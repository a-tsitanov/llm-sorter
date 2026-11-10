# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

llm-sorter classifies messages through a self-growing topic tree: three processes (feeders -> processors -> learner) connected only by RabbitMQ and SQLite, hexagonal layout. Design: [v2 spec](docs/superpowers/specs/2026-10-23-llm-sorter-v2-design.md) (Russian), [plan](docs/superpowers/plans/2026-10-23-llm-sorter-v2.md). The v1 spec/plan (Temporal + Postgres) are kept for history only. Comments and docs are in Russian, identifiers in English.

## Commands

Python `>=3.12,<3.14`, `uv` only (never bare `pip`). `make help` lists everything.

- `make install` / `make lint` / `make fmt` / `make typecheck` / `make test` (unit, `-m "not integration"`)
- Single test: `uv run pytest tests/unit/application/test_process.py::test_gate_failure_goes_to_parent_other -v`
- `make up` / `make down` - docker compose: RabbitMQ 3.13 (streams), Prometheus, Grafana. Nothing else is needed besides an OpenAI-compatible model server (`LLM_URL`; the gate and discover models may live on their own servers, `LLM_GUARD_URL`/`LLM_GUARD_API_KEY` and `LLM_TEXT_URL`/`LLM_TEXT_API_KEY`, empty = `LLM_URL`).
- Processes: `make feed-dir` / `make feed-tg` (feeders), `make process` (any number of processors; second one on the same host needs `--metrics-port 0`), `make learn` (exactly one; enforced by a flock next to the SQLite file).
- Integration (`make test-int`, `-m integration`): needs `make up` only; queue names get a random prefix per test and are deleted after. If RabbitMQ host ports are remapped (`RABBIT_PORT`/`RABBIT_MGMT_PORT`), `RABBIT_URL` must point to them (the repo `.env` does).
- Metrics: processor serves Prometheus on `METRICS_PORT` (9100), learner on `LEARNER_METRICS_PORT` (9102); compose Prometheus scrapes both on `host.docker.internal`, Grafana (`localhost:3000`) has the provisioned dashboard `llm-sorter` (`deploy/grafana/provisioning/dashboards/`). Alerts `OtherLost` and `DiscoverBlocked` (`deploy/prometheus/rules.yml`): `llm_sorter_discover_blocked{node} == 1` for 30m - node at `max_children` with a non-empty buffer, needs a manual tree edit.

## Architecture

Layers under `src/llm_sorter/`: `domain` (pure models/rules) <- `ports` (Protocols) <- `application` (services) <- `adapters` (inbound: cli, feeders, processor, learner_runner; outbound: llm, sqlite_store, rabbit, metrics). `bootstrap.py` is the only place that wires adapters to services; tests inject fakes (`tests/fakes.py`) via `overrides`. Import rule, enforced by `tests/unit/test_layers.py`: `domain` imports nothing from other layers nor httpx/openai/aio_pika/aiosqlite/telethon/prometheus_client; `application` imports only `domain` and `ports`.

Data flow:
- **Feeders** (`feed dir` = `LocalDirSource`, `feed tg` = `TelegramSource`: chats of Telegram folders `TG_FOLDERS`, per-chat cursor in `TG_STATE_FILE`) publish `{"id","text","metadata"}` to queue `llm-sorter.inbox` (DLQ `llm-sorter.inbox.dlq`).
- **Processor** (`ProcessService` + `adapters/inbound/processor.py`): tree from SQLite cached in memory (`TreeCache`, re-checks `tree_rev` every `TREE_REFRESH_S`); descends from `root` with the decide model until a leaf (`DECIDE_PROTOCOL=chat`: JSON over chat/completions; `systemone`: a decision model such as Liquid d1 via `POST /v1/systemone`, one `choice` over all children + `other`, option text cut to `CRITERION_CHARS`), then the leaf's gate (`guard`). Outcomes: gate passed -> topic exchange `llm-sorter.classified`, routing key = topic path by ids (`finance.cards`), always also bound to archive queue `llm-sorter.classified.all` (`#`); no matching child at node X -> stream `llm-sorter.other` with `node=X`; gate failed at leaf L -> `other` of L's **parent**; a passed leaf below `max_depth` also writes a `reason=leaf` copy (leaf growth material).
- **Learner** (`Learner` + `learner_runner.py`): reads the stream from the saved offset (on start it appends a `Marker` record and runs no discover until it reads it back - buffers are complete only then; records at offsets already seen are ignored, so a reconnect replay is harmless), keeps per-node buffers, and when a node has `buffer_threshold` *new* (not yet considered) messages runs `DiscoverService.propose` (cluster, LLM proposals, dedup, guard check), writes topics to SQLite, re-publishes the clustered messages to inbox with `x-reprocess` + 1 and `x-tree-rev` = the new revision (a processor whose cache is older reloads the tree before handling it). A node whose discover keeps failing is skipped (`model_error{discover}`), never blocking other nodes, offset saves or refresh. A leaf splits only into >= 2 subtopics; then its whole buffer is re-published.

## Invariants

- Ack after publish: the processor acks inbox only after the result (and every `other` copy) is confirmed by the broker. Delivery is at-least-once; subscribers drop duplicates by `id`, and for the same `id` the record with the higher `tree_rev` wins (a split leaf re-publishes its messages with a deeper path).
- Outage waits, bad output falls back: LLM/Rabbit/SQLite errors retry the same message in place with backoff (1 s x2, max 300 s, interrupted by stop -> nack requeue) - never to `other` or DLQ. `InvalidModelOutput` (including model HTTP 400/404/413/422, which a retry cannot fix) is retried 3 times, then decide -> `other` / gate -> parent `other`, `reason=model_error`. Only invalid bodies go to the DLQ. Decide/guard prompts carry at most `MAX_TEXT_CHARS` (12000) characters of the message.
- Loop guard: a message whose `x-reprocess >= settings.max_reprocess` (default 3) that lands in `other` again is marked `final` and never buffered again.
- Message state lives only in the stream: `Resolved` (re-published ids) and `Considered` (ids already seen by discover) are records the learner appends; they apply only to copies with `at <= record.at`, so a newer copy of the same id is never dropped. The latest copy of an id wins. SQLite keeps only the tree, the changelog and `learner_state.other_offset` (oldest waiting copy).
- Retention: stream `x-max-age`/`x-max-length-bytes` (`OTHER_MAX_AGE`/`OTHER_MAX_BYTES`); the learner re-publishes waiting copies older than half the max age (new `at`, followed by a `Considered` record for considered ones) every `OTHER_REFRESH_S`, the last run time is kept in SQLite (`learner_state.last_refresh_at`) so restarts do not postpone it. If size retention removed records before the saved offset, the learner logs an error and increments `llm_sorter_other_lost_total` (alert `OtherLost`). RabbitMQ data lives in the compose volume `rabbitdata` - it is the system state now (inbox, archive, stream). Stream arguments of a live stream cannot change (`PRECONDITION_FAILED`); all processes must use the same values.
- Topic `id` is `^[a-z0-9_]+$`, not `root`/`other`, globally unique (it is a routing-key segment); discover suffixes collisions (`cards_2`). A discovered `guard` must be a real question (`guard_is_meaningful`: >= 15 chars, has `?`, not starting with yes/no/да/нет).
- Tree changes are incremental (`domain/changes.py`): add / move / edit / archive (with subtree), each one changelog row and `tree_rev + 1` in one SQLite transaction; no snapshots, no rollback. Archived ids can be reused.

## Editing the rule tree

Via CLI only: `topics show` (tree with routing keys), `topics add ID --title T --description D [--parent P] [--guard G]`, `topics move ID PARENT`, `topics edit ID --title ...`, `topics archive ID`, `topics import FILE [--replace]` / `topics export [-o FILE]` (flat YAML: `settings` + `topics` list with `parent`), `topics log`. Seed for the synthetic dataset: `datasets/synthetic-v1/topics.yml`. Moving a topic changes its routing key - bindings on the old path stop receiving it.

## Git

Never commit or push without explicit user confirmation. Staging (`git add`) is fine.
