# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

llm-sorter classifies transcripts through a self-growing topic tree (hexagonal architecture, Temporal + RabbitMQ + Postgres + LM Studio). Design: [spec](docs/superpowers/specs/2026-10-19-llm-sorter-design.md) (Russian), implementation [plan](docs/superpowers/plans/2026-10-19-llm-sorter.md). Comments and docs are in Russian, identifiers in English.

## Commands

Python `>=3.12,<3.14`, `uv` only (never bare `pip`). `make help` lists everything.

- `make install` / `make lint` / `make fmt` / `make typecheck` / `make test` (unit, `-m "not integration"`)
- Single test: `uv run pytest tests/unit/application/test_classify.py::test_final_is_immutable -v`
- `make up` / `make down` - docker compose (Postgres+pgvector, RabbitMQ, Temporal + UI, Prometheus, Grafana)
- `make migrate` - alembic; `make worker` / `make ingest` / `make consume` - the three long-running processes
- Integration (`make test-int`, `-m integration`): needs `make up` only; the fixture creates and migrates the separate `sorter_test` DB itself (`TEST_DATABASE_URL` optional, default `.../sorter_test`; it refuses any DB not ending in `_test` before TRUNCATE). Models are scripted. If RabbitMQ host ports are remapped (`RABBIT_PORT`/`RABBIT_MGMT_PORT`), set `RABBIT_URL`, e.g. `RABBIT_URL=amqp://guest:guest@localhost:5673/ uv run pytest -m integration`.
- Metrics: worker serves Prometheus on `METRICS_PORT` (9100); compose Prometheus scrapes `host.docker.internal:9100`; Grafana datasource is provisioned from `deploy/grafana/`.
- Alerts: `deploy/prometheus/rules.yml` (loaded via `rule_files`): `TopicMoveAttempt` (any increase of `llm_sorter_topic_move_attempts_total` in 5m) and `DiscoverBlocked` (`llm_sorter_discover_blocked{path} == 1` for 30m: node at `max_children` with a non-empty `other` buffer - needs a manual tree edit). Check with `promtool check rules`.

## Architecture

Layers under `src/llm_sorter/`: `domain` (pure models/rules) <- `ports` (Protocols) <- `application` (services) <- `orchestration` (Temporal workflows/activities/worker) and `adapters` (inbound: poller/rabbit/cli; outbound: lmstudio, postgres, fs, prometheus...). `bootstrap.build_app` is the only place that wires adapters to services; tests inject fakes (`tests/fakes.py`) via its `overrides`.

Import rule, enforced by `tests/unit/test_layers.py`: `domain` imports nothing from other layers nor temporalio/sqlalchemy/httpx/openai/aio_pika; `application` imports only `domain` and `ports`.

Data flow: `ingest` (poller, `LocalDirSource` stub) -> RabbitMQ -> `consume` stores the transcript and starts `ClassifyWorkflow` -> decide/guard walk the tree -> `finalize` (saves the classification, adds `other`/leaf material) -> actions (save/translate/annotate/notify) -> when a node's `other` buffer reaches `buffer_threshold`, `DiscoverWorkflow` clusters it and adds topics -> new version -> files in `other` are reprocessed (`reprocess:{file_id}:{version}`). `SweepWorkflow` (Temporal schedule) picks up due nodes and then runs `reconcile_reprocess`: any file in `X/other` classified at a version where X had fewer live children than now gets `reprocess:{file_id}:{active_version}` (catches reprocess that never started because publish succeeded but the process died before `start_reprocess`). A reprocess first checks `still_in_other`; if another reprocess already moved the file (or it is final) it returns `skipped` without LLM calls and without a `topic_move_attempt`.

## Invariants

- Final is final: a `classified`/`guard_failed` file is immutable. Overwrite attempt -> `FinalClassificationExists`, `llm_sorter_topic_move_attempts_total` +1, no actions. The exception is a replay of the same `run_id` (Temporal retry), which re-applies side effects idempotently.
- `ClassifyWorkflow` pins the tree version at start and uses it to the end; publishing a new version never affects running workflows.
- `other` is never written to YAML; it exists implicitly on `root` and on every node with live children. A leaf has no `other`.
- Workflow ids and reuse policies (`temporal_starter.py`): classify `file_id` and reprocess `reprocess:{file_id}:{version}` use `ALLOW_DUPLICATE_FAILED_ONLY` - a duplicate delivery while running or after success raises `WorkflowAlreadyStartedError` (swallowed, consumer acks), but a failed run can be restarted by redelivery and replays the stored decision via `run_id`. Discover `discover:{node_uid}` uses `ALLOW_DUPLICATE` - rejected only while a run is in progress.
- Retry semantics - outage waits, bad output falls back. `ClassifyService.step`/`guard_check` call the model up to 3 times on `InvalidModelOutput`, then raise it; Temporal does not retry it (`non_retryable_error_types`), and the workflow falls back (decide -> `other`, guard -> `guard_failed`) and records `model_error`. Any other activity error (LM Studio down, Postgres down, timeout) is retried forever with backoff (1s x2, max 5 min), so files wait instead of being misfiled. The OpenAI client has `max_retries=0`, `timeout=110` (below the 120s activity timeout). Actions also retry forever except `ValueError` (non-retryable, fails the workflow).
- `save` action paths with `..` or absolute are rejected (`ValueError`), nothing is written outside `STORAGE_ROOT`. `Transcript.id` must match `^[A-Za-z0-9._-]+$` without `..`; `translate.to` must look like a language tag (`en`, `pt-BR`).

## RabbitMQ consumer

- Invalid body (bad JSON / schema / id) -> `reject(requeue=False)` -> dead-lettered to `<queue>.dlq`.
- Side-effect failure (Postgres/Temporal down) -> after a 1s pause the message is republished to the end of the queue with header `x-retries` +1 and the original is acked; at `x-retries >= 5` it is rejected to `<queue>.dlq`. If republish itself fails -> `nack(requeue=True)`.
- The queue is declared with `x-dead-letter-exchange=""`, `x-dead-letter-routing-key=<queue>.dlq` by both publisher and consumer. Upgrading an existing deployment: an old `transcripts` queue without these arguments makes the declare fail with `PRECONDITION_FAILED` - drain and delete it once (`rabbitmqctl delete_queue transcripts` or the management UI) before starting the new version.

## Editing the rule tree

Rules live in Postgres as immutable versions; edit via CLI, not SQL: `uv run llm-sorter rules pull -o rules.yml` -> edit -> `rules publish rules.yml` (optimistic: base version from the file's `version`, or `--base`; conflict -> pull again) ; `rules rollback <version>` publishes an old version as a new one; `rules show` prints the active tree. A manual YAML whose root `uid` differs from the active root is rejected ("root uid mismatch - run `rules pull` first"). When an edit removes a node or all its children, that node's `other` buffer moves to the nearest live ancestor that still has `other`.

## Known limitations

- Changing `SWEEP_INTERVAL_S` does not update an existing Temporal schedule (delete the schedule `SWEEP_SCHEDULE_ID` to recreate it).
- Re-running a classify workflow after Temporal retention has dropped its history repeats its actions (the stored decision is replayed by `run_id`, but side effects are not deduplicated beyond that).

## Git

Never commit or push without explicit user confirmation. Staging (`git add`) is fine.
