> **Заменён:** [план v2](2026-10-23-llm-sorter-v2.md) (2026-10-23).

# llm-sorter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Сервис, который раскладывает транскрипты разговоров по самовыращиваемому тематическому дереву (d1 decide → guard → actions) и сам дополняет дерево новыми темами из накопленных `other`.

**Architecture:** Гексагональная: `domain` (чистые модели дерева, обход, правила discover) ← `application` (use cases, зависят только от `ports`) ← `adapters` (LM Studio, Postgres, RabbitMQ, ФС, Rocket-заглушка, Prometheus, CLI) и `orchestration` (Temporal workflows/activities — тонкие обёртки над use cases). Workflows детерминированы, всё I/O — в activities.

**Tech Stack:** Python 3.12, uv, temporalio, aio-pika, openai (клиент к LM Studio), pydantic v2 + pydantic-settings, SQLAlchemy 2 async + asyncpg + alembic, Postgres 16 + pgvector, scikit-learn (HDBSCAN), numpy, prometheus-client, typer, PyYAML. Тесты: pytest, pytest-asyncio, respx. Линт: ruff, mypy.

**Spec:** `docs/superpowers/specs/2026-10-19-llm-sorter-design.md`

## Global Constraints

- **Git: НИКАКИХ `git commit` / `git push` без явного «да» пользователя.** Шаг «Stage» в каждой задаче = `git add …` и остановка с вопросом. Это переопределяет «frequent commits» и любые авто-коммиты исполнителей.
- Python `>=3.12,<3.14`; управление зависимостями только `uv` (`uv add`, `uv run`), не `pip`.
- `domain/` не импортирует ничего из `application/`, `adapters/`, `orchestration/`, а также temporalio, sqlalchemy, httpx, openai, aio_pika.
- `application/` импортирует только `domain/` и `ports/`.
- `other` не пишется в YAML; неявно есть у `root` и у каждого узла с живыми детьми. У листа `other` нет.
- Файл со статусом `classified` или `guard_failed` неизменяем. Попытка перезаписать → `FinalClassificationExists`, метрика `llm_sorter_topic_move_attempts_total` (+1), никаких actions.
- ClassifyWorkflow фиксирует версию дерева на старте и идёт по ней до конца.
- `workflow_id`: классификация `file_id`; переобработка `reprocess:{file_id}:{version}`; discover `discover:{node_uid}`.
- Дефолты настроек: `max_depth: 3`, `language: ru`, `buffer_threshold: 50`, `min_cluster_size: 10`, `max_children: 12`, `dedup_similarity: 0.85`, `llm_dedup_floor: 0.70`, `samples_per_cluster: 5`, `sample_chars: 2000`.
- Default action при провале guard: `{type: save, path: "_guard_failed"}`.
- Модели: decide = Liquid LFM2 (env `MODEL_DECIDE`), guard/translate/annotate/discover = Qwen ~4B (`MODEL_GUARD`, `MODEL_TEXT`), embed = `bge-m3` (`MODEL_EMBED`, размерность 1024). Все через LM Studio `http://localhost:1234/v1`. `temperature=0` для decide и guard.
- Язык комментариев/доков — русский, код и идентификаторы — английский.

## Review Focus

1. **d1 вернула id вне списка кандидатов или битый JSON** → activity бросает `InvalidModelOutput`, Temporal ретраит 3 раза, затем workflow считает выбор `other` и пишет `model_error("decide")`. Тест в Task 7.
2. **Повторная доставка того же сообщения из RabbitMQ / повторный id** → второй workflow не стартует (`WorkflowAlreadyStartedError` поглощается), actions второй раз не выполняются. Тесты в Task 7 (повтор finalize → skipped, actions не вызваны) и Task 11 (consumer ack при already-started).
3. **Публикация новой версии, пока файлы в обработке** → бегущий workflow доходит по старой версии. Тест в Task 7.
4. **Пустой / пробельный транскрипт** → `root/other` без вызова моделей. Тест в Task 5.
5. **Путь `save` с `..` или абсолютный** (из ручного YAML) → `ValueError`, файл за пределами `STORAGE_ROOT` не пишется. Тест в Task 6.

---

## File Structure

```
pyproject.toml, uv.lock, Makefile, docker-compose.yml, alembic.ini, .env.example, CLAUDE.md
deploy/prometheus.yml
src/llm_sorter/
  config.py                     # AppConfig (pydantic-settings)
  domain/
    actions.py                  # Action-модели (discriminated union)
    tree.py                     # Settings, Node, RuleTree, навигация
    loader.py                   # YAML ⇄ RuleTree, генерация uid
    validation.py               # validate_tree
    traversal.py                # Candidate, candidates_for, resolve_choice, guard_question
    classification.py           # Transcript, Status, Classification, ошибки
    tree_edit.py                # add_children, merge_archived, gained_children, live_ancestor_with_other
    discover.py                 # should_discover, slots_left, dedup_verdict, TopicProposal
  ports/
    models.py                   # DecisionModel, GuardModel, TextLLM, Embedder, InvalidModelOutput
    storage.py                  # RulesRepository, TranscriptStore, ClassificationRepository,
                                # MaterialRepository, EmbeddingCache, FileStorage, Notifier, VersionConflict
    io.py                       # TranscriptSource, MessagePublisher, WorkflowStarter
    metrics.py                  # Metrics
  application/
    classify.py                 # ClassifyService
    actions.py                  # ActionContext, ActionService
    rules.py                    # RulesService, PublishResult
    clustering.py               # cluster, nearest_to_centroid, cosine
    discover.py                 # DiscoverService, DiscoverResult
  orchestration/
    activities.py               # Activities (@activity.defn методы)
    workflows.py                # ClassifyWorkflow, DiscoverWorkflow, SweepWorkflow, ClassifyInput, ClassifyOutcome
    worker.py                   # run_worker, ensure_sweep_schedule
  adapters/
    outbound/lmstudio.py        # LMStudioDecision, LMStudioGuard, LMStudioText, LMStudioEmbedder
    outbound/postgres/          # tables.py, rules.py, transcripts.py, classifications.py, material.py, embeddings.py
    outbound/fs_storage.py      # LocalFileStorage
    outbound/rocket_stub.py     # LoggingNotifier
    outbound/prometheus_metrics.py
    outbound/temporal_starter.py
    inbound/stub_source.py      # LocalDirSource
    inbound/poller.py           # run_poller
    inbound/rabbit.py           # RabbitPublisher, run_consumer
    inbound/cli.py              # typer app
  bootstrap.py                  # сборка зависимостей из AppConfig
migrations/                     # alembic
tests/
  fakes.py                      # in-memory реализации всех портов
  unit/…  integration/…
```

---

### Task 1: Scaffold проекта

**Files:**
- Create: `pyproject.toml`, `Makefile`, `docker-compose.yml`, `.env.example`, `.gitignore`, `src/llm_sorter/__init__.py`, `src/llm_sorter/config.py`, все `__init__.py` пакетов из File Structure, `tests/unit/test_config.py`

**Interfaces:**
- Produces: `AppConfig` (pydantic-settings, env без префикса) с полями и дефолтами:
  `lmstudio_url="http://localhost:1234/v1"`, `lmstudio_api_key="lm-studio"`, `model_decide="lfm2"`, `model_guard="qwen3-4b"`, `model_text="qwen3-4b"`, `model_embed="bge-m3"`, `database_url="postgresql+asyncpg://sorter:sorter@localhost:5432/sorter"`, `rabbit_url="amqp://guest:guest@localhost:5672/"`, `queue_name="transcripts"`, `temporal_address="localhost:7233"`, `temporal_namespace="default"`, `task_queue="llm-sorter"`, `storage_root=Path("./data/out")`, `source_dir=Path("./data/inbox")`, `poll_interval_s=10.0`, `metrics_port=9100`, `sweep_interval_s=600`.

- [ ] **Step 1:** `git init`; `uv init --package --name llm-sorter --python 3.12`; добавить зависимости из Tech Stack (`uv add …`, dev-группа: pytest, pytest-asyncio, respx, ruff, mypy, types-PyYAML). pytest: `asyncio_mode = "auto"`, маркер `integration`, `testpaths=["tests"]`. ruff: line-length 100.
- [ ] **Step 2: Failing test** `tests/unit/test_config.py`:

```python
def test_defaults(monkeypatch):
    monkeypatch.delenv("MODEL_DECIDE", raising=False)
    cfg = AppConfig()
    assert cfg.task_queue == "llm-sorter"
    assert cfg.lmstudio_url == "http://localhost:1234/v1"

def test_env_override(monkeypatch):
    monkeypatch.setenv("MODEL_DECIDE", "lfm2-1.2b")
    assert AppConfig().model_decide == "lfm2-1.2b"
```

- [ ] **Step 3:** `uv run pytest tests/unit/test_config.py -v` → FAIL (ImportError).
- [ ] **Step 4:** Реализовать `AppConfig` в `src/llm_sorter/config.py`.
- [ ] **Step 5:** `docker-compose.yml`: `postgres` (образ `pgvector/pgvector:pg16`, user/pass/db `sorter`), `rabbitmq:3-management`, `temporal` (`temporalio/auto-setup`, БД — тот же postgres, отдельная база `temporal`), `temporal-ui`, `prometheus` (конфиг `deploy/prometheus.yml`, scrape `host.docker.internal:9100`), `grafana`. Makefile targets: `help install lint fmt typecheck test test-int up down migrate worker ingest consume`. `test` = `uv run pytest -m "not integration"`, `test-int` = `uv run pytest -m integration`.
- [ ] **Step 6:** `uv run pytest tests/unit/test_config.py -v` → PASS; `make lint` → без ошибок.
- [ ] **Step 7: Stage** `git add -A` — **не коммитить**, спросить пользователя.

---

### Task 2: Дерево, actions, YAML-загрузчик

**Files:**
- Create: `src/llm_sorter/domain/actions.py`, `src/llm_sorter/domain/tree.py`, `src/llm_sorter/domain/loader.py`
- Test: `tests/unit/domain/test_tree.py`, `tests/unit/domain/test_loader.py`

**Interfaces:**
- Produces (все модели — `pydantic.BaseModel`, `frozen=True`):
  - `actions.py`: `TranslateAction(type: Literal["translate"], to: str)`, `AnnotateAction(type: Literal["annotate"])`, `SaveAction(type: Literal["save"], path: str | None = None)`, `RocketAction(type: Literal["rocket"], channel: str)`; `Action = Annotated[TranslateAction | AnnotateAction | SaveAction | RocketAction, Field(discriminator="type")]`.
  - `tree.py`: `OTHER_ID = "other"`; `DiscoverSettings` (дефолты из Global Constraints); `Settings(max_depth=3, language="ru", discover=DiscoverSettings())`; `Defaults(guard_failed_action: Action = SaveAction(path="_guard_failed"))`; `Node(uid: str, id: str, title: str = "", description: str = "", examples: list[str] = [], guard: str | None = None, actions: list[Action] = [], origin: Literal["manual","discovered"] = "manual", max_depth: int | None = None, children: list[Node] = [])`; `ArchivedNode(uid: str, path: str, parent_uid: str | None)`; `RuleTree(version: int | None, settings: Settings, defaults: Defaults, root: Node, archived: list[ArchivedNode] = [])` с методами:
    `find(uid) -> Node` (KeyError), `parent_of(uid) -> Node | None`, `path_of(uid) -> str` (root → `""`, иначе `"finance/crypto"`), `depth_of(uid) -> int` (root 0), `max_depth_for(uid) -> int` (ближайший `max_depth` вверх по предкам, иначе `settings.max_depth`), `has_other(uid) -> bool` (root или есть дети), `other_path(uid) -> str` (`"other"` для root, иначе `path + "/other"`), `iter_nodes() -> Iterator[Node]`.
  - `loader.py`: `load_tree(text: str) -> RuleTree` (узлам без `uid` присваивает `uuid4().hex[:12]`; root без `id` → `id="root"`); `dump_tree(tree: RuleTree) -> str` (YAML, `allow_unicode=True`, `sort_keys=False`, без полей равных дефолтам); `class TreeParseError(ValueError)`.

- [ ] **Step 1: Failing tests:**

```python
YAML = """
node:
  id: root
  children:
    - id: finance
      title: Финансы
      description: Деньги
      children:
        - id: crypto
          title: Крипта
          description: Крипта
          actions: [{type: save}, {type: rocket, channel: "#c"}]
"""

def test_load_assigns_uids_and_paths():
    t = load_tree(YAML)
    crypto = next(n for n in t.iter_nodes() if n.id == "crypto")
    assert len(crypto.uid) == 12
    assert t.path_of(crypto.uid) == "finance/crypto"
    assert t.depth_of(crypto.uid) == 2
    assert t.settings.max_depth == 3

def test_other_only_where_children():
    t = load_tree(YAML)
    fin = next(n for n in t.iter_nodes() if n.id == "finance")
    crypto = next(n for n in t.iter_nodes() if n.id == "crypto")
    assert t.has_other(t.root.uid) and t.has_other(fin.uid)
    assert not t.has_other(crypto.uid)
    assert t.other_path(fin.uid) == "finance/other"
    assert t.other_path(t.root.uid) == "other"

def test_roundtrip_keeps_uids():
    t = load_tree(YAML)
    t2 = load_tree(dump_tree(t))
    assert [n.uid for n in t.iter_nodes()] == [n.uid for n in t2.iter_nodes()]

def test_actions_parsed():
    crypto = next(n for n in load_tree(YAML).iter_nodes() if n.id == "crypto")
    assert isinstance(crypto.actions[1], RocketAction) and crypto.actions[1].channel == "#c"

def test_minimal_root_only():
    t = load_tree("node: {id: root}")
    assert t.root.children == [] and t.has_other(t.root.uid)

def test_bad_yaml_raises():
    with pytest.raises(TreeParseError):
        load_tree("node: [")
```

- [ ] **Step 2:** `uv run pytest tests/unit/domain -v` → FAIL.
- [ ] **Step 3:** Реализовать модели и загрузчик.
- [ ] **Step 4:** `uv run pytest tests/unit/domain -v` → PASS.
- [ ] **Step 5: Stage**, не коммитить.

---

### Task 3: Валидация дерева

**Files:**
- Create: `src/llm_sorter/domain/validation.py`
- Test: `tests/unit/domain/test_validation.py`

**Interfaces:**
- Consumes: `RuleTree`, `OTHER_ID` (Task 2).
- Produces: `validate_tree(tree: RuleTree) -> None`; `class TreeValidationError(ValueError)` с атрибутом `errors: list[str]`.
  Правила: `id` матчит `^[a-z0-9_]+$`; `id != "other"`; `id` уникален среди соседей; `uid` уникален глобально (включая `archived`); `depth_of(uid) <= max_depth_for(uid)`; у не-root узлов непустые `title` и `description`. Все ошибки собираются, не первая.

- [ ] **Step 1: Failing tests:** по одному тесту на правило; каждый строит YAML, нарушающий одно правило, и проверяет `pytest.raises(TreeValidationError)` и подстроку в `exc.errors` (`"other"`, `"duplicate id"`, `"duplicate uid"`, `"depth"`, `"title"`). Плюс `test_valid_tree_passes` на YAML из Task 2 и `test_collects_all_errors` (два нарушения → `len(errors) == 2`). Тест глубины: `settings: {max_depth: 1}` и узел на глубине 2.
- [ ] **Step 2:** запустить → FAIL.
- [ ] **Step 3:** Реализовать.
- [ ] **Step 4:** `uv run pytest tests/unit/domain/test_validation.py -v` → PASS.
- [ ] **Step 5: Stage**, не коммитить.

---

### Task 4: Обход, классификация, редактирование дерева, порты, фейки

**Files:**
- Create: `src/llm_sorter/domain/traversal.py`, `src/llm_sorter/domain/classification.py`, `src/llm_sorter/domain/tree_edit.py`, `src/llm_sorter/domain/discover.py`, `src/llm_sorter/ports/models.py`, `src/llm_sorter/ports/storage.py`, `src/llm_sorter/ports/io.py`, `src/llm_sorter/ports/metrics.py`, `tests/fakes.py`
- Test: `tests/unit/domain/test_traversal.py`, `tests/unit/domain/test_tree_edit.py`, `tests/unit/domain/test_discover_rules.py`

**Interfaces:**
- `traversal.py`: `Candidate(id: str, title: str, description: str, examples: list[str] = [])`; `candidates_for(tree, uid) -> list[Candidate]` — живые дети по порядку + последним `Candidate(id="other", title="Other", description="Ничего из перечисленного не подходит")`; `[]` если `has_other` ложно; `resolve_choice(tree, uid, choice_id: str) -> str | None` — uid ребёнка, `None` для `"other"`, `ValueError` для неизвестного; `guard_question(node: Node) -> str` — `node.guard` или `f"Этот разговор действительно на тему «{title}»: {description}?"`.
- `classification.py`: `Transcript(id: str, text: str, metadata: dict[str, Any] = {}, received_at: datetime)`; `class Status(StrEnum): CLASSIFIED="classified"; IN_OTHER="in_other"; GUARD_FAILED="guard_failed"`; `Classification(file_id: str, tree_version: int, node_uid: str, path: str, status: Status)` + property `final -> bool` (`status != IN_OTHER`); `class FinalClassificationExists(Exception)`.
- `discover.py`: `TopicProposal(id: str, title: str, description: str, examples: list[str], guard: str)`; `should_discover(tree, uid, new_count: int) -> bool` — `new_count >= buffer_threshold and depth_of(uid) < max_depth_for(uid) and slots_left(tree, uid) > 0`; `slots_left(tree, uid) -> int` — `max_children - len(children)`; `dedup_verdict(similarity: float, s: DiscoverSettings) -> Literal["dup","ask","new"]` — `>= dedup_similarity` → dup, `>= llm_dedup_floor` → ask, иначе new.
- `tree_edit.py`:
  - `add_children(tree, parent_uid, proposals: list[TopicProposal]) -> RuleTree` — новые узлы `origin="discovered"`, новый uid, без actions; `version=None`.
  - `merge_archived(old: RuleTree, new: RuleTree) -> RuleTree` — узлы из `old` (живые), чьих uid нет в `new`, дописываются в `new.archived` с путём из `old`.
  - `gained_children(old, new) -> list[str]` — uid узлов, у которых в `new` есть живые дети, которых не было в `old`.
  - `live_ancestor_with_other(old, new, uid) -> str` — ближайший предок `uid` по `old`, который есть в `new` и `new.has_other(...)`; fallback root uid.
- `ports/models.py` (Protocols): `DecisionModel.decide(text: str, candidates: list[Candidate]) -> str`; `GuardModel.check(text: str, question: str) -> bool`; `TextLLM.translate(text, to: str) -> str`, `.summarize(text, language: str) -> str`, `.propose_topic(samples: list[str], parent_path: str, siblings: list[Candidate], language: str) -> TopicProposal`, `.same_topic(a: Candidate, b: Candidate) -> bool`; `Embedder.embed(texts: list[str]) -> list[list[float]]`; `class InvalidModelOutput(Exception)`. Все методы async.
- `ports/storage.py` (async Protocols):
  - `RulesRepository`: `active() -> RuleTree`, `get(version: int) -> RuleTree`, `publish(tree: RuleTree, base_version: int | None, source: Literal["manual","discover","rollback"]) -> int` (бросает `VersionConflict`, если активная ≠ `base_version`; возвращает новый номер и делает её активной).
  - `TranscriptStore`: `put(t: Transcript) -> bool` (False, если id уже был), `get(file_id) -> Transcript`.
  - `ClassificationRepository`: `get(file_id) -> Classification | None`, `save(c: Classification) -> None` (бросает `FinalClassificationExists`, если существующая `final`).
  - `MaterialRepository`: `add(node_uid, file_id, kind: Literal["other","leaf"])`, `list(node_uid) -> list[MaterialItem]` (`MaterialItem(file_id, kind)`), `remove(node_uid, file_id)`, `move_other(from_uid, to_uid)`, `count_new(node_uid) -> int` (добавленные после последнего `mark_discovered`), `mark_discovered(node_uid)`, `due_nodes(threshold: int) -> list[str]`, `other_size(node_uid) -> int`.
  - `EmbeddingCache`: `get_many(file_ids: list[str]) -> dict[str, list[float]]`, `put_many(vectors: dict[str, list[float]])`.
  - `FileStorage.write(rel_path: str, content: str) -> str`; `Notifier.send(channel: str, text: str, attachment_name: str, attachment: str) -> None`.
  - `class VersionConflict(Exception)`.
- `ports/io.py`: `TranscriptSource.fetch_new() -> list[Transcript]`; `MessagePublisher.publish(t: Transcript) -> None`; `WorkflowStarter.start_classify(file_id) -> bool`, `.start_discover(node_uid) -> bool`, `.start_reprocess(file_id: str, node_uid: str, version: int) -> bool` (False = уже запущен).
- `ports/metrics.py` (sync Protocol): `file_classified(path, status: Status, version: int)`, `model_call(role: str, latency_s: float, ok: bool)`, `model_error(role: str)`, `guard_result(path, passed: bool)`, `action_done(action_type: str, ok: bool, latency_s: float)`, `buffer_size(path, size: int)`, `discover_run(path, created: int, rejected: int)`, `reprocess_moved(from_path, to_path)`, `topic_move_attempt()`, `active_version(v: int)`, `rules_published(source: str)`.
- `tests/fakes.py`: in-memory реализации всех портов: `FakeRules`, `FakeTranscripts`, `FakeClassifications`, `FakeMaterial`, `FakeEmbeddings`, `FakeStorage` (dict path→content), `FakeNotifier` (list вызовов), `FakeStarter` (списки вызовов, повтор → False), `RecordingMetrics` (Counter: ключ — имя метода, для `model_error` — `"model_error:<role>"`), `ScriptedDecision(answers: dict[str, list[str]])` (по `path`/списку id кандидатов выдаёт заранее заданный ответ), `ScriptedGuard(default=True)`, `FakeText`, `FakeEmbedder(mapping: dict[str, list[float]])`.

- [ ] **Step 1: Failing tests:**

```python
def test_candidates_include_other_last(tree):          # tree: finance/{crypto,banks}
    ids = [c.id for c in candidates_for(tree, uid(tree, "finance"))]
    assert ids == ["crypto", "banks", "other"]

def test_leaf_has_no_candidates(tree):
    assert candidates_for(tree, uid(tree, "crypto")) == []

def test_resolve_choice(tree):
    f = uid(tree, "finance")
    assert resolve_choice(tree, f, "crypto") == uid(tree, "crypto")
    assert resolve_choice(tree, f, "other") is None
    with pytest.raises(ValueError):
        resolve_choice(tree, f, "nft")

def test_add_children_marks_discovered(tree):
    t2 = add_children(tree, uid(tree, "crypto"), [TopicProposal(id="nft", title="NFT", description="d", examples=[], guard="g")])
    nft = next(n for n in t2.iter_nodes() if n.id == "nft")
    assert nft.origin == "discovered" and t2.path_of(nft.uid) == "finance/crypto/nft"
    assert gained_children(tree, t2) == [uid(tree, "crypto")]

def test_merge_archived_and_live_ancestor(tree):
    new = remove_node(tree, "crypto")                   # хелпер теста
    merged = merge_archived(tree, new)
    assert [a.path for a in merged.archived] == ["finance/crypto"]
    assert live_ancestor_with_other(tree, merged, uid(tree, "crypto")) == uid(tree, "finance")

def test_should_discover_threshold_depth_slots(tree):
    f = uid(tree, "finance")
    assert should_discover(tree, f, 50) and not should_discover(tree, f, 49)
    deep = with_settings(tree, max_depth=1)
    assert not should_discover(deep, f, 100)
    full = with_settings(tree, max_children=2)          # у finance 2 ребёнка
    assert not should_discover(full, f, 100)

@pytest.mark.parametrize("sim,verdict", [(0.9, "dup"), (0.85, "dup"), (0.75, "ask"), (0.5, "new")])
def test_dedup_verdict(sim, verdict):
    assert dedup_verdict(sim, DiscoverSettings()) == verdict
```

- [ ] **Step 2:** запустить → FAIL.
- [ ] **Step 3:** Реализовать domain-модули, Protocols в `ports/`, фейки в `tests/fakes.py`.
- [ ] **Step 4:** `uv run pytest tests/unit -v` → PASS; `uv run mypy src` → без ошибок.
- [ ] **Step 5:** Архитектурная проверка — тест `tests/unit/test_layers.py`: обходит `src/llm_sorter/domain/**/*.py` и `application/**/*.py` через `ast`, падает, если `domain` импортирует `llm_sorter.application|adapters|orchestration|ports` или любой из `temporalio, sqlalchemy, httpx, openai, aio_pika`; если `application` импортирует `adapters|orchestration` или эти библиотеки. → PASS.
- [ ] **Step 6: Stage**, не коммитить.

---

### Task 5: ClassifyService (шаг дерева, guard, finalize)

**Files:**
- Create: `src/llm_sorter/application/classify.py`
- Test: `tests/unit/application/test_classify.py`

**Interfaces:**
- Consumes: порты и domain из Task 4.
- Produces:
  - `StepResult(kind: Literal["descend","other","leaf"], next_uid: str | None)` (pydantic).
  - `FinalizeResult(skipped: bool, path: str)`.
  - `ClassifyService(rules: RulesRepository, transcripts: TranscriptStore, classifications: ClassificationRepository, material: MaterialRepository, decision: DecisionModel, guard: GuardModel, starter: WorkflowStarter, metrics: Metrics)`:
    - `async active_version() -> int`
    - `async root_uid(version: int) -> str`
    - `async step(version: int, node_uid: str, file_id: str) -> StepResult` — нет кандидатов → `leaf`; пустой/пробельный текст → `other` без вызова модели; иначе `decide`, `resolve_choice`; неизвестный id → `InvalidModelOutput`. Деревья кэшируются по версии в dict.
    - `async guard_check(version, node_uid, file_id) -> bool` — `metrics.guard_result`.
    - `async finalize(file_id, version, node_uid, status: Status, from_other_uid: str | None = None) -> FinalizeResult` — сохраняет `Classification` (`path` = `other_path(node_uid)` для IN_OTHER, `"_guard_failed/" + path_of` для GUARD_FAILED, иначе `path_of`); `FinalClassificationExists` → `metrics.topic_move_attempt()`, `skipped=True`, больше ничего. При успехе: IN_OTHER → `material.add(node_uid, file_id, "other")`; CLASSIFIED и `depth_of < max_depth_for` → `material.add(node_uid, file_id, "leaf")`; `from_other_uid` задан и статус ≠ IN_OTHER → `material.remove(from_other_uid, file_id)` + `metrics.reprocess_moved`; если IN_OTHER с `from_other_uid == node_uid` — материал уже там, повторно не добавлять. Затем `should_discover(tree_active, node_uid, await material.count_new(node_uid))` → `starter.start_discover(node_uid)`. `metrics.file_classified`.

- [ ] **Step 1: Failing tests** (фейки из `tests/fakes.py`, дерево `finance/{crypto/{trading}, banks}`):

```python
async def test_step_descends_then_leaf(svc, v):
    r = await svc.step(v, root, "f1")                    # ScriptedDecision: root→finance
    assert r == StepResult(kind="descend", next_uid=uid("finance"))
    assert (await svc.step(v, uid("trading"), "f1")).kind == "leaf"

async def test_step_other(svc, v):                       # decision → "other"
    assert (await svc.step(v, root, "f1")) == StepResult(kind="other", next_uid=None)

async def test_unknown_choice_raises(svc, v):            # decision → "nft"
    with pytest.raises(InvalidModelOutput):
        await svc.step(v, root, "f1")

async def test_empty_text_goes_other_without_model(svc, v, decision):
    await transcripts.put(Transcript(id="e", text="  \n", received_at=now))
    assert (await svc.step(v, root, "e")).kind == "other"
    assert decision.calls == 0

async def test_finalize_other_adds_material(svc, v):
    await svc.finalize("f1", v, uid("finance"), Status.IN_OTHER)
    assert (await classifications.get("f1")).path == "finance/other"
    assert await material.list(uid("finance")) == [MaterialItem(file_id="f1", kind="other")]

async def test_finalize_leaf_below_max_depth_is_material(svc, v):
    await svc.finalize("f1", v, uid("crypto"), Status.CLASSIFIED)    # depth 2 < 3
    assert MaterialItem(file_id="f1", kind="leaf") in await material.list(uid("crypto"))

async def test_finalize_at_max_depth_no_material(svc, v):
    await svc.finalize("f1", v, uid("trading"), Status.CLASSIFIED)   # depth 3
    assert await material.list(uid("trading")) == []

async def test_final_is_immutable(svc, v, metrics):
    await svc.finalize("f1", v, uid("trading"), Status.CLASSIFIED)
    r = await svc.finalize("f1", v, uid("banks"), Status.CLASSIFIED)
    assert r.skipped and (await classifications.get("f1")).path == "finance/crypto/trading"
    assert metrics.counts["topic_move_attempt"] == 1

async def test_reprocess_moves_out_of_other(svc, v):
    await svc.finalize("f1", v, uid("finance"), Status.IN_OTHER)
    await svc.finalize("f1", v, uid("banks"), Status.CLASSIFIED, from_other_uid=uid("finance"))
    assert await material.list(uid("finance")) == []

async def test_threshold_triggers_discover(svc, v, starter):  # buffer_threshold=2 в дереве
    for f in ("a", "b"):
        await svc.finalize(f, v, uid("finance"), Status.IN_OTHER)
    assert starter.discover == [uid("finance")]
```

- [ ] **Step 2:** запустить → FAIL.
- [ ] **Step 3:** Реализовать `ClassifyService`.
- [ ] **Step 4:** `uv run pytest tests/unit/application/test_classify.py -v` → PASS.
- [ ] **Step 5: Stage**, не коммитить.

---

### Task 6: Actions + LocalFileStorage + Rocket-заглушка

**Files:**
- Create: `src/llm_sorter/application/actions.py`, `src/llm_sorter/adapters/outbound/fs_storage.py`, `src/llm_sorter/adapters/outbound/rocket_stub.py`
- Test: `tests/unit/application/test_actions.py`, `tests/unit/adapters/test_fs_storage.py`

**Interfaces:**
- Produces:
  - `ActionContext(file_id: str, text: str, translations: dict[str, str] = {}, summary: str | None = None)` (pydantic).
  - `ActionService(transcripts, storage: FileStorage, notifier: Notifier, text_llm: TextLLM, rules: RulesRepository, metrics, language: str)`:
    - `async load_context(file_id) -> ActionContext`
    - `async node_plan(version: int, node_uid: str, guard_failed: bool) -> tuple[list[Action], str]` — `(node.actions, path_of)` либо `([defaults.guard_failed_action], path_of)`.
    - `async run(ctx: ActionContext, action: Action, node_path: str) -> ActionContext`:
      translate → `translations[to] = translate(text, to)`; annotate → `summary = summarize(text, language)`; save → пишет `{base}/{file_id}.txt`, если есть summary — `{base}/{file_id}.summary.txt`, по каждому переводу `{base}/{file_id}.{lang}.txt`, где `base = action.path or node_path`; для guard-failed `base = action.path + "/" + node_path`; rocket → `notifier.send(channel, text=summary or text[:500], attachment_name=f"{file_id}.txt", attachment=text)`. `metrics.action_done`.
  - `LocalFileStorage(root: Path).write(rel_path, content) -> str` — `ValueError` для абсолютного пути или выхода за `root` (через `resolve()` + `is_relative_to`); создаёт каталоги; UTF-8.
  - `LoggingNotifier.send(...)` — `logging.info("rocket stub: channel=%s name=%s text=%s", ...)`, ничего не отправляет.

- [ ] **Step 1: Failing tests:**

```python
async def test_chain_translate_annotate_save(svc, storage):
    ctx = await svc.load_context("f1")
    for a in [TranslateAction(type="translate", to="en"), AnnotateAction(type="annotate"), SaveAction(type="save")]:
        ctx = await svc.run(ctx, a, "finance/crypto/trading")
    assert set(storage.files) == {"finance/crypto/trading/f1.txt",
                                  "finance/crypto/trading/f1.summary.txt",
                                  "finance/crypto/trading/f1.en.txt"}

async def test_rocket_gets_summary(svc, notifier):
    ctx = ActionContext(file_id="f1", text="t", summary="s")
    await svc.run(ctx, RocketAction(type="rocket", channel="#c"), "x")
    assert notifier.calls == [("#c", "s", "f1.txt", "t")]

async def test_guard_failed_plan(svc, v):
    actions, path = await svc.node_plan(v, uid("trading"), guard_failed=True)
    assert actions == [SaveAction(type="save", path="_guard_failed")]

def test_fs_rejects_traversal(tmp_path):
    s = LocalFileStorage(tmp_path)
    with pytest.raises(ValueError):
        asyncio.run(s.write("../evil.txt", "x"))
    with pytest.raises(ValueError):
        asyncio.run(s.write("/etc/evil.txt", "x"))

def test_fs_writes_nested(tmp_path):
    asyncio.run(LocalFileStorage(tmp_path).write("a/b/c.txt", "привет"))
    assert (tmp_path / "a/b/c.txt").read_text("utf-8") == "привет"
```

- [ ] **Step 2:** запустить → FAIL.
- [ ] **Step 3:** Реализовать.
- [ ] **Step 4:** `uv run pytest tests/unit/application/test_actions.py tests/unit/adapters/test_fs_storage.py -v` → PASS.
- [ ] **Step 5: Stage**, не коммитить.

---

### Task 7: Temporal ClassifyWorkflow + activities

**Files:**
- Create: `src/llm_sorter/orchestration/activities.py`, `src/llm_sorter/orchestration/workflows.py`
- Test: `tests/unit/orchestration/test_classify_workflow.py`

**Interfaces:**
- Consumes: `ClassifyService` (Task 5), `ActionService` (Task 6).
- Produces:
  - `ClassifyInput(file_id: str, start_node_uid: str | None = None, version: int | None = None, from_other_uid: str | None = None)`; `ClassifyOutcome(status: Status, path: str, version: int, skipped: bool)`.
  - `class Activities` — конструктор принимает `ClassifyService`, `ActionService` (позже — `DiscoverService`, `MaterialRepository`, `WorkflowStarter`); методы `@activity.defn`: `active_version`, `root_uid`, `step`, `guard_check`, `finalize`, `load_context`, `node_plan`, `run_action`, `record_model_error(role: str)`. Каждый — тонкий прокси.
  - `@workflow.defn class ClassifyWorkflow`, `run(inp: ClassifyInput) -> ClassifyOutcome`:
    1. `version = inp.version or await active_version()`; `node = inp.start_node_uid or await root_uid(version)`.
    2. Цикл (не более `max_depth + 2` итераций — защита): `step`. Если `ActivityError` после ретраев → `record_model_error("decide")` и считать `other`. `descend` → `node = next_uid`; `other` → `finalize(IN_OTHER)`, выход; `leaf` → выход из цикла.
    3. На листе `guard_check`; `ActivityError` → считать `False` + `record_model_error("guard")`.
    4. `finalize(CLASSIFIED | GUARD_FAILED, from_other_uid)`; если `skipped` → вернуть outcome без actions.
    5. `node_plan`, затем `load_context` и `run_action` по очереди.
  - Таймауты/ретраи: модельные activities `start_to_close_timeout=120s`, `RetryPolicy(maximum_attempts=3, initial_interval=1s)`; `run_action` — `maximum_attempts=5`; остальные — 30s, дефолтный retry.
  - Data converter: `temporalio.contrib.pydantic.pydantic_data_converter` везде (клиент, тестовая среда).

- [ ] **Step 1: Failing tests** — `WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter)`, `Worker` с `Activities` на фейках:

```python
async def test_happy_path_runs_actions(env):
    out = await run_classify(env, "f1")          # decision: root→finance→crypto→trading, guard True
    assert out.status == Status.CLASSIFIED and out.path == "finance/crypto/trading"
    assert "finance/crypto/trading/f1.txt" in storage.files

async def test_guard_fail_default_action(env):   # ScriptedGuard(default=False)
    out = await run_classify(env, "f1")
    assert out.status == Status.GUARD_FAILED
    assert "_guard_failed/finance/crypto/trading/f1.txt" in storage.files

async def test_invalid_decide_falls_to_other(env):   # decision всегда "nft"
    out = await run_classify(env, "f1")
    assert out.status == Status.IN_OTHER and out.path == "other"
    assert metrics.counts["model_error:decide"] == 1

async def test_pinned_version(env):
    v1 = (await rules.active()).version
    await rules.publish(tree_without_crypto, base_version=v1, source="manual")
    out = await run_classify(env, "f1", ClassifyInput(file_id="f1", version=v1))
    assert out.version == v1 and out.path == "finance/crypto/trading"

async def test_second_run_skips_actions(env):
    await run_classify(env, "f1")
    storage.files.clear()
    out = await run_classify(env, "f1", ClassifyInput(file_id="f1"), workflow_id="f1-again")
    assert out.skipped and storage.files == {}
```

- [ ] **Step 2:** запустить → FAIL.
- [ ] **Step 3:** Реализовать activities и workflow (импорты сервисов в workflow — через `workflow.unsafe.imports_passed_through()`).
- [ ] **Step 4:** `uv run pytest tests/unit/orchestration -v` → PASS.
- [ ] **Step 5: Stage**, не коммитить.

---

### Task 8: LM Studio адаптеры

**Files:**
- Create: `src/llm_sorter/adapters/outbound/lmstudio.py`
- Test: `tests/unit/adapters/test_lmstudio.py`

**Interfaces:**
- Consumes: Protocols из `ports/models.py`.
- Produces (все принимают `client: openai.AsyncOpenAI, model: str`):
  - `LMStudioDecision.decide` — `chat.completions.create(temperature=0, response_format={"type":"json_schema","json_schema":{"name":"choice","strict":True,"schema":{"type":"object","properties":{"choice":{"type":"string","enum":[ids]}},"required":["choice"]}}})`. System-промпт: «выбери одну тему, к которой относится разговор; если ни одна не подходит — other»; user: список кандидатов (`id`, `title`, `description`, до 3 `examples`) + текст. Невалидный JSON / `choice` не из enum → `InvalidModelOutput`.
  - `LMStudioGuard.check` — схема `{"answer": enum ["yes","no"]}`, `temperature=0`.
  - `LMStudioText` — `translate`, `summarize` (3–5 предложений на `language`), `propose_topic` (schema под `TopicProposal`, `id` — `^[a-z0-9_]+$`; в промпте путь родителя и список соседей, чтобы не повторяться), `same_topic` (yes/no schema). `temperature=0.2` для генерации, 0 для `same_topic`.
  - `LMStudioEmbedder.embed` — `embeddings.create(model, input=texts)`, возвращает векторы по порядку.
  - `make_openai_client(cfg: AppConfig) -> AsyncOpenAI` (`base_url=cfg.lmstudio_url`, `api_key=cfg.lmstudio_api_key`, `timeout=120`).

- [ ] **Step 1: Failing tests** с `respx` на `http://test/v1/chat/completions` и `/v1/embeddings`:
  `test_decide_returns_choice` (ответ `{"choice":"crypto"}` → `"crypto"`; в теле запроса `temperature == 0` и enum `["crypto","banks","other"]`), `test_decide_out_of_enum_raises` (`{"choice":"nft"}` → `InvalidModelOutput`), `test_decide_bad_json_raises` (контент `"crypto"` → `InvalidModelOutput`), `test_guard_yes_no` (`yes` → True, `no` → False), `test_propose_topic_parses`, `test_embed_order` (два вектора в порядке входа).
- [ ] **Step 2:** запустить → FAIL.
- [ ] **Step 3:** Реализовать.
- [ ] **Step 4:** `uv run pytest tests/unit/adapters/test_lmstudio.py -v` → PASS.
- [ ] **Step 5: Stage**, не коммитить.

---

### Task 9: Postgres адаптеры + миграции

**Files:**
- Create: `alembic.ini`, `migrations/env.py`, `migrations/versions/0001_init.py`, `src/llm_sorter/adapters/outbound/postgres/{__init__,tables,rules,transcripts,classifications,material,embeddings}.py`
- Test: `tests/integration/test_postgres.py` (маркер `integration`, требует `make up && make migrate`)

**Interfaces:**
- Consumes: Protocols из `ports/storage.py`.
- Produces: `make_engine(url) -> AsyncEngine`; `PgRules`, `PgTranscripts`, `PgClassifications`, `PgMaterial`, `PgEmbeddings` (каждый принимает `async_sessionmaker`).
- Таблицы:
  - `transcripts(id text pk, text text, metadata jsonb, received_at timestamptz)`
  - `rule_versions(version serial pk, yaml text, source text, base_version int null, created_at timestamptz default now())`
  - `active_rules(id int pk check id=1, version int fk)`
  - `classifications(file_id text pk, tree_version int, node_uid text, path text, status text, updated_at timestamptz)`
  - `material(node_uid text, file_id text, kind text, added_at timestamptz default now(), pk(node_uid, file_id))`
  - `discover_state(node_uid text pk, last_discover_at timestamptz)`
  - `embeddings(file_id text pk, vector vector(1024))`, `CREATE EXTENSION IF NOT EXISTS vector`.
- `PgRules.publish`: в одной транзакции `SELECT version FROM active_rules FOR UPDATE`; не совпало с `base_version` → `VersionConflict`; `INSERT rule_versions` с `dump_tree(tree)`; `UPDATE active_rules`. Если активной нет — `base_version` должен быть `None`. `get` кэширует распарсенные версии.
- `PgClassifications.save`: `SELECT … FOR UPDATE`; если существующая `status != in_other` → `FinalClassificationExists`; иначе upsert.
- `PgMaterial.count_new`: `count(*) where added_at > coalesce(last_discover_at, '-infinity')`; `due_nodes(threshold)` — узлы, где `count_new >= threshold`; `move_other(from, to)` — `UPDATE … SET node_uid=to WHERE node_uid=from AND kind='other' ON CONFLICT DO NOTHING` (через insert-select + delete).

- [ ] **Step 1: Failing tests:** `test_publish_and_active`, `test_publish_conflict` (два publish с одним `base_version` → второй `VersionConflict`), `test_classification_final_immutable`, `test_material_count_new_resets_after_mark`, `test_move_other`, `test_transcript_put_dedup` (второй `put` → False), `test_embeddings_roundtrip`. Каждый тест — в транзакции с откатом / `TRUNCATE` в фикстуре.
- [ ] **Step 2:** `make up && make migrate && make test-int` → FAIL.
- [ ] **Step 3:** Реализовать таблицы, миграцию `0001_init`, репозитории.
- [ ] **Step 4:** `make test-int` → PASS.
- [ ] **Step 5: Stage**, не коммитить.

---

### Task 10: RulesService + CLI для правил

**Files:**
- Create: `src/llm_sorter/application/rules.py`, `src/llm_sorter/adapters/inbound/cli.py`, `src/llm_sorter/bootstrap.py`
- Modify: `pyproject.toml` — `[project.scripts] llm-sorter = "llm_sorter.adapters.inbound.cli:app"`
- Test: `tests/unit/application/test_rules.py`

**Interfaces:**
- Consumes: `load_tree`, `dump_tree`, `validate_tree`, `tree_edit.*`, `RulesRepository`, `MaterialRepository`, `WorkflowStarter`, `Metrics`.
- Produces:
  - `PublishResult(version: int, reprocess_nodes: list[str], reprocess_started: int)`.
  - `RulesService(rules, material, starter, metrics)`:
    - `async publish_tree(new: RuleTree, base_version: int | None, source) -> PublishResult` — `validate_tree`; `merge_archived(old, new)`; `rules.publish`; для каждого uid из `archived`, которого не было в `old.archived`: `material.move_other(uid, live_ancestor_with_other(old, new, uid))`; `reprocess_nodes = [u for u in gained_children(old, new) if await material.other_size(u) > 0]`; по каждому файлу с `kind == "other"` в этих узлах `starter.start_reprocess(file_id, u, version)`; `metrics.rules_published(source)`, `metrics.active_version(version)`.
    - `async publish_yaml(text: str, base_version: int | None) -> PublishResult` (source `manual`).
    - `async add_discovered(parent_uid: str, proposals: list[TopicProposal]) -> PublishResult` — до 3 попыток: `active()` → `add_children` → `publish_tree(..., base_version=active.version, source="discover")`; на `VersionConflict` повтор.
    - `async pull() -> str` — `dump_tree(active())`.
    - `async rollback(version: int) -> PublishResult` — `publish_tree(get(version) с version=None, base_version=active.version, source="rollback")`.
    - `async bootstrap_if_empty() -> None` — если активной версии нет, публикует `node: {id: root}`.
  - `bootstrap.py`: `async build_app(cfg: AppConfig) -> App` (dataclass со всеми адаптерами и сервисами); `App.close()`.
  - CLI (typer): `llm-sorter rules publish FILE [--base N]` (без `--base` берёт версию из поля `version` в файле), `rules pull [-o FILE]`, `rules rollback VERSION`, `rules show` (дерево с путями и origin); `worker`, `ingest`, `consume` — регистрируются в Task 11/12.

- [ ] **Step 1: Failing tests** (фейки):

```python
async def test_publish_invalid_rejected(svc):
    with pytest.raises(TreeValidationError):
        await svc.publish_yaml("node: {id: root, children: [{id: other, title: x, description: y}]}", base_version=1)
    assert (await rules.active()).version == 1

async def test_manual_add_child_reprocesses_other_buffer(svc, starter):
    await material.add(root_uid, "f1", "other")
    res = await svc.publish_yaml(yaml_with_new_root_child, base_version=1)
    assert res.reprocess_nodes == [root_uid]
    assert starter.reprocess == [("f1", root_uid, res.version)]

async def test_delete_moves_other_buffer_up(svc):
    await material.add(uid("crypto_with_children"), "f2", "other")
    await svc.publish_yaml(yaml_without_that_subtree, base_version=1)
    assert MaterialItem(file_id="f2", kind="other") in await material.list(uid("finance"))

async def test_add_discovered_retries_on_conflict(svc, rules):
    rules.conflict_once = True
    res = await svc.add_discovered(root_uid, [proposal("taxes")])
    assert any(n.id == "taxes" for n in (await rules.active()).iter_nodes())

async def test_rollback_archives_discovered(svc):
    v1 = (await rules.active()).version
    await svc.add_discovered(root_uid, [proposal("taxes")])
    await svc.rollback(v1)
    assert [a.path for a in (await rules.active()).archived] == ["taxes"]
```

- [ ] **Step 2:** запустить → FAIL.
- [ ] **Step 3:** Реализовать `RulesService`, `bootstrap.py`, CLI-команды `rules *`.
- [ ] **Step 4:** `uv run pytest tests/unit/application/test_rules.py -v` → PASS; `uv run llm-sorter --help` показывает группу `rules`.
- [ ] **Step 5: Stage**, не коммитить.

---

### Task 11: Ingest: заглушка источника, поллер, RabbitMQ, consumer, worker

**Files:**
- Create: `src/llm_sorter/adapters/inbound/stub_source.py`, `src/llm_sorter/adapters/inbound/poller.py`, `src/llm_sorter/adapters/inbound/rabbit.py`, `src/llm_sorter/adapters/outbound/temporal_starter.py`, `src/llm_sorter/orchestration/worker.py`
- Modify: `src/llm_sorter/adapters/inbound/cli.py` (команды `ingest`, `consume`, `worker`)
- Test: `tests/unit/adapters/test_stub_source.py`, `tests/unit/adapters/test_consumer.py`, `tests/integration/test_e2e.py`

**Interfaces:**
- Produces:
  - `LocalDirSource(dir: Path, seen_file: Path)`: `fetch_new()` читает `*.txt` (id = имя без расширения, text = содержимое) и `*.json` (`{"id","text","metadata"}`), пропускает id из `seen_file` (JSON-список), дописывает новые.
  - `async run_poller(source, publisher, interval_s: float, stop: asyncio.Event) -> None`.
  - `RabbitPublisher(url, queue)` — `publish` в durable-очередь, `delivery_mode=PERSISTENT`, тело `Transcript.model_dump_json()`.
  - `async handle_message(body: bytes, transcripts: TranscriptStore, starter: WorkflowStarter) -> None` — парсит `Transcript`, `transcripts.put`, `starter.start_classify(id)` (False = уже запущен, не ошибка). `run_consumer(url, queue, transcripts, starter)` — `prefetch_count=10`, ack после `handle_message`, исключение → `nack(requeue=True)`; битый JSON → `reject(requeue=False)` + лог.
  - `TemporalStarter(client: Client, task_queue: str)` — `start_workflow` с id из Global Constraints, `id_reuse_policy=REJECT_DUPLICATE`; `WorkflowAlreadyStartedError` → False.
  - `async run_worker(cfg: AppConfig) -> None` — `build_app`, `bootstrap_if_empty`, `start_http_server(cfg.metrics_port)` (если метрики подключены), Worker на `cfg.task_queue` со всеми workflows/activities.

- [ ] **Step 1: Failing tests:**

```python
async def test_stub_source_dedups(tmp_path):
    (tmp_path / "in").mkdir(); (tmp_path / "in/a.txt").write_text("hi")
    src = LocalDirSource(tmp_path / "in", tmp_path / "seen.json")
    assert [t.id for t in await src.fetch_new()] == ["a"]
    assert await src.fetch_new() == []

async def test_handle_message_twice_starts_once(transcripts, starter):
    body = Transcript(id="x", text="t", received_at=now).model_dump_json().encode()
    await handle_message(body, transcripts, starter)
    await handle_message(body, transcripts, starter)       # FakeStarter вернёт False, исключения нет
    assert starter.classify == ["x"]
```

  `tests/integration/test_e2e.py` (маркер `integration`; нужны `make up`, `make migrate`, LM Studio не нужен — подменяем модели на `ScriptedDecision`/`ScriptedGuard` через `build_app(cfg, overrides=...)`): кладём `a.txt` в `source_dir`, поднимаем poller+consumer+worker в фоне, ждём до 30 с, проверяем `classifications.get("a").status == Status.IN_OTHER` и `path == "other"` (холодный старт, дерево только root).
- [ ] **Step 2:** запустить → FAIL.
- [ ] **Step 3:** Реализовать; `build_app` принимает `overrides: dict[str, Any] | None` для подмены адаптеров в тестах.
- [ ] **Step 4:** `make test` → PASS; `make test-int` → PASS.
- [ ] **Step 5: Stage**, не коммитить.

---

### Task 12: Discover + переобработка + sweep

**Files:**
- Create: `src/llm_sorter/application/clustering.py`, `src/llm_sorter/application/discover.py`
- Modify: `src/llm_sorter/orchestration/activities.py` (`discover_run`, `due_nodes`, `start_discover`), `src/llm_sorter/orchestration/workflows.py` (`DiscoverWorkflow`, `SweepWorkflow`), `src/llm_sorter/orchestration/worker.py` (`ensure_sweep_schedule`)
- Test: `tests/unit/application/test_clustering.py`, `tests/unit/application/test_discover.py`, `tests/unit/orchestration/test_discover_workflow.py`

**Interfaces:**
- Consumes: `should_discover`, `slots_left`, `dedup_verdict`, `candidates_for`, `RulesService.add_discovered` (сам запускает переобработку буфера `other`), порты.
- Produces:
  - `clustering.py`: `cluster(vectors: np.ndarray, min_cluster_size: int) -> list[list[int]]` — L2-нормализация, `sklearn.cluster.HDBSCAN(min_cluster_size=..., metric="euclidean")`, метка -1 отбрасывается, кластеры по убыванию размера; меньше `min_cluster_size` точек → `[]`. `nearest_to_centroid(vectors, idxs: list[int], k: int) -> list[int]`. `cosine(a, b) -> float`.
  - `DiscoverResult(version: int | None, created: list[str], rejected: int)`.
  - `DiscoverService(rules, rules_service: RulesService, material, transcripts, cache: EmbeddingCache, embedder, text_llm, metrics)`:
    `async run(node_uid: str) -> DiscoverResult`:
    1. `tree = active()`; узел архивный или `not should_discover(tree, uid, count_new)` → `DiscoverResult(None, [], 0)`.
    2. Материал = `material.list(uid)`; тексты; эмбеддинги (кэш, недостающие — `embed`, батчами по 16, затем `put_many`).
    3. `cluster(...)`; для каждого кластера, пока `slots_left > len(accepted)`: сэмплы = `nearest_to_centroid(k=samples_per_cluster)`, тексты обрезаны до `sample_chars`; `propose_topic(samples, path_of(uid), candidates_for без other, language)`.
    4. Дедуп: эмбеддинг `title + ". " + description` нового против живых соседей и уже принятых: `dedup_verdict(max_sim)`; `ask` → `same_topic`; `dup` или `same_topic=True` → `rejected += 1`. Конфликт `id` с соседом → суффикс `_2`, `_3`.
    5. Принятые есть → `rules_service.add_discovered(uid, accepted)`; `material.mark_discovered(uid)` в любом случае после пункта 1; `metrics.discover_run`.
  - `@workflow.defn DiscoverWorkflow.run(node_uid: str) -> DiscoverResult` — одна activity `discover_run`, `start_to_close_timeout=30min`, `maximum_attempts=2`.
  - `@workflow.defn SweepWorkflow.run() -> int` — `due_nodes(threshold из активного дерева)` → `start_discover` по каждому; возвращает число запущенных.
  - `ensure_sweep_schedule(client, cfg)` — Temporal Schedule `llm-sorter-sweep`, интервал `cfg.sweep_interval_s`, `overlap=SKIP`; создаёт, если нет.

- [ ] **Step 1: Failing tests:**

```python
def test_cluster_two_groups():
    rng = np.random.default_rng(0)
    a = rng.normal([1, 0, 0], 0.01, (12, 3)); b = rng.normal([0, 1, 0], 0.01, (12, 3))
    groups = cluster(np.vstack([a, b]), min_cluster_size=10)
    assert sorted(len(g) for g in groups) == [12, 12]

def test_cluster_too_small():
    assert cluster(np.eye(3), min_cluster_size=10) == []

async def test_discover_creates_topic_and_reprocesses(svc, starter):
    # 12 файлов в root/other с почти одинаковыми эмбеддингами, threshold=10, min_cluster_size=10
    res = await svc.run(root_uid)
    assert res.created == ["taxes"]
    assert len(starter.reprocess) == 12

async def test_discover_leaf_material_not_reprocessed(svc, starter):
    # 12 файлов kind="leaf" в crypto (depth 2)
    res = await svc.run(uid("crypto"))
    assert res.created and starter.reprocess == []
    assert all((await classifications.get(f)).path == "finance/crypto" for f in leaf_files)

async def test_dedup_against_sibling(svc, text_llm):
    text_llm.next_proposal = proposal("banks_2", title="Банки")   # эмбеддинг ≈ существующего banks
    res = await svc.run(uid("finance"))
    assert res.created == [] and res.rejected == 1

async def test_respects_max_children(svc):          # max_children = len(children) → slots 0
    assert (await svc.run(uid("finance"))).version is None

async def test_counter_resets(svc, material):
    await svc.run(root_uid)
    assert await material.count_new(root_uid) == 0
```

  Workflow-тест: `SweepWorkflow` с `FakeMaterial.due_nodes → [u1, u2]` → возвращает 2, `starter.discover == [u1, u2]`.
- [ ] **Step 2:** запустить → FAIL.
- [ ] **Step 3:** Реализовать; переобработка — `TemporalStarter.start_reprocess` запускает `ClassifyWorkflow` с `ClassifyInput(file_id, start_node_uid=node_uid, version=version, from_other_uid=node_uid)`.
- [ ] **Step 4:** `make test` → PASS.
- [ ] **Step 5: Stage**, не коммитить.

---

### Task 13: Метрики Prometheus + CLAUDE.md

**Files:**
- Create: `src/llm_sorter/adapters/outbound/prometheus_metrics.py`, `deploy/prometheus.yml`, `deploy/grafana/provisioning/datasources/prometheus.yml`, `CLAUDE.md`
- Modify: `src/llm_sorter/bootstrap.py` (подключить `PrometheusMetrics`), `docker-compose.yml` (монтирование provisioning)
- Test: `tests/unit/adapters/test_prometheus_metrics.py`

**Interfaces:**
- Consumes: `Metrics` Protocol.
- Produces: `PrometheusMetrics(registry: CollectorRegistry | None = None)`. Метрики (префикс `llm_sorter_`):
  `files_classified_total{path,status,version}`, `model_calls_total{role,ok}`, `model_latency_seconds{role}` (Histogram), `model_errors_total{role}`, `guard_results_total{path,passed}`, `actions_total{type,ok}`, `action_latency_seconds{type}`, `other_buffer_size{path}` (Gauge), `discover_runs_total{path}`, `discover_topics_created_total{path}`, `discover_topics_rejected_total{path}`, `reprocess_moved_total{from_path,to_path}`, `topic_move_attempts_total` (alert: > 0), `active_rules_version` (Gauge), `rules_published_total{source}`.
  `buffer_size` обновляется в `ClassifyService.finalize` и `DiscoverService.run` (`material.other_size`).

- [ ] **Step 1: Failing test:** с отдельным `CollectorRegistry` вызвать `topic_move_attempt()` дважды и `file_classified("finance", Status.CLASSIFIED, 3)`; `registry.get_sample_value("llm_sorter_topic_move_attempts_total") == 2`; `get_sample_value("llm_sorter_files_classified_total", {"path":"finance","status":"classified","version":"3"}) == 1`.
- [ ] **Step 2:** запустить → FAIL.
- [ ] **Step 3:** Реализовать адаптер и подключить в `bootstrap`.
- [ ] **Step 4:** Написать `CLAUDE.md` (стандартный префикс `# CLAUDE.md` + «This file provides guidance…»): команды Makefile (включая одиночный тест `uv run pytest tests/unit/application/test_classify.py::test_final_is_immutable -v`), карта слоёв и правило импортов (проверяется `tests/unit/test_layers.py`), поток ingest → Rabbit → ClassifyWorkflow → discover → reprocess, инварианты (финальность, пин версии, `other` только у узлов с детьми, id workflow), правила работы с деревом через CLI, запрет коммитов без подтверждения. Ссылка на spec.
- [ ] **Step 5:** `make lint typecheck test` → всё зелёное.
- [ ] **Step 6: Stage**, не коммитить; попросить пользователя посмотреть финальный diff.
