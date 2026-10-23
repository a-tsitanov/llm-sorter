# llm-sorter v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Переписать llm-sorter в три процесса (фидеры → обработчики → дообучение), связанных RabbitMQ и SQLite, без Temporal и Postgres.

**Architecture:** Фидеры публикуют в очередь `llm-sorter.inbox`. Обработчики (async, без состояния) держат дерево из SQLite в памяти, спускаются до листа, проверяют гейт и публикуют в topic exchange `llm-sorter.classified` либо в stream `llm-sorter.other`. Дообучение читает stream, копит буферы по узлам, по порогу кластеризует, добавляет темы в SQLite и перегоняет сообщения обратно в inbox. Состояние сообщений живёт только в stream (служебные записи `resolved`/`considered`), в SQLite — дерево, журнал, позиция в stream.

**Tech Stack:** Python 3.12, uv, aio-pika, aiosqlite, openai (async, OpenAI-совместимый сервер), pydantic(-settings), numpy, scikit-learn, telethon, typer, pyyaml, prometheus-client, pytest + pytest-asyncio, RabbitMQ 3.13 (streams через AMQP 0.9.1).

**Spec:** `docs/superpowers/specs/2026-10-23-llm-sorter-v2-design.md`

## Global Constraints

- Python `>=3.12,<3.14`, только `uv` (никогда bare `pip`).
- Комментарии и docstring на русском, идентификаторы на английском (`CLAUDE.md`).
- **Коммит / push — только после явного «да» пользователя.** Каждый шаг «Commit» = подготовить `git add` и спросить; без ответа не коммитить.
- Незакоммиченные правки пользователя (`clustering.py`, `discover.py`, `lmstudio.py`, `tree.py`, `ports/models.py`, `tests/fakes.py`, их тесты) — переносятся, не теряются: в частности `diverse_samples`, `propose_topic(..., level, max_depth)`, `samples_per_cluster=20`, `sample_chars=6000`.
- Зависимости: удалить `temporalio`, `sqlalchemy`, `asyncpg`, `alembic`, `pgvector`; добавить `aiosqlite`. Других новых зависимостей нет.
- Инфраструктура для запуска — только RabbitMQ (+ опционально Prometheus/Grafana). Всё на одной машине.
- Имена Rabbit по умолчанию: inbox `llm-sorter.inbox`, DLQ `llm-sorter.inbox.dlq`, exchange `llm-sorter.classified` (topic, durable), архив `llm-sorter.classified.all` (binding `#`), stream `llm-sorter.other` (`x-max-age=30D`, `x-max-length-bytes=10737418240`).
- SQLite: файл `SQLITE_PATH` (по умолчанию `./data/sorter.db`), `PRAGMA journal_mode=WAL`, схема кодом, версия в `PRAGMA user_version`.
- `Transcript.id`: `^[A-Za-z0-9._-]+$` без `..`. `Topic.id`: `^[a-z0-9_]+$`, не `other`/`root`, глобально уникален.
- Пороги по умолчанию: `buffer_threshold=50`, `min_cluster_size=10`, `max_children=12`, `max_depth=3`, `max_reprocess=3`, `MODEL_ATTEMPTS=3`, backoff сбоя 1 с ×2 до 300 с.
- Слои: `domain` ← `ports` ← `application` ← `adapters`; `tests/unit/test_layers.py` остаётся (domain не импортирует aio_pika/aiosqlite/openai/telethon/httpx; application — только domain и ports).

## Review Focus

1. Пустое дерево (только `root`) — первое сообщение уходит в `other(root)` без вызова LLM. → тест в Task 5.
2. LLM недоступна в момент обработки — сообщение ждёт и обрабатывается после восстановления ровно один раз, не попадает ни в `other`, ни в DLQ. → тест в Task 6.
3. Discover предлагает `id`, который уже занят в другой ветке (`cards` под `finance` и под `tech`) — новый `id` получает суффикс, routing key однозначен. → тест в Task 7.
4. Перезапуск дообучения посреди накопления — буферы восстанавливаются из stream с сохранённой позиции, перегнанные сообщения не возвращаются в буфер. → тест в Task 8.
5. Заголовок `x-reprocess` отсутствует или мусорный (`"abc"`, `-1`) — считается 0, сообщение обрабатывается. → тест в Task 6.

---

## File Structure

```
src/llm_sorter/
  config.py                         # переписать: настройки v2
  domain/
    tree.py                         # переписать: Topic, Settings, Tree (id-based, без uid/версий)
    changes.py                      # новый: AddTopic/MoveTopic/EditTopic/ArchiveTopic, apply_changes
    traversal.py                    # адаптировать под id
    validation.py                   # validate_tree + guard_is_meaningful
    discover.py                     # TopicProposal, should_discover, dedup_verdict, proposal_is_valid
    messages.py                     # новый: Transcript, TopicRef, Classified, OtherItem, Reason, StreamRecord
    loader.py                       # переписать: YAML плоским списком (import/export)
  ports/
    models.py                       # DecisionModel, GuardModel, TextLLM (без translate/summarize), Embedder
    store.py                        # новый: TopicStore
    bus.py                          # новый: Bus
    io.py                           # TranscriptSource, MessagePublisher (без WorkflowStarter)
    metrics.py                      # переписать
  application/
    process.py                      # новый: ProcessService
    discover.py                     # переписать: DiscoverService.propose (без хранилищ)
    learner.py                      # новый: Learner
    topics.py                       # новый: TopicsService (CLI-правки, import/export)
    clustering.py                   # без изменений
  adapters/
    inbound/
      cli.py                        # переписать
      poller.py, stub_source.py, telegram_source.py   # без изменений
      processor.py                  # новый: consumer inbox
      learner_runner.py             # новый: consumer stream
    outbound/
      llm.py                        # переименование lmstudio.py (git mv), без translate/summarize
      sqlite_store.py               # новый: SqliteTopicStore
      rabbit.py                     # новый: topology, RabbitBus, InboxPublisher
      prometheus_metrics.py, noop_metrics.py          # переписать под новый Metrics
  bootstrap.py                      # переписать
```

Удаляются: `orchestration/`, `adapters/outbound/postgres/`, `temporal_starter.py`, `null_starter.py`, `fs_storage.py`, `rocket_stub.py`, `adapters/inbound/rabbit.py` (заменён `processor.py` + `outbound/rabbit.py`), `application/{actions,classify,rules}.py`, `domain/{actions,classification,tree_edit}.py`, `ports/storage.py`, `migrations/`, `alembic.ini`, их тесты, `tests/integration/*` (кроме переписанного `conftest.py`).

---

### Task 0: Базовая точка

**Files:** нет изменений кода.

- [ ] **Step 1:** `git status` — убедиться, что в рабочей копии правки пользователя и TG-фидер (сессия 2026-10-22).
- [ ] **Step 2:** Спросить пользователя: закоммитить текущее состояние на `main` (сообщение `feat: telegram feeder + discover tuning (pre-v2)`) и создать ветку `v2`. **Без ответа — не продолжать.**
- [ ] **Step 3:** После «да»: коммит, `git switch -c v2`. `make test` → все зелёные (237 passed).

---

### Task 1: Снос v1-инфраструктуры

**Files:**
- Delete: всё из списка «Удаляются» выше, кроме `domain/classification.py` (переедет в Task 2) и `application/classify.py` (нужен как источник логики до Task 5).
- Modify: `pyproject.toml`, `uv.lock` (`uv remove temporalio sqlalchemy asyncpg alembic pgvector`, `uv add aiosqlite`), `Makefile` (убрать `migrate`, `worker`, `ingest`, `consume`, `test-int` пока оставить), `docker-compose.yml`, `deploy/prometheus/prometheus.yml`, `deploy/prometheus/rules.yml`.
- Modify: `src/llm_sorter/adapters/inbound/cli.py` — оставить только `tg-folders`; `bootstrap.py` удалить.

- [ ] **Step 1:** Удалить файлы и тесты; `docker-compose.yml`: оставить `rabbitmq` (образ `rabbitmq:3.13-management`, порты через `RABBIT_PORT`/`RABBIT_MGMT_PORT`), `prometheus`, `grafana`; удалить `postgres`, `temporal`, `temporal-ui`. Prometheus scrape: `host.docker.internal:9100` (processor) и `:9101` (learner). Из `rules.yml` убрать `TopicMoveAttempt`, `DiscoverBlocked` переписать на `llm_sorter_discover_blocked{node} == 1` for 30m.
- [ ] **Step 2:** Run `uv run ruff check src tests && uv run pytest -m "not integration" -q`. Expected: зелёные оставшиеся тесты (domain, clustering, telegram_source, stub_source, lmstudio, config — config-тест обновится в Task 2; на этом шаге допустимо пометить его `xfail` с причиной «Task 2»). `uv run python -c "import llm_sorter.adapters.inbound.cli"` без ошибок.
- [ ] **Step 3:** `docker compose config -q` — без ошибок.
- [ ] **Step 4: Commit** (после подтверждения): `chore: remove temporal, postgres and actions (v2 groundwork)`.

---

### Task 2: Домен — дерево, изменения, сообщения, конфиг

**Files:**
- Rewrite: `src/llm_sorter/domain/tree.py`, `traversal.py`, `validation.py`, `discover.py`
- Create: `src/llm_sorter/domain/changes.py`, `src/llm_sorter/domain/messages.py`
- Delete: `src/llm_sorter/domain/classification.py` (`Transcript` → `messages.py`; поправить импорты в `stub_source.py`, `telegram_source.py`, `poller.py`, `ports/io.py`)
- Rewrite: `src/llm_sorter/config.py`
- Test: `tests/unit/domain/test_tree.py`, `test_changes.py`, `test_messages.py`, `test_validation.py`, `test_traversal.py`, `test_discover_rules.py`, `tests/unit/test_config.py`

**Interfaces — Produces:**

```python
# domain/tree.py
OTHER_ID = "other"; ROOT_ID = "root"
class DiscoverSettings(_Frozen): buffer_threshold=50; min_cluster_size=10; max_children=12
    dedup_similarity=0.85; llm_dedup_floor=0.70; samples_per_cluster=20; sample_chars=6000
class Settings(_Frozen): max_depth=3; language="ru"; max_reprocess=3; discover=DiscoverSettings()
class Topic(_Frozen): id: str; parent_id: str | None; title=""; description=""
    examples: list[str]=[]; guard: str | None=None; origin: Literal["manual","discovered"]="manual"
    max_depth: int | None=None
@dataclass(frozen=True) class Tree:
    rev: int; settings: Settings; topics: dict[str, Topic]   # порядок вставки = порядок детей
    def get(self, id: str) -> Topic                           # KeyError
    def children(self, id: str) -> list[Topic]
    def parent(self, id: str) -> Topic | None
    def path(self, id: str) -> list[str]                      # ids без root: ["finance","cards"]
    def routing_key(self, id: str) -> str                     # "finance.cards"; root -> ""
    def depth(self, id: str) -> int                           # root 0
    def max_depth_for(self, id: str) -> int                   # ближайший max_depth у предков, иначе settings
    def has_other(self, id: str) -> bool                      # root или есть дети
    def is_leaf(self, id: str) -> bool                        # не root и нет детей
    @staticmethod def empty(settings: Settings | None = None) -> "Tree"   # только root, rev 0

# domain/changes.py
class AddTopic(_Frozen): topic: Topic
class MoveTopic(_Frozen): id: str; new_parent_id: str
class EditTopic(_Frozen): id: str; fields: dict[str, Any]     # title/description/examples/guard/max_depth
class ArchiveTopic(_Frozen): id: str                           # с потомками
Change = AddTopic | MoveTopic | EditTopic | ArchiveTopic
def apply_changes(tree: Tree, changes: Sequence[Change]) -> Tree   # rev+1; validate_tree; TreeValidationError

# domain/validation.py
ID_RE = re.compile(r"^[a-z0-9_]+$")
class TreeValidationError(ValueError): errors: list[str]
def guard_is_meaningful(q: str | None) -> bool
def validate_tree(tree: Tree) -> None    # id, title, description, глубина, max_children не проверяет (это discover), цикл parent

# domain/traversal.py
class Candidate(_Frozen): id; title; description; examples=[]
def candidates_for(tree: Tree, node_id: str) -> list[Candidate]   # дети + other последним; лист -> []
def resolve_choice(tree: Tree, node_id: str, choice_id: str) -> str | None   # None для other
def guard_question(t: Topic) -> str      # t.guard или прежний fallback-вопрос

# domain/discover.py
class TopicProposal(_Frozen): id; title; description; examples: list[str]; guard: str
def slots_left(tree: Tree, node_id: str) -> int
def should_discover(tree: Tree, node_id: str, new_count: int) -> bool   # порог, глубина < max_depth_for, слоты
def dedup_verdict(similarity: float, s: DiscoverSettings) -> Literal["dup","ask","new"]
def proposal_is_valid(p: TopicProposal) -> bool   # + guard_is_meaningful(p.guard), id не other/root

# domain/messages.py
class Transcript(_Frozen): id (pattern как сейчас, без ".."); text; metadata: dict={}; received_at: datetime
Reason = Literal["no_match", "gate_failed", "model_error", "leaf"]
class TopicRef(_Frozen): id; path: str; title
class Classified(_Frozen): id; text; metadata; topic: TopicRef; trail: list[str]; gate: Literal["passed"]="passed"
    tree_rev: int; classified_at: datetime
class OtherItem(_Frozen): id; text; metadata; received_at: datetime; node: str; trail: list[str]; reason: Reason
    tree_rev: int; reprocess: int = 0; final: bool = False; at: datetime
    def transcript(self) -> Transcript
class Resolved(_Frozen): kind: Literal["resolved"]="resolved"; ids: list[str]; at: datetime
class Considered(_Frozen): kind: Literal["considered"]="considered"; node: str; ids: list[str]; at: datetime
StreamRecord = OtherItem | Resolved | Considered     # OtherItem сериализуется с kind="item"
def dump_record(r: StreamRecord) -> bytes; def load_record(b: bytes) -> StreamRecord
```

`config.AppConfig` (без префикса, `.env`): `llm_url="http://localhost:1234/v1"`, `llm_api_key: SecretStr`, `model_decide`, `model_guard`, `model_text`, `model_embed` (текущие значения по умолчанию), `sqlite_path=Path("./data/sorter.db")`, `rabbit_url`, `inbox_queue`, `classified_exchange`, `classified_archive`, `other_stream`, `other_max_age="30D"`, `other_max_bytes=10737418240`, `process_concurrency=16`, `tree_refresh_s=1.0`, `learner_check_s=30.0`, `other_refresh_s=86400.0`, `metrics_port=9100`, `learner_metrics_port=9101`, `source_dir`, `poll_interval_s`, `tg_*` (как сейчас, без `ingest_source`).

- [ ] **Step 1: Failing tests** (ключевые):

```python
def test_routing_key_and_leaf(tree3):   # root -> finance -> cards; root -> tech
    assert tree3.routing_key("cards") == "finance.cards"
    assert tree3.is_leaf("cards") and not tree3.is_leaf("finance") and not tree3.is_leaf("root")
    assert tree3.has_other("root") and tree3.has_other("finance") and not tree3.has_other("cards")

def test_apply_changes_bumps_rev_and_validates(tree3):
    t = apply_changes(tree3, [AddTopic(topic=Topic(id="loans", parent_id="finance", title="Кредиты", description="..."))])
    assert t.rev == tree3.rev + 1 and [c.id for c in t.children("finance")] == ["cards", "loans"]
    with pytest.raises(TreeValidationError, match="duplicate id"):
        apply_changes(tree3, [AddTopic(topic=Topic(id="cards", parent_id="tech", title="x", description="y"))])
    with pytest.raises(TreeValidationError, match="cycle"):
        apply_changes(tree3, [MoveTopic(id="finance", new_parent_id="cards")])

def test_archive_removes_subtree(tree3):
    assert "cards" not in apply_changes(tree3, [ArchiveTopic(id="finance")]).topics

@pytest.mark.parametrize("q,ok", [
    (None, False), ("", False), ("yes", False), ("Да", False),
    ("Да, если в разговоре упоминаются технологии, тренды, обновления", False),
    ("Пост о банковских картах или их тарифах?", True)])
def test_guard_is_meaningful(q, ok): assert guard_is_meaningful(q) is ok

def test_stream_record_roundtrip(): ...   # OtherItem/Resolved/Considered -> dump -> load == исходное
```

- [ ] **Step 2:** Run `uv run pytest tests/unit/domain tests/unit/test_config.py -q` → FAIL (ImportError).
- [ ] **Step 3:** Реализовать. `guard_is_meaningful`: после `strip()` длина ≥ 15, содержит `?`, первое слово (без пунктуации, lower) не в `{"yes","no","да","нет"}`. `apply_changes` копирует `topics`, применяет по порядку, `rev+1`, вызывает `validate_tree`.
- [ ] **Step 4:** Run → PASS; `uv run mypy src` чисто.
- [ ] **Step 5: Commit** (после подтверждения): `feat(domain): id-based topic tree, changes, stream records`.

---

### Task 3: SQLite-хранилище

**Files:**
- Create: `src/llm_sorter/ports/store.py`, `src/llm_sorter/adapters/outbound/sqlite_store.py`
- Test: `tests/unit/adapters/test_sqlite_store.py`

**Interfaces — Consumes:** `Tree`, `Change`, `apply_changes`, `Settings` (Task 2).
**Produces:**

```python
class TopicStore(Protocol):
    async def tree_rev(self) -> int
    async def load_tree(self) -> Tree                       # пустая БД -> Tree.empty() с root в БД
    async def apply(self, changes: Sequence[Change], actor: Literal["learner","cli"]) -> Tree
    async def save_settings(self, s: Settings) -> Tree      # тоже tree_rev+1 и changelog op="settings"
    async def changelog(self, limit: int = 50) -> list[ChangelogEntry]
    async def get_state(self, key: str) -> str | None
    async def set_state(self, key: str, value: str) -> None

@dataclass(frozen=True) class ChangelogEntry: at: datetime; actor: str; op: str; topic_id: str | None; details: dict

class SqliteTopicStore(TopicStore):
    @classmethod async def open(cls, path: Path) -> "SqliteTopicStore"   # WAL, создаёт схему, user_version=1
    async def close(self) -> None
```

Схема — как в спеке (`topics`, `settings`, `meta(tree_rev)`, `changelog`, `learner_state`); `examples` — JSON-текст; архив = `archived_at IS NOT NULL` (в `load_tree` не попадают). `apply`: читает дерево, `apply_changes` (валидация до записи), одна транзакция `BEGIN IMMEDIATE`: INSERT/UPDATE строк, `changelog` по одной записи на изменение, `tree_rev = tree_rev + 1`. Конфликт двух писателей решает `BEGIN IMMEDIATE` (второй ждёт `busy_timeout=5000`), дерево перечитывается внутри транзакции.

- [ ] **Step 1: Failing tests:**

```python
async def test_empty_db_has_root(tmp_path):
    s = await SqliteTopicStore.open(tmp_path / "s.db")
    t = await s.load_tree()
    assert list(t.topics) == ["root"] and t.rev == 0 and await s.tree_rev() == 0

async def test_apply_persists_and_logs(tmp_path):
    s = await SqliteTopicStore.open(tmp_path / "s.db")
    await s.apply([AddTopic(topic=Topic(id="finance", parent_id="root", title="Финансы", description="d"))], "cli")
    s2 = await SqliteTopicStore.open(tmp_path / "s.db")   # второй коннект видит изменения
    assert (await s2.load_tree()).routing_key("finance") == "finance" and await s2.tree_rev() == 1
    [e] = await s2.changelog()
    assert (e.actor, e.op, e.topic_id) == ("cli", "add", "finance")

async def test_invalid_change_writes_nothing(tmp_path): ...   # TreeValidationError, tree_rev не изменился, changelog пуст
async def test_archive_hides_subtree_but_keeps_rows(tmp_path): ...   # load_tree без них; строки с archived_at в БД
async def test_state_roundtrip(tmp_path): ...    # set_state("other_offset","42") -> get_state == "42"
```

- [ ] **Step 2:** Run `uv run pytest tests/unit/adapters/test_sqlite_store.py -q` → FAIL.
- [ ] **Step 3:** Реализовать на `aiosqlite`.
- [ ] **Step 4:** Run → PASS; mypy чисто.
- [ ] **Step 5: Commit** (после подтверждения): `feat(store): sqlite topic store with changelog`.

---

### Task 4: Темы через CLI, YAML import/export

**Files:**
- Rewrite: `src/llm_sorter/domain/loader.py`
- Create: `src/llm_sorter/application/topics.py`
- Modify: `src/llm_sorter/adapters/inbound/cli.py`
- Test: `tests/unit/domain/test_loader.py`, `tests/unit/application/test_topics.py`, `tests/unit/adapters/test_cli.py`

**Interfaces — Consumes:** `TopicStore`, `SqliteTopicStore` (Task 3), `Change` (Task 2).
**Produces:**

```python
# domain/loader.py — YAML-формат:
# settings: {max_depth: 3, discover: {buffer_threshold: 50}}
# topics:
#   - {id: finance, parent: root, title: Финансы, description: ..., examples: [...], guard: "...?"}
def load_yaml(text: str) -> tuple[Settings, list[Topic]]    # TreeParseError
def dump_yaml(tree: Tree) -> str                             # родители раньше детей, root не пишется

class TopicsService:
    def __init__(self, store: TopicStore) -> None
    async def show(self) -> list[str]            # строки "  cards  [discovered]  finance.cards"
    async def add(self, id, parent, title, description, guard: str | None = None) -> Tree
    async def move(self, id: str, parent: str) -> Tree
    async def edit(self, id: str, **fields: Any) -> Tree
    async def archive(self, id: str) -> Tree
    async def import_yaml(self, text: str) -> Tree   # пустое дерево: settings + AddTopic по списку; непустое — ошибка "tree not empty; use --replace"
    async def import_yaml_replace(self, text: str) -> Tree   # ArchiveTopic всех детей root, затем добавление
    async def export_yaml(self) -> str
```

CLI (`typer`, группа `topics`): `show`, `add ID --parent P --title T --description D [--guard G]`, `move ID PARENT`, `edit ID [--title ...] ...`, `archive ID`, `import FILE [--replace]`, `export [-o FILE]`, `log [--limit N]`. Все открывают `SqliteTopicStore` по `cfg.sqlite_path`. `TreeValidationError` → `typer.Exit(1)` с ошибками в stderr (как текущий `_fail`).

- [ ] **Step 1: Failing tests:** `test_yaml_roundtrip` (load → apply → dump → load равны), `test_import_into_non_empty_requires_replace`, `test_show_prints_routing_keys` (через `CliRunner`, строка содержит `finance.cards`), `test_move_changes_routing_key_and_logs`.
- [ ] **Step 2:** Run `uv run pytest tests/unit/domain/test_loader.py tests/unit/application/test_topics.py tests/unit/adapters/test_cli.py -q` → FAIL.
- [ ] **Step 3:** Реализовать.
- [ ] **Step 4:** Run → PASS. Ручная проверка: `uv run llm-sorter topics import datasets/synthetic-v1/<seed>.yml` — **seed YAML создать** в `datasets/synthetic-v1/topics.yml` из тем каталога `data/inbox` (fin/health/tech/travel × 2 листа), `topics show` печатает 4 ветки.
- [ ] **Step 5: Commit** (после подтверждения): `feat(topics): yaml import/export and cli edits`.

---

### Task 5: Обработка — ProcessService

**Files:**
- Create: `src/llm_sorter/application/process.py`
- Modify: `src/llm_sorter/ports/models.py` (убрать `translate`, `summarize`), `src/llm_sorter/ports/metrics.py`, `adapters/outbound/noop_metrics.py`, `prometheus_metrics.py`
- Rename: `git mv src/llm_sorter/adapters/outbound/lmstudio.py src/llm_sorter/adapters/outbound/llm.py` (классы `LLMDecision`, `LLMGuard`, `LLMText`, `LLMEmbedder`, `make_openai_client(url, api_key)`; удалить translate/summarize; промпты и правки пользователя без изменений), тест `tests/unit/adapters/test_llm.py` (бывший `test_lmstudio.py`).
- Delete: `application/classify.py`, `tests/unit/application/test_classify.py`
- Rewrite: `tests/fakes.py` (оставить `FakeDecision`/`FakeGuard`/`FakeText`/`FakeEmbedder` из текущих фейков со скриптами, удалить фейки хранилищ/стартера; `RecordingMetrics` под новый протокол)
- Test: `tests/unit/application/test_process.py`

**Interfaces — Consumes:** `Tree`, `candidates_for`, `resolve_choice`, `guard_question` (Task 2), `Transcript`, `Classified`, `OtherItem`.
**Produces:**

```python
class Metrics(Protocol):
    def processed(self, outcome: Literal["classified","other","final_other"]) -> None
    def model_call(self, role: str, latency_s: float, ok: bool) -> None
    def model_error(self, role: str) -> None
    def guard_result(self, path: str, passed: bool) -> None
    def buffer_size(self, node: str, size: int) -> None
    def discover_run(self, node: str, created: int, rejected: int) -> None
    def discover_blocked(self, node: str, blocked: bool) -> None
    def tree_rev(self, rev: int) -> None

@dataclass(frozen=True) class Outcome:
    classified: Classified | None
    others: list[OtherItem]          # 0..2: остановка/отказ, плюс копия leaf

MODEL_ATTEMPTS = 3
class ProcessService:
    def __init__(self, decision: DecisionModel, guard: GuardModel, metrics: Metrics,
                 now: Callable[[], datetime] = lambda: datetime.now(UTC)) -> None
    async def process(self, t: Transcript, tree: Tree, reprocess: int = 0) -> Outcome
```

Правила `process` (спека, «Слой 2»):
- узел без кандидатов кроме `other` (пустое дерево) или пустой `text` → `other` на этом узле, `reason="no_match"`, без вызова LLM;
- decide → `InvalidModelOutput` 3 раза → `reason="model_error"` на текущем узле; выбор вне кандидатов = `InvalidModelOutput`;
- лист L → guard; отказ → `OtherItem(node=parent(L).id, reason="gate_failed")`; `InvalidModelOutput` ×3 → `node=parent(L).id, reason="model_error"`;
- пройден → `Classified(topic=TopicRef(id, routing_key, title), trail=path(L), tree_rev=tree.rev)`; если `tree.depth(L) < tree.max_depth_for(L)` — дополнительно `OtherItem(node=L, reason="leaf")`;
- `final = reprocess >= tree.settings.max_reprocess` для `no_match`/`gate_failed`/`model_error` (не для `leaf`);
- прочие исключения (сеть, таймаут) — наружу, без метрик ошибки модели.

- [ ] **Step 1: Failing tests:**

```python
async def test_empty_tree_goes_to_root_other_without_llm():       # Review Focus 1
    dec = FakeDecision([]); out = await svc(dec).process(t("a"), Tree.empty())
    assert out.classified is None and [(o.node, o.reason) for o in out.others] == [("root", "no_match")]
    assert dec.calls == []

async def test_descends_to_leaf_and_passes_gate(tree3):
    out = await svc(FakeDecision(["finance", "cards"]), FakeGuard([True])).process(t("a"), tree3)
    assert out.classified.topic.path == "finance.cards" and out.classified.trail == ["finance", "cards"]
    assert [(o.node, o.reason) for o in out.others] == [("cards", "leaf")]    # depth 2 < max_depth 3

async def test_gate_failure_goes_to_parent_other(tree3):
    out = await svc(FakeDecision(["finance", "cards"]), FakeGuard([False])).process(t("a"), tree3)
    assert out.classified is None and [(o.node, o.reason) for o in out.others] == [("finance", "gate_failed")]

async def test_invalid_output_three_times_falls_back(tree3): ...   # decide 3× "bogus" -> ("root","model_error")
async def test_final_after_max_reprocess(tree3): ...               # reprocess=3 -> others[0].final is True
async def test_no_leaf_copy_at_max_depth(tree_depth3): ...         # лист на глубине 3 -> others == []
async def test_connection_error_propagates(tree3): ...             # FakeDecision бросает ConnectionError -> pytest.raises
```

- [ ] **Step 2:** Run `uv run pytest tests/unit/application/test_process.py -q` → FAIL.
- [ ] **Step 3:** Реализовать (перенести `_call_model` из `classify.py`).
- [ ] **Step 4:** Run `make test` → PASS (включая `test_llm.py`); `make typecheck` чисто.
- [ ] **Step 5: Commit** (после подтверждения): `feat(process): tree descent with gate and other outcomes`.

---

### Task 6: Rabbit-топология, обработчик и фидеры

**Files:**
- Create: `src/llm_sorter/ports/bus.py`, `src/llm_sorter/adapters/outbound/rabbit.py`, `src/llm_sorter/adapters/inbound/processor.py`
- Rewrite: `src/llm_sorter/bootstrap.py`, `src/llm_sorter/adapters/inbound/cli.py` (команды `feed dir`, `feed tg`, `process`)
- Test: `tests/unit/adapters/test_processor.py`, `tests/integration/conftest.py`, `tests/integration/test_rabbit_flow.py`

**Interfaces — Consumes:** `ProcessService.process`, `Outcome` (Task 5), `TopicStore.load_tree/tree_rev` (Task 3), `dump_record` (Task 2), `run_poller`, `LocalDirSource`, `TelegramSource`.
**Produces:**

```python
class Bus(Protocol):
    async def classified(self, msg: Classified) -> None        # routing key = msg.topic.path
    async def other(self, record: StreamRecord) -> None        # в stream
    async def reinject(self, t: Transcript, reprocess: int) -> None   # в inbox, header x-reprocess

@dataclass(frozen=True) class Names:
    inbox: str; exchange: str; archive: str; other: str
    @property def dlq(self) -> str          # f"{inbox}.dlq"
    @classmethod def from_config(cls, cfg: AppConfig) -> "Names"
async def declare_topology(ch: AbstractChannel, n: Names, other_max_age: str, other_max_bytes: int) -> None
class RabbitBus(Bus): def __init__(self, ch: AbstractChannel, n: Names) -> None   # publisher confirms
class InboxPublisher(MessagePublisher):      # для фидеров, заменяет старый RabbitPublisher
    def __init__(self, url: str, n: Names) -> None; async def publish(self, t: Transcript) -> None; async def close()
def reprocess_count(headers: Mapping[str, Any] | None) -> int   # нет/мусор/<0 -> 0

class TreeCache:
    def __init__(self, store: TopicStore, refresh_s: float, clock: Callable[[], float] = time.monotonic) -> None
    async def get(self) -> Tree       # tree_rev не чаще refresh_s; при изменении load_tree
BACKOFF_START_S = 1.0; BACKOFF_MAX_S = 300.0
async def handle_delivery(msg: AbstractIncomingMessage, service: ProcessService, cache: TreeCache,
                          bus: Bus, stop: asyncio.Event, sleep=asyncio.sleep) -> None
async def run_processor(url: str, n: Names, concurrency: int, service: ProcessService,
                        cache: TreeCache, stop: asyncio.Event) -> None
```

`declare_topology`: inbox durable с `x-dead-letter-exchange=""`, `x-dead-letter-routing-key=<dlq>` + сам DLQ; exchange `topic` durable; archive durable + bind `#`; stream: `durable=True`, `arguments={"x-queue-type":"stream","x-max-age":..., "x-max-length-bytes":...}`.

`handle_delivery` (спека «Слой 2» + решение «сбой — ждём»):
1. `Transcript` из тела; ошибка разбора/схемы → `msg.reject(requeue=False)` (→ DLQ), return.
2. Цикл: `tree = await cache.get()`; `outcome = await service.process(t, tree, reprocess_count(msg.headers))`; публикация: `classified` (если есть), затем каждый `other`; `msg.ack()`; return.
3. Любое исключение в шаге 2 (кроме `CancelledError`) → лог, `await sleep(delay)`, `delay = min(delay*2, BACKOFF_MAX_S)`, повтор шага 2 **с тем же сообщением**. Если `stop` выставлен — `msg.nack(requeue=True)`, return.

`run_processor`: `channel.set_qos(prefetch_count=concurrency)`, `declare_topology`, `RabbitBus`, `queue.consume(on_message)`; каждое сообщение — `asyncio.create_task(handle_delivery(...))`; по `stop` — `cancel` consumer, дождаться задач.

CLI: `feed dir`, `feed tg` (бывший `ingest` с `INGEST_SOURCE`; логика старта TG без изменений, `InboxPublisher`), `process [--concurrency N] [--metrics-port P]` (0 — без метрик). `bootstrap.build_processor(cfg, overrides) -> (ProcessService, TreeCache, store)`; `overrides` для фейков, как сейчас.

- [ ] **Step 1: Failing unit tests** (фейковое сообщение с `ack/reject/nack`, `FakeBus` со списками, `sleep` — запись задержек):

```python
async def test_invalid_body_rejected_to_dlq(): ...       # reject(requeue=False), bus пуст
async def test_publish_before_ack(tree3): ...            # порядок вызовов: bus.classified, bus.other(leaf), ack
async def test_llm_down_waits_then_processes_once(tree3):        # Review Focus 2
    dec = FakeDecision([ConnectionError(), ConnectionError(), "finance", "cards"])
    await handle_delivery(m, svc(dec, FakeGuard([True])), cache, bus, stop, sleep=rec)
    assert rec.delays == [1.0, 2.0] and m.acked == 1 and m.rejected == 0
    assert [c.topic.path for c in bus.classified_msgs] == ["finance.cards"] and not any(o.reason == "model_error" for o in bus.others)
async def test_stop_during_outage_requeues(): ...        # nack(requeue=True), ack не вызван
@pytest.mark.parametrize("h,n", [(None,0),({},0),({"x-reprocess":"abc"},0),({"x-reprocess":-1},0),({"x-reprocess":2},2)])
def test_reprocess_count(h, n): assert reprocess_count(h) == n                # Review Focus 5
async def test_tree_cache_reloads_on_rev_change(tmp_path): ...   # clock-фейк: до refresh_s старое, после — новое дерево
```

- [ ] **Step 2:** Run `uv run pytest tests/unit/adapters/test_processor.py -q` → FAIL.
- [ ] **Step 3:** Реализовать; перенести тесты `tests/unit/adapters/test_consumer.py` → удалить (логика заменена), `test_cli.py` дополнить `feed`/`process` в `--help`.
- [ ] **Step 4:** Run `make test` → PASS.
- [ ] **Step 5: Integration** (`-m integration`, нужен `make up`; `conftest.py` — уникальный префикс имён Rabbit на тест, удаление очередей после; SQLite в `tmp_path`): `test_feed_process_classified_and_archive` — `InboxPublisher` публикует 2 сообщения, `run_processor` с фейковыми моделями; в `archive` 1 сообщение с routing key `finance.cards`, в stream 1 запись `no_match` + 1 `leaf` (читать stream с `x-stream-offset="first"`). Run `make up && RABBIT_URL=amqp://guest:guest@localhost:5673/ uv run pytest -m integration -q` → PASS.
- [ ] **Step 6: Commit** (после подтверждения): `feat(processor): rabbit topology, inbox consumer, feeders`.

---

### Task 7: Discover без хранилищ

**Files:**
- Rewrite: `src/llm_sorter/application/discover.py`
- Test: `tests/unit/application/test_discover.py` (переписать; сохранить сценарии пользователя на `diverse_samples`, `level`/`max_depth`)

**Interfaces — Consumes:** `cluster`, `diverse_samples`, `cosine` (clustering), `TextLLM.propose_topic(samples, parent_path, siblings, language, level, max_depth)`, `TextLLM.same_topic`, `Embedder.embed`, `should_discover`, `slots_left`, `dedup_verdict`, `proposal_is_valid` (Task 2).
**Produces:**

```python
@dataclass(frozen=True) class Accepted: topic: Topic; members: list[int]     # индексы в texts
@dataclass(frozen=True) class DiscoverResult:
    accepted: list[Accepted]; considered: list[int]; rejected: int            # considered = все не-принятые индексы
EMBED_BATCH = 16
class DiscoverService:
    def __init__(self, embedder: Embedder, text_llm: TextLLM, metrics: Metrics) -> None
    async def propose(self, tree: Tree, node_id: str, texts: list[str]) -> DiscoverResult
```

Логика — текущий `DiscoverService.run` без хранилищ: эмбеддинги текстов (без кеша), `cluster`, на кластер `propose_topic` (`level=tree.depth(node)+1`, `max_depth=tree.max_depth_for(node)`), `proposal_is_valid` (включая guard), дедуп с соседями и между принятыми, `id` уникализируется **по всем `tree.topics`** (`_free_id` с `taken = set(tree.topics) | {OTHER_ID, ROOT_ID}`), лимит `slots_left`. `Topic(parent_id=node_id, origin="discovered", guard=p.guard, ...)`.
Лист (`tree.is_leaf(node_id)`): соседи для дедупа = `[node как Candidate]`; если принято < 2 — `accepted=[]`, все индексы в `considered`.

- [ ] **Step 1: Failing tests:**

```python
async def test_id_collision_in_other_branch_gets_suffix(tree3):      # Review Focus 3: "cards" есть под finance
    res = await svc(text=FakeText([proposal(id="cards", guard="Пост о картах лояльности магазинов?")])).propose(tree3, "tech", texts)
    assert [a.topic.id for a in res.accepted] == ["cards_2"]

async def test_meaningless_guard_rejected(tree3): ...                 # guard="yes" -> accepted == [], model_error метрика
async def test_dup_of_sibling_rejected(tree3): ...                    # эмбеддинг ≈ соседу -> rejected == 1, индексы в considered
async def test_leaf_needs_two_subtopics(tree3): ...                   # лист cards, 1 кластер -> accepted == []
async def test_leaf_splits_into_two(tree3): ...                       # 2 кластера, 2 валидных -> 2 Accepted с parent_id="cards"
async def test_respects_slots_left(tree_full): ...                    # max_children исчерпан -> accepted == []
```

- [ ] **Step 2:** Run `uv run pytest tests/unit/application/test_discover.py -q` → FAIL.
- [ ] **Step 3:** Реализовать.
- [ ] **Step 4:** Run `make test` → PASS.
- [ ] **Step 5: Commit** (после подтверждения): `feat(discover): stateless proposals with guard validation and leaf split`.

---

### Task 8: Дообучение — Learner

**Files:**
- Create: `src/llm_sorter/application/learner.py`
- Test: `tests/unit/application/test_learner.py`

**Interfaces — Consumes:** `DiscoverService.propose`, `DiscoverResult` (Task 7), `TopicStore.apply/load_tree/get_state/set_state` (Task 3), `Bus.reinject/other` (Task 6), `StreamRecord`, `OtherItem`, `Resolved`, `Considered`, `AddTopic`, `should_discover`, `discover_blocked`-логика (`slots_left(...) <= 0 and buffer`).
**Produces:**

```python
OFFSET_KEY = "other_offset"
@dataclass class Pending: offset: int; item: OtherItem; considered: bool = False
class Learner:
    def __init__(self, store: TopicStore, discover: DiscoverService, bus: Bus, metrics: Metrics,
                 now: Callable[[], datetime] = ...) -> None
    def feed(self, offset: int, record: StreamRecord) -> None
    def buffer(self, node: str) -> list[Pending]
    def watermark(self) -> int | None                 # min offset среди Pending; None если буферы пусты
    async def run_due(self) -> list[str]              # узлы, где прошёл discover
    async def refresh_old(self, max_age: timedelta) -> int   # переиздано записей
    async def save_offset(self, next_offset: int) -> None    # watermark или next_offset, если буферы пусты
```

Правила `feed`:
- `OtherItem` с `final=True` — игнор (метрика `processed` уже учтена обработчиком);
- `OtherItem` — удалить прежнюю копию того же `id` из любого буфера, добавить в `buffer(item.node)`;
- `Resolved(ids)` — удалить эти `id` из всех буферов;
- `Considered(node, ids)` — пометить `considered=True` у этих `id` в `buffer(node)`.

`run_due`: дерево `load_tree`; для каждого узла с `should_discover(tree, node, new_count)` где `new_count` = число не-`considered` (узел отсутствует в дереве — архивирован — пропустить; `slots_left<=0` и буфер не пуст → `metrics.discover_blocked(node, True)` + warning):
1. `res = await discover.propose(tree, node, [p.item.text for p in buf])`;
2. если `res.accepted` → `store.apply([AddTopic(a.topic) ...], "learner")`; перегон: для листа — **все** `buf`, иначе — члены принятых кластеров: `bus.reinject(p.item.transcript(), p.item.reprocess + 1)`; затем `bus.other(Resolved(ids=перегнанные))` и локально удалить их;
3. остальные индексы → `bus.other(Considered(node, ids))` и локально пометить;
4. `metrics.discover_run(node, len(accepted), res.rejected)`, `metrics.buffer_size(node, len(buffer))`.
Ошибки LLM/Rabbit — наружу (раннер ждёт и повторяет).

`refresh_old`: `Pending` с `item.at < now - max_age` → `bus.other(item.model_copy(update={"at": now()}))` (новая копия; старая уйдёт при `feed` новой).

- [ ] **Step 1: Failing tests** (`FakeBus`, реальный `SqliteTopicStore` в `tmp_path`, `FakeDiscover` со скриптом результатов):

```python
async def test_threshold_triggers_discover_and_reinjects_members(store, tree_root_only): ...
    # 50 OtherItem node=root -> run_due() == ["root"]; tree_rev +1; bus.reinjected == членам; Resolved опубликован; буфер без них
async def test_considered_do_not_count_again(): ...      # после discover без принятых второй run_due() == [] пока не придут 50 новых
async def test_latest_copy_wins(): ...                   # один id дважды (node=root, затем node=finance) -> только в finance
async def test_restart_replay_restores_buffers(tmp_path):        # Review Focus 4
    a = learner(); [a.feed(i, rec) for i, rec in enumerate(records)]   # items, Resolved, Considered
    replay = learner(); [replay.feed(i, rec) for i, rec in enumerate(records) if i >= a.watermark()]
    assert snapshot(replay) == snapshot(a)
async def test_leaf_split_reinjects_whole_leaf_buffer(): ...
async def test_final_items_ignored(): ...
async def test_refresh_old_republishes(): ...            # at старше max_age -> bus.other с новым at
async def test_blocked_node_reported(): ...              # max_children исчерпан -> discover_blocked(node, True), discover не вызван
```

- [ ] **Step 2:** Run `uv run pytest tests/unit/application/test_learner.py -q` → FAIL.
- [ ] **Step 3:** Реализовать.
- [ ] **Step 4:** Run `make test` → PASS.
- [ ] **Step 5: Commit** (после подтверждения): `feat(learner): per-node buffers from other stream`.

---

### Task 9: Раннер дообучения и CLI `learn`

**Files:**
- Create: `src/llm_sorter/adapters/inbound/learner_runner.py`
- Modify: `src/llm_sorter/bootstrap.py` (`build_learner`), `src/llm_sorter/adapters/inbound/cli.py` (`learn [--metrics-port P]`)
- Test: `tests/unit/adapters/test_learner_runner.py`, `tests/integration/test_learner_flow.py`

**Interfaces — Consumes:** `Learner` (Task 8), `declare_topology`, `RabbitBus`, `Names`, `load_record` (Task 6, 2), `TopicStore.get_state` (Task 3).
**Produces:**

```python
class LearnerLock:                       # fcntl.flock на f"{sqlite_path}.learner.lock"
    def __init__(self, path: Path) -> None
    def __enter__(self) -> "LearnerLock"   # уже занят -> RuntimeError("another learner is running")
    def __exit__(self, *exc) -> None
def stream_offset(headers: Mapping[str, Any] | None) -> int   # x-stream-offset
async def run_learner(url: str, n: Names, learner: Learner, store: TopicStore, check_s: float,
                      refresh_s: float, max_age: timedelta, stop: asyncio.Event, sleep=asyncio.sleep) -> None
```

`run_learner`: `set_qos(prefetch_count=500)` (обязателен для stream); старт с `x-stream-offset` = `int(get_state(OFFSET_KEY))` или `"first"`; каждое сообщение: `learner.feed(stream_offset(h), load_record(body))`, `ack`. Отдельная корутина раз в `check_s`: `await learner.run_due()` затем `learner.save_offset(последний offset + 1)`; раз в `refresh_s`: `refresh_old(max_age/2)`. Исключение в `run_due` → backoff 1 с ×2 до 300 с, повтор (как в процессоре). Непарсируемая запись stream — warning и пропуск (stream нельзя reject в DLQ).

- [ ] **Step 1: Failing tests:** `test_second_lock_fails(tmp_path)`; `test_stream_offset_header`; `test_run_due_error_backs_off` (фейковый learner бросает дважды → задержки `[1.0, 2.0]`).
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Реализовать.
- [ ] **Step 4:** Run `make test` → PASS.
- [ ] **Step 5: Integration** `test_learner_flow`: в stream 12 `OtherItem` node=root (порог 10, `min_cluster_size` 5 через settings), фейковые модели дают 1 тему → в SQLite новая тема, в inbox 12 сообщений с `x-reprocess=1`, `learner_state.other_offset` сохранён; второй `run_learner` с тем же stream не перегоняет их повторно. Run integration → PASS.
- [ ] **Step 6: Commit** (после подтверждения): `feat(learner): stream runner with single-instance lock`.

---

### Task 10: End-to-end, документация, чистка

**Files:**
- Create: `tests/integration/test_e2e.py` (переписать), `tests/unit/test_throughput_smoke.py`
- Modify: `CLAUDE.md`, `.env.example`, `Makefile` (`feed-tg`, `feed-dir`, `process`, `learn`, `test-int`), `deploy/grafana/*` (дашборд под новые метрики, если существует), старые spec/plan — строка «Заменён: 2026-10-23-llm-sorter-v2*» в начале.
- Modify: `tests/unit/test_layers.py` — `LIBS |= {"aiosqlite", "telethon"}`.

- [ ] **Step 1:** `test_e2e` (integration): `topics import` seed-дерева (2 ветки по 1 листу), `feed dir` из каталога с 30 файлами (20 подходят, 10 — новая тема), `process` и `learn` в одном event loop с фейковыми моделями по ключевым словам → в `archive` 20 сообщений; после discover появляется тема, 10 перегнанных сообщений оказываются в `archive` с её routing key; `changelog` содержит `learner add`.
- [ ] **Step 2:** `test_throughput_smoke` (unit, фейковые модели без задержки, `FakeBus`): 1000 сообщений через `handle_delivery` с `concurrency=16` за < 5 с на дереве глубины 3.
- [ ] **Step 3:** Run `make lint && make typecheck && make test && make up && RABBIT_URL=amqp://guest:guest@localhost:5673/ make test-int` → всё зелёное.
- [ ] **Step 4:** `CLAUDE.md` переписать под v2: команды, три процесса, топология Rabbit, SQLite, инварианты (ack после публикации, at-least-once, больший `tree_rev` побеждает, отказ гейта → `other` родителя, `max_reprocess`, сбой — ждём), правка дерева через `topics *`. `.env.example` — все поля `AppConfig`.
- [ ] **Step 5:** Ручной прогон: `make up`, `topics import datasets/synthetic-v1/topics.yml`, `feed dir`, `process`, `learn` на реальном LLM-сервере; `topics show` и очередь `llm-sorter.classified.all` в UI Rabbit — результат показать пользователю.
- [ ] **Step 6: Commit** (после подтверждения): `docs: v2 architecture, e2e and smoke tests`.
