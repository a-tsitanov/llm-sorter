import time

from pydantic import BaseModel, ConfigDict, Field

from llm_sorter.domain.actions import (
    Action,
    AnnotateAction,
    RocketAction,
    SaveAction,
    TranslateAction,
)
from llm_sorter.ports.metrics import Metrics
from llm_sorter.ports.models import TextLLM
from llm_sorter.ports.storage import FileStorage, Notifier, RulesRepository, TranscriptStore


class ActionContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    file_id: str
    text: str
    translations: dict[str, str] = Field(default_factory=dict)
    summary: str | None = None


class ActionService:
    def __init__(
        self,
        transcripts: TranscriptStore,
        storage: FileStorage,
        notifier: Notifier,
        text_llm: TextLLM,
        rules: RulesRepository,
        metrics: Metrics,
        language: str,
    ) -> None:
        self._transcripts = transcripts
        self._storage = storage
        self._notifier = notifier
        self._text_llm = text_llm
        self._rules = rules
        self._metrics = metrics
        self._default_language = language

    async def load_context(self, file_id: str) -> ActionContext:
        t = await self._transcripts.get(file_id)
        return ActionContext(file_id=file_id, text=t.text)

    async def node_plan(
        self, version: int, node_uid: str, guard_failed: bool
    ) -> tuple[list[Action], str]:
        tree = await self._rules.get(version)
        path = tree.path_of(node_uid)
        if not guard_failed:
            return list(tree.find(node_uid).actions), path
        default = tree.defaults.guard_failed_action
        # путь узла вкладывается в каталог дефолтного save
        if isinstance(default, SaveAction) and default.path:
            default = SaveAction(type="save", path=f"{default.path}/{path}")
        return [default], path

    async def _language(self) -> str:
        """Язык из активного дерева; при отсутствии правил — значение по умолчанию."""
        try:
            return (await self._rules.active()).settings.language
        except LookupError:
            return self._default_language

    async def run(self, ctx: ActionContext, action: Action, node_path: str) -> ActionContext:
        started = time.monotonic()
        try:
            result = await self._execute(ctx, action, node_path)
        except Exception:
            self._metrics.action_done(action.type, False, time.monotonic() - started)
            raise
        self._metrics.action_done(action.type, True, time.monotonic() - started)
        return result

    async def _execute(self, ctx: ActionContext, action: Action, node_path: str) -> ActionContext:
        if isinstance(action, TranslateAction):
            translated = await self._text_llm.translate(ctx.text, action.to)
            return ctx.model_copy(
                update={"translations": {**ctx.translations, action.to: translated}}
            )
        if isinstance(action, AnnotateAction):
            summary = await self._text_llm.summarize(ctx.text, await self._language())
            return ctx.model_copy(update={"summary": summary})
        if isinstance(action, SaveAction):
            base = action.path or node_path
            await self._storage.write(f"{base}/{ctx.file_id}.txt", ctx.text)
            if ctx.summary is not None:
                await self._storage.write(f"{base}/{ctx.file_id}.summary.txt", ctx.summary)
            for lang, body in ctx.translations.items():
                await self._storage.write(f"{base}/{ctx.file_id}.{lang}.txt", body)
            return ctx
        if isinstance(action, RocketAction):
            await self._notifier.send(
                action.channel,
                text=ctx.summary or ctx.text[:500],
                attachment_name=f"{ctx.file_id}.txt",
                attachment=ctx.text,
            )
            return ctx
        raise ValueError(f"unknown action: {action!r}")
