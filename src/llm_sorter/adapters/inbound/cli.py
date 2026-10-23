import asyncio
import contextlib
import logging
import signal
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml

from llm_sorter.adapters.inbound.poller import run_poller
from llm_sorter.adapters.inbound.rabbit import RabbitPublisher, run_consumer
from llm_sorter.adapters.inbound.stub_source import LocalDirSource
from llm_sorter.adapters.inbound.telegram_source import TelegramSource, TelethonGateway
from llm_sorter.adapters.outbound.null_starter import NullStarter
from llm_sorter.application.rules import PublishResult, RulesService
from llm_sorter.bootstrap import build_app
from llm_sorter.config import AppConfig
from llm_sorter.domain.loader import load_tree
from llm_sorter.domain.tree import RuleTree
from llm_sorter.ports.io import TranscriptSource
from llm_sorter.ports.storage import VersionConflict

app = typer.Typer(help="llm-sorter: классификация транскриптов", no_args_is_help=True)
rules_app = typer.Typer(help="Управление деревом правил", no_args_is_help=True)
app.add_typer(rules_app, name="rules")


def _run[T](fn: Callable[[RulesService], Awaitable[T]], *, read_only: bool = False) -> T:
    """read_only: команда ничего не запускает, Temporal не нужен."""
    overrides: dict[str, Any] = {"starter": NullStarter()} if read_only else {}

    async def go() -> T:
        application = await build_app(AppConfig(), overrides)
        try:
            return await fn(application.rules_service)
        finally:
            await application.close()

    return asyncio.run(go())


def _print_result(res: PublishResult) -> None:
    typer.echo(
        f"published version {res.version}; reprocess nodes: {len(res.reprocess_nodes)}, "
        f"workflows started: {res.reprocess_started}"
    )


def _render(tree: RuleTree) -> list[str]:
    lines: list[str] = []

    def walk(node: Any, depth: int) -> None:
        lines.append(f"{'  ' * depth}{tree.path_of(node.uid) or '/'}  [{node.origin}]")
        for c in node.children:
            walk(c, depth + 1)

    walk(tree.root, 0)
    return lines


def _fail(msg: str) -> typer.Exit:
    typer.echo(msg, err=True)
    return typer.Exit(1)


@rules_app.command("publish")
def publish(
    file: Annotated[Path, typer.Argument(help="YAML с деревом")],
    base: Annotated[int | None, typer.Option("--base", help="Базовая версия")] = None,
) -> None:
    try:
        text = file.read_text(encoding="utf-8")
        base_version = base
        if base_version is None:
            data = yaml.safe_load(text)
            raw = data.get("version") if isinstance(data, dict) else None
            base_version = int(raw) if raw is not None else None
    except (OSError, yaml.YAMLError, ValueError, TypeError) as e:
        raise _fail(f"ошибка чтения файла: {e}") from e

    async def go(svc: RulesService) -> PublishResult:
        if base_version is None:
            try:
                await svc.pull()
            except LookupError:
                pass  # активных правил нет: первая публикация
            else:
                raise _fail("в файле нет version; сначала выполните `rules pull`")
        return await svc.publish_yaml(text, base_version)

    try:
        _print_result(_run(go))
    except VersionConflict as e:
        raise _fail(f"конфликт версий: {e}; выполните `rules pull`") from e
    except ValueError as e:  # TreeParseError и TreeValidationError
        raise _fail(f"ошибка: {e}") from e


@rules_app.command("pull")
def pull(
    output: Annotated[Path | None, typer.Option("-o", "--output")] = None,
) -> None:
    text = _run(lambda svc: svc.pull(), read_only=True)
    if output is None:
        typer.echo(text)
    else:
        output.write_text(text, encoding="utf-8")
        typer.echo(f"saved to {output}")


@rules_app.command("rollback")
def rollback(version: int) -> None:
    try:
        _print_result(_run(lambda svc: svc.rollback(version)))
    except LookupError as e:
        raise _fail(f"неизвестная версия: {e}") from e
    except VersionConflict as e:
        raise _fail(f"конфликт версий: {e}; повторите команду") from e
    except ValueError as e:
        raise _fail(f"ошибка: {e}") from e


@rules_app.command("show")
def show() -> None:
    tree = load_tree(_run(lambda svc: svc.pull(), read_only=True))
    typer.echo("\n".join(_render(tree)))


def _serve(main: Callable[[asyncio.Event], Awaitable[None]]) -> None:
    """Запускает долгоживущий процесс; SIGINT/SIGTERM выставляют stop для штатной остановки."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    async def go() -> None:
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop.set)
        await main(stop)

    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(go())


@app.command("worker")
def worker() -> None:
    """Temporal worker."""
    from llm_sorter.orchestration.worker import run_worker

    _serve(lambda stop: run_worker(AppConfig(), stop=stop))


def _tg_folders(cfg: AppConfig) -> list[str]:
    return [f.strip() for f in cfg.tg_folders.split(",") if f.strip()]


async def _tg_gateway(cfg: AppConfig) -> TelethonGateway:
    return await TelethonGateway.connect(
        cfg.tg_api_id, cfg.tg_api_hash.get_secret_value(), cfg.tg_session.get_secret_value()
    )


@app.command("ingest")
def ingest() -> None:
    """Поллер источника -> RabbitMQ. INGEST_SOURCE: dir - каталог SOURCE_DIR,
    telegram - чаты папок TG_FOLDERS."""
    cfg = AppConfig()
    if cfg.ingest_source == "telegram" and not _tg_folders(cfg):
        raise _fail("TG_FOLDERS is empty - list folders with `llm-sorter tg-folders`")

    async def main(stop: asyncio.Event) -> None:
        gateway: TelethonGateway | None = None
        source: TranscriptSource
        if cfg.ingest_source == "telegram":
            gateway = await _tg_gateway(cfg)
            folders = _tg_folders(cfg)
            # падаем на старте, а не на каждом тике, если папки нет
            chats = await gateway.folder_chats(folders)
            logging.getLogger(__name__).info(
                "telegram: %d chats in %s: %s", len(chats), folders, [c.title for c in chats]
            )
            source = TelegramSource(
                gateway,
                folders,
                cfg.tg_state_file,
                backfill=cfg.tg_backfill,
                min_len=cfg.tg_min_len,
            )
        else:
            cfg.source_dir.mkdir(parents=True, exist_ok=True)
            source = LocalDirSource(cfg.source_dir, cfg.source_dir / ".seen.json")
        publisher = RabbitPublisher(cfg.rabbit_url, cfg.queue_name)
        try:
            await run_poller(source, publisher, cfg.poll_interval_s, stop)
        finally:
            await publisher.close()
            if gateway is not None:
                await gateway.close()

    _serve(main)


@app.command("tg-folders")
def tg_folders() -> None:
    """Папки Telegram аккаунта TG_SESSION (значения для TG_FOLDERS)."""
    cfg = AppConfig()

    async def go() -> list[str]:
        gateway = await _tg_gateway(cfg)
        try:
            return await gateway.folder_titles()
        finally:
            await gateway.close()

    typer.echo("\n".join(asyncio.run(go())))


@app.command("consume")
def consume() -> None:
    """Consumer RabbitMQ -> старт ClassifyWorkflow."""
    cfg = AppConfig()

    async def main(stop: asyncio.Event) -> None:
        application = await build_app(cfg)
        try:
            await run_consumer(
                cfg.rabbit_url, cfg.queue_name, application.transcripts, application.starter, stop
            )
        finally:
            await application.close()

    _serve(main)
