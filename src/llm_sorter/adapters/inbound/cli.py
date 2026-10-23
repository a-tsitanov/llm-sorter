import asyncio
import contextlib
import logging
import signal
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated

import typer

from llm_sorter.adapters.inbound.poller import run_poller
from llm_sorter.adapters.inbound.processor import run_processor
from llm_sorter.adapters.inbound.stub_source import LocalDirSource
from llm_sorter.adapters.inbound.telegram_source import TelegramSource, TelethonGateway
from llm_sorter.adapters.outbound.rabbit import InboxPublisher, Names
from llm_sorter.adapters.outbound.sqlite_store import SqliteTopicStore
from llm_sorter.application.topics import TopicsService
from llm_sorter.bootstrap import build_processor
from llm_sorter.config import AppConfig
from llm_sorter.domain.tree import ROOT_ID
from llm_sorter.domain.validation import TreeValidationError
from llm_sorter.ports.io import TranscriptSource

app = typer.Typer(help="llm-sorter: классификация сообщений", no_args_is_help=True)
topics_app = typer.Typer(help="Дерево тем (SQLite)", no_args_is_help=True)
app.add_typer(topics_app, name="topics")
feed_app = typer.Typer(help="Фидеры: источник -> очередь inbox", no_args_is_help=True)
app.add_typer(feed_app, name="feed")


def _fail(msg: str) -> typer.Exit:
    typer.echo(msg, err=True)
    return typer.Exit(code=1)


def _topics[T](fn: Callable[[TopicsService], Awaitable[T]]) -> T:
    """Открывает хранилище, выполняет действие, ошибки правок -> код 1."""

    async def go() -> T:
        store = await SqliteTopicStore.open(AppConfig().sqlite_path)
        try:
            return await fn(TopicsService(store))
        finally:
            await store.close()

    try:
        return asyncio.run(go())
    except TreeValidationError as e:
        raise _fail("ошибки дерева:\n  " + "\n  ".join(e.errors)) from e
    except ValueError as e:  # TreeParseError и прочие ошибки ввода
        raise _fail(f"ошибка: {e}") from e


@topics_app.command("show")
def topics_show() -> None:
    """Дерево с routing key каждой темы."""
    typer.echo("\n".join(_topics(lambda s: s.show())))


@topics_app.command("add")
def topics_add(
    id: str,
    title: Annotated[str, typer.Option()],
    description: Annotated[str, typer.Option()],
    parent: Annotated[str, typer.Option()] = ROOT_ID,
    guard: Annotated[str | None, typer.Option()] = None,
) -> None:
    _topics(lambda s: s.add(id, parent, title, description, guard))


@topics_app.command("move")
def topics_move(id: str, parent: str) -> None:
    _topics(lambda s: s.move(id, parent))


@topics_app.command("edit")
def topics_edit(
    id: str,
    title: Annotated[str | None, typer.Option()] = None,
    description: Annotated[str | None, typer.Option()] = None,
    guard: Annotated[str | None, typer.Option()] = None,
) -> None:
    fields = {
        k: v
        for k, v in {"title": title, "description": description, "guard": guard}.items()
        if v is not None
    }
    if not fields:
        raise _fail("нечего менять: укажите --title / --description / --guard")
    _topics(lambda s: s.edit(id, **fields))


@topics_app.command("archive")
def topics_archive(id: str) -> None:
    """Убрать тему вместе с подтемами."""
    _topics(lambda s: s.archive(id))


@topics_app.command("import")
def topics_import(file: Path, replace: Annotated[bool, typer.Option("--replace")] = False) -> None:
    """Засеять дерево из YAML; --replace архивирует текущие темы."""
    text = file.read_text(encoding="utf-8")
    _topics(lambda s: s.import_yaml_replace(text) if replace else s.import_yaml(text))


@topics_app.command("export")
def topics_export(output: Annotated[Path | None, typer.Option("-o", "--output")] = None) -> None:
    text = _topics(lambda s: s.export_yaml())
    if output is None:
        typer.echo(text)
    else:
        output.write_text(text, encoding="utf-8")


@topics_app.command("log")
def topics_log(limit: int = 50) -> None:
    """Журнал изменений дерева, новые первыми."""

    async def go(s: TopicsService) -> list[str]:
        return [
            f"{e.at:%Y-%m-%d %H:%M:%S}  {e.actor:<7}  {e.op:<8}  {e.topic_id or '-'}  {e.details}"
            for e in await s.changelog(limit)
        ]

    typer.echo("\n".join(_topics(go)))


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


def _tg_folders(cfg: AppConfig) -> list[str]:
    return [f.strip() for f in cfg.tg_folders.split(",") if f.strip()]


async def _tg_gateway(cfg: AppConfig) -> TelethonGateway:
    return await TelethonGateway.connect(
        cfg.tg_api_id, cfg.tg_api_hash.get_secret_value(), cfg.tg_session.get_secret_value()
    )


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


async def _feed(cfg: AppConfig, source: TranscriptSource, stop: asyncio.Event) -> None:
    publisher = InboxPublisher(cfg.rabbit_url, Names.from_config(cfg))
    try:
        await run_poller(source, publisher, cfg.poll_interval_s, stop)
    finally:
        await publisher.close()


@feed_app.command("dir")
def feed_dir() -> None:
    """Файлы *.txt / *.json из SOURCE_DIR -> inbox."""
    cfg = AppConfig()

    async def main(stop: asyncio.Event) -> None:
        cfg.source_dir.mkdir(parents=True, exist_ok=True)
        await _feed(cfg, LocalDirSource(cfg.source_dir, cfg.source_dir / ".seen.json"), stop)

    _serve(main)


@feed_app.command("tg")
def feed_tg() -> None:
    """Посты чатов папок TG_FOLDERS -> inbox."""
    cfg = AppConfig()
    folders = _tg_folders(cfg)
    if not folders:
        raise _fail("TG_FOLDERS is empty - list folders with `llm-sorter tg-folders`")

    async def main(stop: asyncio.Event) -> None:
        gateway = await _tg_gateway(cfg)
        try:
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
            await _feed(cfg, source, stop)
        finally:
            await gateway.close()

    _serve(main)


@app.command("process")
def process(
    concurrency: Annotated[int | None, typer.Option(help="сообщений одновременно")] = None,
    metrics_port: Annotated[int | None, typer.Option(help="0 - без метрик")] = None,
) -> None:
    """Обработчик inbox: спуск по дереву, гейт, публикация результата."""
    cfg = AppConfig()

    async def main(stop: asyncio.Event) -> None:
        port = cfg.metrics_port if metrics_port is None else metrics_port
        service, cache, store = await build_processor(cfg, port)
        try:
            await run_processor(
                cfg.rabbit_url,
                Names.from_config(cfg),
                concurrency or cfg.process_concurrency,
                service,
                cache,
                stop,
                cfg.other_max_age,
                cfg.other_max_bytes,
            )
        finally:
            await store.close()

    _serve(main)
