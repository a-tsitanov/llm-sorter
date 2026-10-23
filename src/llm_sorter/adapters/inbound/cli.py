import asyncio
import contextlib
import logging
import signal
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated

import typer

from llm_sorter.adapters.inbound.telegram_source import TelethonGateway
from llm_sorter.adapters.outbound.sqlite_store import SqliteTopicStore
from llm_sorter.application.topics import TopicsService
from llm_sorter.config import AppConfig
from llm_sorter.domain.tree import ROOT_ID
from llm_sorter.domain.validation import TreeValidationError

app = typer.Typer(help="llm-sorter: классификация сообщений", no_args_is_help=True)
topics_app = typer.Typer(help="Дерево тем (SQLite)", no_args_is_help=True)
app.add_typer(topics_app, name="topics")


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
