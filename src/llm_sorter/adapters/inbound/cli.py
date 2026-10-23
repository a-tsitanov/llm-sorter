import asyncio
import contextlib
import logging
import signal
from collections.abc import Awaitable, Callable

import typer

from llm_sorter.adapters.inbound.telegram_source import TelethonGateway
from llm_sorter.config import AppConfig

app = typer.Typer(help="llm-sorter: классификация сообщений", no_args_is_help=True)


@app.callback()
def _main() -> None:
    """Группа команд (пока одна: typer иначе свернёт её в корень)."""


def _fail(msg: str) -> typer.Exit:
    typer.echo(msg, err=True)
    return typer.Exit(code=1)


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
