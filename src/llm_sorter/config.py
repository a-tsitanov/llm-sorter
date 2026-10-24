from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    """Настройки приложения из переменных окружения (без префикса) и .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # OpenAI-совместимый сервер моделей (vLLM, LM Studio ...)
    llm_url: str = "http://localhost:1234/v1"
    llm_api_key: SecretStr = SecretStr("lm-studio")
    # гейт и модель discover на отдельных серверах; пусто - тот же LLM_URL / LLM_API_KEY
    llm_guard_url: str = ""
    llm_guard_api_key: SecretStr = SecretStr("")
    llm_text_url: str = ""
    llm_text_api_key: SecretStr = SecretStr("")
    model_decide: str = "lfm2"
    # chat - JSON по схеме через chat/completions; systemone - decision-модель (d1), /v1/systemone
    decide_protocol: Literal["chat", "systemone"] = "chat"
    model_guard: str = "qwen3-4b"
    model_text: str = "qwen3-4b"
    model_embed: str = "bge-m3"

    # дерево тем, журнал и позиция дообучения в stream
    sqlite_path: Path = Path("./data/sorter.db")

    rabbit_url: str = "amqp://guest:guest@localhost:5672/"
    inbox_queue: str = "llm-sorter.inbox"
    classified_exchange: str = "llm-sorter.classified"
    classified_archive: str = "llm-sorter.classified.all"
    other_stream: str = "llm-sorter.other"
    other_max_age: str = "30D"
    other_max_bytes: int = 10 * 1024**3

    # одновременно обрабатываемых сообщений в одном процессе-обработчике
    process_concurrency: int = 16
    # как часто обработчик сверяет tree_rev
    tree_refresh_s: float = 1.0
    # как часто дообучение проверяет пороги и сохраняет позицию в stream
    learner_check_s: float = 30.0
    # как часто дообучение переиздаёт старые ждущие сообщения (против ретеншна stream)
    other_refresh_s: float = 86400.0
    metrics_port: int = 9100
    learner_metrics_port: int = 9101

    # фидер каталога
    source_dir: Path = Path("./data/inbox")
    poll_interval_s: float = 10.0

    # фидер Telegram
    tg_api_id: int = 0
    tg_api_hash: SecretStr = SecretStr("")
    # Telethon StringSession
    tg_session: SecretStr = SecretStr("")
    # названия папок Telegram через запятую, как в клиенте
    tg_folders: str = ""
    tg_state_file: Path = Path("./data/tg_state.json")
    # сколько последних сообщений брать из чата при первом опросе
    tg_backfill: int = 20
    tg_min_len: int = 80
