from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    """Настройки приложения из переменных окружения (без префикса) и .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    lmstudio_url: str = "http://localhost:1234/v1"
    lmstudio_api_key: str = "lm-studio"
    model_decide: str = "lfm2"
    model_guard: str = "qwen3-4b"
    model_text: str = "qwen3-4b"
    model_embed: str = "bge-m3"
    database_url: str = "postgresql+asyncpg://sorter:sorter@localhost:5432/sorter"
    rabbit_url: str = "amqp://guest:guest@localhost:5672/"
    queue_name: str = "transcripts"
    temporal_address: str = "localhost:7233"
    temporal_namespace: str = "default"
    task_queue: str = "llm-sorter"
    storage_root: Path = Path("./data/out")
    source_dir: Path = Path("./data/inbox")
    poll_interval_s: float = 10.0
    metrics_port: int = 9100
    sweep_interval_s: int = 600
    # активностей на воркер одновременно; LM Studio обслуживает ~4 запроса на модель,
    # лишние ждут в очереди и падают по таймауту
    max_concurrent_activities: int = 8
    sweep_schedule_id: str = "llm-sorter-sweep"
