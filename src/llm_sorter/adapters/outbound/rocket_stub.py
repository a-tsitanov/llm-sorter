import logging


class LoggingNotifier:
    """Заглушка Rocket.Chat: только логирует, ничего не отправляет."""

    async def send(self, channel: str, text: str, attachment_name: str, attachment: str) -> None:
        logging.info("rocket stub: channel=%s name=%s text=%s", channel, attachment_name, text)
