import logging


class NullStarter:
    """Заглушка WorkflowStarter до подключения Temporal: логирует и ничего не запускает."""

    async def start_classify(self, file_id: str) -> bool:
        logging.warning("null starter: classify %s", file_id)
        return False

    async def start_discover(self, node_uid: str) -> bool:
        logging.warning("null starter: discover %s", node_uid)
        return False

    async def start_reprocess(self, file_id: str, node_uid: str, version: int) -> bool:
        logging.warning("null starter: reprocess %s %s v%d", file_id, node_uid, version)
        return False
