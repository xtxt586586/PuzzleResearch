from __future__ import annotations

import logging
from typing import Any


logger = logging.getLogger(__name__)

def _create_postgres_checkpointer(database_url: str) -> Any:
    import psycopg
    from langgraph.checkpoint.postgres import PostgresSaver

    connection = None

    try:
        connection = psycopg.connect(database_url)
        checkpointer = PostgresSaver(connection)
        checkpointer.setup()
        return checkpointer
    except Exception:
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass
        raise


def create_checkpointer(
        database_url: str | None = None,
        *,
        fallback_to_memory: bool = True,
) -> Any:
    normalized_url = str(database_url or "").strip()

    if not normalized_url:
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver()

    try:
        checkpointer = _create_postgres_checkpointer(
            normalized_url,
        )
    except Exception as exc:
        if not fallback_to_memory:
            raise RuntimeError(
                f"PostgreSQL checkpointer 初始化失败：{exc} "
            )

        logger.warning(
            f"PostgreSQL checkpointer 初始化失败，回退到内存检查点 checkpointer: %s",
            exc,
        )

        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver()

    logger.info("PostgreSQL checkpointer 初始化完成。")
    return checkpointer