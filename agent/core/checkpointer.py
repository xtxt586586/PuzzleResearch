from __future__ import annotations

import asyncio
import logging
from typing import Any


logger = logging.getLogger(__name__)


def close_checkpointer(checkpointer: Any) -> None:
    connection = getattr(checkpointer, "conn", None)
    close = getattr(connection, "close", None)
    if callable(close):
        close()


def _create_postgres_checkpointer(database_url: str) -> Any:
    import psycopg
    from langgraph.checkpoint.postgres import PostgresSaver

    class AsyncCompatiblePostgresSaver(PostgresSaver):
        async def aget_tuple(self, config: Any) -> Any:
            return await asyncio.to_thread(self.get_tuple, config)

        async def alist(
            self,
            config: Any,
            *,
            filter: Any = None,
            before: Any = None,
            limit: int | None = None,
        ):
            checkpoints = await asyncio.to_thread(
                lambda: list(
                    self.list(
                        config,
                        filter=filter,
                        before=before,
                        limit=limit,
                    )
                )
            )
            for checkpoint in checkpoints:
                yield checkpoint

        async def aput(
            self,
            config: Any,
            checkpoint: Any,
            metadata: Any,
            new_versions: Any,
        ) -> Any:
            return await asyncio.to_thread(
                self.put,
                config,
                checkpoint,
                metadata,
                new_versions,
            )

        async def aput_writes(
            self,
            config: Any,
            writes: Any,
            task_id: str,
            task_path: str = "",
        ) -> None:
            await asyncio.to_thread(
                self.put_writes,
                config,
                writes,
                task_id,
                task_path,
            )

        async def adelete_thread(self, thread_id: str) -> None:
            await asyncio.to_thread(self.delete_thread, thread_id)

    connection = None

    try:
        connection = psycopg.connect(
            database_url,
            autocommit=True,
            prepare_threshold=0,
        )
        checkpointer = AsyncCompatiblePostgresSaver(connection)
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
