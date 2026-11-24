from __future__ import annotations

from typing import Any


_CONTEXT_MANAGER_ATTRIBUTE = "_puzzleresearch_context_manager"


def _setup_context_managed_store(context_manager: Any) -> Any:
    store = context_manager.__enter__()
    try:
        store.setup()
    except BaseException as exc:
        context_manager.__exit__(type(exc), exc, exc.__traceback__)
        raise

    setattr(store, _CONTEXT_MANAGER_ATTRIBUTE, context_manager)
    return store


def close_long_term_store(store: Any) -> None:
    context_manager = getattr(store, _CONTEXT_MANAGER_ATTRIBUTE, None)
    if context_manager is not None:
        setattr(store, _CONTEXT_MANAGER_ATTRIBUTE, None)
        context_manager.__exit__(None, None, None)
        return

    close = getattr(store, "close", None)
    if callable(close):
        close()


def create_long_term_store(
    backend: str,
    url: str = "",
) -> Any:
    normalized_backend = str(backend or "memory").strip().lower()
    normalized_url = str(url or "").strip()

    if normalized_backend in {"", "memory"}:
        from langgraph.store.memory import InMemoryStore

        return InMemoryStore()

    if normalized_backend == "postgres":
        if not normalized_url:
            raise ValueError(
                "memory_store_url is required when memory_store_backend=postgres"
            )
        from langgraph.store.postgres import PostgresStore

        return _setup_context_managed_store(
            PostgresStore.from_conn_string(normalized_url)
        )

    if normalized_backend == "redis":
        if not normalized_url:
            raise ValueError(
                "memory_store_url is required when memory_store_backend=redis"
            )
        from langgraph.store.redis import RedisStore

        return _setup_context_managed_store(
            RedisStore.from_conn_string(normalized_url)
        )

    raise ValueError(
        "memory_store_backend must be memory, postgres, or redis"
    )
