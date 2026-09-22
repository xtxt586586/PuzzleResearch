from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Principal:
    user_id: str
    authenticated: bool = False

    @property
    def is_anonymous(self) -> bool:
        return not self.authenticated


ANONYMOUS_PRINCIPAL = Principal(
    user_id="anonymous",
    authenticated=False,
)


def resolve_principal(
        *,
        user_id: str | None = None,
        api_key: str | None = None,
        configured_api_key: str | None = None,
) -> Principal:
    normalized_user_id = str(user_id or "").strip()

    if configured_api_key:
        if api_key != configured_api_key:
            raise PermissionError("API Key 无效。")

        return Principal(
            user_id=normalized_user_id or "api-user",
            authenticated=True,
        )

    return Principal(
        user_id=normalized_user_id or "anonymous",
        authenticated=False,
    )
