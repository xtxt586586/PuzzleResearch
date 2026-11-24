from __future__ import annotations

import json
import os
import tempfile
import uuid
from collections.abc import Iterable,Mapping
from datetime import datetime,timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel,ConfigDict,Field,StrictBool,field_validator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENTS_PATH = PROJECT_ROOT / "data" / "agents.json"

DEFAULT_AGENT_ID = "default"


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


class AgentStoreError(RuntimeError):
    pass

class AgentAlreadyExistError(RuntimeError):
    pass

class AgentNotFoundError(RuntimeError):
    pass

class DefaultAgentProtectedError(RuntimeError):
    pass

class AgentRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = ""
    system_prompt: str = ""
    model: str = ""
    enabled_tools: dict[str,bool] = Field(
        default_factory=dict
    )
    mcp_servers: dict[str,StrictBool] | None = None
    metadata: dict[str,Any] = Field(
        default_factory=dict
    )
    created_at: str = Field(
        default_factory=_timestamp
    )
    updated_at: str = Field(
        default_factory=_timestamp
    )

    @field_validator("id","name")
    @classmethod
    def _validate_text(
            cls,
            value: str
    ) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Agent 的 id 和 name 不能为空。")

        return value




def create_default_agent() -> AgentRecord:
    timestamp = _timestamp()

    return AgentRecord(
        id=DEFAULT_AGENT_ID,
        name="default",
        description="PuzzleResearch 默认 Agent",
        system_prompt="",
        model="",
        enabled_tools={},
        mcp_servers=None,
        metadata={},
        created_at=timestamp,
        updated_at=timestamp,
    )


def _coerce_record(
        value: AgentRecord | Mapping[str,Any]
) -> AgentRecord:
    if isinstance(value,AgentRecord):
        return value

    if isinstance(value,Mapping):
        try:
            return AgentRecord.model_validate(value)
        except Exception as exc:
            raise AgentStoreError(
                "Agent 存储记录格式无效。"
            ) from exc

    raise AgentStoreError(
        "Agent 存储记录必须是对象。"
    )


def _normalize_records(
        records: Iterable[
            AgentRecord | Mapping[str,Any]
        ],
) -> list[AgentRecord]:
    normalized: list[AgentRecord] = []
    seen_ids: set[str] = set()

    for item in records:
        record = _coerce_record(item)

        if record.id in seen_ids:
            raise AgentStoreError(
                f"Agent id 重复：{record.id}"
            )

        seen_ids.add(record.id)
        normalized.append(record)

    if DEFAULT_AGENT_ID not in seen_ids:
        normalized.insert(
            0,
            create_default_agent(),
        )

    return normalized


class AgentStore:
    def __init__(
            self,
            path: str | Path | None = None,
    ) -> None:
        self.path = Path(path) if path is not None else AGENTS_PATH

    def _read_records(self) -> list[AgentRecord]:
        if not self.path.exists():
            return [create_default_agent()]

        try:
            raw_text = self.path.read_text(
                encoding="utf-8",
            )

            raw_data = json.loads(
                raw_text or "[]"
            )

        except FileNotFoundError:
            return [create_default_agent()]

        except (
            OSError,
            json.JSONDecodeError,
        ) as exc:
            raise AgentStoreError(
                f"无法读取 Agent 存储文件：{self.path}"
            ) from exc

        if isinstance(raw_data,list):
            raw_records = raw_data

        elif isinstance(raw_data,Mapping):
            raw_records = raw_data.get(
                "agents"
            )

        else:
            raw_records = None

        if not isinstance(raw_records,list):
            raise AgentStoreError(
                "Agent 存储文件格式无效，必须是数组或包含 agents 数组的对象。"
            )

        return _normalize_records(raw_records)


    def _atomic_write(
            self,
            records: list[AgentRecord],
    ) -> None:
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "agents": [
                record.model_dump(mode="json")
                for record in records
            ]
        }

        temporary_path: Path | None = None

        try:
            descriptor,temporary_name = tempfile.mkstemp(
                prefix=".agents-",
                suffix=".tmp",
                dir=self.path.parent,
            )

            temporary_path = Path(temporary_name)

            with os.fdopen(
                descriptor,
                "w",
                encoding="utf-8",
                newline="\n",
            ) as handle:
                json.dump(
                    payload,
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())

            os.replace(
                temporary_path,
                self.path,
            )

        except (
            OSError,
            TypeError,
            ValueError,
        ) as exc:
            raise AgentStoreError(
                f"无法写入 Agent 存储文件：{self.path}"
            ) from exc

        finally:
            if (
                temporary_path is not None
                and temporary_path.exists()
            ):
                try:
                    temporary_path.unlink()
                except OSError:
                    pass


    def list(self) -> list[AgentRecord]:
        return self._read_records()

    def get(
            self,
            agent_id: str,
    ) -> AgentRecord | None:
        normalized_id = str(agent_id).strip()

        for record in self._read_records():
            if record.id == normalized_id:
                return record

        return None

    def save(
            self,
            records: Iterable[AgentRecord | Mapping[str,Any]],
    ) -> list[AgentRecord]:
        normalized = _normalize_records(records)
        self._atomic_write(normalized)
        return normalized


    def create(
            self,
            record: AgentRecord,
    ) -> AgentRecord:
        records = self._read_records()

        if any(
            item.id == record.id
            for item in records
        ):
            raise AgentAlreadyExistError(
                f"Agent 已存在：{record.id}"
            )

        records.append(record)
        self._atomic_write(records)

        return record


    def update(
            self,
            agent_id: str,
            record: AgentRecord,
    ) -> AgentRecord:
        normalized_id = str(agent_id).strip()

        if normalized_id == DEFAULT_AGENT_ID:
            raise DefaultAgentProtectedError(
                "默认 Agent 不允许更新。"
            )

        records = self._read_records()

        for index,current in enumerate(records):
            if current.id != normalized_id:
                continue

            updated = record.model_copy(
                update={
                    "id": normalized_id,
                    "created_at": current.created_at,
                    "updated_at": _timestamp(),
                }
            )

            records[index] = updated
            self._atomic_write(records)
            return updated

        raise AgentNotFoundError(
            f"Agent 不存在：{normalized_id}"
        )


    def upsert(
            self,
            record: AgentRecord,
    ) -> AgentRecord:
        existing = self.get(record.id)

        if existing is None:
            return self.create(record)

        return self.update(
            record.id,
            record,
        )


    def delete(
            self,
            agent_id: str,
    ) -> None:
        normalized_id = str(agent_id).strip()

        if normalized_id == DEFAULT_AGENT_ID:
            raise DefaultAgentProtectedError(
                "默认 Agent 不允许删除。"
            )

        records = self._read_records()

        remaining = [
            record
            for record in records
            if record.id != normalized_id
        ]

        if len(remaining) == len(records):
            raise AgentNotFoundError(
                f"Agent 不存在：{normalized_id}"
            )

        self._atomic_write(remaining)


def load_agents(
        path: str | Path | None = None,
) -> list[AgentRecord]:
    return list_agents(path)

def save_agents(
        records: Iterable[
            AgentRecord | Mapping[str,Any]
        ],
        path: str | Path | None = None,
) -> list[AgentRecord]:
    return AgentStore(path).save(records)

def list_agents(
        path: str | Path | None = None,
) -> list[AgentRecord]:
    return AgentStore(path).list()

def get_agent(
        agent_id: str,
        path: str | Path | None = None,
) -> AgentRecord | None:
    return AgentStore(path).get(agent_id)


def create_agent(
        record: AgentRecord,
        path: str | Path | None = None,
) -> AgentRecord:
    return AgentStore(path).create(record)

def update_agent(
        agent_id: str,
        record: AgentRecord,
        path: str | Path | None = None,
) -> AgentRecord:
    return AgentStore(path).update(agent_id,record)



def upsert_agent(
        record: AgentRecord,
        path: str | Path | None = None,
) -> AgentRecord:
    return AgentStore(path).upsert(record)

def delete_agent(
        agent_id: str,
        path: str | Path | None = None,
) -> None:
    return AgentStore(path).delete(agent_id)
