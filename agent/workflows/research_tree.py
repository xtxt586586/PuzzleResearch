from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ResearchNodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    EMPTY = "empty"
    FAILED = "failed"


def _normalize(value: Any) -> str:
    return " ".join(
        str(value or "").strip().split()
    )


def _node_id(
    *,
    topic: str,
    query: str,
    epoch: int,
) -> str:
    raw = (
        f"{epoch}|{_normalize(topic).lower()}|"
        f"{_normalize(query).lower()}"
    )

    digest = hashlib.sha1(
        raw.encode("utf-8")
    ).hexdigest()[:12]

    return f"research-{epoch}-{digest}"


@dataclass
class ResearchTreeNode:
    id: str
    topic: str
    query: str
    parent_id: str | None = None
    depth: int = 0
    epoch: int = 1
    status: ResearchNodeStatus = (
        ResearchNodeStatus.PENDING
    )
    result_count: int = 0
    source_urls: list[str] = field(
        default_factory=list
    )
    result_keys: list[str] = field(
        default_factory=list
    )
    summary: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "topic": self.topic,
            "query": self.query,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "epoch": self.epoch,
            "status": self.status.value,
            "result_count": self.result_count,
            "source_urls": list(
                self.source_urls
            ),
            "summary": self.summary,
            "error": self.error,
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
    ) -> "ResearchTreeNode":
        return cls(
            id=str(payload.get("id", "")),
            topic=str(
                payload.get("topic", "")
            ),
            query=str(
                payload.get("query", "")
            ),
            parent_id=payload.get(
                "parent_id"
            ),
            depth=int(
                payload.get("depth", 0)
                or 0
            ),
            epoch=int(
                payload.get("epoch", 1)
                or 1
            ),
            status=ResearchNodeStatus(
                payload.get(
                    "status",
                    ResearchNodeStatus.PENDING.value,
                )
            ),
            result_count=int(
                payload.get("result_count", 0)
                or 0
            ),
            source_urls=[
                str(item)
                for item in (
                    payload.get(
                        "source_urls",
                        [],
                    )
                    or []
                )
            ],
            result_keys=[],
            summary=str(
                payload.get("summary", "")
            ),
            error=str(
                payload.get("error", "")
            ),
        )


@dataclass
class ResearchTree:
    topic: str
    epoch: int = 1
    root_id: str = "research-root"
    nodes: dict[str, ResearchTreeNode] = field(
        default_factory=dict
    )

    @classmethod
    def from_queries(
        cls,
        topic: str,
        queries: list[str],
        *,
        epoch: int = 1,
        max_branches: int = 4,
    ) -> "ResearchTree":
        tree = cls(
            topic=_normalize(topic),
            epoch=max(int(epoch), 1),
        )

        root = ResearchTreeNode(
            id=tree.root_id,
            topic=tree.topic,
            query=tree.topic,
            depth=0,
            epoch=tree.epoch,
            status=ResearchNodeStatus.RUNNING,
        )

        tree.nodes[root.id] = root
        tree.add_queries(
            queries,
            epoch=tree.epoch,
            max_branches=max_branches,
        )

        return tree

    @classmethod
    def from_dict(
        cls,
        payload: Any,
    ) -> "ResearchTree":
        if not isinstance(payload, dict):
            return cls(topic="")

        tree = cls(
            topic=str(
                payload.get("topic", "")
            ),
            epoch=int(
                payload.get("epoch", 1)
                or 1
            ),
            root_id=str(
                payload.get(
                    "root_id",
                    "research-root",
                )
            ),
        )

        raw_nodes = payload.get(
            "nodes",
            [],
        )

        if isinstance(raw_nodes, dict):
            raw_nodes = list(
                raw_nodes.values()
            )

        for item in raw_nodes:
            if not isinstance(item, dict):
                continue

            node = ResearchTreeNode.from_dict(
                item
            )

            if node.id:
                tree.nodes[node.id] = node

        if tree.root_id not in tree.nodes:
            tree.nodes[tree.root_id] = (
                ResearchTreeNode(
                    id=tree.root_id,
                    topic=tree.topic,
                    query=tree.topic,
                    depth=0,
                    epoch=tree.epoch,
                    status=ResearchNodeStatus.RUNNING,
                )
            )

        return tree

    def add_queries(
        self,
        queries: list[str],
        *,
        epoch: int | None = None,
        max_branches: int = 4,
        parent_id: str | None = None,
    ) -> list[str]:
        normalized_epoch = max(
            int(epoch or self.epoch),
            1,
        )

        parent = parent_id or self.root_id
        created: list[str] = []
        seen: set[str] = set()

        for raw_query in queries:
            query = _normalize(raw_query)

            if not query:
                continue

            key = query.casefold()

            if key in seen:
                continue

            seen.add(key)

            existing = next(
                (
                    node
                    for node in self.nodes.values()
                    if (
                        node.query.casefold()
                        == key
                    )
                ),
                None,
            )

            if existing is not None:
                continue

            node_id = _node_id(
                topic=self.topic,
                query=query,
                epoch=normalized_epoch,
            )

            node = ResearchTreeNode(
                id=node_id,
                topic=self.topic,
                query=query,
                parent_id=parent,
                depth=1,
                epoch=normalized_epoch,
            )

            self.nodes[node.id] = node
            created.append(node.id)

            if len(created) >= max(
                int(max_branches),
                1,
            ):
                break

        return created

    def record_results(
        self,
        results: list[dict[str, Any]],
    ) -> None:
        grouped: dict[str, list[dict[str, Any]]] = {}

        for result in results:
            if not isinstance(result, dict):
                continue

            query = _normalize(
                result.get("query", "")
            )

            if query:
                grouped.setdefault(
                    query.casefold(),
                    [],
                ).append(result)

        for node in self.nodes.values():
            if node.depth == 0:
                continue

            items = grouped.get(
                node.query.casefold(),
                [],
            )

            if not items:
                continue

            for item in items:
                url = _normalize(
                    item.get("url", "")
                )
                title = _normalize(
                    item.get("title", "")
                )
                result_key = url or title

                if result_key:
                    if (
                        result_key
                        not in node.result_keys
                    ):
                        node.result_keys.append(
                            result_key
                        )

                    if (
                        url
                        and url not in node.source_urls
                    ):
                        node.source_urls.append(
                            url
                        )

            node.result_count = len(
                node.result_keys
            )
            node.status = (
                ResearchNodeStatus.COMPLETED
            )

        self._refresh_root()

    def finalize(self) -> None:
        for node in self.nodes.values():
            if node.depth == 0:
                continue

            if node.status in {
                ResearchNodeStatus.PENDING,
                ResearchNodeStatus.RUNNING,
            }:
                node.status = (
                    ResearchNodeStatus.COMPLETED
                    if node.result_count
                    else ResearchNodeStatus.EMPTY
                )

        self._refresh_root()

    def _refresh_root(self) -> None:
        root = self.nodes.get(self.root_id)

        if root is None:
            return

        children = [
            node
            for node in self.nodes.values()
            if node.parent_id == self.root_id
        ]

        root.result_count = sum(
            node.result_count
            for node in children
        )

        root.source_urls = list(
            dict.fromkeys(
                url
                for node in children
                for url in node.source_urls
            )
        )

        if children and all(
            node.status
            in {
                ResearchNodeStatus.COMPLETED,
                ResearchNodeStatus.EMPTY,
                ResearchNodeStatus.FAILED,
            }
            for node in children
        ):
            root.status = (
                ResearchNodeStatus.COMPLETED
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "epoch": self.epoch,
            "root_id": self.root_id,
            "nodes": [
                node.to_dict()
                for node in self.nodes.values()
            ],
        }