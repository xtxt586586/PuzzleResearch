from __future__ import annotations

import hashlib
import asyncio
import json
import logging
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agent.prompts import get_prompt_manager
from agent.workflows.research_agent import ResearchAgent, ResearchTask
from common.config import settings
from agent.workflows.research_budget import (
    DEFAULT_DEEPSEARCH_MAX_SECONDS,
    DEFAULT_DEEPSEARCH_MAX_TOKENS,
)


logger = logging.getLogger(__name__)


class ResearchNodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    EMPTY = "empty"
    FAILED = "failed"


NodeStatus = ResearchNodeStatus


class TreeExplorationBudgetExceeded(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = str(reason or "").strip() or "budget_exceeded"


def _estimate_tokens(value: Any) -> int:
    text = value if isinstance(value, str) else str(value or "")
    return (len(text) + 3) // 4 if text else 0


def _reported_token_usage(response: Any) -> int | None:
    response_metadata = getattr(response, "response_metadata", {})
    if not isinstance(response_metadata, Mapping):
        response_metadata = {}
    candidates = [
        getattr(response, "usage_metadata", None),
        response_metadata.get("token_usage"),
    ]
    for usage in candidates:
        if not isinstance(usage, Mapping):
            continue
        try:
            total = usage.get("total_tokens")
            if total is None:
                total = int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
                total += int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0)
            return max(0, int(total))
        except (TypeError, ValueError):
            continue
    return None


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


def _compact_node_findings(findings: list[dict[str, Any]], limit: int = 5) -> list[dict[str, str]]:
    compacted: list[dict[str, str]] = []
    for finding in findings[-max(int(limit), 0):]:
        result = finding.get("result", {})
        if not isinstance(result, Mapping):
            continue
        compacted.append({
            "query": str(finding.get("query", "")),
            "title": str(result.get("title", ""))[:200],
            "url": str(result.get("url", ""))[:2_000],
            "snippet": str(
                result.get("snippet") or result.get("summary") or result.get("content") or ""
            )[:400],
            "provider": str(result.get("provider", ""))[:100],
        })
    return compacted


def _restore_node_findings(value: Any) -> list[dict[str, Any]]:
    restored: list[dict[str, Any]] = []
    if not isinstance(value, list):
        return restored
    for finding in value:
        if not isinstance(finding, Mapping):
            continue
        result = finding.get("result")
        if isinstance(result, Mapping):
            normalized_result = dict(result)
        else:
            normalized_result = {
                key: finding[key]
                for key in ("title", "url", "snippet", "provider")
                if key in finding
            }
        if not normalized_result:
            continue
        restored_finding = {
            "query": str(finding.get("query", "")),
            "result": normalized_result,
        }
        branch_id = finding.get("branch_id")
        if branch_id:
            restored_finding["branch_id"] = str(branch_id)
        restored.append(restored_finding)
    return restored


@dataclass
class ResearchTreeNode:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    topic: str = ""
    query: str = ""
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
    children_ids: list[str] = field(default_factory=list)
    findings: list[dict[str, Any]] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    agent_result: dict[str, Any] = field(default_factory=dict)
    relevance_score: float = 1.0
    created_at: str = field(default_factory=lambda: str(time.time()))
    completed_at: str | None = None

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
            "children_ids": list(self.children_ids),
            "findings_count": max(self.result_count, len(self.findings)),
            "findings": _compact_node_findings(self.findings),
            "sources_count": len(set(self.sources) | set(self.source_urls)),
            "queries": list(self.queries) if self.queries else (
                [self.query] if self.query else []
            ),
            "agent_result": dict(self.agent_result),
            "relevance_score": self.relevance_score,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }

    def mark_complete(self, summary: str = "") -> None:
        self.status = ResearchNodeStatus.COMPLETED
        self.completed_at = str(time.time())
        if summary:
            self.summary = summary

    def mark_failed(self, error: str = "") -> None:
        self.status = ResearchNodeStatus.FAILED
        self.completed_at = str(time.time())
        if error:
            self.error = str(error)
            self.summary = f"[FAILED] {error}"

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
    ) -> "ResearchTreeNode":
        findings = _restore_node_findings(payload.get("findings", []))
        source_urls = [
            str(item)
            for item in (payload.get("source_urls", []) or [])
        ]
        sources = [
            str(item)
            for item in (payload.get("sources", []) or source_urls)
        ]
        if not source_urls:
            source_urls = list(sources)
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
                max(
                    int(payload.get("result_count", 0) or 0),
                    int(payload.get("findings_count", 0) or 0),
                    len(findings),
                )
            ),
            source_urls=source_urls,
            result_keys=[],
            queries=[str(item) for item in payload.get("queries", []) or []],
            agent_result=(
                dict(payload.get("agent_result"))
                if isinstance(payload.get("agent_result"), Mapping)
                else {}
            ),
            summary=str(
                payload.get("summary", "")
            ),
            error=str(
                payload.get("error", "")
            ),
            children_ids=[str(item) for item in payload.get("children_ids", []) or []],
            findings=findings,
            sources=sources,
            relevance_score=float(payload.get("relevance_score", 1.0) or 1.0),
            created_at=str(payload.get("created_at", "")),
            completed_at=payload.get("completed_at"),
        )


@dataclass
class ResearchTree:
    topic: str
    epoch: int = 1
    root_id: str = "research-root"
    max_depth: int = 99
    max_branches: int = 4
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
            max_depth=int(payload.get("max_depth", 99) or 99),
            max_branches=max(int(payload.get("max_branches", 4) or 4), 1),
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
            root = self.nodes.get(parent)
            if root is not None and node.id not in root.children_ids:
                root.children_ids.append(node.id)
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
            "max_depth": self.max_depth,
            "max_branches": self.max_branches,
            "nodes": [
                node.to_dict()
                for node in self.nodes.values()
            ],
        }

    def create_root(self, topic: str) -> ResearchTreeNode:
        root = ResearchTreeNode(
            topic=_normalize(topic),
            query=_normalize(topic),
            depth=0,
            parent_id=None,
            status=ResearchNodeStatus.RUNNING,
        )
        self.root_id = root.id
        self.topic = root.topic
        self.nodes = {root.id: root}
        return root

    def add_child(
        self,
        parent_id: str,
        topic: str,
        relevance_score: float = 1.0,
    ) -> ResearchTreeNode | None:
        parent = self.nodes.get(parent_id)
        if parent is None or parent.depth >= self.max_depth:
            return None
        children = [item for item in self.nodes.values() if item.parent_id == parent_id]
        max_branches = max(int(self.max_branches or 4), 1)
        if len(children) >= max_branches:
            return None
        child = ResearchTreeNode(
            topic=_normalize(topic),
            query=_normalize(topic),
            parent_id=parent_id,
            depth=parent.depth + 1,
            epoch=self.epoch,
            relevance_score=max(0.0, min(float(relevance_score), 1.0)),
        )
        self.nodes[child.id] = child
        parent.children_ids.append(child.id)
        return child

    def get_node(self, node_id: str) -> ResearchTreeNode | None:
        return self.nodes.get(node_id)

    def get_root(self) -> ResearchTreeNode | None:
        return self.nodes.get(self.root_id)

    def get_children(self, node_id: str) -> list[ResearchTreeNode]:
        node = self.nodes.get(node_id)
        if node is None:
            return []
        return [self.nodes[item] for item in node.children_ids if item in self.nodes]

    def get_pending_nodes(self) -> list[ResearchTreeNode]:
        return [node for node in self.nodes.values() if node.status == ResearchNodeStatus.PENDING]

    def get_completed_nodes(self) -> list[ResearchTreeNode]:
        return [node for node in self.nodes.values() if node.status == ResearchNodeStatus.COMPLETED]

    def get_nodes_at_depth(self, depth: int) -> list[ResearchTreeNode]:
        return [node for node in self.nodes.values() if node.depth == depth]

    def get_all_sources(self) -> list[str]:
        return list(dict.fromkeys(
            url for node in self.nodes.values() for url in (node.source_urls + node.sources) if url
        ))

    def get_all_findings(self) -> list[dict[str, Any]]:
        return [item for node in self.nodes.values() for item in node.findings]

    def get_merged_summary(self) -> str:
        lines: list[str] = []
        for node in self.get_completed_nodes():
            if node.summary:
                lines.extend([f"## {node.topic}", node.summary, ""])
        return "\n".join(lines).strip()


def _parse_json_output(value: Any) -> Any:
    text = str(value or "").strip()
    if "```" in text:
        text = text.replace("```json", "").replace("```", "").strip()
    start = min(
        [position for position in (text.find("{"), text.find("[")) if position >= 0],
        default=-1,
    )
    if start >= 0:
        end = max(text.rfind("}"), text.rfind("]")) + 1
        text = text[start:end]
    try:
        return json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return {}


class TreeExplorer:
    """可同步或异步执行的树状研究分支探索器。"""

    def __init__(
        self,
        planner_llm: Any,
        researcher_llm: Any,
        writer_llm: Any,
        search_func: Any,
        config: dict[str, Any] | None = None,
        max_depth: int = 2,
        max_branches: int = 4,
        queries_per_branch: int = 3,
        event_callback: Any = None,
        max_searches: int | None = None,
        max_seconds: float | None = None,
        max_tokens: int | None = None,
        initial_elapsed_seconds: float = 0.0,
        initial_tokens_used: int = 0,
        count_topic_tokens: bool = True,
    ) -> None:
        self.planner_llm = planner_llm
        self.researcher_llm = researcher_llm
        self.writer_llm = writer_llm
        self.search_func = search_func
        self.config = config or {}
        self.max_depth = max(0, int(max_depth))
        self.max_branches = max(1, int(max_branches))
        self.queries_per_branch = max(1, int(queries_per_branch))
        self.event_callback = event_callback
        self.tree: ResearchTree | None = None
        self.all_searched_urls: list[str] = []
        self.start_time = 0.0
        self.search_count = 0
        configured_max_seconds = (
            getattr(settings, "deepsearch_max_seconds", 0.0)
            if max_seconds is None
            else max_seconds
        )
        configured_max_tokens = (
            getattr(settings, "deepsearch_max_tokens", 0)
            if max_tokens is None
            else max_tokens
        )
        self.max_seconds = max(0.0, float(configured_max_seconds or 0.0)) or DEFAULT_DEEPSEARCH_MAX_SECONDS
        self.max_tokens = max(0, int(configured_max_tokens or 0)) or DEFAULT_DEEPSEARCH_MAX_TOKENS
        self.initial_elapsed_seconds = max(0.0, float(initial_elapsed_seconds or 0.0))
        self.tokens_used = max(0, int(initial_tokens_used or 0))
        self.count_topic_tokens = bool(count_topic_tokens)
        self.budget_stop_reason = ""
        self._topic_tokens_counted = False
        configured_max_searches = (
            getattr(settings, "deepsearch_tree_max_searches", 0)
            if max_searches is None
            else max_searches
        )
        self.max_searches = max(0, int(configured_max_searches or 0))
        self._budget_lock = threading.Lock()
        self._model_call_lock = threading.Lock()

    def _start_budget(self, topic: str = "") -> None:
        with self._budget_lock:
            if not self.start_time:
                self.start_time = time.monotonic()
            if topic and not self._topic_tokens_counted and self.count_topic_tokens:
                self.tokens_used += _estimate_tokens(topic)
            if topic:
                self._topic_tokens_counted = True
            self._refresh_budget_locked()

    def _elapsed_seconds_locked(self) -> float:
        current = time.monotonic() - self.start_time if self.start_time else 0.0
        return self.initial_elapsed_seconds + max(0.0, current)

    def _refresh_budget_locked(self) -> str:
        if self.budget_stop_reason:
            return self.budget_stop_reason
        if self.max_seconds and self._elapsed_seconds_locked() >= self.max_seconds:
            self.budget_stop_reason = "time_budget_exceeded"
        elif self.max_tokens and self.tokens_used >= self.max_tokens:
            self.budget_stop_reason = "token_budget_exceeded"
        return self.budget_stop_reason

    def _check_budget(self) -> None:
        self._start_budget()
        with self._budget_lock:
            reason = self._refresh_budget_locked()
        if reason:
            raise TreeExplorationBudgetExceeded(reason)

    def _record_tokens(self, count: int) -> None:
        with self._budget_lock:
            self.tokens_used += max(0, int(count))
            self._refresh_budget_locked()

    def _invoke(self, model: Any, messages: Any) -> Any:
        with self._model_call_lock:
            self._check_budget()
            with self._budget_lock:
                remaining_tokens = (
                    self.max_tokens
                    - self.tokens_used
                    - _estimate_tokens(messages)
                )
                remaining_seconds = (
                    self.max_seconds
                    - self._elapsed_seconds_locked()
                )
                if remaining_tokens <= 0:
                    self.budget_stop_reason = "token_budget_exceeded"
                    raise TreeExplorationBudgetExceeded(
                        self.budget_stop_reason
                    )
                if remaining_seconds <= 0:
                    self.budget_stop_reason = "time_budget_exceeded"
                    raise TreeExplorationBudgetExceeded(
                        self.budget_stop_reason
                    )

            bounded_model = model
            model_copy = getattr(model, "model_copy", None)
            if callable(model_copy):
                bounded_model = model_copy(
                    update={"timeout": max(0.1, remaining_seconds)}
                )
            bind = getattr(bounded_model, "bind", None)
            if callable(bind):
                bounded_model = bind(max_tokens=remaining_tokens)

            response = bounded_model.invoke(messages, config=self.config)
            usage = _reported_token_usage(response)
            if usage is None:
                usage = _estimate_tokens(messages) + _estimate_tokens(
                    getattr(response, "content", "")
                )
            self._record_tokens(usage)
            return response

    def get_budget_stats(self) -> dict[str, Any]:
        self._start_budget()
        with self._budget_lock:
            self._refresh_budget_locked()
            return {
                "max_seconds": self.max_seconds,
                "max_tokens": self.max_tokens,
                "elapsed_seconds": round(self._elapsed_seconds_locked(), 3),
                "tokens_used": self.tokens_used,
                "searches_used": self.search_count,
                "stop_reason": self.budget_stop_reason,
            }

    def _publish_event(
        self,
        event_type: str,
        data: dict[str, Any],
    ) -> None:
        if self.event_callback is None:
            return
        try:
            self.event_callback(event_type, dict(data))
        except Exception as exc:
            logger.debug("树研究事件发布失败：%s", exc)

    @staticmethod
    def _compact_results(
        results: list[dict[str, Any]],
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        compacted: list[dict[str, Any]] = []
        for item in results[: max(0, int(limit))]:
            compacted.append(
                {
                    "title": str(item.get("title", "")),
                    "url": str(item.get("url", "")),
                    "snippet": str(item.get("snippet", "") or item.get("summary", ""))[:500],
                    "provider": str(item.get("provider", "")),
                }
            )
        return compacted

    def _check_cancel(self, state: dict[str, Any]) -> None:
        if state.get("is_cancelled"):
            raise asyncio.CancelledError("Task was cancelled")

    def decompose_topic(
        self,
        topic: str,
        existing_knowledge: str = "",
        num_subtopics: int = 4,
    ) -> list[tuple[str, float]]:
        response = self._invoke(
            self.planner_llm,
            [{"role": "user", "content": f"将主题分解为 {num_subtopics} 个子主题：{topic}\n{existing_knowledge}"}],
        )
        data = _parse_json_output(getattr(response, "content", ""))
        items = data.get("subtopics", []) if isinstance(data, dict) else data
        result: list[tuple[str, float]] = []
        for item in items if isinstance(items, list) else []:
            if isinstance(item, dict):
                value = str(item.get("topic", "")).strip()
                score = float(item.get("relevance", 0.8) or 0.8)
            else:
                value, score = str(item).strip(), 0.8
            if value:
                result.append((value, score))
        return result[: max(0, int(num_subtopics))]

    @staticmethod
    def _parse_list_output(value: Any) -> list[str]:
        data = _parse_json_output(value)
        if isinstance(data, dict):
            data = data.get("queries", data.get("items", []))
        return [str(item).strip() for item in data if str(item).strip()] if isinstance(data, list) else []

    def _search(
        self,
        node: ResearchTreeNode,
        query: str,
        state: dict[str, Any],
        max_results: int,
    ) -> list[dict[str, Any]]:
        self._check_cancel(state)
        with self._budget_lock:
            reason = self._refresh_budget_locked()
            if reason:
                raise TreeExplorationBudgetExceeded(reason)
            if self.max_searches and self.search_count >= self.max_searches:
                self.budget_stop_reason = "search_budget_exceeded"
                raise TreeExplorationBudgetExceeded("search_budget_exceeded")
            self.search_count += 1
        try:
            raw = self.search_func(
                {"query": query, "max_results": max_results},
                config=self.config,
            )
        except TypeError:
            raw = self.search_func({"query": query, "max_results": max_results})
        results = raw.get("results", []) if isinstance(raw, dict) else raw
        normalized_results = [
            item
            for item in results
            if isinstance(item, dict)
        ] if isinstance(results, list) else []
        result_tokens = sum(
            _estimate_tokens(
                " ".join(
                    str(item.get(key, "") or "")
                    for key in ("title", "snippet", "summary", "content")
                )[:2400]
            )
            for item in normalized_results
        )
        self._record_tokens(_estimate_tokens(query) + result_tokens)
        for item in normalized_results:
            url = str(item.get("url", "")).strip()
            if url:
                with self._budget_lock:
                    if url not in self.all_searched_urls:
                        self.all_searched_urls.append(url)
            if url and url not in node.sources:
                node.sources.append(url)
            if url and url not in node.source_urls:
                node.source_urls.append(url)
            node.findings.append({"query": query, "result": item})
        node.result_count += len(normalized_results)
        self._publish_event(
            "search",
            {
                "query": query,
                "mode": "tree",
                "count": len(normalized_results),
                "result_count": len(node.findings),
                "sources_count": len(node.sources),
                "status": node.status.value,
                "results": self._compact_results(normalized_results),
                "node_id": node.id,
                "topic": node.topic,
                "depth": node.depth,
                "parent_id": node.parent_id,
            },
        )
        return normalized_results

    def explore_branch(self, node: ResearchTreeNode, state: dict[str, Any], per_query_results: int = 5) -> None:
        self._start_budget(node.topic)
        self._check_cancel(state)
        self._publish_event(
            "research_node_start",
            {
                "node_id": node.id,
                "topic": node.topic,
                "depth": node.depth,
                "parent_id": node.parent_id,
            },
        )
        node.status = ResearchNodeStatus.RUNNING
        node.query = node.query or node.topic
        node.queries = [node.topic] if node.topic else []

        def web_search(query: str, max_results: int) -> list[dict[str, Any]]:
            return self._search(node, query, state, max_results)

        assigned = state.get("active_research_task") or {}
        if isinstance(assigned, Mapping) and assigned.get("task_id"):
            task_payload = dict(assigned)
            task_payload.setdefault("topic", node.topic)
            task_payload.setdefault("objective", f"为研究主题提供可引用的证据：{node.topic}")
            task_payload.setdefault("context", {})
            task_payload.setdefault("allowed_tools", ["web_search"])
            task = ResearchTask.model_validate(task_payload)
        else:
            task = ResearchTask(
                task_id=node.id,
                topic=node.topic,
                objective=f"为研究主题提供可引用的证据：{node.topic}",
                priority=min(5, max(1, round(float(node.relevance_score or 1.0) * 5))),
                context={
                    "overall_topic": str(state.get("input", "") or "")[:500],
                    "parent_id": node.parent_id,
                    "depth": node.depth,
                    "epoch": node.epoch,
                },
                allowed_tools={"web_search"},
            )

        def execute(agent_context: Any) -> dict[str, Any]:
            if node.queries:
                agent_context.call_tool(
                    "web_search",
                    query=node.queries[0],
                    max_results=per_query_results,
                )
            remaining = max(0, self.queries_per_branch - len(node.queries))
            if assigned:
                remaining = 0
            if remaining:
                response = self._invoke(
                    self.planner_llm,
                    [{"role": "user", "content": f"为主题生成 {remaining} 个补充查询：{node.topic}"}],
                )
                seen = {item.casefold() for item in node.queries}
                for query in self._parse_list_output(getattr(response, "content", "")):
                    if query.casefold() not in seen:
                        seen.add(query.casefold())
                        node.queries.append(query)
                    if len(node.queries) >= self.queries_per_branch:
                        break
            for query in node.queries[1:]:
                agent_context.call_tool(
                    "web_search",
                    query=query,
                    max_results=per_query_results,
                )
            return {"summary": self._summarize_branch(node)}

        result = ResearchAgent({"web_search": web_search}).run(task, execute)
        node.agent_result = result.model_dump(mode="json")
        if result.status == "failed":
            node.mark_failed(result.error)
        elif result.status == "empty":
            node.status = ResearchNodeStatus.EMPTY
            node.completed_at = str(time.time())
        else:
            summary = result.summary or self._local_branch_summary(node)
            node.mark_complete(summary)

        self._publish_event(
            "research_node_complete",
            {
                "node_id": node.id,
                "agent_id": result.task_id,
                "task_id": result.task_id,
                "allowed_tools": sorted(task.allowed_tools),
                "tool_calls": list(result.tool_calls),
                "topic": node.topic,
                "depth": node.depth,
                "status": node.status.value,
                "summary": node.summary,
                "sources": list(node.sources),
                "finding_count": len(node.findings),
                "findings": _compact_node_findings(node.findings),
                "error": result.error,
                "budget_stop_reason": result.stop_reason,
            },
        )
        if result.stop_reason:
            raise TreeExplorationBudgetExceeded(result.stop_reason)

    @staticmethod
    def _local_branch_summary(node: ResearchTreeNode) -> str:
        lines: list[str] = []
        for finding in node.findings[:10]:
            result = finding.get("result", {})
            if not isinstance(result, dict):
                continue
            title = str(result.get("title", "")).strip() or "未命名来源"
            excerpt = str(
                result.get("content")
                or result.get("snippet")
                or result.get("summary", "")
            ).strip()
            url = str(result.get("url", "")).strip()
            if excerpt:
                excerpt = excerpt[:500]
            suffix = f"（{url}）" if url else ""
            lines.append(f"- {title}：{excerpt}{suffix}".strip())
        return "\n".join(lines)

    def _summarize_branch(self, node: ResearchTreeNode) -> str:
        """用研究模型总结单个分支，模型失败时退回本地摘要。"""
        if not node.findings:
            return ""

        findings_text: list[str] = []
        for index, finding in enumerate(node.findings[:10], start=1):
            result = finding.get("result", {})
            if not isinstance(result, dict):
                continue
            findings_text.append(
                f"[{index}] 标题：{str(result.get('title', '')).strip()}\n"
                f"URL：{str(result.get('url', '')).strip()}\n"
                f"摘要：{str(result.get('content') or result.get('snippet') or result.get('summary', '')).strip()[:800]}"
            )

        prompt = (
            f"{get_prompt_manager(self.config).get_prompt('tree_branch_summarizer')}\n"
            f"分支主题：{node.topic}\n"
            "使用简洁要点，最多 500 字。\n\n"
            + "\n\n".join(findings_text)
        )
        try:
            response = self._invoke(
                self.researcher_llm,
                [{"role": "user", "content": prompt}],
            )
            summary = str(getattr(response, "content", "") or "").strip()
            if summary:
                return summary
        except TreeExplorationBudgetExceeded:
            return self._local_branch_summary(node)
        except Exception as exc:
            logger.debug("分支摘要模型不可用，使用本地摘要：%s", exc)
        return self._local_branch_summary(node)

    def merge_branches(self, nodes: list[ResearchTreeNode]) -> str:
        """合并已完成分支，模型失败时按主题拼接分支摘要。"""
        summaries = [
            f"## {node.topic}\n{node.summary}"
            for node in nodes
            if node.status == ResearchNodeStatus.COMPLETED and node.summary
        ]
        if not summaries:
            return ""

        prompt = (
            f"{get_prompt_manager(self.config).get_prompt('tree_merge_summarizer')}\n\n"
            + "\n\n".join(summaries)
        )
        try:
            response = self._invoke(
                self.writer_llm,
                [{"role": "user", "content": prompt}],
            )
            merged = str(getattr(response, "content", "") or "").strip()
            if merged:
                return merged
        except TreeExplorationBudgetExceeded:
            return "\n\n".join(summaries)
        except Exception as exc:
            logger.debug("分支合并模型不可用，使用本地合并：%s", exc)
        return "\n\n".join(summaries)

    async def explore_branch_async(self, node: ResearchTreeNode, state: dict[str, Any], per_query_results: int = 5) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.explore_branch, node, state, per_query_results)

    def _explore_children(self, parent: ResearchTreeNode, state: dict[str, Any]) -> None:
        if parent.depth >= self.max_depth or self.tree is None:
            return
        for topic, score in self.decompose_topic(parent.topic, parent.summary, min(2, self.max_branches)):
            child = self.tree.add_child(parent.id, topic, score)
            if child is None:
                continue
            self.explore_branch(child, state)
            self._explore_children(child, state)

    async def _explore_children_async(self, parent: ResearchTreeNode, state: dict[str, Any], semaphore: asyncio.Semaphore | None = None) -> None:
        if parent.depth >= self.max_depth or self.tree is None:
            return
        semaphore = semaphore or asyncio.Semaphore(max(
            1,
            int(getattr(
                settings,
                "deepsearch_tree_parallel_branches",
                getattr(settings, "tree_parallel_branches", 3),
            ) or 3),
        ))
        children = [
            child for topic, score in self.decompose_topic(parent.topic, parent.summary, min(2, self.max_branches))
            if (child := self.tree.add_child(parent.id, topic, score)) is not None
        ]

        async def explore_child(child: ResearchTreeNode) -> None:
            async with semaphore:
                await self.explore_branch_async(child, dict(state))
            await self._explore_children_async(child, dict(state), semaphore)

        await asyncio.gather(*(explore_child(child) for child in children))

    def run(self, topic: str, state: dict[str, Any], decompose_root: bool = True) -> ResearchTree:
        self._start_budget(topic)
        self._check_cancel(state)
        self.tree = ResearchTree(topic=_normalize(topic), max_depth=self.max_depth, max_branches=self.max_branches)
        root = self.tree.create_root(topic)
        self.explore_branch(root, state)
        if decompose_root and self.max_depth > 0:
            self._explore_children(root, state)
        return self.tree

    async def run_async(self, topic: str, state: dict[str, Any], decompose_root: bool = True) -> ResearchTree:
        self._start_budget(topic)
        self._check_cancel(state)
        self.tree = ResearchTree(topic=_normalize(topic), max_depth=self.max_depth, max_branches=self.max_branches)
        root = self.tree.create_root(topic)
        await self.explore_branch_async(root, state)
        if decompose_root and self.max_depth > 0:
            await self._explore_children_async(root, state)
        return self.tree

    def get_final_summary(self) -> str:
        if self.tree is None:
            return ""
        merged = self.merge_branches(self.tree.get_completed_nodes())
        return merged or self.tree.get_merged_summary()

    def get_all_sources(self) -> list[str]:
        return self.tree.get_all_sources() if self.tree else list(self.all_searched_urls)

    def get_all_findings(self) -> list[dict[str, Any]]:
        return self.tree.get_all_findings() if self.tree else []
