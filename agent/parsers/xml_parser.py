from __future__ import annotations

import hashlib
import html
import json
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any


logger = logging.getLogger(__name__)
_FUNCTION_CALLS = re.compile(
    r"<function_calls\b[^>]*>.*?</function_calls\s*>",
    flags=re.IGNORECASE | re.DOTALL,
)
_INVOKE_BLOCK = re.compile(
    r"<invoke\b(?P<attributes>[^>]*)>(?P<body>.*?)</invoke\s*>",
    flags=re.IGNORECASE | re.DOTALL,
)
_ATTRIBUTE_NAME = re.compile(
    r"\bname\s*=\s*(?P<quote>[\"'])(?P<name>.*?)(?P=quote)",
    flags=re.IGNORECASE | re.DOTALL,
)
_CHILD_NAME = re.compile(
    r"<name\b[^>]*>(.*?)</name\s*>",
    flags=re.IGNORECASE | re.DOTALL,
)
_PARAMETER_BLOCK = re.compile(
    r"<parameter\b(?P<attributes>[^>]*)>(?P<value>.*?)</parameter\s*>",
    flags=re.IGNORECASE | re.DOTALL,
)


@dataclass
class XMLToolCall:
    function_name: str
    parameters: dict[str, Any]
    raw_xml: str = ""
    call_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "function_name": self.function_name,
            "parameters": self.parameters,
            "raw_xml": self.raw_xml,
            "call_id": self.call_id,
        }

    def to_openai_format(self) -> dict[str, Any]:
        call_id = self.call_id or "call_" + hashlib.sha256(
            self.raw_xml.encode("utf-8")
        ).hexdigest()[:16]
        return {
            "id": call_id,
            "type": "function",
            "function": {
                "name": self.function_name,
                "arguments": json.dumps(self.parameters, ensure_ascii=False),
            },
        }


class XMLToolParser:
    """Parse PuzzleResearch-style ``function_calls`` blocks into validated data shapes."""

    def __init__(self) -> None:
        self.parsed_calls = 0
        self.parse_errors = 0

    def parse_content(self, content: str) -> list[XMLToolCall]:
        calls: list[XMLToolCall] = []
        if not isinstance(content, str) or not content.strip():
            return calls

        for match in _FUNCTION_CALLS.finditer(content):
            raw_block = match.group(0)
            try:
                root = ET.fromstring(raw_block)
            except ET.ParseError as exc:
                self.parse_errors += 1
                recovered = self._parse_malformed_block(raw_block)
                if recovered:
                    calls.extend(recovered)
                    self.parsed_calls += len(recovered)
                    logger.debug(
                        "从格式不规范的 XML 中恢复了 %s 个工具调用：%s",
                        len(recovered),
                        exc,
                    )
                else:
                    logger.debug("忽略格式错误的 XML 工具调用：%s", exc)
                continue

            for invoke in root.iter():
                if invoke.tag.rsplit("}", 1)[-1].lower() != "invoke":
                    continue
                name = str(invoke.attrib.get("name", "")).strip()
                if not name:
                    name_node = next(
                        (
                            node
                            for node in invoke
                            if node.tag.rsplit("}", 1)[-1].lower() == "name"
                        ),
                        None,
                    )
                    name = "".join(name_node.itertext()).strip() if name_node is not None else ""
                if not name:
                    self.parse_errors += 1
                    continue

                parameters: dict[str, Any] = {}
                for parameter in invoke.iter():
                    if parameter.tag.rsplit("}", 1)[-1].lower() != "parameter":
                        continue
                    parameter_name = str(parameter.attrib.get("name", "")).strip()
                    if not parameter_name:
                        self.parse_errors += 1
                        continue
                    value = "".join(parameter.itertext()).strip()
                    parameters[parameter_name] = self._parse_value(value)

                call_xml = ET.tostring(invoke, encoding="unicode")
                calls.append(XMLToolCall(name, parameters, call_xml))
                self.parsed_calls += 1

        return calls

    def _parse_malformed_block(self, raw_block: str) -> list[XMLToolCall]:
        """Recover complete invoke blocks when strict XML parsing fails."""
        invoke_open_count = len(
            re.findall(r"<invoke\b", raw_block, flags=re.IGNORECASE)
        )
        invoke_close_count = len(
            re.findall(r"</invoke\s*>", raw_block, flags=re.IGNORECASE)
        )
        parameter_open_count = len(
            re.findall(r"<parameter\b", raw_block, flags=re.IGNORECASE)
        )
        parameter_close_count = len(
            re.findall(r"</parameter\s*>", raw_block, flags=re.IGNORECASE)
        )
        if (
            invoke_open_count != invoke_close_count
            or parameter_open_count != parameter_close_count
        ):
            return []

        recovered: list[XMLToolCall] = []
        for match in _INVOKE_BLOCK.finditer(raw_block):
            attributes = match.group("attributes")
            body = match.group("body")
            name_match = _ATTRIBUTE_NAME.search(attributes)
            name = name_match.group("name").strip() if name_match else ""
            if not name:
                child_name = _CHILD_NAME.search(body)
                if child_name:
                    name = html.unescape(child_name.group(1)).strip()
            if not name:
                continue

            parameters: dict[str, Any] = {}
            for parameter in _PARAMETER_BLOCK.finditer(body):
                parameter_name_match = _ATTRIBUTE_NAME.search(
                    parameter.group("attributes")
                )
                if not parameter_name_match:
                    continue
                parameter_name = parameter_name_match.group("name").strip()
                if not parameter_name:
                    continue

                cdata_values: list[str] = []

                def preserve_cdata(match: re.Match[str]) -> str:
                    cdata_values.append(match.group(1))
                    return f"\x00{len(cdata_values) - 1}\x00"

                value = re.sub(
                    r"<!\[CDATA\[(.*?)\]\]>",
                    preserve_cdata,
                    parameter.group("value"),
                    flags=re.IGNORECASE | re.DOTALL,
                )
                value = re.sub(r"<[^>]*>", "", value)
                value = html.unescape(value).strip()
                for index, cdata_value in enumerate(cdata_values):
                    value = value.replace(f"\x00{index}\x00", cdata_value)
                parameters[parameter_name] = self._parse_value(value)

            recovered.append(
                XMLToolCall(
                    function_name=html.unescape(name),
                    parameters=parameters,
                    raw_xml=match.group(0),
                )
            )

        return recovered

    @staticmethod
    def _parse_value(value: str) -> Any:
        if not value:
            return ""
        if value.startswith(("{", "[")):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered in {"null", "none"}:
            return None
        try:
            if any(marker in lowered for marker in (".", "e")):
                return float(value)
            return int(value)
        except ValueError:
            return value

    def parse_streaming_content(
        self,
        accumulated_content: str,
        previous_calls: list[XMLToolCall],
    ) -> list[XMLToolCall]:
        return self.parse_content(accumulated_content)[len(previous_calls):]

    def has_complete_function_calls(self, content: str) -> bool:
        return bool(_FUNCTION_CALLS.search(content or ""))

    def extract_thinking_and_calls(
        self,
        content: str,
    ) -> tuple[str | None, list[XMLToolCall]]:
        match = _FUNCTION_CALLS.search(content or "")
        if not match:
            return content.strip() or None, []
        thinking = content[:match.start()].strip() or None
        return thinking, self.parse_content(content)

    @staticmethod
    def validate_tool_call(call: XMLToolCall) -> tuple[bool, str | None]:
        if not call.function_name.strip():
            return False, "工具名称不能为空"
        if any(not str(name).strip() for name in call.parameters):
            return False, "工具参数名称不能为空"
        return True, None

    def get_statistics(self) -> dict[str, int]:
        return {"parsed_calls": self.parsed_calls, "parse_errors": self.parse_errors}


def convert_xml_to_openai_format(calls: list[XMLToolCall]) -> list[dict[str, Any]]:
    return [call.to_openai_format() for call in calls]


def extract_tool_calls_from_response(
    response_content: str,
    format_type: str = "auto",
) -> tuple[str | None, list[XMLToolCall]]:
    if format_type not in {"auto", "xml", "native"}:
        raise ValueError("format_type 必须是 auto、xml 或 native")
    if format_type == "native":
        return response_content, []
    parser = XMLToolParser()
    if format_type == "xml" or _FUNCTION_CALLS.search(response_content or ""):
        return parser.extract_thinking_and_calls(response_content)
    return response_content, []
