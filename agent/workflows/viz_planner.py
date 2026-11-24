from __future__ import annotations

import base64
import hashlib
import html
import json
import math
import re
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping
from urllib.parse import urlsplit


_VALUE_RE = re.compile(
    r"^\s*([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*(.*?)\s*$"
)
_VALID_UNIT_RE = re.compile(r"^[a-z0-9\u4e00-\u9fff%$\u00a3\u00a5/._-]*$")
_REPORT_MARKER = "<!-- PuzzleResearch generated visualizations -->"


def _parse_statistic_value(raw_value: Any) -> tuple[Decimal, str, str] | None:
    if isinstance(raw_value, bool) or raw_value is None:
        return None

    display = str(raw_value).strip()
    if not display or len(display) > 64:
        return None

    if isinstance(raw_value, (int, float, Decimal)):
        number_text = display
        unit = ""
    else:
        match = _VALUE_RE.fullmatch(display)
        if match is None:
            return None
        number_text = match.group(1).replace(",", "")
        unit = "".join(match.group(2).casefold().split())

    if len(unit) > 32 or not _VALID_UNIT_RE.fullmatch(unit):
        return None

    try:
        value = Decimal(number_text)
    except (InvalidOperation, ValueError):
        return None

    if not value.is_finite() or value < 0:
        return None
    try:
        if not math.isfinite(float(value)):
            return None
    except (OverflowError, ValueError):
        return None

    return value, unit, display


def _source_label(source: str) -> str:
    try:
        parsed = urlsplit(source)
        label = parsed.netloc or source
        if parsed.netloc:
            final_path_part = parsed.path.rstrip("/").rsplit("/", 1)[-1]
            if final_path_part:
                label = f"{label}/{final_path_part}"
    except ValueError:
        label = source
    if label.casefold().startswith("www."):
        label = label[4:]
    return label.strip()[:48] or "未知来源"


def _render_svg(
    *,
    title: str,
    unit: str,
    points: list[dict[str, Any]],
) -> str:
    width = 920
    row_height = 52
    height = 112 + row_height * len(points)
    label_x = 32
    bar_x = 300
    bar_width = 450
    largest = max((point["value"] for point in points), default=Decimal(0))
    safe_title = html.escape(title, quote=True)
    safe_unit = html.escape(unit or "未标注单位", quote=True)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{safe_title}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="32" y="42" fill="#202a34" font-family="Arial,sans-serif" '
        f'font-size="24" font-weight="700">{safe_title}</text>',
        f'<text x="32" y="72" fill="#64727e" font-family="Arial,sans-serif" '
        f'font-size="14">单位：{safe_unit}；按来源比较</text>',
        f'<line x1="{bar_x}" y1="88" x2="{bar_x}" y2="{height - 14}" '
        'stroke="#d7dee3" stroke-width="1"/>',
    ]

    for index, point in enumerate(points):
        y = 116 + index * row_height
        label = html.escape(point["label"], quote=True)
        source = html.escape(point["source"], quote=True)
        display_value = html.escape(point["display"], quote=True)
        ratio = float(point["value"] / largest) if largest else 0.0
        drawn_width = max(1, int(bar_width * ratio)) if point["value"] else 0
        parts.extend(
            [
                f'<text x="{label_x}" y="{y + 20}" fill="#35424c" '
                'font-family="Arial,sans-serif" font-size="14">'
                f'<title>{source}</title>{label}</text>',
                f'<rect x="{bar_x}" y="{y}" width="{drawn_width}" height="24" '
                'rx="3" fill="#237c70"/>',
                f'<text x="{bar_x + drawn_width + 12}" y="{y + 18}" fill="#202a34" '
                'font-family="Arial,sans-serif" font-size="14">'
                f'{display_value}</text>',
            ]
        )

    return "".join(parts) + "</svg>"


def build_report_charts(
    compressed_knowledge: Mapping[str, Any] | None,
    *,
    max_charts: int = 3,
    max_points_per_chart: int = 8,
) -> list[dict[str, str]]:
    """Build conservative source-comparison charts from structured statistics only."""
    if max_charts <= 0 or not isinstance(compressed_knowledge, Mapping):
        return []

    statistics = compressed_knowledge.get("statistics")
    if not isinstance(statistics, list):
        return []

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    metrics: dict[tuple[str, str], str] = {}
    for statistic in statistics[:200]:
        if not isinstance(statistic, Mapping):
            continue
        metric = " ".join(str(statistic.get("metric", "")).split())[:80]
        source = " ".join(str(statistic.get("source", "")).split())
        if len(source) > 1_000:
            continue
        parsed = _parse_statistic_value(statistic.get("value"))
        if not metric or not source or parsed is None:
            continue

        value, unit, display = parsed
        key = (metric.casefold(), unit)
        points = grouped.setdefault(key, [])
        source_key = source.casefold()
        if any(point["source"].casefold() == source_key for point in points):
            continue
        points.append(
            {
                "source": source,
                "label": _source_label(source),
                "value": value,
                "display": display,
            }
        )
        metrics[key] = metric

    chart_items: list[dict[str, str]] = []
    for key, points in grouped.items():
        if len(points) < 2:
            continue
        selected = points[:max(2, max_points_per_chart)]
        metric = metrics[key]
        unit = key[1]
        title = f"{metric} 来源比较"
        svg = _render_svg(title=title, unit=unit, points=selected)
        encoded_svg = base64.b64encode(svg.encode("utf-8")).decode("ascii")
        identity_payload = json.dumps(
            {
                "metric": key[0],
                "unit": unit,
                "points": [
                    [point["source"], point["display"]]
                    for point in selected
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        identity = hashlib.sha256(identity_payload.encode("utf-8")).hexdigest()[:20]
        source_list = "\n".join(f"- {point['source']}" for point in selected)
        chart_items.append(
            {
                "id": f"research-chart:{identity}",
                "type": "chart",
                "title": title,
                "url": f"data:image/svg+xml;base64,{encoded_svg}",
                "mime_type": "image/svg+xml",
                "content": (
                    f"指标：{metric}\n单位：{unit or '未标注'}\n数据来源：\n{source_list}"
                ),
            }
        )
        if len(chart_items) >= max(0, max_charts):
            break

    return chart_items


def embed_charts_in_report(
    report: str,
    charts: list[Mapping[str, Any]],
) -> str:
    """Embed generated chart images in Markdown and replace any previous chart section."""
    clean_report = str(report or "").split(_REPORT_MARKER, 1)[0].rstrip()
    valid_charts = [
        chart
        for chart in charts
        if isinstance(chart, Mapping)
        and str(chart.get("url", "")).startswith("data:image/svg+xml;base64,")
    ]
    if not valid_charts:
        return clean_report

    sections = [clean_report, _REPORT_MARKER, "## 数据可视化"]
    for chart in valid_charts:
        title = str(chart.get("title", "研究图表"))
        markdown_title = re.sub(
            r"([\\`*_{}\[\]<>()#+\-.!|])",
            r"\\\1",
            title,
        )
        content = str(chart.get("content", ""))
        source_names = [
            line[2:].strip()
            for line in content.splitlines()
            if line.startswith("- ") and line[2:].strip()
        ]
        sections.extend(
            [
                f"### {markdown_title}",
                f"![{markdown_title}]({chart['url']})",
                "> 数据按来源比较；来源信息见图表制品。",
            ]
        )
        if source_names:
            escaped_sources = [
                re.sub(
                    r"([\\`*_{}\[\]<>()#+\-.!|])",
                    r"\\\1",
                    source,
                )
                for source in source_names
            ]
            sections[-1] = "> 数据来源：" + "、".join(escaped_sources)

    return "\n\n".join(part for part in sections if part).strip()


def finalize_report_visualizations(
    report: str,
    compressed_knowledge: Mapping[str, Any] | None,
) -> tuple[str, list[dict[str, str]]]:
    charts = build_report_charts(compressed_knowledge)
    return embed_charts_in_report(report, charts), charts
