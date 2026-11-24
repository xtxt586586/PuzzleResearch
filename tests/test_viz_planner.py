from __future__ import annotations

import base64
import unittest

from agent.workflows.viz_planner import (
    build_report_charts,
    embed_charts_in_report,
    finalize_report_visualizations,
)


class ReportVisualizationTests(unittest.TestCase):
    def test_builds_chart_only_for_same_metric_and_unit_from_distinct_sources(self):
        charts = build_report_charts(
            {
                "statistics": [
                    {
                        "metric": "转化率",
                        "value": "12.5%",
                        "source": "https://one.example/report",
                    },
                    {
                        "metric": "转化率",
                        "value": "14%",
                        "source": "https://two.example/report",
                    },
                    {
                        "metric": "收入",
                        "value": "14%",
                        "source": "https://three.example/report",
                    },
                    {
                        "metric": "转化率",
                        "value": "125",
                        "source": "https://four.example/report",
                    },
                ]
            }
        )

        self.assertEqual(len(charts), 1)
        self.assertEqual(charts[0]["type"], "chart")
        self.assertEqual(charts[0]["mime_type"], "image/svg+xml")
        encoded_svg = charts[0]["url"].split(",", 1)[1]
        svg = base64.b64decode(encoded_svg).decode("utf-8")
        self.assertIn("one.example", svg)
        self.assertIn("two.example", svg)
        self.assertIn("12.5%", svg)

    def test_ignores_single_source_missing_source_and_invalid_values(self):
        charts = build_report_charts(
            {
                "statistics": [
                    {"metric": "指标", "value": "10%", "source": "https://one.example"},
                    {"metric": "指标", "value": "11%", "source": "https://one.example"},
                    {"metric": "指标", "value": "12%"},
                    {"metric": "指标", "value": "不确定", "source": "https://two.example"},
                ]
            }
        )

        self.assertEqual(charts, [])

    def test_embeds_charts_and_replaces_previous_generated_section(self):
        chart = {
            "title": "转化率 来源比较",
            "url": "data:image/svg+xml;base64,PHN2Zz48L3N2Zz4=",
            "content": "指标：转化率\n数据来源：\n- one.example\n- two.example",
        }

        first = embed_charts_in_report("报告正文", [chart])
        second = embed_charts_in_report(first, [chart])

        self.assertEqual(second.count("## 数据可视化"), 1)
        self.assertEqual(second.count("data:image/svg+xml;base64,"), 1)
        self.assertTrue(second.startswith("报告正文"))

    def test_no_chartable_statistics_leave_report_unchanged(self):
        self.assertEqual(embed_charts_in_report("报告正文", []), "报告正文")

    def test_zero_chart_limit_disables_generation(self):
        charts = build_report_charts(
            {
                "statistics": [
                    {"metric": "指标", "value": "10%", "source": "https://one.example"},
                    {"metric": "指标", "value": "12%", "source": "https://two.example"},
                ]
            },
            max_charts=0,
        )

        self.assertEqual(charts, [])

    def test_escapes_untrusted_source_text_in_report_markdown(self):
        report = embed_charts_in_report(
            "报告正文",
            [
                {
                    "title": "图表",
                    "url": "data:image/svg+xml;base64,PHN2Zz48L3N2Zz4=",
                    "content": "数据来源：\n- evil](javascript:alert(1))",
                }
            ],
        )

        self.assertIn(r"evil\]\(javascript:alert\(1\)\)", report)

    def test_finalizes_report_and_artifact_from_structured_evidence(self):
        report, charts = finalize_report_visualizations(
            "研究结论。",
            {
                "statistics": [
                    {"metric": "准确率", "value": "91%", "source": "https://one.example/a"},
                    {"metric": "准确率", "value": "94%", "source": "https://two.example/b"},
                ]
            },
        )

        self.assertEqual(len(charts), 1)
        self.assertIn("## 数据可视化", report)
        self.assertIn(charts[0]["url"], report)


if __name__ == "__main__":
    unittest.main()
