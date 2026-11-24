from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openpyxl import load_workbook

from tools.builtin import create_default_registry
from tools.registry import ToolRegistry
from tools.workspace_files import SessionWorkspace, WorkspaceFileError
from tools.workspace_sheets import (
    WorkspaceSpreadsheetError,
    add_sheet,
    build_workspace_sheet_tool_definitions,
    create_chart,
    create_spreadsheet,
    format_cells,
    read_data,
    write_data,
)


class WorkspaceSheetTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_create_write_read_and_format_xlsx(self):
        with patch(
            "tools.workspace_sheets.get_session_workspace",
            return_value=SessionWorkspace("sheet-thread", storage_root=self.root),
        ):
            created = create_spreadsheet(
                "sheet-thread",
                "reports/research.xlsx",
                headers=["研究主题", "来源数量", "可信度"],
            )
            self.assertEqual(created["type"], "xlsx")
            written = write_data(
                "sheet-thread",
                "reports/research.xlsx",
                [["模型评测", 12, 0.92], ["数据治理", 8, 0.87]],
                start_row=2,
            )
            self.assertEqual(written["rows_written"], 2)
            result = read_data("sheet-thread", "reports/research.xlsx")
            self.assertEqual(
                result["data"],
                [
                    ["研究主题", "来源数量", "可信度"],
                    ["模型评测", 12, 0.92],
                    ["数据治理", 8, 0.87],
                ],
            )
            formatted = format_cells(
                "sheet-thread",
                "reports/research.xlsx",
                1,
                1,
                1,
                3,
                bold=True,
                bg_color="#D9EAF7",
                border=True,
            )
            self.assertTrue(formatted["success"])

        target = next((self.root / "").rglob("research.xlsx"))
        workbook = load_workbook(target)
        self.assertTrue(workbook.active["A1"].font.bold)
        self.assertEqual(workbook.active["A1"].fill.fgColor.rgb, "00D9EAF7")

    def test_csv_round_trip_and_offset_write(self):
        with patch(
            "tools.workspace_sheets.get_session_workspace",
            return_value=SessionWorkspace("csv-thread", storage_root=self.root),
        ):
            create_spreadsheet("csv-thread", "data.csv", headers=["名称", "数量"])
            write_data("csv-thread", "data.csv", [["样本", 3]], start_row=2, start_col=1)
            result = read_data("csv-thread", "data.csv")
        self.assertEqual(result["data"], [["名称", "数量"], ["样本", "3"]])

    def test_add_sheet_and_chart(self):
        with patch(
            "tools.workspace_sheets.get_session_workspace",
            return_value=SessionWorkspace("chart-thread", storage_root=self.root),
        ):
            create_spreadsheet(
                "chart-thread",
                "chart.xlsx",
                headers=["月份", "访问量"],
            )
            write_data(
                "chart-thread",
                "chart.xlsx",
                [["一月", 10], ["二月", 14]],
                start_row=2,
            )
            added = add_sheet("chart-thread", "chart.xlsx", "备注")
            chart = create_chart(
                "chart-thread",
                "chart.xlsx",
                "bar",
                "A1:B3",
                title="访问量",
                position="D2",
            )
        self.assertEqual(added["sheets"], ["Sheet1", "备注"])
        self.assertEqual(chart["chart_type"], "bar")
        target = next((self.root / "").rglob("chart.xlsx"))
        workbook = load_workbook(target)
        self.assertEqual(len(workbook["Sheet1"]._charts), 1)

    def test_paths_and_extensions_are_rejected(self):
        with self.assertRaises(WorkspaceFileError):
            create_spreadsheet("sheet-thread", "../outside.xlsx")
        with self.assertRaises(WorkspaceSpreadsheetError):
            create_spreadsheet("sheet-thread", "notes.txt")
        with self.assertRaises(WorkspaceFileError):
            read_data("sheet-thread", "missing.xlsx")

    def test_registry_exposes_and_invokes_sheet_tools(self):
        names = {tool.name for tool in create_default_registry().list_tools()}
        self.assertTrue(
            {
                "create_spreadsheet",
                "write_data",
                "read_data",
                "format_cells",
                "add_sheet",
                "create_chart",
            } <= names
        )
        workspace = SessionWorkspace("registry-thread", storage_root=self.root)
        with patch(
            "tools.workspace_sheets.get_session_workspace",
            return_value=workspace,
        ):
            registry = ToolRegistry(build_workspace_sheet_tool_definitions("registry-thread"))
            registry.invoke(
                "create_spreadsheet",
                {"file_path": "registry.xlsx", "headers": ["A"]},
            )
            result = registry.invoke(
                "write_data",
                {"file_path": "registry.xlsx", "data": [[1]], "start_row": 2},
            )
        self.assertTrue(result["success"])


if __name__ == "__main__":
    unittest.main()
