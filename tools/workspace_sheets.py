from __future__ import annotations

import csv
import io
import re
import tempfile
from copy import copy
from pathlib import Path
from typing import Any

from tools.registry import ToolDefinition
from tools.workspace_files import (
    MAX_FILE_BYTES,
    SessionWorkspace,
    WorkspaceFileConflictError,
    WorkspaceFileError,
    WorkspaceFileNotFoundError,
    get_session_workspace,
)


SUPPORTED_EXTENSIONS = {".csv", ".xlsx"}
MAX_ROWS = 5_000
MAX_COLUMNS = 200
MAX_READ_ROWS = 500
MAX_READ_COLUMNS = 200
_SHEET_NAME_RE = re.compile(r"^[^\\/*?:\[\]]{1,31}$")
_CELL_RANGE_RE = re.compile(r"^[A-Za-z]{1,3}[1-9][0-9]*:[A-Za-z]{1,3}[1-9][0-9]*$")
_COLOR_RE = re.compile(r"^(?:[0-9A-Fa-f]{6}|[0-9A-Fa-f]{8})$")


class WorkspaceSpreadsheetError(WorkspaceFileError):
    """工作区表格操作错误。"""


def _openpyxl() -> Any:
    try:
        import openpyxl
    except ImportError as exc:
        raise WorkspaceSpreadsheetError(
            "Excel 工具需要 openpyxl；请先安装项目 requirements.txt 中的依赖。"
        ) from exc
    return openpyxl


def _validate_table_path(path: str) -> tuple[str, str]:
    normalized = str(path or "").strip().replace("\\", "/")
    extension = Path(normalized).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise WorkspaceSpreadsheetError("表格文件只支持 .xlsx 或 .csv 格式。")
    return normalized, extension


def _validate_positive(value: int, name: str, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise WorkspaceSpreadsheetError(
            f"{name} 必须是 1 到 {maximum} 之间的整数。"
        )
    return value


def _validate_sheet_name(sheet_name: str) -> str:
    value = str(sheet_name or "").strip()
    if not _SHEET_NAME_RE.fullmatch(value) or value in {"", ".", ".."}:
        raise WorkspaceSpreadsheetError(
            "工作表名称不能为空、不能超过 31 个字符，且不能包含 \\/ * ? : [ ]。"
        )
    return value


def _validate_range(start_row: int, end_row: int, start_col: int, end_col: int) -> None:
    _validate_positive(start_row, "start_row", MAX_ROWS)
    _validate_positive(end_row, "end_row", MAX_ROWS)
    _validate_positive(start_col, "start_col", MAX_COLUMNS)
    _validate_positive(end_col, "end_col", MAX_COLUMNS)
    if end_row < start_row or end_col < start_col:
        raise WorkspaceSpreadsheetError("结束行列必须大于或等于起始行列。")


def _validate_color(value: str | None, name: str) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lstrip("#")
    if not _COLOR_RE.fullmatch(normalized):
        raise WorkspaceSpreadsheetError(f"{name} 必须是 6 位或 8 位十六进制颜色。")
    return normalized.upper()


def _validate_values(data: list[list[Any]]) -> list[list[Any]]:
    if not isinstance(data, list):
        raise WorkspaceSpreadsheetError("data 必须是二维数组。")
    if len(data) > MAX_ROWS:
        raise WorkspaceSpreadsheetError(f"数据行数不能超过 {MAX_ROWS}。")
    result: list[list[Any]] = []
    width = 0
    for row in data:
        if not isinstance(row, list):
            raise WorkspaceSpreadsheetError("data 中的每一行必须是数组。")
        if len(row) > MAX_COLUMNS:
            raise WorkspaceSpreadsheetError(f"数据列数不能超过 {MAX_COLUMNS}。")
        for value in row:
            if isinstance(value, (dict, list, tuple, set)):
                raise WorkspaceSpreadsheetError("单元格只支持文本、数字、布尔值或空值。")
        width = max(width, len(row))
        result.append(list(row))
    return result


def _save_workbook(workbook: Any, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".spreadsheet-",
        suffix=".xlsx",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        import os

        os.close(descriptor)
        workbook.save(temporary)
        if temporary.stat().st_size > MAX_FILE_BYTES:
            raise WorkspaceSpreadsheetError(
                f"表格文件不能超过 {MAX_FILE_BYTES} 字节。"
            )
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def _load_workbook(workspace: SessionWorkspace, path: str, *, data_only: bool = False) -> tuple[Path, Any]:
    normalized, extension = _validate_table_path(path)
    if extension != ".xlsx":
        raise WorkspaceSpreadsheetError("该操作只支持 .xlsx 文件。")
    target = workspace._target(normalized)
    if not target.is_file():
        raise WorkspaceFileNotFoundError("工作区表格文件不存在。")
    openpyxl = _openpyxl()
    try:
        return target, openpyxl.load_workbook(target, data_only=data_only)
    except Exception as exc:
        raise WorkspaceSpreadsheetError("Excel 文件无法读取或格式无效。") from exc


def _write_csv(target: Path, data: list[list[Any]], *, overwrite: bool) -> None:
    if target.exists() and not overwrite:
        raise WorkspaceFileConflictError("文件已存在；如需覆盖请设置 overwrite=true。")
    output = io.StringIO(newline="")
    csv.writer(output, lineterminator="\n").writerows(data)
    content = output.getvalue().encode("utf-8")
    if len(content) > MAX_FILE_BYTES:
        raise WorkspaceSpreadsheetError(f"表格文件不能超过 {MAX_FILE_BYTES} 字节。")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)


def create_spreadsheet(
    thread_id: str,
    file_path: str,
    sheet_name: str = "Sheet1",
    headers: list[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    workspace = get_session_workspace(thread_id)
    normalized, extension = _validate_table_path(file_path)
    if type(overwrite) is not bool:
        raise WorkspaceSpreadsheetError("overwrite 必须是布尔值。")
    values: list[list[Any]] = []
    if headers is not None:
        if not isinstance(headers, list) or any(not isinstance(item, str) for item in headers):
            raise WorkspaceSpreadsheetError("headers 必须是字符串数组。")
        _validate_values([headers])
        values.append(headers)

    target = workspace._target(normalized)
    if extension == ".csv":
        _write_csv(target, values, overwrite=overwrite)
        return {
            "success": True,
            "path": normalized,
            "type": "csv",
            "sheet_name": None,
            "rows_written": len(values),
            "output": f"已创建工作区表格：{normalized}",
        }

    selected_sheet = _validate_sheet_name(sheet_name)
    if target.exists() and not overwrite:
        raise WorkspaceFileConflictError("文件已存在；如需覆盖请设置 overwrite=true。")
    openpyxl = _openpyxl()
    workbook = openpyxl.Workbook()
    workbook.active.title = selected_sheet
    worksheet = workbook[selected_sheet]
    for column, value in enumerate(values[0] if values else [], start=1):
        cell = worksheet.cell(row=1, column=column, value=value)
        font = copy(cell.font)
        font.bold = True
        cell.font = font
    _save_workbook(workbook, target)
    return {
        "success": True,
        "path": normalized,
        "type": "xlsx",
        "sheet_name": selected_sheet,
        "sheets": workbook.sheetnames,
        "rows_written": len(values),
        "output": f"已创建工作区表格：{normalized}",
    }


def write_data(
    thread_id: str,
    file_path: str,
    data: list[list[Any]],
    start_row: int = 1,
    start_col: int = 1,
    sheet_name: str | None = None,
) -> dict[str, Any]:
    workspace = get_session_workspace(thread_id)
    normalized, extension = _validate_table_path(file_path)
    values = _validate_values(data)
    _validate_positive(start_row, "start_row", MAX_ROWS)
    _validate_positive(start_col, "start_col", MAX_COLUMNS)
    if start_row + len(values) - 1 > MAX_ROWS or start_col + max((len(row) for row in values), default=0) - 1 > MAX_COLUMNS:
        raise WorkspaceSpreadsheetError("写入范围超出表格工具允许的最大行列。")

    if extension == ".csv":
        target = workspace._target(normalized)
        if not target.is_file():
            raise WorkspaceFileNotFoundError("工作区表格文件不存在。")
        try:
            rows = list(csv.reader(io.StringIO(target.read_text(encoding="utf-8"))))
        except (OSError, UnicodeDecodeError) as exc:
            raise WorkspaceSpreadsheetError("CSV 文件无法读取。") from exc
        required_rows = start_row - 1 + len(values)
        while len(rows) < required_rows:
            rows.append([])
        for row_index, row_values in enumerate(values, start=start_row - 1):
            while len(rows[row_index]) < start_col - 1 + len(row_values):
                rows[row_index].append("")
            rows[row_index][start_col - 1 : start_col - 1 + len(row_values)] = [
                "" if value is None else value for value in row_values
            ]
        _write_csv(target, rows, overwrite=True)
        sheets = None
    else:
        target, workbook = _load_workbook(workspace, normalized)
        selected_sheet = _validate_sheet_name(sheet_name) if sheet_name else workbook.active.title
        if selected_sheet not in workbook.sheetnames:
            raise WorkspaceSpreadsheetError(f"工作表不存在：{selected_sheet}")
        worksheet = workbook[selected_sheet]
        for row_index, row_values in enumerate(values, start=start_row):
            for column_index, value in enumerate(row_values, start=start_col):
                worksheet.cell(row=row_index, column=column_index, value=value)
        _save_workbook(workbook, target)
        sheets = workbook.sheetnames

    return {
        "success": True,
        "path": normalized,
        "sheet_name": sheet_name,
        "rows_written": len(values),
        "start_row": start_row,
        "start_col": start_col,
        "sheets": sheets,
        "output": f"已写入工作区表格：{normalized}",
    }


def read_data(
    thread_id: str,
    file_path: str,
    sheet_name: str | None = None,
    max_rows: int = 100,
    start_row: int = 1,
    start_col: int = 1,
    max_columns: int = MAX_READ_COLUMNS,
) -> dict[str, Any]:
    workspace = get_session_workspace(thread_id)
    normalized, extension = _validate_table_path(file_path)
    _validate_positive(max_rows, "max_rows", MAX_READ_ROWS)
    _validate_positive(start_row, "start_row", MAX_ROWS)
    _validate_positive(start_col, "start_col", MAX_COLUMNS)
    _validate_positive(max_columns, "max_columns", MAX_READ_COLUMNS)

    if extension == ".csv":
        target = workspace._target(normalized)
        if not target.is_file():
            raise WorkspaceFileNotFoundError("工作区表格文件不存在。")
        try:
            rows = list(csv.reader(io.StringIO(target.read_text(encoding="utf-8"))))
        except (OSError, UnicodeDecodeError) as exc:
            raise WorkspaceSpreadsheetError("CSV 文件无法读取。") from exc
        data = [
            row[start_col - 1 : start_col - 1 + max_columns]
            for row in rows[start_row - 1 : start_row - 1 + max_rows]
        ]
        sheets = None
    else:
        _, workbook = _load_workbook(workspace, normalized, data_only=True)
        selected_sheet = _validate_sheet_name(sheet_name) if sheet_name else workbook.active.title
        if selected_sheet not in workbook.sheetnames:
            raise WorkspaceSpreadsheetError(f"工作表不存在：{selected_sheet}")
        worksheet = workbook[selected_sheet]
        if worksheet.max_row < start_row or worksheet.max_column < start_col:
            data = []
        else:
            last_row = min(worksheet.max_row, start_row + max_rows - 1)
            last_col = min(worksheet.max_column, start_col + max_columns - 1)
            data = [
                [cell.value for cell in row]
                for row in worksheet.iter_rows(
                    min_row=start_row,
                    max_row=last_row,
                    min_col=start_col,
                    max_col=last_col,
                )
            ]
        sheets = workbook.sheetnames
    return {
        "success": True,
        "path": normalized,
        "sheet_name": sheet_name,
        "data": data,
        "rows": len(data),
        "columns": max((len(row) for row in data), default=0),
        "sheets": sheets,
        "output": f"已读取工作区表格：{normalized}",
    }


def format_cells(
    thread_id: str,
    file_path: str,
    start_row: int,
    end_row: int,
    start_col: int,
    end_col: int,
    bold: bool = False,
    italic: bool = False,
    font_size: int | None = None,
    font_color: str | None = None,
    bg_color: str | None = None,
    border: bool = False,
    sheet_name: str | None = None,
) -> dict[str, Any]:
    workspace = get_session_workspace(thread_id)
    normalized, extension = _validate_table_path(file_path)
    if extension != ".xlsx":
        raise WorkspaceSpreadsheetError("单元格格式只支持 .xlsx 文件。")
    _validate_range(start_row, end_row, start_col, end_col)
    if type(bold) is not bool or type(italic) is not bool or type(border) is not bool:
        raise WorkspaceSpreadsheetError("bold、italic、border 必须是布尔值。")
    if font_size is not None:
        _validate_positive(font_size, "font_size", 200)
    font_color = _validate_color(font_color, "font_color")
    bg_color = _validate_color(bg_color, "bg_color")
    target, workbook = _load_workbook(workspace, normalized)
    selected_sheet = _validate_sheet_name(sheet_name) if sheet_name else workbook.active.title
    if selected_sheet not in workbook.sheetnames:
        raise WorkspaceSpreadsheetError(f"工作表不存在：{selected_sheet}")
    openpyxl = _openpyxl()
    worksheet = workbook[selected_sheet]
    for row in worksheet.iter_rows(
        min_row=start_row,
        max_row=end_row,
        min_col=start_col,
        max_col=end_col,
    ):
        for cell in row:
            font = copy(cell.font)
            if bold:
                font.bold = True
            if italic:
                font.italic = True
            if font_size is not None:
                font.sz = font_size
            if font_color is not None:
                font.color = font_color
            cell.font = font
            if bg_color is not None:
                cell.fill = openpyxl.styles.PatternFill(
                    fill_type="solid",
                    fgColor=bg_color,
                )
            if border:
                side = openpyxl.styles.Side(style="thin", color="808080")
                cell.border = openpyxl.styles.Border(
                    left=side,
                    right=side,
                    top=side,
                    bottom=side,
                )
    _save_workbook(workbook, target)
    return {
        "success": True,
        "path": normalized,
        "sheet_name": selected_sheet,
        "range": f"{start_row}:{end_row},{start_col}:{end_col}",
        "output": f"已设置工作区表格格式：{normalized}",
    }


def add_sheet(
    thread_id: str,
    file_path: str,
    sheet_name: str,
    position: int | None = None,
) -> dict[str, Any]:
    workspace = get_session_workspace(thread_id)
    normalized, extension = _validate_table_path(file_path)
    if extension != ".xlsx":
        raise WorkspaceSpreadsheetError("新增工作表只支持 .xlsx 文件。")
    selected_sheet = _validate_sheet_name(sheet_name)
    if position is not None and (type(position) is not int or position < 0):
        raise WorkspaceSpreadsheetError("position 必须是非负整数。")
    target, workbook = _load_workbook(workspace, normalized)
    if selected_sheet in workbook.sheetnames:
        raise WorkspaceSpreadsheetError(f"工作表已存在：{selected_sheet}")
    if position is not None and position > len(workbook.sheetnames):
        raise WorkspaceSpreadsheetError("position 超出工作表数量范围。")
    workbook.create_sheet(selected_sheet, index=position)
    _save_workbook(workbook, target)
    return {
        "success": True,
        "path": normalized,
        "sheet_name": selected_sheet,
        "sheets": workbook.sheetnames,
        "output": f"已新增工作表：{selected_sheet}",
    }


def create_chart(
    thread_id: str,
    file_path: str,
    chart_type: str,
    data_range: str,
    title: str = "",
    x_axis_title: str = "",
    y_axis_title: str = "",
    position: str = "E1",
    sheet_name: str | None = None,
) -> dict[str, Any]:
    workspace = get_session_workspace(thread_id)
    normalized, extension = _validate_table_path(file_path)
    if extension != ".xlsx":
        raise WorkspaceSpreadsheetError("图表只支持 .xlsx 文件。")
    selected_type = str(chart_type or "").strip().lower()
    if selected_type not in {"bar", "line", "pie", "scatter", "area"}:
        raise WorkspaceSpreadsheetError("chart_type 必须是 bar、line、pie、scatter 或 area。")
    if not _CELL_RANGE_RE.fullmatch(str(data_range or "")):
        raise WorkspaceSpreadsheetError("data_range 必须使用 A1:D10 格式。")
    if not re.fullmatch(r"^[A-Za-z]{1,3}[1-9][0-9]*$", str(position or "")):
        raise WorkspaceSpreadsheetError("position 必须是单元格位置，例如 E1。")
    target, workbook = _load_workbook(workspace, normalized)
    selected_sheet = _validate_sheet_name(sheet_name) if sheet_name else workbook.active.title
    if selected_sheet not in workbook.sheetnames:
        raise WorkspaceSpreadsheetError(f"工作表不存在：{selected_sheet}")
    openpyxl = _openpyxl()
    from openpyxl.chart import AreaChart, BarChart, LineChart, PieChart, ScatterChart
    from openpyxl.chart import Reference
    from openpyxl.utils.cell import range_boundaries

    min_col, min_row, max_col, max_row = range_boundaries(data_range)
    if max_col <= min_col or max_row <= min_row:
        raise WorkspaceSpreadsheetError("data_range 至少需要一列分类和一列数据。")
    worksheet = workbook[selected_sheet]
    chart_class = {
        "bar": BarChart,
        "line": LineChart,
        "pie": PieChart,
        "scatter": ScatterChart,
        "area": AreaChart,
    }[selected_type]
    chart = chart_class()
    chart.title = str(title or "")
    if hasattr(chart, "x_axis"):
        chart.x_axis.title = str(x_axis_title or "")
    if hasattr(chart, "y_axis"):
        chart.y_axis.title = str(y_axis_title or "")
    chart.add_data(
        Reference(worksheet, min_col=min_col + 1, min_row=min_row, max_col=max_col, max_row=max_row),
        titles_from_data=True,
    )
    chart.set_categories(
        Reference(worksheet, min_col=min_col, min_row=min_row + 1, max_row=max_row)
    )
    worksheet.add_chart(chart, position)
    _save_workbook(workbook, target)
    return {
        "success": True,
        "path": normalized,
        "sheet_name": selected_sheet,
        "chart_type": selected_type,
        "data_range": data_range,
        "position": position,
        "output": f"已创建工作区图表：{normalized}",
    }


def build_workspace_sheet_tool_definitions(thread_id: str) -> list[ToolDefinition]:
    def create_spreadsheet_tool(
        file_path: str,
        sheet_name: str = "Sheet1",
        headers: list[str] | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        return create_spreadsheet(thread_id, file_path, sheet_name, headers, overwrite)

    def write_data_tool(
        file_path: str,
        data: list[list[Any]],
        start_row: int = 1,
        start_col: int = 1,
        sheet_name: str | None = None,
    ) -> dict[str, Any]:
        return write_data(thread_id, file_path, data, start_row, start_col, sheet_name)

    def read_data_tool(
        file_path: str,
        sheet_name: str | None = None,
        max_rows: int = 100,
        start_row: int = 1,
        start_col: int = 1,
        max_columns: int = MAX_READ_COLUMNS,
    ) -> dict[str, Any]:
        return read_data(thread_id, file_path, sheet_name, max_rows, start_row, start_col, max_columns)

    def format_cells_tool(
        file_path: str,
        start_row: int,
        end_row: int,
        start_col: int,
        end_col: int,
        bold: bool = False,
        italic: bool = False,
        font_size: int | None = None,
        font_color: str | None = None,
        bg_color: str | None = None,
        border: bool = False,
        sheet_name: str | None = None,
    ) -> dict[str, Any]:
        return format_cells(
            thread_id, file_path, start_row, end_row, start_col, end_col,
            bold, italic, font_size, font_color, bg_color, border, sheet_name,
        )

    def add_sheet_tool(
        file_path: str,
        sheet_name: str,
        position: int | None = None,
    ) -> dict[str, Any]:
        return add_sheet(thread_id, file_path, sheet_name, position)

    def create_chart_tool(
        file_path: str,
        chart_type: str,
        data_range: str,
        title: str = "",
        x_axis_title: str = "",
        y_axis_title: str = "",
        position: str = "E1",
        sheet_name: str | None = None,
    ) -> dict[str, Any]:
        return create_chart(
            thread_id, file_path, chart_type, data_range, title,
            x_axis_title, y_axis_title, position, sheet_name,
        )

    return [
        ToolDefinition(
            "create_spreadsheet",
            "在当前会话工作区创建 Excel 或 CSV 表格，可选创建表头。",
            create_spreadsheet_tool,
        ),
        ToolDefinition(
            "write_data",
            "向当前会话工作区表格的指定起始单元格写入二维数据。",
            write_data_tool,
        ),
        ToolDefinition(
            "read_data",
            "读取当前会话工作区 Excel 或 CSV 表格中的二维数据。",
            read_data_tool,
        ),
        ToolDefinition(
            "format_cells",
            "设置当前会话工作区 Excel 表格区域的字体、颜色和边框。",
            format_cells_tool,
        ),
        ToolDefinition(
            "add_sheet",
            "向当前会话工作区 Excel 文件新增工作表。",
            add_sheet_tool,
        ),
        ToolDefinition(
            "create_chart",
            "根据当前会话工作区 Excel 数据区域创建柱状、折线、饼图、散点或面积图。",
            create_chart_tool,
        ),
    ]
