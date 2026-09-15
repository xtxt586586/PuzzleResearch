from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class PythonExecutionResult:
    """
    Python 代码执行结果
    :return:
    """

    output: str
    error: str
    return_code: int | None
    timed_out: bool


class PythonExecutor:
    """
    使用独立 Python 子进程执行代码。

    注意：
    当前实现具备进程隔离和超时控制，
    但还不是严格安全沙箱。
    """

    def execute(
            self,
            code: str,
            *,
            timeout_seconds: float = 5.0,
            max_output_chars: int = 4_000,
    ) -> PythonExecutionResult:
        """
        执行一段 Python 代码。
        """

        code = code.strip()

        if not code:
            raise ValueError("Python 代码不能为空。")

        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds 必须大于 0")

        if max_output_chars <= 0:
            raise ValueError("max_output_chars 必须大于 0")

        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-c",
                    code,
                ],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
                check=False,
            )

        except subprocess.TimeoutExpired as exc:
            return PythonExecutionResult(
                output=self._clip_text(
                    self._to_text(exc.stdout),
                    max_output_chars,
                ),
                error=(
                    f"python 执行超时，限制为 "
                    f"{timeout_seconds} 秒。"
                    + self._clip_text(
                        self._to_text(exc.stderr),
                        max_output_chars,
                    )
                ),
                return_code=None,
                timed_out=True,
            )

        output = self._clip_text(
            completed.stdout,
            max_output_chars,
        )

        error = self._clip_text(
            completed.stderr,
            max_output_chars,
        )

        if completed.returncode != 0 and not error:
            error = (
                f"python 执行退出，返回码： "
                f"{completed.returncode}。"
            )

        return PythonExecutionResult(
            output=output,
            error=error,
            return_code=completed.returncode,
            timed_out=False,
        )


    @staticmethod
    def _to_text(value: str | bytes | None) -> str:
        """
        兼容 TimeoutExpired 返回的字符串或字节数据。
        """

        if value is None:
            return ""

        if isinstance(value,bytes):
            return value.decode(
                "utf-8",
                errors="replace",
            )

        return str(value)

    @staticmethod
    def _clip_text(
            value: str,
            max_output_chars: int,
    ) -> str:
        """
        限制输出长度，避免程序产生过大结果。
        """

        if len(value) <= max_output_chars:
            return value

        return (
            value[:max_output_chars]
            + "\n...[输出已截断]"
        )