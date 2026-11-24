import unittest

from tools.python_executor import PythonExecutor


class PythonExecutorTests(unittest.TestCase):
    def setUp(self):
        self.executor = PythonExecutor()

    def test_execute_python_code(self):
        result = self.executor.execute(
            "print(2 + 3)",
            timeout_seconds=2,
        )

        self.assertEqual(result.output.strip(), "5")
        self.assertEqual(result.error, "")
        self.assertEqual(result.return_code, 0)
        self.assertFalse(result.timed_out)

    def test_isolated_runtime_supports_langchain_agents(self):
        result = self.executor.execute(
            "from langchain.agents import create_agent\n"
            "print(create_agent.__name__)",
            timeout_seconds=10,
        )

        self.assertEqual(result.output.strip(), "create_agent")
        self.assertEqual(result.error, "")
        self.assertEqual(result.return_code, 0)

    def test_runtime_error_is_returned(self):
        result = self.executor.execute(
            "raise ValueError('test error')",
            timeout_seconds=2,
        )

        self.assertNotEqual(result.return_code, 0)
        self.assertIn("ValueError", result.error)
        self.assertFalse(result.timed_out)

    def test_timeout_is_handled(self):
        result = self.executor.execute(
            "while True:\n    pass",
            timeout_seconds=0.2,
        )

        self.assertTrue(result.timed_out)
        self.assertIsNone(result.return_code)
        self.assertIn("超时", result.error)

    def test_empty_code_is_rejected(self):
        with self.assertRaises(ValueError):
            self.executor.execute("")

    def test_output_is_limited(self):
        result = self.executor.execute(
            "print('x' * 1000)",
            timeout_seconds=2,
            max_output_chars=100,
        )

        self.assertLessEqual(
            len(result.output),
            120,
        )
        self.assertIn("输出已截断", result.output)


if __name__ == "__main__":
    unittest.main()
