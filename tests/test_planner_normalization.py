import unittest

from agent.workflows.nodes import _parse_plan_response


class PlannerNormalizationTests(unittest.TestCase):
    def test_model_plan_collapses_whitespace_and_casefolds_duplicates(self):
        response = type(
            "Response",
            (),
            {
                "content": (
                    '{"queries": ["  Official   Docs  ", '
                    '"official docs", "' + ("A" * 600) + '"]}'
                )
            },
        )()

        queries, _ = _parse_plan_response(response)

        self.assertEqual(queries, ["Official Docs", "A" * 500])


if __name__ == "__main__":
    unittest.main()
