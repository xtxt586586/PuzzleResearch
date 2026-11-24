import unittest

from langchain_core.messages import AIMessage, HumanMessage

from agent.workflows.stuck_middleware import (
    STUCK_HINT,
    detect_stuck,
    inject_stuck_hint,
)


class StuckMiddlewareTests(unittest.TestCase):
    def test_detects_consecutive_duplicate_assistant_messages(self):
        messages = [AIMessage(content="same"), AIMessage(content="same")]

        self.assertTrue(detect_stuck(messages, threshold=1))
        self.assertFalse(detect_stuck(messages, threshold=2))

    def test_does_not_detect_non_consecutive_or_empty_assistant_messages(self):
        self.assertFalse(
            detect_stuck(
                [
                    AIMessage(content="same"),
                    HumanMessage(content="new turn"),
                    AIMessage(content="same"),
                ],
                threshold=1,
            )
        )
        self.assertFalse(
            detect_stuck(
                [AIMessage(content=""), AIMessage(content="")],
                threshold=1,
            )
        )

    def test_hint_injection_returns_a_new_list(self):
        messages = [AIMessage(content="same")]

        hinted = inject_stuck_hint(messages)

        self.assertEqual(len(messages), 1)
        self.assertEqual(len(hinted), 2)
        self.assertEqual(hinted[-1].content, STUCK_HINT)
