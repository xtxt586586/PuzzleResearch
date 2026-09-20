import unittest

from agent.core.profile_adapter import (
    runtime_profile_from_record,
)
from common.agent_store import AgentRecord


class ProfileAdapterTests(unittest.TestCase):
    def test_empty_tool_map_keeps_default_tool_behavior(self):
        record = AgentRecord(
            id="default",
            name="Default Agent",
        )

        profile = runtime_profile_from_record(
            record
        )

        self.assertEqual(
            profile.name,
            "Default Agent",
        )
        self.assertIsNone(
            profile.enabled_tools
        )
        self.assertIsNone(
            profile.blocked_tools
        )

    def test_tool_flags_map_to_enabled_and_blocked_sets(self):
        record = AgentRecord(
            id="restricted-agent",
            name="Restricted Agent",
            enabled_tools={
                "python_executor": True,
                "echo": False,
            },
        )

        profile = runtime_profile_from_record(
            record
        )

        self.assertEqual(
            profile.enabled_tools,
            frozenset({"python_executor"}),
        )
        self.assertEqual(
            profile.blocked_tools,
            frozenset({"echo"}),
        )

    def test_runtime_policy_is_mapped_from_metadata(self):
        record = AgentRecord(
            id="policy-agent",
            name="Policy Agent",
            metadata={
                "runtime_policy": {
                    "tool_call_limit": 2,
                    "require_tool_approval": True,
                    "tool_retry_limit": 1,
                    "deduplicate_tool_calls": True,
                    "cache_tool_results": True,
                    "max_tool_result_chars": 1000,
                }
            },
        )

        profile = runtime_profile_from_record(
            record
        )

        self.assertEqual(
            profile.tool_call_limit,
            2,
        )
        self.assertTrue(
            profile.require_tool_approval
        )
        self.assertEqual(
            profile.tool_retry_limit,
            1,
        )
        self.assertTrue(
            profile.deduplicate_tool_calls
        )
        self.assertTrue(
            profile.cache_tool_results
        )
        self.assertEqual(
            profile.max_tool_result_chars,
            1000,
        )

    def test_invalid_runtime_policy_is_rejected(self):
        record = AgentRecord(
            id="invalid-agent",
            name="Invalid Agent",
            metadata={
                "runtime_policy": "invalid"
            },
        )

        with self.assertRaises(ValueError):
            runtime_profile_from_record(record)

    def test_agent_fields_are_mapped(self):
        record = AgentRecord(
            id="custom-agent",
            name="Custom Agent",
            system_prompt="你是一个严谨的研究助手。",
            model="deepseek-reasoner",
        )

        profile = runtime_profile_from_record(record)

        self.assertEqual(
            profile.system_prompt,
            "你是一个严谨的研究助手。",
        )
        self.assertEqual(
            profile.model,
            "deepseek-reasoner",
        )


if __name__ == "__main__":
    unittest.main()