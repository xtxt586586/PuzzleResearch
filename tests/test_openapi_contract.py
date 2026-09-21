import unittest

import main


class OpenApiContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = main.app.openapi()

    def test_research_and_event_paths_are_exposed(self):
        paths = self.spec["paths"]

        self.assertIn("/api/research", paths)
        self.assertIn("/api/research/sse", paths)
        self.assertIn("/api/events/{thread_id}", paths)
        self.assertIn(
            "/api/sessions/{thread_id}/evidence",
            paths,
        )

    def test_research_request_schema_contains_contract_fields(self):
        schemas = self.spec["components"]["schemas"]
        research_schema = schemas["ResearchRequest"]

        self.assertEqual(
            set(research_schema["required"]),
            {"query"},
        )
        self.assertEqual(
            set(research_schema["properties"]),
            {
                "query",
                "model",
                "search_mode",
                "agent_id",
                "user_id",
                "images",
            },
        )

    def test_evidence_schema_contains_all_top_level_sections(self):
        schemas = self.spec["components"]["schemas"]
        evidence_schema = schemas["EvidenceResponse"]
        properties = evidence_schema["properties"]

        self.assertEqual(
            set(properties),
            {
                "sources",
                "claims",
                "quality_summary",
                "fetched_pages",
                "passages",
            },
        )

        for field_name in (
            "sources",
            "claims",
            "fetched_pages",
            "passages",
        ):
            self.assertEqual(
                properties[field_name]["type"],
                "array",
            )


if __name__ == "__main__":
    unittest.main()
