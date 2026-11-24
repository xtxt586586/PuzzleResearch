import unittest

from agent.parsers import XMLToolParser, extract_tool_calls_from_response


class XMLToolParserTests(unittest.TestCase):
    def test_parses_multiple_invocations_and_typed_parameters(self):
        parser = XMLToolParser()
        content = (
            "先检查文件。"
            "<function_calls>"
            '<invoke name="read_file">'
            '<parameter name="path">notes.txt</parameter>'
            "</invoke>"
            '<invoke name="update_record">'
            '<parameter name="enabled">true</parameter>'
            '<parameter name="count">3</parameter>'
            '<parameter name="metadata">{"source":"test"}</parameter>'
            "</invoke>"
            "</function_calls>"
        )

        thinking, calls = parser.extract_thinking_and_calls(content)

        self.assertEqual(thinking, "先检查文件。")
        self.assertEqual([call.function_name for call in calls], ["read_file", "update_record"])
        self.assertEqual(calls[0].parameters, {"path": "notes.txt"})
        self.assertEqual(
            calls[1].parameters,
            {"enabled": True, "count": 3, "metadata": {"source": "test"}},
        )
        self.assertEqual(parser.get_statistics(), {"parsed_calls": 2, "parse_errors": 0})

    def test_extract_auto_returns_no_calls_for_plain_text(self):
        text, calls = extract_tool_calls_from_response("普通回答")

        self.assertEqual(text, "普通回答")
        self.assertEqual(calls, [])

    def test_native_format_does_not_parse_xml(self):
        content = '<function_calls><invoke name="echo"/></function_calls>'

        text, calls = extract_tool_calls_from_response(content, "native")

        self.assertEqual(text, content)
        self.assertEqual(calls, [])

    def test_ignores_malformed_xml_and_counts_parse_error(self):
        parser = XMLToolParser()

        calls = parser.parse_content(
            '<function_calls><invoke name="echo"></function_calls>'
        )

        self.assertEqual(calls, [])
        self.assertEqual(parser.get_statistics()["parse_errors"], 1)

    def test_recovers_complete_invoke_with_unescaped_ampersand(self):
        parser = XMLToolParser()
        content = (
            "<function_calls>"
            "<invoke name='echo'>"
            "<parameter name='text'>Tom & Jerry</parameter>"
            "</invoke>"
            "</function_calls>"
        )

        calls = parser.parse_content(content)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].function_name, "echo")
        self.assertEqual(calls[0].parameters, {"text": "Tom & Jerry"})
        self.assertEqual(parser.get_statistics(), {"parsed_calls": 1, "parse_errors": 1})

    def test_recovers_child_function_name_and_cdata_parameter(self):
        parser = XMLToolParser()
        content = (
            "<function_calls>"
            "<invoke><name>echo</name>"
            "<parameter name='text'><![CDATA[<hello>]]></parameter>"
            "</invoke>"
            "</function_calls>"
        )

        calls = parser.parse_content(content)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].function_name, "echo")
        self.assertEqual(calls[0].parameters, {"text": "<hello>"})

    def test_does_not_recover_unclosed_invoke(self):
        parser = XMLToolParser()

        calls = parser.parse_content(
            '<function_calls><invoke name="echo">'
            '<parameter name="text">incomplete</parameter>'
            "</function_calls>"
        )

        self.assertEqual(calls, [])
        self.assertEqual(parser.get_statistics()["parsed_calls"], 0)

    def test_rejects_entire_malformed_batch_if_a_later_call_is_incomplete(self):
        parser = XMLToolParser()
        content = (
            "<function_calls>"
            '<invoke name="echo"><parameter name="text">first</parameter></invoke>'
            '<invoke name="echo"><parameter name="text">second</parameter>'
            "</function_calls>"
        )

        calls = parser.parse_content(content)

        self.assertEqual(calls, [])
        self.assertEqual(parser.get_statistics()["parsed_calls"], 0)

    def test_recovery_preserves_markup_inside_cdata_when_outer_xml_is_malformed(self):
        parser = XMLToolParser()
        content = (
            "<function_calls>"
            "<invoke><name>echo</name>"
            "<parameter name='text'><![CDATA[<hello>]]></parameter>"
            "<parameter name='note'>Tom & Jerry</parameter>"
            "</invoke>"
            "</function_calls>"
        )

        calls = parser.parse_content(content)

        self.assertEqual(len(calls), 1)
        self.assertEqual(
            calls[0].parameters,
            {"text": "<hello>", "note": "Tom & Jerry"},
        )

    def test_openai_format_serializes_arguments(self):
        call = XMLToolParser().parse_content(
            '<function_calls><invoke name="echo">'
            '<parameter name="text">你好</parameter>'
            "</invoke></function_calls>"
        )[0]

        converted = call.to_openai_format()

        self.assertEqual(converted["type"], "function")
        self.assertEqual(converted["function"]["name"], "echo")
        self.assertEqual(converted["function"]["arguments"], '{"text": "你好"}')


if __name__ == "__main__":
    unittest.main()
