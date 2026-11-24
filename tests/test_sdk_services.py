import unittest
from urllib.parse import urlsplit

import httpx

from sdk.python.puzzleresearch_sdk import (
    PuzzleResearchClient,
)


class PuzzleResearchServiceSdkTests(
    unittest.TestCase
):
    def test_metrics_preserves_prometheus_text(self):
        def handler(request):
            self.assertEqual(request.method, "GET")
            self.assertEqual(request.url.path, "/metrics")
            return httpx.Response(
                200,
                text="# HELP puzzleresearch_requests_total\n",
                headers={"content-type": "text/plain; version=0.0.4"},
                request=request,
            )

        http = httpx.Client(
            transport=httpx.MockTransport(handler)
        )
        client = PuzzleResearchClient(
            base_url="http://example.test",
            http=http,
        )

        self.assertEqual(
            client.get_metrics(),
            "# HELP puzzleresearch_requests_total\n",
        )
        client.close()

    def test_audio_support_cache_and_trigger_routes(
        self,
    ):
        calls = []

        def handler(request):
            parsed = urlsplit(str(request.url))

            calls.append(
                (
                    request.method,
                    parsed.path,
                    request.url.query.decode(),
                )
            )

            if parsed.path == "/api/asr/upload":
                self.assertIn(
                    "multipart/form-data",
                    request.headers["content-type"],
                )

            return httpx.Response(
                200,
                json={"ok": True},
                request=request,
            )

        http = httpx.Client(
            transport=httpx.MockTransport(handler)
        )

        client = PuzzleResearchClient(
            base_url="http://example.test",
            http=http,
        )

        client.recognize_speech(
            {
                "audio_data": "YXVkaW8=",
                "format": "wav",
                "sample_rate": 16000,
            }
        )

        client.recognize_speech_upload(
            "audio.wav",
            b"audio",
            sample_rate=16000,
        )

        client.get_asr_status()

        client.synthesize_speech(
            {
                "text": "hello",
                "voice": "default",
            }
        )

        client.get_tts_voices()
        client.get_tts_status()

        client.support_chat(
            {
                "message": "How does research work?",
                "user_id": "user-001",
            }
        )

        client.get_memory_status()
        client.get_search_cache_stats()
        client.clear_search_cache()

        client.create_scheduled_trigger(
            {
                "name": "Daily research",
                "schedule": "0 9 * * *",
                "task": "Run research",
            }
        )

        client.create_webhook_trigger(
            {
                "name": "Webhook research",
                "task": "Run webhook",
            }
        )

        client.create_event_trigger(
            {
                "name": "Completion handler",
                "event_type": "research.completed",
                "task": "Process completion",
            }
        )

        client.list_triggers(
            trigger_type="event",
            status="active",
        )

        client.get_trigger("trigger/one")
        client.pause_trigger("trigger/one")
        client.resume_trigger("trigger/one")

        client.get_trigger_executions(
            "trigger/one",
            limit=20,
        )

        client.invoke_webhook(
            "trigger/one",
            {
                "value": 1,
            },
        )

        client.delete_trigger("trigger/one")

        self.assertEqual(
            calls,
            [
                (
                    "POST",
                    "/api/asr/recognize",
                    "",
                ),
                (
                    "POST",
                    "/api/asr/upload",
                    "sample_rate=16000",
                ),
                (
                    "GET",
                    "/api/asr/status",
                    "",
                ),
                (
                    "POST",
                    "/api/tts/synthesize",
                    "",
                ),
                (
                    "GET",
                    "/api/tts/voices",
                    "",
                ),
                (
                    "GET",
                    "/api/tts/status",
                    "",
                ),
                (
                    "POST",
                    "/api/support/chat",
                    "",
                ),
                (
                    "GET",
                    "/api/memory/status",
                    "",
                ),
                (
                    "GET",
                    "/api/search/cache/stats",
                    "",
                ),
                (
                    "POST",
                    "/api/search/cache/clear",
                    "",
                ),
                (
                    "POST",
                    "/api/triggers/scheduled",
                    "",
                ),
                (
                    "POST",
                    "/api/triggers/webhook",
                    "",
                ),
                (
                    "POST",
                    "/api/triggers/event",
                    "",
                ),
                (
                    "GET",
                    "/api/triggers",
                    "trigger_type=event&status=active",
                ),
                (
                    "GET",
                    "/api/triggers/trigger%2Fone",
                    "",
                ),
                (
                    "POST",
                    "/api/triggers/trigger%2Fone/pause",
                    "",
                ),
                (
                    "POST",
                    "/api/triggers/trigger%2Fone/resume",
                    "",
                ),
                (
                    "GET",
                    "/api/triggers/trigger%2Fone/executions",
                    "limit=20",
                ),
                (
                    "POST",
                    "/api/webhook/trigger%2Fone",
                    "",
                ),
                (
                    "DELETE",
                    "/api/triggers/trigger%2Fone",
                    "",
                ),
            ],
        )

        client.close()


if __name__ == "__main__":
    unittest.main()
