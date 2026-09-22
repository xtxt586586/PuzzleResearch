import base64
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

import main


class AudioApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(main.app)

    def test_asr_status_is_disabled_without_key(self):
        with patch.object(
            main.settings,
            "dashscope_api_key",
            "",
        ):
            response = self.client.get(
                "/api/asr/status"
            )

        self.assertEqual(
            response.status_code,
            200,
        )
        self.assertFalse(
            response.json()["enabled"]
        )
        self.assertFalse(
            response.json()["api_key_configured"]
        )

    def test_asr_returns_503_when_disabled(self):
        with patch.object(
            main.settings,
            "dashscope_api_key",
            "",
        ):
            response = self.client.post(
                "/api/asr/recognize",
                json={
                    "audio_data": "aGVsbG8=",
                },
            )

        self.assertEqual(
            response.status_code,
            503,
        )
        self.assertFalse(
            response.json()["success"]
        )

    def test_asr_validates_base64(self):
        service = MagicMock()
        service.enabled = True

        with patch.object(
            main,
            "get_asr_service",
            return_value=service,
        ):
            response = self.client.post(
                "/api/asr/recognize",
                json={
                    "audio_data": "invalid-base64",
                },
            )

        self.assertEqual(
            response.status_code,
            400,
        )
        service.recognize_bytes.assert_not_called()

    def test_asr_uses_service(self):
        service = MagicMock()
        service.enabled = True
        service.recognize_bytes.return_value = {
            "success": True,
            "text": "测试文本",
            "metrics": {},
        }

        with patch.object(
            main,
            "get_asr_service",
            return_value=service,
        ):
            response = self.client.post(
                "/api/asr/recognize",
                json={
                    "audio_data": base64.b64encode(
                        b"audio"
                    ).decode("ascii"),
                    "format": "wav",
                    "sample_rate": 16000,
                },
            )

        self.assertEqual(
            response.status_code,
            200,
        )
        self.assertEqual(
            response.json()["text"],
            "测试文本",
        )

    def test_tts_status_is_disabled_without_key(self):
        with patch.object(
            main.settings,
            "dashscope_api_key",
            "",
        ):
            response = self.client.get(
                "/api/tts/status"
            )

        self.assertEqual(
            response.status_code,
            200,
        )
        self.assertFalse(
            response.json()["enabled"]
        )

    def test_tts_voices_returns_default(self):
        response = self.client.get(
            "/api/tts/voices"
        )

        self.assertEqual(
            response.status_code,
            200,
        )
        self.assertEqual(
            response.json()["default"],
            "longxiaochun",
        )
        self.assertIn(
            "longxiaochun",
            response.json()["voices"],
        )

    def test_tts_returns_503_when_disabled(self):
        with patch.object(
            main.settings,
            "dashscope_api_key",
            "",
        ):
            response = self.client.post(
                "/api/tts/synthesize",
                json={
                    "text": "测试语音",
                },
            )

        self.assertEqual(
            response.status_code,
            503,
        )


if __name__ == "__main__":
    unittest.main()