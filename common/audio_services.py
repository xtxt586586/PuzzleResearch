from __future__ import annotations

import base64
import importlib.util
import os
import tempfile
import time
from functools import lru_cache
from typing import Any


VALID_AUDIO_FORMATS = {
    "wav",
    "mp3",
    "m4a",
    "flac",
    "ogg",
    "webm",
    "pcm",
}

AVAILABLE_VOICES = {
    "longanyang": "男声，普通话",
    "longxiaochun": "女声，普通话",
    "longwan": "女声，温和",
    "longyue": "女声，活泼",
    "longfei": "男声，磁性",
    "longjielidou": "男童声",
    "longshuo": "男声，深沉",
    "longshu": "女声，知性",
    "loongstella": "女声，英语",
    "loongbella": "女声，英语",
}

DEFAULT_VOICE = "longxiaochun"


def _sdk_available() -> bool:
    return importlib.util.find_spec("dashscope") is not None


class ASRService:
    def __init__(
        self,
        api_key: str | None = None,
    ) -> None:
        self.api_key = (
            os.getenv("DASHSCOPE_API_KEY", "")
            if api_key is None
            else api_key
        ).strip()

        self.enabled = bool(
            self.api_key
            and _sdk_available()
        )

        self.last_error: str | None = None
        self.last_error_time: float | None = None

    def recognize_bytes(
        self,
        audio_data: bytes,
        *,
        format: str = "wav",
        sample_rate: int = 16000,
        language_hints: list[str] | None = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {
                "success": False,
                "text": "",
                "error": "ASR service not available.",
            }

        temporary_path = ""

        try:
            import dashscope

            from dashscope.audio.asr import (
                Recognition,
                RecognitionCallback,
            )

            dashscope.api_key = self.api_key

            class Callback(RecognitionCallback):
                def on_open(self):
                    return None

                def on_complete(self):
                    return None

                def on_error(self, result):
                    return None

                def on_close(self):
                    return None

                def on_event(self, result):
                    return None

            with tempfile.NamedTemporaryFile(
                suffix=f".{format}",
                delete=False,
            ) as handle:
                handle.write(audio_data)
                temporary_path = handle.name

            recognition = Recognition(
                model="fun-asr-realtime-2025-11-07",
                callback=Callback(),
                format=format,
                sample_rate=sample_rate,
            )

            result = recognition.call(
                temporary_path
            )

            status_code = int(
                getattr(
                    result,
                    "status_code",
                    0,
                )
                or 0
            )

            if status_code != 200:
                error = str(
                    getattr(
                        result,
                        "message",
                        "ASR request failed.",
                    )
                )
                self.last_error = error
                self.last_error_time = time.time()

                return {
                    "success": False,
                    "text": "",
                    "error": error,
                }

            sentence = result.get_sentence()
            text = self._extract_text(sentence)

            self.last_error = None
            self.last_error_time = None

            return {
                "success": True,
                "text": text,
                "metrics": {
                    "request_id": str(
                        recognition.get_last_request_id()
                        or ""
                    ),
                },
            }

        except Exception as exc:
            self.last_error = str(exc)
            self.last_error_time = time.time()

            return {
                "success": False,
                "text": "",
                "error": str(exc),
            }

        finally:
            if temporary_path:
                try:
                    os.unlink(temporary_path)
                except OSError:
                    pass

    @staticmethod
    def _extract_text(value: Any) -> str:
        if value is None:
            return ""

        if isinstance(value, list):
            parts = []

            for item in value:
                if isinstance(item, dict):
                    parts.append(
                        str(
                            item.get("text")
                            or item.get("sentence")
                            or ""
                        )
                    )
                else:
                    parts.append(str(item))

            return " ".join(
                part for part in parts if part
            )

        if isinstance(value, dict):
            return str(
                value.get("text")
                or value.get("sentence")
                or ""
            )

        return str(value)


class TTSService:
    def __init__(
        self,
        api_key: str | None = None,
    ) -> None:
        self.api_key = (
            os.getenv("DASHSCOPE_API_KEY", "")
            if api_key is None
            else api_key
        ).strip()

        self.enabled = bool(
            self.api_key
            and _sdk_available()
        )

        self.last_error: str | None = None
        self.last_error_time: float | None = None

    def synthesize(
        self,
        text: str,
        *,
        voice: str = DEFAULT_VOICE,
    ) -> dict[str, Any]:
        normalized_text = str(text or "").strip()

        if not normalized_text:
            raise ValueError(
                "text 不能为空。"
            )

        if not self.enabled:
            return {
                "success": False,
                "error": "TTS service not available.",
            }

        try:
            import dashscope

            from dashscope.audio.tts_v2 import (
                AudioFormat,
                SpeechSynthesizer,
            )

            dashscope.api_key = self.api_key

            synthesizer = SpeechSynthesizer(
                model="cosyvoice-v1",
                voice=voice,
                format=AudioFormat.MP3_22050HZ_MONO_256KBPS,
            )

            audio_data = synthesizer.call(
                normalized_text[:2000]
            )

            if not audio_data:
                raise RuntimeError(
                    "TTS returned empty audio."
                )

            self.last_error = None
            self.last_error_time = None

            return {
                "success": True,
                "audio": base64.b64encode(
                    audio_data
                ).decode("utf-8"),
                "format": "mp3",
                "voice": voice,
            }

        except Exception as exc:
            self.last_error = str(exc)
            self.last_error_time = time.time()

            return {
                "success": False,
                "error": str(exc),
            }

    @staticmethod
    def voices() -> dict[str, str]:
        return dict(AVAILABLE_VOICES)


@lru_cache(maxsize=8)
def get_asr_service(
    api_key: str | None = None,
) -> ASRService:
    return ASRService(api_key)


@lru_cache(maxsize=8)
def get_tts_service(
    api_key: str | None = None,
) -> TTSService:
    return TTSService(api_key)