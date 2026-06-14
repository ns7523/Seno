from typing import Protocol


class VoiceTranscriber(Protocol):
    def transcribe(self, audio_bytes: bytes) -> str:
        ...


class StaticVoiceTranscriber:
    """Deterministic test/local transcriber.

    Production can replace this with an OpenAI/Groq-compatible speech adapter
    without changing Telegram webhook or email approval logic.
    """

    def __init__(self, text: str = "") -> None:
        self.text = text

    def transcribe(self, audio_bytes: bytes) -> str:
        return self.text
