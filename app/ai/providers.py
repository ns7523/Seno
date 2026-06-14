import json
from abc import ABC, abstractmethod
from typing import Any

from openai import AsyncOpenAI

from app.config import Settings


class BaseLLMProvider(ABC):
    """Provider boundary for OpenAI-compatible chat JSON generation.

    Groq is the default production provider. The OpenAI adapter remains optional
    and is only constructed when explicitly selected and configured.
    """

    @abstractmethod
    async def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        raise NotImplementedError


class OpenAIProvider(BaseLLMProvider):
    def __init__(self, *, api_key: str, model: str, timeout: float = 30, base_url: str | None = None) -> None:
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key, timeout=timeout, base_url=base_url)

    async def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content or "{}")


class GroqProvider(OpenAIProvider):
    def __init__(self, *, api_key: str, model: str, timeout: float = 30) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
            timeout=timeout,
            base_url="https://api.groq.com/openai/v1",
        )


def build_llm_provider(settings: Settings) -> BaseLLMProvider | None:
    if settings.llm_provider == "groq":
        if not settings.groq_api_key:
            return None
        return GroqProvider(api_key=settings.groq_api_key, model=settings.groq_model, timeout=settings.openai_timeout_seconds)
    if not settings.openai_api_key:
        return None
    return OpenAIProvider(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        timeout=settings.openai_timeout_seconds,
        base_url=settings.openai_base_url,
    )
