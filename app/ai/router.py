from dataclasses import dataclass
from typing import Any

from app.ai.providers import BaseLLMProvider


@dataclass(slots=True)
class RoutingPolicy:
    summary_provider: str = "groq"
    casual_drafting_provider: str = "groq"
    legal_reasoning_provider: str = "groq"
    safety_provider: str = "local"
    embeddings_provider: str = "local"

    def provider_for_task(self, task: str) -> str | None:
        if task == "summary":
            return self.summary_provider
        if task in {"draft", "casual_draft"}:
            return self.casual_drafting_provider
        if task in {"reasoning", "legal_reasoning"}:
            return self.legal_reasoning_provider
        if task == "safety":
            return self.safety_provider
        if task in {"embedding", "embeddings"}:
            return self.embeddings_provider
        return None


@dataclass(slots=True)
class LLMTaskRouter:
    """Routes future AI tasks to task-specific providers.

    The default policy is Groq-first and local-first: LLM tasks use Groq, while
    safety and embeddings stay local unless explicitly configured otherwise.
    """

    default_provider: BaseLLMProvider | str | None = None
    summary_provider: BaseLLMProvider | str | None = None
    drafting_provider: BaseLLMProvider | str | None = None
    reasoning_provider: BaseLLMProvider | str | None = None
    safety_provider: BaseLLMProvider | str | None = None
    embeddings_provider: BaseLLMProvider | str | None = None
    policy: RoutingPolicy | None = None

    def for_task(self, task: str) -> Any | None:
        if task == "summary":
            return self.summary_provider or self.default_provider
        if task in {"draft", "casual_draft"}:
            return self.drafting_provider or self.default_provider
        if task in {"reasoning", "legal_reasoning"}:
            return self.reasoning_provider or self.default_provider
        if task == "safety":
            return self.safety_provider or self.reasoning_provider or self.default_provider
        if task in {"embedding", "embeddings"}:
            return self.embeddings_provider or self.default_provider
        return self.default_provider

    def provider_name_for_task(self, task: str) -> str | None:
        if self.policy:
            policy_choice = self.policy.provider_for_task(task)
            if policy_choice:
                return policy_choice
        provider = self.for_task(task)
        if isinstance(provider, str) or provider is None:
            return provider
        return provider.__class__.__name__
