from app.ai.providers import GroqProvider, OpenAIProvider, build_llm_provider
from app.config import Settings


def test_build_llm_provider_returns_openai_provider():
    settings = Settings(openai_api_key="openai-key", llm_provider="openai")

    provider = build_llm_provider(settings)

    assert isinstance(provider, OpenAIProvider)


def test_build_llm_provider_returns_groq_provider():
    settings = Settings(groq_api_key="groq-key", llm_provider="groq")

    provider = build_llm_provider(settings)

    assert isinstance(provider, GroqProvider)


def test_build_llm_provider_returns_none_without_required_key():
    settings = Settings(llm_provider="groq", groq_api_key=None)

    assert build_llm_provider(settings) is None


def test_production_settings_do_not_require_openai_when_groq_is_configured():
    settings = Settings(
        environment="production",
        llm_provider="groq",
        groq_api_key="groq-key",
        openai_api_key=None,
        gmail_client_secrets_file="client_secret.json",
        telegram_bot_token="telegram",
        telegram_chat_id="chat",
        telegram_webhook_secret="secret",
        admin_secret="admin",
    )

    settings.validate_runtime()


def test_default_llm_provider_is_groq_first():
    settings = Settings(_env_file=None)

    assert settings.llm_provider == "groq"
    assert settings.groq_model == "llama-3.3-70b-versatile"
