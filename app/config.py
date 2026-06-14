from functools import lru_cache
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Autonomous Gmail AI Agent"
    environment: Literal["development", "test", "production"] = "development"
    database_url: str = "sqlite:///./agent.db"
    log_level: str = "INFO"

    llm_provider: Literal["openai", "groq"] = "groq"
    summary_llm_provider: Literal["openai", "groq", "default"] = "default"
    drafting_llm_provider: Literal["openai", "groq", "default"] = "default"
    reasoning_llm_provider: Literal["openai", "groq", "default"] = "default"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4.1-mini"
    openai_base_url: str | None = None
    openai_timeout_seconds: float = 30
    groq_api_key: str | None = None
    groq_model: str = "llama-3.3-70b-versatile"

    gmail_client_secrets_file: str | None = None
    gmail_token_file: str = "token.json"
    gmail_user_id: str = "me"
    gmail_poll_interval_seconds: int = 60
    gmail_query: str = "is:unread in:inbox -category:promotions -category:social -in:spam"
    enable_inbox_monitor: bool | None = None
    google_calendar_enabled: bool = False
    google_calendar_id: str = "primary"

    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None
    telegram_webhook_secret: str | None = None
    diagnostics_secret: str | None = None
    admin_secret: str | None = None

    auto_reply_risk_threshold: int = 55
    min_auto_reply_confidence: float = 0.75
    max_email_body_chars: int = 8000
    public_email: str = "contact@nsakash.in"
    personal_email: str = "nsakash752003@gmail.com"
    sender_aliases: str = "contact@nsakash.in,developer@nsakash.in,craftiq@nsakash.in"
    default_sender_alias: str = "contact@nsakash.in"
    default_from_email: str = "contact@nsakash.in"
    reply_from_original_recipient: bool = True
    allow_sender_alias_context_override: bool = False
    email_footer_mode: Literal["minimal", "professional", "executive", "stealth"] = "professional"
    vector_memory_enabled: bool = False
    vector_memory_provider: Literal["sqlite", "chroma", "faiss", "pgvector"] = "sqlite"
    voice_transcription_enabled: bool = False

    enable_contextual_drafting: bool = True
    enable_long_email_reasoning: bool = True
    enable_intent_extraction: bool = True
    enable_action_item_extraction: bool = True

    telegram_enable_callback_locking: bool = True
    telegram_enable_idempotency: bool = True
    telegram_enable_duplicate_protection: bool = True
    telegram_enable_safe_fallbacks: bool = True
    telegram_enable_payload_truncation: bool = True

    ai_enable_context_summarization: bool = True
    ai_enable_thread_memory: bool = True
    ai_enable_relationship_context: bool = True
    ai_enable_conversational_reasoning: bool = True
    ai_enable_scheduling_reasoning: bool = True

    enable_safe_webhook_mode: bool = True
    enable_webhook_exception_guard: bool = True
    debug_gmail_pipeline: bool = False
    seno_debug_workflow: bool = False

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @field_validator("gmail_poll_interval_seconds")
    @classmethod
    def validate_poll_interval(cls, value: int) -> int:
        if value < 15:
            raise ValueError("GMAIL_POLL_INTERVAL_SECONDS must be at least 15")
        return value

    @field_validator("auto_reply_risk_threshold")
    @classmethod
    def validate_risk_threshold(cls, value: int) -> int:
        if not 0 <= value <= 100:
            raise ValueError("AUTO_REPLY_RISK_THRESHOLD must be between 0 and 100")
        return value

    @field_validator("sender_aliases")
    @classmethod
    def validate_sender_aliases(cls, value: str) -> str:
        aliases = [item.strip().lower() for item in value.split(",") if item.strip()]
        if not aliases:
            raise ValueError("SENDER_ALIASES must contain at least one configured Gmail send-as alias")
        return ",".join(dict.fromkeys(aliases))

    @field_validator("default_sender_alias")
    @classmethod
    def normalize_default_sender_alias(cls, value: str) -> str:
        return value.strip().lower()

    @field_validator("default_from_email")
    @classmethod
    def normalize_default_from_email(cls, value: str) -> str:
        return value.strip().lower()

    def validate_runtime(self) -> None:
        missing: list[str] = []
        if self.llm_provider == "groq" and not self.groq_api_key:
            missing.append("GROQ_API_KEY")
        if not self.gmail_client_secrets_file:
            missing.append("GMAIL_CLIENT_SECRETS_FILE")
        if not self.telegram_bot_token:
            missing.append("TELEGRAM_BOT_TOKEN")
        if not self.telegram_chat_id:
            missing.append("TELEGRAM_CHAT_ID")
        if not self.telegram_webhook_secret:
            missing.append("TELEGRAM_WEBHOOK_SECRET")
        if not self.admin_secret:
            missing.append("ADMIN_SECRET")
        if missing and self.environment == "production":
            raise RuntimeError(f"Missing required production environment variables: {', '.join(missing)}")

    @property
    def inbox_monitor_enabled(self) -> bool:
        if self.enable_inbox_monitor is not None:
            return self.enable_inbox_monitor
        return self.environment == "production"

    @property
    def sender_alias_list(self) -> list[str]:
        return [item.strip().lower() for item in self.sender_aliases.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
