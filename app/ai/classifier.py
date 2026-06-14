import json
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI

from app.ai.providers import BaseLLMProvider
from app.ai.risk_engine import RiskAssessment
from app.models.email import EmailMessage
from app.utils.logger import get_logger


logger = get_logger(__name__)


SYSTEM_PROMPT = """
You are an autonomous email assistant.
IMPORTANT SAFETY RULES:
- Never confirm attendance, purchases, payments, meetings, or commitments.
- Never say the user will attend, accept, buy, sign, approve, or join.
- Never generate emotionally committed responses.
- Use neutral and acknowledgment-based wording.
- Do not mention AI, automation, approval workflows, or internal tooling in the reply.
- Keep responses concise and professional.
- If uncertain, avoid commitment language.
- For auto replies, acknowledge receipt only. Do not imply another follow-up, confirmation, or later automatic message will happen.
GOOD:
- "Thanks for the message. NS has received it."
- "Thanks for reaching out."
- "Your message has been noted."
BAD:
- "I will attend."
- "I would love to join."
- "See you there."
- "Looking forward to meeting."
- "A follow-up confirmation will be shared separately."
- "A personal follow-up will be shared separately."
Return only valid JSON.
""".strip()


@dataclass(slots=True)
class EmailAnalysis:
    intent: str
    urgency: str
    risk_score: int
    requires_approval: bool
    never_reply: bool
    confidence: float
    summary: str
    suggested_reply: str
    reasons: list[str] = field(default_factory=list)
    tone: str = "neutral"


class EmailAnalyzer:
    def __init__(
        self,
        *,
        api_key: str | None,
        model: str = "gpt-4.1-mini",
        timeout: float = 30,
        auto_reply_threshold: int = 55,
        min_confidence: float = 0.75,
        provider: BaseLLMProvider | None = None,
    ) -> None:
        # Backward compatibility for tests or direct construction. Production
        # inference should flow through BaseLLMProvider when build_components is used.
        self.client = AsyncOpenAI(api_key=api_key, timeout=timeout) if api_key else None
        self.provider = provider
        self.model = model
        self.auto_reply_threshold = auto_reply_threshold
        self.min_confidence = min_confidence

    async def analyze(self, email: EmailMessage, memory_context: dict[str, Any], risk_hint: RiskAssessment) -> EmailAnalysis:
        if not self.provider and not self.client:
            logger.warning("LLM provider is not configured; using approval-required fallback", extra={"gmail_id": email.gmail_id})
            return self._fallback(email, risk_hint, "LLM provider is not configured")

        prompt = self._build_prompt(email, memory_context, risk_hint)
        try:
            logger.info("Generating AI classification", extra={"gmail_id": email.gmail_id, "model": self.model})
            if self.provider:
                data = await self.provider.complete_json(system_prompt=SYSTEM_PROMPT, user_prompt=prompt)
            else:
                assert self.client is not None
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content or "{}"
                data = json.loads(content)

            logger.info("LLM classification response received", extra={"gmail_id": email.gmail_id})
            return self._coerce(data, risk_hint)
        except Exception as exc:
            logger.exception("llm_analysis_failed", extra={"gmail_id": email.gmail_id})
            return self._fallback(email, risk_hint, f"LLM failure: {exc}")

    def _build_prompt(self, email: EmailMessage, memory_context: dict[str, Any], risk_hint: RiskAssessment) -> str:
        return json.dumps(
            {
                "email": {
                    "sender": email.sender,
                    "subject": email.subject,
                    "body": email.body[:1200],
                    "has_attachments": email.has_attachments,
                },
                "memory": memory_context,
                "deterministic_risk_hint": {
                    "risk_score": risk_hint.risk_score,
                    "requires_approval": risk_hint.requires_approval,
                    "never_reply": risk_hint.never_reply,
                    "reasons": risk_hint.reasons,
                },
                "required_json_schema": {
                    "intent": "short intent label",
                    "urgency": "low|normal|high|critical",
                    "risk_score": "integer 0-100",
                    "requires_approval": "boolean",
                    "never_reply": "boolean",
                    "confidence": "number 0-1",
                    "summary": "one sentence",
                    "suggested_reply": "professional reply draft, empty if never_reply",
                    "reasons": ["short reasons"],
                    "tone": "detected tone",
                },
            },
            ensure_ascii=True,
        )

    def _coerce(self, data: dict[str, Any], risk_hint: RiskAssessment) -> EmailAnalysis:
        risk_score = max(int(data.get("risk_score", risk_hint.risk_score)), risk_hint.risk_score)
        confidence = float(data.get("confidence", 0.0))
        never_reply = bool(data.get("never_reply", False)) or risk_hint.never_reply
        requires_approval = (
            bool(data.get("requires_approval", False))
            or risk_hint.requires_approval
            or risk_score >= self.auto_reply_threshold
            or confidence < self.min_confidence
        )
        suggested_reply = "" if never_reply else str(data.get("suggested_reply", "")).strip()
        if not never_reply and not suggested_reply:
            suggested_reply = "Thanks for the message. NS has received it."

        return EmailAnalysis(
            intent=str(data.get("intent", "unknown")),
            urgency=str(data.get("urgency", "normal")),
            risk_score=min(100, max(0, risk_score)),
            requires_approval=requires_approval,
            never_reply=never_reply,
            confidence=min(1.0, max(0.0, confidence)),
            summary=str(data.get("summary", "No summary available")),
            suggested_reply=suggested_reply,
            reasons=list(data.get("reasons") or risk_hint.reasons),
            tone=str(data.get("tone", "neutral")),
        )

    def _fallback(self, email: EmailMessage, risk_hint: RiskAssessment, reason: str) -> EmailAnalysis:
        never_reply = risk_hint.never_reply
        return EmailAnalysis(
            intent="unknown",
            urgency="normal",
            risk_score=max(risk_hint.risk_score, 80),
            requires_approval=True,
            never_reply=never_reply,
            confidence=0.0,
            summary=f"Automated analysis unavailable for: {email.subject}",
            suggested_reply="" if never_reply else "Thanks for the message. NS has received it.",
            reasons=[reason, *risk_hint.reasons],
            tone="unknown",
        )
