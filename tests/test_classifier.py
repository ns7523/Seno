import pytest

from app.ai.classifier import EmailAnalyzer
from app.ai.risk_engine import RiskAssessment
from app.models.email import EmailMessage


class FakeMessage:
    content = (
        '{"intent":"scheduling","urgency":"normal","risk_score":22,'
        '"requires_approval":false,"never_reply":false,"confidence":0.9,'
        '"summary":"A simple scheduling email","suggested_reply":"Yes, that works.",'
        '"reasons":["safe"],"tone":"friendly"}'
    )


class FakeChoice:
    message = FakeMessage()


class FakeChatResponse:
    choices = [FakeChoice()]


class FakeCompletions:
    def __init__(self):
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return FakeChatResponse()


class FakeChat:
    def __init__(self):
        self.completions = FakeCompletions()


class FakeOpenAIClient:
    def __init__(self):
        self.chat = FakeChat()


class FakeProvider:
    def __init__(self):
        self.called = False

    async def complete_json(self, *, system_prompt, user_prompt):
        self.called = True
        return {
            "intent": "acknowledgement",
            "urgency": "normal",
            "risk_score": 12,
            "requires_approval": False,
            "never_reply": False,
            "confidence": 0.98,
            "summary": "Simple acknowledgement",
            "suggested_reply": "NS has received your message.",
            "reasons": ["provider path"],
            "tone": "neutral",
        }


def make_email():
    return EmailMessage(
        gmail_id="gmail-1",
        thread_id="thread-1",
        sender="person@example.com",
        subject="Meeting",
        body="Can we meet tomorrow?",
        timestamp=None,
    )


@pytest.mark.asyncio
async def test_analyzer_uses_chat_completions_json_mode():
    client = FakeOpenAIClient()
    analyzer = EmailAnalyzer(api_key=None)
    analyzer.client = client

    analysis = await analyzer.analyze(
        make_email(),
        memory_context={"trust_score": 80},
        risk_hint=RiskAssessment(risk_score=20, requires_approval=False, never_reply=False, reasons=["safe"]),
    )

    assert analysis.intent == "scheduling"
    assert analysis.suggested_reply == "Yes, that works."
    assert client.chat.completions.kwargs["response_format"] == {"type": "json_object"}
    system_prompt = client.chat.completions.kwargs["messages"][0]["content"]
    assert "Never confirm attendance" in system_prompt
    assert "Do not mention AI" in system_prompt
    assert "automated AI-assisted response" not in system_prompt


@pytest.mark.asyncio
async def test_analyzer_uses_provider_without_legacy_client_fallback():
    provider = FakeProvider()
    analyzer = EmailAnalyzer(api_key=None, provider=provider)

    analysis = await analyzer.analyze(
        make_email(),
        memory_context={"trust_score": 80},
        risk_hint=RiskAssessment(risk_score=10, requires_approval=False, never_reply=False, reasons=["safe"]),
    )

    assert provider.called is True
    assert analysis.intent == "acknowledgement"
    assert analysis.confidence == 0.98
    assert analysis.reasons == ["provider path"]


@pytest.mark.asyncio
async def test_analyzer_fallback_requires_approval_without_api_key():
    analyzer = EmailAnalyzer(api_key=None)

    analysis = await analyzer.analyze(
        make_email(),
        memory_context={},
        risk_hint=RiskAssessment(risk_score=30, requires_approval=False, never_reply=False, reasons=["default"]),
    )

    assert analysis.requires_approval is True
    assert analysis.confidence == 0.0
    assert analysis.suggested_reply == "Thanks for the message. NS has received it."
    assert "follow-up" not in analysis.suggested_reply.lower()
    assert "shared separately" not in analysis.suggested_reply.lower()
