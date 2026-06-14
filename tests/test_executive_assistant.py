import pytest
from fastapi.testclient import TestClient

from app.ai.classifier import EmailAnalysis
from app.ai.priority_engine import PriorityEngine
from app.ai.responder import ReplySafetyError
from app.ai.router import LLMTaskRouter, RoutingPolicy
from app.calendar.service import CalendarEvent, CalendarService
from app.config import get_settings
from app.database import Database
from app.memory.memory import MemoryStore
from app.models.email import ApprovalRequest, EmailMessage
from app.services.email_processor import EmailProcessor
from app.telegram.bot import TelegramBot
from app.telegram.voice import StaticVoiceTranscriber
from app.main import create_app


class FakeHTTPClient:
    def __init__(self):
        self.requests = []

    async def post(self, url, json=None, timeout=None):
        self.requests.append((url, json, timeout))
        return {"ok": True}


class FakeAnalyzer:
    def __init__(self, analysis):
        self.analysis = analysis

    async def analyze(self, email, memory_context, risk_hint):
        return self.analysis


class FakeGmailSender:
    def __init__(self):
        self.sent = []

    def route_sender_identity(self, email):
        email.selected_sender_alias = email.selected_sender_alias or "contact@nsakash.in"
        email.alias_selection_reason = email.alias_selection_reason or "test default sender"
        return type(
            "SenderIdentityDecision",
            (),
            {
                "selected_sender_alias": email.selected_sender_alias,
                "original_recipient": email.original_recipient,
                "detection_source": email.recipient_detection_source,
                "reason": email.alias_selection_reason,
            },
        )()

    async def send_reply(self, email, body):
        self.sent.append((email.gmail_id, body))


class FakeTelegram:
    def __init__(self):
        self.approvals = []
        self.previews = []

    async def send_approval_request(self, approval):
        self.approvals.append(approval)

    async def send_draft_preview(self, approval_id, draft):
        self.previews.append((approval_id, draft))


class ConflictCalendar(CalendarService):
    async def conflicts_for_email(self, email):
        return [CalendarEvent(title="Existing call", starts_at="2026-05-16T18:00:00", ends_at="2026-05-16T19:00:00")]

    async def create_event_from_email(self, email):
        return CalendarEvent(title=email.subject, starts_at="2026-05-17T09:00:00", ends_at="2026-05-17T09:30:00")

    async def suggest_alternative_times(self, email):
        return ["2026-05-17T10:00:00", "2026-05-17T11:00:00"]


def make_email(gmail_id="gmail-1", sender="Rahul <rahul@example.com>", subject="Breakfast", body="Breakfast near UVCE tomorrow at 9 AM?"):
    return EmailMessage(
        gmail_id=gmail_id,
        thread_id="thread-1",
        sender=sender,
        subject=subject,
        body=body,
        timestamp=None,
        original_recipient="contact@nsakash.in",
        recipient_detection_source="delivered-to",
        selected_sender_alias="contact@nsakash.in",
        alias_selection_reason="matched original recipient",
    )


@pytest.mark.asyncio
async def test_telegram_email_summary_card_contains_executive_fields():
    client = FakeHTTPClient()
    bot = TelegramBot(token="token", chat_id="123", http_client=client)
    approval = ApprovalRequest(
        id=7,
        email=make_email(),
        summary="Breakfast invite near UVCE tomorrow at 9 AM.",
        suggested_reply="9 AM works. See you then.",
        risk_score=18,
        category="Casual / Social",
        urgency="normal",
        suggested_tone="Friendly",
        priority="Casual",
        reply_recommendation="Approve with friendly tone.",
    )

    await bot.send_approval_request(approval)

    payload = client.requests[0][1]
    assert "Executive Brief" in payload["text"]
    assert "Context: Casual / Social" in payload["text"]
    assert "Priority: Casual" in payload["text"]
    assert "Replying as: contact@nsakash.in" in payload["text"]
    assert "Recommendation\nApprove with friendly tone." in payload["text"]
    assert "Confidence: Low" in payload["text"]
    assert "Risk:" not in payload["text"]
    assert "Routing:" not in payload["text"]
    assert payload["reply_markup"]["inline_keyboard"][0][0]["callback_data"] == "approve:7"
    assert payload["reply_markup"]["inline_keyboard"][0][1]["callback_data"] == "reject:7"
    assert payload["reply_markup"]["inline_keyboard"][1][0]["callback_data"] == "more:7"


@pytest.mark.asyncio
async def test_style_selection_generates_draft_preview_without_sending(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()
    sender = FakeGmailSender()
    telegram = FakeTelegram()
    processor = EmailProcessor(
        db=db,
        memory=MemoryStore(db),
        analyzer=FakeAnalyzer(
            EmailAnalysis(
                intent="social",
                urgency="normal",
                risk_score=70,
                requires_approval=True,
                never_reply=False,
                confidence=0.9,
                summary="Breakfast invite",
                suggested_reply="Breakfast at 9 AM works. See you then.",
                reasons=["breakfast"],
                tone="friendly",
            )
        ),
        gmail_sender=sender,
        telegram=telegram,
    )
    await processor.process_email(make_email())
    approval = db.get_pending_approval_by_email("gmail-1")

    assert approval is not None
    assert await processor.begin_approval(approval.id) is True
    draft = await processor.preview_approved_reply(approval.id, style="friendly")

    assert sender.sent == []
    assert "ai-assisted" not in draft.lower()
    assert "Hey Rahul" in draft
    assert telegram.previews == [(approval.id, draft)]

    assert await processor.send_previewed_reply(approval.id) is True
    assert sender.sent == [("gmail-1", draft)]


@pytest.mark.asyncio
async def test_previewed_finance_reply_remains_blocked(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()
    processor = EmailProcessor(
        db=db,
        memory=MemoryStore(db),
        analyzer=FakeAnalyzer(
            EmailAnalysis(
                intent="payment",
                urgency="high",
                risk_score=90,
                requires_approval=True,
                never_reply=False,
                confidence=0.9,
                summary="Payment confirmation",
                suggested_reply="I confirm payment has been sent.",
                reasons=["payment"],
                tone="formal",
            )
        ),
        gmail_sender=FakeGmailSender(),
        telegram=FakeTelegram(),
    )
    await processor.process_email(make_email(subject="Invoice", body="Please confirm invoice payment."))
    approval = db.get_pending_approval_by_email("gmail-1")

    assert approval is not None
    assert await processor.begin_approval(approval.id) is True
    with pytest.raises(ReplySafetyError):
        await processor.preview_approved_reply(approval.id, style="friendly")


def test_priority_engine_assigns_urgent_medium_and_casual():
    engine = PriorityEngine()

    urgent = engine.assess(make_email(sender="recruiter@company.com", subject="Urgent interview deadline", body="Please reply today."), risk_score=30, urgency="high")
    medium = engine.assess(make_email(sender="professor@uvce.edu", subject="Project review", body="Please review this week."), risk_score=45, urgency="normal")
    casual = engine.assess(make_email(sender="friend@example.com", subject="Coffee", body="coffee at campus?"), risk_score=12, urgency="normal")

    assert urgent.level == "Executive Attention Required"
    assert medium.level == "Medium"
    assert casual.level == "Casual"


def test_relationship_memory_learns_role_tone_and_signoff(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()
    memory = MemoryStore(db)

    memory.record_tone_selection("recruiter@company.com", "formal")
    memory.record_user_edit("recruiter@company.com", "Hi,\n\nWorks.\n\n- NS")
    profile = memory.get_relationship_profile("recruiter@company.com")

    assert profile.preferred_tone == "formal"
    assert profile.preferred_signoff == "- NS"
    assert profile.relationship_type == "recruiter"


@pytest.mark.asyncio
async def test_calendar_conflict_warning_is_added_to_approval(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()
    telegram = FakeTelegram()
    processor = EmailProcessor(
        db=db,
        memory=MemoryStore(db),
        analyzer=FakeAnalyzer(
            EmailAnalysis(
                intent="meeting",
                urgency="normal",
                risk_score=70,
                requires_approval=True,
                never_reply=False,
                confidence=0.9,
                summary="Meeting at 6 PM",
                suggested_reply="6 PM works. See you then.",
                reasons=["meeting"],
                tone="normal",
            )
        ),
        gmail_sender=FakeGmailSender(),
        telegram=telegram,
        calendar_service=ConflictCalendar(),
    )

    await processor.process_email(make_email(subject="Meeting", body="Meeting at 6 PM?"))

    assert telegram.approvals
    assert "Schedule warning" in telegram.approvals[0].summary


def test_dashboard_apis_report_approvals_and_metrics(monkeypatch, tmp_path):
    get_settings.cache_clear()
    monkeypatch.setenv("ADMIN_SECRET", "secret")
    app = create_app(database_url=f"sqlite:///{tmp_path / 'agent.db'}", start_scheduler=False)
    with TestClient(app) as client:
        response = client.get("/dashboard/api/summary", headers={"X-ADMIN-SECRET": "secret"})

    assert response.status_code == 200
    assert {"pending_approvals", "sent_emails", "top_contacts"} <= set(response.json())
    get_settings.cache_clear()


def test_voice_transcriber_returns_polished_instruction():
    transcriber = StaticVoiceTranscriber("Tell him 9 AM works and we will discuss project ideas.")

    assert transcriber.transcribe(b"voice-bytes") == "Tell him 9 AM works and we will discuss project ideas."


@pytest.mark.asyncio
async def test_style_learning_adapts_future_greeting_and_signoff(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()
    memory = MemoryStore(db)
    memory.record_user_edit("rahul@example.com", "Hello Rahul,\n\n9 AM works.\n\nCheers,\nNS")
    processor = EmailProcessor(
        db=db,
        memory=memory,
        analyzer=FakeAnalyzer(
            EmailAnalysis(
                intent="social",
                urgency="normal",
                risk_score=70,
                requires_approval=True,
                never_reply=False,
                confidence=0.9,
                summary="Coffee invite",
                suggested_reply="Coffee at 9 AM works. See you then.",
                reasons=["coffee"],
                tone="friendly",
            )
        ),
        gmail_sender=FakeGmailSender(),
        telegram=FakeTelegram(),
    )

    await processor.process_email(make_email(sender="Rahul <rahul@example.com>", subject="Coffee", body="coffee at 9 AM?"))
    approval = db.get_pending_approval_by_email("gmail-1")
    assert approval is not None
    await processor.begin_approval(approval.id)
    draft = await processor.preview_approved_reply(approval.id, style="friendly")

    assert "Hello Rahul," in draft
    assert "Cheers,\nNS" in draft
    assert draft.rstrip().endswith("Sent via Seno, NS's executive communication assistant.")
    profile = memory.get_relationship_profile("rahul@example.com")
    assert profile.tone_confidence > 0


@pytest.mark.asyncio
async def test_short_direct_edit_style_shapes_future_professional_drafts(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()
    memory = MemoryStore(db)
    memory.record_user_edit("maya@example.com", "Hi Maya,\n\nTuesday at 6 PM works.\n\n- NS")
    memory.record_user_edit("maya@example.com", "Hi Maya,\n\nSounds good.\n\n- NS")
    processor = EmailProcessor(
        db=db,
        memory=memory,
        analyzer=FakeAnalyzer(
            EmailAnalysis(
                intent="collaboration",
                urgency="normal",
                risk_score=70,
                requires_approval=True,
                never_reply=False,
                confidence=0.9,
                summary="Collaboration discussion",
                suggested_reply="Thank you for reaching out",
                reasons=["collaboration"],
                tone="normal",
            )
        ),
        gmail_sender=FakeGmailSender(),
        telegram=FakeTelegram(),
    )

    await processor.process_email(
        make_email(
            sender="Maya <maya@example.com>",
            subject="Seno collaboration",
            body="Can we discuss Seno and the approval-driven workflow next Tuesday at 6 PM?",
        )
    )
    approval = db.get_pending_approval_by_email("gmail-1")
    assert approval is not None
    draft = await processor.preview_approved_reply(approval.id, style="normal")

    assert "Hi Maya," in draft
    assert "- NS" in draft
    assert "I can also talk through" not in draft
    assert "next Tuesday around 6 PM" in draft


@pytest.mark.asyncio
async def test_thread_continuity_warns_before_conflicting_time(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()
    prior = make_email(gmail_id="old", subject="Meeting", body="Meeting at 6 PM?", sender="Rahul <rahul@example.com>")
    db.record_email(prior, status="approved_sent")
    prior_approval = db.create_approval(prior, "Prior meeting", "6 PM works. See you then.", 60)
    db.set_approval_draft(prior_approval.id, "6 PM works. See you then.", "friendly")
    db.decide_approval(prior_approval.id, True)
    processor = EmailProcessor(
        db=db,
        memory=MemoryStore(db),
        analyzer=FakeAnalyzer(
            EmailAnalysis(
                intent="meeting",
                urgency="normal",
                risk_score=70,
                requires_approval=True,
                never_reply=False,
                confidence=0.9,
                summary="Conflicting meeting time",
                suggested_reply="7 PM works. See you then.",
                reasons=["meeting"],
                tone="friendly",
            )
        ),
        gmail_sender=FakeGmailSender(),
        telegram=FakeTelegram(),
    )

    await processor.process_email(make_email(gmail_id="new", subject="Meeting update", body="Actually meet at 7 PM?", sender="Rahul <rahul@example.com>"))
    approval = db.get_pending_approval_by_email("new")
    assert approval is not None
    await processor.begin_approval(approval.id)
    draft = await processor.preview_approved_reply(approval.id, style="friendly")

    assert "previously had 6 PM noted" in draft


@pytest.mark.asyncio
async def test_calendar_create_event_and_alternative_times(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()
    processor = EmailProcessor(
        db=db,
        memory=MemoryStore(db),
        analyzer=FakeAnalyzer(
            EmailAnalysis(
                intent="meeting",
                urgency="normal",
                risk_score=70,
                requires_approval=True,
                never_reply=False,
                confidence=0.9,
                summary="Breakfast meeting",
                suggested_reply="9 AM works. See you then.",
                reasons=["calendar"],
                tone="normal",
            )
        ),
        gmail_sender=FakeGmailSender(),
        telegram=FakeTelegram(),
        calendar_service=ConflictCalendar(),
    )
    await processor.process_email(make_email())
    approval = db.get_pending_approval_by_email("gmail-1")
    assert approval is not None

    event = await processor.create_calendar_event(approval.id)
    alternatives = await processor.suggest_alternative_times(approval.id)

    assert event is not None
    assert event.title == "Breakfast"
    assert alternatives == ["2026-05-17T10:00:00", "2026-05-17T11:00:00"]


def test_approval_card_includes_confidence_and_risk_explanation():
    approval = ApprovalRequest(
        id=7,
        email=make_email(),
        summary="Invoice with attachment.",
        suggested_reply="I will review this.",
        risk_score=88,
        confidence=0.81,
        risk_explanation=["attachment detected", "finance keywords found", "external sender"],
    )

    assert "81%" in approval.confidence_label
    assert "attachment detected" in approval.risk_explanation_text


def test_routing_policy_selects_task_specific_providers():
    router = LLMTaskRouter(
        default_provider="groq",
        summary_provider="groq",
        drafting_provider="groq",
        reasoning_provider="groq",
        safety_provider="local",
        embeddings_provider="local",
        policy=RoutingPolicy(),
    )

    assert router.provider_name_for_task("summary") == "groq"
    assert router.provider_name_for_task("casual_draft") == "groq"
    assert router.provider_name_for_task("legal_reasoning") == "groq"
    assert router.provider_name_for_task("safety") == "local"
    assert router.provider_name_for_task("embedding") == "local"


def test_vector_memory_recalls_semantic_context(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()

    db.add_vector_memory("rahul@example.com", "relationship", "Rahul prefers friendly short replies about UVCE plans.")
    recalled = db.recall_memory("rahul@example.com", "UVCE friendly plans")

    assert recalled
    assert "friendly short replies" in recalled[0]["content"]


def test_dashboard_summary_includes_intelligence_metrics(monkeypatch, tmp_path):
    get_settings.cache_clear()
    monkeypatch.setenv("ADMIN_SECRET", "secret")
    app = create_app(database_url=f"sqlite:///{tmp_path / 'agent.db'}", start_scheduler=False)

    with TestClient(app) as client:
        response = client.get("/dashboard/api/summary", headers={"X-ADMIN-SECRET": "secret"})
        html = client.get("/dashboard", headers={"X-ADMIN-SECRET": "secret"})

    assert response.status_code == 200
    data = response.json()
    assert {"tone_statistics", "risk_distribution", "relationship_profiles"} <= set(data)
    assert "Executive Communication Assistant" in html.text
    assert "Risk Distribution" in html.text
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_internship_opportunity_generates_full_contextual_formal_draft(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()
    processor = EmailProcessor(
        db=db,
        memory=MemoryStore(db),
        analyzer=FakeAnalyzer(
            EmailAnalysis(
                intent="internship_opportunity",
                urgency="high",
                risk_score=72,
                requires_approval=True,
                never_reply=False,
                confidence=0.92,
                summary="Internship discussion with project expectations and timing.",
                suggested_reply="Thanks for the message. NS has received it.",
                reasons=["recruiter outreach"],
                tone="formal",
            )
        ),
        gmail_sender=FakeGmailSender(),
        telegram=FakeTelegram(),
    )
    body = (
        "We reviewed your portfolio and would like to discuss a backend AI internship. "
        "Please confirm if you are available tomorrow at 5:30 PM. We would also like you "
        "to share your experience with FastAPI, Gmail automation, and your recent AI project."
    )
    await processor.process_email(make_email(sender="Arun <arun@startup.com>", subject="AI Backend Internship Opportunity", body=body))
    approval = db.get_pending_approval_by_email("gmail-1")

    assert approval is not None
    await processor.begin_approval(approval.id)
    draft = await processor.preview_approved_reply(approval.id, style="formal")

    assert "Dear Arun," in draft
    assert "internship opportunity" in draft.lower()
    assert "5:30 PM" in draft
    assert "FastAPI" in draft
    assert "Gmail automation" in draft
    assert "AI project" in draft
    assert "Thanks for the message. NS has received it." not in draft


@pytest.mark.asyncio
async def test_formal_collaboration_draft_rejects_generic_opener_and_uses_context(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()
    processor = EmailProcessor(
        db=db,
        memory=MemoryStore(db),
        analyzer=FakeAnalyzer(
            EmailAnalysis(
                intent="collaboration_discussion",
                urgency="normal",
                risk_score=67,
                requires_approval=True,
                never_reply=False,
                confidence=0.9,
                summary="Collaboration request about Seno workflow architecture and scheduling.",
                suggested_reply="Hello,\nThank you for reaching out",
                reasons=["collaboration", "scheduling intent", "technical architecture discussion"],
                tone="formal",
            )
        ),
        gmail_sender=FakeGmailSender(),
        telegram=FakeTelegram(),
    )
    body = (
        "Hi NS, I reviewed Seno and wanted to discuss the AI workflow and communication workflow architecture. "
        "The approval-driven approach and contextual drafting system look interesting. "
        "Could we connect Tuesday or Wednesday evening around 6 PM IST to discuss orchestration, "
        "communication infrastructure, and possible collaboration?"
    )
    await processor.process_email(
        make_email(
            sender="Reviewer <reviewer7523@gmail.com>",
            subject="Collaboration Discussion & Availability Next Week",
            body=body,
        )
    )
    approval = db.get_pending_approval_by_email("gmail-1")

    assert approval is not None
    await processor.begin_approval(approval.id)
    draft = await processor.preview_approved_reply(approval.id, style="formal")
    lowered = draft.lower()

    assert "thank you for taking the time to look through" in lowered
    assert "thank you for reaching out" not in lowered[:140]
    assert "seno" in lowered
    assert "workflow" in lowered
    assert "architecture decisions" in lowered
    assert "collaboration" in lowered
    assert "tuesday or wednesday evening around 6 pm ist" in lowered
    assert "approval flow" in lowered
    assert "orchestration" not in lowered
    assert "communication infrastructure" not in lowered
    assert draft.strip().lower() != "hello,\nthank you for reaching out"


@pytest.mark.asyncio
async def test_send_button_uses_exact_contextual_preview_as_source_of_truth(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()
    sender = FakeGmailSender()
    processor = EmailProcessor(
        db=db,
        memory=MemoryStore(db),
        analyzer=FakeAnalyzer(
            EmailAnalysis(
                intent="recruiter_outreach",
                urgency="high",
                risk_score=70,
                requires_approval=True,
                never_reply=False,
                confidence=0.91,
                summary="Recruiter wants to discuss backend AI internship details.",
                suggested_reply="Thanks for the message. NS has received it.",
                reasons=["recruiter outreach"],
                tone="formal",
            )
        ),
        gmail_sender=sender,
        telegram=FakeTelegram(),
    )
    email = make_email(
        sender="Ravi <ravi@company.com>",
        subject="Backend AI Internship Discussion",
        body=(
            "Hi NS, we liked your Gmail automation and FastAPI work. "
            "Can we discuss the backend AI internship tomorrow at 5:30 PM?"
        ),
    )
    await processor.process_email(email)
    approval = db.get_pending_approval_by_email("gmail-1")

    assert approval is not None
    await processor.begin_approval(approval.id)
    preview = await processor.preview_approved_reply(approval.id, style="formal")
    persisted = db.get_pending_approval(approval.id)

    assert persisted is not None
    assert persisted[0].final_reply == preview
    assert persisted[0].selected_style == "formal"
    assert persisted[0].draft_checksum

    assert await processor.send_previewed_reply(approval.id) is True
    assert sender.sent == [("gmail-1", preview)]
    assert "Thanks for the message. NS has received it." not in sender.sent[0][1]


@pytest.mark.asyncio
async def test_send_button_aborts_if_persisted_preview_would_be_rewritten(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()
    sender = FakeGmailSender()
    processor = EmailProcessor(
        db=db,
        memory=MemoryStore(db),
        analyzer=FakeAnalyzer(
            EmailAnalysis(
                intent="meeting",
                urgency="normal",
                risk_score=65,
                requires_approval=True,
                never_reply=False,
                confidence=0.9,
                summary="Meeting request",
                suggested_reply="6 PM works.",
                reasons=["meeting"],
                tone="normal",
            )
        ),
        gmail_sender=sender,
        telegram=FakeTelegram(),
    )
    await processor.process_email(make_email(subject="Meeting", body="Meeting at UVCE at 6 PM?"))
    approval = db.get_pending_approval_by_email("gmail-1")

    assert approval is not None
    await processor.begin_approval(approval.id)
    assert db.set_approval_draft(approval.id, "This is an AI-assisted response approved by NS.\n\n6 PM works.", "normal") is True

    assert await processor.send_previewed_reply(approval.id) is False
    assert sender.sent == []


@pytest.mark.asyncio
async def test_long_multi_topic_collaboration_email_answers_all_major_points(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()
    processor = EmailProcessor(
        db=db,
        memory=MemoryStore(db),
        analyzer=FakeAnalyzer(
            EmailAnalysis(
                intent="collaboration",
                urgency="normal",
                risk_score=68,
                requires_approval=True,
                never_reply=False,
                confidence=0.88,
                summary="Collaboration proposal covering API design, dashboard, and timeline.",
                suggested_reply="Noted.",
                reasons=["collaboration"],
                tone="normal",
            )
        ),
        gmail_sender=FakeGmailSender(),
        telegram=FakeTelegram(),
    )
    body = (
        "Hi NS, I saw your executive email assistant and want to collaborate. "
        "Could you review the API design, suggest improvements for the dashboard, "
        "and let me know whether we can schedule a project discussion on Friday at 4 PM? "
        "Also please share any thoughts on deployment and testing."
    )
    await processor.process_email(make_email(sender="Maya <maya@example.com>", subject="Collaboration on AI assistant", body=body))
    approval = db.get_pending_approval_by_email("gmail-1")

    assert approval is not None
    await processor.begin_approval(approval.id)
    draft = await processor.preview_approved_reply(approval.id, style="normal")

    lowered = draft.lower()
    assert "collaborat" in lowered
    assert "api design" in lowered
    assert "dashboard" in lowered
    assert "friday" in lowered
    assert "4 PM" in draft
    assert "deployment" in lowered
    assert "testing" in lowered


@pytest.mark.asyncio
async def test_friendly_dinner_invite_remains_contextual_not_generic(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()
    processor = EmailProcessor(
        db=db,
        memory=MemoryStore(db),
        analyzer=FakeAnalyzer(
            EmailAnalysis(
                intent="social_invite",
                urgency="normal",
                risk_score=62,
                requires_approval=True,
                never_reply=False,
                confidence=0.9,
                summary="Dinner invite at Season Club.",
                suggested_reply="Thanks for the message. NS has received it.",
                reasons=["dinner"],
                tone="friendly",
            )
        ),
        gmail_sender=FakeGmailSender(),
        telegram=FakeTelegram(),
    )
    await processor.process_email(make_email(subject="Dinner tonight", body="Let's head to dinner tonight at 8 PM at Season Club."))
    approval = db.get_pending_approval_by_email("gmail-1")

    assert approval is not None
    await processor.begin_approval(approval.id)
    draft = await processor.preview_approved_reply(approval.id, style="friendly")

    assert "Hey Rahul," in draft
    assert "8 PM" in draft
    assert "Season Club" in draft
    assert "dinner" in draft.lower()
    assert "Thanks for the message. NS has received it." not in draft
