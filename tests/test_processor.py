import pytest

from app.ai.classifier import EmailAnalysis
from app.ai.responder import ReplySafetyError
from app.database import Database
from app.memory.memory import MemoryStore
from app.models.email import EmailMessage
from app.services.email_processor import EmailProcessor


class FakeAnalyzer:
    def __init__(self, analysis):
        self.analysis = analysis
        self.memory_contexts = []

    async def analyze(self, email, memory_context, risk_hint):
        self.memory_contexts.append(memory_context)
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


class FailingGmailSender(FakeGmailSender):
    async def send_reply(self, email, body):
        raise RuntimeError("gmail unavailable")


class FakeGmailManager:
    def __init__(self):
        self.archived = []

    async def archive_email(self, email):
        self.archived.append(email.gmail_id)


class FakeTelegram:
    def __init__(self):
        self.requests = []
        self.style_requests = []
        self.messages = []
        self.previews = []

    async def send_approval_request(self, approval):
        self.requests.append(approval)

    async def send_style_selection(self, approval_id):
        self.style_requests.append(approval_id)

    async def send_draft_preview(self, approval_id, draft):
        self.previews.append((approval_id, draft))

    async def send_message(self, text):
        self.messages.append(text)


class FailingTelegram:
    async def send_approval_request(self, approval):
        raise RuntimeError("telegram unavailable")


class ExplodingAnalyzer:
    async def analyze(self, email, memory_context, risk_hint):
        raise AssertionError("LLM should not be called for deterministic never-reply messages")


def test_processor_routing_threshold_policy_respects_contextual_zone():
    from app.ai.risk_engine import RiskAssessment

    processor = EmailProcessor.__new__(EmailProcessor)
    processor.min_confidence = 0.75
    safe = EmailAnalysis(
        intent="greeting",
        urgency="normal",
        risk_score=49,
        requires_approval=True,
        never_reply=False,
        confidence=0.9,
        summary="Simple greeting",
        suggested_reply="Thanks.",
    )
    contextual = EmailAnalysis(
        intent="collaboration",
        urgency="normal",
        risk_score=52,
        requires_approval=False,
        never_reply=False,
        confidence=0.9,
        summary="Collaboration discussion",
        suggested_reply="Thanks.",
    )
    high = EmailAnalysis(
        intent="discussion",
        urgency="normal",
        risk_score=60,
        requires_approval=False,
        never_reply=False,
        confidence=0.9,
        summary="Discussion",
        suggested_reply="Thanks.",
    )
    neutral_hint = RiskAssessment(risk_score=20, requires_approval=False, never_reply=False, reasons=[])

    assert processor._requires_approval(safe, neutral_hint, executive_signal=False) is False
    assert processor._requires_approval(contextual, neutral_hint, executive_signal=True) is True
    assert processor._requires_approval(high, neutral_hint, executive_signal=False) is True


@pytest.mark.asyncio
async def test_ignored_sender_bypasses_analyzer_drafting_and_notifications(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()
    db.ignore_sender("alerts@example.com", reason="noisy sender")
    sender = FakeGmailSender()
    telegram = FakeTelegram()
    processor = EmailProcessor(
        db=db,
        memory=MemoryStore(db),
        analyzer=ExplodingAnalyzer(),
        gmail_sender=sender,
        telegram=telegram,
    )

    await processor.process_email(make_email(gmail_id="ignored-1", sender="Alerts <alerts@example.com>", subject="Alert", body="FYI"))

    assert sender.sent == []
    assert telegram.requests == []
    assert db.is_processed("ignored-1") is True
    assert db.get_email_status("ignored-1") == "ignored_sender"


def test_ignore_sender_refuses_protected_addresses(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()
    processor = EmailProcessor(
        db=db,
        memory=MemoryStore(db),
        analyzer=FakeAnalyzer(None),
        gmail_sender=FakeGmailSender(),
        telegram=FakeTelegram(),
        protected_ignore_addresses=["nsakash752003@gmail.com", "contact@nsakash.in"],
    )
    email = make_email(gmail_id="protected-1", sender="NS <nsakash752003@gmail.com>")
    db.record_email(email, "pending_approval")
    approval = db.create_approval(email, "summary", "draft", 20)

    assert processor.ignore_sender(approval.id) is False
    assert db.is_sender_ignored("nsakash752003@gmail.com") is False
    admin_email = make_email(gmail_id="protected-2", sender="Admin <admin@example.com>")
    db.record_email(admin_email, "pending_approval")
    admin_approval = db.create_approval(admin_email, "summary", "draft", 20)

    assert processor.ignore_sender(admin_approval.id) is False
    assert db.is_sender_ignored("admin@example.com") is False


def test_ignore_sender_requires_confirmation_for_moderate_risk(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()
    processor = EmailProcessor(
        db=db,
        memory=MemoryStore(db),
        analyzer=FakeAnalyzer(None),
        gmail_sender=FakeGmailSender(),
        telegram=FakeTelegram(),
    )
    email = make_email(gmail_id="moderate-1", sender="Alerts <alerts@example.com>")
    db.record_email(email, "pending_approval")
    approval = db.create_approval(email, "summary", "draft", 55)

    assert processor.ignore_sender(approval.id) is False
    assert db.is_sender_ignored("alerts@example.com") is False
    assert processor.ignore_sender(approval.id, confirmed=True) is True
    assert db.is_sender_ignored("alerts@example.com") is True


@pytest.mark.asyncio
async def test_ignored_thread_bypasses_analyzer_drafting_and_notifications(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()
    db.ignore_thread("thread-muted", reason="thread mute")
    sender = FakeGmailSender()
    telegram = FakeTelegram()
    processor = EmailProcessor(
        db=db,
        memory=MemoryStore(db),
        analyzer=ExplodingAnalyzer(),
        gmail_sender=sender,
        telegram=telegram,
    )

    await processor.process_email(
        EmailMessage("thread-muted-1", "thread-muted", "Person <person@example.com>", "Re: noisy thread", "FYI", None)
    )

    assert sender.sent == []
    assert telegram.requests == []
    assert db.get_email_status("thread-muted-1") == "ignored_thread"


@pytest.mark.asyncio
async def test_auto_handle_similar_low_risk_bypasses_notifications_without_reply(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()
    db.auto_handle_sender("alerts@example.com", reason="transactional alerts")
    sender = FakeGmailSender()
    telegram = FakeTelegram()
    processor = EmailProcessor(
        db=db,
        memory=MemoryStore(db),
        analyzer=ExplodingAnalyzer(),
        gmail_sender=sender,
        telegram=telegram,
    )

    await processor.process_email(make_email(gmail_id="auto-handle-1", sender="Alerts <alerts@example.com>", subject="FYI", body="Hi"))

    assert sender.sent == []
    assert telegram.requests == []
    assert db.get_email_status("auto-handle-1") == "auto_handled_similar"


@pytest.mark.asyncio
async def test_pinned_sender_raises_approval_priority_without_auto_whitelisting(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()
    db.pin_sender("founder@example.com", reason="important")
    telegram = FakeTelegram()
    processor = EmailProcessor(
        db=db,
        memory=MemoryStore(db),
        analyzer=FakeAnalyzer(
            EmailAnalysis(
                intent="discussion",
                urgency="normal",
                risk_score=62,
                requires_approval=True,
                never_reply=False,
                confidence=0.9,
                summary="Project discussion",
                suggested_reply="Thanks.",
                reasons=["discussion"],
                tone="formal",
            )
        ),
        gmail_sender=FakeGmailSender(),
        telegram=telegram,
    )

    await processor.process_email(make_email(gmail_id="pinned-1", sender="Founder <founder@example.com>", subject="Project", body="Can we discuss?"))

    assert telegram.requests
    assert telegram.requests[0].priority == "Executive Attention Required"
    assert any("pinned sender" in reason for reason in telegram.requests[0].risk_explanation)


@pytest.mark.asyncio
async def test_thread_summary_is_passed_to_analyzer_context(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()
    analyzer = FakeAnalyzer(
        EmailAnalysis(
            intent="meeting",
            urgency="normal",
            risk_score=65,
            requires_approval=True,
            never_reply=False,
            confidence=0.9,
            summary="Follow-up meeting",
            suggested_reply="Tuesday at 6 PM works.",
            reasons=["meeting"],
            tone="normal",
        )
    )
    memory = MemoryStore(db)
    prior = make_email(gmail_id="prior-thread", subject="Prior agenda", body="Please share API notes. Can we meet Tuesday at 6 PM?")
    memory.record_thread_observation(prior, reply_text="I will share API notes.")
    processor = EmailProcessor(
        db=db,
        memory=memory,
        analyzer=analyzer,
        gmail_sender=FakeGmailSender(),
        telegram=FakeTelegram(),
    )

    await processor.process_email(make_email(gmail_id="thread-new", subject="Re: Prior agenda", body="Following up on the API notes."))

    assert analyzer.memory_contexts
    thread_summary = analyzer.memory_contexts[-1]["thread_summary"]
    assert thread_summary["commitments"]
    assert thread_summary["unresolved_items"]


def make_email(gmail_id="gmail-1", subject="Meeting", body="Can we meet?", sender="person@example.com"):
    return EmailMessage(
        gmail_id=gmail_id,
        thread_id="thread-1",
        sender=sender,
        subject=subject,
        body=body,
        timestamp=None,
    )


def test_database_persists_sender_routing_metadata(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()
    email = make_email()
    email.original_recipient = "developer@nsakash.in"
    email.recipient_detection_source = "delivered-to"
    email.selected_sender_alias = "developer@nsakash.in"
    email.alias_selection_reason = "matched original recipient"

    db.record_email(email, status="received")

    with db.connect() as conn:
        row = conn.execute(
            """
            SELECT original_recipient, selected_sender_alias,
                   alias_selection_reason, recipient_detection_source
            FROM emails
            WHERE gmail_id = ?
            """,
            (email.gmail_id,),
        ).fetchone()

    assert dict(row) == {
        "original_recipient": "developer@nsakash.in",
        "selected_sender_alias": "developer@nsakash.in",
        "alias_selection_reason": "matched original recipient",
        "recipient_detection_source": "delivered-to",
    }


@pytest.mark.asyncio
async def test_scoped_edit_sessions_do_not_cross_wire_replies(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()
    sender = FakeGmailSender()
    processor = EmailProcessor(
        db=db,
        memory=MemoryStore(db),
        analyzer=FakeAnalyzer(None),
        gmail_sender=sender,
        telegram=FakeTelegram(),
    )
    first = make_email(gmail_id="gmail-edit-1", subject="First", body="First email")
    second = make_email(gmail_id="gmail-edit-2", subject="Second", body="Second email")
    db.record_email(first, "pending_approval")
    db.record_email(second, "pending_approval")
    first_approval = db.create_approval(first, "summary", "draft", 60)
    second_approval = db.create_approval(second, "summary", "draft", 60)

    assert processor.start_edit_reply(
        first_approval.id,
        telegram_chat_id="chat-1",
        telegram_message_id="message-1",
        telegram_user_id="user-1",
    )
    assert processor.start_edit_reply(
        second_approval.id,
        telegram_chat_id="chat-1",
        telegram_message_id="message-2",
        telegram_user_id="user-2",
    )

    assert await processor.send_edited_reply(
        "Hi,\n\nEdited second reply.\n\nRegards,\nNS",
        telegram_chat_id="chat-1",
        telegram_user_id="user-2",
        reply_to_message_id="message-2",
    )

    assert sender.sent == [("gmail-edit-2", "Hi,\n\nEdited second reply.\n\nRegards,\nNS\n\nSent via Seno, NS's executive communication assistant.")]
    assert db.get_approval_status(first_approval.id) == "editing"
    assert db.get_approval_status(second_approval.id) == "approved"


@pytest.mark.asyncio
async def test_unscoped_ambiguous_edit_reply_is_rejected(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()
    sender = FakeGmailSender()
    processor = EmailProcessor(
        db=db,
        memory=MemoryStore(db),
        analyzer=FakeAnalyzer(None),
        gmail_sender=sender,
        telegram=FakeTelegram(),
    )
    first = make_email(gmail_id="gmail-ambiguous-1")
    second = make_email(gmail_id="gmail-ambiguous-2")
    db.record_email(first, "pending_approval")
    db.record_email(second, "pending_approval")
    first_approval = db.create_approval(first, "summary", "draft", 60)
    second_approval = db.create_approval(second, "summary", "draft", 60)
    processor.start_edit_reply(first_approval.id, telegram_chat_id="chat-1", telegram_message_id="message-1", telegram_user_id="user-1")
    processor.start_edit_reply(second_approval.id, telegram_chat_id="chat-1", telegram_message_id="message-2", telegram_user_id="user-1")

    handled = await processor.send_edited_reply(
        "Hi,\n\nAmbiguous reply.\n\nRegards,\nNS",
        telegram_chat_id="chat-1",
        telegram_user_id="user-1",
    )

    assert handled is False
    assert sender.sent == []


@pytest.mark.asyncio
async def test_processor_auto_replies_to_safe_low_context_email_with_full_format(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()
    sender = FakeGmailSender()
    telegram = FakeTelegram()
    processor = EmailProcessor(
        db=db,
        memory=MemoryStore(db),
        analyzer=FakeAnalyzer(
            EmailAnalysis(
                intent="greeting",
                urgency="normal",
                risk_score=20,
                requires_approval=False,
                never_reply=False,
                confidence=0.91,
                summary="Simple greeting",
                suggested_reply="Thanks for saying hello. I’ve seen your message and appreciate you reaching out.",
                reasons=["low risk"],
                tone="neutral",
            )
        ),
        gmail_sender=sender,
        telegram=telegram,
    )

    await processor.process_email(make_email(subject="Hello", body="Hi NS, just saying hello."))

    assert sender.sent
    sent_body = sender.sent[0][1]
    assert sent_body.startswith("Dear Person,")
    assert "Thank you for saying hello" in sent_body
    assert "Kind regards,\nNS" in sent_body
    assert sent_body.endswith("Sent via Seno, NS's executive communication assistant.")
    assert telegram.requests == []
    assert db.is_processed("gmail-1") is True


@pytest.mark.asyncio
async def test_processor_routes_professional_scheduling_to_telegram_approval(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()
    sender = FakeGmailSender()
    telegram = FakeTelegram()
    processor = EmailProcessor(
        db=db,
        memory=MemoryStore(db),
        analyzer=FakeAnalyzer(
            EmailAnalysis(
                intent="scheduling",
                urgency="normal",
                risk_score=20,
                requires_approval=False,
                never_reply=False,
                confidence=0.95,
                summary="Internship scheduling discussion",
                suggested_reply="Tomorrow at 6 PM works.",
                reasons=["safe"],
                tone="normal",
            )
        ),
        gmail_sender=sender,
        telegram=telegram,
    )

    await processor.process_email(
        make_email(
            subject="Internship discussion",
            body="Would tomorrow at 6 PM work to discuss internship opportunities?",
        )
    )

    assert sender.sent == []
    assert len(telegram.requests) == 1
    assert db.get_pending_approval_by_email("gmail-1") is not None


@pytest.mark.asyncio
async def test_processor_creates_approval_for_risky_email(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()
    sender = FakeGmailSender()
    telegram = FakeTelegram()
    processor = EmailProcessor(
        db=db,
        memory=MemoryStore(db),
        analyzer=FakeAnalyzer(
            EmailAnalysis(
                intent="legal",
                urgency="high",
                risk_score=88,
                requires_approval=True,
                never_reply=False,
                confidence=0.82,
                summary="Legal contract question",
                suggested_reply="I will review and respond soon.",
                reasons=["legal"],
                tone="formal",
            )
        ),
        gmail_sender=sender,
        telegram=telegram,
    )

    await processor.process_email(make_email())

    assert sender.sent == []
    assert len(telegram.requests) == 1
    assert telegram.requests[0].suggested_reply == "I will review and respond soon."


@pytest.mark.asyncio
async def test_processor_keeps_approval_retryable_when_telegram_fails(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()
    processor = EmailProcessor(
        db=db,
        memory=MemoryStore(db),
        analyzer=FakeAnalyzer(
            EmailAnalysis(
                intent="legal",
                urgency="high",
                risk_score=88,
                requires_approval=True,
                never_reply=False,
                confidence=0.82,
                summary="Legal contract question",
                suggested_reply="Neutral reply",
                reasons=["legal"],
                tone="formal",
            )
        ),
        gmail_sender=FakeGmailSender(),
        telegram=FailingTelegram(),
    )

    with pytest.raises(RuntimeError):
        await processor.process_email(make_email())

    approval = db.get_pending_approval_by_email("gmail-1")
    assert approval is not None
    assert approval.notification_status == "failed"
    assert db.is_processed("gmail-1") is False


@pytest.mark.asyncio
async def test_processor_skips_llm_for_noreply_newsletter(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()
    processor = EmailProcessor(
        db=db,
        memory=MemoryStore(db),
        analyzer=ExplodingAnalyzer(),
        gmail_sender=FakeGmailSender(),
        telegram=FakeTelegram(),
    )

    await processor.process_email(
        EmailMessage(
            gmail_id="newsletter-1",
            thread_id="thread-1",
            sender="noreply@example.com",
            subject="Newsletter",
            body="Unsubscribe from this promotional newsletter.",
            timestamp=None,
        )
    )

    assert db.is_processed("newsletter-1") is True


@pytest.mark.asyncio
async def test_processor_approval_sends_reply_once(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()
    sender = FakeGmailSender()
    processor = EmailProcessor(
        db=db,
        memory=MemoryStore(db),
        analyzer=FakeAnalyzer(
            EmailAnalysis(
                intent="legal",
                urgency="high",
                risk_score=88,
                requires_approval=True,
                never_reply=False,
                confidence=0.82,
                summary="Legal contract question",
                suggested_reply="Approved reply",
                reasons=["legal"],
                tone="formal",
            )
        ),
        gmail_sender=sender,
        telegram=FakeTelegram(),
    )
    await processor.process_email(make_email())
    approval = db.get_pending_approval_by_email("gmail-1")

    assert approval is not None
    assert await processor.begin_approval(approval.id) is True
    assert sender.sent == []
    preview = await processor.preview_approved_reply(approval.id, style="normal")
    assert await processor.send_previewed_reply(approval.id) is True
    assert await processor.send_previewed_reply(approval.id) is False
    assert sender.sent == [("gmail-1", preview)]


@pytest.mark.asyncio
async def test_preview_send_failure_keeps_approval_retryable(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()
    processor = EmailProcessor(
        db=db,
        memory=MemoryStore(db),
        analyzer=FakeAnalyzer(
            EmailAnalysis(
                intent="collaboration",
                urgency="normal",
                risk_score=75,
                requires_approval=True,
                never_reply=False,
                confidence=0.9,
                summary="Collaboration request",
                suggested_reply="Hi Person,\n\nThanks for the note. Tuesday at 6 PM works for me.\n\nRegards,\nNS",
                reasons=["professional"],
                tone="normal",
            )
        ),
        gmail_sender=FailingGmailSender(),
        telegram=FakeTelegram(),
    )
    await processor.process_email(make_email(subject="Collaboration", body="Can we discuss Tuesday at 6 PM?"))
    approval = db.get_pending_approval_by_email("gmail-1")

    assert approval is not None
    assert await processor.begin_approval(approval.id) is True
    await processor.preview_approved_reply(approval.id, style="normal")
    with pytest.raises(RuntimeError, match="gmail unavailable"):
        await processor.send_previewed_reply(approval.id)

    assert db.get_approval_status(approval.id) == "send_failed"
    assert db.get_pending_approval(approval.id) is not None


def test_snooze_approval_does_not_start_edit_session(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()
    processor = EmailProcessor(
        db=db,
        memory=MemoryStore(db),
        analyzer=FakeAnalyzer(None),
        gmail_sender=FakeGmailSender(),
        telegram=FakeTelegram(),
    )
    email = make_email(gmail_id="gmail-snooze")
    db.record_email(email, "pending_approval")
    approval = db.create_approval(email, "summary", "draft", 60)

    assert processor.snooze_approval(approval.id, option="tonight") is True

    assert db.get_approval_status(approval.id) == "snoozed"
    assert db.get_editing_approval() is None


def test_risk_analysis_uses_current_preview_draft(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()
    processor = EmailProcessor(
        db=db,
        memory=MemoryStore(db),
        analyzer=FakeAnalyzer(None),
        gmail_sender=FakeGmailSender(),
        telegram=FakeTelegram(),
    )
    email = make_email(gmail_id="gmail-risk")
    db.record_email(email, "pending_approval")
    approval = db.create_approval(email, "summary", "old draft", 60)
    assert db.set_approval_draft(approval.id, "new preview draft", "formal")

    text = processor.risk_analysis_text(approval.id)

    assert "new preview draft" in text
    assert "old draft" not in text


@pytest.mark.asyncio
async def test_regenerate_preserves_approval_session_tone_and_preview_state(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()
    telegram = FakeTelegram()
    processor = EmailProcessor(
        db=db,
        memory=MemoryStore(db),
        analyzer=FakeAnalyzer(
            EmailAnalysis(
                intent="collaboration",
                urgency="normal",
                risk_score=72,
                requires_approval=True,
                never_reply=False,
                confidence=0.9,
                summary="Seno collaboration",
                suggested_reply="Hello,\nThank you for reaching out",
                reasons=["collaboration"],
                tone="formal",
            )
        ),
        gmail_sender=FakeGmailSender(),
        telegram=telegram,
    )
    email = make_email(
        subject="AI executive assistant opportunity",
        body=(
            "Hi NS, I reviewed Seno and the workflow architecture. "
            "Could we connect Tuesday or Wednesday evening around 6 PM IST "
            "to discuss collaboration and deployment strategy?"
        ),
    )
    await processor.process_email(email)
    approval = db.get_pending_approval_by_email("gmail-1")

    assert approval is not None
    original_approval_count = len(telegram.requests)
    preview = await processor.preview_approved_reply(approval.id, style="formal")
    regenerated = await processor.regenerate_reply(approval.id)
    refreshed = db.get_pending_approval(approval.id)

    assert regenerated
    assert regenerated != preview
    assert refreshed is not None
    refreshed_approval, _ = refreshed
    assert refreshed_approval.id == approval.id
    assert refreshed_approval.status == "draft_preview"
    assert refreshed_approval.selected_style == "formal"
    assert refreshed_approval.final_reply == regenerated
    assert len(telegram.requests) == original_approval_count


@pytest.mark.asyncio
async def test_quick_tone_regenerate_preserves_session_with_explicit_strategy(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()
    processor = EmailProcessor(
        db=db,
        memory=MemoryStore(db),
        analyzer=FakeAnalyzer(
            EmailAnalysis(
                intent="collaboration",
                urgency="normal",
                risk_score=72,
                requires_approval=True,
                never_reply=False,
                confidence=0.9,
                summary="Seno collaboration",
                suggested_reply="Hello,\nThank you for reaching out",
                reasons=["collaboration"],
                tone="formal",
            )
        ),
        gmail_sender=FakeGmailSender(),
        telegram=FakeTelegram(),
    )
    await processor.process_email(make_email(subject="Seno collaboration", body="Can we discuss Seno next Tuesday at 6 PM?"))
    approval = db.get_pending_approval_by_email("gmail-1")

    assert approval is not None
    regenerated = await processor.regenerate_reply(
        approval.id,
        style="normal",
        strategy="concise_direct",
        reason="quick_tone_short",
    )
    refreshed = db.get_pending_approval(approval.id)

    assert regenerated
    assert refreshed is not None
    refreshed_approval, _ = refreshed
    assert refreshed_approval.id == approval.id
    assert refreshed_approval.status == "draft_preview"
    assert refreshed_approval.selected_style == "normal"


@pytest.mark.asyncio
async def test_processor_has_no_legacy_direct_approved_send_path(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()
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
        gmail_sender=FakeGmailSender(),
        telegram=FakeTelegram(),
    )

    assert not hasattr(processor, "send_approved_reply")


@pytest.mark.asyncio
async def test_handle_approval_requires_existing_preview_and_sends_exact_preview(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()
    sender = FakeGmailSender()
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
                summary="Coffee plan",
                suggested_reply="Coffee at 4 PM works. See you then.",
                reasons=["coffee"],
                tone="friendly",
            )
        ),
        gmail_sender=sender,
        telegram=FakeTelegram(),
    )
    await processor.process_email(make_email(subject="Coffee", body="coffee at 4 PM?"))
    approval = db.get_pending_approval_by_email("gmail-1")

    assert approval is not None
    assert await processor.handle_approval(approval.id, approved=True) is False
    assert sender.sent == []
    assert await processor.begin_approval(approval.id) is True
    preview = await processor.preview_approved_reply(approval.id, style="friendly")
    assert await processor.handle_approval(approval.id, approved=True) is True
    assert sender.sent == [("gmail-1", preview)]


@pytest.mark.asyncio
async def test_processor_human_approved_dinner_invitation_can_be_conversational(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()
    sender = FakeGmailSender()
    processor = EmailProcessor(
        db=db,
        memory=MemoryStore(db),
        analyzer=FakeAnalyzer(
            EmailAnalysis(
                intent="social_invitation",
                urgency="normal",
                risk_score=72,
                requires_approval=True,
                never_reply=False,
                confidence=0.9,
                summary="Dinner invitation",
                suggested_reply="Sounds great, see you there.",
                reasons=["dinner invitation"],
                tone="friendly",
            )
        ),
        gmail_sender=sender,
        telegram=FakeTelegram(),
    )
    await processor.process_email(make_email(subject="Dinner tonight", body="Let's head to dinner tonight."))
    approval = db.get_pending_approval_by_email("gmail-1")

    assert approval is not None
    assert await processor.begin_approval(approval.id) is True
    preview = await processor.preview_approved_reply(approval.id, style="friendly")
    assert await processor.send_previewed_reply(approval.id) is True
    assert "Hey" in sender.sent[0][1]
    assert "Sounds great, see you there." in sender.sent[0][1]
    assert sender.sent[0][1] == preview


@pytest.mark.asyncio
async def test_processor_human_approved_casual_meeting_can_be_conversational(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()
    sender = FakeGmailSender()
    processor = EmailProcessor(
        db=db,
        memory=MemoryStore(db),
        analyzer=FakeAnalyzer(
            EmailAnalysis(
                intent="casual_meeting",
                urgency="normal",
                risk_score=68,
                requires_approval=True,
                never_reply=False,
                confidence=0.9,
                summary="Casual campus meeting",
                suggested_reply="6 PM works. See you then.",
                reasons=["casual meeting at campus"],
                tone="friendly",
            )
        ),
        gmail_sender=sender,
        telegram=FakeTelegram(),
    )
    await processor.process_email(make_email(subject="Meeting at UVCE", body="Meeting at campus around 6 PM?"))
    approval = db.get_pending_approval_by_email("gmail-1")

    assert approval is not None
    assert await processor.begin_approval(approval.id) is True
    preview = await processor.preview_approved_reply(approval.id, style="friendly")
    assert await processor.send_previewed_reply(approval.id) is True
    assert "6 PM works. See you then." in sender.sent[0][1]
    assert sender.sent[0][1] == preview


@pytest.mark.asyncio
async def test_processor_human_approved_robotic_uvce_reply_is_natural(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()
    sender = FakeGmailSender()
    processor = EmailProcessor(
        db=db,
        memory=MemoryStore(db),
        analyzer=FakeAnalyzer(
            EmailAnalysis(
                intent="casual_meeting",
                urgency="normal",
                risk_score=68,
                requires_approval=True,
                never_reply=False,
                confidence=0.9,
                summary="Casual UVCE meeting",
                suggested_reply="A follow-up confirmation will be shared separately.",
                reasons=["casual meeting at UVCE"],
                tone="neutral",
            )
        ),
        gmail_sender=sender,
        telegram=FakeTelegram(),
    )
    await processor.process_email(make_email(subject="Meeting at UVCE", body="hi, meeting at 6:00 PM about UVCE"))
    approval = db.get_pending_approval_by_email("gmail-1")

    assert approval is not None
    assert await processor.begin_approval(approval.id) is True
    preview = await processor.preview_approved_reply(approval.id, style="friendly")
    assert await processor.send_previewed_reply(approval.id) is True
    assert "6:00 PM works. See you then." in sender.sent[0][1]
    assert sender.sent[0][1] == preview


@pytest.mark.asyncio
async def test_processor_human_approved_movie_plan_is_natural(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()
    sender = FakeGmailSender()
    processor = EmailProcessor(
        db=db,
        memory=MemoryStore(db),
        analyzer=FakeAnalyzer(
            EmailAnalysis(
                intent="social_plan",
                urgency="normal",
                risk_score=62,
                requires_approval=True,
                never_reply=False,
                confidence=0.9,
                summary="Movie plan",
                suggested_reply="Your message has been received. A follow-up confirmation will be shared separately.",
                reasons=["movie plan"],
                tone="friendly",
            )
        ),
        gmail_sender=sender,
        telegram=FakeTelegram(),
    )
    await processor.process_email(make_email(subject="Movie plan", body="movie this weekend?"))
    approval = db.get_pending_approval_by_email("gmail-1")

    assert approval is not None
    assert await processor.begin_approval(approval.id) is True
    preview = await processor.preview_approved_reply(approval.id, style="friendly")
    assert await processor.send_previewed_reply(approval.id) is True
    sent_body = sender.sent[0][1]
    assert sent_body == preview
    assert "ai-assisted" not in sent_body.lower()
    assert "follow-up confirmation" not in sent_body.lower()
    assert "sounds good" in sent_body.lower()


@pytest.mark.asyncio
async def test_processor_human_approved_payment_reply_still_restricted(tmp_path):
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
                confidence=0.88,
                summary="Payment confirmation",
                suggested_reply="I confirm payment has been sent.",
                reasons=["payment"],
                tone="formal",
            )
        ),
        gmail_sender=FakeGmailSender(),
        telegram=FakeTelegram(),
    )
    await processor.process_email(make_email(subject="Payment confirmation", body="Please confirm payment for invoice 10."))
    approval = db.get_pending_approval_by_email("gmail-1")

    assert approval is not None
    assert await processor.begin_approval(approval.id) is True
    with pytest.raises(ReplySafetyError):
        await processor.preview_approved_reply(approval.id, style="friendly")


@pytest.mark.asyncio
async def test_processor_rejects_expired_approval(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()
    sender = FakeGmailSender()
    processor = EmailProcessor(
        db=db,
        memory=MemoryStore(db),
        analyzer=FakeAnalyzer(
            EmailAnalysis(
                intent="legal",
                urgency="high",
                risk_score=88,
                requires_approval=True,
                never_reply=False,
                confidence=0.82,
                summary="Legal contract question",
                suggested_reply="Approved reply",
                reasons=["legal"],
                tone="formal",
            )
        ),
        gmail_sender=sender,
        telegram=FakeTelegram(),
    )
    await processor.process_email(make_email())
    approval = db.get_pending_approval_by_email("gmail-1")
    assert approval is not None
    with db.connect() as conn:
        conn.execute("UPDATE approvals SET expires_at = datetime('now', '-1 minute') WHERE id = ?", (approval.id,))

    assert await processor.begin_approval(approval.id) is False
    assert sender.sent == []


@pytest.mark.asyncio
async def test_processor_rejects_approval_without_sending(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()
    sender = FakeGmailSender()
    processor = EmailProcessor(
        db=db,
        memory=MemoryStore(db),
        analyzer=FakeAnalyzer(
            EmailAnalysis(
                intent="legal",
                urgency="high",
                risk_score=88,
                requires_approval=True,
                never_reply=False,
                confidence=0.82,
                summary="Legal contract question",
                suggested_reply="Approved reply",
                reasons=["legal"],
                tone="formal",
            )
        ),
        gmail_sender=sender,
        telegram=FakeTelegram(),
    )
    await processor.process_email(make_email())
    approval = db.get_pending_approval_by_email("gmail-1")

    assert approval is not None
    assert await processor.reject_approval(approval.id) is True
    assert sender.sent == []


@pytest.mark.asyncio
async def test_processor_delete_email_marks_handled_and_archives(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()
    manager = FakeGmailManager()
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
                confidence=0.82,
                summary="Social plan",
                suggested_reply="Sounds good.",
                reasons=["social"],
                tone="friendly",
            )
        ),
        gmail_sender=FakeGmailSender(),
        telegram=FakeTelegram(),
        gmail_manager=manager,
    )
    await processor.process_email(make_email(subject="Dinner", body="dinner tonight?"))
    approval = db.get_pending_approval_by_email("gmail-1")

    assert approval is not None
    assert await processor.delete_email(approval.id) is True
    assert manager.archived == ["gmail-1"]
    assert db.is_processed("gmail-1") is True


@pytest.mark.asyncio
async def test_processor_edit_reply_sends_custom_text(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()
    sender = FakeGmailSender()
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
                confidence=0.82,
                summary="Social plan",
                suggested_reply="Sounds good.",
                reasons=["social"],
                tone="friendly",
            )
        ),
        gmail_sender=sender,
        telegram=FakeTelegram(),
    )
    await processor.process_email(make_email(subject="Coffee", body="coffee tomorrow?"))
    approval = db.get_pending_approval_by_email("gmail-1")

    assert approval is not None
    assert processor.start_edit_reply(approval.id) is True
    assert await processor.send_edited_reply("Coffee tomorrow works. See you then.") is True
    assert sender.sent == [("gmail-1", "Coffee tomorrow works. See you then.\n\nSent via Seno, NS's executive communication assistant.")]


@pytest.mark.asyncio
async def test_processor_formal_normal_friendly_styles(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()
    sender = FakeGmailSender()
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
                confidence=0.82,
                summary="Breakfast plan",
                suggested_reply="Breakfast at 9 AM works. See you then.",
                reasons=["breakfast"],
                tone="friendly",
            )
        ),
        gmail_sender=sender,
        telegram=FakeTelegram(),
    )

    for gmail_id, style, expected in [
        ("formal", "formal", "Dear"),
        ("normal", "normal", "Hi Person,"),
        ("friendly", "friendly", "Hey"),
    ]:
        await processor.process_email(make_email(gmail_id=gmail_id, subject="Breakfast", body="breakfast at 9 AM?"))
        approval = db.get_pending_approval_by_email(gmail_id)
        assert approval is not None
        assert await processor.begin_approval(approval.id) is True
        preview = await processor.preview_approved_reply(approval.id, style=style)
        assert await processor.send_previewed_reply(approval.id) is True
        assert expected in sender.sent[-1][1]
        assert sender.sent[-1][1] == preview
