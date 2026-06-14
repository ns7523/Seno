import pytest

from app.ai.classifier import EmailAnalysis
from app.ai.responder import ReplySafetyError, validate_reply
from app.models.email import EmailMessage


def make_analysis(reply: str, never_reply: bool = False) -> EmailAnalysis:
    return EmailAnalysis(
        intent="social_invitation",
        urgency="normal",
        risk_score=20,
        requires_approval=False,
        never_reply=never_reply,
        confidence=0.91,
        summary="Dinner invitation",
        suggested_reply=reply,
        reasons=["low risk"],
        tone="friendly",
    )


def test_validate_reply_does_not_expose_ai_disclosure_text():
    reply = validate_reply(
        make_analysis(
            "Thank you for your message regarding dinner tonight. "
            "NS has received the invitation. A personal confirmation will follow separately if available."
        )
    )

    assert "ai-assisted" not in reply.lower()
    assert "automated" not in reply.lower()


def test_validate_reply_auto_mode_removes_misleading_future_confirmation_wording():
    reply = validate_reply(
        make_analysis(
            "NS has received your invitation. "
            "A follow-up confirmation will be shared separately if available."
        )
    )

    assert reply == "Thanks for the message. NS has received it."
    assert "follow-up confirmation" not in reply.lower()
    assert "shared separately" not in reply.lower()


@pytest.mark.parametrize(
    "unsafe_reply",
    [
        "I will attend dinner tonight.",
        "Looking forward to seeing you.",
        "See you there.",
        "I would love to join.",
        "Count me in.",
        "I confirm the meeting.",
        "I accept the invitation.",
        "I will join the call.",
        "Yes, that works.",
        "Sure, tomorrow works.",
        "That works for me.",
        "I am available at 8 PM.",
    ],
)
def test_validate_reply_blocks_social_and_legal_commitments(unsafe_reply: str):
    with pytest.raises(ReplySafetyError):
        validate_reply(make_analysis(unsafe_reply))


def test_validate_reply_strips_stale_disclosure():
    reply = validate_reply(make_analysis("This is an AI-assisted response approved by NS.\n\nYour message has been received."))

    assert reply == "Your message has been received."


def test_validate_reply_allows_human_approved_social_commitment():
    email = EmailMessage(
        gmail_id="gmail-1",
        thread_id="thread-1",
        sender="friend@example.com",
        subject="Breakfast tomorrow",
        body="Breakfast at 9:00 AM?",
        timestamp=None,
    )

    reply = validate_reply(
        make_analysis("Breakfast at 9:00 AM sounds good. See you then."),
        human_approved=True,
        original_email=email,
    )

    assert reply == "Breakfast at 9:00 AM sounds good. See you then."


def test_validate_reply_keeps_finance_restrictions_after_human_approval():
    email = EmailMessage(
        gmail_id="gmail-1",
        thread_id="thread-1",
        sender="billing@example.com",
        subject="Payment confirmation",
        body="Please confirm payment for the invoice.",
        timestamp=None,
    )

    with pytest.raises(ReplySafetyError):
        validate_reply(
            make_analysis("I confirm payment has been sent."),
            human_approved=True,
            original_email=email,
        )


def test_validate_reply_humanizes_robotic_social_reply_after_approval():
    email = EmailMessage(
        gmail_id="gmail-1",
        thread_id="thread-1",
        sender="friend@example.com",
        subject="Dinner tonight",
        body="Let's head to dinner tonight.",
        timestamp=None,
    )

    reply = validate_reply(
        make_analysis(
            "NS has received your invitation. "
            "A follow-up confirmation will be shared separately if available."
        ),
        human_approved=True,
        original_email=email,
    )

    assert "ai-assisted" not in reply.lower()
    assert "follow-up confirmation" not in reply.lower()
    assert "see you then" in reply.lower()


def test_validate_reply_humanizes_robotic_uvce_meeting_after_approval():
    email = EmailMessage(
        gmail_id="gmail-1",
        thread_id="thread-1",
        sender="friend@example.com",
        subject="Meeting at UVCE",
        body="hi, meeting at 6:00 PM about UVCE",
        timestamp=None,
    )

    reply = validate_reply(
        make_analysis("A follow-up confirmation will be shared separately."),
        human_approved=True,
        original_email=email,
    )

    assert reply == "6:00 PM works. See you then."


def test_validate_reply_humanizes_robotic_breakfast_after_approval():
    email = EmailMessage(
        gmail_id="gmail-1",
        thread_id="thread-1",
        sender="friend@example.com",
        subject="Breakfast tomorrow",
        body="breakfast at 9?",
        timestamp=None,
    )

    reply = validate_reply(
        make_analysis("Your message has been received. A follow-up confirmation will be shared separately."),
        human_approved=True,
        original_email=email,
    )

    assert reply == "Breakfast at 9 works. See you then."


@pytest.mark.parametrize(
    ("subject", "body", "expected_fragment"),
    [
        ("Coffee", "coffee tomorrow?", "Coffee sounds good"),
        ("Movie plan", "movie this weekend?", "Sounds good"),
        ("Hangout", "hangout with friends?", "Sounds good"),
        ("Campus discussion", "campus discussion after class?", "Sounds good"),
    ],
)
def test_validate_reply_humanizes_approved_social_variants(subject: str, body: str, expected_fragment: str):
    email = EmailMessage(
        gmail_id="gmail-1",
        thread_id="thread-1",
        sender="friend@example.com",
        subject=subject,
        body=body,
        timestamp=None,
    )

    reply = validate_reply(
        make_analysis("Your message has been received. A follow-up confirmation will be shared separately."),
        human_approved=True,
        original_email=email,
    )

    assert "ai-assisted" not in reply.lower()
    assert expected_fragment.lower() in reply.lower()
    assert "follow-up confirmation" not in reply.lower()
    assert "your message has been received" not in reply.lower()


def test_validate_reply_auto_mode_keeps_invites_non_committal():
    email = EmailMessage(
        gmail_id="gmail-1",
        thread_id="thread-1",
        sender="friend@example.com",
        subject="Dinner tonight",
        body="Let's get dinner tonight.",
        timestamp=None,
    )

    with pytest.raises(ReplySafetyError):
        validate_reply(make_analysis("Sounds good, see you then."), original_email=email)


@pytest.mark.parametrize(
    "unsafe_reply",
    [
        "I accept the legal agreement.",
        "I confirm payment has been sent.",
        "I approve the invoice.",
        "I accept the HR offer.",
    ],
)
def test_validate_reply_blocks_sensitive_confirmations_even_after_approval(unsafe_reply: str):
    email = EmailMessage(
        gmail_id="gmail-1",
        thread_id="thread-1",
        sender="ops@example.com",
        subject="Invoice and legal agreement",
        body="Please approve the invoice payment and contract.",
        timestamp=None,
    )

    with pytest.raises(ReplySafetyError):
        validate_reply(make_analysis(unsafe_reply), human_approved=True, original_email=email)


def test_validate_reply_does_not_relax_generic_business_meeting_after_approval():
    email = EmailMessage(
        gmail_id="gmail-1",
        thread_id="thread-1",
        sender="ops@example.com",
        subject="Meeting",
        body="Meeting tomorrow about contract terms.",
        timestamp=None,
    )

    with pytest.raises(ReplySafetyError):
        validate_reply(make_analysis("Sounds good, see you then."), human_approved=True, original_email=email)


def test_validate_reply_does_not_relax_generic_invitation_after_approval():
    email = EmailMessage(
        gmail_id="gmail-1",
        thread_id="thread-1",
        sender="events@example.com",
        subject="Invitation",
        body="You have an invitation from the business event portal.",
        timestamp=None,
    )

    analysis = EmailAnalysis(
        intent="event_invitation",
        urgency="normal",
        risk_score=35,
        requires_approval=True,
        never_reply=False,
        confidence=0.9,
        summary="Business event invitation",
        suggested_reply="Sounds good, see you then.",
        reasons=["event"],
        tone="normal",
    )

    with pytest.raises(ReplySafetyError):
        validate_reply(analysis, human_approved=True, original_email=email)


@pytest.mark.parametrize("unsafe_reply", ["I accept.", "I confirm."])
def test_validate_reply_blocks_accept_confirm_even_for_approved_social_context(unsafe_reply: str):
    email = EmailMessage(
        gmail_id="gmail-1",
        thread_id="thread-1",
        sender="friend@example.com",
        subject="Dinner tonight",
        body="Dinner tonight?",
        timestamp=None,
    )

    with pytest.raises(ReplySafetyError):
        validate_reply(make_analysis(unsafe_reply), human_approved=True, original_email=email)
