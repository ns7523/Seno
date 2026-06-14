import pytest

from app.models.email import ApprovalRequest, EmailMessage
from app.telegram.bot import TelegramBot


class FakeHTTPClient:
    def __init__(self):
        self.requests = []

    async def post(self, url, json=None, timeout=None):
        self.requests.append((url, json, timeout))
        return {"ok": True}


@pytest.mark.asyncio
async def test_telegram_approval_request_contains_inline_buttons():
    client = FakeHTTPClient()
    bot = TelegramBot(token="token", chat_id="123", http_client=client)
    approval = ApprovalRequest(
        id=7,
        email=EmailMessage(
            gmail_id="gmail-1",
            thread_id="thread-1",
            sender="person@example.com",
            subject="Important",
            body="Please review this.",
            timestamp=None,
        ),
        summary="Needs review",
        suggested_reply="I will check.",
        risk_score=77,
    )

    await bot.send_approval_request(approval)

    payload = client.requests[0][1]
    assert payload["text"].startswith("Executive Brief")
    assert "person@example.com" in payload["text"]
    assert "Confidence: Low" in payload["text"]
    assert "Risk:" not in payload["text"]
    assert "Urgency:" not in payload["text"]
    assert "Routing:" not in payload["text"]
    assert payload["reply_markup"]["inline_keyboard"][0][0]["callback_data"] == "approve:7"
    assert payload["reply_markup"]["inline_keyboard"][0][1]["callback_data"] == "reject:7"
    assert payload["reply_markup"]["inline_keyboard"][1][0]["callback_data"] == "more:7"
    flattened = [
        button["callback_data"]
        for row in payload["reply_markup"]["inline_keyboard"]
        for button in row
    ]
    assert "edit:7" not in flattened
    assert "delete:7" not in flattened
    assert "full:7" not in flattened
    assert "risk:7" not in flattened


@pytest.mark.asyncio
async def test_telegram_approval_brief_deduplicates_and_hides_noisy_signals():
    client = FakeHTTPClient()
    bot = TelegramBot(token="token", chat_id="123", http_client=client)
    approval = ApprovalRequest(
        id=7,
        email=EmailMessage(
            gmail_id="gmail-1",
            thread_id="thread-1",
            sender="Recruiter <recruiter@example.com>",
            subject="Project discussion",
            body="Can we connect tomorrow?",
            timestamp=None,
            selected_sender_alias="contact@nsakash.in",
        ),
        summary="Recruiter wants to connect tomorrow about a project collaboration.",
        suggested_reply="Thanks.",
        risk_score=55,
        category="General",
        priority="Executive Attention Required",
        reply_recommendation="Preview and approve a contextual response before sending.",
        confidence=0.8,
        risk_explanation=[
            "neutral low-risk context: hi",
            "executive attention signal",
            "availability",
            "available",
            "collaboration",
            "collaboration",
            "connect",
        ],
    )

    await bot.send_approval_request(approval)

    text = client.requests[0][1]["text"]
    assert "Executive Brief" in text
    assert "Key Signals" in text
    assert text.count("Scheduling intent detected") == 1
    assert text.count("Collaboration request") == 1
    assert text.count("Networking opportunity") == 1
    assert "neutral low-risk" not in text
    assert "executive attention signal" not in text
    assert "Confidence: High" in text
    assert "Risk: 55/100" not in text


@pytest.mark.asyncio
async def test_telegram_debug_mode_keeps_diagnostics_available():
    client = FakeHTTPClient()
    bot = TelegramBot(token="token", chat_id="123", http_client=client, debug_workflow=True)
    approval = ApprovalRequest(
        id=7,
        email=EmailMessage(
            gmail_id="gmail-1",
            thread_id="thread-1",
            sender="person@example.com",
            subject="Important",
            body="Please review this.",
            timestamp=None,
            original_recipient="contact@nsakash.in",
            selected_sender_alias="contact@nsakash.in",
            alias_selection_reason="matched original recipient",
        ),
        summary="Needs review",
        suggested_reply="I will check.",
        risk_score=77,
        confidence=0.91,
    )

    await bot.send_approval_request(approval)

    text = client.requests[0][1]["text"]
    assert "Diagnostics" in text
    assert "Confidence: 91%" in text
    assert "Risk: 77/100" in text
    assert "Routing: matched original recipient" in text


@pytest.mark.asyncio
async def test_telegram_approval_request_uses_defaults_when_optional_metadata_breaks():
    class BrokenApproval:
        id = 9
        email = EmailMessage(
            gmail_id="gmail-9",
            thread_id="thread-9",
            sender="person@example.com",
            subject="Needs review",
            body="Please review this long message.",
            timestamp=None,
        )
        suggested_reply = "Thanks for the context."

        @property
        def category(self):
            raise RuntimeError("metadata unavailable")

    client = FakeHTTPClient()
    bot = TelegramBot(token="token", chat_id="123", http_client=client)

    await bot.send_approval_request(BrokenApproval())

    payload = client.requests[0][1]
    assert "Executive Brief" in payload["text"]
    assert "Context: General" in payload["text"]
    assert "person@example.com" in payload["text"]
    assert payload["reply_markup"]["inline_keyboard"][0][0]["callback_data"] == "approve:9"
    assert payload["reply_markup"]["inline_keyboard"][1][0]["callback_data"] == "more:9"


@pytest.mark.asyncio
async def test_telegram_approval_request_supports_regenerated_minimal_approval_object():
    class ApprovalForTelegram:
        id = 11
        email = EmailMessage(
            gmail_id="gmail-11",
            thread_id="thread-11",
            sender="Reviewer <reviewer@example.com>",
            subject="Collaboration",
            body="Can we discuss collaboration next week?",
            timestamp=None,
            selected_sender_alias="contact@nsakash.in",
        )
        summary = "Reviewer wants to discuss a collaboration next week."
        suggested_reply = "Thanks for the context."
        risk_score = 55
        category = "General"
        urgency = "normal"
        suggested_tone = "Formal"
        priority = "Executive Attention Required"
        reply_recommendation = "Choose a tone and preview before sending."

    client = FakeHTTPClient()
    bot = TelegramBot(token="token", chat_id="123", http_client=client)

    await bot.send_approval_request(ApprovalForTelegram())

    payload = client.requests[0][1]
    assert payload["text"].startswith("Executive Brief")
    assert "Reviewer <reviewer@example.com>" in payload["text"]
    assert "Replying as: contact@nsakash.in" in payload["text"]
    assert "Key Signals" in payload["text"]
    assert "Confidence: Low" in payload["text"]
    assert payload["reply_markup"]["inline_keyboard"][1][0]["callback_data"] == "more:11"


@pytest.mark.asyncio
async def test_telegram_style_selection_contains_reply_styles():
    client = FakeHTTPClient()
    bot = TelegramBot(token="token", chat_id="123", http_client=client)

    await bot.send_style_selection(approval_id=7)

    payload = client.requests[0][1]
    assert payload["text"].startswith("Choose tone")
    assert payload["reply_markup"]["inline_keyboard"][0][0]["callback_data"] == "style:7:formal"
    assert payload["reply_markup"]["inline_keyboard"][0][1]["callback_data"] == "style:7:normal"
    assert payload["reply_markup"]["inline_keyboard"][0][2]["callback_data"] == "style:7:friendly"


@pytest.mark.asyncio
async def test_telegram_more_actions_contains_secondary_buttons_and_back():
    client = FakeHTTPClient()
    bot = TelegramBot(token="token", chat_id="123", http_client=client)

    await bot.send_more_actions(approval_id=7)

    payload = client.requests[0][1]
    assert payload["text"].startswith("More actions")
    buttons = [
        button["callback_data"]
        for row in payload["reply_markup"]["inline_keyboard"]
        for button in row
    ]
    assert buttons == [
        "menu_tone:7",
        "menu_regen:7",
        "menu_controls:7",
        "menu_snooze:7",
        "menu_info:7",
        "back:7",
    ]


@pytest.mark.asyncio
async def test_telegram_more_actions_can_include_ignore_sender_for_low_risk():
    client = FakeHTTPClient()
    bot = TelegramBot(token="token", chat_id="123", http_client=client)

    await bot.send_sender_controls(approval_id=7, include_ignore_sender=True)

    buttons = [
        button["callback_data"]
        for row in client.requests[0][1]["reply_markup"]["inline_keyboard"]
        for button in row
    ]
    assert "ignore_sender:7" in buttons


@pytest.mark.asyncio
async def test_telegram_more_actions_hides_ignore_sender_by_default():
    client = FakeHTTPClient()
    bot = TelegramBot(token="token", chat_id="123", http_client=client)

    await bot.send_sender_controls(approval_id=7)

    buttons = [
        button["callback_data"]
        for row in client.requests[0][1]["reply_markup"]["inline_keyboard"]
        for button in row
    ]
    assert "ignore_sender:7" not in buttons


@pytest.mark.asyncio
async def test_telegram_quick_tone_and_regenerate_reason_menus_are_compact():
    client = FakeHTTPClient()
    bot = TelegramBot(token="token", chat_id="123", http_client=client)

    await bot.send_quick_tone_actions(approval_id=7)
    await bot.send_regenerate_reason_menu(approval_id=7)

    quick_buttons = [
        button["callback_data"]
        for row in client.requests[0][1]["reply_markup"]["inline_keyboard"]
        for button in row
    ]
    reason_buttons = [
        button["callback_data"]
        for row in client.requests[1][1]["reply_markup"]["inline_keyboard"]
        for button in row
    ]
    assert {"qtone:7:short", "qtone:7:warm", "qtone:7:executive", "qtone:7:formal", "qtone:7:casual"} <= set(quick_buttons)
    assert {"regen_reason_apply:7:too_robotic", "regen_reason_apply:7:more_direct", "regen_reason_apply:7:stronger_negotiation"} <= set(reason_buttons)


@pytest.mark.asyncio
async def test_telegram_ignore_sender_warning_and_snooze_menu():
    client = FakeHTTPClient()
    bot = TelegramBot(token="token", chat_id="123", http_client=client)

    await bot.send_ignore_sender_warning(approval_id=7)
    await bot.send_snooze_menu(approval_id=7)

    warning = client.requests[0][1]
    snooze = client.requests[1][1]
    assert "classified as important/moderate risk" in warning["text"]
    assert warning["reply_markup"]["inline_keyboard"][0][0]["callback_data"] == "confirm_ignore_sender:7"
    buttons = [
        button["callback_data"]
        for row in snooze["reply_markup"]["inline_keyboard"]
        for button in row
    ]
    assert {"snooze:7:1h", "snooze:7:tonight", "snooze:7:tomorrow_morning", "snooze:7:monday", "snooze:7:after_meeting"} <= set(buttons)


@pytest.mark.asyncio
async def test_telegram_delete_confirmation_prompt_and_closed_message():
    client = FakeHTTPClient()
    bot = TelegramBot(token="token", chat_id="123", http_client=client)

    await bot.send_delete_confirmation_prompt(approval_id=7)
    await bot.send_delete_confirmation()

    prompt = client.requests[0][1]
    closed = client.requests[1][1]
    assert "Delete this email?" in prompt["text"]
    assert prompt["reply_markup"]["inline_keyboard"][0][0]["callback_data"] == "delete:7"
    assert prompt["reply_markup"]["inline_keyboard"][0][1]["callback_data"] == "back:7"
    assert "Email deleted." in closed["text"]
    assert "Status:\nClosed" in closed["text"]


@pytest.mark.asyncio
async def test_telegram_back_restores_primary_actions():
    client = FakeHTTPClient()
    bot = TelegramBot(token="token", chat_id="123", http_client=client)

    await bot.send_primary_actions(approval_id=7)

    payload = client.requests[0][1]
    assert payload["text"] == "Approval actions"
    assert payload["reply_markup"]["inline_keyboard"][0][0]["callback_data"] == "approve:7"
    assert payload["reply_markup"]["inline_keyboard"][0][1]["callback_data"] == "reject:7"
    assert payload["reply_markup"]["inline_keyboard"][1][0]["callback_data"] == "more:7"


@pytest.mark.asyncio
async def test_telegram_send_confirmation_and_keyboard_clear_payloads():
    client = FakeHTTPClient()
    bot = TelegramBot(token="token", chat_id="123", http_client=client)

    await bot.clear_inline_keyboard(chat_id="123", message_id="456")
    await bot.send_send_confirmation(sender_alias="contact@nsakash.in", tone="formal")

    clear_payload = client.requests[0][1]
    confirm_payload = client.requests[1][1]
    assert client.requests[0][0].endswith("/editMessageReplyMarkup")
    assert clear_payload["chat_id"] == "123"
    assert clear_payload["message_id"] == 456
    assert clear_payload["reply_markup"] == {"inline_keyboard": []}
    assert "Reply sent successfully." in confirm_payload["text"]
    assert "contact@nsakash.in" in confirm_payload["text"]
    assert "Formal" in confirm_payload["text"]
    assert "Completed" in confirm_payload["text"]


@pytest.mark.asyncio
async def test_telegram_edit_draft_preview_updates_existing_message():
    client = FakeHTTPClient()
    bot = TelegramBot(token="token", chat_id="123", http_client=client)

    edited = await bot.edit_draft_preview(chat_id="123", message_id="456", approval_id=7, draft="Fresh draft")

    assert edited is True
    assert client.requests[0][0].endswith("/editMessageText")
    payload = client.requests[0][1]
    assert payload["chat_id"] == "123"
    assert payload["message_id"] == "456"
    assert "Draft regenerated." in payload["text"]
    assert payload["reply_markup"]["inline_keyboard"][0][1]["callback_data"] == "regenerate:7"
