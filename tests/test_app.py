import httpx
from fastapi.testclient import TestClient
import pytest

from app.config import get_settings
import app.main as app_main
from app.main import create_app


def test_health_endpoint_reports_ok(tmp_path):
    app = create_app(database_url=f"sqlite:///{tmp_path / 'agent.db'}", start_scheduler=False)
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_root_endpoint_reports_agent_running(tmp_path):
    app = create_app(database_url=f"sqlite:///{tmp_path / 'agent.db'}", start_scheduler=False)
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"status": "AI Agent Running"}


def test_status_endpoint_requires_admin_secret(monkeypatch, tmp_path):
    get_settings.cache_clear()
    monkeypatch.setenv("ADMIN_SECRET", "secret")
    app = create_app(database_url=f"sqlite:///{tmp_path / 'agent.db'}", start_scheduler=False)
    with TestClient(app) as client:
        response = client.get("/status")

    assert response.status_code == 401
    get_settings.cache_clear()


def test_status_endpoint_reports_scheduler_and_config_with_admin_secret(monkeypatch, tmp_path):
    get_settings.cache_clear()
    monkeypatch.setenv("ADMIN_SECRET", "secret")
    app = create_app(database_url=f"sqlite:///{tmp_path / 'agent.db'}", start_scheduler=False)
    with TestClient(app) as client:
        response = client.get("/status", headers={"X-ADMIN-SECRET": "secret"})

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["scheduler"]["running"] is False
    assert data["gmail"]["scopes"] == [
        "https://www.googleapis.com/auth/gmail.modify",
        "https://www.googleapis.com/auth/gmail.send",
        "https://www.googleapis.com/auth/calendar",
        "https://www.googleapis.com/auth/calendar.events",
    ]
    assert data["calendar"] == {"enabled": False, "calendar_id": "primary"}
    assert "OPENAI_API_KEY" in data["integrations"]
    get_settings.cache_clear()


def test_ignored_senders_admin_list_and_unignore(monkeypatch, tmp_path):
    get_settings.cache_clear()
    monkeypatch.setenv("ADMIN_SECRET", "secret")
    app = create_app(database_url=f"sqlite:///{tmp_path / 'agent.db'}", start_scheduler=False)
    headers = {"X-ADMIN-SECRET": "secret"}
    with TestClient(app) as client:
        app.state.db.ignore_sender("alerts@example.com", reason="test")
        listed = client.get("/admin/ignored-senders", headers=headers)
        removed = client.delete("/admin/ignored-senders/alerts@example.com", headers=headers)
        listed_again = client.get("/admin/ignored-senders", headers=headers)

    assert listed.status_code == 200
    assert listed.json()["ignored_senders"][0]["email"] == "alerts@example.com"
    assert removed.status_code == 200
    assert removed.json() == {"ok": True, "sender": "alerts@example.com", "removed": True}
    assert listed_again.json()["ignored_senders"] == []
    get_settings.cache_clear()


def test_settings_expose_configured_sender_aliases(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("SENDER_ALIASES", "contact@nsakash.in, developer@nsakash.in, craftiq@nsakash.in")
    monkeypatch.setenv("DEFAULT_SENDER_ALIAS", "developer@nsakash.in")
    monkeypatch.setenv("DEFAULT_FROM_EMAIL", "contact@nsakash.in")
    monkeypatch.setenv("REPLY_FROM_ORIGINAL_RECIPIENT", "true")

    settings = get_settings()

    assert settings.sender_alias_list == ["contact@nsakash.in", "developer@nsakash.in", "craftiq@nsakash.in"]
    assert settings.default_sender_alias == "developer@nsakash.in"
    assert settings.default_from_email == "contact@nsakash.in"
    assert settings.reply_from_original_recipient is True
    get_settings.cache_clear()


def test_gmail_diagnostics_reports_not_configured_without_credentials(monkeypatch, tmp_path):
    get_settings.cache_clear()
    monkeypatch.setenv("GMAIL_CLIENT_SECRETS_FILE", "")
    monkeypatch.setenv("GMAIL_TOKEN_FILE", str(tmp_path / "missing-token.json"))
    monkeypatch.setenv("ADMIN_SECRET", "secret")
    app = create_app(database_url=f"sqlite:///{tmp_path / 'agent.db'}", start_scheduler=False)
    with TestClient(app) as client:
        response = client.get("/diagnostics/gmail", headers={"X-ADMIN-SECRET": "secret"})

    assert response.status_code == 200
    assert response.json()["status"] == "not_configured"
    get_settings.cache_clear()


def test_default_development_app_starts_without_external_credentials(monkeypatch, tmp_path):
    get_settings.cache_clear()
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'agent.db'}")
    monkeypatch.delenv("GMAIL_CLIENT_SECRETS_FILE", raising=False)

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ADMIN_SECRET", "test-admin")
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)

    app = create_app()
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    get_settings.cache_clear()


def test_scheduler_starts_single_polling_job_when_monitor_enabled(monkeypatch, tmp_path):
    class FakeMonitor:
        async def poll_once(self):
            return None

    get_settings.cache_clear()
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("ENABLE_INBOX_MONITOR", "true")
    monkeypatch.setenv("ADMIN_SECRET", "secret")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'agent.db'}")
    monkeypatch.setattr(app_main, "build_components", lambda settings, db: {"monitor": FakeMonitor()})

    app = create_app()
    with TestClient(app) as client:
        response = client.get("/status", headers={"X-ADMIN-SECRET": "secret"})

    data = response.json()
    assert response.status_code == 200
    assert data["scheduler"]["running"] is True
    assert data["scheduler"]["poll_interval_seconds"] == 60
    assert [job["id"] for job in data["scheduler"]["jobs"]] == ["gmail-inbox-poll"]
    get_settings.cache_clear()


def test_production_startup_requires_telegram_webhook_secret(monkeypatch, tmp_path):
    get_settings.cache_clear()
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("ENABLE_INBOX_MONITOR", "false")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'agent.db'}")
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GMAIL_CLIENT_SECRETS_FILE", "client_secret.json")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
    
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat")
    monkeypatch.setenv("ADMIN_SECRET", "test-admin")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "")

    app = create_app()
    with pytest.raises(RuntimeError, match="TELEGRAM_WEBHOOK_SECRET"):
        with TestClient(app):
            pass
    get_settings.cache_clear()


def test_telegram_webhook_rejects_missing_invalid_and_replayed_secret(monkeypatch, tmp_path):
    get_settings.cache_clear()


def test_telegram_webhook_routes_approval_style_and_delete_callbacks(monkeypatch, tmp_path):
    class FakeProcessor:
        def __init__(self):
            self.events = []

        async def begin_approval(self, approval_id):
            self.events.append(("begin", approval_id))
            return True

        async def reject_approval(self, approval_id):
            self.events.append(("reject", approval_id))
            return True

        async def preview_approved_reply(self, approval_id, style):
            self.events.append(("style", approval_id, style))
            return "draft"

        def start_edit_reply(self, approval_id):
            self.events.append(("edit", approval_id))
            return True

        async def delete_email(self, approval_id):
            self.events.append(("delete", approval_id))
            return True

        async def regenerate_reply(self, approval_id, style=None, strategy=None, reason=None):
            self.events.append(("regenerate", approval_id, style, strategy, reason))
            return True

        def risk_analysis_text(self, approval_id):
            self.events.append(("risk", approval_id))
            return "risk details"

        def full_email_text(self, approval_id):
            self.events.append(("full", approval_id))
            return "full email"

    class FakeTelegram:
        def __init__(self):
            self.style_requests = []
            self.messages = []
            self.callbacks = []

        async def send_style_selection(self, approval_id):
            self.style_requests.append(approval_id)

        async def send_message(self, text):
            self.messages.append(text)

        async def answer_callback(self, callback_query_id, text):
            self.callbacks.append((callback_query_id, text))

    get_settings.cache_clear()
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "telegram-secret")
    processor = FakeProcessor()
    telegram = FakeTelegram()
    app = create_app(database_url=f"sqlite:///{tmp_path / 'agent.db'}", start_scheduler=False)
    with TestClient(app) as client:
        app.state.components["processor"] = processor
        app.state.components["telegram"] = telegram
        headers = {"X-Telegram-Bot-Api-Secret-Token": "telegram-secret"}
        for update_id, data in enumerate(
            [
                "approve:7",
                "style:7:friendly",
                "edit:7",
                "confirm_delete:7",
                "delete:7",
                "regenerate:7",
                "qtone:7:short",
                "regen_reason_apply:7:too_robotic",
                "risk:7",
            ],
            start=100,
        ):
            response = client.post(
                "/telegram/webhook",
                json={"update_id": update_id, "callback_query": {"id": f"cb-{update_id}", "data": data}},
                headers=headers,
            )
            assert response.status_code == 200

    assert processor.events == [
        ("begin", 7),
        ("style", 7, "friendly"),
        ("edit", 7),
        ("delete", 7),
        ("regenerate", 7, None, None, None),
        ("regenerate", 7, "normal", "concise_direct", "quick_tone_short"),
        ("regenerate", 7, None, "warmer_executive", "too_robotic"),
        ("risk", 7),
    ]
    assert telegram.style_requests == [7]
    assert any("custom edited reply" in message.lower() for message in telegram.messages)
    assert any("risk details" in message for message in telegram.messages)
    get_settings.cache_clear()


def test_telegram_webhook_more_and_back_callbacks_restore_clean_menus(monkeypatch, tmp_path):
    from app.models.email import EmailMessage

    class FakeProcessor:
        pass

    class FakeTelegram:
        def __init__(self):
            self.more = []
            self.primary = []
            self.callbacks = []

        async def answer_callback(self, callback_query_id, text):
            self.callbacks.append((callback_query_id, text))

        async def send_more_actions(self, approval_id):
            self.more.append(approval_id)

        async def send_primary_actions(self, approval_id):
            self.primary.append(approval_id)

        async def send_message(self, text):
            self.primary.append(("message", text))

    get_settings.cache_clear()
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "telegram-secret")
    telegram = FakeTelegram()
    app = create_app(database_url=f"sqlite:///{tmp_path / 'agent.db'}", start_scheduler=False)
    headers = {"X-Telegram-Bot-Api-Secret-Token": "telegram-secret"}
    with TestClient(app) as client:
        email = EmailMessage(
            gmail_id="gmail-menu",
            thread_id="thread-menu",
            sender="Ravi <ravi@example.com>",
            subject="Meeting",
            body="Can we discuss this tomorrow?",
            timestamp=None,
        )
        app.state.db.record_email(email, "pending_approval")
        approval = app.state.db.create_approval(email, "summary", "draft", 70)
        app.state.components["processor"] = FakeProcessor()
        app.state.components["telegram"] = telegram

        more = client.post(
            "/telegram/webhook",
            json={"update_id": 2750, "callback_query": {"id": "cb-more", "data": f"more:{approval.id}"}},
            headers=headers,
        )
        back = client.post(
            "/telegram/webhook",
            json={"update_id": 2751, "callback_query": {"id": "cb-back", "data": f"back:{approval.id}"}},
            headers=headers,
        )

    assert more.status_code == 200
    assert back.status_code == 200
    assert telegram.more == [approval.id]
    assert telegram.primary == [approval.id]
    assert telegram.callbacks == [("cb-more", "Processing"), ("cb-back", "Processing")]
    get_settings.cache_clear()


def test_telegram_webhook_secondary_actions_route_to_processor(monkeypatch, tmp_path):
    class FakeProcessor:
        def __init__(self):
            self.events = []

        def snooze_approval(self, approval_id):
            self.events.append(("snooze", approval_id, None))
            return True

        def mark_handled(self, approval_id):
            self.events.append(("handled", approval_id))
            return True

        async def create_calendar_event(self, approval_id):
            self.events.append(("calendar", approval_id))
            return type(
                "CalendarEvent",
                (),
                {"title": "Meeting", "starts_at": "2026-05-20T18:00:00+05:30", "meet_link": "https://meet.google.com/test"},
            )()

        async def suggest_alternative_times(self, approval_id):
            self.events.append(("alt_time", approval_id))
            return ["Wednesday 6 PM", "Thursday 5 PM"]

        def ignore_sender(self, approval_id, confirmed=False):
            self.events.append(("ignore_sender", approval_id, confirmed))
            return True

        def ignore_thread(self, approval_id):
            self.events.append(("ignore_thread", approval_id))
            return True

        def pin_sender(self, approval_id):
            self.events.append(("pin_sender", approval_id))
            return True

        def auto_handle_similar(self, approval_id):
            self.events.append(("auto_handle_similar", approval_id))
            return True

    class FakeTelegram:
        def __init__(self):
            self.callbacks = []
            self.messages = []
            self.snoozed = 0
            self.more_calls = []

        async def answer_callback(self, callback_query_id, text):
            self.callbacks.append((callback_query_id, text))

        async def send_message(self, text):
            self.messages.append(text)

        async def send_snooze_confirmation(self, option=None):
            self.snoozed += 1

        async def send_more_actions(self, approval_id, include_ignore_sender=False):
            self.more_calls.append((approval_id, include_ignore_sender))

    get_settings.cache_clear()
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "telegram-secret")
    processor = FakeProcessor()
    telegram = FakeTelegram()
    app = create_app(database_url=f"sqlite:///{tmp_path / 'agent.db'}", start_scheduler=False)
    headers = {"X-Telegram-Bot-Api-Secret-Token": "telegram-secret"}
    with TestClient(app) as client:
        app.state.components["processor"] = processor
        app.state.components["telegram"] = telegram
        for update_id, action in enumerate(
            ["snooze", "handled", "calendar", "alt_time", "confirm_ignore_sender", "ignore_thread", "pin_sender", "auto_handle_similar"],
            start=2760,
        ):
            response = client.post(
                "/telegram/webhook",
                json={"update_id": update_id, "callback_query": {"id": f"cb-{action}", "data": f"{action}:7"}},
                headers=headers,
            )
            assert response.status_code == 200
            assert response.json()["handled"] is True

    assert processor.events == [
        ("snooze", 7, None),
        ("handled", 7),
        ("calendar", 7),
        ("alt_time", 7),
        ("ignore_sender", 7, True),
        ("ignore_thread", 7),
        ("pin_sender", 7),
        ("auto_handle_similar", 7),
    ]
    assert telegram.snoozed == 1
    assert any("Marked as handled" in message for message in telegram.messages)
    assert any("Calendar event created" in message for message in telegram.messages)
    assert any("Wednesday 6 PM" in message for message in telegram.messages)
    assert any("Future emails from this sender will be ignored." in message for message in telegram.messages)
    assert any("Future notifications for this thread will be ignored." in message for message in telegram.messages)
    assert any("Sender pinned." in message for message in telegram.messages)
    assert any("Similar low-risk emails will be auto-handled." in message for message in telegram.messages)
    get_settings.cache_clear()


def test_telegram_more_menu_includes_ignore_sender_for_moderate_nonprotected_sender(monkeypatch, tmp_path):
    from app.models.email import EmailMessage

    class FakeProcessor:
        pass

    class FakeTelegram:
        def __init__(self):
            self.more_calls = []

        async def answer_callback(self, callback_query_id, text):
            return None

        async def send_more_actions(self, approval_id, include_ignore_sender=False):
            self.more_calls.append((approval_id, include_ignore_sender))

    get_settings.cache_clear()
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "telegram-secret")
    telegram = FakeTelegram()
    app = create_app(database_url=f"sqlite:///{tmp_path / 'agent.db'}", start_scheduler=False)
    headers = {"X-Telegram-Bot-Api-Secret-Token": "telegram-secret"}
    with TestClient(app) as client:
        email = EmailMessage("moderate-gmail", "thread-moderate", "Alerts <alerts@example.com>", "FYI", "Moderate update", None)
        app.state.db.record_email(email, "pending_approval")
        approval = app.state.db.create_approval(email, "summary", "draft", 55)
        app.state.components["processor"] = FakeProcessor()
        app.state.components["telegram"] = telegram
        response = client.post(
            "/telegram/webhook",
            json={"update_id": 2850, "callback_query": {"id": "cb-more-ignore", "data": f"more:{approval.id}"}},
            headers=headers,
        )

    assert response.status_code == 200
    assert telegram.more_calls == [(approval.id, True)]
    get_settings.cache_clear()


def test_telegram_more_menu_hides_ignore_sender_for_protected_sender(monkeypatch, tmp_path):
    from app.models.email import EmailMessage

    class FakeProcessor:
        pass

    class FakeTelegram:
        def __init__(self):
            self.more_calls = []

        async def answer_callback(self, callback_query_id, text):
            return None

        async def send_more_actions(self, approval_id, include_ignore_sender=False):
            self.more_calls.append((approval_id, include_ignore_sender))

    get_settings.cache_clear()
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "telegram-secret")
    telegram = FakeTelegram()
    app = create_app(database_url=f"sqlite:///{tmp_path / 'agent.db'}", start_scheduler=False)
    headers = {"X-Telegram-Bot-Api-Secret-Token": "telegram-secret"}
    with TestClient(app) as client:
        email = EmailMessage("protected-gmail", "thread-protected", "NS <nsakash752003@gmail.com>", "FYI", "Protected", None)
        app.state.db.record_email(email, "pending_approval")
        approval = app.state.db.create_approval(email, "summary", "draft", 20)
        app.state.components["processor"] = FakeProcessor()
        app.state.components["telegram"] = telegram
        response = client.post(
            "/telegram/webhook",
            json={"update_id": 2851, "callback_query": {"id": "cb-more-protected", "data": f"more:{approval.id}"}},
            headers=headers,
        )

    assert response.status_code == 200
    assert telegram.more_calls == [(approval.id, False)]
    get_settings.cache_clear()


def test_telegram_ignore_sender_moderate_risk_requires_confirmation(monkeypatch, tmp_path):
    from app.models.email import EmailMessage

    class FakeProcessor:
        def __init__(self):
            self.events = []

        def ignore_sender(self, approval_id, confirmed=False):
            self.events.append((approval_id, confirmed))
            return confirmed

    class FakeTelegram:
        def __init__(self):
            self.warnings = []
            self.messages = []

        async def answer_callback(self, callback_query_id, text):
            return None

        async def send_ignore_sender_warning(self, approval_id):
            self.warnings.append(approval_id)

        async def send_message(self, text):
            self.messages.append(text)

    get_settings.cache_clear()
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "telegram-secret")
    processor = FakeProcessor()
    telegram = FakeTelegram()
    app = create_app(database_url=f"sqlite:///{tmp_path / 'agent.db'}", start_scheduler=False)
    headers = {"X-Telegram-Bot-Api-Secret-Token": "telegram-secret"}
    with TestClient(app) as client:
        email = EmailMessage("moderate-ignore", "thread-ignore", "Alerts <alerts@example.com>", "FYI", "Moderate", None)
        app.state.db.record_email(email, "pending_approval")
        approval = app.state.db.create_approval(email, "summary", "draft", 55)
        app.state.components["processor"] = processor
        app.state.components["telegram"] = telegram
        warn = client.post(
            "/telegram/webhook",
            json={"update_id": 2860, "callback_query": {"id": "cb-ignore-warn", "data": f"ignore_sender:{approval.id}"}},
            headers=headers,
        )
        confirm = client.post(
            "/telegram/webhook",
            json={"update_id": 2861, "callback_query": {"id": "cb-ignore-confirm", "data": f"confirm_ignore_sender:{approval.id}"}},
            headers=headers,
        )

    assert warn.status_code == 200
    assert confirm.status_code == 200
    assert telegram.warnings == [approval.id]
    assert processor.events == [(approval.id, True)]
    assert any("Future emails from this sender will be ignored." in message for message in telegram.messages)
    get_settings.cache_clear()


def test_telegram_webhook_duplicate_callback_id_is_idempotent(monkeypatch, tmp_path):
    class FakeProcessor:
        def __init__(self):
            self.begin_calls = 0

        async def begin_approval(self, approval_id):
            self.begin_calls += 1
            return True

    class FakeTelegram:
        def __init__(self):
            self.style_requests = []
            self.callbacks = []

        async def send_style_selection(self, approval_id):
            self.style_requests.append(approval_id)

        async def answer_callback(self, callback_query_id, text):
            self.callbacks.append((callback_query_id, text))

    get_settings.cache_clear()
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "telegram-secret")
    processor = FakeProcessor()
    telegram = FakeTelegram()
    app = create_app(database_url=f"sqlite:///{tmp_path / 'agent.db'}", start_scheduler=False)
    headers = {"X-Telegram-Bot-Api-Secret-Token": "telegram-secret"}
    with TestClient(app) as client:
        app.state.components["processor"] = processor
        app.state.components["telegram"] = telegram
        first = client.post(
            "/telegram/webhook",
            json={"update_id": 501, "callback_query": {"id": "same-callback", "data": "approve:7"}},
            headers=headers,
        )
        duplicate = client.post(
            "/telegram/webhook",
            json={"update_id": 502, "callback_query": {"id": "same-callback", "data": "approve:7"}},
            headers=headers,
        )

    assert first.status_code == 200
    assert duplicate.status_code == 200
    assert duplicate.json()["duplicate_callback"] is True
    assert processor.begin_calls == 1
    assert telegram.style_requests == [7]
    assert telegram.callbacks == [("same-callback", "Processing")]
    get_settings.cache_clear()


def test_telegram_preview_callbacks_ack_immediately_and_continue_workflow(monkeypatch, tmp_path):
    events = []

    class FakeProcessor:
        async def send_previewed_reply(self, approval_id):
            events.append(("send_entered", approval_id))
            assert ("callback_ack", "cb-send", "Processing") in events
            return True

        async def regenerate_reply(self, approval_id):
            events.append(("regenerate_entered", approval_id))
            assert ("callback_ack", "cb-regenerate", "Processing") in events
            return True

        def start_edit_reply(self, approval_id):
            events.append(("edit_entered", approval_id))
            assert ("callback_ack", "cb-edit", "Processing") in events
            return True

        def cancel_approval(self, approval_id):
            events.append(("cancel_entered", approval_id))
            assert ("callback_ack", "cb-cancel", "Processing") in events
            return True

    class FakeTelegram:
        async def answer_callback(self, callback_query_id, text):
            events.append(("callback_ack", callback_query_id, text))

        async def send_message(self, text):
            events.append(("message", text))

    get_settings.cache_clear()
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "telegram-secret")
    app = create_app(database_url=f"sqlite:///{tmp_path / 'agent.db'}", start_scheduler=False)
    headers = {"X-Telegram-Bot-Api-Secret-Token": "telegram-secret"}
    with TestClient(app) as client:
        app.state.components["processor"] = FakeProcessor()
        app.state.components["telegram"] = FakeTelegram()
        for update_id, action in enumerate(["send", "regenerate", "edit", "cancel"], start=801):
            response = client.post(
                "/telegram/webhook",
                json={"update_id": update_id, "callback_query": {"id": f"cb-{action}", "data": f"{action}:7"}},
                headers=headers,
            )
            assert response.status_code == 200
            assert response.json()["handled"] is True

    assert ("send_entered", 7) in events
    assert ("regenerate_entered", 7) in events
    assert ("edit_entered", 7) in events
    assert ("cancel_entered", 7) in events
    get_settings.cache_clear()


def test_telegram_regenerate_edits_existing_preview_without_recreating_card(monkeypatch, tmp_path):
    events = []

    class FakeProcessor:
        async def regenerate_reply(self, approval_id):
            events.append(("regenerate", approval_id))
            return "Updated contextual draft"

    class FakeTelegram:
        async def answer_callback(self, callback_query_id, text):
            events.append(("ack", callback_query_id, text))

        async def edit_draft_preview(self, *, chat_id, message_id, approval_id, draft):
            events.append(("edit_preview", chat_id, message_id, approval_id, draft))
            return True

        async def send_draft_preview(self, approval_id, draft):
            events.append(("send_preview", approval_id, draft))

    get_settings.cache_clear()
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "telegram-secret")
    app = create_app(database_url=f"sqlite:///{tmp_path / 'agent.db'}", start_scheduler=False)
    headers = {"X-Telegram-Bot-Api-Secret-Token": "telegram-secret"}
    with TestClient(app) as client:
        app.state.components["processor"] = FakeProcessor()
        app.state.components["telegram"] = FakeTelegram()
        response = client.post(
            "/telegram/webhook",
            json={
                "update_id": 901,
                "callback_query": {
                    "id": "cb-regenerate-edit",
                    "data": "regenerate:7",
                    "message": {"message_id": 456, "chat": {"id": 123}},
                },
            },
            headers=headers,
        )

    assert response.status_code == 200
    assert response.json()["handled"] is True
    assert ("edit_preview", "123", "456", 7, "Updated contextual draft") in events
    assert not any(event[0] == "send_preview" for event in events)
    get_settings.cache_clear()


def test_telegram_send_callback_clears_keyboard_and_confirms_completion(monkeypatch, tmp_path):
    from app.models.email import EmailMessage

    class FakeProcessor:
        async def send_previewed_reply(self, approval_id):
            return True

    class FakeTelegram:
        def __init__(self):
            self.callbacks = []
            self.cleared = []
            self.confirmations = []

        async def answer_callback(self, callback_query_id, text):
            self.callbacks.append((callback_query_id, text))

        async def clear_inline_keyboard(self, *, chat_id, message_id):
            self.cleared.append((chat_id, message_id))

        async def send_send_confirmation(self, *, sender_alias, tone, status="Completed"):
            self.confirmations.append((sender_alias, tone, status))

    get_settings.cache_clear()
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "telegram-secret")
    telegram = FakeTelegram()
    app = create_app(database_url=f"sqlite:///{tmp_path / 'agent.db'}", start_scheduler=False)
    headers = {"X-Telegram-Bot-Api-Secret-Token": "telegram-secret"}
    with TestClient(app) as client:
        email = EmailMessage(
            gmail_id="gmail-send-confirm",
            thread_id="thread-send-confirm",
            sender="Ravi <ravi@example.com>",
            subject="Meeting",
            body="Can we discuss this tomorrow?",
            timestamp=None,
            selected_sender_alias="contact@nsakash.in",
        )
        app.state.db.record_email(email, "pending_approval")
        approval = app.state.db.create_approval(email, "summary", "draft", 70)
        app.state.db.set_approval_draft(approval.id, "Dear Ravi,\n\nDraft\n\nBest regards,\nNS", "formal")
        app.state.components["processor"] = FakeProcessor()
        app.state.components["telegram"] = telegram
        response = client.post(
            "/telegram/webhook",
            json={
                "update_id": 806,
                "callback_query": {
                    "id": "cb-send-confirm",
                    "data": f"send:{approval.id}",
                    "message": {"message_id": 321, "chat": {"id": 123}},
                },
            },
            headers=headers,
        )

    assert response.status_code == 200
    assert response.json()["handled"] is True
    assert telegram.cleared == [("123", "321")]
    assert telegram.confirmations == [("contact@nsakash.in", "formal", "Completed")]
    get_settings.cache_clear()


def test_telegram_callback_lock_releases_after_guarded_exception(monkeypatch, tmp_path):
    class FlakyProcessor:
        def __init__(self):
            self.calls = 0

        async def send_previewed_reply(self, approval_id):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary send failure")
            return True

    class FakeTelegram:
        def __init__(self):
            self.callbacks = []

        async def answer_callback(self, callback_query_id, text):
            self.callbacks.append((callback_query_id, text))

    get_settings.cache_clear()
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "telegram-secret")
    processor = FlakyProcessor()
    telegram = FakeTelegram()
    app = create_app(database_url=f"sqlite:///{tmp_path / 'agent.db'}", start_scheduler=False)
    headers = {"X-Telegram-Bot-Api-Secret-Token": "telegram-secret"}
    with TestClient(app) as client:
        app.state.components["processor"] = processor
        app.state.components["telegram"] = telegram
        first = client.post(
            "/telegram/webhook",
            json={"update_id": 850, "callback_query": {"id": "cb-send-fail", "data": "send:7"}},
            headers=headers,
        )
        second = client.post(
            "/telegram/webhook",
            json={"update_id": 851, "callback_query": {"id": "cb-send-retry", "data": "send:7"}},
            headers=headers,
        )

    assert first.status_code == 200
    assert first.json()["ok"] is False
    assert second.status_code == 200
    assert second.json()["handled"] is True
    assert processor.calls == 2
    assert telegram.callbacks == [("cb-send-fail", "Processing"), ("cb-send-retry", "Processing")]
    get_settings.cache_clear()


def test_failed_telegram_update_and_callback_are_retryable(monkeypatch, tmp_path):
    class FlakyProcessor:
        def __init__(self):
            self.calls = 0

        async def begin_approval(self, approval_id):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary workflow failure")
            return True

    class FakeTelegram:
        def __init__(self):
            self.messages = []
            self.callbacks = []
            self.style_requests = []

        async def answer_callback(self, callback_query_id, text):
            self.callbacks.append((callback_query_id, text))

        async def send_message(self, text):
            self.messages.append(text)

        async def send_style_selection(self, approval_id):
            self.style_requests.append(approval_id)

    get_settings.cache_clear()
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "telegram-secret")
    processor = FlakyProcessor()
    telegram = FakeTelegram()
    app = create_app(database_url=f"sqlite:///{tmp_path / 'agent.db'}", start_scheduler=False)
    payload = {"update_id": 875, "callback_query": {"id": "cb-retryable", "data": "approve:7"}}
    headers = {"X-Telegram-Bot-Api-Secret-Token": "telegram-secret"}
    with TestClient(app) as client:
        app.state.components["processor"] = processor
        app.state.components["telegram"] = telegram
        first = client.post("/telegram/webhook", json=payload, headers=headers)
        second = client.post("/telegram/webhook", json=payload, headers=headers)

    assert first.status_code == 200
    assert first.json()["ok"] is False
    assert second.status_code == 200
    assert second.json()["handled"] is True
    assert processor.calls == 2
    assert telegram.style_requests == [7]
    assert any("temporary workflow issue" in message for message in telegram.messages)
    get_settings.cache_clear()


def test_durable_approval_lock_blocks_second_owner_until_release(tmp_path):
    from app.database import Database

    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()

    assert db.acquire_approval_lock(7, "owner-1", ttl_seconds=30) is True
    assert db.acquire_approval_lock(7, "owner-2", ttl_seconds=30) is False

    db.release_approval_lock(7, "owner-1")

    assert db.acquire_approval_lock(7, "owner-2", ttl_seconds=30) is True


def test_approval_debug_snapshot_reads_lock_acquired_at_without_created_at(tmp_path):
    from app.database import Database
    from app.models.email import EmailMessage

    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()
    email = EmailMessage(
        gmail_id="gmail-debug-lock",
        thread_id="thread-debug-lock",
        sender="Ravi <ravi@example.com>",
        subject="Meeting",
        body="Can we meet tomorrow?",
        timestamp=None,
    )
    db.record_email(email, "pending_approval")
    approval = db.create_approval(email, "summary", "draft", 65)
    assert db.acquire_approval_lock(approval.id, "owner-1", ttl_seconds=30)

    snapshot = db.get_approval_debug_snapshot(approval.id)

    assert snapshot["locks"][0]["owner"] == "owner-1"
    assert "acquired_at" in snapshot["locks"][0]
    assert "created_at" not in snapshot["locks"][0]


def test_stale_workflow_cleanup_expires_abandoned_rows(tmp_path):
    from app.database import Database

    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()
    db.begin_telegram_update(77)
    db.begin_telegram_callback("cb-stale", approval_id=7, action="send")
    db.acquire_approval_lock(7, "owner", ttl_seconds=30)
    with db.connect() as conn:
        conn.execute("UPDATE telegram_updates SET processing_started_at = datetime('now', '-20 minutes') WHERE update_id = 77")
        conn.execute("UPDATE telegram_callbacks SET processing_started_at = datetime('now', '-20 minutes') WHERE callback_query_id = 'cb-stale'")
        conn.execute("UPDATE approval_locks SET expires_at = datetime('now', '-1 minute') WHERE approval_id = 7")

    counts = db.cleanup_stale_workflows()

    assert counts["updates"] == 1
    assert counts["callbacks"] == 1
    assert counts["locks"] == 1


def test_approval_state_machine_rejects_invalid_direct_send_transition(tmp_path):
    from app.database import Database
    from app.models.email import EmailMessage

    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()
    email = EmailMessage(
        gmail_id="gmail-state",
        thread_id="thread-state",
        sender="Ravi <ravi@example.com>",
        subject="Meeting",
        body="Can we meet tomorrow?",
        timestamp=None,
    )
    db.record_email(email, "pending_approval")
    approval = db.create_approval(email, "summary", "draft", 65)

    assert db.decide_approval(approval.id, True) is False
    assert db.get_approval_status(approval.id) == "pending"


def test_telegram_webhook_update_replay_returns_ok_without_duplicate_action(monkeypatch, tmp_path):
    class FakeProcessor:
        def __init__(self):
            self.begin_calls = 0

        async def begin_approval(self, approval_id):
            self.begin_calls += 1
            return True

    class FakeTelegram:
        async def send_style_selection(self, approval_id):
            return None

        async def answer_callback(self, callback_query_id, text):
            return None

    get_settings.cache_clear()
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "telegram-secret")
    processor = FakeProcessor()
    app = create_app(database_url=f"sqlite:///{tmp_path / 'agent.db'}", start_scheduler=False)
    payload = {"update_id": 601, "callback_query": {"id": "cb-601", "data": "approve:7"}}
    headers = {"X-Telegram-Bot-Api-Secret-Token": "telegram-secret"}
    with TestClient(app) as client:
        app.state.components["processor"] = processor
        app.state.components["telegram"] = FakeTelegram()
        first = client.post("/telegram/webhook", json=payload, headers=headers)
        replay = client.post("/telegram/webhook", json=payload, headers=headers)

    assert first.status_code == 200
    assert replay.status_code == 200
    assert replay.json()["duplicate_update"] is True
    assert processor.begin_calls == 1
    get_settings.cache_clear()


@pytest.mark.parametrize("status_code", [400, 409])
def test_telegram_answer_callback_expired_or_conflicted_errors_are_swallowed(status_code):
    from app.telegram.bot import TelegramBot

    class BadRequestHTTPClient:
        async def post(self, url, json=None, timeout=None):
            request = httpx.Request("POST", url)
            response = httpx.Response(status_code, request=request)
            raise httpx.HTTPStatusError("Telegram callback error", request=request, response=response)

    bot = TelegramBot(token="token", chat_id="chat", http_client=BadRequestHTTPClient())

    import anyio

    anyio.run(bot.answer_callback, "expired-callback", "Recorded")


def test_telegram_webhook_internal_exception_returns_200(monkeypatch, tmp_path):
    class FailingProcessor:
        async def begin_approval(self, approval_id):
            raise RuntimeError("boom")

    class FakeTelegram:
        async def answer_callback(self, callback_query_id, text):
            return None

    get_settings.cache_clear()
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "telegram-secret")
    app = create_app(database_url=f"sqlite:///{tmp_path / 'agent.db'}", start_scheduler=False)
    with TestClient(app) as client:
        app.state.components["processor"] = FailingProcessor()
        app.state.components["telegram"] = FakeTelegram()
        response = client.post(
            "/telegram/webhook",
            json={"update_id": 701, "callback_query": {"id": "cb-701", "data": "approve:7"}},
            headers={"X-Telegram-Bot-Api-Secret-Token": "telegram-secret"},
        )

    assert response.status_code == 200
    assert response.json()["ok"] is False
    assert response.json()["error"] == "webhook_exception_handled"
    get_settings.cache_clear()


def test_telegram_callback_exception_logs_action_context_and_notifies_recovery(monkeypatch, tmp_path):
    class FailingProcessor:
        async def send_previewed_reply(self, approval_id):
            raise RuntimeError("gmail send exploded")

    class FakeTelegram:
        def __init__(self):
            self.messages = []

        async def answer_callback(self, callback_query_id, text):
            return None

        async def send_message(self, text):
            self.messages.append(text)

    exception_logs = []

    def capture_exception(message, *args, **kwargs):
        exception_logs.append((message, kwargs.get("extra", {})))

    monkeypatch.setattr(app_main.logger, "exception", capture_exception)
    get_settings.cache_clear()
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "telegram-secret")
    monkeypatch.setenv("SENO_DEBUG_WORKFLOW", "true")
    telegram = FakeTelegram()
    app = create_app(database_url=f"sqlite:///{tmp_path / 'agent.db'}", start_scheduler=False)
    with TestClient(app) as client:
        app.state.components["processor"] = FailingProcessor()
        app.state.components["telegram"] = telegram
        response = client.post(
            "/telegram/webhook",
            json={"update_id": 711, "callback_query": {"id": "cb-send-debug", "data": "send:7"}},
            headers={"X-Telegram-Bot-Api-Secret-Token": "telegram-secret"},
        )

    assert response.status_code == 200
    assert response.json()["ok"] is False
    assert any("Send could not complete" in message for message in telegram.messages)
    action_logs = [extra for message, extra in exception_logs if message == "telegram_callback_action_exception"]
    assert action_logs
    assert action_logs[-1]["action"] == "send"
    assert action_logs[-1]["callback_query_id"] == "cb-send-debug"
    assert action_logs[-1]["stage"] == "action_exception"
    get_settings.cache_clear()


def test_telegram_callback_unhandled_action_sends_stale_state_recovery(monkeypatch, tmp_path):
    class StaleProcessor:
        async def send_previewed_reply(self, approval_id):
            return False

    class FakeTelegram:
        def __init__(self):
            self.messages = []

        async def answer_callback(self, callback_query_id, text):
            return None

        async def send_message(self, text):
            self.messages.append(text)

    get_settings.cache_clear()
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "telegram-secret")
    telegram = FakeTelegram()
    app = create_app(database_url=f"sqlite:///{tmp_path / 'agent.db'}", start_scheduler=False)
    with TestClient(app) as client:
        app.state.components["processor"] = StaleProcessor()
        app.state.components["telegram"] = telegram
        response = client.post(
            "/telegram/webhook",
            json={"update_id": 712, "callback_query": {"id": "cb-send-stale", "data": "send:7"}},
            headers={"X-Telegram-Bot-Api-Secret-Token": "telegram-secret"},
        )

    assert response.status_code == 200
    assert response.json()["handled"] is False
    assert any("Send button is no longer valid" in message for message in telegram.messages)
    get_settings.cache_clear()


def test_telegram_webhook_rejects_missing_invalid_and_replayed_secret(monkeypatch, tmp_path):
    get_settings.cache_clear()
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "telegram-secret")
    app = create_app(database_url=f"sqlite:///{tmp_path / 'agent.db'}", start_scheduler=False)
    payload = {"update_id": 42, "callback_query": {"id": "cb1", "data": "approve:1"}}
    with TestClient(app) as client:
        assert client.post("/telegram/webhook", json=payload).status_code == 401
        assert client.post(
            "/telegram/webhook",
            json=payload,
            headers={"X-Telegram-Bot-Api-Secret-Token": "bad"},
        ).status_code == 401
        first = client.post(
            "/telegram/webhook",
            json=payload,
            headers={"X-Telegram-Bot-Api-Secret-Token": "telegram-secret"},
        )
        replay = client.post(
            "/telegram/webhook",
            json=payload,
            headers={"X-Telegram-Bot-Api-Secret-Token": "telegram-secret"},
        )

    assert first.status_code in {200, 503}
    assert replay.status_code == 200
    assert replay.json()["ok"] is False
    get_settings.cache_clear()


def test_telegram_webhook_replay_protection_persists_across_app_restart(monkeypatch, tmp_path):
    database_url = f"sqlite:///{tmp_path / 'agent.db'}"
    payload = {"update_id": 99, "callback_query": {"id": "cb1", "data": "approve:1"}}
    get_settings.cache_clear()
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "telegram-secret")

    first_app = create_app(database_url=database_url, start_scheduler=False)
    with TestClient(first_app) as client:
        first = client.post(
            "/telegram/webhook",
            json=payload,
            headers={"X-Telegram-Bot-Api-Secret-Token": "telegram-secret"},
        )

    second_app = create_app(database_url=database_url, start_scheduler=False)
    with TestClient(second_app) as client:
        replay = client.post(
            "/telegram/webhook",
            json=payload,
            headers={"X-Telegram-Bot-Api-Secret-Token": "telegram-secret"},
        )

    assert first.status_code in {200, 503}
    assert replay.status_code == 200
    assert replay.json()["ok"] is False
    get_settings.cache_clear()
