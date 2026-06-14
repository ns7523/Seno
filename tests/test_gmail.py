import pytest
import logging
import base64
from email import message_from_bytes
from pathlib import Path
from google.oauth2.credentials import Credentials

from app.gmail.auth import GmailAuth
from app.gmail.reader import GmailReader
from app.gmail.sender import GmailSender


class FakeExecute:
    def __init__(self, value):
        self.value = value

    def execute(self):
        return self.value


class FakeMessages:
    def __init__(self):
        self.sent_body = None
        self.modified = []
        self.list_kwargs = []
        self.headers = [
            {"name": "From", "value": "Alice <alice@example.com>"},
            {"name": "Subject", "value": "Hello"},
        ]

    def list(self, **kwargs):
        self.list_kwargs.append(kwargs)
        return FakeExecute({"messages": [{"id": "gmail-1"}]})

    def get(self, **kwargs):
        return FakeExecute(
            {
                "id": "gmail-1",
                "threadId": "thread-1",
                "payload": {
                    "headers": [
                        *self.headers,
                    ],
                    "body": {"data": "SGVsbG8="},
                },
            }
        )

    def send(self, **kwargs):
        self.sent_body = kwargs["body"]
        return FakeExecute({"id": "sent-1"})

    def modify(self, **kwargs):
        self.modified.append(kwargs)
        return FakeExecute({})


class PaginatedMessages(FakeMessages):
    def __init__(self):
        super().__init__()
        self.pages = {
            None: ({"messages": [{"id": "gmail-1"}, {"id": "gmail-2"}], "nextPageToken": "page-2", "resultSizeEstimate": 3}),
            "page-2": ({"messages": [{"id": "gmail-3"}], "resultSizeEstimate": 3}),
        }

    def list(self, **kwargs):
        self.list_kwargs.append(kwargs)
        return FakeExecute(self.pages.get(kwargs.get("pageToken"), {"messages": []}))

    def get(self, **kwargs):
        gmail_id = kwargs["id"]
        return FakeExecute(
            {
                "id": gmail_id,
                "threadId": f"thread-{gmail_id}",
                "payload": {
                    "headers": [
                        {"name": "From", "value": "Alice <alice@example.com>"},
                        {"name": "Subject", "value": f"Hello {gmail_id}"},
                    ],
                    "body": {"data": "SGVsbG8="},
                },
            }
        )


class FakeUsers:
    def __init__(self, messages):
        self._messages = messages

    def messages(self):
        return self._messages


class FakeGmailService:
    def __init__(self):
        self.messages_api = FakeMessages()

    def users(self):
        return FakeUsers(self.messages_api)


def test_gmail_auth_requires_client_secrets_when_no_token(tmp_path):
    auth = GmailAuth(client_secrets_file=None, token_file=str(tmp_path / "missing-token.json"))

    with pytest.raises(RuntimeError, match="GMAIL_CLIENT_SECRETS_FILE"):
        auth.get_credentials()


def test_gmail_auth_token_write_failure_does_not_crash(monkeypatch, tmp_path):
    auth = GmailAuth(client_secrets_file=None, token_file=str(tmp_path / "token.json"))
    creds = Credentials(token="access-token")

    def fail_write(*args, **kwargs):
        raise PermissionError("read-only filesystem")

    monkeypatch.setattr(Path, "write_text", fail_write)

    auth._write_token(creds)


@pytest.mark.asyncio
async def test_gmail_reader_fetches_unread_and_marks_processed(caplog):
    caplog.set_level(logging.INFO)
    service = FakeGmailService()
    reader = GmailReader(service)

    emails = await reader.fetch_unread("is:unread")
    await reader.mark_processed("gmail-1")

    assert emails[0].sender == "Alice <alice@example.com>"
    assert emails[0].body == "Hello"
    assert service.messages_api.modified[0]["body"] == {"removeLabelIds": ["UNREAD"]}
    messages = [record.getMessage() for record in caplog.records]
    assert "Checking Gmail inbox..." in messages
    assert "Unread email detected" in messages
    assert "Extracting email body" in messages


@pytest.mark.asyncio
async def test_gmail_reader_extracts_original_recipient_from_delivered_to():
    service = FakeGmailService()
    service.messages_api.headers = [
        {"name": "Delivered-To", "value": "developer@nsakash.in"},
        {"name": "To", "value": "Contact <contact@nsakash.in>"},
        {"name": "From", "value": "Alice <alice@example.com>"},
        {"name": "Subject", "value": "Hello"},
    ]

    email = (await GmailReader(service).fetch_unread("is:unread"))[0]

    assert email.original_recipient == "developer@nsakash.in"
    assert email.recipient_detection_source == "delivered-to"


@pytest.mark.asyncio
async def test_gmail_reader_logs_recipient_headers_for_forwarded_aliases(caplog):
    caplog.set_level(logging.INFO)
    service = FakeGmailService()
    service.messages_api.headers = [
        {"name": "X-Forwarded-To", "value": "developer@nsakash.in"},
        {"name": "Delivered-To", "value": "nsakash752003@gmail.com"},
        {"name": "To", "value": "NS <nsakash752003@gmail.com>"},
        {"name": "Reply-To", "value": "sender@example.com"},
        {"name": "From", "value": "Alice <alice@example.com>"},
        {"name": "Subject", "value": "Forwarded alias"},
    ]

    await GmailReader(service).fetch_unread("is:unread")

    extraction_records = [record for record in caplog.records if record.getMessage() == "Gmail message extraction complete"]
    assert extraction_records
    assert extraction_records[-1].original_recipient == "developer@nsakash.in"
    assert extraction_records[-1].recipient_detection_source == "x-forwarded-to"
    assert extraction_records[-1].recipient_headers["x-forwarded-to"] == ["developer@nsakash.in"]


@pytest.mark.asyncio
async def test_gmail_reader_resolves_forwarded_icloud_alias_recipient():
    service = FakeGmailService()
    service.messages_api.headers = [
        {"name": "X-Forwarded-To", "value": "craftiq@nsakash.in"},
        {"name": "Delivered-To", "value": "nsakash752003@gmail.com"},
        {"name": "To", "value": "NS <nsakash752003@gmail.com>"},
        {"name": "From", "value": "Designer <designer@example.com>"},
        {"name": "Subject", "value": "Branding"},
    ]

    email = (await GmailReader(service).fetch_unread("is:unread"))[0]

    assert email.original_recipient == "craftiq@nsakash.in"
    assert email.recipient_detection_source == "x-forwarded-to"


@pytest.mark.asyncio
async def test_gmail_reader_debug_pipeline_uses_broad_unread_query_and_logs_counts(caplog):
    caplog.set_level(logging.INFO)
    service = FakeGmailService()
    reader = GmailReader(service, debug_pipeline=True)

    emails = await reader.fetch_unread("is:unread in:inbox -category:promotions -category:social -in:spam")

    assert len(emails) == 1
    assert service.messages_api.list_kwargs[0]["q"] == "is:unread"
    assert any(record.getMessage() == "Gmail raw unread response loaded" for record in caplog.records)
    assert any(record.getMessage() == "Gmail inbox check complete" for record in caplog.records)


@pytest.mark.asyncio
async def test_gmail_reader_paginates_unread_backlog():
    service = FakeGmailService()
    service.messages_api = PaginatedMessages()
    reader = GmailReader(service)

    emails = await reader.fetch_unread("is:unread", max_results=2, max_pages=5)

    assert [email.gmail_id for email in emails] == ["gmail-1", "gmail-2", "gmail-3"]
    assert service.messages_api.list_kwargs[0].get("pageToken") is None
    assert service.messages_api.list_kwargs[1]["pageToken"] == "page-2"


@pytest.mark.asyncio
async def test_gmail_reader_truncates_giant_corrupt_payloads():
    class GiantMessages(FakeMessages):
        def get(self, **kwargs):
            return FakeExecute(
                {
                    "id": "gmail-giant",
                    "threadId": "thread-giant",
                    "payload": {
                        "headers": [
                            {"name": "From", "value": "Broken <broken@example.com>"},
                            {"name": "Subject", "value": "Broken payload"},
                        ],
                        "body": {"data": "not-valid-base64" + ("A" * 20000)},
                    },
                }
            )

    service = FakeGmailService()
    service.messages_api = GiantMessages()
    reader = GmailReader(service, max_body_chars=128)

    emails = await reader.fetch_unread("is:unread")

    assert emails[0].gmail_id == "gmail-giant"
    assert len(emails[0].body) <= 128


@pytest.mark.asyncio
async def test_gmail_reader_logs_malformed_and_empty_payload_decisions(caplog):
    class MixedMessages(FakeMessages):
        def list(self, **kwargs):
            self.list_kwargs.append(kwargs)
            return FakeExecute({"messages": [{"id": "malformed"}, {"id": "empty"}], "resultSizeEstimate": 2})

        def get(self, **kwargs):
            if kwargs["id"] == "malformed":
                return FakeExecute({"threadId": "missing-id", "payload": {"headers": []}})
            return FakeExecute(
                {
                    "id": "empty",
                    "threadId": "thread-empty",
                    "labelIds": ["UNREAD", "INBOX"],
                    "payload": {
                        "headers": [
                            {"name": "From", "value": "No Body <empty@example.com>"},
                            {"name": "Subject", "value": "Empty"},
                        ],
                        "body": {"data": ""},
                    },
                }
            )

    caplog.set_level(logging.INFO)
    service = FakeGmailService()
    service.messages_api = MixedMessages()
    reader = GmailReader(service)

    emails = await reader.fetch_unread("is:unread")

    assert [email.gmail_id for email in emails] == ["empty"]
    assert any(getattr(record, "skip_reason", None) == "malformed_payload" for record in caplog.records)
    assert any(getattr(record, "skip_reason", None) == "empty_body_fallback" for record in caplog.records)


@pytest.mark.asyncio
async def test_gmail_sender_preserves_thread_id():
    service = FakeGmailService()
    sender = GmailSender(service, from_email="contact@example.com")
    email = (await GmailReader(service).fetch_unread("is:unread"))[0]

    result = await sender.send_reply(email, "Thanks\n\nSent via Seno.")

    assert result == {"id": "sent-1"}
    assert service.messages_api.sent_body["threadId"] == "thread-1"
    raw = base64.urlsafe_b64decode(service.messages_api.sent_body["raw"])
    mime = message_from_bytes(raw)
    assert mime.is_multipart()
    html_parts = [part for part in mime.walk() if part.get_content_type() == "text/html"]
    assert html_parts
    html_body = html_parts[0].get_payload(decode=True).decode()
    assert "Sent via Seno." in html_body
    assert "border-top:1px solid #e5e7eb" in html_body
    assert "font-size:12px" in html_body


@pytest.mark.asyncio
@pytest.mark.parametrize("alias", ["contact@nsakash.in", "developer@nsakash.in", "craftiq@nsakash.in"])
async def test_gmail_sender_supports_configured_sender_aliases(alias):
    service = FakeGmailService()
    sender = GmailSender(
        service,
        from_email=alias,
        sender_aliases=["contact@nsakash.in", "developer@nsakash.in", "craftiq@nsakash.in"],
    )
    email = (await GmailReader(service).fetch_unread("is:unread"))[0]

    await sender.send_reply(email, "Thanks")

    raw = base64.urlsafe_b64decode(service.messages_api.sent_body["raw"])
    mime = message_from_bytes(raw)
    assert mime["From"] == alias


@pytest.mark.asyncio
async def test_gmail_sender_falls_back_when_alias_is_not_configured():
    service = FakeGmailService()
    sender = GmailSender(
        service,
        from_email="unknown@nsakash.in",
        sender_aliases=["contact@nsakash.in", "developer@nsakash.in", "craftiq@nsakash.in"],
    )
    email = (await GmailReader(service).fetch_unread("is:unread"))[0]

    await sender.send_reply(email, "Thanks")

    raw = base64.urlsafe_b64decode(service.messages_api.sent_body["raw"])
    mime = message_from_bytes(raw)
    assert mime["From"] == "contact@nsakash.in"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("recipient", "expected_from"),
    [
        ("contact@nsakash.in", "contact@nsakash.in"),
        ("developer@nsakash.in", "developer@nsakash.in"),
        ("craftiq@nsakash.in", "craftiq@nsakash.in"),
        ("nsakash752003@gmail.com", "nsakash752003@gmail.com"),
    ],
)
async def test_gmail_sender_preserves_original_recipient_identity(recipient, expected_from):
    service = FakeGmailService()
    service.messages_api.headers = [
        {"name": "Delivered-To", "value": recipient},
        {"name": "From", "value": "Alice <alice@example.com>"},
        {"name": "Subject", "value": "Hello"},
    ]
    sender = GmailSender(
        service,
        from_email="contact@nsakash.in",
        sender_aliases=["contact@nsakash.in", "developer@nsakash.in", "craftiq@nsakash.in"],
        personal_email="nsakash752003@gmail.com",
        reply_from_original_recipient=True,
    )
    email = (await GmailReader(service).fetch_unread("is:unread"))[0]

    await sender.send_reply(email, "Thanks")

    raw = base64.urlsafe_b64decode(service.messages_api.sent_body["raw"])
    mime = message_from_bytes(raw)
    assert mime["From"] == expected_from
    assert email.selected_sender_alias == expected_from
    assert "original recipient" in email.alias_selection_reason


@pytest.mark.asyncio
async def test_gmail_sender_blocks_spoofed_original_recipient_and_uses_default():
    service = FakeGmailService()
    service.messages_api.headers = [
        {"name": "Delivered-To", "value": "spoofed@example.net"},
        {"name": "From", "value": "Alice <alice@example.com>"},
        {"name": "Subject", "value": "Hello"},
    ]
    sender = GmailSender(
        service,
        from_email="contact@nsakash.in",
        sender_aliases=["contact@nsakash.in", "developer@nsakash.in", "craftiq@nsakash.in"],
        personal_email="nsakash752003@gmail.com",
        reply_from_original_recipient=True,
    )
    email = (await GmailReader(service).fetch_unread("is:unread"))[0]

    await sender.send_reply(email, "Thanks")

    raw = base64.urlsafe_b64decode(service.messages_api.sent_body["raw"])
    mime = message_from_bytes(raw)
    assert mime["From"] == "contact@nsakash.in"
    assert email.selected_sender_alias == "contact@nsakash.in"
    assert email.alias_selection_reason == "fallback_to_default_sender"


@pytest.mark.asyncio
async def test_gmail_sender_can_contextually_override_when_explicitly_enabled():
    service = FakeGmailService()
    service.messages_api.headers = [
        {"name": "Delivered-To", "value": "contact@nsakash.in"},
        {"name": "From", "value": "Alice <alice@example.com>"},
        {"name": "Subject", "value": "Python API project"},
    ]
    sender = GmailSender(
        service,
        from_email="contact@nsakash.in",
        sender_aliases=["contact@nsakash.in", "developer@nsakash.in", "craftiq@nsakash.in"],
        personal_email="nsakash752003@gmail.com",
        reply_from_original_recipient=True,
        allow_contextual_sender_override=True,
    )
    email = (await GmailReader(service).fetch_unread("is:unread"))[0]
    email.body = "Can we discuss backend code, GitHub, and API integration?"

    await sender.send_reply(email, "Thanks")

    raw = base64.urlsafe_b64decode(service.messages_api.sent_body["raw"])
    mime = message_from_bytes(raw)
    assert mime["From"] == "developer@nsakash.in"
    assert email.alias_selection_reason == "contextual_override_developer"
