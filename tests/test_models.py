from datetime import datetime, timezone

from app.models.email import EmailMessage, extract_plain_text


def test_extract_plain_text_prefers_text_plain_part():
    payload = {
        "payload": {
            "parts": [
                {
                    "mimeType": "text/html",
                    "body": {"data": "PGI-SGk8L2I-"},
                },
                {
                    "mimeType": "text/plain",
                    "body": {"data": "SGVsbG8gZnJvbSBHbWFpbA=="},
                },
            ]
        }
    }

    assert extract_plain_text(payload) == "Hello from Gmail"


def test_email_message_from_gmail_extracts_headers_and_body():
    raw = {
        "id": "msg-1",
        "threadId": "thread-1",
        "internalDate": "1700000000000",
        "labelIds": ["INBOX", "UNREAD"],
        "payload": {
            "headers": [
                {"name": "From", "value": "Alice <alice@example.com>"},
                {"name": "Subject", "value": "Meeting"},
                {"name": "Message-ID", "value": "<m1@example.com>"},
            ],
            "body": {"data": "Q2FuIHdlIG1lZXQgdG9tb3Jyb3c_"},
        },
    }

    email = EmailMessage.from_gmail(raw)

    assert email.gmail_id == "msg-1"
    assert email.sender == "Alice <alice@example.com>"
    assert email.subject == "Meeting"
    assert email.body == "Can we meet tomorrow?"
    assert email.thread_id == "thread-1"
    assert email.message_id == "<m1@example.com>"
    assert email.timestamp == datetime.fromtimestamp(1700000000, tz=timezone.utc)
