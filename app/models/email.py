import base64
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import getaddresses
from typing import Any

from app.utils.helpers import strip_html


def _decode_body(data: str | None) -> str:
    if not data:
        return ""
    padded = data + "=" * (-len(data) % 4)
    try:
        return base64.urlsafe_b64decode(padded.encode("utf-8")).decode("utf-8", errors="replace")
    except Exception:
        return ""


def _walk_parts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    parts = payload.get("parts") or []
    if not parts:
        return [payload]
    flattened: list[dict[str, Any]] = []
    for part in parts:
        flattened.extend(_walk_parts(part))
    return flattened


def extract_plain_text(raw_message: dict[str, Any]) -> str:
    payload = raw_message.get("payload", raw_message)
    parts = _walk_parts(payload)

    for part in parts:
        if part.get("mimeType") == "text/plain":
            text = _decode_body(part.get("body", {}).get("data"))
            if text.strip():
                return text.strip()

    for part in parts:
        if part.get("mimeType") == "text/html":
            text = strip_html(_decode_body(part.get("body", {}).get("data")))
            if text.strip():
                return text.strip()

    return _decode_body(payload.get("body", {}).get("data")).strip()


RECIPIENT_HEADER_PRIORITY = (
    "x-forwarded-to",
    "x-original-to",
    "x-envelope-to",
    "envelope-to",
    "delivered-to",
    "apparently-to",
    "resent-to",
    "to",
    "cc",
)


def _headers_by_name(raw: dict[str, Any]) -> dict[str, list[str]]:
    headers: dict[str, list[str]] = {}
    for item in raw.get("payload", {}).get("headers", []):
        name = item.get("name", "").lower()
        value = item.get("value", "")
        if name and value:
            headers.setdefault(name, []).append(value)
    return headers


def detect_original_recipient(raw: dict[str, Any]) -> tuple[str | None, str | None]:
    headers = _headers_by_name(raw)
    for header_name in RECIPIENT_HEADER_PRIORITY:
        values = headers.get(header_name, [])
        for _, address in getaddresses(values):
            normalized = address.strip().lower()
            if normalized:
                return normalized, header_name
    return None, None


@dataclass(slots=True)
class EmailMessage:
    gmail_id: str
    thread_id: str
    sender: str
    subject: str
    body: str
    timestamp: datetime | None
    message_id: str | None = None
    label_ids: list[str] = field(default_factory=list)
    has_attachments: bool = False
    original_recipient: str | None = None
    recipient_detection_source: str | None = None
    selected_sender_alias: str | None = None
    alias_selection_reason: str | None = None

    @classmethod
    def from_gmail(cls, raw: dict[str, Any], max_body_chars: int | None = None) -> "EmailMessage":
        header_values = _headers_by_name(raw)
        headers = {name: values[-1] for name, values in header_values.items()}
        body = extract_plain_text(raw)
        if max_body_chars:
            body = body[:max_body_chars]
        internal_date = raw.get("internalDate")
        timestamp = None
        if internal_date:
            timestamp = datetime.fromtimestamp(int(internal_date) / 1000, tz=timezone.utc)
        original_recipient, recipient_detection_source = detect_original_recipient(raw)

        return cls(
            gmail_id=raw["id"],
            thread_id=raw.get("threadId", raw["id"]),
            sender=headers.get("from", ""),
            subject=headers.get("subject", ""),
            body=body,
            timestamp=timestamp,
            message_id=headers.get("message-id"),
            label_ids=raw.get("labelIds", []),
            has_attachments=_has_attachments(raw.get("payload", {})),
            original_recipient=original_recipient,
            recipient_detection_source=recipient_detection_source,
        )


def _has_attachments(payload: dict[str, Any]) -> bool:
    for part in _walk_parts(payload):
        filename = part.get("filename")
        body = part.get("body", {})
        if filename or body.get("attachmentId"):
            return True
    return False


@dataclass(slots=True)
class ApprovalRequest:
    id: int
    email: EmailMessage
    summary: str
    suggested_reply: str
    risk_score: int
    category: str = "General"
    urgency: str = "normal"
    suggested_tone: str = "Normal"
    priority: str = "Medium"
    reply_recommendation: str = "Review and choose a reply style."
    confidence: float = 0.0
    risk_explanation: list[str] = field(default_factory=list)

    @property
    def confidence_label(self) -> str:
        return f"{round(max(0.0, min(1.0, self.confidence)) * 100)}%"

    @property
    def risk_explanation_text(self) -> str:
        return "\n".join(f"- {reason}" for reason in self.risk_explanation)
