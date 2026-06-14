from typing import Any

from app.models.email import EmailMessage
from app.utils.logger import get_logger


logger = get_logger(__name__)


class GmailReader:
    def __init__(self, service: Any, user_id: str = "me", max_body_chars: int = 8000, debug_pipeline: bool = False) -> None:
        self.service = service
        self.user_id = user_id
        self.max_body_chars = max_body_chars
        self.debug_pipeline = debug_pipeline

    async def fetch_unread(self, query: str, max_results: int = 10, max_pages: int = 10) -> list[EmailMessage]:
        effective_query = "is:unread" if self.debug_pipeline else query
        logger.info(
            "Checking Gmail inbox...",
            extra={"query": effective_query, "configured_query": query, "debug_gmail_pipeline": self.debug_pipeline},
        )
        # The Gmail query excludes spam/promotions/social before full bodies are
        # fetched. A future optimization can fetch metadata first and only call
        # format=full for messages that pass header-level filters.
        messages: list[dict[str, Any]] = []
        page_token: str | None = None
        result_size_estimate = 0
        for page_index in range(max(1, max_pages)):
            request_kwargs: dict[str, Any] = {"userId": self.user_id, "q": effective_query, "maxResults": max_results}
            if page_token:
                request_kwargs["pageToken"] = page_token
            result = self.service.users().messages().list(**request_kwargs).execute()
            page_messages = result.get("messages", [])
            result_size_estimate = max(result_size_estimate, int(result.get("resultSizeEstimate") or 0))
            messages.extend(page_messages)
            logger.info(
                "Gmail unread page loaded",
                extra={
                    "query": effective_query,
                    "page_index": page_index,
                    "page_message_count": len(page_messages),
                    "next_page_token_present": bool(result.get("nextPageToken")),
                },
            )
            page_token = result.get("nextPageToken")
            if not page_token:
                break
        message_ids = [str(item.get("id", "")) for item in messages if item.get("id")]
        logger.info(
            "Gmail raw unread response loaded",
            extra={
                "query": effective_query,
                "result_size_estimate": result_size_estimate,
                "raw_message_count": len(messages),
                "gmail_message_ids": message_ids[:25],
            },
        )
        emails: list[EmailMessage] = []
        for item in messages:
            gmail_id = item.get("id")
            thread_id = item.get("threadId")
            if not gmail_id:
                logger.warning("Skipping Gmail list item because message id is missing", extra={"skip_reason": "missing_gmail_id"})
                continue
            logger.info("Unread email detected", extra={"gmail_id": gmail_id, "thread_id": thread_id})
            raw = (
                self.service.users()
                .messages()
                .get(userId=self.user_id, id=gmail_id, format="full")
                .execute()
            )
            label_ids = raw.get("labelIds", [])
            logger.info(
                "Extracting email body",
                extra={
                    "gmail_id": gmail_id,
                    "thread_id": raw.get("threadId"),
                    "label_ids": label_ids,
                    "internal_date": raw.get("internalDate"),
                    "recipient_headers": _recipient_header_snapshot(raw),
                },
            )
            try:
                email = EmailMessage.from_gmail(raw, max_body_chars=self.max_body_chars)
            except Exception as exc:
                logger.exception(
                    "Skipping Gmail message because payload parsing failed",
                    extra={"gmail_id": gmail_id, "skip_reason": "malformed_payload", "error": str(exc)},
                )
                continue
            if not email.body.strip():
                logger.warning(
                    "Gmail message has empty extracted body; processing minimal metadata anyway",
                    extra={"gmail_id": gmail_id, "skip_reason": "empty_body_fallback", "label_ids": email.label_ids},
                )
            logger.info(
                "Gmail message extraction complete",
                extra={
                    "gmail_id": email.gmail_id,
                    "thread_id": email.thread_id,
                    "sender": email.sender,
                    "subject": email.subject,
                    "body_chars": len(email.body),
                    "label_ids": email.label_ids,
                    "has_attachments": email.has_attachments,
                    "timestamp": email.timestamp.isoformat() if email.timestamp else None,
                    "original_recipient": email.original_recipient,
                    "recipient_detection_source": email.recipient_detection_source,
                    "recipient_headers": _recipient_header_snapshot(raw),
                },
            )
            emails.append(email)
        logger.info(
            "Gmail inbox check complete",
            extra={
                "raw_message_count": len(messages),
                "parsed_email_count": len(emails),
                "after_filtering_count": len(emails),
                "after_deduplication_count": len(emails),
                "query": effective_query,
            },
        )
        return emails

    async def mark_processed(self, gmail_id: str) -> None:
        logger.info("Marking Gmail email as processed", extra={"gmail_id": gmail_id})
        self.service.users().messages().modify(
            userId=self.user_id,
            id=gmail_id,
            body={"removeLabelIds": ["UNREAD"]},
        ).execute()

    async def mark_ignored(self, gmail_id: str) -> None:
        await self.mark_processed(gmail_id)

    async def archive_email(self, email: EmailMessage) -> None:
        logger.info("Archiving Gmail email without reply", extra={"gmail_id": email.gmail_id})
        self.service.users().messages().modify(
            userId=self.user_id,
            id=email.gmail_id,
            body={"removeLabelIds": ["INBOX", "UNREAD"]},
        ).execute()


def _recipient_header_snapshot(raw: dict[str, Any]) -> dict[str, list[str]]:
    wanted = {
        "delivered-to",
        "to",
        "cc",
        "x-forwarded-to",
        "x-original-to",
        "x-envelope-to",
        "envelope-to",
        "apparently-to",
        "resent-to",
        "reply-to",
    }
    snapshot: dict[str, list[str]] = {}
    for item in raw.get("payload", {}).get("headers", []):
        name = str(item.get("name", "")).lower()
        value = str(item.get("value", ""))
        if name in wanted and value:
            snapshot.setdefault(name, []).append(value)
    return snapshot
