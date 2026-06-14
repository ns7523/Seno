import base64
import html
from dataclasses import dataclass
from email.message import EmailMessage as MIMEEmailMessage
from email.utils import parseaddr
from typing import Any

from app.branding.footer import split_seno_footer
from app.models.email import EmailMessage
from app.utils.logger import get_logger


logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SenderIdentityDecision:
    selected_sender_alias: str | None
    original_recipient: str | None
    detection_source: str | None
    reason: str


class GmailSender:
    def __init__(
        self,
        service: Any,
        user_id: str = "me",
        from_email: str | None = None,
        sender_aliases: list[str] | None = None,
        personal_email: str | None = None,
        reply_from_original_recipient: bool = False,
        allow_contextual_sender_override: bool = False,
    ) -> None:
        self.service = service
        self.user_id = user_id
        configured_aliases = [alias.strip().lower() for alias in (sender_aliases or []) if alias.strip()]
        if from_email and not configured_aliases:
            configured_aliases.insert(0, from_email.strip().lower())
        self.sender_aliases = list(dict.fromkeys(configured_aliases))
        self.personal_email = personal_email.strip().lower() if personal_email else None
        self.allowed_sender_identities = self._allowed_sender_identities(from_email)
        self.from_email = self._resolve_alias(from_email)
        self.reply_from_original_recipient = reply_from_original_recipient
        self.allow_contextual_sender_override = allow_contextual_sender_override

    async def send_reply(self, email: EmailMessage, body: str) -> dict[str, Any]:
        identity = self.route_sender_identity(email)
        logger.info(
            "Sending Gmail reply",
            extra={
                "gmail_id": email.gmail_id,
                "thread_id": email.thread_id,
                "sender_alias": identity.selected_sender_alias,
                "original_recipient": identity.original_recipient,
                "alias_selection_reason": identity.reason,
            },
        )
        _, to_address = parseaddr(email.sender)
        message = MIMEEmailMessage()
        message["To"] = to_address
        if identity.selected_sender_alias:
            message["From"] = identity.selected_sender_alias
        subject = email.subject if email.subject.lower().startswith("re:") else f"Re: {email.subject}"
        message["Subject"] = subject
        if email.message_id:
            message["In-Reply-To"] = email.message_id
            message["References"] = email.message_id
        message.set_content(body)
        message.add_alternative(_render_html_email(body), subtype="html")
        encoded = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
        result = (
            self.service.users()
            .messages()
            .send(userId=self.user_id, body={"raw": encoded, "threadId": email.thread_id})
            .execute()
        )
        logger.info("Email sent successfully", extra={"gmail_id": email.gmail_id, "thread_id": email.thread_id})
        return result

    def route_sender_identity(self, email: EmailMessage) -> SenderIdentityDecision:
        contextual_alias, contextual_reason = self._contextual_override(email)
        if contextual_alias:
            selected = contextual_alias
            reason = contextual_reason
        elif self.reply_from_original_recipient and self._is_allowed(email.original_recipient):
            selected = email.original_recipient.strip().lower() if email.original_recipient else None
            reason = "matched original recipient"
        else:
            selected = self.from_email
            reason = "fallback_to_default_sender"

        decision = SenderIdentityDecision(
            selected_sender_alias=selected,
            original_recipient=email.original_recipient,
            detection_source=email.recipient_detection_source,
            reason=reason,
        )
        email.selected_sender_alias = selected
        email.alias_selection_reason = reason
        return decision

    def _resolve_alias(self, requested_alias: str | None) -> str | None:
        if not requested_alias:
            return self.allowed_sender_identities[0] if self.allowed_sender_identities else None
        normalized = requested_alias.strip().lower()
        if normalized in self.allowed_sender_identities:
            return normalized
        logger.warning(
            "Requested sender alias is not configured; falling back to default alias",
            extra={"requested_alias": normalized, "default_alias": self.allowed_sender_identities[0] if self.allowed_sender_identities else None},
        )
        return self.allowed_sender_identities[0] if self.allowed_sender_identities else None

    def _allowed_sender_identities(self, from_email: str | None) -> list[str]:
        identities = [*self.sender_aliases]
        if self.personal_email:
            identities.append(self.personal_email)
        if from_email and not identities:
            identities.append(from_email.strip().lower())
        return list(dict.fromkeys(identity for identity in identities if identity))

    def _is_allowed(self, value: str | None) -> bool:
        return bool(value and value.strip().lower() in self.allowed_sender_identities)

    def _contextual_override(self, email: EmailMessage) -> tuple[str | None, str]:
        if not self.allow_contextual_sender_override:
            return None, ""
        text = f"{email.subject} {email.body}".lower()
        developer_terms = ("code", "coding", "developer", "github", "backend", "api", "python", "software")
        craftiq_terms = ("brand", "branding", "design", "creative", "craftiq", "logo", "visual")
        if "developer@nsakash.in" in self.allowed_sender_identities and any(term in text for term in developer_terms):
            return "developer@nsakash.in", "contextual_override_developer"
        if "craftiq@nsakash.in" in self.allowed_sender_identities and any(term in text for term in craftiq_terms):
            return "craftiq@nsakash.in", "contextual_override_craftiq"
        return None, ""


def _render_html_email(body: str) -> str:
    main_body, footer = split_seno_footer(body)
    paragraphs = [part.strip() for part in main_body.split("\n\n") if part.strip()]
    rendered = "\n".join(f"<p>{html.escape(paragraph).replace(chr(10), '<br>')}</p>" for paragraph in paragraphs)
    footer_html = ""
    if footer:
        footer_html = (
            '<div style="margin-top:22px;padding-top:12px;border-top:1px solid #e5e7eb;'
            'font-size:12px;line-height:1.45;color:#8a94a6;">'
            f"{html.escape(footer)}"
            "</div>"
        )
    return f"""<!doctype html>
<html>
  <body style="margin:0;padding:0;background:#ffffff;">
    <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;font-size:15px;line-height:1.58;color:#1f2933;max-width:640px;padding:4px 0;">
      {rendered}
      {footer_html}
    </div>
  </body>
</html>"""
