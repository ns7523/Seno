from typing import Any, Protocol

import httpx

from app.models.email import ApprovalRequest
from app.utils.logger import get_logger


logger = get_logger(__name__)


class HTTPClient(Protocol):
    async def post(self, url: str, json: dict[str, Any] | None = None, timeout: float | None = None) -> Any:
        ...


class TelegramBot:
    def __init__(
        self,
        token: str | None,
        chat_id: str | None,
        http_client: HTTPClient | None = None,
        *,
        debug_workflow: bool = False,
    ) -> None:
        self.token = token
        self.chat_id = chat_id
        self.http_client = http_client
        self.debug_workflow = debug_workflow

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    async def send_approval_request(self, approval: ApprovalRequest | Any) -> None:
        if not self.enabled:
            logger.warning("Telegram approval requested but bot is not configured", extra={"approval_id": _safe_attr(approval, "id", "unknown")})
            return
        try:
            payload = self._approval_payload(approval)
        except Exception as exc:
            logger.exception(
                "telegram_approval_card_fallback_used",
                extra={
                    "approval_id": _safe_attr(approval, "id", "unknown"),
                    "rendering_stage": "approval_payload",
                    "missing_attribute": getattr(exc, "name", None) if isinstance(exc, AttributeError) else None,
                    "approval_snapshot": _approval_snapshot(approval),
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            )
            payload = self._fallback_approval_payload(approval)
        await self._post("sendMessage", payload)
        logger.info(
            "Telegram approval requested",
            extra={
                "approval_id": _safe_attr(approval, "id", "unknown"),
                "gmail_id": _safe_attr(_safe_attr(approval, "email", None), "gmail_id", "unknown"),
            },
        )

    async def send_style_selection(self, approval_id: int) -> None:
        if not self.enabled:
            return
        await self._post(
            "sendMessage",
            {
                "chat_id": self.chat_id,
                "text": "Choose tone\n\nFormal for official conversations, Normal for balanced replies, Friendly for warmer personal notes.",
                "reply_markup": {
                    "inline_keyboard": [
                        [
                            {"text": "Formal", "callback_data": f"style:{approval_id}:formal"},
                            {"text": "Normal", "callback_data": f"style:{approval_id}:normal"},
                            {"text": "Friendly", "callback_data": f"style:{approval_id}:friendly"},
                        ]
                    ]
                },
            },
        )

    async def send_draft_preview(self, approval_id: int, draft: str) -> None:
        if not self.enabled:
            return
        payload = self._draft_preview_payload(approval_id, draft)
        payload["chat_id"] = self.chat_id
        await self._post("sendMessage", payload)

    async def edit_draft_preview(self, *, chat_id: str | None, message_id: str | None, approval_id: int, draft: str) -> bool:
        if not self.enabled or not chat_id or not message_id:
            return False
        payload = self._draft_preview_payload(approval_id, draft, regenerated=True)
        payload.update({"chat_id": chat_id, "message_id": message_id})
        try:
            await self._post("editMessageText", payload)
            return True
        except Exception as exc:
            logger.warning("telegram_draft_preview_edit_failed", extra={"approval_id": approval_id, "error": str(exc), "error_type": type(exc).__name__})
            return False

    async def send_more_actions(self, approval_id: int, *, include_ignore_sender: bool = False) -> None:
        if not self.enabled:
            return
        await self._post(
            "sendMessage",
            {
                "chat_id": self.chat_id,
                "text": "More actions\nChoose one control group.",
                "reply_markup": self._more_actions_reply_markup(approval_id),
            },
        )

    async def send_quick_tone_actions(self, approval_id: int) -> None:
        if not self.enabled:
            return
        await self._post(
            "sendMessage",
            {
                "chat_id": self.chat_id,
                "text": "Quick tone\nReshape the current draft without starting a new approval.",
                "reply_markup": {
                    "inline_keyboard": [
                        [
                            {"text": "Short", "callback_data": f"qtone:{approval_id}:short"},
                            {"text": "Warm", "callback_data": f"qtone:{approval_id}:warm"},
                            {"text": "Executive", "callback_data": f"qtone:{approval_id}:executive"},
                        ],
                        [
                            {"text": "Formal", "callback_data": f"qtone:{approval_id}:formal"},
                            {"text": "Casual", "callback_data": f"qtone:{approval_id}:casual"},
                        ],
                        [{"text": "Back", "callback_data": f"more:{approval_id}"}],
                    ]
                },
            },
        )

    async def send_regenerate_actions(self, approval_id: int) -> None:
        if not self.enabled:
            return
        await self._post(
            "sendMessage",
            {
                "chat_id": self.chat_id,
                "text": "Draft tools",
                "reply_markup": {
                    "inline_keyboard": [
                        [
                            {"text": "Regenerate", "callback_data": f"regenerate:{approval_id}"},
                            {"text": "Draft Better", "callback_data": f"draft_better:{approval_id}"},
                        ],
                        [
                            {"text": "Regenerate Reason", "callback_data": f"regen_reason:{approval_id}"},
                            {"text": "Back", "callback_data": f"more:{approval_id}"},
                        ],
                    ]
                },
            },
        )

    async def send_regenerate_reason_menu(self, approval_id: int) -> None:
        if not self.enabled:
            return
        await self._post(
            "sendMessage",
            {
                "chat_id": self.chat_id,
                "text": "Regenerate reason\nTell Seno what to improve.",
                "reply_markup": {
                    "inline_keyboard": [
                        [
                            {"text": "Too formal", "callback_data": f"regen_reason_apply:{approval_id}:too_formal"},
                            {"text": "Too long", "callback_data": f"regen_reason_apply:{approval_id}:too_long"},
                        ],
                        [
                            {"text": "Too robotic", "callback_data": f"regen_reason_apply:{approval_id}:too_robotic"},
                            {"text": "More direct", "callback_data": f"regen_reason_apply:{approval_id}:more_direct"},
                        ],
                        [
                            {"text": "More friendly", "callback_data": f"regen_reason_apply:{approval_id}:more_friendly"},
                            {"text": "Stronger negotiation", "callback_data": f"regen_reason_apply:{approval_id}:stronger_negotiation"},
                        ],
                        [{"text": "Back", "callback_data": f"menu_regen:{approval_id}"}],
                    ]
                },
            },
        )

    async def send_sender_controls(self, approval_id: int, *, include_ignore_sender: bool = False) -> None:
        if not self.enabled:
            return
        rows = [
            [
                {"text": "Ignore Thread", "callback_data": f"ignore_thread:{approval_id}"},
                {"text": "Pin Sender", "callback_data": f"pin_sender:{approval_id}"},
            ],
            [
                {"text": "Auto Handle Similar", "callback_data": f"auto_handle_similar:{approval_id}"},
                {"text": "Mark Handled", "callback_data": f"handled:{approval_id}"},
            ],
        ]
        if include_ignore_sender:
            rows.insert(0, [{"text": "Ignore Sender", "callback_data": f"ignore_sender:{approval_id}"}])
        rows.append([{"text": "Back", "callback_data": f"more:{approval_id}"}])
        await self._post(
            "sendMessage",
            {
                "chat_id": self.chat_id,
                "text": "Thread and sender controls",
                "reply_markup": {"inline_keyboard": rows},
            },
        )

    async def send_snooze_menu(self, approval_id: int) -> None:
        if not self.enabled:
            return
        await self._post(
            "sendMessage",
            {
                "chat_id": self.chat_id,
                "text": "Snooze",
                "reply_markup": {
                    "inline_keyboard": [
                        [
                            {"text": "1 Hour", "callback_data": f"snooze:{approval_id}:1h"},
                            {"text": "Tonight", "callback_data": f"snooze:{approval_id}:tonight"},
                        ],
                        [
                            {"text": "Tomorrow AM", "callback_data": f"snooze:{approval_id}:tomorrow_morning"},
                            {"text": "Monday", "callback_data": f"snooze:{approval_id}:monday"},
                        ],
                        [
                            {"text": "After Meeting", "callback_data": f"snooze:{approval_id}:after_meeting"},
                            {"text": "Custom Time", "callback_data": f"snooze:{approval_id}:custom"},
                        ],
                        [{"text": "Back", "callback_data": f"more:{approval_id}"}],
                    ]
                },
            },
        )

    async def send_info_actions(self, approval_id: int) -> None:
        if not self.enabled:
            return
        await self._post(
            "sendMessage",
            {
                "chat_id": self.chat_id,
                "text": "Information",
                "reply_markup": {
                    "inline_keyboard": [
                        [
                            {"text": "Risk", "callback_data": f"risk:{approval_id}"},
                            {"text": "Full Email", "callback_data": f"full:{approval_id}"},
                        ],
                        [
                            {"text": "Calendar", "callback_data": f"calendar:{approval_id}"},
                            {"text": "New Time", "callback_data": f"alt_time:{approval_id}"},
                        ],
                        [
                            {"text": "Edit", "callback_data": f"edit:{approval_id}"},
                            {"text": "Delete", "callback_data": f"confirm_delete:{approval_id}"},
                        ],
                        [{"text": "Back", "callback_data": f"more:{approval_id}"}],
                    ]
                },
            },
        )

    async def send_ignore_sender_warning(self, approval_id: int) -> None:
        if not self.enabled:
            return
        await self._post(
            "sendMessage",
            {
                "chat_id": self.chat_id,
                "text": (
                    "Ignore sender?\n\n"
                    "This sender was classified as important/moderate risk. "
                    "Future emails from this exact sender will skip Seno notifications, but Gmail will still receive them."
                ),
                "reply_markup": {
                    "inline_keyboard": [
                        [
                            {"text": "Confirm Ignore Sender", "callback_data": f"confirm_ignore_sender:{approval_id}"},
                            {"text": "Back", "callback_data": f"menu_controls:{approval_id}"},
                        ]
                    ]
                },
            },
        )

    def _draft_preview_payload(self, approval_id: int, draft: str, *, regenerated: bool = False) -> dict[str, Any]:
        title = "Draft regenerated.\n\nDraft Preview" if regenerated else "Draft Preview"
        return {
            "text": self._truncate(f"{title}\n\n{draft}"),
            "reply_markup": {
                "inline_keyboard": [
                    [
                        {"text": "Send", "callback_data": f"send:{approval_id}"},
                        {"text": "Regenerate", "callback_data": f"regenerate:{approval_id}"},
                    ],
                    [
                        {"text": "Edit", "callback_data": f"edit:{approval_id}"},
                        {"text": "Cancel", "callback_data": f"cancel:{approval_id}"},
                    ],
                ]
            },
        }

    async def send_primary_actions(self, approval_id: int) -> None:
        if not self.enabled:
            return
        await self._post(
            "sendMessage",
            {
                "chat_id": self.chat_id,
                "text": "Approval actions",
                "reply_markup": self._approval_reply_markup(approval_id),
            },
        )

    async def clear_inline_keyboard(self, *, chat_id: str | None, message_id: str | None) -> None:
        if not self.enabled or not chat_id or not message_id:
            return
        try:
            await self._post(
                "editMessageReplyMarkup",
                {
                    "chat_id": chat_id,
                    "message_id": int(message_id),
                    "reply_markup": {"inline_keyboard": []},
                },
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in {400, 409}:
                logger.warning(
                    "Telegram inline keyboard clear skipped",
                    extra={"status_code": exc.response.status_code, "message_id": message_id},
                )
                return
            raise
        except (ValueError, httpx.HTTPError) as exc:
            logger.warning("Telegram inline keyboard clear failed", extra={"error": str(exc), "message_id": message_id})

    async def send_send_confirmation(
        self,
        *,
        sender_alias: str | None,
        tone: str | None,
        status: str = "Completed",
    ) -> None:
        if not self.enabled:
            return
        text = (
            "Reply sent successfully.\n\n"
            f"From:\n{sender_alias or 'Default sender'}\n\n"
            f"Tone:\n{(tone or 'Selected').title()}\n\n"
            f"Status:\n{status}"
        )
        await self.send_message(text)

    async def send_reject_confirmation(self) -> None:
        if not self.enabled:
            return
        await self.send_message("Approval rejected.\n\nStatus:\nClosed\n\nNo reply was sent.")

    async def send_delete_confirmation_prompt(self, approval_id: int) -> None:
        if not self.enabled:
            return
        await self._post(
            "sendMessage",
            {
                "chat_id": self.chat_id,
                "text": "Delete this email?\n\nNo reply will be sent.",
                "reply_markup": {
                    "inline_keyboard": [
                        [
                            {"text": "Delete Email", "callback_data": f"delete:{approval_id}"},
                            {"text": "Back", "callback_data": f"back:{approval_id}"},
                        ]
                    ]
                },
            },
        )

    async def send_delete_confirmation(self) -> None:
        if not self.enabled:
            return
        await self.send_message("Email deleted.\n\nStatus:\nClosed\n\nNo reply was sent.")

    async def send_snooze_confirmation(self, option: str | None = None) -> None:
        if not self.enabled:
            return
        label = _snooze_label(option)
        await self.send_message(f"Approval snoozed{f' until {label}' if label else ''}.\n\nStatus:\nPaused\n\nYou can return to this email later.")

    async def send_message(self, text: str) -> None:
        if not self.enabled:
            return
        await self._post("sendMessage", {"chat_id": self.chat_id, "text": text[:3900]})

    async def answer_callback(self, callback_query_id: str, text: str) -> None:
        if not self.enabled:
            return
        try:
            await self._post("answerCallbackQuery", {"callback_query_id": callback_query_id, "text": text})
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in {400, 409}:
                logger.warning(
                    "Telegram callback acknowledgement skipped because it was already answered, expired, or conflicted",
                    extra={"status_code": exc.response.status_code},
                )
                return
            raise
        except httpx.HTTPError as exc:
            logger.warning("Telegram callback acknowledgement failed", extra={"error": str(exc)})
            return

    async def _post(self, method: str, payload: dict[str, Any]) -> Any:
        if not self.token:
            return None
        url = f"https://api.telegram.org/bot{self.token}/{method}"
        if self.http_client:
            return await self.http_client.post(url, json=payload, timeout=15)
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()

    @staticmethod
    def _escape(value: str) -> str:
        return (value or "").replace("_", "\\_").replace("*", "\\*").replace("[", "\\[")

    def _approval_payload(self, approval: ApprovalRequest | Any) -> dict[str, Any]:
        text = self._approval_brief_text(approval)
        approval_id = _safe_attr(approval, "id", "unknown")
        return {
            "chat_id": self.chat_id,
            "text": self._truncate(text),
            "reply_markup": self._approval_reply_markup(approval_id),
        }

    def _approval_brief_text(self, approval: ApprovalRequest | Any) -> str:
        email = _safe_attr(approval, "email", None)
        context = _safe_attr(approval, "category", "General") or "General"
        summary = _safe_attr(approval, "summary", "No summary available.")
        recommendation = _safe_attr(
            approval,
            "reply_recommendation",
            "Preview and approve a contextual response before sending.",
        ) or "Preview and approve a contextual response before sending."
        sections = [
            "Executive Brief",
            f"Priority: {_safe_attr(approval, 'priority', 'Medium') or 'Medium'}",
            f"Context: {context}",
            f"Confidence: {self._confidence_band(approval)}",
            f"From: {_safe_attr(email, 'sender', 'Unknown sender')}",
            f"Replying as: {_safe_attr(email, 'selected_sender_alias', None) or _safe_attr(email, 'original_recipient', None) or 'Default sender'}",
            f"Subject: {_safe_attr(email, 'subject', '(no subject)') or '(no subject)'}",
            "",
            "Summary",
            self._clean_summary(summary),
            "",
            "Key Signals",
            self._key_signals_text(approval),
            "",
            "Recommendation",
            recommendation,
        ]
        if self.debug_workflow:
            sections.extend(
                [
                    "",
                    "Diagnostics",
                    f"Confidence: {_safe_attr(approval, 'confidence_label', 'unknown')}",
                    f"Risk: {_safe_attr(approval, 'risk_score', 'unknown')}/100",
                    f"Urgency: {_safe_attr(approval, 'urgency', 'unknown')}",
                    f"Original Recipient: {_safe_attr(email, 'original_recipient', 'Unknown') or 'Unknown'}",
                    f"Routing: {_safe_attr(email, 'alias_selection_reason', None) or _safe_attr(email, 'recipient_detection_source', None) or 'default'}",
                ]
            )
        return "\n".join(sections)

    @staticmethod
    def _clean_summary(summary: str) -> str:
        clean = " ".join((summary or "No summary available.").split())
        if len(clean) <= 520:
            return clean
        return clean[:517].rstrip() + "..."

    def _key_signals_text(self, approval: ApprovalRequest | Any) -> str:
        signals = self._executive_signals(approval)
        if not signals:
            return "- Review requested"
        return "\n".join(f"- {signal}" for signal in signals[:5])

    @staticmethod
    def _executive_signals(approval: ApprovalRequest | Any) -> list[str]:
        raw_items = [_safe_attr(approval, "category", "")]
        raw_items.extend(_safe_list_attr(approval, "risk_explanation"))
        signals: list[str] = []
        seen: set[str] = set()
        noisy_fragments = (
            "low-risk",
            "low risk",
            "neutral",
            "risk score",
            "confidence",
            "urgency",
            "executive attention signal",
            "professional opportunity signal",
        )
        replacements = (
            ("availability", "Scheduling intent detected"),
            ("available", "Scheduling intent detected"),
            ("schedule", "Scheduling intent detected"),
            ("meeting", "Scheduling intent detected"),
            ("calendar", "Scheduling intent detected"),
            ("recruiter", "Recruiter outreach"),
            ("internship", "Opportunity discussion"),
            ("opportunity", "Opportunity discussion"),
            ("collaboration", "Collaboration request"),
            ("connect", "Networking opportunity"),
            ("network", "Networking opportunity"),
            ("technical", "Technical discussion"),
            ("project", "Project discussion"),
            ("attachment", "Attachment included"),
            ("invoice", "Finance-sensitive topic"),
            ("payment", "Finance-sensitive topic"),
            ("contract", "Contract/legal topic"),
            ("legal", "Contract/legal topic"),
        )
        for item in raw_items:
            normalized = " ".join(str(item or "").replace("_", " ").split()).strip(" -")
            lowered = normalized.lower()
            if not lowered or any(fragment in lowered for fragment in noisy_fragments):
                continue
            signal = normalized[:1].upper() + normalized[1:]
            for needle, replacement in replacements:
                if needle in lowered:
                    signal = replacement
                    break
            key = signal.lower()
            if key not in seen:
                seen.add(key)
                signals.append(signal)
        return signals

    def _fallback_approval_payload(self, approval: Any) -> dict[str, Any]:
        approval_id = _safe_attr(approval, "id", "unknown")
        email = _safe_attr(approval, "email", None)
        text = (
            "Approval needed\n\n"
            f"From: {_safe_attr(email, 'sender', 'Unknown sender')}\n"
            f"Subject: {_safe_attr(email, 'subject', '(no subject)')}\n\n"
            f"Suggested reply:\n{_safe_attr(approval, 'suggested_reply', '')}"
        )
        return {
            "chat_id": self.chat_id,
            "text": self._truncate(text),
            "reply_markup": self._approval_reply_markup(approval_id),
        }

    @staticmethod
    def _approval_reply_markup(approval_id: Any) -> dict[str, Any]:
        return {
            "inline_keyboard": [
                [
                    {"text": "Approve", "callback_data": f"approve:{approval_id}"},
                    {"text": "Reject", "callback_data": f"reject:{approval_id}"},
                ],
                [
                    {"text": "More", "callback_data": f"more:{approval_id}"},
                ],
            ]
        }

    @staticmethod
    def _more_actions_reply_markup(approval_id: Any) -> dict[str, Any]:
        return {
            "inline_keyboard": [
                [
                    {"text": "Quick Tone", "callback_data": f"menu_tone:{approval_id}"},
                    {"text": "Draft Tools", "callback_data": f"menu_regen:{approval_id}"},
                ],
                [
                    {"text": "Thread/Sender", "callback_data": f"menu_controls:{approval_id}"},
                    {"text": "Snooze", "callback_data": f"menu_snooze:{approval_id}"},
                ],
                [
                    {"text": "Info", "callback_data": f"menu_info:{approval_id}"},
                    {"text": "Back", "callback_data": f"back:{approval_id}"},
                ],
            ]
        }

    @staticmethod
    def _confidence_band(approval: ApprovalRequest | Any) -> str:
        confidence = float(_safe_attr(approval, "confidence", 0.0) or 0.0)
        if confidence >= 0.8:
            return "High"
        if confidence >= 0.45:
            return "Medium"
        return "Low"

    @staticmethod
    def _truncate(text: str, limit: int = 3900) -> str:
        if len(text) <= limit:
            return text
        return text[: limit - 18].rstrip() + "\n\n[truncated]"


def _safe_attr(value: Any, name: str, default: Any) -> Any:
    try:
        return getattr(value, name)
    except Exception:
        return default


def _safe_list_attr(value: Any, name: str) -> list[Any]:
    raw = _safe_attr(value, name, [])
    if raw is None:
        return []
    if isinstance(raw, list | tuple | set):
        return list(raw)
    if isinstance(raw, str):
        return [raw]
    return []


def _snooze_label(option: str | None) -> str | None:
    labels = {
        "1h": "1 hour",
        "tonight": "tonight",
        "tomorrow_morning": "tomorrow morning",
        "monday": "Monday",
        "after_meeting": "after the meeting",
        "custom": "a custom time",
    }
    return labels.get(option or "")


def _approval_snapshot(approval: Any) -> dict[str, Any]:
    email = _safe_attr(approval, "email", None)
    return {
        "type": type(approval).__name__,
        "approval_id": _safe_attr(approval, "id", "unknown"),
        "has_email": email is not None,
        "gmail_id": _safe_attr(email, "gmail_id", "unknown") if email else "unknown",
        "category": _safe_attr(approval, "category", None),
        "priority": _safe_attr(approval, "priority", None),
        "has_risk_explanation": hasattr(approval, "risk_explanation"),
        "has_summary": hasattr(approval, "summary"),
    }
