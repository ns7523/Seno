import hashlib
from difflib import SequenceMatcher
from typing import Any, Protocol

from app.ai.classifier import EmailAnalyzer
from app.ai.drafting import ContextualDraftGenerator
from app.ai.priority_engine import PriorityEngine
from app.ai.responder import ReplySafetyError, format_reply_style, humanize_reply, score_email_quality, validate_reply
from app.ai.risk_engine import RiskEngine
from app.branding.footer import apply_seno_footer
from app.calendar.service import CalendarService
from app.database import Database
from app.gmail.reader import GmailReader
from app.gmail.sender import GmailSender
from app.memory.memory import MemoryStore
from app.models.email import EmailMessage
from app.telegram.bot import TelegramBot
from app.utils.helpers import normalize_email_address, retry_async
from app.utils.logger import get_logger


logger = get_logger(__name__)


class GmailManager(Protocol):
    async def archive_email(self, email: EmailMessage) -> None:
        ...


class EmailProcessor:
    def __init__(
        self,
        *,
        db: Database,
        memory: MemoryStore,
        analyzer: EmailAnalyzer,
        gmail_sender: GmailSender,
        telegram: TelegramBot,
        gmail_manager: GmailManager | None = None,
        calendar_service: CalendarService | None = None,
        risk_engine: RiskEngine | None = None,
        auto_reply_threshold: int = 55,
        min_confidence: float = 0.75,
        debug_gmail_pipeline: bool = False,
        debug_workflow: bool = False,
        email_footer_mode: str = "professional",
        protected_ignore_addresses: list[str] | None = None,
    ) -> None:
        self.db = db
        self.memory = memory
        self.analyzer = analyzer
        self.gmail_sender = gmail_sender
        self.telegram = telegram
        self.gmail_manager = gmail_manager
        self.calendar_service = calendar_service
        self.risk_engine = risk_engine or RiskEngine(auto_reply_threshold)
        self.priority_engine = PriorityEngine()
        self.draft_generator = ContextualDraftGenerator()
        self.auto_reply_threshold = auto_reply_threshold
        self.min_confidence = min_confidence
        self.debug_gmail_pipeline = debug_gmail_pipeline
        self.debug_workflow = debug_workflow
        self.email_footer_mode = email_footer_mode
        self.protected_ignore_addresses = {
            normalize_email_address(address)
            for address in (protected_ignore_addresses or [])
            if normalize_email_address(address)
        }

    def _workflow_extra(self, approval_id: int, **extra: Any) -> dict[str, Any]:
        snapshot = self.db.get_approval_debug_snapshot(approval_id) if self.debug_workflow else {"approval_id": approval_id}
        snapshot.update(extra)
        return snapshot

    def is_protected_sender(self, sender: str) -> bool:
        sender_email = normalize_email_address(sender)
        local_part = sender_email.split("@", 1)[0] if "@" in sender_email else sender_email
        system_local_parts = {"admin", "system", "security", "postmaster"}
        return bool(sender_email and (sender_email in self.protected_ignore_addresses or local_part in system_local_parts))

    async def process_email(self, email: EmailMessage) -> None:
        already_processed = self.db.is_processed(email.gmail_id)
        logger.info(
            "Processed cache decision",
            extra={
                "gmail_id": email.gmail_id,
                "thread_id": email.thread_id,
                "sender": normalize_email_address(email.sender),
                "subject": email.subject,
                "original_recipient": email.original_recipient,
                "recipient_detection_source": email.recipient_detection_source,
                "already_processed": already_processed,
                "debug_gmail_pipeline": self.debug_gmail_pipeline,
            },
        )
        if already_processed and not self.debug_gmail_pipeline:
            logger.info(
                "Duplicate email skipped",
                extra={
                    "gmail_id": email.gmail_id,
                    "thread_id": email.thread_id,
                    "skip_reason": "already_processed",
                    "original_recipient": email.original_recipient,
                    "recipient_detection_source": email.recipient_detection_source,
                },
            )
            self.db.log_action("duplicate_skipped", email.gmail_id)
            return
        if already_processed and self.debug_gmail_pipeline:
            logger.warning(
                "DEBUG_GMAIL_PIPELINE enabled; processing email despite processed cache hit",
                extra={"gmail_id": email.gmail_id, "skip_reason": "processed_cache_bypassed"},
            )

        if self.db.is_thread_ignored(email.thread_id) and not self.debug_gmail_pipeline:
            logger.info(
                "Email skipped because thread is ignored",
                extra={
                    "gmail_id": email.gmail_id,
                    "thread_id": email.thread_id,
                    "ignored_thread": True,
                    "skip_reason": "ignored_thread",
                },
            )
            self.db.record_email(email, status="ignored_thread")
            self.db.log_action("ignored_thread_skipped", email.gmail_id, {"thread_id": email.thread_id, "ignored_thread": True})
            return

        sender_email = normalize_email_address(email.sender)
        if self.db.is_sender_ignored(sender_email) and not self.debug_gmail_pipeline:
            logger.info(
                "Email skipped because sender is ignored",
                extra={
                    "gmail_id": email.gmail_id,
                    "thread_id": email.thread_id,
                    "sender": sender_email,
                    "ignored_sender": True,
                    "skip_reason": "ignored_sender",
                },
            )
            self.db.record_email(email, status="ignored_sender")
            self.db.log_action("ignored_sender_skipped", email.gmail_id, {"sender": sender_email, "ignored_sender": True})
            return

        identity = self.gmail_sender.route_sender_identity(email)
        logger.info(
            "Sender identity routing decision",
            extra={
                "gmail_id": email.gmail_id,
                "sender": normalize_email_address(email.sender),
                "original_recipient": identity.original_recipient,
                "recipient_detection_source": identity.detection_source,
                "selected_sender_alias": identity.selected_sender_alias,
                "alias_selection_reason": identity.reason,
            },
        )
        logger.info("Processing Gmail email", extra={"gmail_id": email.gmail_id, "sender": normalize_email_address(email.sender)})
        profile = self.memory.get_sender_profile(email.sender)
        risk_hint = self.risk_engine.assess(email, has_attachments=email.has_attachments, sender_trust=profile.trust_score)
        logger.info(
            "Risk score calculated",
            extra={
                "gmail_id": email.gmail_id,
                "risk_score": risk_hint.risk_score,
                "requires_approval": risk_hint.requires_approval,
                "routing_reasons": risk_hint.reasons,
            },
        )
        self.db.record_email(email, status="received")
        self.db.log_action(
            "email_received",
            email.gmail_id,
            {
                "sender": normalize_email_address(email.sender),
                "original_recipient": email.original_recipient,
                "selected_sender_alias": email.selected_sender_alias,
                "alias_selection_reason": email.alias_selection_reason,
                "recipient_detection_source": email.recipient_detection_source,
            },
        )
        self.memory.record_thread_observation(email)
        thread_summary = self.memory.get_thread_summary(email.thread_id).as_context()

        if (
            self.db.is_sender_auto_handled(sender_email)
            and risk_hint.risk_score < 50
            and not risk_hint.requires_approval
            and not risk_hint.never_reply
        ):
            logger.info(
                "Email auto-handled by similar sender rule",
                extra={
                    "gmail_id": email.gmail_id,
                    "sender": sender_email,
                    "risk_score": risk_hint.risk_score,
                    "auto_handle_similar": True,
                },
            )
            self.db.update_email_status(email.gmail_id, "auto_handled_similar")
            self.db.log_action("auto_handle_similar_skipped", email.gmail_id, {"sender": sender_email, "risk_score": risk_hint.risk_score})
            return

        # Deterministic never-reply categories are blocked before any LLM call.
        # This reduces cost/latency and avoids sending suspicious content to a model.
        if risk_hint.never_reply:
            logger.info("Email ignored by deterministic never-reply policy", extra={"gmail_id": email.gmail_id})
            self.db.update_email_status(email.gmail_id, "ignored")
            self.memory.record_interaction(email.sender, approved=False, auto_replied=False, risk_score=risk_hint.risk_score)
            self.db.log_action("never_reply_prefilter_ignored", email.gmail_id, {"reasons": risk_hint.reasons})
            return

        logger.info("Generating AI classification", extra={"gmail_id": email.gmail_id})
        relationship = self.memory.get_relationship_profile(email.sender)
        analysis = await self.analyzer.analyze(
            email,
            memory_context={
                "sender_email": profile.email,
                "trust_score": profile.trust_score,
                "total_interactions": profile.total_interactions,
                "avg_risk": profile.avg_risk,
                "relationship_type": relationship.relationship_type,
                "preferred_tone": relationship.preferred_tone,
                "preferred_signoff": relationship.preferred_signoff,
                "thread_history": self.db.get_thread_history(email.thread_id),
                "thread_summary": thread_summary,
            },
            risk_hint=risk_hint,
        )
        self.db.record_decision(email.gmail_id, analysis)
        logger.info(
            "AI classification complete",
            extra={"gmail_id": email.gmail_id, "intent": analysis.intent, "risk_score": analysis.risk_score},
        )
        self.db.log_action("ai_decision", email.gmail_id, {"risk_score": analysis.risk_score, "intent": analysis.intent})

        executive_signal = self._has_executive_signal(email, analysis, risk_hint)
        should_approve = self._requires_approval(analysis, risk_hint, executive_signal=executive_signal)
        approval_reasons = self._approval_reasons(analysis, risk_hint)
        logger.info(
            "Email routing decision",
            extra={
                "gmail_id": email.gmail_id,
                "route": "approval" if should_approve else "auto_reply",
                "routing_confidence": round(max(analysis.confidence, 0.0), 2),
                "approval_reasons": approval_reasons,
                "risk_reasons": risk_hint.reasons,
                "executive_signal": executive_signal,
            },
        )

        if analysis.never_reply:
            logger.info("Email ignored by never-reply policy", extra={"gmail_id": email.gmail_id})
            self.db.update_email_status(email.gmail_id, "ignored")
            self.memory.record_interaction(email.sender, approved=False, auto_replied=False, risk_score=analysis.risk_score)
            self.db.log_action("never_reply_ignored", email.gmail_id, {"reasons": analysis.reasons})
            return

        if should_approve:
            priority = self.priority_engine.assess(email, risk_score=analysis.risk_score, urgency=analysis.urgency)
            if self.db.is_sender_pinned(sender_email):
                priority.level = "Executive Attention Required"
                priority.reasons.append("pinned sender")
            summary = await self._summary_with_calendar_warning(email, analysis.summary)
            approval = self.db.create_approval(
                email,
                summary,
                analysis.suggested_reply,
                analysis.risk_score,
                category=self._category_for(email, analysis),
                urgency=analysis.urgency,
                suggested_tone=self._suggested_tone(analysis, relationship.preferred_tone),
                priority=priority.level,
                reply_recommendation=self._reply_recommendation(analysis),
                confidence=analysis.confidence,
                risk_explanation=[*risk_hint.reasons, *analysis.reasons, *priority.reasons],
            )
            self.db.update_email_status(email.gmail_id, "pending_approval")
            logger.info("Telegram approval requested", extra={"gmail_id": email.gmail_id, "approval_id": approval.id})
            try:
                await retry_async(lambda: self.telegram.send_approval_request(approval), attempts=3)
            except Exception as exc:
                self.db.update_approval_notification(approval.id, "failed", str(exc))
                self.db.update_email_status(email.gmail_id, "approval_notification_failed")
                self.db.log_action("approval_notification_failed", email.gmail_id, {"approval_id": approval.id})
                raise
            self.db.update_approval_notification(approval.id, "sent")
            self.db.log_action("approval_requested", email.gmail_id, {"approval_id": approval.id})
            return

        reply = self._apply_footer(self._build_auto_reply(analysis, email))
        logger.info("Auto reply triggered", extra={"gmail_id": email.gmail_id})
        await retry_async(lambda: self.gmail_sender.send_reply(email, reply), attempts=3)
        logger.info("Email sent successfully", extra={"gmail_id": email.gmail_id})
        self.db.update_email_status(email.gmail_id, "auto_replied")
        self.memory.record_interaction(email.sender, approved=True, auto_replied=True, risk_score=analysis.risk_score)
        self.db.log_action("auto_reply_sent", email.gmail_id)

    async def begin_approval(self, approval_id: int) -> bool:
        pending = self.db.get_pending_approval(approval_id)
        if not pending:
            logger.info("Approval callback ignored because it was already handled", extra={"approval_id": approval_id})
            return False
        approval, email = pending
        if not self.db.set_approval_status(approval_id, "awaiting_style"):
            return False
        self.db.log_action("approval_style_requested", email.gmail_id, {"approval_id": approval_id})
        return True

    async def reject_approval(self, approval_id: int) -> bool:
        pending = self.db.get_pending_approval(approval_id)
        if not pending:
            logger.info("Approval rejection ignored because it was already handled", extra={"approval_id": approval_id})
            return False
        approval, email = pending
        if not self.db.decide_approval(approval_id, False):
            return False
        logger.info("Telegram approval rejected", extra={"gmail_id": email.gmail_id, "approval_id": approval_id})
        self.db.update_email_status(email.gmail_id, "approval_rejected")
        self.memory.record_interaction(email.sender, approved=False, auto_replied=False, risk_score=75, rejected=True)
        self.memory.record_rejection(email.sender)
        self.db.log_action("approval_rejected", email.gmail_id, {"approval_id": approval_id})
        return True

    async def preview_approved_reply(self, approval_id: int, *, style: str = "normal") -> str:
        logger.info("workflow_stage_preview_start", extra=self._workflow_extra(approval_id, selected_style=style))
        pending = self.db.get_pending_approval(approval_id)
        if not pending:
            logger.warning("workflow_stage_preview_missing_approval", extra=self._workflow_extra(approval_id, selected_style=style))
            raise ReplySafetyError("Approval request is no longer active")
        approval, email = pending
        logger.info("workflow_stage_preview_state_loaded", extra=self._workflow_extra(approval_id, selected_style=style))
        try:
            reply = self._apply_footer(self._build_approved_reply(approval.suggested_reply, email, style, contextual=True))
            reply = self._apply_thread_continuity(reply, email)
            generated_checksum = hashlib.sha256(reply.encode("utf-8")).hexdigest()
            logger.info(
                "workflow_stage_preview_draft_generated",
                extra=self._workflow_extra(
                    approval_id,
                    selected_style=style,
                    generated_checksum=generated_checksum,
                    draft_length=len(reply),
                ),
            )
        except ReplySafetyError:
            self.db.update_email_status(email.gmail_id, "approval_failed")
            logger.exception("workflow_stage_preview_safety_failed", extra=self._workflow_extra(approval_id, selected_style=style))
            raise
        if not self.db.set_approval_draft(approval_id, reply, style):
            logger.error("workflow_stage_preview_save_failed", extra=self._workflow_extra(approval_id, selected_style=style))
            raise ReplySafetyError("Unable to save draft preview")
        logger.info("workflow_stage_preview_saved", extra=self._workflow_extra(approval_id, selected_style=style))
        self.memory.record_tone_selection(email.sender, style)
        await self.telegram.send_draft_preview(approval_id, reply)
        logger.info("workflow_stage_preview_telegram_sent", extra=self._workflow_extra(approval_id, selected_style=style))
        self.db.log_action("draft_preview_generated", email.gmail_id, {"approval_id": approval_id, "style": style})
        return reply

    async def send_previewed_reply(self, approval_id: int) -> bool:
        logger.info("workflow_stage_send_start", extra=self._workflow_extra(approval_id))
        pending = self.db.get_pending_approval(approval_id)
        if not pending:
            logger.warning("workflow_stage_send_missing_approval", extra=self._workflow_extra(approval_id))
            return False
        approval, email = pending
        logger.info("workflow_stage_send_state_loaded", extra=self._workflow_extra(approval_id))
        if not approval.final_reply:
            logger.error("Preview send aborted because no persisted draft exists", extra={"approval_id": approval_id})
            return False
        reply = approval.final_reply
        checksum = hashlib.sha256(reply.encode("utf-8")).hexdigest()
        logger.info(
            "workflow_stage_send_checksum_validation",
            extra=self._workflow_extra(
                approval_id,
                generated_checksum=checksum,
                stored_checksum=approval.draft_checksum,
                checksum_matches=(not approval.draft_checksum or checksum == approval.draft_checksum),
            ),
        )
        if approval.draft_checksum and checksum != approval.draft_checksum:
            logger.error(
                "Preview send aborted because persisted draft checksum changed",
                extra={"gmail_id": email.gmail_id, "approval_id": approval_id, "selected_style": approval.selected_style},
            )
            self.db.update_email_status(email.gmail_id, "approval_failed")
            return False
        logger.info("workflow_stage_send_safety_validation_start", extra=self._workflow_extra(approval_id))
        validated = validate_reply(
            self._analysis_from_reply(approval.final_reply),
            human_approved=True,
            original_email=email,
        )
        if validated != reply:
            logger.error(
                "Preview send aborted because safety validation would mutate the approved draft",
                extra={"gmail_id": email.gmail_id, "approval_id": approval_id, "selected_style": approval.selected_style},
            )
            self.db.update_email_status(email.gmail_id, "approval_failed")
            return False
        logger.info("workflow_stage_send_state_validation_start", extra=self._workflow_extra(approval_id, expected_states=["draft_preview", "editing", "send_failed"]))
        if not self.db.begin_approval_send(approval_id):
            logger.warning("workflow_stage_send_state_validation_rejected", extra=self._workflow_extra(approval_id, expected_states=["draft_preview", "editing", "send_failed"]))
            return False
        logger.info(
            "Sending persisted Telegram preview draft",
            extra={
                "gmail_id": email.gmail_id,
                "approval_id": approval_id,
                "selected_style": approval.selected_style,
                "fallback_used": False,
                "draft_checksum": checksum,
                "final_email_length": len(reply),
            },
        )
        logger.info("workflow_stage_send_gmail_start", extra=self._workflow_extra(approval_id))
        try:
            await retry_async(lambda: self.gmail_sender.send_reply(email, reply), attempts=3)
        except Exception:
            self.db.mark_approval_send_failed(approval_id)
            logger.exception("workflow_stage_send_gmail_failed", extra=self._workflow_extra(approval_id))
            raise
        logger.info("workflow_stage_send_gmail_complete", extra=self._workflow_extra(approval_id))
        self.db.mark_approval_sent(approval_id)
        self.db.update_email_status(email.gmail_id, "approved_sent")
        self.memory.record_interaction(email.sender, approved=True, auto_replied=False, risk_score=approval.risk_score or 60)
        self.memory.record_thread_observation(email, reply_text=reply)
        self.memory.record_approved_draft(email.sender, reply, approval.selected_style)
        if approval.selected_style:
            self.memory.record_tone_selection(email.sender, approval.selected_style)
        self.db.log_action("previewed_reply_sent", email.gmail_id, {"approval_id": approval_id, "style": approval.selected_style})
        logger.info("workflow_stage_send_complete", extra=self._workflow_extra(approval_id))
        return True

    async def create_calendar_event(self, approval_id: int) -> Any | None:
        pending = self.db.get_pending_approval(approval_id)
        if not pending or not self.calendar_service:
            return None
        _, email = pending
        event = await self.calendar_service.create_event_from_email(email)
        if event:
            self.db.log_action(
                "calendar_event_created",
                email.gmail_id,
                {"title": event.title, "starts_at": event.starts_at, "meet_link": getattr(event, "meet_link", None)},
            )
        return event

    async def suggest_alternative_times(self, approval_id: int) -> list[str]:
        pending = self.db.get_pending_approval(approval_id)
        if not pending or not self.calendar_service:
            return []
        _, email = pending
        alternatives = await self.calendar_service.suggest_alternative_times(email)
        self.db.log_action("calendar_alternatives_suggested", email.gmail_id, {"alternatives": alternatives})
        return alternatives

    async def handle_approval(self, approval_id: int, approved: bool) -> bool:
        if not approved:
            return await self.reject_approval(approval_id)
        return await self.send_previewed_reply(approval_id)

    def start_edit_reply(
        self,
        approval_id: int,
        *,
        telegram_chat_id: str | None = None,
        telegram_message_id: str | None = None,
        telegram_user_id: str | None = None,
    ) -> bool:
        logger.info("workflow_stage_edit_start", extra=self._workflow_extra(approval_id))
        pending = self.db.get_pending_approval(approval_id)
        if not pending:
            logger.warning("workflow_stage_edit_missing_approval", extra=self._workflow_extra(approval_id))
            return False
        approval, email = pending
        if not self.db.set_approval_status(approval_id, "editing"):
            logger.warning("workflow_stage_edit_state_validation_rejected", extra=self._workflow_extra(approval_id, expected_states=["pending", "awaiting_style", "draft_preview"]))
            return False
        if telegram_chat_id:
            self.db.start_edit_session(
                approval_id,
                telegram_chat_id=telegram_chat_id,
                telegram_message_id=telegram_message_id,
                telegram_user_id=telegram_user_id,
            )
            logger.info(
                "workflow_stage_edit_session_started",
                extra=self._workflow_extra(
                    approval_id,
                    telegram_chat_id=telegram_chat_id,
                    telegram_message_id=telegram_message_id,
                    telegram_user_id=telegram_user_id,
                ),
            )
        self.db.log_action("approval_edit_requested", email.gmail_id, {"approval_id": approval_id})
        return True

    def snooze_approval(self, approval_id: int, option: str | None = None) -> bool:
        pending = self.db.get_pending_approval(approval_id)
        if not pending:
            logger.info("Approval snooze ignored because it is no longer active", extra={"approval_id": approval_id})
            return False
        _, email = pending
        if not self.db.set_approval_status(approval_id, "snoozed"):
            logger.warning("workflow_stage_snooze_state_validation_rejected", extra=self._workflow_extra(approval_id))
            return False
        self.db.log_action("approval_snoozed", email.gmail_id, {"approval_id": approval_id, "option": option})
        return True

    async def send_edited_reply(
        self,
        edited_reply: str,
        *,
        telegram_chat_id: str | None = None,
        telegram_user_id: str | None = None,
        reply_to_message_id: str | None = None,
    ) -> bool:
        logger.info(
            "workflow_stage_edit_send_start",
            extra={
                "telegram_chat_id": telegram_chat_id,
                "telegram_user_id": telegram_user_id,
                "reply_to_message_id": reply_to_message_id,
                "edited_reply_length": len(edited_reply),
            },
        )
        pending = (
            self.db.get_editing_approval_for_session(
                telegram_chat_id=telegram_chat_id,
                telegram_user_id=telegram_user_id,
                reply_to_message_id=reply_to_message_id,
            )
            if telegram_chat_id
            else self.db.get_editing_approval()
        )
        if not pending:
            logger.warning(
                "workflow_stage_edit_send_no_session",
                extra={
                    "telegram_chat_id": telegram_chat_id,
                    "telegram_user_id": telegram_user_id,
                    "reply_to_message_id": reply_to_message_id,
                },
            )
            return False
        approval, email = pending
        logger.info("workflow_stage_edit_send_session_resolved", extra=self._workflow_extra(approval.id))
        try:
            reply = self._apply_footer(validate_reply(
                self._analysis_from_reply(edited_reply),
                human_approved=True,
                original_email=email,
            ))
        except ReplySafetyError:
            self.db.update_email_status(email.gmail_id, "approval_failed")
            logger.exception("workflow_stage_edit_send_safety_failed", extra=self._workflow_extra(approval.id))
            raise
        logger.info("workflow_stage_edit_send_state_validation_start", extra=self._workflow_extra(approval.id, expected_states=["editing", "send_failed"]))
        if not self.db.begin_approval_send(approval.id):
            logger.warning("workflow_stage_edit_send_state_validation_rejected", extra=self._workflow_extra(approval.id, expected_states=["editing", "send_failed"]))
            return False
        logger.info("workflow_stage_edit_send_gmail_start", extra=self._workflow_extra(approval.id))
        try:
            await retry_async(lambda: self.gmail_sender.send_reply(email, reply), attempts=3)
        except Exception:
            self.db.mark_approval_send_failed(approval.id)
            logger.exception("workflow_stage_edit_send_gmail_failed", extra=self._workflow_extra(approval.id))
            raise
        logger.info("workflow_stage_edit_send_gmail_complete", extra=self._workflow_extra(approval.id))
        self.db.mark_approval_sent(approval.id)
        self.db.update_email_status(email.gmail_id, "approved_sent")
        self.memory.record_interaction(email.sender, approved=True, auto_replied=False, risk_score=60)
        self.memory.record_user_edit(email.sender, reply)
        self.memory.record_thread_observation(email, reply_text=reply)
        self.db.complete_edit_session(approval.id)
        self.db.log_action("edited_reply_sent", email.gmail_id, {"approval_id": approval.id})
        logger.info("workflow_stage_edit_send_complete", extra=self._workflow_extra(approval.id))
        return True

    async def delete_email(self, approval_id: int) -> bool:
        pending = self.db.get_pending_approval(approval_id)
        if not pending:
            return False
        approval, email = pending
        if not self.db.set_approval_status(approval_id, "deleted"):
            return False
        if self.gmail_manager:
            await retry_async(lambda: self.gmail_manager.archive_email(email), attempts=3)
        self.db.update_email_status(email.gmail_id, "deleted")
        self.memory.record_interaction(email.sender, approved=False, auto_replied=False, risk_score=approval.risk_score or 75)
        self.db.log_action("email_deleted_without_reply", email.gmail_id, {"approval_id": approval_id})
        return True

    def cancel_approval(self, approval_id: int) -> bool:
        pending = self.db.get_pending_approval(approval_id)
        if not pending:
            return False
        approval, email = pending
        if not self.db.set_approval_status(approval_id, "cancelled"):
            return False
        self.db.update_email_status(email.gmail_id, "approval_cancelled")
        self.db.log_action("approval_cancelled", email.gmail_id, {"approval_id": approval.id})
        return True

    def mark_handled(self, approval_id: int) -> bool:
        pending = self.db.get_pending_approval(approval_id)
        if not pending:
            return False
        approval, email = pending
        if not self.db.set_approval_status(approval_id, "handled"):
            return False
        self.db.update_email_status(email.gmail_id, "handled")
        self.db.log_action("approval_marked_handled", email.gmail_id, {"approval_id": approval.id})
        return True

    async def regenerate_reply(
        self,
        approval_id: int,
        *,
        style: str | None = None,
        strategy: str | None = None,
        reason: str | None = None,
    ) -> str | None:
        logger.info("workflow_stage_regenerate_start", extra=self._workflow_extra(approval_id))
        pending = self.db.get_pending_approval(approval_id)
        if not pending:
            logger.warning("workflow_stage_regenerate_missing_approval", extra=self._workflow_extra(approval_id))
            return None
        approval, email = pending
        style = (style or approval.selected_style or approval.suggested_tone or "normal").lower()
        previous_reply = approval.final_reply or approval.suggested_reply or ""
        regenerated, strategy = self._regenerate_reply(email, style=style, previous_reply=previous_reply, forced_strategy=strategy)
        if not self.db.set_approval_draft(approval_id, regenerated, style):
            logger.warning("workflow_stage_regenerate_state_validation_rejected", extra=self._workflow_extra(approval_id))
            return None
        logger.info(
            "workflow_stage_regenerate_reply_saved",
            extra=self._workflow_extra(approval_id, regenerated_length=len(regenerated), selected_style=style, strategy=strategy),
        )
        self.memory.record_regeneration_choice(email.sender, strategy=reason or strategy, draft=regenerated, tone=style)
        self.db.log_action(
            "reply_regenerated",
            email.gmail_id,
            {"approval_id": approval_id, "style": style, "strategy": strategy, "reason": reason},
        )
        return regenerated

    def ignore_sender(self, approval_id: int, *, confirmed: bool = False) -> bool:
        pending = self.db.get_pending_approval(approval_id)
        if not pending:
            logger.info("Ignore sender ignored because approval is no longer active", extra={"approval_id": approval_id})
            return False
        approval, email = pending
        sender_email = normalize_email_address(email.sender)
        if not sender_email or self.is_protected_sender(sender_email):
            logger.warning(
                "Sender ignore refused for protected address",
                extra={"approval_id": approval_id, "sender": sender_email, "protected_ignore": True},
            )
            return False
        if int(approval.risk_score or 100) >= 50 and not confirmed:
            logger.warning(
                "Sender ignore refused for non-low-risk approval",
                extra={"approval_id": approval_id, "sender": sender_email, "risk_score": approval.risk_score},
            )
            return False
        self.db.ignore_sender(sender_email, reason=f"telegram approval {approval_id}")
        self.db.set_approval_status(approval_id, "handled")
        self.db.update_email_status(email.gmail_id, "ignored_sender")
        self.db.log_action(
            "sender_ignored",
            email.gmail_id,
            {"approval_id": approval.id, "sender": sender_email, "ignored_sender": True, "risk_score": approval.risk_score},
        )
        logger.info(
            "Sender ignored from Telegram workflow",
            extra={"approval_id": approval_id, "sender": sender_email, "ignored_sender": True},
        )
        return True

    def ignore_thread(self, approval_id: int) -> bool:
        pending = self.db.get_pending_approval(approval_id)
        if not pending:
            return False
        approval, email = pending
        self.db.ignore_thread(email.thread_id, reason=f"telegram approval {approval_id}")
        self.db.set_approval_status(approval_id, "handled")
        self.db.update_email_status(email.gmail_id, "ignored_thread")
        self.db.log_action("thread_ignored", email.gmail_id, {"approval_id": approval.id, "thread_id": email.thread_id})
        return True

    def pin_sender(self, approval_id: int) -> bool:
        pending = self.db.get_pending_approval(approval_id)
        if not pending:
            return False
        approval, email = pending
        sender_email = normalize_email_address(email.sender)
        if not sender_email:
            return False
        self.db.pin_sender(sender_email, reason=f"telegram approval {approval_id}")
        self.db.log_action("sender_pinned", email.gmail_id, {"approval_id": approval.id, "sender": sender_email})
        return True

    def auto_handle_similar(self, approval_id: int) -> bool:
        pending = self.db.get_pending_approval(approval_id)
        if not pending:
            return False
        approval, email = pending
        sender_email = normalize_email_address(email.sender)
        if not sender_email or int(approval.risk_score or 100) >= 50:
            return False
        self.db.auto_handle_sender(sender_email, reason=f"telegram approval {approval_id}")
        self.db.set_approval_status(approval_id, "handled")
        self.db.update_email_status(email.gmail_id, "auto_handled_similar")
        self.db.log_action("auto_handle_similar_enabled", email.gmail_id, {"approval_id": approval.id, "sender": sender_email})
        return True

    def risk_analysis_text(self, approval_id: int) -> str:
        pending = self.db.get_pending_approval(approval_id)
        if not pending:
            return "Approval request is no longer active."
        approval, email = pending
        return (
            "Risk analysis\n\n"
            f"From: {email.sender}\n"
            f"Subject: {email.subject}\n"
            f"Risk score: {getattr(approval, 'risk_score', 'unknown')}\n"
            f"Current status: {approval.status}\n\n"
            f"Summary:\n{getattr(approval, 'summary', 'No summary available')}\n\n"
            f"Current draft:\n{approval.final_reply or approval.suggested_reply}"
        )

    def full_email_text(self, approval_id: int) -> str:
        pending = self.db.get_pending_approval(approval_id)
        if not pending:
            return "Approval request is no longer active."
        _, email = pending
        return f"Original email\n\nFrom: {email.sender}\nSubject: {email.subject}\n\n{email.body}"

    def _build_approved_reply(
        self,
        suggested_reply: str,
        email: EmailMessage,
        style: str,
        *,
        contextual: bool = False,
        variation: str | None = None,
    ) -> str:
        style_preferences = self.memory.get_style_preferences(email.sender)
        if contextual and variation is None and style_preferences.directness == "direct":
            variation = "concise_direct"
        draft_source = (
            self.draft_generator.build(
                email,
                suggested_reply=suggested_reply,
                style=style,
                variation=variation,
                thread_context=self.memory.get_thread_summary(email.thread_id).as_context(),
                style_preferences=style_preferences,
            )
            if contextual
            else suggested_reply
        )
        base = validate_reply(
            self._analysis_from_reply(draft_source),
            human_approved=True,
            original_email=email,
        )
        relationship = self.memory.get_relationship_profile(email.sender)
        styled = format_reply_style(
            base,
            style=style,
            original_email=email,
            preferred_greeting=style_preferences.preferred_greeting or relationship.preferred_greeting,
            preferred_signoff=style_preferences.preferred_signoff or relationship.preferred_signoff,
        )
        quality = score_email_quality(styled)
        if quality.needs_improvement:
            logger.info(
                "Draft quality improvement applied",
                extra={
                    "gmail_id": email.gmail_id,
                    "style": style,
                    "human_likeness": quality.human_likeness,
                    "ai_likeness": quality.ai_likeness,
                    "repetition": quality.repetition,
                },
            )
            styled = humanize_reply(styled, style=style, original_email=email)
        return validate_reply(
            self._analysis_from_reply(styled),
            human_approved=True,
            original_email=email,
        )

    def _apply_footer(self, reply: str) -> str:
        return apply_seno_footer(reply, self.email_footer_mode)

    def _apply_thread_continuity(self, reply: str, email: EmailMessage) -> str:
        prior = self.db.get_thread_commitments(email.thread_id, exclude_gmail_id=email.gmail_id)
        if not prior:
            return reply
        current_time = self._extract_time(reply)
        if not current_time:
            return reply
        for item in prior:
            previous_time = self._extract_time(item.get("final_reply") or item.get("suggested_reply") or "")
            if previous_time and previous_time != current_time:
                return (
                    reply.rstrip()
                    + f"\n\nNote: I previously had {previous_time} noted for this thread; please confirm if this changes."
                )
        return reply

    @staticmethod
    def _extract_time(text: str) -> str | None:
        import re

        match = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(AM|PM|am|pm)?\b", text)
        if not match:
            return None
        hour = match.group(1)
        minute = match.group(2)
        suffix = (match.group(3) or "").upper()
        return f"{hour}:{minute} {suffix}".strip() if minute else f"{hour} {suffix}".strip()

    async def _summary_with_calendar_warning(self, email: EmailMessage, summary: str) -> str:
        if not self.calendar_service:
            return summary
        conflicts = await self.calendar_service.conflicts_for_email(email)
        if not conflicts:
            return summary
        conflict_text = "; ".join(f"{event.title} at {event.starts_at}" for event in conflicts[:2])
        return f"{summary}\n\nSchedule warning: possible conflict with {conflict_text}."

    @staticmethod
    def _category_for(email: EmailMessage, analysis: Any) -> str:
        text = f"{analysis.intent} {analysis.summary} {email.subject} {email.body}".lower()
        if any(term in text for term in ("payment", "invoice", "bank", "finance")):
            return "Finance"
        if any(term in text for term in ("legal", "contract", "agreement")):
            return "Legal / Contract"
        if any(term in text for term in ("hr", "salary", "recruiter", "interview")):
            return "Professional / HR"
        if any(term in text for term in ("breakfast", "lunch", "dinner", "coffee", "campus", "uvce", "movie", "hangout")):
            return "Casual / Social"
        return "General"

    @staticmethod
    def _suggested_tone(analysis: Any, preferred_tone: str) -> str:
        if preferred_tone:
            return preferred_tone.title()
        tone = str(getattr(analysis, "tone", "normal") or "normal").lower()
        if tone in {"formal", "friendly"}:
            return tone.title()
        return "Normal"

    @staticmethod
    def _reply_recommendation(analysis: Any) -> str:
        if getattr(analysis, "risk_score", 0) >= 75:
            return "Review carefully before sending."
        if getattr(analysis, "requires_approval", False):
            return "Choose a tone and preview before sending."
        return "Safe to acknowledge."

    @staticmethod
    def _approval_reasons(analysis: Any, risk_hint: Any) -> list[str]:
        reasons: list[str] = []
        if getattr(risk_hint, "requires_approval", False):
            reasons.extend(getattr(risk_hint, "reasons", []) or [])
        if getattr(analysis, "requires_approval", False):
            reasons.append("model requested approval")
        if getattr(analysis, "confidence", 1.0) < 0.75:
            reasons.append("low confidence")
        if getattr(analysis, "risk_score", 0) >= 55:
            reasons.append("risk threshold exceeded")
        return reasons or ["safe low-context acknowledgement"]

    def _requires_approval(self, analysis: Any, risk_hint: Any, *, executive_signal: bool) -> bool:
        if getattr(analysis, "never_reply", False):
            return True
        if getattr(risk_hint, "never_reply", False):
            return False
        risk_score = max(int(getattr(analysis, "risk_score", 0) or 0), int(getattr(risk_hint, "risk_score", 0) or 0))
        if risk_score >= 60:
            return True
        if executive_signal:
            return True
        if risk_score < 50:
            return False
        return bool(
            getattr(analysis, "requires_approval", False)
            or getattr(risk_hint, "requires_approval", False)
            or getattr(analysis, "confidence", 1.0) < self.min_confidence
        )

    @staticmethod
    def _has_executive_signal(email: EmailMessage, analysis: Any, risk_hint: Any) -> bool:
        text = " ".join(
            str(value)
            for value in [
                email.subject,
                email.body,
                getattr(analysis, "intent", ""),
                getattr(analysis, "summary", ""),
                " ".join(getattr(analysis, "reasons", []) or []),
                " ".join(getattr(risk_hint, "reasons", []) or []),
            ]
        ).lower()
        return any(
            signal in text
            for signal in (
                "recruiter",
                "internship",
                "collaboration",
                "collaborate",
                "schedule",
                "scheduling",
                "meeting",
                "available",
                "availability",
                "networking",
                "connect",
                "partnership",
                "deployment",
                "business",
                "opportunity",
                "interview",
            )
        )

    def _build_auto_reply(self, analysis: Any, email: EmailMessage) -> str:
        raw = validate_reply(analysis, original_email=email)
        styled = format_reply_style(raw, style="formal", original_email=email)
        return validate_reply(self._analysis_from_reply(styled), original_email=email)

    @staticmethod
    def _analysis_from_reply(suggested_reply: str) -> Any:
        return type(
            "ApprovedAnalysis",
            (),
            {"suggested_reply": suggested_reply, "never_reply": False},
        )()

    def _regenerate_reply(
        self,
        email: EmailMessage,
        *,
        style: str,
        previous_reply: str,
        forced_strategy: str | None = None,
    ) -> tuple[str, str]:
        strategies = ["warmer_executive", "technical", "concise_direct", "collaborative"]
        if forced_strategy:
            reply = self._apply_footer(self._build_approved_reply("", email, style, contextual=True, variation=forced_strategy))
            reply = self._apply_thread_continuity(reply, email)
            return reply, forced_strategy
        best_reply = ""
        best_strategy = strategies[0]
        lowest_similarity = 1.0
        for strategy in strategies:
            reply = self._apply_footer(self._build_approved_reply("", email, style, contextual=True, variation=strategy))
            reply = self._apply_thread_continuity(reply, email)
            similarity = SequenceMatcher(None, _normalize_similarity_text(previous_reply), _normalize_similarity_text(reply)).ratio()
            if similarity < lowest_similarity:
                best_reply = reply
                best_strategy = strategy
                lowest_similarity = similarity
            if similarity < 0.72:
                return reply, strategy
        return best_reply, best_strategy


def _normalize_similarity_text(value: str) -> str:
    return " ".join(value.lower().split())


class InboxMonitor:
    def __init__(self, reader: GmailReader, processor: EmailProcessor, query: str, debug_pipeline: bool = False) -> None:
        self.reader = reader
        self.processor = processor
        self.query = query
        self.debug_pipeline = debug_pipeline

    async def poll_once(self) -> None:
        logger.info("Checking Gmail inbox...", extra={"query": self.query, "debug_gmail_pipeline": self.debug_pipeline})
        emails = await self.reader.fetch_unread(self.query)
        logger.info(
            "Inbox polling batch loaded",
            extra={
                "email_count": len(emails),
                "gmail_ids": [email.gmail_id for email in emails],
                "thread_ids": [email.thread_id for email in emails],
                "debug_gmail_pipeline": self.debug_pipeline,
            },
        )
        for email in emails:
            already_processed = (
                self.processor.db.is_processed(email.gmail_id)
                if hasattr(self.processor, "db")
                else None
            )
            logger.info(
                "Unread email detected",
                extra={"gmail_id": email.gmail_id, "thread_id": email.thread_id, "sender": normalize_email_address(email.sender), "subject": email.subject},
            )
            logger.info(
                "Inbox email processing eligibility",
                extra={
                    "gmail_id": email.gmail_id,
                    "thread_id": email.thread_id,
                    "sender": normalize_email_address(email.sender),
                    "subject": email.subject,
                    "original_recipient": email.original_recipient,
                    "recipient_detection_source": email.recipient_detection_source,
                    "selected_sender_alias": email.selected_sender_alias,
                    "label_ids": email.label_ids,
                    "has_attachments": email.has_attachments,
                    "timestamp": email.timestamp.isoformat() if email.timestamp else None,
                    "already_processed": already_processed,
                    "debug_gmail_pipeline": self.debug_pipeline,
                    "processing_eligible": bool(self.debug_pipeline or not already_processed),
                },
            )
            try:
                await self.processor.process_email(email)
                logger.info(
                    "Inbox email processing complete",
                    extra={
                        "gmail_id": email.gmail_id,
                        "thread_id": email.thread_id,
                        "already_processed_before_processing": already_processed,
                    },
                )
                await self.reader.mark_processed(email.gmail_id)
            except Exception as exc:
                logger.exception("email_processing_failed", extra={"gmail_id": email.gmail_id, "error": str(exc)})
                self.processor.db.log_action("processing_failed", email.gmail_id, {"error": str(exc)})
