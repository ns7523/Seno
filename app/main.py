import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status as http_status
from fastapi.responses import HTMLResponse

from app.ai.classifier import EmailAnalyzer
from app.ai.providers import build_llm_provider
from app.ai.risk_engine import RiskEngine
from app.config import Settings, get_settings
from app.database import Database
from app.gmail.auth import GmailAuth, SCOPES
from app.gmail.reader import GmailReader
from app.gmail.sender import GmailSender
from app.integrations.calendar import GoogleCalendarService
from app.memory.memory import MemoryStore
from app.models.email import detect_original_recipient
from app.services.email_processor import EmailProcessor, InboxMonitor
from app.telegram.bot import TelegramBot
from app.utils.helpers import normalize_email_address
from app.utils.logger import configure_logging, get_logger


logger = get_logger(__name__)


def _mask_email(value: str | None) -> str | None:
    if not value or "@" not in value:
        return value
    local, domain = value.split("@", 1)
    if len(local) <= 2:
        masked_local = local[0] + "*"
    else:
        masked_local = f"{local[:2]}***{local[-1]}"
    return f"{masked_local}@{domain}"


def _telegram_payload_log_context(payload: dict[str, Any]) -> dict[str, Any]:
    callback = payload.get("callback_query") or {}
    message = payload.get("message") or {}
    data = str(callback.get("data", ""))
    return {
        "update_id": payload.get("update_id"),
        "callback_query_id": callback.get("id"),
        "callback_data": data[:80],
        "message_id": message.get("message_id"),
    }


def _nested_str(value: dict[str, Any], *path: str) -> str | None:
    current: Any = value
    for item in path:
        if not isinstance(current, dict):
            return None
        current = current.get(item)
    if current is None:
        return None
    return str(current)


def _ignore_sender_visible(settings: Settings, processor: Any, sender: str | None) -> bool:
    sender_email = normalize_email_address(sender or "")
    if not sender_email:
        return False
    if hasattr(processor, "is_protected_sender"):
        return not processor.is_protected_sender(sender_email)
    protected = {
        normalize_email_address(settings.personal_email),
        normalize_email_address(settings.public_email),
        normalize_email_address(settings.default_from_email),
        normalize_email_address(settings.default_sender_alias),
        *(normalize_email_address(alias) for alias in settings.sender_alias_list),
    }
    return sender_email not in protected


def require_admin(request: Request, x_admin_secret: str | None = Header(default=None)) -> None:
    """Protect operational endpoints that can expose metadata or trigger work."""
    settings: Settings = request.app.state.settings
    if not settings.admin_secret or x_admin_secret != settings.admin_secret:
        raise HTTPException(status_code=http_status.HTTP_401_UNAUTHORIZED, detail="Invalid admin secret")


def build_components(settings: Settings, db: Database) -> dict[str, Any]:
    allow_interactive_google_oauth = settings.environment != "production"
    gmail_service = GmailAuth(
        settings.gmail_client_secrets_file,
        settings.gmail_token_file,
        allow_interactive_oauth=allow_interactive_google_oauth,
    ).build_service()
    calendar_service = (
        GoogleCalendarService(
            client_secrets_file=settings.gmail_client_secrets_file,
            token_file=settings.gmail_token_file,
            calendar_id=settings.google_calendar_id,
            allow_interactive_oauth=allow_interactive_google_oauth,
        )
        if settings.google_calendar_enabled
        else None
    )
    memory = MemoryStore(db)
    telegram = TelegramBot(settings.telegram_bot_token, settings.telegram_chat_id, debug_workflow=settings.seno_debug_workflow)
    provider_model = settings.groq_model if settings.llm_provider == "groq" else settings.openai_model
    analyzer = EmailAnalyzer(
        api_key=settings.openai_api_key,
        model=provider_model,
        timeout=settings.openai_timeout_seconds,
        auto_reply_threshold=settings.auto_reply_risk_threshold,
        min_confidence=settings.min_auto_reply_confidence,
        provider=build_llm_provider(settings),
    )
    reader = GmailReader(
        gmail_service,
        settings.gmail_user_id,
        settings.max_email_body_chars,
        debug_pipeline=settings.debug_gmail_pipeline,
    )
    processor = EmailProcessor(
        db=db,
        memory=memory,
        analyzer=analyzer,
        gmail_sender=GmailSender(
            gmail_service,
            settings.gmail_user_id,
            from_email=settings.default_from_email or settings.default_sender_alias or settings.public_email,
            sender_aliases=settings.sender_alias_list,
            personal_email=settings.personal_email,
            reply_from_original_recipient=settings.reply_from_original_recipient,
            allow_contextual_sender_override=settings.allow_sender_alias_context_override,
        ),
        telegram=telegram,
        gmail_manager=reader,
        risk_engine=RiskEngine(settings.auto_reply_risk_threshold),
        auto_reply_threshold=settings.auto_reply_risk_threshold,
        min_confidence=settings.min_auto_reply_confidence,
        debug_gmail_pipeline=settings.debug_gmail_pipeline,
        debug_workflow=settings.seno_debug_workflow,
        email_footer_mode=settings.email_footer_mode,
        calendar_service=calendar_service,
        protected_ignore_addresses=[
            settings.personal_email,
            settings.public_email,
            settings.default_from_email,
            settings.default_sender_alias,
            *settings.sender_alias_list,
        ],
    )
    monitor = InboxMonitor(
        reader=reader,
        processor=processor,
        query=settings.gmail_query,
        debug_pipeline=settings.debug_gmail_pipeline,
    )
    return {"memory": memory, "telegram": telegram, "processor": processor, "monitor": monitor}


def create_app(database_url: str | None = None, start_scheduler: bool = True) -> FastAPI:
    settings = get_settings()
    if database_url:
        settings = settings.model_copy(update={"database_url": database_url, "environment": "test"})
    configure_logging(settings.log_level)
    db = Database(settings.database_url)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        db.init_db()
        cleanup_counts = db.cleanup_stale_workflows()
        logger.info("stale_workflow_cleanup_complete", extra=cleanup_counts)
        settings.validate_runtime()
        scheduler: AsyncIOScheduler | None = None
        components: dict[str, Any] = {}
        should_start_monitor = start_scheduler and settings.inbox_monitor_enabled
        if should_start_monitor:
            components = build_components(settings, db)
            scheduler = AsyncIOScheduler(timezone="UTC")
            scheduler.add_job(
                components["monitor"].poll_once,
                "interval",
                seconds=settings.gmail_poll_interval_seconds,
                id="gmail-inbox-poll",
                max_instances=3,
                coalesce=True,
                misfire_grace_time=60,
                replace_existing=True,
            )
            scheduler.start()
            logger.info("scheduler_started", extra={"job_id": "gmail-inbox-poll", "seconds": settings.gmail_poll_interval_seconds})
        else:
            components = {
                "telegram": TelegramBot(
                    settings.telegram_bot_token,
                    settings.telegram_chat_id,
                    debug_workflow=settings.seno_debug_workflow,
                ),
                "processor": None,
            }
            logger.info("inbox_monitor_disabled")
        app.state.settings = settings
        app.state.db = db
        app.state.scheduler = scheduler
        app.state.components = components
        app.state.telegram_callback_locks = {}
        try:
            yield
        finally:
            if scheduler:
                scheduler.shutdown(wait=False)

    app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)

    def _approval_workflow_context(
        db: Database,
        *,
        approval_id: int | None,
        action: str | None = None,
        callback_id: str | None = None,
        callback_data: str | None = None,
        update_id: int | None = None,
        lock_owner: str | None = None,
        stage: str | None = None,
    ) -> dict[str, Any]:
        context: dict[str, Any] = {
            "stage": stage,
            "approval_id": approval_id,
            "action": action,
            "callback_query_id": callback_id,
            "callback_data": (callback_data or "")[:120],
            "update_id": update_id,
            "lock_owner": lock_owner,
        }
        if approval_id is not None:
            try:
                context["approval"] = db.get_approval_debug_snapshot(approval_id)
            except Exception as exc:
                context["approval"] = {
                    "approval_id": approval_id,
                    "snapshot_error": str(exc),
                    "snapshot_error_type": type(exc).__name__,
                }
        if callback_id:
            try:
                context["callback"] = db.get_telegram_callback_debug_snapshot(callback_id)
            except Exception as exc:
                context["callback"] = {
                    "callback_query_id": callback_id,
                    "snapshot_error": str(exc),
                    "snapshot_error_type": type(exc).__name__,
                }
        if update_id is not None:
            try:
                context["update"] = db.get_telegram_update_debug_snapshot(update_id)
            except Exception as exc:
                context["update"] = {
                    "update_id": update_id,
                    "snapshot_error": str(exc),
                    "snapshot_error_type": type(exc).__name__,
                }
        return context

    def _log_workflow_stage(
        stage: str,
        db: Database,
        *,
        approval_id: int | None,
        action: str | None = None,
        callback_id: str | None = None,
        callback_data: str | None = None,
        update_id: int | None = None,
        lock_owner: str | None = None,
        level: str = "info",
    ) -> None:
        context = _approval_workflow_context(
            db,
            approval_id=approval_id,
            action=action,
            callback_id=callback_id,
            callback_data=callback_data,
            update_id=update_id,
            lock_owner=lock_owner,
            stage=stage,
        )
        getattr(logger, level)("telegram_workflow_stage", extra=context)

    async def _safe_send_workflow_message(telegram: Any, text: str, *, fallback_text: str | None = None) -> None:
        if not telegram:
            return
        try:
            await telegram.send_message(text)
        except Exception as exc:
            logger.warning(
                "telegram_workflow_message_send_failed",
                extra={"error": str(exc), "error_type": type(exc).__name__},
            )
            if fallback_text:
                try:
                    await telegram.send_message(fallback_text)
                except Exception as fallback_exc:
                    logger.warning(
                        "telegram_workflow_fallback_message_failed",
                        extra={"error": str(fallback_exc), "error_type": type(fallback_exc).__name__},
                    )

    def _workflow_recovery_message(action: str | None, state: str | None) -> str:
        if action == "send":
            if state not in {"draft_preview", "editing"}:
                return (
                    "This Send button is no longer valid for the current approval state. "
                    "Please regenerate the draft or use the latest preview message."
                )
            return "Send could not complete. Please retry Send once, or regenerate the draft if it happens again."
        if action == "regenerate":
            return "Regeneration could not complete. Please retry Regenerate from the current draft preview."
        if action == "edit":
            return "Edit mode could not start. Please retry Edit from the latest approval message."
        if action == "style":
            return "Draft preview generation failed. Please choose the tone again or retry from the latest approval message."
        if action == "cancel":
            return "Cancel could not complete. Please retry from the latest approval message."
        return "That approval action could not be completed. Please retry from the latest approval message."

    def _workflow_exception_message(action: str | None) -> str:
        if action == "send":
            return "Send could not complete. Please retry Send once, or regenerate the draft if it happens again."
        if action == "regenerate":
            return "Regeneration could not complete. Please retry Regenerate from the current draft preview."
        if action == "edit":
            return "Edit mode could not start. Please retry Edit from the latest approval message."
        if action == "style":
            return "Draft preview generation failed. Please choose the tone again or retry from the latest approval message."
        return "Seno could not complete that action. Please retry from the latest approval message."

    @app.get("/")
    async def root() -> dict[str, str]:
        return {"status": "AI Agent Running"}

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/status", dependencies=[Depends(require_admin)])
    async def status(request: Request) -> dict[str, Any]:
        runtime_settings: Settings = request.app.state.settings
        scheduler: AsyncIOScheduler | None = request.app.state.scheduler
        scheduler_jobs = []
        if scheduler:
            scheduler_jobs = [
                {
                    "id": job.id,
                    "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
                }
                for job in scheduler.get_jobs()
            ]
        return {
            "status": "ok",
            "environment": runtime_settings.environment,
            "scheduler": {
                "enabled": runtime_settings.inbox_monitor_enabled,
                "running": bool(scheduler and scheduler.running),
                "jobs": scheduler_jobs,
                "poll_interval_seconds": runtime_settings.gmail_poll_interval_seconds,
            },
            "gmail": {
                "client_secret_configured": bool(runtime_settings.gmail_client_secrets_file),
                "client_secret_exists": bool(
                    runtime_settings.gmail_client_secrets_file
                    and Path(runtime_settings.gmail_client_secrets_file).exists()
                ),
                "token_file": runtime_settings.gmail_token_file,
                "token_exists": Path(runtime_settings.gmail_token_file).exists(),
                "query": runtime_settings.gmail_query,
                "scopes": SCOPES,
            },
            "calendar": {
                "enabled": runtime_settings.google_calendar_enabled,
                "calendar_id": runtime_settings.google_calendar_id,
            },
            "integrations": {
                "LLM_PROVIDER": runtime_settings.llm_provider,
                "GROQ_API_KEY": bool(runtime_settings.groq_api_key),
                "OPENAI_API_KEY": bool(runtime_settings.openai_api_key),
                "TELEGRAM_BOT_TOKEN": bool(runtime_settings.telegram_bot_token),
                "TELEGRAM_CHAT_ID": bool(runtime_settings.telegram_chat_id),
            },
        }

    @app.get("/diagnostics/gmail", dependencies=[Depends(require_admin)])
    async def diagnostics_gmail(
        request: Request,
    ) -> dict[str, Any]:
        runtime_settings: Settings = request.app.state.settings
        client_secret_exists = bool(
            runtime_settings.gmail_client_secrets_file
            and Path(runtime_settings.gmail_client_secrets_file).exists()
        )
        token_exists = Path(runtime_settings.gmail_token_file).exists()
        if not runtime_settings.gmail_client_secrets_file or not client_secret_exists or not token_exists:
            return {
                "status": "not_configured",
                "client_secret_exists": client_secret_exists,
                "token_exists": token_exists,
                "scopes": SCOPES,
            }

        logger.info("Running Gmail diagnostics")
        service = GmailAuth(
            runtime_settings.gmail_client_secrets_file,
            runtime_settings.gmail_token_file,
            allow_interactive_oauth=runtime_settings.environment != "production",
        ).build_service()
        profile = service.users().getProfile(userId=runtime_settings.gmail_user_id).execute()
        result = (
            service.users()
            .messages()
            .list(userId=runtime_settings.gmail_user_id, q=runtime_settings.gmail_query, maxResults=5)
            .execute()
        )
        messages = result.get("messages", [])
        logger.info("Gmail diagnostics complete", extra={"unread_count": len(messages)})
        return {
            "status": "ok",
            "profile_email": _mask_email(profile.get("emailAddress")),
            "messages_total": profile.get("messagesTotal"),
            "threads_total": profile.get("threadsTotal"),
            "unread_batch_count": len(messages),
            "result_size_estimate": result.get("resultSizeEstimate", 0),
            "client_secret_exists": client_secret_exists,
            "token_exists": token_exists,
            "query": runtime_settings.gmail_query,
            "scopes": SCOPES,
        }

    @app.get("/diagnostics/gmail/unread", dependencies=[Depends(require_admin)])
    async def diagnostics_gmail_unread(request: Request, limit: int = 10) -> dict[str, Any]:
        runtime_settings: Settings = request.app.state.settings
        service = GmailAuth(
            runtime_settings.gmail_client_secrets_file,
            runtime_settings.gmail_token_file,
            allow_interactive_oauth=runtime_settings.environment != "production",
        ).build_service()
        query = "is:unread" if runtime_settings.debug_gmail_pipeline else runtime_settings.gmail_query
        max_results = max(1, min(limit, 25))
        result = (
            service.users()
            .messages()
            .list(userId=runtime_settings.gmail_user_id, q=query, maxResults=max_results)
            .execute()
        )
        messages = result.get("messages", [])
        previews: list[dict[str, Any]] = []
        for item in messages[:max_results]:
            gmail_id = item.get("id")
            if not gmail_id:
                previews.append({"id": None, "skip_reason": "missing_gmail_id"})
                continue
            raw = (
                service.users()
                .messages()
                .get(
                    userId=runtime_settings.gmail_user_id,
                    id=gmail_id,
                    format="metadata",
                    metadataHeaders=[
                        "From",
                        "Subject",
                        "To",
                        "Cc",
                        "Delivered-To",
                        "X-Forwarded-To",
                        "X-Original-To",
                        "X-Envelope-To",
                        "Envelope-To",
                        "Reply-To",
                    ],
                )
                .execute()
            )
            headers = {
                header.get("name", "").lower(): header.get("value", "")
                for header in raw.get("payload", {}).get("headers", [])
            }
            original_recipient, detection_source = detect_original_recipient(raw)
            previews.append(
                {
                    "id": gmail_id,
                    "thread_id": raw.get("threadId") or item.get("threadId"),
                    "label_ids": raw.get("labelIds", []),
                    "sender": headers.get("from", ""),
                    "recipient_headers": {
                        key: value
                        for key, value in headers.items()
                        if key
                        in {
                            "to",
                            "cc",
                            "delivered-to",
                            "x-forwarded-to",
                            "x-original-to",
                            "x-envelope-to",
                            "envelope-to",
                            "reply-to",
                        }
                    },
                    "detected_original_recipient": original_recipient,
                    "recipient_detection_source": detection_source,
                    "subject": headers.get("subject", ""),
                    "internal_date": raw.get("internalDate"),
                }
            )
        logger.info(
            "Gmail unread diagnostics complete",
            extra={"query": query, "raw_message_count": len(messages), "gmail_message_ids": [item.get("id") for item in messages]},
        )
        return {
            "status": "ok",
            "query": query,
            "result_size_estimate": result.get("resultSizeEstimate", 0),
            "raw_message_count": len(messages),
            "messages": previews,
        }

    async def _handle_telegram_webhook(request: Request) -> dict[str, Any]:
        runtime_settings: Settings = request.app.state.settings
        payload = await request.json()
        logger.info("telegram_webhook_received", extra=_telegram_payload_log_context(payload))
        # Telegram update IDs are monotonic; keeping seen IDs blocks simple replay attempts.
        update_id = payload.get("update_id")
        update_id_int: int | None = None
        if runtime_settings.telegram_enable_duplicate_protection and update_id is not None:
            try:
                update_id_int = int(update_id)
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=400, detail="Invalid Telegram update id") from exc
            update_state = request.app.state.db.begin_telegram_update(update_id_int)
            if update_state == "completed":
                logger.info("telegram_duplicate_update_ignored", extra={"update_id": update_id_int, "update_state": update_state})
                return {"ok": True, "duplicate_update": True}
            if update_state == "processing_duplicate":
                logger.info("telegram_processing_update_ignored", extra={"update_id": update_id_int, "update_state": update_state})
                return {"ok": True, "processing_update": True}
        callback = payload.get("callback_query")
        message = payload.get("message")
        if message and not callback:
            processor = request.app.state.components.get("processor")
            if processor is None:
                raise HTTPException(status_code=503, detail="Processor not initialized")
            text = str(message.get("text", "")).strip()
            if not text:
                if update_id_int is not None:
                    request.app.state.db.complete_telegram_update(update_id_int)
                return {"ok": True, "ignored": "empty_message"}
            chat_id = _nested_str(message, "chat", "id")
            user_id = _nested_str(message, "from", "id")
            reply_to_message_id = _nested_str(message, "reply_to_message", "message_id")
            handled = await processor.send_edited_reply(
                text,
                telegram_chat_id=chat_id,
                telegram_user_id=user_id,
                reply_to_message_id=reply_to_message_id,
            )
            telegram = request.app.state.components.get("telegram")
            if telegram:
                await telegram.send_message("Edited reply sent." if handled else "No active edit request found.")
            if update_id_int is not None:
                request.app.state.db.complete_telegram_update(update_id_int)
            return {"ok": True, "handled": handled}
        if not callback:
            if update_id_int is not None:
                request.app.state.db.complete_telegram_update(update_id_int)
            return {"ok": True, "ignored": "not_callback"}
        data = callback.get("data", "")
        try:
            parts = data.split(":")
            action = parts[0]
            approval_id_raw = parts[1]
            approval_id = int(approval_id_raw)
        except (IndexError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="Invalid callback data") from exc
        if action not in {
            "approve",
            "reject",
            "style",
            "send",
            "edit",
            "delete",
            "confirm_delete",
            "regenerate",
            "risk",
            "full",
            "cancel",
            "snooze",
            "handled",
            "calendar",
            "alt_time",
            "ignore_sender",
            "confirm_ignore_sender",
            "ignore_thread",
            "pin_sender",
            "auto_handle_similar",
            "menu_tone",
            "menu_regen",
            "menu_controls",
            "menu_snooze",
            "menu_info",
            "qtone",
            "draft_better",
            "regen_reason",
            "regen_reason_apply",
            "more",
            "back",
        }:
            raise HTTPException(status_code=400, detail="Unknown action")

        processor = request.app.state.components.get("processor")
        telegram = request.app.state.components.get("telegram")
        if processor is None:
            raise HTTPException(status_code=503, detail="Processor not initialized")
        callback_id = str(callback.get("id") or "")
        _log_workflow_stage(
            "callback_parsed",
            request.app.state.db,
            approval_id=approval_id,
            action=action,
            callback_id=callback_id,
            callback_data=data,
            update_id=update_id_int,
        )
        if (
            runtime_settings.telegram_enable_idempotency
            and runtime_settings.telegram_enable_callback_locking
            and callback_id
        ):
            callback_state = request.app.state.db.begin_telegram_callback(callback_id, approval_id=approval_id, action=action)
            if callback_state == "completed":
                logger.info("telegram_duplicate_callback_ignored", extra={"callback_query_id": callback_id, "approval_id": approval_id, "action": action})
                if update_id_int is not None:
                    request.app.state.db.complete_telegram_update(update_id_int)
                return {"ok": True, "handled": False, "duplicate_callback": True}
            if callback_state == "processing_duplicate":
                logger.info("telegram_processing_callback_ignored", extra={"callback_query_id": callback_id, "approval_id": approval_id, "action": action})
                if update_id_int is not None:
                    request.app.state.db.complete_telegram_update(update_id_int)
                return {"ok": True, "handled": False, "processing_callback": True}
        elif callback_id:
            callback_state = request.app.state.db.begin_telegram_callback(callback_id, approval_id=approval_id, action=action)
            if callback_state == "completed":
                if update_id_int is not None:
                    request.app.state.db.complete_telegram_update(update_id_int)
                return {"ok": True, "handled": False, "duplicate_callback": True}
        if telegram and callback_id:
            logger.info(
                "telegram_callback_acknowledging",
                extra={"callback_query_id": callback_id, "approval_id": approval_id, "action": action},
            )
            await telegram.answer_callback(callback_id, "Processing")
            logger.info(
                "telegram_callback_transition_requested",
                extra=_approval_workflow_context(
                    request.app.state.db,
                    approval_id=approval_id,
                    action=action,
                    callback_id=callback_id,
                    callback_data=data,
                    update_id=update_id_int,
                    stage="transition_requested",
                ),
            )

        lock_owner = callback_id or f"{action}:{approval_id}:{update_id_int or 'no-update'}"
        durable_lock_acquired = False
        if runtime_settings.telegram_enable_callback_locking:
            lock_key = str(approval_id)
            request.app.state.db.cleanup_stale_workflows()
            logger.info(
                "telegram_callback_lock_waiting",
                extra=_approval_workflow_context(
                    request.app.state.db,
                    approval_id=approval_id,
                    action=action,
                    callback_id=callback_id,
                    callback_data=data,
                    update_id=update_id_int,
                    lock_owner=lock_owner,
                    stage="lock_waiting",
                ) | {"lock_key": lock_key},
            )
            for _ in range(50):
                if request.app.state.db.acquire_approval_lock(approval_id, lock_owner, ttl_seconds=30):
                    durable_lock_acquired = True
                    break
                await asyncio.sleep(0.1)
            if not durable_lock_acquired:
                logger.warning(
                    "telegram_callback_lock_timeout",
                    extra=_approval_workflow_context(
                        request.app.state.db,
                        approval_id=approval_id,
                        action=action,
                        callback_id=callback_id,
                        callback_data=data,
                        update_id=update_id_int,
                        lock_owner=lock_owner,
                        stage="lock_timeout",
                    ),
                )
                if telegram:
                    await telegram.send_message("This approval is already processing. Please try again shortly.")
                if callback_id:
                    request.app.state.db.fail_telegram_callback(callback_id, "approval lock timeout")
                if update_id_int is not None:
                    request.app.state.db.fail_telegram_update(update_id_int, "approval lock timeout")
                return {"ok": True, "handled": False, "lock_timeout": True}
            logger.info(
                "telegram_callback_lock_acquired",
                extra=_approval_workflow_context(
                    request.app.state.db,
                    approval_id=approval_id,
                    action=action,
                    callback_id=callback_id,
                    callback_data=data,
                    update_id=update_id_int,
                    lock_owner=lock_owner,
                    stage="lock_acquired",
                ) | {"lock_key": lock_key},
            )
        try:
            _log_workflow_stage(
                "action_enter",
                request.app.state.db,
                approval_id=approval_id,
                action=action,
                callback_id=callback_id,
                callback_data=data,
                update_id=update_id_int,
                lock_owner=lock_owner,
            )
            if action == "approve":
                handled = await processor.begin_approval(approval_id)
                if handled and telegram:
                    _log_workflow_stage(
                        "telegram_style_selection_send_start",
                        request.app.state.db,
                        approval_id=approval_id,
                        action=action,
                        callback_id=callback_id,
                        callback_data=data,
                        update_id=update_id_int,
                        lock_owner=lock_owner,
                    )
                    await telegram.send_style_selection(approval_id)
            elif action == "reject":
                handled = await processor.reject_approval(approval_id)
                if handled and telegram:
                    if hasattr(telegram, "clear_inline_keyboard"):
                        await telegram.clear_inline_keyboard(
                            chat_id=_nested_str(callback, "message", "chat", "id"),
                            message_id=_nested_str(callback, "message", "message_id"),
                        )
                    if hasattr(telegram, "send_reject_confirmation"):
                        await telegram.send_reject_confirmation()
            elif action == "style":
                if len(parts) != 3:
                    raise HTTPException(status_code=400, detail="Invalid style callback data")
                logger.info("entered style callback", extra={"approval_id": approval_id, "style": parts[2]})
                draft = await processor.preview_approved_reply(approval_id, style=parts[2])
                handled = bool(draft)
            elif action == "send":
                logger.info("entered send callback", extra={"approval_id": approval_id})
                handled = await processor.send_previewed_reply(approval_id)
                if handled and telegram:
                    snapshot = request.app.state.db.get_approval_debug_snapshot(approval_id)
                    if hasattr(telegram, "clear_inline_keyboard"):
                        await telegram.clear_inline_keyboard(
                            chat_id=_nested_str(callback, "message", "chat", "id"),
                            message_id=_nested_str(callback, "message", "message_id"),
                        )
                    if hasattr(telegram, "send_send_confirmation"):
                        await telegram.send_send_confirmation(
                            sender_alias=snapshot.get("selected_sender_alias"),
                            tone=snapshot.get("selected_style"),
                        )
            elif action == "edit":
                logger.info("entered edit callback", extra={"approval_id": approval_id})
                try:
                    handled = processor.start_edit_reply(
                        approval_id,
                        telegram_chat_id=_nested_str(callback, "message", "chat", "id"),
                        telegram_message_id=_nested_str(callback, "message", "message_id"),
                        telegram_user_id=_nested_str(callback, "from", "id"),
                    )
                except TypeError:
                    handled = processor.start_edit_reply(approval_id)
                if handled and telegram:
                    await telegram.send_message(
                        "Send your custom edited reply as the next Telegram message. "
                        "I will use that exact text after safety validation."
                    )
            elif action == "delete":
                handled = await processor.delete_email(approval_id)
                if handled and telegram:
                    if hasattr(telegram, "clear_inline_keyboard"):
                        await telegram.clear_inline_keyboard(
                            chat_id=_nested_str(callback, "message", "chat", "id"),
                            message_id=_nested_str(callback, "message", "message_id"),
                        )
                    if hasattr(telegram, "send_delete_confirmation"):
                        await telegram.send_delete_confirmation()
            elif action == "confirm_delete":
                handled = True
                if telegram:
                    if request.app.state.db.get_approval_status(approval_id):
                        if hasattr(telegram, "send_delete_confirmation_prompt"):
                            await telegram.send_delete_confirmation_prompt(approval_id)
                        else:
                            await _safe_send_workflow_message(telegram, "Delete this email? Confirm from the latest menu.")
                    else:
                        await _safe_send_workflow_message(telegram, "This approval is no longer available.")
            elif action == "more":
                handled = True
                if telegram:
                    if request.app.state.db.get_approval_status(approval_id):
                        snapshot = request.app.state.db.get_approval_debug_snapshot(approval_id)
                        include_ignore_sender = bool(
                            snapshot.get("exists")
                            and _ignore_sender_visible(runtime_settings, processor, str(snapshot.get("sender") or ""))
                        )
                        try:
                            await telegram.send_more_actions(approval_id, include_ignore_sender=include_ignore_sender)
                        except TypeError:
                            await telegram.send_more_actions(approval_id)
                    else:
                        await _safe_send_workflow_message(telegram, "This approval is no longer available.")
            elif action == "menu_tone":
                handled = True
                if telegram and hasattr(telegram, "send_quick_tone_actions"):
                    await telegram.send_quick_tone_actions(approval_id)
            elif action == "menu_regen":
                handled = True
                if telegram and hasattr(telegram, "send_regenerate_actions"):
                    await telegram.send_regenerate_actions(approval_id)
            elif action == "menu_controls":
                handled = True
                if telegram:
                    snapshot = request.app.state.db.get_approval_debug_snapshot(approval_id)
                    include_ignore_sender = bool(
                        snapshot.get("exists")
                        and _ignore_sender_visible(runtime_settings, processor, str(snapshot.get("sender") or ""))
                    )
                    if hasattr(telegram, "send_sender_controls"):
                        await telegram.send_sender_controls(approval_id, include_ignore_sender=include_ignore_sender)
                    else:
                        await telegram.send_more_actions(approval_id, include_ignore_sender=include_ignore_sender)
            elif action == "menu_snooze":
                handled = True
                if telegram and hasattr(telegram, "send_snooze_menu"):
                    await telegram.send_snooze_menu(approval_id)
            elif action == "menu_info":
                handled = True
                if telegram and hasattr(telegram, "send_info_actions"):
                    await telegram.send_info_actions(approval_id)
            elif action == "back":
                handled = True
                if telegram:
                    if request.app.state.db.get_approval_status(approval_id):
                        await telegram.send_primary_actions(approval_id)
                    else:
                        await _safe_send_workflow_message(telegram, "This approval is no longer available.")
            elif action == "regenerate":
                logger.info("entered regenerate callback", extra={"approval_id": approval_id})
                regenerated = await processor.regenerate_reply(approval_id)
                handled = bool(regenerated)
                if handled and telegram:
                    edited = False
                    if hasattr(telegram, "edit_draft_preview"):
                        edited = await telegram.edit_draft_preview(
                            chat_id=_nested_str(callback, "message", "chat", "id"),
                            message_id=_nested_str(callback, "message", "message_id"),
                            approval_id=approval_id,
                            draft=regenerated,
                        )
                    if not edited and hasattr(telegram, "send_draft_preview"):
                        await telegram.send_draft_preview(approval_id, regenerated)
            elif action == "qtone":
                if len(parts) != 3:
                    raise HTTPException(status_code=400, detail="Invalid quick tone callback data")
                tone = parts[2]
                tone_map = {
                    "short": ("normal", "concise_direct"),
                    "warm": ("friendly", "warmer_executive"),
                    "executive": ("formal", "concise_direct"),
                    "formal": ("formal", "collaborative"),
                    "casual": ("friendly", "warmer_executive"),
                }
                style, strategy = tone_map.get(tone, ("normal", "concise_direct"))
                regenerated = await processor.regenerate_reply(
                    approval_id,
                    style=style,
                    strategy=strategy,
                    reason=f"quick_tone_{tone}",
                )
                handled = bool(regenerated)
                if handled and telegram and hasattr(telegram, "send_draft_preview"):
                    await telegram.send_draft_preview(approval_id, regenerated)
            elif action == "draft_better":
                regenerated = await processor.regenerate_reply(
                    approval_id,
                    strategy="collaborative",
                    reason="draft_better",
                )
                handled = bool(regenerated)
                if handled and telegram and hasattr(telegram, "send_draft_preview"):
                    await telegram.send_draft_preview(approval_id, regenerated)
            elif action == "regen_reason":
                handled = True
                if telegram and hasattr(telegram, "send_regenerate_reason_menu"):
                    await telegram.send_regenerate_reason_menu(approval_id)
            elif action == "regen_reason_apply":
                if len(parts) != 3:
                    raise HTTPException(status_code=400, detail="Invalid regenerate reason callback data")
                reason = parts[2]
                reason_map = {
                    "too_formal": ("friendly", "warmer_executive"),
                    "too_long": (None, "concise_direct"),
                    "too_robotic": (None, "warmer_executive"),
                    "more_direct": (None, "concise_direct"),
                    "more_friendly": ("friendly", "warmer_executive"),
                    "stronger_negotiation": (None, "collaborative"),
                }
                style, strategy = reason_map.get(reason, (None, "collaborative"))
                regenerated = await processor.regenerate_reply(
                    approval_id,
                    style=style,
                    strategy=strategy,
                    reason=reason,
                )
                handled = bool(regenerated)
                if handled and telegram and hasattr(telegram, "send_draft_preview"):
                    await telegram.send_draft_preview(approval_id, regenerated)
            elif action == "risk":
                handled = True
                if telegram:
                    await _safe_send_workflow_message(telegram, processor.risk_analysis_text(approval_id), fallback_text="Risk details are unavailable right now.")
            elif action == "full":
                handled = True
                if telegram:
                    await _safe_send_workflow_message(telegram, processor.full_email_text(approval_id), fallback_text="Full email view is unavailable right now.")
            elif action == "cancel":
                logger.info("entered cancel callback", extra={"approval_id": approval_id})
                handled = processor.cancel_approval(approval_id)
                if telegram:
                    await _safe_send_workflow_message(telegram, "Approval cancelled. No reply will be sent.")
            elif action == "snooze":
                snooze_option = parts[2] if len(parts) > 2 else None
                if hasattr(processor, "snooze_approval"):
                    try:
                        handled = processor.snooze_approval(approval_id, option=snooze_option)
                    except TypeError:
                        handled = processor.snooze_approval(approval_id)
                else:
                    handled = False
                if telegram:
                    if hasattr(telegram, "send_snooze_confirmation"):
                        try:
                            await telegram.send_snooze_confirmation(snooze_option)
                        except TypeError:
                            await telegram.send_snooze_confirmation()
                    else:
                        await _safe_send_workflow_message(telegram, "Approval paused. You can return to this email later.")
            elif action == "handled":
                handled = processor.mark_handled(approval_id)
                if telegram:
                    await _safe_send_workflow_message(telegram, "Marked as handled. No reply will be sent.")
            elif action == "ignore_sender":
                snapshot = request.app.state.db.get_approval_debug_snapshot(approval_id)
                risk_score = int(snapshot.get("risk_score") or 100)
                if risk_score >= 50 and telegram and hasattr(telegram, "send_ignore_sender_warning"):
                    await telegram.send_ignore_sender_warning(approval_id)
                    handled = True
                else:
                    handled = processor.ignore_sender(approval_id) if hasattr(processor, "ignore_sender") else False
                if telegram:
                    if risk_score < 50 or not handled:
                        await _safe_send_workflow_message(
                            telegram,
                            "Future emails from this sender will be ignored."
                            if handled
                            else "This sender cannot be ignored from this approval.",
                        )
            elif action == "confirm_ignore_sender":
                handled = processor.ignore_sender(approval_id, confirmed=True) if hasattr(processor, "ignore_sender") else False
                if telegram:
                    await _safe_send_workflow_message(
                        telegram,
                        "Future emails from this sender will be ignored."
                        if handled
                        else "This sender cannot be ignored from this approval.",
                    )
            elif action == "ignore_thread":
                handled = processor.ignore_thread(approval_id) if hasattr(processor, "ignore_thread") else False
                if telegram:
                    await _safe_send_workflow_message(
                        telegram,
                        "Future notifications for this thread will be ignored."
                        if handled
                        else "This thread cannot be ignored from this approval.",
                    )
            elif action == "pin_sender":
                handled = processor.pin_sender(approval_id) if hasattr(processor, "pin_sender") else False
                if telegram:
                    await _safe_send_workflow_message(
                        telegram,
                        "Sender pinned." if handled else "This sender could not be pinned.",
                    )
            elif action == "auto_handle_similar":
                handled = processor.auto_handle_similar(approval_id) if hasattr(processor, "auto_handle_similar") else False
                if telegram:
                    await _safe_send_workflow_message(
                        telegram,
                        "Similar low-risk emails will be auto-handled."
                        if handled
                        else "Similar emails cannot be auto-handled from this approval.",
                    )
            elif action == "calendar":
                event = await processor.create_calendar_event(approval_id)
                handled = bool(event)
                if telegram:
                    if event and getattr(event, "meet_link", None):
                        calendar_message = f"Calendar event created: {event.title} at {event.starts_at}\nMeet: {event.meet_link}"
                    elif event:
                        calendar_message = f"Calendar event created: {event.title} at {event.starts_at}"
                    else:
                        calendar_message = "No calendar event was created."
                    await _safe_send_workflow_message(
                        telegram,
                        calendar_message,
                    )
            elif action == "alt_time":
                alternatives = await processor.suggest_alternative_times(approval_id)
                handled = bool(alternatives)
                if telegram:
                    await _safe_send_workflow_message(telegram, "Suggested alternative times:\n" + "\n".join(alternatives) if alternatives else "No alternatives available.")
            else:
                handled = False
            _log_workflow_stage(
                "action_exit",
                request.app.state.db,
                approval_id=approval_id,
                action=action,
                callback_id=callback_id,
                callback_data=data,
                update_id=update_id_int,
                lock_owner=lock_owner,
            )
        except Exception:
            logger.exception(
                "telegram_callback_action_exception",
                extra=_approval_workflow_context(
                    request.app.state.db,
                    approval_id=approval_id,
                    action=action,
                    callback_id=callback_id,
                    callback_data=data,
                    update_id=update_id_int,
                    lock_owner=lock_owner,
                    stage="action_exception",
                ),
            )
            if telegram:
                await _safe_send_workflow_message(
                    telegram,
                    _workflow_exception_message(action),
                    fallback_text="Seno could not complete that action. Please retry from the latest approval message.",
                )
            raise
        finally:
            if durable_lock_acquired:
                request.app.state.db.release_approval_lock(approval_id, lock_owner)
                logger.info(
                    "telegram_callback_lock_released",
                    extra=_approval_workflow_context(
                        request.app.state.db,
                        approval_id=approval_id,
                        action=action,
                        callback_id=callback_id,
                        callback_data=data,
                        update_id=update_id_int,
                        lock_owner=lock_owner,
                        stage="lock_released",
                    ),
                )
        if not handled and telegram:
            logger.warning(
                "telegram_callback_action_unhandled",
                extra=_approval_workflow_context(
                    request.app.state.db,
                    approval_id=approval_id,
                    action=action,
                    callback_id=callback_id,
                    callback_data=data,
                    update_id=update_id_int,
                    lock_owner=lock_owner,
                    stage="action_unhandled",
                ),
            )
            await _safe_send_workflow_message(
                telegram,
                _workflow_recovery_message(action, request.app.state.db.get_approval_status(approval_id)),
                fallback_text="That approval action could not be completed. Please regenerate the draft or use the latest buttons.",
            )
        logger.info(
            "telegram_callback_transition_completed",
            extra=_approval_workflow_context(
                request.app.state.db,
                approval_id=approval_id,
                action=action,
                callback_id=callback_id,
                callback_data=data,
                update_id=update_id_int,
                lock_owner=lock_owner,
                stage="transition_completed",
            ) | {"handled": handled},
        )
        if callback_id:
            request.app.state.db.complete_telegram_callback(callback_id)
        if update_id_int is not None:
            request.app.state.db.complete_telegram_update(update_id_int)
        return {"ok": True, "handled": handled}

    @app.post("/telegram/webhook")
    async def telegram_webhook(request: Request, x_telegram_bot_api_secret_token: str | None = Header(default=None)) -> dict[str, Any]:
        runtime_settings: Settings = request.app.state.settings
        if not runtime_settings.telegram_webhook_secret or x_telegram_bot_api_secret_token != runtime_settings.telegram_webhook_secret:
            raise HTTPException(status_code=http_status.HTTP_401_UNAUTHORIZED, detail="Invalid Telegram webhook secret")
        if not runtime_settings.enable_webhook_exception_guard:
            return await _handle_telegram_webhook(request)
        try:
            return await _handle_telegram_webhook(request)
        except Exception as exc:
            # Telegram retries every non-2xx webhook response. In safe webhook mode,
            # authenticate first, then log and acknowledge internal workflow failures
            # so retries do not duplicate approval actions or callback processing.
            logger.exception(
                "telegram_webhook_exception_guarded_unparsed",
                extra={"error": str(exc), "error_type": type(exc).__name__},
            )
            try:
                payload = await request.json()
                update_id = payload.get("update_id")
                callback = payload.get("callback_query") or {}
                callback_id = str(callback.get("id") or "")
                callback_data = str(callback.get("data") or "")
                approval_id_for_log: int | None = None
                action_for_log: str | None = None
                try:
                    callback_parts = callback_data.split(":")
                    action_for_log = callback_parts[0] if callback_parts else None
                    approval_id_for_log = int(callback_parts[1]) if len(callback_parts) > 1 else None
                except (TypeError, ValueError):
                    approval_id_for_log = None
                logger.exception(
                    "telegram_webhook_exception_guarded",
                    extra=_approval_workflow_context(
                        request.app.state.db,
                        approval_id=approval_id_for_log,
                        action=action_for_log,
                        callback_id=callback_id,
                        callback_data=callback_data,
                        update_id=int(update_id) if update_id is not None else None,
                        stage="webhook_exception_guard",
                    ) | {"error": str(exc), "error_type": type(exc).__name__},
                )
                if update_id is not None:
                    request.app.state.db.fail_telegram_update(int(update_id), str(exc))
                if callback_id:
                    request.app.state.db.fail_telegram_callback(callback_id, str(exc))
                telegram = request.app.state.components.get("telegram")
                if telegram:
                    await telegram.send_message(
                        "Seno hit a temporary workflow issue while handling that action. "
                        "Please retry the button, regenerate the draft, or send the edited reply again."
                    )
            except Exception as notify_exc:
                logger.warning(
                    "telegram_workflow_failure_notification_failed",
                    extra={"error": str(notify_exc), "error_type": type(notify_exc).__name__},
                )
            return {"ok": False, "error": "webhook_exception_handled"}

    @app.post("/tasks/poll-once", dependencies=[Depends(require_admin)])
    async def poll_once(request: Request) -> dict[str, bool]:
        monitor = request.app.state.components.get("monitor")
        if monitor is None:
            raise HTTPException(status_code=503, detail="Monitor not initialized")
        await monitor.poll_once()
        return {"ok": True}

    @app.get("/dashboard", dependencies=[Depends(require_admin)], response_class=HTMLResponse)
    async def dashboard(request: Request) -> str:
        data = request.app.state.db.dashboard_summary()
        return f"""
        <!doctype html>
        <html>
        <head>
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <title>Executive Communication Assistant</title>
          <style>
            body {{ margin: 0; font-family: Inter, system-ui, sans-serif; background: #f7f8fb; color: #18202f; }}
            header {{ padding: 28px 36px; background: #111827; color: white; }}
            main {{ padding: 28px 36px; display: grid; gap: 20px; }}
            .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; }}
            .card {{ background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 18px; box-shadow: 0 1px 2px rgba(0,0,0,.04); }}
            .metric {{ font-size: 30px; font-weight: 700; }}
            pre {{ white-space: pre-wrap; background: #f3f4f6; padding: 12px; border-radius: 6px; }}
          </style>
        </head>
        <body>
          <header><h1>Executive Communication Assistant</h1><p>Adaptive approval, risk, tone, and relationship intelligence.</p></header>
          <main>
            <section class="grid">
              <div class="card"><div>Pending Approvals</div><div class="metric">{data["pending_approvals"]}</div></div>
              <div class="card"><div>Sent Emails</div><div class="metric">{data["sent_emails"]}</div></div>
              <div class="card"><div>Total Approvals</div><div class="metric">{data["total_approvals"]}</div></div>
            </section>
            <section class="grid">
              <div class="card"><h2>Risk Distribution</h2><pre>{data["risk_distribution"]}</pre></div>
              <div class="card"><h2>Tone Statistics</h2><pre>{data["tone_statistics"]}</pre></div>
              <div class="card"><h2>Relationship Profiles</h2><pre>{data["relationship_profiles"]}</pre></div>
            </section>
          </main>
        </body>
        </html>
        """

    @app.get("/dashboard/api/summary", dependencies=[Depends(require_admin)])
    async def dashboard_summary(request: Request) -> dict[str, Any]:
        return request.app.state.db.dashboard_summary()

    @app.get("/dashboard/api/approvals", dependencies=[Depends(require_admin)])
    async def dashboard_approvals(request: Request) -> dict[str, Any]:
        return {"approvals": request.app.state.db.pending_approvals()}

    @app.get("/admin/ignored-senders", dependencies=[Depends(require_admin)])
    async def ignored_senders(request: Request) -> dict[str, Any]:
        return {"ignored_senders": request.app.state.db.list_ignored_senders()}

    @app.delete("/admin/ignored-senders/{sender_email}", dependencies=[Depends(require_admin)])
    async def unignore_sender(sender_email: str, request: Request) -> dict[str, Any]:
        removed = request.app.state.db.unignore_sender(sender_email)
        return {"ok": True, "sender": sender_email, "removed": removed}

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
