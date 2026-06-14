import hashlib
import json
import re
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from app.models.email import ApprovalRequest, EmailMessage
from app.utils.helpers import normalize_email_address


def _sqlite_path(database_url: str) -> str:
    if database_url.startswith("sqlite:///"):
        path = database_url.removeprefix("sqlite:///")
        if path == ":memory:":
            return path
        return str(Path(path).expanduser())
    if database_url == ":memory:":
        return database_url
    raise ValueError("Only sqlite:/// URLs are supported by the built-in adapter")


@dataclass(slots=True)
class PendingApproval:
    id: int
    email_gmail_id: str
    status: str
    suggested_reply: str
    summary: str = ""
    risk_score: int = 0
    category: str = "General"
    urgency: str = "normal"
    suggested_tone: str = "Normal"
    priority: str = "Medium"
    final_reply: str | None = None
    selected_style: str | None = None
    draft_checksum: str | None = None
    notification_status: str = "pending"
    expires_at: str | None = None


class Database:
    """SQLite adapter.

    Future PostgreSQL migration should keep this public method surface and move
    SQL into a second adapter. Sensitive fields such as email bodies and reply
    drafts are intentionally centralized here so encryption-at-rest hooks can
    be added before writes and after reads without touching Gmail/AI services.
    A production retention job should periodically purge old email bodies,
    decisions, and action logs after the audit window expires.
    """
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self.path = _sqlite_path(database_url)
        self._lock = threading.RLock()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_db(self) -> None:
        with self._lock, self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS emails (
                    gmail_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    sender TEXT NOT NULL,
                    sender_email TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    body TEXT NOT NULL,
                    timestamp TEXT,
                    message_id TEXT,
                    label_ids TEXT NOT NULL DEFAULT '[]',
                    has_attachments INTEGER NOT NULL DEFAULT 0,
                    original_recipient TEXT,
                    selected_sender_alias TEXT,
                    alias_selection_reason TEXT,
                    recipient_detection_source TEXT,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    gmail_id TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    urgency TEXT NOT NULL,
                    risk_score INTEGER NOT NULL,
                    requires_approval INTEGER NOT NULL,
                    never_reply INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    summary TEXT NOT NULL,
                    suggested_reply TEXT NOT NULL,
                    reasons TEXT NOT NULL,
                    tone TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(gmail_id) REFERENCES emails(gmail_id)
                );

                CREATE TABLE IF NOT EXISTS approvals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    gmail_id TEXT NOT NULL UNIQUE,
                    summary TEXT NOT NULL,
                    suggested_reply TEXT NOT NULL,
                    risk_score INTEGER NOT NULL,
                    category TEXT NOT NULL DEFAULT 'General',
                    urgency TEXT NOT NULL DEFAULT 'normal',
                    suggested_tone TEXT NOT NULL DEFAULT 'Normal',
                    priority TEXT NOT NULL DEFAULT 'Medium',
                    final_reply TEXT,
                    selected_style TEXT,
                    draft_checksum TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    telegram_message_id TEXT,
                    notification_status TEXT NOT NULL DEFAULT 'pending',
                    notification_attempts INTEGER NOT NULL DEFAULT 0,
                    last_notification_error TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    expires_at TEXT NOT NULL DEFAULT (datetime('now', '+24 hours')),
                    decided_at TEXT,
                    FOREIGN KEY(gmail_id) REFERENCES emails(gmail_id)
                );

                CREATE TABLE IF NOT EXISTS telegram_updates (
                    update_id INTEGER PRIMARY KEY,
                    status TEXT NOT NULL DEFAULT 'completed',
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    failure_reason TEXT,
                    processing_started_at TEXT,
                    completed_at TEXT,
                    failed_at TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS telegram_callbacks (
                    callback_query_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL DEFAULT 'completed',
                    approval_id INTEGER,
                    action TEXT,
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    failure_reason TEXT,
                    processing_started_at TEXT,
                    completed_at TEXT,
                    failed_at TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS telegram_edit_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    approval_id INTEGER NOT NULL,
                    telegram_chat_id TEXT NOT NULL,
                    telegram_message_id TEXT,
                    telegram_user_id TEXT,
                    status TEXT NOT NULL DEFAULT 'active',
                    expires_at TEXT NOT NULL DEFAULT (datetime('now', '+30 minutes')),
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    completed_at TEXT,
                    FOREIGN KEY(approval_id) REFERENCES approvals(id)
                );

                CREATE INDEX IF NOT EXISTS idx_telegram_edit_sessions_active
                ON telegram_edit_sessions(telegram_chat_id, telegram_user_id, status, expires_at);

                CREATE TABLE IF NOT EXISTS approval_locks (
                    approval_id INTEGER PRIMARY KEY,
                    owner TEXT NOT NULL,
                    acquired_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    expires_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sender_memory (
                    email TEXT PRIMARY KEY,
                    total_interactions INTEGER NOT NULL DEFAULT 0,
                    approvals INTEGER NOT NULL DEFAULT 0,
                    rejections INTEGER NOT NULL DEFAULT 0,
                    auto_replies INTEGER NOT NULL DEFAULT 0,
                    avg_risk REAL NOT NULL DEFAULT 50,
                    last_seen_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    notes TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS relationship_memory (
                    email TEXT PRIMARY KEY,
                    relationship_type TEXT NOT NULL DEFAULT 'unknown',
                    preferred_tone TEXT NOT NULL DEFAULT 'normal',
                    preferred_signoff TEXT,
                    preferred_greeting TEXT,
                    tone_confidence REAL NOT NULL DEFAULT 0,
                    tone_history TEXT NOT NULL DEFAULT '[]',
                    edit_history TEXT NOT NULL DEFAULT '[]',
                    rejection_count INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS thread_summaries (
                    thread_id TEXT PRIMARY KEY,
                    summary TEXT NOT NULL DEFAULT '',
                    commitments TEXT NOT NULL DEFAULT '[]',
                    pending_questions TEXT NOT NULL DEFAULT '[]',
                    scheduling_context TEXT NOT NULL DEFAULT '[]',
                    tone_shifts TEXT NOT NULL DEFAULT '[]',
                    unresolved_items TEXT NOT NULL DEFAULT '[]',
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS ignored_senders (
                    email TEXT PRIMARY KEY,
                    reason TEXT,
                    ignored_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS ignored_threads (
                    thread_id TEXT PRIMARY KEY,
                    reason TEXT,
                    ignored_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS pinned_senders (
                    email TEXT PRIMARY KEY,
                    reason TEXT,
                    pinned_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS auto_handle_senders (
                    email TEXT PRIMARY KEY,
                    reason TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS vector_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT NOT NULL,
                    namespace TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS action_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    gmail_id TEXT,
                    action TEXT NOT NULL,
                    details TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            self._ensure_approval_delivery_columns(conn)
            self._ensure_email_routing_columns(conn)
            self._ensure_telegram_workflow_columns(conn)

    def _ensure_approval_delivery_columns(self, conn: sqlite3.Connection) -> None:
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(approvals)").fetchall()}
        for name, ddl in {
            "notification_status": "ALTER TABLE approvals ADD COLUMN notification_status TEXT NOT NULL DEFAULT 'pending'",
            "notification_attempts": "ALTER TABLE approvals ADD COLUMN notification_attempts INTEGER NOT NULL DEFAULT 0",
            "last_notification_error": "ALTER TABLE approvals ADD COLUMN last_notification_error TEXT",
            "expires_at": "ALTER TABLE approvals ADD COLUMN expires_at TEXT",
            "category": "ALTER TABLE approvals ADD COLUMN category TEXT NOT NULL DEFAULT 'General'",
            "urgency": "ALTER TABLE approvals ADD COLUMN urgency TEXT NOT NULL DEFAULT 'normal'",
            "suggested_tone": "ALTER TABLE approvals ADD COLUMN suggested_tone TEXT NOT NULL DEFAULT 'Normal'",
            "priority": "ALTER TABLE approvals ADD COLUMN priority TEXT NOT NULL DEFAULT 'Medium'",
            "final_reply": "ALTER TABLE approvals ADD COLUMN final_reply TEXT",
            "selected_style": "ALTER TABLE approvals ADD COLUMN selected_style TEXT",
            "draft_checksum": "ALTER TABLE approvals ADD COLUMN draft_checksum TEXT",
        }.items():
            if name not in existing:
                conn.execute(ddl)
        relationship_existing = {row["name"] for row in conn.execute("PRAGMA table_info(relationship_memory)").fetchall()}
        for name, ddl in {
            "preferred_greeting": "ALTER TABLE relationship_memory ADD COLUMN preferred_greeting TEXT",
            "tone_confidence": "ALTER TABLE relationship_memory ADD COLUMN tone_confidence REAL NOT NULL DEFAULT 0",
        }.items():
            if name not in relationship_existing:
                conn.execute(ddl)
        conn.execute(
            "UPDATE approvals SET expires_at = datetime(created_at, '+24 hours') WHERE expires_at IS NULL"
        )

    def _ensure_email_routing_columns(self, conn: sqlite3.Connection) -> None:
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(emails)").fetchall()}
        for name, ddl in {
            "original_recipient": "ALTER TABLE emails ADD COLUMN original_recipient TEXT",
            "selected_sender_alias": "ALTER TABLE emails ADD COLUMN selected_sender_alias TEXT",
            "alias_selection_reason": "ALTER TABLE emails ADD COLUMN alias_selection_reason TEXT",
            "recipient_detection_source": "ALTER TABLE emails ADD COLUMN recipient_detection_source TEXT",
        }.items():
            if name not in existing:
                conn.execute(ddl)

    def _ensure_telegram_workflow_columns(self, conn: sqlite3.Connection) -> None:
        update_existing = {row["name"] for row in conn.execute("PRAGMA table_info(telegram_updates)").fetchall()}
        for name, ddl in {
            "status": "ALTER TABLE telegram_updates ADD COLUMN status TEXT NOT NULL DEFAULT 'completed'",
            "retry_count": "ALTER TABLE telegram_updates ADD COLUMN retry_count INTEGER NOT NULL DEFAULT 0",
            "failure_reason": "ALTER TABLE telegram_updates ADD COLUMN failure_reason TEXT",
            "processing_started_at": "ALTER TABLE telegram_updates ADD COLUMN processing_started_at TEXT",
            "completed_at": "ALTER TABLE telegram_updates ADD COLUMN completed_at TEXT",
            "failed_at": "ALTER TABLE telegram_updates ADD COLUMN failed_at TEXT",
            "created_at": "ALTER TABLE telegram_updates ADD COLUMN created_at TEXT",
        }.items():
            if name not in update_existing:
                conn.execute(ddl)
        conn.execute("UPDATE telegram_updates SET created_at = CURRENT_TIMESTAMP WHERE created_at IS NULL")
        callback_existing = {row["name"] for row in conn.execute("PRAGMA table_info(telegram_callbacks)").fetchall()}
        for name, ddl in {
            "status": "ALTER TABLE telegram_callbacks ADD COLUMN status TEXT NOT NULL DEFAULT 'completed'",
            "approval_id": "ALTER TABLE telegram_callbacks ADD COLUMN approval_id INTEGER",
            "action": "ALTER TABLE telegram_callbacks ADD COLUMN action TEXT",
            "retry_count": "ALTER TABLE telegram_callbacks ADD COLUMN retry_count INTEGER NOT NULL DEFAULT 0",
            "failure_reason": "ALTER TABLE telegram_callbacks ADD COLUMN failure_reason TEXT",
            "processing_started_at": "ALTER TABLE telegram_callbacks ADD COLUMN processing_started_at TEXT",
            "completed_at": "ALTER TABLE telegram_callbacks ADD COLUMN completed_at TEXT",
            "failed_at": "ALTER TABLE telegram_callbacks ADD COLUMN failed_at TEXT",
            "created_at": "ALTER TABLE telegram_callbacks ADD COLUMN created_at TEXT",
        }.items():
            if name not in callback_existing:
                conn.execute(ddl)
        conn.execute("UPDATE telegram_callbacks SET created_at = CURRENT_TIMESTAMP WHERE created_at IS NULL")
        edit_existing = {row["name"] for row in conn.execute("PRAGMA table_info(telegram_edit_sessions)").fetchall()}
        for name, ddl in {
            "created_at": "ALTER TABLE telegram_edit_sessions ADD COLUMN created_at TEXT",
            "completed_at": "ALTER TABLE telegram_edit_sessions ADD COLUMN completed_at TEXT",
        }.items():
            if name not in edit_existing:
                conn.execute(ddl)
        conn.execute("UPDATE telegram_edit_sessions SET created_at = CURRENT_TIMESTAMP WHERE created_at IS NULL")
        lock_existing = {row["name"] for row in conn.execute("PRAGMA table_info(approval_locks)").fetchall()}
        if "acquired_at" not in lock_existing:
            conn.execute("ALTER TABLE approval_locks ADD COLUMN acquired_at TEXT")
        conn.execute("UPDATE approval_locks SET acquired_at = CURRENT_TIMESTAMP WHERE acquired_at IS NULL")

    def is_processed(self, gmail_id: str) -> bool:
        with self._lock, self.connect() as conn:
            row = conn.execute("SELECT status FROM emails WHERE gmail_id = ?", (gmail_id,)).fetchone()
            return bool(
                row
                and row["status"]
                in {
                    "processed",
                    "auto_replied",
                    "pending_approval",
                    "ignored",
                    "ignored_sender",
                    "ignored_thread",
                    "auto_handled_similar",
                    "approved_sent",
                    "deleted",
                    "handled",
                    "snoozed",
                }
            )

    def record_email(self, email: EmailMessage, status: str) -> None:
        with self._lock, self.connect() as conn:
            conn.execute(
                """
                INSERT INTO emails (
                    gmail_id, thread_id, sender, sender_email, subject, body, timestamp,
                    message_id, label_ids, has_attachments, original_recipient,
                    selected_sender_alias, alias_selection_reason, recipient_detection_source, status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(gmail_id) DO UPDATE SET
                    status=excluded.status,
                    original_recipient=excluded.original_recipient,
                    selected_sender_alias=excluded.selected_sender_alias,
                    alias_selection_reason=excluded.alias_selection_reason,
                    recipient_detection_source=excluded.recipient_detection_source,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    email.gmail_id,
                    email.thread_id,
                    email.sender,
                    normalize_email_address(email.sender),
                    email.subject,
                    email.body,
                    email.timestamp.isoformat() if email.timestamp else None,
                    email.message_id,
                    json.dumps(email.label_ids),
                    int(email.has_attachments),
                    email.original_recipient,
                    email.selected_sender_alias,
                    email.alias_selection_reason,
                    email.recipient_detection_source,
                    status,
                ),
            )

    def update_email_status(self, gmail_id: str, status: str) -> None:
        with self._lock, self.connect() as conn:
            conn.execute("UPDATE emails SET status = ?, updated_at=CURRENT_TIMESTAMP WHERE gmail_id = ?", (status, gmail_id))

    def get_email_status(self, gmail_id: str) -> str | None:
        with self._lock, self.connect() as conn:
            row = conn.execute("SELECT status FROM emails WHERE gmail_id = ?", (gmail_id,)).fetchone()
        return row["status"] if row else None

    def record_decision(self, gmail_id: str, analysis: Any) -> None:
        with self._lock, self.connect() as conn:
            conn.execute(
                """
                INSERT INTO decisions (
                    gmail_id, intent, urgency, risk_score, requires_approval, never_reply,
                    confidence, summary, suggested_reply, reasons, tone
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    gmail_id,
                    analysis.intent,
                    analysis.urgency,
                    analysis.risk_score,
                    int(analysis.requires_approval),
                    int(analysis.never_reply),
                    analysis.confidence,
                    analysis.summary,
                    analysis.suggested_reply,
                    json.dumps(analysis.reasons),
                    analysis.tone,
                ),
            )

    def create_approval(
        self,
        email: EmailMessage,
        summary: str,
        suggested_reply: str,
        risk_score: int,
        *,
        category: str = "General",
        urgency: str = "normal",
        suggested_tone: str = "Normal",
        priority: str = "Medium",
        reply_recommendation: str = "Review and choose a reply style.",
        confidence: float = 0.0,
        risk_explanation: list[str] | None = None,
    ) -> ApprovalRequest:
        with self._lock, self.connect() as conn:
            conn.execute(
                """
                INSERT INTO approvals (
                    gmail_id, summary, suggested_reply, risk_score, category,
                    urgency, suggested_tone, priority, status, expires_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', datetime('now', '+24 hours'))
                ON CONFLICT(gmail_id) DO NOTHING
                """,
                (email.gmail_id, summary, suggested_reply, risk_score, category, urgency, suggested_tone, priority),
            )
            row = conn.execute(
                "SELECT id FROM approvals WHERE gmail_id = ?",
                (email.gmail_id,),
            ).fetchone()
            assert row is not None
            return ApprovalRequest(
                id=row["id"],
                email=email,
                summary=summary,
                suggested_reply=suggested_reply,
                risk_score=risk_score,
                category=category,
                urgency=urgency,
                suggested_tone=suggested_tone,
                priority=priority,
                reply_recommendation=reply_recommendation,
                confidence=confidence,
                risk_explanation=risk_explanation or [],
            )

    def get_pending_approval_by_email(self, gmail_id: str) -> PendingApproval | None:
        with self._lock, self.connect() as conn:
            row = conn.execute(
                """
                SELECT id, gmail_id, status, suggested_reply, summary, risk_score,
                       category, urgency, suggested_tone, priority, final_reply,
                       selected_style, draft_checksum, notification_status, expires_at
                FROM approvals
                WHERE gmail_id = ? AND status IN ('pending', 'awaiting_style', 'editing', 'draft_preview', 'send_failed')
                """,
                (gmail_id,),
            ).fetchone()
        if not row:
            return None
        return PendingApproval(
            row["id"],
            row["gmail_id"],
            row["status"],
            row["suggested_reply"],
            row["summary"],
            row["risk_score"],
            row["category"],
            row["urgency"],
            row["suggested_tone"],
            row["priority"],
            row["final_reply"],
            row["selected_style"],
            row["draft_checksum"],
            row["notification_status"],
            row["expires_at"],
        )

    def get_pending_approval(self, approval_id: int) -> tuple[PendingApproval, EmailMessage] | None:
        with self._lock, self.connect() as conn:
            row = conn.execute(
                """
                SELECT a.id, a.gmail_id, a.status, a.suggested_reply, a.summary, a.risk_score,
                       a.category, a.urgency, a.suggested_tone, a.priority, a.final_reply,
                       a.selected_style, a.draft_checksum, a.notification_status, a.expires_at,
                       e.thread_id, e.sender, e.subject,
                       e.body, e.timestamp, e.message_id, e.label_ids, e.has_attachments,
                       e.original_recipient, e.selected_sender_alias, e.alias_selection_reason,
                       e.recipient_detection_source
                FROM approvals a
                JOIN emails e ON e.gmail_id = a.gmail_id
                WHERE a.id = ? AND a.status IN ('pending', 'awaiting_style', 'editing', 'draft_preview', 'send_failed')
                """,
                (approval_id,),
            ).fetchone()
            if row and self._is_expired(row["expires_at"]):
                conn.execute(
                    "UPDATE approvals SET status = 'expired', decided_at=CURRENT_TIMESTAMP WHERE id = ? AND status = 'pending'",
                    (approval_id,),
                )
                conn.execute(
                    "UPDATE emails SET status = 'approval_expired', updated_at=CURRENT_TIMESTAMP WHERE gmail_id = ?",
                    (row["gmail_id"],),
                )
                return None
        if not row:
            return None
        email = EmailMessage(
            gmail_id=row["gmail_id"],
            thread_id=row["thread_id"],
            sender=row["sender"],
            subject=row["subject"],
            body=row["body"],
            timestamp=None,
            message_id=row["message_id"],
            label_ids=json.loads(row["label_ids"] or "[]"),
            has_attachments=bool(row["has_attachments"]),
            original_recipient=row["original_recipient"],
            selected_sender_alias=row["selected_sender_alias"],
            alias_selection_reason=row["alias_selection_reason"],
            recipient_detection_source=row["recipient_detection_source"],
        )
        approval = PendingApproval(
            row["id"],
            row["gmail_id"],
            row["status"],
            row["suggested_reply"],
            row["summary"],
            row["risk_score"],
            row["category"],
            row["urgency"],
            row["suggested_tone"],
            row["priority"],
            row["final_reply"],
            row["selected_style"],
            row["draft_checksum"],
            row["notification_status"],
            row["expires_at"],
        )
        return approval, email

    def update_approval_notification(self, approval_id: int, status: str, error: str | None = None) -> None:
        with self._lock, self.connect() as conn:
            conn.execute(
                """
                UPDATE approvals
                SET notification_status = ?,
                    notification_attempts = notification_attempts + 1,
                    last_notification_error = ?
                WHERE id = ?
                """,
                (status, error, approval_id),
            )

    def get_approval_status(self, approval_id: int) -> str | None:
        with self._lock, self.connect() as conn:
            row = conn.execute("SELECT status FROM approvals WHERE id = ?", (approval_id,)).fetchone()
            return row["status"] if row else None

    def get_approval_debug_snapshot(self, approval_id: int) -> dict[str, Any]:
        """Return non-secret workflow state for production callback tracing."""
        with self._lock, self.connect() as conn:
            row = conn.execute(
                """
                SELECT a.id, a.gmail_id, a.status, a.risk_score, a.category,
                       a.urgency, a.suggested_tone, a.priority, a.selected_style,
                       a.draft_checksum, a.notification_status, a.expires_at,
                       LENGTH(COALESCE(a.final_reply, '')) AS final_reply_length,
                       CASE WHEN a.final_reply IS NULL OR a.final_reply = '' THEN 0 ELSE 1 END AS preview_exists,
                       e.thread_id, e.sender, e.subject, e.original_recipient,
                       e.selected_sender_alias, e.alias_selection_reason,
                       e.recipient_detection_source
                FROM approvals a
                LEFT JOIN emails e ON e.gmail_id = a.gmail_id
                WHERE a.id = ?
                """,
                (approval_id,),
            ).fetchone()
            if not row:
                return {"approval_id": approval_id, "exists": False}
            sessions = conn.execute(
                """
                SELECT id, status, telegram_chat_id, telegram_message_id,
                       telegram_user_id, expires_at, created_at, completed_at
                FROM telegram_edit_sessions
                WHERE approval_id = ?
                ORDER BY created_at DESC
                LIMIT 5
                """,
                (approval_id,),
            ).fetchall()
            try:
                locks = conn.execute(
                    """
                    SELECT owner, expires_at, acquired_at,
                           expires_at <= CURRENT_TIMESTAMP AS expired
                    FROM approval_locks
                    WHERE approval_id = ?
                    ORDER BY acquired_at DESC
                    """,
                    (approval_id,),
                ).fetchall()
            except sqlite3.OperationalError as exc:
                locks = [{"snapshot_error": str(exc)}]
        return {
            "approval_id": row["id"],
            "exists": True,
            "gmail_id": row["gmail_id"],
            "workflow_state": row["status"],
            "risk_score": row["risk_score"],
            "category": row["category"],
            "urgency": row["urgency"],
            "suggested_tone": row["suggested_tone"],
            "selected_style": row["selected_style"],
            "priority": row["priority"],
            "preview_exists": bool(row["preview_exists"]),
            "final_reply_length": row["final_reply_length"],
            "stored_checksum": row["draft_checksum"],
            "notification_status": row["notification_status"],
            "expires_at": row["expires_at"],
            "thread_id": row["thread_id"],
            "sender": normalize_email_address(row["sender"] or ""),
            "subject": (row["subject"] or "")[:120],
            "original_recipient": row["original_recipient"],
            "selected_sender_alias": row["selected_sender_alias"],
            "alias_selection_reason": row["alias_selection_reason"],
            "recipient_detection_source": row["recipient_detection_source"],
            "edit_sessions": [dict(item) for item in sessions],
            "locks": [dict(item) for item in locks],
        }

    def get_telegram_callback_debug_snapshot(self, callback_query_id: str) -> dict[str, Any]:
        with self._lock, self.connect() as conn:
            row = conn.execute(
                """
                SELECT callback_query_id, status, approval_id, action, retry_count,
                       failure_reason, processing_started_at, completed_at, failed_at
                FROM telegram_callbacks
                WHERE callback_query_id = ?
                """,
                (callback_query_id,),
            ).fetchone()
        return dict(row) if row else {"callback_query_id": callback_query_id, "exists": False}

    def get_telegram_update_debug_snapshot(self, update_id: int) -> dict[str, Any]:
        with self._lock, self.connect() as conn:
            row = conn.execute(
                """
                SELECT update_id, status, retry_count, failure_reason,
                       processing_started_at, completed_at, failed_at
                FROM telegram_updates
                WHERE update_id = ?
                """,
                (update_id,),
            ).fetchone()
        return dict(row) if row else {"update_id": update_id, "exists": False}

    def decide_approval(self, approval_id: int, approved: bool) -> bool:
        status = "approved" if approved else "rejected"
        allowed_statuses = ("draft_preview", "editing", "send_failed") if approved else ("pending", "awaiting_style", "editing", "draft_preview", "send_failed")
        placeholders = ", ".join("?" for _ in allowed_statuses)
        with self._lock, self.connect() as conn:
            cursor = conn.execute(
                f"""
                UPDATE approvals
                SET status = ?, decided_at=CURRENT_TIMESTAMP
                WHERE id = ? AND status IN ({placeholders})
                """,
                (status, approval_id, *allowed_statuses),
            )
            return cursor.rowcount == 1

    def set_approval_status(self, approval_id: int, status: str) -> bool:
        with self._lock, self.connect() as conn:
            cursor = conn.execute(
                """
                UPDATE approvals
                SET status = ?, decided_at = CASE
                    WHEN ? IN ('approved', 'rejected', 'deleted', 'cancelled', 'handled', 'snoozed') THEN CURRENT_TIMESTAMP
                    ELSE decided_at
                END
                  WHERE id = ?
                  AND status IN ('pending', 'awaiting_style', 'editing', 'draft_preview', 'send_failed')
                  AND (
                    (status = 'pending' AND ? IN ('awaiting_style', 'editing', 'rejected', 'deleted', 'cancelled', 'handled', 'snoozed'))
                    OR (status = 'awaiting_style' AND ? IN ('draft_preview', 'editing', 'rejected', 'deleted', 'cancelled', 'handled', 'snoozed'))
                    OR (status = 'draft_preview' AND ? IN ('approved', 'editing', 'deleted', 'cancelled', 'handled', 'sending', 'snoozed'))
                    OR (status = 'editing' AND ? IN ('approved', 'draft_preview', 'deleted', 'cancelled', 'handled', 'sending', 'snoozed'))
                    OR (status = 'send_failed' AND ? IN ('draft_preview', 'editing', 'deleted', 'cancelled', 'handled', 'sending', 'snoozed'))
                  )
                """,
                (status, status, approval_id, status, status, status, status, status),
            )
            return cursor.rowcount == 1

    def begin_approval_send(self, approval_id: int) -> bool:
        with self._lock, self.connect() as conn:
            cursor = conn.execute(
                """
                UPDATE approvals
                SET status = 'sending'
                WHERE id = ? AND status IN ('draft_preview', 'editing', 'send_failed')
                """,
                (approval_id,),
            )
            return cursor.rowcount == 1

    def mark_approval_sent(self, approval_id: int) -> bool:
        with self._lock, self.connect() as conn:
            cursor = conn.execute(
                """
                UPDATE approvals
                SET status = 'approved', decided_at=CURRENT_TIMESTAMP
                WHERE id = ? AND status = 'sending'
                """,
                (approval_id,),
            )
            return cursor.rowcount == 1

    def mark_approval_send_failed(self, approval_id: int) -> None:
        with self._lock, self.connect() as conn:
            conn.execute(
                """
                UPDATE approvals
                SET status = 'send_failed'
                WHERE id = ? AND status = 'sending'
                """,
                (approval_id,),
            )

    def update_approval_reply(self, approval_id: int, suggested_reply: str) -> bool:
        with self._lock, self.connect() as conn:
            cursor = conn.execute(
                """
                UPDATE approvals
                SET suggested_reply = ?
                WHERE id = ? AND status IN ('pending', 'awaiting_style', 'editing', 'draft_preview', 'send_failed')
                """,
                (suggested_reply, approval_id),
            )
            return cursor.rowcount == 1

    def set_approval_draft(self, approval_id: int, final_reply: str, selected_style: str) -> bool:
        checksum = hashlib.sha256(final_reply.encode("utf-8")).hexdigest()
        with self._lock, self.connect() as conn:
            cursor = conn.execute(
                """
                UPDATE approvals
                SET final_reply = ?, selected_style = ?, draft_checksum = ?, status = 'draft_preview'
                WHERE id = ? AND status IN ('pending', 'awaiting_style', 'editing', 'draft_preview', 'send_failed')
                """,
                (final_reply, selected_style, checksum, approval_id),
            )
            return cursor.rowcount == 1

    def get_editing_approval(self) -> tuple[PendingApproval, EmailMessage] | None:
        with self._lock, self.connect() as conn:
            row = conn.execute(
                """
                SELECT a.id, a.gmail_id, a.status, a.suggested_reply, a.summary, a.risk_score,
                       a.category, a.urgency, a.suggested_tone, a.priority, a.final_reply,
                       a.selected_style, a.draft_checksum, a.notification_status, a.expires_at,
                       e.thread_id, e.sender, e.subject,
                       e.body, e.timestamp, e.message_id, e.label_ids, e.has_attachments,
                       e.original_recipient, e.selected_sender_alias, e.alias_selection_reason,
                       e.recipient_detection_source
                FROM approvals a
                JOIN emails e ON e.gmail_id = a.gmail_id
                WHERE a.status IN ('editing', 'send_failed')
                ORDER BY a.created_at DESC
                LIMIT 1
                """
            ).fetchone()
        if not row:
            return None
        email = EmailMessage(
            gmail_id=row["gmail_id"],
            thread_id=row["thread_id"],
            sender=row["sender"],
            subject=row["subject"],
            body=row["body"],
            timestamp=None,
            message_id=row["message_id"],
            label_ids=json.loads(row["label_ids"] or "[]"),
            has_attachments=bool(row["has_attachments"]),
            original_recipient=row["original_recipient"],
            selected_sender_alias=row["selected_sender_alias"],
            alias_selection_reason=row["alias_selection_reason"],
            recipient_detection_source=row["recipient_detection_source"],
        )
        approval = PendingApproval(
            row["id"],
            row["gmail_id"],
            row["status"],
            row["suggested_reply"],
            row["summary"],
            row["risk_score"],
            row["category"],
            row["urgency"],
            row["suggested_tone"],
            row["priority"],
            row["final_reply"],
            row["selected_style"],
            row["draft_checksum"],
            row["notification_status"],
            row["expires_at"],
        )
        return approval, email

    def _approval_email_from_row(self, row: sqlite3.Row) -> tuple[PendingApproval, EmailMessage]:
        email = EmailMessage(
            gmail_id=row["gmail_id"],
            thread_id=row["thread_id"],
            sender=row["sender"],
            subject=row["subject"],
            body=row["body"],
            timestamp=None,
            message_id=row["message_id"],
            label_ids=json.loads(row["label_ids"] or "[]"),
            has_attachments=bool(row["has_attachments"]),
            original_recipient=row["original_recipient"],
            selected_sender_alias=row["selected_sender_alias"],
            alias_selection_reason=row["alias_selection_reason"],
            recipient_detection_source=row["recipient_detection_source"],
        )
        approval = PendingApproval(
            row["id"],
            row["gmail_id"],
            row["status"],
            row["suggested_reply"],
            row["summary"],
            row["risk_score"],
            row["category"],
            row["urgency"],
            row["suggested_tone"],
            row["priority"],
            row["final_reply"],
            row["selected_style"],
            row["draft_checksum"],
            row["notification_status"],
            row["expires_at"],
        )
        return approval, email

    def begin_telegram_update(self, update_id: int, *, stale_after_seconds: int = 300) -> str:
        """Begin durable update processing.

        Completed updates are idempotently ignored. Failed or stale processing
        updates are retried, which prevents safe-webhook mode from permanently
        losing work after an internal exception.
        """
        with self._lock, self.connect() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO telegram_updates (update_id, status, processing_started_at)
                    VALUES (?, 'processing', CURRENT_TIMESTAMP)
                    """,
                    (update_id,),
                )
                return "processing"
            except sqlite3.IntegrityError:
                row = conn.execute("SELECT * FROM telegram_updates WHERE update_id = ?", (update_id,)).fetchone()
                if not row:
                    return "processing"
                if row["status"] == "completed":
                    return "completed"
                stale = conn.execute(
                    """
                    SELECT datetime(COALESCE(processing_started_at, created_at), '+' || ? || ' seconds') <= CURRENT_TIMESTAMP AS is_stale
                    FROM telegram_updates
                    WHERE update_id = ?
                    """,
                    (stale_after_seconds, update_id),
                ).fetchone()["is_stale"]
                if row["status"] == "failed" or stale:
                    conn.execute(
                        """
                        UPDATE telegram_updates
                        SET status = 'processing',
                            retry_count = retry_count + 1,
                            failure_reason = NULL,
                            processing_started_at = CURRENT_TIMESTAMP,
                            failed_at = NULL
                        WHERE update_id = ?
                        """,
                        (update_id,),
                    )
                    return "processing"
                return "processing_duplicate"

    def complete_telegram_update(self, update_id: int) -> None:
        with self._lock, self.connect() as conn:
            conn.execute(
                """
                UPDATE telegram_updates
                SET status = 'completed', completed_at = CURRENT_TIMESTAMP
                WHERE update_id = ?
                """,
                (update_id,),
            )

    def fail_telegram_update(self, update_id: int, reason: str) -> None:
        with self._lock, self.connect() as conn:
            conn.execute(
                """
                UPDATE telegram_updates
                SET status = 'failed', failure_reason = ?, failed_at = CURRENT_TIMESTAMP
                WHERE update_id = ?
                """,
                (reason[:500], update_id),
            )

    def record_telegram_update(self, update_id: int) -> bool:
        return self.begin_telegram_update(update_id) == "processing"

    def begin_telegram_callback(self, callback_query_id: str, *, approval_id: int | None = None, action: str | None = None) -> str:
        """Begin callback processing with retryable failed state."""
        with self._lock, self.connect() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO telegram_callbacks (
                        callback_query_id, status, approval_id, action, processing_started_at
                    )
                    VALUES (?, 'processing', ?, ?, CURRENT_TIMESTAMP)
                    """,
                    (callback_query_id, approval_id, action),
                )
                return "processing"
            except sqlite3.IntegrityError:
                row = conn.execute("SELECT status FROM telegram_callbacks WHERE callback_query_id = ?", (callback_query_id,)).fetchone()
                if not row:
                    return "processing"
                if row["status"] == "failed":
                    conn.execute(
                        """
                        UPDATE telegram_callbacks
                        SET status = 'processing',
                            retry_count = retry_count + 1,
                            failure_reason = NULL,
                            processing_started_at = CURRENT_TIMESTAMP,
                            failed_at = NULL
                        WHERE callback_query_id = ?
                        """,
                        (callback_query_id,),
                    )
                    return "processing"
                if row["status"] == "processing":
                    return "processing_duplicate"
                return row["status"]

    def complete_telegram_callback(self, callback_query_id: str) -> None:
        with self._lock, self.connect() as conn:
            conn.execute(
                "UPDATE telegram_callbacks SET status = 'completed', completed_at = CURRENT_TIMESTAMP WHERE callback_query_id = ?",
                (callback_query_id,),
            )

    def fail_telegram_callback(self, callback_query_id: str, reason: str) -> None:
        with self._lock, self.connect() as conn:
            conn.execute(
                """
                UPDATE telegram_callbacks
                SET status = 'failed', failure_reason = ?, failed_at = CURRENT_TIMESTAMP
                WHERE callback_query_id = ?
                """,
                (reason[:500], callback_query_id),
            )

    def record_telegram_callback(self, callback_query_id: str) -> bool:
        return self.begin_telegram_callback(callback_query_id) == "processing"

    def acquire_approval_lock(self, approval_id: int, owner: str, *, ttl_seconds: int = 30) -> bool:
        with self._lock, self.connect() as conn:
            conn.execute("DELETE FROM approval_locks WHERE expires_at <= CURRENT_TIMESTAMP")
            try:
                conn.execute(
                    """
                    INSERT INTO approval_locks (approval_id, owner, expires_at)
                    VALUES (?, ?, datetime('now', '+' || ? || ' seconds'))
                    """,
                    (approval_id, owner, ttl_seconds),
                )
                return True
            except sqlite3.IntegrityError:
                return False

    def release_approval_lock(self, approval_id: int, owner: str) -> None:
        with self._lock, self.connect() as conn:
            conn.execute("DELETE FROM approval_locks WHERE approval_id = ? AND owner = ?", (approval_id, owner))

    def start_edit_session(
        self,
        approval_id: int,
        *,
        telegram_chat_id: str,
        telegram_message_id: str | None = None,
        telegram_user_id: str | None = None,
        ttl_minutes: int = 30,
    ) -> None:
        with self._lock, self.connect() as conn:
            conn.execute(
                """
                UPDATE telegram_edit_sessions
                SET status = 'cancelled', completed_at = CURRENT_TIMESTAMP
                WHERE approval_id = ? AND status = 'active'
                """,
                (approval_id,),
            )
            conn.execute(
                """
                INSERT INTO telegram_edit_sessions (
                    approval_id, telegram_chat_id, telegram_message_id,
                    telegram_user_id, expires_at
                )
                VALUES (?, ?, ?, ?, datetime('now', '+' || ? || ' minutes'))
                """,
                (approval_id, telegram_chat_id, telegram_message_id, telegram_user_id, ttl_minutes),
            )

    def get_editing_approval_for_session(
        self,
        *,
        telegram_chat_id: str,
        telegram_user_id: str | None = None,
        reply_to_message_id: str | None = None,
    ) -> tuple[PendingApproval, EmailMessage] | None:
        with self._lock, self.connect() as conn:
            conn.execute(
                """
                UPDATE telegram_edit_sessions
                SET status = 'expired', completed_at = CURRENT_TIMESTAMP
                WHERE status = 'active' AND expires_at <= CURRENT_TIMESTAMP
                """
            )
            rows = conn.execute(
                """
                SELECT s.id AS session_id, s.telegram_message_id,
                       a.id, a.gmail_id, a.status, a.suggested_reply, a.summary, a.risk_score,
                       a.category, a.urgency, a.suggested_tone, a.priority, a.final_reply,
                       a.selected_style, a.draft_checksum, a.notification_status, a.expires_at,
                       e.thread_id, e.sender, e.subject,
                       e.body, e.timestamp, e.message_id, e.label_ids, e.has_attachments,
                       e.original_recipient, e.selected_sender_alias, e.alias_selection_reason,
                       e.recipient_detection_source
                FROM telegram_edit_sessions s
                JOIN approvals a ON a.id = s.approval_id
                JOIN emails e ON e.gmail_id = a.gmail_id
                WHERE s.status = 'active'
                  AND s.telegram_chat_id = ?
                  AND (? IS NULL OR s.telegram_user_id IS NULL OR s.telegram_user_id = ?)
                  AND a.status IN ('editing', 'send_failed')
                ORDER BY s.created_at DESC
                LIMIT 2
                """,
                (telegram_chat_id, telegram_user_id, telegram_user_id),
            ).fetchall()
        if not rows:
            return None
        if len(rows) > 1 and reply_to_message_id:
            rows = [row for row in rows if str(row["telegram_message_id"] or "") == str(reply_to_message_id)]
        if len(rows) != 1:
            return None
        row = rows[0]
        return self._approval_email_from_row(row)

    def complete_edit_session(self, approval_id: int) -> None:
        with self._lock, self.connect() as conn:
            conn.execute(
                """
                UPDATE telegram_edit_sessions
                SET status = 'completed', completed_at = CURRENT_TIMESTAMP
                WHERE approval_id = ? AND status = 'active'
                """,
                (approval_id,),
            )

    def cleanup_stale_workflows(self) -> dict[str, int]:
        """Expire abandoned workflow rows.

        This is intentionally database-local today and keeps the adapter
        surface ready for a PostgreSQL implementation with row-level locks and
        scheduled cleanup workers later.
        """
        with self._lock, self.connect() as conn:
            locks = conn.execute("DELETE FROM approval_locks WHERE expires_at <= CURRENT_TIMESTAMP").rowcount
            edits = conn.execute(
                """
                UPDATE telegram_edit_sessions
                SET status = 'expired', completed_at = CURRENT_TIMESTAMP
                WHERE status = 'active' AND expires_at <= CURRENT_TIMESTAMP
                """
            ).rowcount
            updates = conn.execute(
                """
                UPDATE telegram_updates
                SET status = 'failed',
                    failure_reason = COALESCE(failure_reason, 'processing timeout'),
                    failed_at = CURRENT_TIMESTAMP
                WHERE status = 'processing'
                  AND datetime(COALESCE(processing_started_at, created_at), '+10 minutes') <= CURRENT_TIMESTAMP
                """
            ).rowcount
            callbacks = conn.execute(
                """
                UPDATE telegram_callbacks
                SET status = 'failed',
                    failure_reason = COALESCE(failure_reason, 'processing timeout'),
                    failed_at = CURRENT_TIMESTAMP
                WHERE status = 'processing'
                  AND datetime(COALESCE(processing_started_at, created_at), '+10 minutes') <= CURRENT_TIMESTAMP
                """
            ).rowcount
        return {"locks": locks, "edit_sessions": edits, "updates": updates, "callbacks": callbacks}

    def expire_approval(self, approval_id: int) -> bool:
        with self._lock, self.connect() as conn:
            cursor = conn.execute(
                """
                UPDATE approvals
                SET status = 'expired', decided_at=CURRENT_TIMESTAMP
                WHERE id = ? AND status = 'pending'
                """,
                (approval_id,),
            )
            return cursor.rowcount == 1

    def update_sender_memory(
        self,
        sender_email: str,
        *,
        approved: bool,
        auto_replied: bool,
        risk_score: int,
        rejected: bool = False,
    ) -> None:
        sender_email = normalize_email_address(sender_email)
        with self._lock, self.connect() as conn:
            row = conn.execute("SELECT * FROM sender_memory WHERE email = ?", (sender_email,)).fetchone()
            if not row:
                conn.execute(
                    """
                    INSERT INTO sender_memory (
                        email, total_interactions, approvals, rejections, auto_replies, avg_risk
                    )
                    VALUES (?, 1, ?, ?, ?, ?)
                    """,
                    (sender_email, int(approved), int(rejected), int(auto_replied), float(risk_score)),
                )
                return
            total = row["total_interactions"] + 1
            avg_risk = ((row["avg_risk"] * row["total_interactions"]) + risk_score) / total
            conn.execute(
                """
                UPDATE sender_memory
                SET total_interactions = ?, approvals = approvals + ?, rejections = rejections + ?,
                    auto_replies = auto_replies + ?, avg_risk = ?, last_seen_at=CURRENT_TIMESTAMP
                WHERE email = ?
                """,
                (total, int(approved), int(rejected), int(auto_replied), avg_risk, sender_email),
            )

    def get_sender_memory(self, sender_email: str) -> sqlite3.Row | None:
        with self._lock, self.connect() as conn:
            return conn.execute("SELECT * FROM sender_memory WHERE email = ?", (normalize_email_address(sender_email),)).fetchone()

    def get_thread_history(self, thread_id: str, limit: int = 5) -> list[dict[str, Any]]:
        with self._lock, self.connect() as conn:
            rows = conn.execute(
                """
                SELECT sender, subject, body, status, created_at
                FROM emails
                WHERE thread_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (thread_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_thread_commitments(self, thread_id: str, *, exclude_gmail_id: str | None = None, limit: int = 5) -> list[dict[str, Any]]:
        with self._lock, self.connect() as conn:
            rows = conn.execute(
                """
                SELECT a.suggested_reply, a.final_reply, a.selected_style, a.status, e.gmail_id, e.subject
                FROM approvals a
                JOIN emails e ON e.gmail_id = a.gmail_id
                WHERE e.thread_id = ?
                  AND (? IS NULL OR e.gmail_id != ?)
                  AND a.status IN ('approved', 'draft_preview')
                ORDER BY a.decided_at DESC, a.created_at DESC
                LIMIT ?
                """,
                (thread_id, exclude_gmail_id, exclude_gmail_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def upsert_relationship_memory(
        self,
        sender_email: str,
        *,
        relationship_type: str | None = None,
        preferred_tone: str | None = None,
        preferred_signoff: str | None = None,
        preferred_greeting: str | None = None,
        tone: str | None = None,
        edited_reply: str | None = None,
        rejected: bool = False,
    ) -> None:
        sender_email = normalize_email_address(sender_email)
        with self._lock, self.connect() as conn:
            row = conn.execute("SELECT * FROM relationship_memory WHERE email = ?", (sender_email,)).fetchone()
            if not row:
                conn.execute(
                    """
                    INSERT INTO relationship_memory (
                        email, relationship_type, preferred_tone, preferred_signoff,
                        preferred_greeting, tone_confidence, tone_history, edit_history, rejection_count
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        sender_email,
                        relationship_type or self._infer_relationship_type(sender_email),
                        preferred_tone or tone or "normal",
                        preferred_signoff,
                        preferred_greeting,
                        0.25 if (tone or edited_reply) else 0.0,
                        json.dumps([tone] if tone else []),
                        json.dumps([edited_reply] if edited_reply else []),
                        int(rejected),
                    ),
                )
                return
            tone_history = json.loads(row["tone_history"] or "[]")
            edit_history = json.loads(row["edit_history"] or "[]")
            if tone:
                tone_history.append(tone)
            if edited_reply:
                edit_history.append(edited_reply)
            conn.execute(
                """
                UPDATE relationship_memory
                SET relationship_type = ?, preferred_tone = ?, preferred_signoff = ?,
                    preferred_greeting = ?, tone_confidence = ?,
                    tone_history = ?, edit_history = ?,
                    rejection_count = rejection_count + ?, updated_at=CURRENT_TIMESTAMP
                WHERE email = ?
                """,
                (
                    relationship_type or row["relationship_type"],
                    preferred_tone or tone or row["preferred_tone"],
                    preferred_signoff or row["preferred_signoff"],
                    preferred_greeting or row["preferred_greeting"],
                    min(1.0, float(row["tone_confidence"] or 0) + (0.2 if tone else 0) + (0.15 if edited_reply else 0)),
                    json.dumps(tone_history[-20:]),
                    json.dumps(edit_history[-20:]),
                    int(rejected),
                    sender_email,
                ),
            )

    def get_relationship_memory(self, sender_email: str) -> sqlite3.Row | None:
        sender_email = normalize_email_address(sender_email)
        with self._lock, self.connect() as conn:
            return conn.execute("SELECT * FROM relationship_memory WHERE email = ?", (sender_email,)).fetchone()

    def upsert_thread_summary(
        self,
        thread_id: str,
        *,
        summary: str,
        commitments: list[str],
        pending_questions: list[str],
        scheduling_context: list[str],
        tone_shifts: list[str],
        unresolved_items: list[str],
    ) -> None:
        with self._lock, self.connect() as conn:
            conn.execute(
                """
                INSERT INTO thread_summaries (
                    thread_id, summary, commitments, pending_questions,
                    scheduling_context, tone_shifts, unresolved_items
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    summary = excluded.summary,
                    commitments = excluded.commitments,
                    pending_questions = excluded.pending_questions,
                    scheduling_context = excluded.scheduling_context,
                    tone_shifts = excluded.tone_shifts,
                    unresolved_items = excluded.unresolved_items,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    thread_id,
                    summary,
                    json.dumps(commitments[-8:]),
                    json.dumps(pending_questions[-8:]),
                    json.dumps(scheduling_context[-8:]),
                    json.dumps(tone_shifts[-8:]),
                    json.dumps(unresolved_items[-8:]),
                ),
            )

    def get_thread_summary(self, thread_id: str) -> sqlite3.Row | None:
        with self._lock, self.connect() as conn:
            return conn.execute("SELECT * FROM thread_summaries WHERE thread_id = ?", (thread_id,)).fetchone()

    def ignore_sender(self, sender_email: str, *, reason: str | None = None) -> None:
        sender_email = normalize_email_address(sender_email)
        with self._lock, self.connect() as conn:
            conn.execute(
                """
                INSERT INTO ignored_senders (email, reason)
                VALUES (?, ?)
                ON CONFLICT(email) DO UPDATE SET
                    reason = COALESCE(excluded.reason, ignored_senders.reason),
                    updated_at = CURRENT_TIMESTAMP
                """,
                (sender_email, reason),
            )

    def unignore_sender(self, sender_email: str) -> bool:
        sender_email = normalize_email_address(sender_email)
        with self._lock, self.connect() as conn:
            cursor = conn.execute("DELETE FROM ignored_senders WHERE email = ?", (sender_email,))
            return cursor.rowcount == 1

    def is_sender_ignored(self, sender_email: str) -> bool:
        sender_email = normalize_email_address(sender_email)
        with self._lock, self.connect() as conn:
            row = conn.execute("SELECT 1 FROM ignored_senders WHERE email = ?", (sender_email,)).fetchone()
        return row is not None

    def list_ignored_senders(self) -> list[dict[str, Any]]:
        with self._lock, self.connect() as conn:
            rows = conn.execute(
                "SELECT email, reason, ignored_at, updated_at FROM ignored_senders ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def ignore_thread(self, thread_id: str, *, reason: str | None = None) -> None:
        thread_id = (thread_id or "").strip()
        if not thread_id:
            return
        with self._lock, self.connect() as conn:
            conn.execute(
                """
                INSERT INTO ignored_threads (thread_id, reason)
                VALUES (?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    reason = COALESCE(excluded.reason, ignored_threads.reason),
                    updated_at = CURRENT_TIMESTAMP
                """,
                (thread_id, reason),
            )

    def unignore_thread(self, thread_id: str) -> bool:
        with self._lock, self.connect() as conn:
            cursor = conn.execute("DELETE FROM ignored_threads WHERE thread_id = ?", ((thread_id or "").strip(),))
            return cursor.rowcount == 1

    def is_thread_ignored(self, thread_id: str) -> bool:
        with self._lock, self.connect() as conn:
            row = conn.execute("SELECT 1 FROM ignored_threads WHERE thread_id = ?", ((thread_id or "").strip(),)).fetchone()
        return row is not None

    def pin_sender(self, sender_email: str, *, reason: str | None = None) -> None:
        sender_email = normalize_email_address(sender_email)
        if not sender_email:
            return
        with self._lock, self.connect() as conn:
            conn.execute(
                """
                INSERT INTO pinned_senders (email, reason)
                VALUES (?, ?)
                ON CONFLICT(email) DO UPDATE SET
                    reason = COALESCE(excluded.reason, pinned_senders.reason),
                    updated_at = CURRENT_TIMESTAMP
                """,
                (sender_email, reason),
            )

    def unpin_sender(self, sender_email: str) -> bool:
        sender_email = normalize_email_address(sender_email)
        with self._lock, self.connect() as conn:
            cursor = conn.execute("DELETE FROM pinned_senders WHERE email = ?", (sender_email,))
            return cursor.rowcount == 1

    def is_sender_pinned(self, sender_email: str) -> bool:
        sender_email = normalize_email_address(sender_email)
        with self._lock, self.connect() as conn:
            row = conn.execute("SELECT 1 FROM pinned_senders WHERE email = ?", (sender_email,)).fetchone()
        return row is not None

    def auto_handle_sender(self, sender_email: str, *, reason: str | None = None) -> None:
        sender_email = normalize_email_address(sender_email)
        if not sender_email:
            return
        with self._lock, self.connect() as conn:
            conn.execute(
                """
                INSERT INTO auto_handle_senders (email, reason)
                VALUES (?, ?)
                ON CONFLICT(email) DO UPDATE SET
                    reason = COALESCE(excluded.reason, auto_handle_senders.reason),
                    updated_at = CURRENT_TIMESTAMP
                """,
                (sender_email, reason),
            )

    def is_sender_auto_handled(self, sender_email: str) -> bool:
        sender_email = normalize_email_address(sender_email)
        with self._lock, self.connect() as conn:
            row = conn.execute("SELECT 1 FROM auto_handle_senders WHERE email = ?", (sender_email,)).fetchone()
        return row is not None

    def dashboard_summary(self) -> dict[str, Any]:
        with self._lock, self.connect() as conn:
            pending = conn.execute("SELECT COUNT(*) AS c FROM approvals WHERE status IN ('pending', 'awaiting_style', 'editing', 'draft_preview')").fetchone()["c"]
            sent = conn.execute("SELECT COUNT(*) AS c FROM emails WHERE status IN ('auto_replied', 'approved_sent')").fetchone()["c"]
            approvals = conn.execute("SELECT COUNT(*) AS c FROM approvals").fetchone()["c"]
            tone_rows = conn.execute(
                "SELECT COALESCE(selected_style, suggested_tone) AS tone, COUNT(*) AS total FROM approvals GROUP BY tone"
            ).fetchall()
            risk_rows = conn.execute(
                """
                SELECT
                    CASE
                        WHEN risk_score >= 75 THEN 'high'
                        WHEN risk_score >= 35 THEN 'medium'
                        ELSE 'low'
                    END AS bucket,
                    COUNT(*) AS total
                FROM approvals
                GROUP BY bucket
                """
            ).fetchall()
            relationship_rows = conn.execute(
                """
                SELECT email, relationship_type, preferred_tone, preferred_signoff, preferred_greeting, tone_confidence
                FROM relationship_memory
                ORDER BY updated_at DESC
                LIMIT 10
                """
            ).fetchall()
            top_contacts = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT sender_email AS email, COUNT(*) AS total
                    FROM emails
                    GROUP BY sender_email
                    ORDER BY total DESC
                    LIMIT 5
                    """
                ).fetchall()
            ]
        return {
            "pending_approvals": pending,
            "sent_emails": sent,
            "total_approvals": approvals,
            "top_contacts": top_contacts,
            "tone_statistics": {row["tone"] or "unknown": row["total"] for row in tone_rows},
            "risk_distribution": {row["bucket"]: row["total"] for row in risk_rows},
            "relationship_profiles": [dict(row) for row in relationship_rows],
        }

    def pending_approvals(self) -> list[dict[str, Any]]:
        with self._lock, self.connect() as conn:
            rows = conn.execute(
                """
                SELECT a.id, a.status, a.summary, a.risk_score, a.priority, a.category,
                       e.sender, e.subject
                FROM approvals a
                JOIN emails e ON e.gmail_id = a.gmail_id
                WHERE a.status IN ('pending', 'awaiting_style', 'editing', 'draft_preview')
                ORDER BY a.created_at DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def add_vector_memory(self, sender_email: str, namespace: str, content: str, embedding: list[float] | None = None) -> None:
        sender_email = normalize_email_address(sender_email)
        with self._lock, self.connect() as conn:
            conn.execute(
                "INSERT INTO vector_memory (email, namespace, content, embedding) VALUES (?, ?, ?, ?)",
                (sender_email, namespace, content, json.dumps(embedding or [])),
            )

    def recall_memory(self, sender_email: str, query: str, limit: int = 5) -> list[dict[str, Any]]:
        sender_email = normalize_email_address(sender_email)
        query_terms = {term.lower() for term in re.findall(r"[A-Za-z0-9]+", query) if len(term) > 2}
        with self._lock, self.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, email, namespace, content, created_at
                FROM vector_memory
                WHERE email = ?
                ORDER BY created_at DESC
                LIMIT 50
                """,
                (sender_email,),
            ).fetchall()
        ranked = []
        for row in rows:
            content_terms = set(re.findall(r"[A-Za-z0-9]+", row["content"].lower()))
            score = len(query_terms & content_terms)
            if score:
                item = dict(row)
                item["score"] = score
                ranked.append(item)
        return sorted(ranked, key=lambda item: item["score"], reverse=True)[:limit]

    @staticmethod
    def _infer_relationship_type(sender_email: str) -> str:
        lowered = sender_email.lower()
        if "recruiter" in lowered or "talent" in lowered:
            return "recruiter"
        if ".edu" in lowered or "professor" in lowered:
            return "professor"
        if "client" in lowered:
            return "client"
        return "unknown"

    def log_action(self, action: str, gmail_id: str | None = None, details: dict[str, Any] | None = None) -> None:
        with self._lock, self.connect() as conn:
            conn.execute(
                "INSERT INTO action_logs (gmail_id, action, details) VALUES (?, ?, ?)",
                (gmail_id, action, json.dumps(details or {})),
            )

    @staticmethod
    def _is_expired(value: str | None) -> bool:
        if not value:
            return False
        normalized = value.replace("Z", "+00:00")
        try:
            expires_at = datetime.fromisoformat(normalized)
        except ValueError:
            return False
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        return expires_at <= datetime.now(timezone.utc)
