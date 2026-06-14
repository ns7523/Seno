import pytest
import logging

from app.database import Database
from app.models.email import EmailMessage
from app.services.email_processor import InboxMonitor


class FakeReader:
    def __init__(self):
        self.marked = []

    async def fetch_unread(self, query):
        return [
            EmailMessage("ok-1", "thread-1", "a@example.com", "Hi", "Hello", None),
            EmailMessage("bad-1", "thread-2", "b@example.com", "Hi", "Boom", None),
        ]

    async def mark_processed(self, gmail_id):
        self.marked.append(gmail_id)


class FakeProcessor:
    def __init__(self, db, debug_gmail_pipeline=False):
        self.db = db
        self.debug_gmail_pipeline = debug_gmail_pipeline
        self.processed = []

    async def process_email(self, email):
        self.processed.append(email.gmail_id)
        if email.gmail_id == "bad-1":
            raise RuntimeError("simulated failure")
        if self.db.is_processed(email.gmail_id) and not self.debug_gmail_pipeline:
            self.db.log_action("duplicate_skipped", email.gmail_id)
            return
        self.db.record_email(email, "processed")


@pytest.mark.asyncio
async def test_monitor_logs_failures_without_crashing(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()
    reader = FakeReader()
    monitor = InboxMonitor(reader, FakeProcessor(db), "is:unread")

    await monitor.poll_once()

    assert reader.marked == ["ok-1"]


@pytest.mark.asyncio
async def test_monitor_emits_inbox_pipeline_logs(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()
    reader = FakeReader()
    monitor = InboxMonitor(reader, FakeProcessor(db), "is:unread")

    await monitor.poll_once()

    messages = [record.getMessage() for record in caplog.records]
    assert "Checking Gmail inbox..." in messages
    assert "Unread email detected" in messages
    assert "Inbox email processing eligibility" in messages
    eligibility = [record for record in caplog.records if record.getMessage() == "Inbox email processing eligibility"][0]
    assert eligibility.processing_eligible is True
    assert eligibility.already_processed is False


@pytest.mark.asyncio
async def test_monitor_debug_pipeline_processes_even_when_cache_has_email(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()
    cached = EmailMessage("ok-1", "thread-1", "a@example.com", "Hi", "Hello", None)
    db.record_email(cached, "processed")
    reader = FakeReader()
    processor = FakeProcessor(db, debug_gmail_pipeline=True)
    monitor = InboxMonitor(reader, processor, "is:unread", debug_pipeline=True)

    await monitor.poll_once()

    assert "ok-1" in processor.processed
    assert reader.marked == ["ok-1"]
