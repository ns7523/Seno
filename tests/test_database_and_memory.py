from app.database import Database
from app.memory.memory import MemoryStore
from app.models.email import EmailMessage


def make_email(gmail_id="gmail-1", sender="person@example.com"):
    return EmailMessage(
        gmail_id=gmail_id,
        thread_id="thread-1",
        sender=sender,
        subject="Hello",
        body="Hi there",
        timestamp=None,
    )


def test_database_prevents_duplicate_processing(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()

    assert db.is_processed("gmail-1") is False
    db.record_email(make_email(), status="processed")

    assert db.is_processed("gmail-1") is True


def test_memory_updates_sender_history_and_trust(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()
    memory = MemoryStore(db)

    memory.record_interaction("alice@example.com", approved=True, auto_replied=False, risk_score=40)
    memory.record_interaction("alice@example.com", approved=True, auto_replied=True, risk_score=25)
    profile = memory.get_sender_profile("alice@example.com")

    assert profile.email == "alice@example.com"
    assert profile.total_interactions == 2
    assert profile.approvals == 2
    assert profile.trust_score > 50


def test_thread_summary_tracks_compact_continuity_state(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()
    memory = MemoryStore(db)
    email = EmailMessage(
        gmail_id="gmail-thread-1",
        thread_id="thread-seno",
        sender="maya@example.com",
        subject="Seno follow-up next Tuesday",
        body=(
            "Can we discuss the dashboard and deployment next Tuesday at 6 PM? "
            "Please share the API notes. I will confirm the final agenda after your reply."
        ),
        timestamp=None,
    )

    memory.record_thread_observation(email, reply_text="Tuesday at 6 PM works. I will share the API notes.")
    summary = memory.get_thread_summary("thread-seno")

    assert summary.thread_id == "thread-seno"
    assert any("dashboard" in item.lower() for item in summary.unresolved_items)
    assert any("next Tuesday at 6 PM" in item or "6 PM" in item for item in summary.scheduling_context)
    assert any("share the api notes" in item.lower() for item in summary.commitments)
    assert summary.as_context()["pending_questions"]


def test_style_preferences_learn_directness_and_sentence_length(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()
    memory = MemoryStore(db)

    memory.record_user_edit("rahul@example.com", "Hi Rahul,\n\n9 AM works.\n\n- NS")
    memory.record_user_edit("rahul@example.com", "Hi Rahul,\n\nSounds good.\n\n- NS")
    prefs = memory.get_style_preferences("rahul@example.com")

    assert prefs.directness == "direct"
    assert prefs.sentence_length == "short"
    assert prefs.preferred_greeting == "Hi Rahul,"
    assert prefs.preferred_signoff == "- NS"


def test_sender_and_thread_control_preferences_persist(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    db.init_db()

    db.ignore_thread("thread-noisy", reason="telegram mute")
    db.pin_sender("founder@example.com", reason="important contact")
    db.auto_handle_sender("alerts@example.com", reason="low-risk alerts")

    assert db.is_thread_ignored("thread-noisy") is True
    assert db.is_sender_pinned("founder@example.com") is True
    assert db.is_sender_auto_handled("alerts@example.com") is True
