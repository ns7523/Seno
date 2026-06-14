from __future__ import annotations

from pathlib import Path

import pytest

import app.integrations.google_auth as google_auth
from app.integrations.calendar import GoogleCalendarService
from app.integrations.google_auth import CALENDAR_SCOPES, GMAIL_SCOPES, GOOGLE_SCOPES, GoogleAuth
from app.models.email import EmailMessage


class FakeCreds:
    def __init__(self, *, valid=True, expired=False, refresh_token=None, scopes=None):
        self.valid = valid
        self.expired = expired
        self.refresh_token = refresh_token
        self.scopes = scopes or GOOGLE_SCOPES
        self.refreshed = False

    def has_scopes(self, scopes):
        return set(scopes).issubset(set(self.scopes))

    def refresh(self, request):
        self.refreshed = True
        self.valid = True
        self.expired = False

    def to_json(self):
        return '{"token":"fake"}'


class FakeFlow:
    def __init__(self, creds):
        self.creds = creds

    def run_local_server(self, **kwargs):
        return self.creds


def make_email(subject="Meeting tomorrow at 6 PM", body="Can we discuss the Seno workflow tomorrow at 6 PM?"):
    return EmailMessage(
        gmail_id="gmail-1",
        thread_id="thread-1",
        sender="Ravi <ravi@example.com>",
        subject=subject,
        body=body,
        timestamp=None,
    )


def test_shared_google_scopes_include_gmail_and_calendar():
    assert GMAIL_SCOPES == [
        "https://www.googleapis.com/auth/gmail.modify",
        "https://www.googleapis.com/auth/gmail.send",
    ]
    assert CALENDAR_SCOPES == [
        "https://www.googleapis.com/auth/calendar",
        "https://www.googleapis.com/auth/calendar.events",
    ]
    assert GOOGLE_SCOPES == [*GMAIL_SCOPES, *CALENDAR_SCOPES]


def test_google_auth_regenerates_missing_token(monkeypatch, tmp_path):
    client_secret = tmp_path / "client_secret.json"
    client_secret.write_text("{}", encoding="utf-8")
    generated = FakeCreds(valid=True)
    calls = []

    def fake_flow(path, scopes):
        calls.append((path, scopes))
        return FakeFlow(generated)

    monkeypatch.setattr(google_auth.InstalledAppFlow, "from_client_secrets_file", fake_flow)

    creds = GoogleAuth(str(client_secret), str(tmp_path / "token.json")).get_credentials()

    assert creds is generated
    assert calls == [(str(client_secret), GOOGLE_SCOPES)]
    assert (tmp_path / "token.json").exists()


def test_google_auth_can_fail_fast_without_interactive_oauth(tmp_path):
    client_secret = tmp_path / "client_secret.json"
    client_secret.write_text("{}", encoding="utf-8")

    auth = GoogleAuth(str(client_secret), str(tmp_path / "token.json"), allow_interactive_oauth=False)

    with pytest.raises(RuntimeError, match="Interactive Google OAuth is disabled"):
        auth.get_credentials()


def test_google_auth_force_regenerates_even_with_valid_token(monkeypatch, tmp_path):
    client_secret = tmp_path / "client_secret.json"
    token_file = tmp_path / "token.json"
    client_secret.write_text("{}", encoding="utf-8")
    token_file.write_text('{"scopes":["https://www.googleapis.com/auth/gmail.modify"]}', encoding="utf-8")
    generated = FakeCreds(valid=True)
    monkeypatch.setattr(google_auth.InstalledAppFlow, "from_client_secrets_file", lambda path, scopes: FakeFlow(generated))

    creds = GoogleAuth(str(client_secret), str(token_file)).regenerate_credentials()

    assert creds is generated
    assert token_file.read_text(encoding="utf-8") == '{"token":"fake"}'


def test_google_auth_regenerates_token_missing_calendar_scopes(monkeypatch, tmp_path):
    client_secret = tmp_path / "client_secret.json"
    token_file = tmp_path / "token.json"
    client_secret.write_text("{}", encoding="utf-8")
    token_file.write_text("{}", encoding="utf-8")
    old_creds = FakeCreds(valid=True, scopes=GMAIL_SCOPES)
    new_creds = FakeCreds(valid=True, scopes=GOOGLE_SCOPES)

    monkeypatch.setattr(google_auth.Credentials, "from_authorized_user_file", lambda path, scopes: old_creds)
    monkeypatch.setattr(google_auth.InstalledAppFlow, "from_client_secrets_file", lambda path, scopes: FakeFlow(new_creds))

    creds = GoogleAuth(str(client_secret), str(token_file)).get_credentials()

    assert creds is new_creds
    assert token_file.read_text(encoding="utf-8") == '{"token":"fake"}'


def test_google_auth_refreshes_expired_token(monkeypatch, tmp_path):
    token_file = tmp_path / "token.json"
    token_file.write_text("{}", encoding="utf-8")
    creds = FakeCreds(valid=True, expired=True, refresh_token="refresh-token")

    monkeypatch.setattr(google_auth.Credentials, "from_authorized_user_file", lambda path, scopes: creds)

    result = GoogleAuth(None, str(token_file)).get_credentials()

    assert result is creds
    assert creds.refreshed is True
    assert token_file.read_text(encoding="utf-8") == '{"token":"fake"}'


class FakeExecute:
    def __init__(self, value):
        self.value = value

    def execute(self):
        return self.value


class FakeFreebusy:
    def __init__(self):
        self.queries = []
        self.busy = []

    def query(self, body):
        self.queries.append(body)
        return FakeExecute({"calendars": {"primary": {"busy": self.busy}}})


class FakeEvents:
    def __init__(self):
        self.insert_calls = []

    def insert(self, **kwargs):
        self.insert_calls.append(kwargs)
        return FakeExecute(
            {
                "id": "event-1",
                "summary": kwargs["body"]["summary"],
                "start": kwargs["body"]["start"],
                "end": kwargs["body"]["end"],
                "hangoutLink": "https://meet.google.com/abc-defg-hij",
            }
        )


class FakeCalendarAPI:
    def __init__(self):
        self.freebusy_api = FakeFreebusy()
        self.events_api = FakeEvents()

    def freebusy(self):
        return self.freebusy_api

    def events(self):
        return self.events_api


@pytest.mark.asyncio
async def test_google_calendar_creates_event_with_meet_link():
    api = FakeCalendarAPI()
    service = GoogleCalendarService(client_secrets_file=None, token_file="token.json", service=api)

    event = await service.create_event_from_email(make_email())

    assert event is not None
    assert event.title == "Meeting tomorrow at 6 PM"
    assert event.meet_link == "https://meet.google.com/abc-defg-hij"
    insert = api.events_api.insert_calls[0]
    assert insert["calendarId"] == "primary"
    assert insert["conferenceDataVersion"] == 1
    assert insert["body"]["conferenceData"]["createRequest"]["conferenceSolutionKey"]["type"] == "hangoutsMeet"


@pytest.mark.asyncio
async def test_google_calendar_conflict_lookup_uses_freebusy():
    api = FakeCalendarAPI()
    api.freebusy_api.busy = [{"start": "2026-05-17T12:30:00+00:00", "end": "2026-05-17T13:00:00+00:00"}]
    service = GoogleCalendarService(client_secrets_file=None, token_file="token.json", service=api)

    conflicts = await service.conflicts_for_email(make_email())

    assert conflicts[0].title == "Busy"
    assert api.freebusy_api.queries[0]["items"] == [{"id": "primary"}]


@pytest.mark.asyncio
async def test_google_calendar_handles_malformed_schedule_safely():
    api = FakeCalendarAPI()
    service = GoogleCalendarService(client_secrets_file=None, token_file="token.json", service=api)

    event = await service.create_event_from_email(make_email(subject="Hello", body="No time here."))

    assert event is None
    assert api.events_api.insert_calls == []


@pytest.mark.asyncio
async def test_google_calendar_api_failure_returns_safe_fallback():
    class BrokenEvents(FakeEvents):
        def insert(self, **kwargs):
            raise RuntimeError("calendar unavailable")

    api = FakeCalendarAPI()
    api.events_api = BrokenEvents()
    service = GoogleCalendarService(client_secrets_file=None, token_file="token.json", service=api)

    event = await service.create_event_from_email(make_email())

    assert event is None


@pytest.mark.asyncio
async def test_google_calendar_understands_time_ranges_and_formats_alternatives():
    api = FakeCalendarAPI()
    service = GoogleCalendarService(client_secrets_file=None, token_file="token.json", service=api)
    email = make_email(
        subject="Planning discussion next Tuesday",
        body="Could we meet next Tuesday between 5 and 6 PM to discuss the deployment plan?",
    )

    event = await service.create_event_from_email(email)
    alternatives = await service.suggest_alternative_times(email)

    assert event is not None
    assert "T17:00:00" in event.starts_at
    assert all(" at " in item for item in alternatives)
