from __future__ import annotations

import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from app.calendar.service import CalendarEvent
from app.integrations.google_auth import GoogleAuth
from app.models.email import EmailMessage
from app.utils.logger import get_logger


logger = get_logger(__name__)


class GoogleCalendarService:
    """Google Calendar integration backed by the shared Gmail/Calendar token."""

    def __init__(
        self,
        *,
        client_secrets_file: str | None,
        token_file: str,
        calendar_id: str = "primary",
        service: Any | None = None,
        allow_interactive_oauth: bool = True,
    ) -> None:
        self.calendar_id = calendar_id
        self.service = service or GoogleAuth(
            client_secrets_file,
            token_file,
            allow_interactive_oauth=allow_interactive_oauth,
        ).build_service("calendar", "v3")

    async def conflicts_for_email(self, email: EmailMessage) -> list[CalendarEvent]:
        window = self._window_from_email(email)
        if not window:
            return []
        start, end = window
        try:
            response = (
                self.service.freebusy()
                .query(
                    body={
                        "timeMin": start.isoformat(),
                        "timeMax": end.isoformat(),
                        "items": [{"id": self.calendar_id}],
                    }
                )
                .execute()
            )
            busy = response.get("calendars", {}).get(self.calendar_id, {}).get("busy", [])
            return [
                CalendarEvent(title="Busy", starts_at=item.get("start", ""), ends_at=item.get("end", ""))
                for item in busy
                if item.get("start") and item.get("end")
            ]
        except Exception as exc:
            logger.warning("calendar_conflict_lookup_failed", extra={"gmail_id": email.gmail_id, "error": str(exc)})
            return []

    async def create_event_from_email(self, email: EmailMessage) -> CalendarEvent | None:
        window = self._window_from_email(email)
        if not window:
            logger.info("calendar_event_skipped_no_schedule", extra={"gmail_id": email.gmail_id})
            return None
        start, end = window
        title = self._event_title(email)
        body = {
            "summary": title,
            "description": self._event_description(email),
            "start": {"dateTime": start.isoformat()},
            "end": {"dateTime": end.isoformat()},
            "conferenceData": {
                "createRequest": {
                    "requestId": f"seno-{email.gmail_id}-{uuid.uuid4().hex[:10]}",
                    "conferenceSolutionKey": {"type": "hangoutsMeet"},
                }
            },
        }
        try:
            created = (
                self.service.events()
                .insert(calendarId=self.calendar_id, body=body, conferenceDataVersion=1)
                .execute()
            )
            meet_link = created.get("hangoutLink")
            logger.info("calendar_event_created", extra={"gmail_id": email.gmail_id, "event_id": created.get("id")})
            return CalendarEvent(
                title=created.get("summary", title),
                starts_at=created.get("start", {}).get("dateTime", start.isoformat()),
                ends_at=created.get("end", {}).get("dateTime", end.isoformat()),
                meet_link=meet_link,
                event_id=created.get("id"),
            )
        except Exception as exc:
            logger.warning("calendar_event_create_failed", extra={"gmail_id": email.gmail_id, "error": str(exc)})
            return None

    async def suggest_alternative_times(self, email: EmailMessage) -> list[str]:
        base = self._base_datetime_from_email(email) or datetime.now(timezone.utc)
        suggestions: list[str] = []
        for day_offset in range(0, 5):
            day = base + timedelta(days=day_offset)
            for hour in (10, 11, 15, 16, 18):
                candidate_start = day.replace(hour=hour, minute=0, second=0, microsecond=0)
                if candidate_start <= datetime.now(timezone.utc):
                    continue
                candidate_end = candidate_start + timedelta(minutes=30)
                if await self._is_free(candidate_start, candidate_end):
                    suggestions.append(candidate_start.strftime("%A at %I:%M %p").replace(" 0", " "))
                if len(suggestions) >= 3:
                    return suggestions
        return suggestions

    async def lookup_freebusy(self, starts_at: datetime, ends_at: datetime) -> list[CalendarEvent]:
        try:
            response = (
                self.service.freebusy()
                .query(
                    body={
                        "timeMin": starts_at.isoformat(),
                        "timeMax": ends_at.isoformat(),
                        "items": [{"id": self.calendar_id}],
                    }
                )
                .execute()
            )
        except Exception as exc:
            logger.warning("calendar_freebusy_lookup_failed", extra={"error": str(exc)})
            return []
        busy = response.get("calendars", {}).get(self.calendar_id, {}).get("busy", [])
        return [CalendarEvent(title="Busy", starts_at=item.get("start", ""), ends_at=item.get("end", "")) for item in busy]

    async def _is_free(self, starts_at: datetime, ends_at: datetime) -> bool:
        return not await self.lookup_freebusy(starts_at, ends_at)

    def _window_from_email(self, email: EmailMessage) -> tuple[datetime, datetime] | None:
        start = self._base_datetime_from_email(email)
        if not start:
            return None
        return start, start + timedelta(minutes=30)

    def _base_datetime_from_email(self, email: EmailMessage) -> datetime | None:
        text = f"{email.subject} {email.body}"
        now = datetime.now(timezone.utc)
        day_offset = 0
        lowered = text.lower()
        if "tomorrow" in lowered:
            day_offset = 1
        else:
            weekdays = {
                "monday": 0,
                "tuesday": 1,
                "wednesday": 2,
                "thursday": 3,
                "friday": 4,
                "saturday": 5,
                "sunday": 6,
            }
            for name, weekday in weekdays.items():
                if name in lowered:
                    day_offset = (weekday - now.weekday()) % 7 or 7
                    break
        range_match = re.search(
            r"\b(?:between|from)\s+(\d{1,2})(?::(\d{2}))?\s*(AM|PM|am|pm)?\s*(?:and|to|-)\s*(\d{1,2})(?::(\d{2}))?\s*(AM|PM|am|pm)?\b",
            text,
            re.IGNORECASE,
        )
        if range_match:
            hour = int(range_match.group(1))
            minute = int(range_match.group(2) or 0)
            meridiem = (range_match.group(3) or range_match.group(6) or "").lower()
            
            if meridiem == "pm" and hour < 12:
                hour += 12

            if meridiem == "am" and hour == 12:
                hour = 0

            if hour < 0 or hour > 23:
                logger.warning(
                    "invalid_calendar_hour_range_match",
                    extra={
                        "hour": hour,
                        "email_subject": email.subject,
                    },
                )
                return None

            if minute < 0 or minute > 59:
                logger.warning(
                    "invalid_calendar_minute_range_match",
                    extra={
                        "minute": minute,
                        "email_subject": email.subject,
                    },
                )
                return None

            return (now + timedelta(days=day_offset)).replace(
                hour=hour,
                minute=minute,
                second=0,
                microsecond=0,
            )

        time_match = re.search(r"\b(?:around|at|after|by)?\s*(\d{1,2})(?::(\d{2}))?\s*(AM|PM|am|pm)?\b", text)
        if not time_match:
            return None
        hour = int(time_match.group(1))
        minute = int(time_match.group(2) or 0)
        meridiem = (time_match.group(3) or "").lower()
        if meridiem == "pm" and hour < 12:
            hour += 12

        if meridiem == "am" and hour == 12:
            hour = 0

        if not meridiem and hour < 8:
            hour += 12

        if hour < 0 or hour > 23:
            logger.warning(
                "invalid_calendar_hour_time_match",
                extra={
                    "hour": hour,
                    "email_subject": email.subject,
                },
            )
            return None

        if minute < 0 or minute > 59:
            logger.warning(
                "invalid_calendar_minute_time_match",
                extra={
                    "minute": minute,
                    "email_subject": email.subject,
                },
            )
            return None

        return (now + timedelta(days=day_offset)).replace(
            hour=hour,
            minute=minute,
            second=0,
            microsecond=0,
        )

    @staticmethod
    def _event_title(email: EmailMessage) -> str:
        subject = email.subject.strip() or "Seno scheduled conversation"
        return subject[:120]

    @staticmethod
    def _event_description(email: EmailMessage) -> str:
        body = email.body.strip()
        if len(body) > 1200:
            body = body[:1200].rstrip() + "..."
        return f"Created by Seno from email thread {email.thread_id}.\n\nFrom: {email.sender}\n\n{body}"
