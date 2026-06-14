from dataclasses import dataclass
from typing import Protocol

from app.models.email import EmailMessage


@dataclass(slots=True)
class CalendarEvent:
    title: str
    starts_at: str
    ends_at: str
    meet_link: str | None = None
    event_id: str | None = None


class CalendarService(Protocol):
    async def conflicts_for_email(self, email: EmailMessage) -> list[CalendarEvent]:
        ...

    async def create_event_from_email(self, email: EmailMessage) -> CalendarEvent | None:
        ...

    async def suggest_alternative_times(self, email: EmailMessage) -> list[str]:
        ...


class NoopCalendarService:
    async def conflicts_for_email(self, email: EmailMessage) -> list[CalendarEvent]:
        return []

    async def create_event_from_email(self, email: EmailMessage) -> CalendarEvent | None:
        return None

    async def suggest_alternative_times(self, email: EmailMessage) -> list[str]:
        return []
