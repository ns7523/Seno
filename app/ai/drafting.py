import re
from dataclasses import dataclass, field
from typing import Any

from app.models.email import EmailMessage


GENERIC_DRAFT_MARKERS = {
    "thanks for the message. ns has received it.",
    "noted.",
    "approved reply",
    "i will review and respond soon.",
}


@dataclass(slots=True)
class DraftContext:
    topics: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)
    schedule: str | None = None
    location: str | None = None


class ContextualDraftGenerator:
    """Builds a detailed deterministic draft from the full email context.

    Groq can still provide the upstream classification and suggested reply, but
    this local layer prevents approved previews from collapsing into generic
    acknowledgements when model output is sparse or safety-biased.
    """

    def build(
        self,
        email: EmailMessage,
        *,
        suggested_reply: str,
        style: str,
        variation: str | None = None,
        thread_context: dict[str, Any] | None = None,
        style_preferences: Any | None = None,
    ) -> str:
        lowered = f"{email.subject} {email.body}".lower()
        if variation is None and getattr(style_preferences, "directness", None) == "direct":
            variation = "concise_direct"
        if suggested_reply and self._contains_sensitive_action(suggested_reply):
            return suggested_reply.strip()
        if suggested_reply and self._is_social_plan(lowered) and not self._is_generic(suggested_reply):
            return suggested_reply.strip()
        if suggested_reply and not self._is_generic(suggested_reply) and not self._is_contextually_weak(suggested_reply, lowered):
            return suggested_reply.strip()
        context = self.extract_context(email, thread_context=thread_context)
        if self._is_social_plan(lowered):
            return self._social_reply(email, context, variation=variation)
        if self._is_professional_opportunity(lowered):
            return self._professional_reply(email, context, opportunity=True, style=style, variation=variation)
        if self._is_collaboration(lowered):
            return self._professional_reply(email, context, opportunity=False, style=style, variation=variation)
        if self._is_meeting(lowered):
            return self._meeting_reply(email, context)
        return self._general_contextual_reply(email, context)

    def extract_context(self, email: EmailMessage, *, thread_context: dict[str, Any] | None = None) -> DraftContext:
        text = f"{email.subject}. {email.body}"
        lowered = text.lower()
        topics = []
        for topic in [
            "internship opportunity",
            "backend AI internship",
            "FastAPI",
            "Gmail automation",
            "Seno",
            "AI workflow",
            "workflow architecture",
            "communication workflow architecture",
            "approval-driven approach",
            "approval-driven workflow",
            "contextual drafting",
            "orchestration",
            "communication infrastructure",
            "executive communication",
            "AI project",
            "API design",
            "dashboard",
            "deployment",
            "testing",
            "project discussion",
            "collaboration",
            "networking",
        ]:
            if topic.lower() in lowered:
                topics.append(topic)
        questions = [sentence.strip() for sentence in re.split(r"(?<=[?.!])\s+", email.body) if "?" in sentence]
        action_items = []
        for marker in ("please ", "could you ", "would like you to ", "let me know", "share "):
            if marker in lowered:
                action_items.append(marker.strip())
        if thread_context:
            for item in thread_context.get("unresolved_items", []) or []:
                if isinstance(item, str) and item and item not in questions:
                    questions.append(item)
            for item in thread_context.get("scheduling_context", []) or []:
                if isinstance(item, str) and item and not self._extract_schedule(text):
                    text = f"{text} {item}"
                    break
        return DraftContext(
            topics=_dedupe(topics),
            questions=questions[:4],
            action_items=_dedupe(action_items),
            schedule=self._extract_schedule(text),
            location=self._extract_location(text),
        )

    @staticmethod
    def _is_generic(reply: str) -> bool:
        cleaned = re.sub(r"\s+", " ", reply.strip().lower())
        if cleaned in GENERIC_DRAFT_MARKERS:
            return True
        return any(marker in cleaned for marker in ("ns has received it", "your message has been noted"))

    @staticmethod
    def _is_contextually_weak(reply: str, source_text: str) -> bool:
        cleaned = re.sub(r"\s+", " ", reply.strip().lower())
        if len(cleaned.split()) < 14:
            return True
        generic_openers = (
            "hello, thank you for reaching out",
            "thank you for reaching out",
            "thanks for reaching out",
            "thank you for your message",
        )
        professional_context = any(
            marker in source_text
            for marker in (
                "collaboration",
                "collaborate",
                "ai workflow",
                "workflow architecture",
                "architecture",
                "technical",
                "orchestration",
                "communication infrastructure",
                "networking",
                "schedule",
                "available",
                "connect",
                "discussion",
                "internship",
                "opportunity",
            )
        )
        if any(cleaned.startswith(opener) for opener in generic_openers):
            return True
        if professional_context:
            contextual_terms = (
                "collaboration",
                "workflow",
                "architecture",
                "technical",
                "orchestration",
                "infrastructure",
                "schedule",
                "available",
                "connect",
                "discussion",
                "internship",
                "opportunity",
                "seno",
            )
            return not any(term in cleaned for term in contextual_terms)
        return False

    @staticmethod
    def _contains_sensitive_action(reply: str) -> bool:
        lowered = reply.lower()
        return any(
            phrase in lowered
            for phrase in (
                "confirm payment",
                "payment has been sent",
                "approve the invoice",
                "approve payment",
                "signed the contract",
                "accept the legal",
                "accept the hr",
                "password is",
                "otp is",
            )
        )

    @staticmethod
    def _is_social_plan(text: str) -> bool:
        return any(word in text for word in ("dinner", "breakfast", "lunch", "coffee", "movie", "hangout", "uvce", "campus", "casual meeting"))

    @staticmethod
    def _is_professional_opportunity(text: str) -> bool:
        return any(word in text for word in ("internship", "recruiter", "interview", "opportunity", "portfolio"))

    @staticmethod
    def _is_collaboration(text: str) -> bool:
        return any(
            word in text
            for word in (
                "collaborate",
                "collaboration",
                "proposal",
                "project discussion",
                "api design",
                "dashboard",
                "ai workflow",
                "workflow architecture",
                "communication workflow",
                "approval-driven",
                "contextual drafting",
                "orchestration",
                "communication infrastructure",
                "technical architecture",
                "networking",
                "connect",
            )
        )

    @staticmethod
    def _is_meeting(text: str) -> bool:
        return any(word in text for word in ("meeting", "schedule", "available", "discussion"))

    def _social_reply(self, email: EmailMessage, context: DraftContext, *, variation: str | None = None) -> str:
        plan = self._first_matching(f"{email.subject} {email.body}", ["dinner", "breakfast", "lunch", "coffee", "movie", "hangout"]) or "plan"
        if plan in {"movie", "hangout"} and not context.schedule and not context.location:
            if variation == "concise_direct":
                return "That works. See you then."
            if variation == "warmer_executive":
                return "Sounds good. Looking forward to it."
            return "Sounds good. See you then."
        details = []
        if context.schedule:
            details.append(context.schedule)
        if context.location:
            details.append(context.location)
        detail_text = f" at {' at '.join(details)}" if details else ""
        if variation == "concise_direct":
            return f"{plan.title()}{detail_text} works."
        if variation == "warmer_executive":
            return f"Sounds good — {plan}{detail_text} works for me."
        return f"{plan.title()}{detail_text} works. See you then."

    def _professional_reply(self, email: EmailMessage, context: DraftContext, *, opportunity: bool, style: str, variation: str | None = None) -> str:
        lines = []
        focus_topics = self._prioritized_focus_topics(context.topics)
        topic_text = self._natural_topic_phrase(focus_topics) if focus_topics else "Seno and the communication workflow"
        if variation == "warmer_executive":
            opening = (
                f"I’m glad {topic_text} caught your attention. "
                "The conversation sounds useful, and I’d be happy to explore it further."
            )
        elif variation == "technical":
            opening = (
                f"I’d be glad to walk through {topic_text} from a technical angle. "
                "There are a few design choices and tradeoffs that would be worth discussing."
            )
        elif variation == "concise_direct":
            opening = f"This sounds worthwhile. I’m open to discussing {topic_text} and next steps."
        elif variation == "collaborative":
            opening = (
                f"It would be useful to compare notes on {topic_text}. "
                "I’d be glad to discuss the direction and where collaboration could fit."
            )
        elif opportunity:
            opening = (
                "Thanks for taking a close look at the project. "
                f"The opportunity to discuss {topic_text} sounds worthwhile."
            )
        elif style.lower() == "formal":
            opening = (
                f"Thanks for taking the time to look through {topic_text}. "
                "The collaboration angle sounds useful, and I’d be glad to discuss where the work could go next."
            )
        else:
            opening = (
                f"Thanks for sharing this. The discussion around {topic_text} sounds worthwhile, "
                "and I’d be glad to talk through the direction and next steps."
            )
        lines.append(opening)
        if context.schedule:
            if variation == "concise_direct":
                lines.append(f"{context.schedule} works for me.")
            elif variation == "warmer_executive":
                lines.append(f"{context.schedule} should work well on my side.")
            else:
                lines.append(f"{context.schedule} should work well for me.")
        if context.topics:
            detail_topics = self._natural_topic_phrase(self._discussion_topics(context.topics))
            if variation == "technical":
                lines.append(f"I can go deeper on {detail_topics} and how the pieces fit together.")
            elif variation == "collaborative":
                lines.append(f"We can use the conversation to cover {detail_topics} and possible next steps.")
            elif variation != "concise_direct":
                lines.append(f"I can also talk through {detail_topics} in more detail.")
        if context.questions and variation != "concise_direct":
            lines.append("I’ll come prepared to walk through the questions and next steps.")
        return "\n\n".join(lines)

    def _meeting_reply(self, email: EmailMessage, context: DraftContext) -> str:
        pieces = ["Thanks for the update."]
        if context.schedule:
            pieces.append(f"{context.schedule} works for the meeting.")
        if context.topics:
            pieces.append(f"I’ll be ready to discuss {self._natural_topic_phrase(context.topics[:4])}.")
        return " ".join(pieces)

    def _general_contextual_reply(self, email: EmailMessage, context: DraftContext) -> str:
        pieces = ["Thanks for sending this over."]
        if context.topics:
            pieces.append(f"I’ve noted the main points around {self._natural_topic_phrase(context.topics[:4])}.")
        if context.questions:
            pieces.append("I will address the questions you raised and follow the expected next steps.")
        return " ".join(pieces)

    @staticmethod
    def _extract_schedule(text: str) -> str | None:
        multi_day = re.search(
            r"\b((?:next\s+)?(?:today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)(?:\s+or\s+(?:next\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))?)\b(?:\s+(morning|afternoon|evening))?.{0,40}?\b(?:around\s+|at\s+|by\s+)?(\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm))(?:\s*(IST|UTC|GMT))?",
            text,
            re.IGNORECASE,
        )
        if multi_day:
            day = multi_day.group(1)
            period = multi_day.group(2)
            time = multi_day.group(3)
            zone = multi_day.group(4)
            parts = [day]
            if period:
                parts.append(period)
            parts.append(f"around {time}")
            if zone:
                parts.append(zone.upper())
            return " ".join(parts)
        weekday = re.search(r"\b((?:next\s+)?(?:today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b", text, re.IGNORECASE)
        time = re.search(r"\b\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm)(?:\s*(?:IST|UTC|GMT))?\b", text)
        if weekday and time:
            return f"{weekday.group(0)} at {time.group(0)}"
        if time:
            return time.group(0)
        return weekday.group(0) if weekday else None

    @staticmethod
    def _extract_location(text: str) -> str | None:
        match = re.search(r"\bat\s+([A-Z][A-Za-z0-9 &.-]{2,40})(?:[.!?]|$)", text)
        if not match:
            return None
        value = match.group(1).strip()
        if re.match(r"\d", value):
            return None
        return value

    @staticmethod
    def _first_matching(text: str, words: list[str]) -> str | None:
        lowered = text.lower()
        for word in words:
            if word in lowered:
                return word
        return None

    @staticmethod
    def _natural_topic_phrase(topics: list[str]) -> str:
        if not topics:
            return "the details"
        if len(topics) == 1:
            return topics[0]
        if len(topics) == 2:
            return f"{topics[0]} and {topics[1]}"
        return f"{', '.join(topics[:-1])}, and {topics[-1]}"

    @staticmethod
    def _prioritized_focus_topics(topics: list[str]) -> list[str]:
        lowered = {topic.lower() for topic in topics}
        result: list[str] = []
        if "seno" in lowered or any("workflow" in topic for topic in lowered):
            result.append("Seno’s communication workflow")
        if any(topic in lowered for topic in {"internship opportunity", "backend ai internship"}):
            result.append("the internship opportunity")
        if "collaboration" in lowered:
            result.append("possible collaboration")
        if not result and topics:
            result.append(topics[0])
        return result[:3]

    @staticmethod
    def _discussion_topics(topics: list[str]) -> list[str]:
        lowered = {topic.lower() for topic in topics}
        result: list[str] = []
        if any(topic in lowered for topic in {"approval-driven approach", "approval-driven workflow", "contextual drafting"}):
            result.append("the approval flow")
        if any(topic in lowered for topic in {"workflow architecture", "communication workflow architecture", "orchestration", "communication infrastructure"}):
            result.append("the architecture decisions")
        if "deployment" in lowered:
            result.append("deployment tradeoffs")
        if "testing" in lowered:
            result.append("testing")
        if "dashboard" in lowered:
            result.append("the dashboard")
        if "api design" in lowered:
            result.append("API design")
        if any(topic in lowered for topic in {"fastapi", "gmail automation", "ai project"}):
            result.append("the FastAPI, Gmail automation, and AI project work")
        if "collaboration" in lowered:
            result.append("where collaboration might make sense")
        return result[:4] or ContextualDraftGenerator._prioritized_focus_topics(topics)


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        key = value.lower()
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result
