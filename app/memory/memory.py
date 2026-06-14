import json
import re
from dataclasses import dataclass, field

from app.database import Database
from app.utils.helpers import normalize_email_address


@dataclass(slots=True)
class SenderProfile:
    email: str
    total_interactions: int = 0
    approvals: int = 0
    rejections: int = 0
    auto_replies: int = 0
    avg_risk: float = 50.0
    trust_score: int = 50


@dataclass(slots=True)
class RelationshipProfile:
    email: str
    relationship_type: str = "unknown"
    preferred_tone: str = "normal"
    preferred_signoff: str | None = None
    preferred_greeting: str | None = None
    tone_confidence: float = 0.0


@dataclass(slots=True)
class ThreadSummary:
    thread_id: str
    summary: str = ""
    commitments: list[str] = field(default_factory=list)
    pending_questions: list[str] = field(default_factory=list)
    scheduling_context: list[str] = field(default_factory=list)
    tone_shifts: list[str] = field(default_factory=list)
    unresolved_items: list[str] = field(default_factory=list)

    def as_context(self) -> dict[str, object]:
        return {
            "summary": self.summary,
            "commitments": self.commitments,
            "pending_questions": self.pending_questions,
            "scheduling_context": self.scheduling_context,
            "tone_shifts": self.tone_shifts,
            "unresolved_items": self.unresolved_items,
        }


@dataclass(slots=True)
class StylePreferences:
    sentence_length: str = "balanced"
    directness: str = "balanced"
    formality: str = "normal"
    preferred_greeting: str | None = None
    preferred_signoff: str | None = None
    confidence: float = 0.0


class MemoryStore:
    def __init__(self, db: Database) -> None:
        self.db = db

    def get_sender_profile(self, sender: str) -> SenderProfile:
        sender_email = normalize_email_address(sender)
        row = self.db.get_sender_memory(sender_email)
        if not row:
            return SenderProfile(email=sender_email)
        trust = self._trust_score(
            total=row["total_interactions"],
            approvals=row["approvals"],
            rejections=row["rejections"],
            auto_replies=row["auto_replies"],
            avg_risk=row["avg_risk"],
        )
        return SenderProfile(
            email=sender_email,
            total_interactions=row["total_interactions"],
            approvals=row["approvals"],
            rejections=row["rejections"],
            auto_replies=row["auto_replies"],
            avg_risk=row["avg_risk"],
            trust_score=trust,
        )

    def record_interaction(self, sender: str, *, approved: bool, auto_replied: bool, risk_score: int, rejected: bool = False) -> None:
        self.db.update_sender_memory(
            normalize_email_address(sender),
            approved=approved,
            auto_replied=auto_replied,
            risk_score=risk_score,
            rejected=rejected,
        )

    def get_relationship_profile(self, sender: str) -> RelationshipProfile:
        sender_email = normalize_email_address(sender)
        row = self.db.get_relationship_memory(sender_email)
        if not row:
            relationship_type = self.db._infer_relationship_type(sender_email)
            default_tone = "formal" if relationship_type in {"recruiter", "professor", "client"} else "normal"
            return RelationshipProfile(email=sender_email, relationship_type=relationship_type, preferred_tone=default_tone)
        return RelationshipProfile(
            email=sender_email,
            relationship_type=row["relationship_type"],
            preferred_tone=row["preferred_tone"],
            preferred_signoff=row["preferred_signoff"],
            preferred_greeting=row["preferred_greeting"] if "preferred_greeting" in row.keys() else None,
            tone_confidence=float(row["tone_confidence"]) if "tone_confidence" in row.keys() else 0.0,
        )

    def record_tone_selection(self, sender: str, tone: str) -> None:
        sender_email = normalize_email_address(sender)
        self.db.upsert_relationship_memory(
            sender_email,
            relationship_type=self.db._infer_relationship_type(sender_email),
            preferred_tone=tone,
            tone=tone,
        )

    def record_user_edit(self, sender: str, edited_reply: str) -> None:
        signoff = self._extract_signoff(edited_reply)
        greeting = self._extract_greeting(edited_reply)
        sender_email = normalize_email_address(sender)
        self.db.upsert_relationship_memory(
            sender_email,
            relationship_type=self.db._infer_relationship_type(sender_email),
            preferred_signoff=signoff,
            preferred_greeting=greeting,
            edited_reply=edited_reply,
        )

    def record_approved_draft(self, sender: str, draft: str, tone: str | None = None) -> None:
        sender_email = normalize_email_address(sender)
        self.db.upsert_relationship_memory(
            sender_email,
            relationship_type=self.db._infer_relationship_type(sender_email),
            tone=tone,
        )

    def record_regeneration_choice(self, sender: str, *, strategy: str, draft: str, tone: str | None = None) -> None:
        sender_email = normalize_email_address(sender)
        self.db.upsert_relationship_memory(
            sender_email,
            relationship_type=self.db._infer_relationship_type(sender_email),
            tone=tone,
            edited_reply=f"regenerated:{strategy}\n{draft}",
        )

    def record_rejection(self, sender: str) -> None:
        sender_email = normalize_email_address(sender)
        self.db.upsert_relationship_memory(
            sender_email,
            relationship_type=self.db._infer_relationship_type(sender_email),
            rejected=True,
        )

    def get_style_preferences(self, sender: str) -> StylePreferences:
        profile = self.get_relationship_profile(sender)
        row = self.db.get_relationship_memory(normalize_email_address(sender))
        edits: list[str] = []
        tones: list[str] = []
        if row:
            edits = [item for item in json.loads(row["edit_history"] or "[]") if isinstance(item, str)]
            tones = [item for item in json.loads(row["tone_history"] or "[]") if isinstance(item, str)]
        sentence_lengths = [_average_sentence_words(edit) for edit in edits if edit.strip()]
        avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 12
        sentence_length = "short" if avg_sentence_length <= 7 else "long" if avg_sentence_length >= 18 else "balanced"
        direct_markers = sum(_looks_direct(edit) for edit in edits[-8:])
        directness = "direct" if edits and direct_markers >= max(1, len(edits[-8:]) // 2) else "balanced"
        formality = profile.preferred_tone or (tones[-1] if tones else "normal")
        return StylePreferences(
            sentence_length=sentence_length,
            directness=directness,
            formality=formality,
            preferred_greeting=profile.preferred_greeting,
            preferred_signoff=profile.preferred_signoff,
            confidence=profile.tone_confidence,
        )

    def get_thread_summary(self, thread_id: str) -> ThreadSummary:
        row = self.db.get_thread_summary(thread_id)
        if not row:
            return ThreadSummary(thread_id=thread_id)
        return ThreadSummary(
            thread_id=thread_id,
            summary=row["summary"],
            commitments=json.loads(row["commitments"] or "[]"),
            pending_questions=json.loads(row["pending_questions"] or "[]"),
            scheduling_context=json.loads(row["scheduling_context"] or "[]"),
            tone_shifts=json.loads(row["tone_shifts"] or "[]"),
            unresolved_items=json.loads(row["unresolved_items"] or "[]"),
        )

    def record_thread_observation(self, email, *, reply_text: str | None = None) -> None:
        existing = self.get_thread_summary(email.thread_id)
        source = f"{email.subject}. {email.body}"
        combined = f"{source}\n{reply_text or ''}"
        summary = _compact_summary(existing.summary, source)
        commitments = _dedupe([*existing.commitments, *_extract_commitments(reply_text or combined)])
        pending_questions = _dedupe([*existing.pending_questions, *_extract_questions(source)])
        scheduling_context = _dedupe([*existing.scheduling_context, *_extract_scheduling(combined)])
        tone_shifts = _dedupe([*existing.tone_shifts, *_extract_tone_shifts(source)])
        unresolved_items = _dedupe([*existing.unresolved_items, *_extract_unresolved(source)])
        self.db.upsert_thread_summary(
            email.thread_id,
            summary=summary,
            commitments=commitments,
            pending_questions=pending_questions,
            scheduling_context=scheduling_context,
            tone_shifts=tone_shifts,
            unresolved_items=unresolved_items,
        )

    @staticmethod
    def _extract_signoff(reply: str) -> str | None:
        lines = [line.strip() for line in reply.splitlines() if line.strip()]
        for index in range(len(lines) - 1, max(-1, len(lines) - 5), -1):
            line = lines[index]
            lowered = line.lower()
            if lowered == "ns" and index > 0 and lines[index - 1].lower() in {"cheers,", "cheers", "thanks,", "thanks"}:
                return f"{lines[index - 1]}\n{line}"
            if lowered in {"- ns", "– ns", "regards", "best regards"} or lowered.endswith("ns"):
                return line
        return None

    @staticmethod
    def _extract_greeting(reply: str) -> str | None:
        for line in reply.splitlines()[:4]:
            stripped = line.strip()
            if stripped.lower().startswith(("hello ", "hi ", "hey ", "dear ")):
                return stripped
        return None

    @staticmethod
    def _trust_score(total: int, approvals: int, rejections: int, auto_replies: int, avg_risk: float) -> int:
        if total <= 0:
            return 50
        score = 50 + approvals * 8 + auto_replies * 5 - rejections * 12 - int(avg_risk / 5)
        return max(0, min(100, score))


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = " ".join(str(value).split()).strip()
        key = clean.lower()
        if clean and key not in seen:
            seen.add(key)
            result.append(clean)
    return result[-8:]


def _compact_summary(previous: str, source: str) -> str:
    subject = source.split(".", 1)[0].strip()
    if previous and subject.lower() in previous.lower():
        return previous[:420]
    text = f"{previous}; {subject}" if previous else subject
    return text.strip("; ")[:420]


def _extract_questions(text: str) -> list[str]:
    return [sentence.strip() for sentence in re.split(r"(?<=[?.!])\s+", text) if "?" in sentence][:4]


def _extract_scheduling(text: str) -> list[str]:
    patterns = [
        r"\b(?:next\s+)?(?:today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b(?:.{0,40}?\b\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm))?",
        r"\b(?:between|from|after|around)\s+\d{1,2}(?::\d{2})?\s*(?:and|-|to)?\s*\d{0,2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b",
        r"\b\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm)\b",
    ]
    matches: list[str] = []
    for pattern in patterns:
        matches.extend(match.group(0).strip() for match in re.finditer(pattern, text, re.IGNORECASE))
    return _dedupe(matches)


def _extract_commitments(text: str) -> list[str]:
    if not text:
        return []
    commitments = []
    for sentence in re.split(r"(?<=[?.!])\s+", text):
        lowered = sentence.lower()
        if any(marker in lowered for marker in ("works", "i will", "i'll", "will share", "will send", "see you", "confirmed")):
            commitments.append(sentence.strip())
    return commitments[:5]


def _extract_unresolved(text: str) -> list[str]:
    unresolved = []
    for sentence in re.split(r"(?<=[?.!])\s+", text):
        lowered = sentence.lower()
        if "?" in sentence or any(marker in lowered for marker in ("please ", "could you", "can you", "let me know", "share ")):
            unresolved.append(sentence.strip())
    return unresolved[:6]


def _extract_tone_shifts(text: str) -> list[str]:
    lowered = text.lower()
    shifts = []
    if any(word in lowered for word in ("urgent", "asap", "immediately")):
        shifts.append("urgent")
    if any(word in lowered for word in ("frustrated", "upset", "angry", "disappointed")):
        shifts.append("tense")
    if any(word in lowered for word in ("excited", "great", "glad", "happy")):
        shifts.append("positive")
    return shifts


def _average_sentence_words(text: str) -> float:
    sentences = [item for item in re.split(r"[.!?]\s+|\n{2,}", text) if item.strip()]
    if not sentences:
        return 12
    lengths = [len(re.findall(r"[A-Za-z0-9']+", sentence)) for sentence in sentences]
    return sum(lengths) / max(1, len(lengths))


def _looks_direct(text: str) -> bool:
    lowered = text.lower()
    return (
        _average_sentence_words(text) <= 8
        or any(marker in lowered for marker in ("works.", "sounds good.", "confirmed.", "see you then."))
    )
