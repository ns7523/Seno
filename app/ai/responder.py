import re
from email.utils import parseaddr
from dataclasses import dataclass
from typing import Any

from app.ai.classifier import EmailAnalysis


DISCLAIMER = ""
APPROVED_DISCLAIMER = ""

SENSITIVE_CONTEXT_KEYWORDS = {
    "payment",
    "paid",
    "bank",
    "banking",
    "invoice",
    "refund",
    "finance",
    "financial",
    "wire",
    "transfer",
    "legal",
    "contract",
    "agreement",
    "signed",
    "hr",
    "human resources",
    "salary",
    "compensation",
    "offer letter",
    "termination",
    "otp",
    "password",
    "security code",
    "verification code",
}

SOCIAL_CONTEXT_KEYWORDS = {
    "breakfast",
    "lunch",
    "dinner",
    "coffee",
    "invitation",
    "invite",
    "hang out",
    "catch up",
    "party",
    "movie",
    "uvce",
    "campus",
    "class",
    "friends",
    "friend",
    "meetup",
    "hangout",
    "casual meeting",
}

ALWAYS_UNSAFE_PHRASES = [
    "i have sent the payment",
    "payment has been sent",
    "i confirm payment",
    "i approve the invoice",
    "approve the invoice",
    "approve payment",
    "i confirm",
    "i accept",
    "i signed the contract",
    "signed the contract",
    "approved on your behalf",
    "password is",
    "otp is",
    "security code is",
    "verification code is",
]

CONVERSATIONAL_COMMITMENT_PHRASES = [
    "i will attend",
    "looking forward to seeing you",
    "looking forward to it",
    "see you there",
    "i would love to join",
    "i would be happy to join",
    "happy to join",
    "count me in",
    "i will join",
    "i'll attend",
    "i'll join",
    "i can attend",
    "i can join",
    "yes, that works",
    "sure, tomorrow works",
    "that works for me",
    "works for me",
    "i am available",
    "i'm available",
    "sounds good",
]

VALID_REPLY_STYLES = {"formal", "normal", "friendly"}
GENERIC_NAMES = {"reviewer", "team", "there", "user", "sir", "madam", "noreply", "no reply"}


class ReplySafetyError(ValueError):
    """Raised when a generated reply violates local safety constraints."""


@dataclass(slots=True)
class EmailQualityScore:
    professionalism: int
    warmth: int
    clarity: int
    human_likeness: int
    repetition: int
    ai_likeness: int

    @property
    def needs_improvement(self) -> bool:
        return self.human_likeness < 70 or self.ai_likeness > 35 or self.repetition > 45


def validate_reply(
    analysis: EmailAnalysis,
    *,
    human_approved: bool = False,
    original_email: Any | None = None,
) -> str:
    reply = _strip_known_disclaimer((analysis.suggested_reply or "").strip())
    if analysis.never_reply:
        raise ReplySafetyError("Reply blocked because the message is marked never_reply")
    if not reply:
        raise ReplySafetyError("Reply blocked because the suggested reply is empty")
    if len(reply) > 4000:
        raise ReplySafetyError("Reply blocked because it exceeds the maximum safe length")
    context = _reply_context(analysis, original_email).lower()
    sensitive_context = _has_sensitive_context(context)
    social_context = _is_social_context(context)
    allow_human_commitment = human_approved and not sensitive_context and (
        social_context or _is_scheduling_or_professional_context(context)
    )
    if allow_human_commitment and social_context:
        reply = _humanize_approved_social_reply(reply, original_email)
    elif not human_approved:
        reply = _normalize_auto_acknowledgement(reply)

    lowered = reply.lower()

    if any(phrase in lowered for phrase in ALWAYS_UNSAFE_PHRASES):
        raise ReplySafetyError("Reply blocked because it contains a sensitive unsafe action")
    if (not allow_human_commitment or sensitive_context) and any(phrase in lowered for phrase in CONVERSATIONAL_COMMITMENT_PHRASES):
        raise ReplySafetyError("Reply blocked because it contains an unsafe commitment")
    if DISCLAIMER and reply.startswith(DISCLAIMER):
        return reply
    if APPROVED_DISCLAIMER and reply.startswith(APPROVED_DISCLAIMER):
        return reply
    return reply


def format_reply_style(
    reply: str,
    *,
    style: str,
    original_email: Any | None = None,
    preferred_greeting: str | None = None,
    preferred_signoff: str | None = None,
) -> str:
    """Apply a human-selected email tone after safety validation.

    The caller should validate the source reply first, then validate the styled
    result again before sending. This keeps style formatting separate from the
    safety policy and avoids treating a button click as permission for sensitive
    finance/legal/security actions.
    """
    style = style.lower().strip()
    if style not in VALID_REPLY_STYLES:
        raise ReplySafetyError(f"Unknown reply style: {style}")
    body = humanize_reply(_strip_known_disclaimer(reply).strip(), style=style, original_email=original_email)
    if not body:
        raise ReplySafetyError("Reply blocked because the styled reply body is empty")
    if _has_email_structure(body):
        return body

    name = _sender_first_name(original_email)
    if style == "formal":
        greeting = preferred_greeting or (f"Dear {name}," if name else "Hello,")
        signoff = preferred_signoff or _smart_signoff(style, original_email)
        return f"{greeting}\n\n{body}\n\n{signoff}"
    if style == "normal":
        greeting = preferred_greeting or (f"Hi {name}," if name else "Hello,")
        signoff = preferred_signoff or _smart_signoff(style, original_email)
        return f"{greeting}\n\n{body}\n\n{signoff}"
    greeting = preferred_greeting or (f"Hey {name}," if name else "Hi,")
    signoff = preferred_signoff or _smart_signoff(style, original_email)
    return f"{greeting}\n\n{body}\n\n{signoff}"


def _reply_context(analysis: EmailAnalysis, original_email: Any | None) -> str:
    values = [
        getattr(analysis, "intent", ""),
        getattr(analysis, "summary", ""),
        " ".join(getattr(analysis, "reasons", []) or []),
        getattr(analysis, "tone", ""),
        getattr(analysis, "suggested_reply", ""),
    ]
    if original_email is not None:
        values.extend(
            [
                getattr(original_email, "sender", ""),
                getattr(original_email, "subject", ""),
                getattr(original_email, "body", ""),
            ]
        )
    return " ".join(str(value) for value in values if value)


def _has_sensitive_context(context: str) -> bool:
    for keyword in SENSITIVE_CONTEXT_KEYWORDS:
        if len(keyword) <= 3 or " " not in keyword:
            if re.search(rf"\b{re.escape(keyword)}\b", context):
                return True
        elif keyword in context:
            return True
    return False


def _is_scheduling_or_professional_context(context: str) -> bool:
    if re.search(
        r"\b(?:today|tomorrow|next\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b|\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b",
        context,
    ):
        return True
    return any(
        marker in context
        for marker in (
            "meeting",
            "schedule",
            "scheduling",
            "available",
            "availability",
            "call",
            "discussion",
            "collaboration",
            "collaborate",
            "connect",
            "internship",
            "interview",
            "project",
        )
    )


def _strip_known_disclaimer(reply: str) -> str:
    if DISCLAIMER and reply.startswith(DISCLAIMER):
        return reply.removeprefix(DISCLAIMER)
    if APPROVED_DISCLAIMER and reply.startswith(APPROVED_DISCLAIMER):
        return reply.removeprefix(APPROVED_DISCLAIMER)
    for stale in (
        "This is an automated AI-assisted response on behalf of NS.",
        "This is an AI-assisted response approved by NS.",
    ):
        if reply.startswith(stale):
            return reply.removeprefix(stale).strip()
    return reply


def _has_email_structure(reply: str) -> bool:
    lowered = reply.lower().strip()
    return (
        lowered.startswith(("dear ", "hi ", "hi,", "hey ", "hey,"))
        and any(signoff in lowered for signoff in ("\nbest regards", "\nregards", "\n- ns", "\n– ns"))
    )


def _sender_first_name(original_email: Any | None) -> str | None:
    if original_email is None:
        return None
    display_name, address = parseaddr(getattr(original_email, "sender", ""))
    candidate = display_name or address.split("@", 1)[0]
    candidate = re.sub(r"[^A-Za-z\s.-]", " ", candidate).strip()
    if not candidate:
        return None
    first = candidate.split()[0].strip(".-")
    if not first or first.lower() in GENERIC_NAMES:
        return None
    return first.title() if first else None


def score_email_quality(reply: str) -> EmailQualityScore:
    lowered = reply.lower()
    sentences = [sentence.strip() for sentence in re.split(r"[.!?]\s+", reply) if sentence.strip()]
    repeated_openers = len(sentences) - len({sentence.split(" ", 1)[0].lower() for sentence in sentences if sentence})
    ai_markers = sum(
        marker in lowered
        for marker in (
            "thank you for reaching out",
            "i appreciate your message",
            "i hope you are doing well",
            "key points around",
            "the points you mentioned",
            "ai-assisted",
            "automated",
        )
    )
    human_likeness = max(35, 88 - ai_markers * 14 - repeated_openers * 5)
    repetition = min(100, repeated_openers * 20 + ai_markers * 8)
    warmth = 78 if any(word in lowered for word in ("glad", "happy", "sounds good", "looking forward", "thanks")) else 62
    professionalism = 86 if any(word in lowered for word in ("dear", "regards", "discuss", "available")) else 74
    clarity = 88 if len(reply.strip()) > 40 else 65
    return EmailQualityScore(
        professionalism=professionalism,
        warmth=warmth,
        clarity=clarity,
        human_likeness=human_likeness,
        repetition=repetition,
        ai_likeness=min(100, ai_markers * 22 + repeated_openers * 8),
    )


def humanize_reply(reply: str, *, style: str = "normal", original_email: Any | None = None) -> str:
    body = re.sub(r"\s+\n", "\n", reply.strip())
    replacements = {
        "Thank you for reaching out regarding": "Thanks for sharing the details on",
        "Thank you for reaching out about": "Thanks for sending this over about",
        "I appreciate you reviewing my portfolio and considering me for": "I’m glad the portfolio work was useful context for",
        "I can also cover the points you mentioned, including": "I’d also be happy to talk through",
        "I will come prepared to discuss the questions and next steps you outlined.": "I’ll come prepared to walk through the questions and next steps.",
        "I noted the key points around": "I’ve noted the main points around",
        "Thank you for the update.": "Thanks for the update.",
    }
    for before, after in replacements.items():
        body = body.replace(before, after)
    body = _smooth_topic_lists(body)
    if style == "formal":
        body = body.replace("Thanks for", "Thank you for", 1) if body.startswith("Thanks for") else body
    elif style == "friendly":
        body = body.replace("I would be glad to discuss it further.", "Sounds good, I’d be happy to talk it through.")
        body = body.replace("works for the discussion.", "works for me.")
    return body


def _smooth_topic_lists(body: str) -> str:
    return re.sub(
        r"including ([A-Za-z0-9 ._-]+), ([A-Za-z0-9 ._-]+), and ([A-Za-z0-9 ._-]+)",
        r"including \1, \2, and \3 in more detail",
        body,
    )


def _is_social_context(context: str) -> bool:
    direct_social_terms = SOCIAL_CONTEXT_KEYWORDS - {"invite", "invitation"}
    if any(keyword in context for keyword in direct_social_terms):
        return True
    if any(keyword in context for keyword in ("invite", "invitation")):
        paired_terms = {
            "breakfast",
            "lunch",
            "dinner",
            "coffee",
            "meetup",
            "hangout",
            "hang out",
            "campus",
            "uvce",
            "class",
            "friends",
            "friend",
            "movie",
            "party",
            "catch up",
        }
        return any(term in context for term in paired_terms)
    return False


def _smart_signoff(style: str, original_email: Any | None) -> str:
    context = f"{getattr(original_email, 'sender', '')} {getattr(original_email, 'subject', '')} {getattr(original_email, 'body', '')}".lower()
    if style == "formal":
        if any(word in context for word in ("interview", "recruiter", "professor", "client")):
            return "Best regards,\nNS"
        return "Kind regards,\nNS"
    if style == "friendly":
        if any(word in context for word in ("coffee", "dinner", "breakfast", "movie", "hangout")):
            return "Cheers,\nNS"
        return "- NS"
    if any(word in context for word in ("discuss", "meeting", "interview")):
        return "Looking forward,\nNS"
    return "Regards,\nNS"


def _humanize_approved_social_reply(reply: str, original_email: Any | None) -> str:
    robotic_markers = [
        "follow-up confirmation",
        "personal confirmation",
        "follow up confirmation",
        "has received your invitation",
        "has received the invitation",
        "your message has been received",
        "message has been received",
    ]
    if not any(marker in reply.lower() for marker in robotic_markers):
        return reply

    source_text = " ".join(
        str(value)
        for value in [
            getattr(original_email, "subject", "") if original_email is not None else "",
            getattr(original_email, "body", "") if original_email is not None else "",
        ]
        if value
    )
    time_text = _extract_time(source_text)
    lowered_source = source_text.lower()
    if "breakfast" in lowered_source:
        if time_text:
            return f"Breakfast at {time_text} works. See you then."
        return "Breakfast sounds good. See you then."
    if "dinner" in lowered_source:
        if time_text:
            return f"Dinner at {time_text} works. See you then."
        return "Sounds good, see you then."
    if "coffee" in lowered_source:
        if time_text:
            return f"Coffee at {time_text} works. See you then."
        return "Coffee sounds good. See you then."
    if any(keyword in lowered_source for keyword in ("movie", "hangout", "hang out", "friends", "party")):
        return "Sounds good, see you then."
    if any(keyword in lowered_source for keyword in ("meeting", "meet", "uvce", "campus", "class")):
        if time_text:
            return f"{time_text} works. See you then."
        return "Sounds good, let's meet there."
    return reply


def _normalize_auto_acknowledgement(reply: str) -> str:
    misleading_markers = [
        "follow-up confirmation",
        "personal confirmation",
        "follow up confirmation",
        "personal follow-up",
        "follow-up will be shared",
        "shared separately",
        "will get back",
    ]
    if any(marker in reply.lower() for marker in misleading_markers):
        return "Thanks for the message. NS has received it."
    return reply


def _extract_time(text: str) -> str | None:
    match = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?\b", text, flags=re.IGNORECASE)
    if not match:
        return None
    hour = match.group(1)
    minute = match.group(2)
    suffix = (match.group(3) or "").replace(".", "").upper()
    if minute:
        return f"{hour}:{minute}{(' ' + suffix) if suffix else ''}"
    return f"{hour}{(' ' + suffix) if suffix else ''}"
