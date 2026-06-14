from dataclasses import dataclass, field

from app.models.email import EmailMessage


@dataclass(slots=True)
class PriorityAssessment:
    level: str
    reasons: list[str] = field(default_factory=list)


class PriorityEngine:
    """Small deterministic priority layer for executive-assistant triage."""

    URGENT_TERMS = {"urgent", "asap", "deadline", "today", "immediately", "interview"}
    MEDIUM_SENDERS = {"professor", "recruiter", "client", "hr"}
    EXECUTIVE_TERMS = {"internship", "opportunity", "collaboration", "partnership", "recruiter", "interview", "project discussion"}
    CASUAL_TERMS = {"coffee", "breakfast", "lunch", "dinner", "campus", "uvce", "movie", "hangout"}

    def assess(self, email: EmailMessage, *, risk_score: int, urgency: str) -> PriorityAssessment:
        text = f"{email.sender} {email.subject} {email.body}".lower()
        reasons: list[str] = []
        if urgency in {"high", "critical"} or risk_score >= 75 or any(term in text for term in self.URGENT_TERMS):
            reasons.append("urgent/risky/deadline signal")
            return PriorityAssessment("Executive Attention Required", reasons)
        if any(term in text for term in self.EXECUTIVE_TERMS):
            reasons.append("professional opportunity signal")
            return PriorityAssessment("Executive Attention Required", reasons)
        if any(sender_hint in text for sender_hint in self.MEDIUM_SENDERS) or 35 <= risk_score < 75:
            reasons.append("important sender or moderate risk")
            return PriorityAssessment("Medium", reasons)
        if any(term in text for term in self.CASUAL_TERMS) or risk_score < 30:
            reasons.append("casual or low-risk context")
            return PriorityAssessment("Casual", reasons)
        return PriorityAssessment("Medium", ["default executive priority"])
