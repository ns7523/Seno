import re
from dataclasses import dataclass, field

from app.models.email import EmailMessage
from app.utils.helpers import is_noreply_address


NEVER_REPLY_KEYWORDS = {
    "unsubscribe",
    "newsletter",
    "marketing",
    "promotion",
    "promotions",
    "sale",
    "discount",
    "limited offer",
    "scam",
    "phishing",
    "verify your account",
    "password reset",
    "password",
    "otp",
    "one-time password",
    "invoice attached from unknown",
    "verify your account",
    "click here",
}

HIGH_RISK_BUSINESS_KEYWORDS = {
    "legal",
    "contract",
    "agreement",
    "lawsuit",
    "salary",
    "compensation",
    "offer letter",
    "termination",
    "resignation",
    "hr",
    "human resources",
    "payment",
    "bank",
    "finance",
    "financial",
    "refund",
    "angry",
    "urgent complaint",
    "confidential",
    "ssn",
    "aadhaar",
    "passport",
    "invoice",
}

HIGH_RISK_BUSINESS_PHRASES = {
    "approve invoice",
    "invoice approval",
    "payment approval",
    "approve payment",
    "sign contract",
    "contract signing",
    "legal documents",
    "sign legal",
}

CASUAL_SOCIAL_KEYWORDS = {
    "breakfast",
    "dinner",
    "coffee",
    "lunch",
    "meetup",
    "hangout",
    "hang out",
    "campus",
    "class",
    "uvce",
    "movie",
    "party",
    "catch up",
    "casual meeting",
}

SOCIAL_INVITE_KEYWORDS = {"invite", "invitation"}

NEUTRAL_LOW_RISK_KEYWORDS = {
    "thanks",
    "thank you",
    "hello",
    "hi",
    "acknowledge",
    "received",
}

EXECUTIVE_CONTEXT_KEYWORDS = {
    "meeting",
    "schedule",
    "scheduled",
    "scheduling",
    "interview",
    "internship",
    "opportunity",
    "collaboration",
    "collaborate",
    "networking",
    "recruiter",
    "project review",
    "project discussion",
    "deployment",
    "partnership",
    "introduction",
    "availability",
    "available",
    "discussion",
    "call",
    "connect",
    "reviewed your project",
    "reviewed your portfolio",
    "tomorrow",
}


@dataclass(slots=True)
class RiskAssessment:
    risk_score: int
    requires_approval: bool
    never_reply: bool
    reasons: list[str] = field(default_factory=list)


class RiskEngine:
    def __init__(self, approval_threshold: int = 55) -> None:
        self.approval_threshold = approval_threshold

    def assess(self, email: EmailMessage, *, has_attachments: bool | None = None, sender_trust: int = 50) -> RiskAssessment:
        text = f"{email.sender} {email.subject} {email.body}".lower()
        score = 15
        reasons: list[str] = []
        never_reply = False

        if is_noreply_address(email.sender):
            score += 65
            never_reply = True
            reasons.append("noreply sender")

        if self._matches_any(text, NEVER_REPLY_KEYWORDS) or self._has_suspicious_link(text):
            score += 45
            never_reply = True
            reasons.append("newsletter/spam/phishing pattern")

        if re.search(r"\b(category:promotions|unsubscribe|marketing preferences)\b", text):
            score += 30
            never_reply = True
            reasons.append("promotional content")

        matched_business = sorted(keyword for keyword in HIGH_RISK_BUSINESS_KEYWORDS if self._contains_keyword(text, keyword))
        matched_business.extend(sorted(phrase for phrase in HIGH_RISK_BUSINESS_PHRASES if phrase in text))
        matched_social = sorted(keyword for keyword in CASUAL_SOCIAL_KEYWORDS if self._contains_keyword(text, keyword))
        matched_social_invites = sorted(keyword for keyword in SOCIAL_INVITE_KEYWORDS if self._contains_keyword(text, keyword))
        if matched_social_invites and matched_social:
            matched_social.extend(matched_social_invites)
        matched_neutral = sorted(keyword for keyword in NEUTRAL_LOW_RISK_KEYWORDS if self._contains_keyword(text, keyword))
        matched_executive = sorted(keyword for keyword in EXECUTIVE_CONTEXT_KEYWORDS if self._contains_keyword(text, keyword))

        if matched_business:
            score = max(score, 55)
            score += min(35, 10 * len(matched_business))
            reasons.append(f"high-risk business topic: {', '.join(matched_business[:4])}")

        if matched_social and not matched_business and not never_reply:
            score = min(score, 18)
            reasons.append(f"casual/social context: {', '.join(matched_social[:4])}")

        if matched_neutral and not matched_business and not never_reply:
            score = min(score, 20)
            reasons.append(f"neutral low-risk context: {', '.join(matched_neutral[:4])}")

        if matched_executive:
            score = max(score, 55)
            reasons.append(f"executive attention signal: {', '.join(matched_executive[:5])}")

        if self._has_meeting_language(text) and matched_business:
            score = max(score, 75)
            reasons.append("business-sensitive meeting context")

        attachment_present = has_attachments if has_attachments is not None else email.has_attachments
        if attachment_present:
            score += 25
            reasons.append("attachment present")

        if sender_trust < 30:
            score += 10
            reasons.append("low sender trust")
        elif sender_trust > 75:
            score -= 8
            reasons.append("trusted sender")

        score = max(0, min(100, score))
        requires_approval = (
            never_reply
            or bool(matched_business)
            or bool(matched_executive)
            or bool(attachment_present)
            or score >= self.approval_threshold
        )
        return RiskAssessment(
            risk_score=score,
            requires_approval=requires_approval,
            never_reply=never_reply,
            reasons=reasons or ["default risk assessment"],
        )

    @staticmethod
    def _has_meeting_language(text: str) -> bool:
        return bool(re.search(r"\b(meeting|meet|schedule|calendar|available)\b", text))

    @classmethod
    def _matches_any(cls, text: str, keywords: set[str]) -> bool:
        return any(cls._contains_keyword(text, keyword) for keyword in keywords)

    @staticmethod
    def _contains_keyword(text: str, keyword: str) -> bool:
        if " " in keyword or "-" in keyword:
            return keyword in text
        if keyword == "call":
            return bool(re.search(r"\b(call|phone call|video call)\b", text))
        return bool(re.search(rf"\b{re.escape(keyword)}\b", text))

    @staticmethod
    def _has_suspicious_link(text: str) -> bool:
        has_url = bool(re.search(r"https?://|www\.", text))
        suspicious_terms = ("verify", "login", "reset", "claim", "prize", "urgent", "password", "account", "scam")
        return has_url and any(term in text for term in suspicious_terms)
