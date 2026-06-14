from typing import Literal


FooterMode = Literal["minimal", "professional", "executive", "stealth"]


FOOTER_TEXT: dict[str, str] = {
    "minimal": "Sent via Seno.",
    "professional": "Sent via Seno, NS's executive communication assistant.",
    "executive": "Delivered through Seno - NS's personal executive assistant.",
    "stealth": "",
}


def footer_text(mode: str) -> str:
    return FOOTER_TEXT.get(mode.lower().strip(), FOOTER_TEXT["professional"])


def apply_seno_footer(body: str, mode: str) -> str:
    text = body.strip()
    footer = footer_text(mode)
    if not footer:
        return _strip_existing_footer(text).strip()
    cleaned = _strip_existing_footer(text).strip()
    return f"{cleaned}\n\n{footer}" if cleaned else footer


def split_seno_footer(body: str) -> tuple[str, str | None]:
    text = body.strip()
    for footer in sorted((value for value in FOOTER_TEXT.values() if value), key=len, reverse=True):
        if text == footer:
            return "", footer
        suffix = f"\n\n{footer}"
        if text.endswith(suffix):
            return text[: -len(suffix)].strip(), footer
    return text, None


def _strip_existing_footer(body: str) -> str:
    main, _ = split_seno_footer(body)
    return main
