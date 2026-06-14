import asyncio
import html
import random
import re
from collections.abc import Awaitable, Callable
from email.utils import parseaddr
from typing import TypeVar


T = TypeVar("T")


def normalize_email_address(value: str) -> str:
    _, address = parseaddr(value or "")
    return address.strip().lower()


def is_noreply_address(value: str) -> bool:
    address = normalize_email_address(value)
    local = address.split("@", 1)[0]
    return bool(re.search(r"(^|[._-])(no[-_]?reply|donotreply|notifications?|mailer-daemon)($|[._-])", local))


def strip_html(value: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", value or "")
    text = re.sub(r"(?s)<br\s*/?>", "\n", text)
    text = re.sub(r"(?s)<.*?>", " ", text)
    text = html.unescape(text)
    return re.sub(r"[ \t\r\f\v]+", " ", text).strip()


async def retry_async(
    operation: Callable[[], Awaitable[T]],
    attempts: int = 3,
    base_delay: float = 0.5,
    retry_exceptions: tuple[type[Exception], ...] = (Exception,),
) -> T:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await operation()
        except retry_exceptions as exc:
            last_error = exc
            if attempt == attempts:
                break
            await asyncio.sleep(base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.1))
    assert last_error is not None
    raise last_error
