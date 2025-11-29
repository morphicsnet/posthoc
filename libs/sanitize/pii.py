from __future__ import annotations

"""
Lightweight PII scrubbing utilities (pure stdlib).

Environment switches (read once at import):
- PII_MASK_EMAIL=1 (default 1)
- PII_MASK_PHONE=1 (default 1)
- PII_MASK_NUMBERS=0 (default 0)

Functions:
- mask_email(text: str) - replace email addresses with "[EMAIL]"
- mask_phone(text: str) - replace common phone patterns with "[PHONE]"
- mask_numbers(text: str, min_digits=9) - replace digit runs of length >= min_digits with "[NUMBER]"
- sanitize_message(text: str) - apply configured pipeline deterministically

Notes:
- Fast, deterministic regexes only (no catastrophic backtracking).
- Intended for sanitizing log copies only; do not apply on data path.
"""

import os
import re
from typing import Optional

# Compile once; fast patterns with bounded constructs
_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}\b"
)

# Matches typical phone numbers:
# - Optional country code +1 / +44 etc
# - (XXX) XXX-XXXX, XXX-XXX-XXXX, XXX.XXX.XXXX, XXX XXX XXXX
# - Avoids over-greedy digit matching to keep it fast
_PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[\s.\-]?)?(?:\(?\d{3}\)?[\s.\-]?)\d{3}[\s.\-]?\d{4}"
)

def _numbers_re(min_digits: int) -> re.Pattern[str]:
    # Replace digit runs of at least min_digits, with word boundary-like guards
    m = max(1, int(min_digits))
    return re.compile(rf"(?<!\d)\d{{{m},}}(?!\d)")

def mask_email(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text
    return _EMAIL_RE.sub("[EMAIL]", text)

def mask_phone(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text
    return _PHONE_RE.sub("[PHONE]", text)

def mask_numbers(text: str, min_digits: int = 9) -> str:
    if not isinstance(text, str) or not text:
        return text
    return _numbers_re(min_digits).sub("[NUMBER]", text)

def _is_enabled(name: str, default: str) -> bool:
    v = (os.getenv(name, default) or default).strip()
    return v not in ("0", "false", "False", "FALSE", "")

_P_EMAIL = _is_enabled("PII_MASK_EMAIL", "1")
_P_PHONE = _is_enabled("PII_MASK_PHONE", "1")
_P_NUMS = _is_enabled("PII_MASK_NUMBERS", "0")
_P_NUMS_MIN = 9
try:
    _P_NUMS_MIN = max(1, int(os.getenv("PII_MASK_MIN_DIGITS", "9")))
except Exception:
    _P_NUMS_MIN = 9

def sanitize_message(text: str) -> str:
    """
    Apply configured sanitizer pipeline. Idempotent under repeated calls.
    """
    if not isinstance(text, str) or not text:
        return text
    out = text
    if _P_EMAIL:
        out = mask_email(out)
    if _P_PHONE:
        out = mask_phone(out)
    if _P_NUMS:
        out = mask_numbers(out, _P_NUMS_MIN)
    return out

__all__ = ["mask_email", "mask_phone", "mask_numbers", "sanitize_message"]