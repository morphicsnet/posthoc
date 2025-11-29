from __future__ import annotations

import os
import sys
import importlib
from typing import Optional

# Simple env patch helper (mirrors style used in other tests)
class EnvPatch:
    def __init__(self, **overrides: str) -> None:
        self.overrides = overrides
        self.prev: dict[str, Optional[str]] = {}

    def __enter__(self):
        for k, v in self.overrides.items():
            self.prev[k] = os.getenv(k)
            os.environ[k] = v
        return self

    def __exit__(self, exc_type, exc, tb):
        for k, old in self.prev.items():
            if old is None:
                try:
                    del os.environ[k]
                except KeyError:
                    pass
            else:
                os.environ[k] = old


def _reload_sanitizers():
    # Ensure repo root is on sys.path for namespace import 'libs.sanitize.pii'
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    # Invalidate and reload to pick up environment changes
    if "libs.sanitize.pii" in sys.modules:
        del sys.modules["libs.sanitize.pii"]
    if "libs.sanitize" in sys.modules:
        del sys.modules["libs.sanitize"]
    import libs.sanitize.pii as pii  # type: ignore
    importlib.reload(pii)
    return pii


def test_mask_email_and_phone_and_numbers() -> None:
    pii = _reload_sanitizers()

    text = (
        "Contact me at alice@example.com or bob.smith+tag@sub.domain.co.uk. "
        "My US phone is +1 (415) 555-2671, alt 415-555-2671 or 415.555.2671. "
        "Short numbers like 1234 stay; long ID 123456789012 should mask."
    )

    # Direct functions
    masked_email = pii.mask_email(text)
    assert "[EMAIL]" in masked_email and "alice@example.com" not in masked_email, "email masking failed"

    masked_phone = pii.mask_phone(text)
    assert "[PHONE]" in masked_phone and "555-2671" not in masked_phone, "phone masking failed"

    masked_numbers = pii.mask_numbers(text, min_digits=9)
    assert "[NUMBER]" in masked_numbers and "123456789012" not in masked_numbers, "numbers masking failed"


def test_sanitize_message_pipeline_defaults_and_toggle_numbers() -> None:
    # Defaults: PII_MASK_EMAIL=1, PII_MASK_PHONE=1, PII_MASK_NUMBERS=0
    with EnvPatch(PII_MASK_EMAIL="1", PII_MASK_PHONE="1", PII_MASK_NUMBERS="0"):
        pii = _reload_sanitizers()
        inp = "Mail: a@b.com, Phone: 415-555-2671, Card: 4111111111111111"
        out = pii.sanitize_message(inp)
        assert "[EMAIL]" in out, "email should be masked by default"
        assert "[PHONE]" in out, "phone should be masked by default"
        # numbers disabled by default
        assert "4111111111111111" in out and "[NUMBER]" not in out, "numbers should not be masked by default"

    # Enable numbers masking
    with EnvPatch(PII_MASK_EMAIL="1", PII_MASK_PHONE="1", PII_MASK_NUMBERS="1", PII_MASK_MIN_DIGITS="12"):
        pii = _reload_sanitizers()
        inp = "Card: 411111111111"
        out = pii.sanitize_message(inp)
        assert "[NUMBER]" in out and "411111111111" not in out, "numbers should be masked when enabled"


def test_idempotence_and_no_catastrophic_backtracking() -> None:
    # Idempotence
    with EnvPatch(PII_MASK_EMAIL="1", PII_MASK_PHONE="1", PII_MASK_NUMBERS="1", PII_MASK_MIN_DIGITS="9"):
        pii = _reload_sanitizers()
        inp = (
            "User alice@example.com, phone 415-555-2671, id 123456789."
            " Another email bob@co.io and digits 9876543210."
        )
        once = pii.sanitize_message(inp)
        twice = pii.sanitize_message(once)
        assert once == twice, "sanitize_message should be idempotent"

        # Simple stress: long digits string should not blow up (no time assertion, just ensure it returns)
        long_digits = "x" * 1000 + " " + ("1234567890" * 2000)  # 20k digits
        out = pii.sanitize_message(long_digits)
        # Should replace large digit runs
        assert "[NUMBER]" in out, "long digit runs should be masked when enabled"


def _run_all():
    test_mask_email_and_phone_and_numbers()
    test_sanitize_message_pipeline_defaults_and_toggle_numbers()
    test_idempotence_and_no_catastrophic_backtracking()
    print("OK: PII sanitizer tests passed")


if __name__ == "__main__":
    _run_all()