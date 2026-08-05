"""
Guardrails layer — Phase 3.

Everything here runs BEFORE text reaches an LLM call. Never log the raw
matched value, only that a redaction happened.
"""
import re

PATTERNS = {
    "EMAIL": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "PAN": re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),               # Indian PAN format
    "AADHAAR": re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),            # Indian Aadhaar format
    "CARD_NUMBER": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "PHONE": re.compile(r"\b\d{10}\b"),
}


def redact_pii(text: str) -> tuple[str, list[str]]:
    """
    Returns (redacted_text, list_of_pii_types_found).
    The caller can log the list of types without ever logging the values.
    """
    found: list[str] = []
    redacted = text
    for label, pattern in PATTERNS.items():
        if pattern.search(redacted):
            found.append(label)
            redacted = pattern.sub(f"[REDACTED_{label}]", redacted)
    return redacted, found


INJECTION_KEYWORDS = [
    "ignore previous instructions",
    "ignore all previous",
    "disregard the above",
    "you are now",
    "system prompt",
]


def looks_like_injection(text: str) -> bool:
    lowered = text.lower()
    return any(kw in lowered for kw in INJECTION_KEYWORDS)


def apply_guardrails(text: str) -> tuple[str, dict]:
    """
    Single entry point the rest of the app calls.
    Returns (safe_text, metadata) where metadata never contains raw PII.
    """
    redacted, pii_found = redact_pii(text)
    injection_flag = looks_like_injection(text)

    metadata = {
        "pii_types_found": pii_found,
        "injection_suspected": injection_flag,
    }

    if injection_flag:
        raise ValueError("Potential prompt injection detected — request blocked.")

    return redacted, metadata
