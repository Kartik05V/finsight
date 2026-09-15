"""
Guardrails: runs before any text reaches an LLM call.
- PII redaction: email, PAN, Aadhaar, card numbers (Luhn-validated), phone.
- Injection detection: risk-score model (soft flag vs hard block).
  NOTE: keyword-based — adds friction but is NOT a security boundary.
"""
import re

from finsight.logging_setup import get_logger

logger = get_logger(__name__)


PATTERNS = {
    "EMAIL": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "PAN": re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),
    # Requires separator between groups — cuts false positives on bare 12-digit runs
    "AADHAAR": re.compile(r"\b\d{4}[\s\-]\d{4}[\s\-]\d{4}\b"),
    # Broad shape match — Luhn validation decides the final verdict
    "_CARD_CANDIDATE": re.compile(r"\b(?:\d[ \-]*?){13,16}\b"),
    "PHONE": re.compile(r"\b\d{10}\b"),
}


def _luhn_valid(num_str: str) -> bool:
    """Return True only for digit strings that pass the Luhn checksum."""
    digits = [int(c) for c in num_str if c.isdigit()]
    if len(digits) < 13:
        return False
    digits.reverse()
    total = 0
    for i, d in enumerate(digits):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def redact_pii(text: str) -> tuple[str, list[str]]:
    """Returns (redacted_text, list_of_pii_types_found)."""
    found: list[str] = []
    redacted = text

    for label, pattern in PATTERNS.items():
        if label == "_CARD_CANDIDATE":
            continue
        if pattern.search(redacted):
            found.append(label)
            redacted = pattern.sub(f"[REDACTED_{label}]", redacted)

    def _redact_card(match: re.Match) -> str:
        raw = match.group(0)
        if _luhn_valid(raw):
            if "CARD_NUMBER" not in found:
                found.append("CARD_NUMBER")
            return "[REDACTED_CARD_NUMBER]"
        return raw

    redacted = PATTERNS["_CARD_CANDIDATE"].sub(_redact_card, redacted)
    return redacted, found


# Injection risk scores: higher = stronger signal. Soft flag vs hard block thresholds below.
_INJECTION_SCORED_KEYWORDS: list[tuple[str, int]] = [
    ("ignore previous instructions", 3),
    ("ignore all previous", 3),
    ("disregard the above", 3),
    ("disregard all", 2),
    ("you are now", 2),
    ("system prompt", 2),
    ("forget your instructions", 3),
    ("act as if", 1),
    ("pretend you are", 1),
    ("new persona", 1),
]

INJECTION_SOFT_THRESHOLD = 2  # flag in metadata but do NOT block
INJECTION_HARD_THRESHOLD = 3  # block and raise ValueError


def _injection_score(text: str) -> int:
    lowered = text.lower()
    return sum(score for kw, score in _INJECTION_SCORED_KEYWORDS if kw in lowered)


def looks_like_injection(text: str) -> bool:
    """Returns True if the text scores at or above the soft threshold."""
    return _injection_score(text) >= INJECTION_SOFT_THRESHOLD


def apply_guardrails(text: str) -> tuple[str, dict]:
    """
    Single entry point. Returns (safe_text, metadata).
    Raises ValueError only for high-confidence injection (score >= HARD_THRESHOLD).
    """
    redacted, pii_found = redact_pii(text)
    score = _injection_score(text)
    hard_block = score >= INJECTION_HARD_THRESHOLD
    soft_flag = score >= INJECTION_SOFT_THRESHOLD

    if pii_found:
        logger.info("pii_redacted", extra={"pii_types": pii_found})

    if soft_flag:
        logger.warning(
            "injection_suspected",
            extra={"score": score, "hard_block": hard_block},
        )

    metadata = {
        "pii_types_found": pii_found,
        "injection_suspected": soft_flag,
        "injection_score": score,
    }

    if hard_block:
        raise ValueError(
            f"Potential prompt injection detected (score={score}) — request blocked."
        )

    return redacted, metadata
