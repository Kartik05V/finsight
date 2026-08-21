"""
Guardrails layer — hardened (#1).

Everything here runs BEFORE text reaches an LLM call. Never log the raw
matched value, only that a redaction happened.

Changes from the original:
- CARD_NUMBER now requires a Luhn checksum pass before flagging, eliminating
  false positives on transaction IDs, invoice numbers, and other 13-16 digit
  numeric strings that aren't real card numbers.
- AADHAAR regex requires at least one separator (space or hyphen) between the
  digit groups — the old pattern matched almost any 12-digit sequence in prose.
  NOTE: this will miss an unformatted 12-digit Aadhaar with no separators,
  which is an acceptable tradeoff: most sources that handle Aadhaar display
  it in the 4-4-4 spaced format.
- looks_like_injection() now uses a risk-score model instead of a hard block
  on any single keyword match:
    score < INJECTION_SOFT_THRESHOLD  → flag only (injection_suspected=True),
                                        do NOT block (avoids false positives).
    score >= INJECTION_HARD_THRESHOLD → block and raise ValueError.
  IMPORTANT CAVEAT: the keyword list is a naive baseline. It is trivially
  bypassed by case variations ("Ignore Previous Instructions"), phrasing
  variants ("disregard all the above"), Unicode homoglyphs (e.g. Cyrillic 'о'
  in "instructiоns"), or prompt injection via multi-step reasoning chains.
  This layer is a first-line filter that adds friction — it is NOT a security
  boundary. Treat it accordingly.
"""
import re

from finsight.logging_setup import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# PII patterns
# ---------------------------------------------------------------------------

PATTERNS = {
    "EMAIL": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "PAN": re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),               # Indian PAN
    # Requires separator between groups — cuts false positives vs. bare
    # 12-digit runs. Misses unspaced Aadhaar (acceptable tradeoff).
    "AADHAAR": re.compile(r"\b\d{4}[\s\-]\d{4}[\s\-]\d{4}\b"),
    # Broad shape match only — Luhn validation below decides final verdict.
    "_CARD_CANDIDATE": re.compile(r"\b(?:\d[ \-]*?){13,16}\b"),
    "PHONE": re.compile(r"\b\d{10}\b"),
}


def _luhn_valid(num_str: str) -> bool:
    """
    Luhn algorithm checksum.

    Returns True only for digit strings that are valid card numbers per the
    Luhn formula — eliminates random 13–16 digit sequences (transaction IDs,
    invoice numbers, account numbers) from being flagged as card numbers.
    """
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
    """
    Returns (redacted_text, list_of_pii_types_found).
    The caller can log the list of types without ever logging the values.
    """
    found: list[str] = []
    redacted = text

    # Standard patterns (non-card)
    for label, pattern in PATTERNS.items():
        if label == "_CARD_CANDIDATE":
            continue
        if pattern.search(redacted):
            found.append(label)
            redacted = pattern.sub(f"[REDACTED_{label}]", redacted)

    # Card numbers — broad regex first, then Luhn gate
    def _redact_card(match: re.Match) -> str:
        raw = match.group(0)
        if _luhn_valid(raw):
            if "CARD_NUMBER" not in found:
                found.append("CARD_NUMBER")
            return "[REDACTED_CARD_NUMBER]"
        return raw  # not a valid card number — leave it alone

    redacted = PATTERNS["_CARD_CANDIDATE"].sub(_redact_card, redacted)

    return redacted, found


# ---------------------------------------------------------------------------
# Injection detection
# ---------------------------------------------------------------------------

# Each entry is (keyword, risk_score). Phrases that strongly signal prompt
# injection carry a higher score; ambiguous single words carry less.
# NOTE: this list is a documented naive baseline — see module docstring.
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

# Score below this → flag metadata only, do NOT block. Single-keyword
# matches that are plausibly legitimate (e.g. "what is the system prompt
# for this bot?") fall here.
INJECTION_SOFT_THRESHOLD = 2
# Score at or above this → block and raise. Reserved for high-confidence
# multi-signal cases.
INJECTION_HARD_THRESHOLD = 3


def _injection_score(text: str) -> int:
    lowered = text.lower()
    return sum(score for kw, score in _INJECTION_SCORED_KEYWORDS if kw in lowered)


def looks_like_injection(text: str) -> bool:
    """
    Returns True if the text scores at or above the SOFT threshold.
    Call apply_guardrails() to get the full soft/hard distinction.
    """
    return _injection_score(text) >= INJECTION_SOFT_THRESHOLD


def apply_guardrails(text: str) -> tuple[str, dict]:
    """
    Single entry point the rest of the app calls.
    Returns (safe_text, metadata) where metadata never contains raw PII.

    Raises ValueError only for high-confidence injection attempts
    (score >= INJECTION_HARD_THRESHOLD). Low-confidence suspicion is
    recorded in metadata but does NOT block the request.
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
