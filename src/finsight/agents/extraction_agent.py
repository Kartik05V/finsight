"""
Extraction agent — Phase 2.

Job: take raw statement text (or a pandas DataFrame from a CSV) and return
a validated ExtractionResult. This is where `.with_structured_output()`
does the heavy lifting — the LLM is forced to match the Pydantic schema,
so you don't have to hand-write a parser for every bank's format.

Uses the gateway's "extraction" task model (fast/cheap) — parsing rows
into a fixed schema is repetitive structured work, not the kind of thing
that needs your strongest model.
"""
from pathlib import Path

import pandas as pd

from finsight.config import get_llm, llm_call_with_retry
from finsight.logging_setup import get_logger
from finsight.models import ExtractionResult, Transaction

logger = get_logger(__name__)

EXTRACTION_SYSTEM_PROMPT = """You are a financial statement extraction assistant.
Given raw statement text, extract every transaction into structured data.
- Use ISO date format (YYYY-MM-DD).
- Positive amount = money spent, negative = refund/credit.
- Pick the closest matching category; use 'other' if unsure.
- If a row is unparseable, add a note to `warnings` instead of guessing wildly.
"""


def build_extraction_agent():
    llm = get_llm(task="extraction")
    return llm.with_structured_output(ExtractionResult)


def extract_transactions(raw_statement_text: str) -> ExtractionResult:
    messages = [
        ("system", EXTRACTION_SYSTEM_PROMPT),
        ("human", raw_statement_text),
    ]

    # Try the fast/cheap model first (the gateway's normal routing for
    # this task). If it fails on this schema — smaller models can be
    # less reliable at strict function-calling format for complex,
    # multi-field schemas even when the underlying content is correct —
    # fall back to the strong model instead of crashing. This is itself
    # a real gateway pattern: automatic failover, not just cost routing.
    try:
        agent = build_extraction_agent()
        result: ExtractionResult = llm_call_with_retry(agent.invoke, messages, task="extraction")
        return result
    except Exception as e:
        logger.warning(
            f"fast model failed on extraction ({type(e).__name__}), "
            f"retrying with strong model..."
        )
        llm = get_llm(task="extraction_fallback")
        agent = llm.with_structured_output(ExtractionResult)
        result = llm_call_with_retry(agent.invoke, messages, task="extraction_fallback")
        return result


def load_statement_text(csv_path: str | Path, chunk_size: int = 40) -> list[str]:
    """
    Reads a raw CSV statement and splits it into text chunks of `chunk_size`
    rows each. Chunking keeps each LLM call small, cheap, and reliable —
    a 500-row statement in one call is where extraction quality falls apart.
    """
    df = pd.read_csv(csv_path)
    rows_as_text = df.to_csv(index=False).splitlines()
    header, rows = rows_as_text[0], rows_as_text[1:]

    chunks = []
    for i in range(0, len(rows), chunk_size):
        chunk_rows = rows[i : i + chunk_size]
        chunks.append("\n".join([header] + chunk_rows))
    return chunks


def validate_extraction(result: ExtractionResult) -> ExtractionResult:
    """
    Post-extraction validation pass (#4).

    Checks for common extraction quality issues and appends warnings
    instead of raising — the caller decides what to do with them.
    Doesn't modify transactions, only adds metadata warnings.
    """
    warnings = list(result.warnings)

    # 1. Check dates fall within the stated period (if the LLM gave us one)
    if result.statement_period_start and result.statement_period_end:
        for t in result.transactions:
            if not (result.statement_period_start <= t.date <= result.statement_period_end):
                warnings.append(
                    f"Transaction date {t.date} ({t.description[:30]!r}) "
                    f"is outside statement period "
                    f"{result.statement_period_start}–{result.statement_period_end}"
                )

    # 2. Flag transactions the LLM left as OTHER — low-confidence category
    other_count = sum(
        1 for t in result.transactions if t.category.value == "other"
    )
    if other_count:
        warnings.append(
            f"{other_count} transaction(s) have category='other' "
            f"(low category confidence — review manually)"
        )

    # 3. Plausibility check: total spend should be positive for a real statement
    total_spend = sum(t.amount for t in result.transactions if t.amount > 0)
    if result.transactions and total_spend == 0:
        warnings.append(
            "Total spend is 0 — extraction may have failed or statement has no debits"
        )

    if warnings != result.warnings:
        logger.info(f"validate_extraction added {len(warnings) - len(result.warnings)} warning(s)")

    return ExtractionResult(
        transactions=result.transactions,
        statement_period_start=result.statement_period_start,
        statement_period_end=result.statement_period_end,
        warnings=warnings,
    )


def extract_from_csv(csv_path: str | Path, chunk_size: int = 40) -> ExtractionResult:
    """
    Full pipeline: read CSV -> chunk -> extract each chunk -> merge results.
    This is what main.py and the evals runner both call.
    """
    import os
    provider = os.getenv("FINSIGHT_PROVIDER", "groq")

    # If using hybrid mode with Gemini, we can pass the whole CSV at once
    # because Gemini 1.5 Flash has a 1M token context window.
    if provider == "hybrid":
        df = pd.read_csv(csv_path)
        raw_text = df.to_csv(index=False)
        logger.info(f"Hybrid mode: bypassing chunking, sending entire {len(df)} row CSV to Gemini")
        result = extract_transactions(raw_text)
        return validate_extraction(result)

    chunks = load_statement_text(csv_path, chunk_size=chunk_size)
    all_transactions: list[Transaction] = []
    all_warnings: list[str] = []

    for chunk in chunks:
        result = extract_transactions(chunk)
        all_transactions.extend(result.transactions)
        all_warnings.extend(result.warnings)

    raw_result = ExtractionResult(transactions=all_transactions, warnings=all_warnings)
    return validate_extraction(raw_result)


if __name__ == "__main__":
    # Phase 2 checkpoint: run `python -m finsight.agents.extraction_agent`
    # and confirm you get back 15 clean Transaction objects matching
    # data/sample_statement.csv (no LLM math, no hallucinated rows).
    sample_path = Path(__file__).parent.parent.parent.parent / "data" / "sample_statement.csv"
    result = extract_from_csv(sample_path)
    print(f"Extracted {len(result.transactions)} transactions, "
          f"{len(result.warnings)} warnings\n")
    for t in result.transactions:
        print(t)
