"""
Extraction agent: converts raw statement text / CSV into validated Transaction objects.
Uses .with_structured_output() so the LLM is forced to match the Pydantic schema.
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

    # Try fast model first; fall back to strong model if schema validation fails
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
    """Read a CSV and split into text chunks to keep each LLM call small."""
    df = pd.read_csv(csv_path)
    rows_as_text = df.to_csv(index=False).splitlines()
    header, rows = rows_as_text[0], rows_as_text[1:]

    chunks = []
    for i in range(0, len(rows), chunk_size):
        chunk_rows = rows[i : i + chunk_size]
        chunks.append("\n".join([header] + chunk_rows))
    return chunks


def validate_extraction(result: ExtractionResult) -> ExtractionResult:
    """Post-extraction quality checks: date range, uncategorised rows, zero spend."""
    warnings = list(result.warnings)

    if result.statement_period_start and result.statement_period_end:
        for t in result.transactions:
            if not (result.statement_period_start <= t.date <= result.statement_period_end):
                warnings.append(
                    f"Transaction date {t.date} ({t.description[:30]!r}) "
                    f"is outside statement period "
                    f"{result.statement_period_start}–{result.statement_period_end}"
                )

    other_count = sum(
        1 for t in result.transactions if t.category.value == "other"
    )
    if other_count:
        warnings.append(
            f"{other_count} transaction(s) have category='other' "
            f"(low category confidence — review manually)"
        )

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
    """Full pipeline: read CSV -> chunk -> extract -> merge -> validate."""
    import os
    provider = os.getenv("FINSIGHT_PROVIDER", "groq")

    if provider == "hybrid":
        # Gemini has a 1M token context window — send the whole CSV at once
        df = pd.read_csv(csv_path)
        raw_text = df.to_csv(index=False)
        logger.info(f"Hybrid mode: sending entire {len(df)} row CSV to Gemini")
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
    sample_path = Path(__file__).parent.parent.parent.parent / "data" / "sample_statement.csv"
    result = extract_from_csv(sample_path)
    print(f"Extracted {len(result.transactions)} transactions, "
          f"{len(result.warnings)} warnings\n")
    for t in result.transactions:
        print(t)
