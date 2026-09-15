"""
Analysis agent: aggregates transactions by category in Python,
then uses the LLM only for narrative insights and severity judgment.
"""
import statistics
from collections import defaultdict

from pydantic import BaseModel

from finsight.config import get_llm, llm_call_with_retry
from finsight.logging_setup import get_logger
from finsight.models import MonthlyReport, SpendingInsight, Transaction

logger = get_logger(__name__)


class InsightList(BaseModel):
    insights: list[SpendingInsight]


NARRATIVE_PROMPT = """You are a financial analyst writing a short monthly summary.
Given category totals and any anomaly flags already detected by the system,
write 2-4 SpendingInsight entries. Be specific and use the real numbers
given to you — never invent a number that isn't in the data below.
Keep each `detail` to one sentence."""


def aggregate_by_category(transactions: list[Transaction]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for t in transactions:
        if t.amount > 0:  # spends only, ignore credits/refunds
            totals[t.category.value] += t.amount
    return dict(totals)


def flag_anomalies(
    current_month: dict[str, float], history: list[dict[str, float]]
) -> list[SpendingInsight]:
    """Flag a category if this month's spend is more than 1.5x the historical average."""
    insights = []
    for category, amount in current_month.items():
        past_values = [h.get(category, 0) for h in history if h.get(category, 0) > 0]
        if not past_values:
            continue
        avg = statistics.mean(past_values)
        if avg > 0 and amount > 1.5 * avg:
            insights.append(
                SpendingInsight(
                    title=f"{category.title()} spending spike",
                    detail=f"Spent {amount:.2f} this month vs avg {avg:.2f}",
                    category=category,
                    severity="warning",
                )
            )
    return insights


def generate_narrative_insights(
    totals: dict[str, float], rule_based_flags: list[SpendingInsight]
) -> list[SpendingInsight]:
    llm = get_llm(task="analysis")
    schema_str = InsightList.model_json_schema()
    structured_llm = llm.with_structured_output(InsightList, method="json_mode")

    context = (
        f"Category totals this month: {totals}\n\n"
        f"Rule-based anomaly flags already detected: "
        f"{[f.model_dump() for f in rule_based_flags]}"
    )
    system_prompt = (
        f"{NARRATIVE_PROMPT}\n\n"
        f"You MUST respond with valid JSON matching EXACTLY this schema "
        f"(use these exact field names, no extras):\n{schema_str}"
    )
    result: InsightList = llm_call_with_retry(
        structured_llm.invoke,
        [("system", system_prompt), ("human", context)],
        task="analysis",
    )
    return result.insights


def generate_report(
    month: str,
    transactions: list[Transaction],
    history: list[dict[str, float]] | None = None,
    use_llm_narrative: bool = True,
) -> MonthlyReport:
    totals = aggregate_by_category(transactions)
    total_spend = sum(totals.values())
    rule_based_flags = flag_anomalies(totals, history or [])

    if use_llm_narrative:
        insights = generate_narrative_insights(totals, rule_based_flags)
    else:
        insights = rule_based_flags

    return MonthlyReport(
        month=month,
        total_spend=total_spend,
        total_by_category=totals,
        insights=insights,
    )


if __name__ == "__main__":
    from pathlib import Path
    from finsight.agents.extraction_agent import extract_from_csv

    sample_path = Path(__file__).parent.parent.parent.parent / "data" / "sample_statement.csv"
    result = extract_from_csv(sample_path)
    report = generate_report("2026-06", result.transactions)
    print(report.model_dump_json(indent=2))
