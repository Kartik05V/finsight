"""
RAG agent: plans what transaction data to retrieve, filters it in Python,
pre-computes stats, then hands clean context to the LLM to answer.
"""
import re
from datetime import datetime

from pydantic import BaseModel

from finsight.config import get_llm, llm_call_with_retry
from finsight.logging_setup import get_logger
from finsight.models import Category, Transaction

logger = get_logger(__name__)


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models like Qwen."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


class RetrievalPlan(BaseModel):
    needs_data: bool
    category_filter: Category | None = None
    # LLM may return natural-language dates — always run through normalize_month_filter()
    month_filter: str | None = None
    reasoning: str


PLANNER_PROMPT = """/no_think
Given a user's question about their finances, decide whether
you need to look up their transaction data, and if so, what filters to apply.

category_filter must be one of exactly these values (or null): food, groceries,
transport, rent, utilities, entertainment, shopping, health, travel,
subscriptions, other. Do not invent a category not in this list — pick the
closest match, or leave it null if the question isn't about one category.

month_filter must be null unless the user explicitly names a month or says
"this month" / "last month". When you DO set it, it must be formatted
EXACTLY as "YYYY-MM" (e.g. "2026-06") — never a natural language date like
"June 2026" or "06/2026".

If the question can be answered without data (e.g. "what is a good savings
rate?"), set needs_data to false.

If the question is ambiguous or lacks context (e.g. "what about those?"), set needs_data to false.
"""

_MONTH_FORMATS = ["%Y-%m", "%B %Y", "%b %Y", "%m/%Y"]


def normalize_month_filter(month_filter: str | None) -> str | None:
    """Normalize any date format the LLM returns into YYYY-MM."""
    if not month_filter:
        return None
    month_filter = month_filter.strip()
    if re.match(r"^\d{4}-\d{2}$", month_filter):
        return month_filter
    for fmt in _MONTH_FORMATS:
        try:
            return datetime.strptime(month_filter, fmt).strftime("%Y-%m")
        except ValueError:
            continue
    logger.warning(f"couldn't parse month_filter {month_filter!r}, ignoring it")
    return None


def plan_retrieval(question: str, chat_history: list[dict] | None = None) -> RetrievalPlan:
    # Strong model for retrieval planning — category ambiguity needs real judgment
    llm = get_llm(task="rag")
    schema_str = RetrievalPlan.model_json_schema()
    planner = llm.with_structured_output(RetrievalPlan, method="json_mode")

    history_text = ""
    if chat_history:
        recent = chat_history[-6:]
        history_text = "\n".join(
            f"{turn['role']}: {turn['content']}" for turn in recent
        )
        history_text = (
            f"\n\nRecent conversation (use this to resolve references like "
            f"'those transactions', 'that category', or 'instead' — a "
            f"follow-up question is asking about the SAME category/month "
            f"as the previous turn unless it explicitly says otherwise):\n"
            f"{history_text}\n"
        )

    system_prompt = (
        f"{PLANNER_PROMPT}\n\n"
        f"You MUST respond with valid JSON matching EXACTLY this schema "
        f"(use these exact field names, no extras):\n{schema_str}"
    )
    plan = llm_call_with_retry(
        planner.invoke,
        [("system", system_prompt), ("human", f"{question}{history_text}")],
        task="rag_planner"
    )
    logger.info(
        f"plan: category={plan.category_filter} month={plan.month_filter} "
        f"reasoning={plan.reasoning!r}"
    )
    return plan


def filter_transactions(
    transactions: list[Transaction], plan: RetrievalPlan
) -> list[Transaction]:
    """Apply category and month filters independently; fall back to full list only if nothing matched."""
    filtered = transactions

    if plan.category_filter:
        by_category = [t for t in filtered if t.category == plan.category_filter]
        if by_category:
            filtered = by_category

    normalized_month = normalize_month_filter(plan.month_filter)
    if normalized_month:
        by_month = [t for t in filtered if t.date.strftime("%Y-%m") == normalized_month]
        if by_month:
            filtered = by_month

    if not filtered and transactions:
        filtered = transactions
    return filtered


def _precompute_stats(relevant: list[Transaction]) -> str:
    """
    Pre-compute totals in Python instead of asking the LLM to sum numbers.
    LLMs are unreliable at multi-term arithmetic — verified directly in testing.
    """
    spends = [t for t in relevant if t.amount > 0]
    credits = [t for t in relevant if t.amount < 0]

    total_spend = sum(t.amount for t in spends)
    total_credits = sum(t.amount for t in credits)
    count = len(relevant)

    lines = [
        f"Pre-computed total spend (sum of positive amounts): {total_spend:.2f}",
        f"Pre-computed total credits/refunds (sum of negative amounts): {total_credits:.2f}",
        f"Pre-computed transaction count: {count}",
    ]
    if spends:
        biggest = max(spends, key=lambda t: t.amount)
        lines.append(
            f"Pre-computed single biggest expense: {biggest.amount:.2f} "
            f"({biggest.description} on {biggest.date})"
        )
    return "\n".join(lines)


def answer_question(
    question: str, transactions: list[Transaction], chat_history: list[dict] | None = None
) -> str:
    plan = plan_retrieval(question, chat_history=chat_history)
    if not plan.needs_data:
        llm = get_llm(task="rag")
        raw = llm.invoke(question).content
        return _strip_think_tags(raw)

    relevant = filter_transactions(transactions, plan)
    context = "\n".join(
        f"{t.date} | {t.description} | {t.amount}" for t in relevant
    )
    stats = _precompute_stats(relevant)
    logger.info(f"filtered context ({len(relevant)} rows): {context[:200]}...")

    history_text = ""
    if chat_history:
        recent = chat_history[-6:]
        history_text = "\n".join(
            f"{turn['role']}: {turn['content']}" for turn in recent
        )
        history_text = f"\nRecent conversation for context:\n{history_text}\n"

    llm = get_llm(task="rag")
    prompt = (
        "/no_think\n"
        f"Raw transaction data (this list is COMPLETE and EXCLUSIVE for the "
        f"filter applied -- do not add, assume, or reason about any other "
        f"transactions, even ones you think might also fit the category):\n"
        f"{context}\n\n"
        f"{stats}\n"
        f"{history_text}\n"
        f"Question: {question}\n"
        "IMPORTANT: use the pre-computed statistics above directly for any "
        "totals, counts, or 'biggest expense' questions -- do NOT manually "
        "add up the raw transaction amounts yourself, that's how arithmetic "
        "errors happen. Do NOT expand the category definition to include "
        "transactions not shown above, even if they seem related. Use the "
        "raw transaction list only for details like merchant names or "
        "individual amounts. Answer in one clear, concise paragraph -- do "
        "not repeat yourself or second-guess your own answer. If the "
        "question refers to something from earlier in the conversation "
        "(e.g. 'what about transport instead'), use the recent "
        "conversation to resolve what it's asking."
    )
    raw = llm_call_with_retry(llm.invoke, prompt, task="rag").content
    return _strip_think_tags(raw)
