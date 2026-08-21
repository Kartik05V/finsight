"""
Pydantic schemas — this is what forces the LLM to give you structured data
instead of prose.
"""
from datetime import date
from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field


class Category(str, Enum):
    FOOD = "food"
    GROCERIES = "groceries"
    TRANSPORT = "transport"
    RENT = "rent"
    UTILITIES = "utilities"
    ENTERTAINMENT = "entertainment"
    SHOPPING = "shopping"
    HEALTH = "health"
    TRAVEL = "travel"
    SUBSCRIPTIONS = "subscriptions"
    OTHER = "other"


class Transaction(BaseModel):
    """One line item from a bank/credit card statement."""
    date: date
    description: str = Field(..., description="Raw merchant/description text")
    amount: float = Field(..., description="Positive = spend, negative = refund/credit")
    category: Category = Category.OTHER
    merchant: str | None = Field(default=None, description="Cleaned merchant name")


class ExtractionResult(BaseModel):
    """What the extraction agent returns for one statement."""
    transactions: list[Transaction]
    statement_period_start: date | None = None
    statement_period_end: date | None = None
    warnings: list[str] = Field(default_factory=list, description="e.g. rows it couldn't parse")


class SpendingInsight(BaseModel):
    """One finding for the monthly report — keep these atomic."""
    title: str
    detail: str
    category: Category | None = None
    severity: str = Field(default="info", description="info | warning | alert")


class MonthlyReport(BaseModel):
    """What the analysis agent returns — Phase 6."""
    month: str  # e.g. "2026-06"
    total_spend: float
    total_by_category: dict[str, float]
    insights: list[SpendingInsight]


class FinSightState(BaseModel):
    """
    Shared state passed between LangGraph nodes (Phase 5).
    """
    user_query: str
    redacted_query: str | None = None
    transactions: list[Transaction] = Field(default_factory=list)
    answer: str | None = None
    report: MonthlyReport | None = None
    # Literal type keeps this in sync with RouteDecision.destination and
    # catches typos at type-check time instead of silently routing wrong.
    route: Literal["rag", "analyze"] | None = None
    chat_history: list[dict] = Field(
        default_factory=list,
        description="Running list of {'role': 'user'|'assistant', 'content': str} "
        "so follow-up questions like 'what about last month?' have context.",
    )
    # --- Self-correction loop fields (#3) -----------------------------------
    # attempts counts how many times rag_node has run for this query.
    attempts: int = 0
    # answer_score is set by grade_node: 1.0 = programmatic check passed,
    # 0.0 = number not found in answer, None = not yet graded.
    answer_score: float | None = None
    # low_confidence is True when all retry attempts exhausted without a
    # passing grade — signals the UI to show a "please double-check" banner.
    low_confidence: bool = False
    # best_answer preserves the highest-scoring answer across loop iterations
    # so if we exhaust retries the user still gets the best attempt, not
    # just the last one.
    best_answer: str | None = None
