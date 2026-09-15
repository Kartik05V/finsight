"""Pydantic schemas — forces the LLM to return structured data instead of prose."""
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
    """What the analysis agent returns."""
    month: str  # e.g. "2026-06"
    total_spend: float
    total_by_category: dict[str, float]
    insights: list[SpendingInsight]


class FinSightState(BaseModel):
    """Shared state passed between LangGraph nodes."""
    user_query: str
    redacted_query: str | None = None
    transactions: list[Transaction] = Field(default_factory=list)
    answer: str | None = None
    report: MonthlyReport | None = None
    route: Literal["rag", "analyze"] | None = None
    chat_history: list[dict] = Field(
        default_factory=list,
        description="Running list of {'role': 'user'|'assistant', 'content': str}",
    )
    attempts: int = 0
    answer_score: float | None = None
    low_confidence: bool = False  # True when all retry attempts exhausted without passing grade
    best_answer: str | None = None  # highest-scoring answer across retry iterations
