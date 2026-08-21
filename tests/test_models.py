"""
Pydantic model edge-case tests (#10).

Zero API calls — purely validates that schema enforcement works correctly.
"""
import pytest
from datetime import date
from pydantic import ValidationError

from finsight.models import (
    Category,
    ExtractionResult,
    FinSightState,
    Transaction,
)


class TestFinSightStateDefaults:
    def test_default_attempts_is_zero(self):
        state = FinSightState(user_query="test")
        assert state.attempts == 0

    def test_default_low_confidence_is_false(self):
        state = FinSightState(user_query="test")
        assert state.low_confidence is False

    def test_default_answer_score_is_none(self):
        state = FinSightState(user_query="test")
        assert state.answer_score is None

    def test_default_best_answer_is_none(self):
        state = FinSightState(user_query="test")
        assert state.best_answer is None

    def test_default_route_is_none(self):
        state = FinSightState(user_query="test")
        assert state.route is None

    def test_route_literal_accepts_rag(self):
        state = FinSightState(user_query="test", route="rag")
        assert state.route == "rag"

    def test_route_literal_accepts_analyze(self):
        state = FinSightState(user_query="test", route="analyze")
        assert state.route == "analyze"

    def test_route_literal_rejects_invalid(self):
        """route='bad' should raise a ValidationError (Literal enforcement)."""
        with pytest.raises(ValidationError):
            FinSightState(user_query="test", route="bad")  # type: ignore[arg-type]

    def test_chat_history_defaults_to_empty_list(self):
        state = FinSightState(user_query="test")
        assert state.chat_history == []

    def test_transactions_defaults_to_empty_list(self):
        state = FinSightState(user_query="test")
        assert state.transactions == []


class TestTransaction:
    def test_positive_amount_is_spend(self):
        t = Transaction(
            date=date(2026, 6, 1),
            description="Swiggy order",
            amount=450.0,
            category=Category.FOOD,
        )
        assert t.amount > 0

    def test_negative_amount_is_credit(self):
        """Negative amounts (salary credits, refunds) must be accepted."""
        t = Transaction(
            date=date(2026, 6, 1),
            description="Salary credit",
            amount=-55000.0,
            category=Category.OTHER,
        )
        assert t.amount < 0

    def test_default_category_is_other(self):
        t = Transaction(
            date=date(2026, 6, 1),
            description="Unknown merchant",
            amount=100.0,
        )
        assert t.category == Category.OTHER

    def test_merchant_is_optional(self):
        t = Transaction(
            date=date(2026, 6, 1),
            description="Some desc",
            amount=50.0,
        )
        assert t.merchant is None


class TestExtractionResult:
    def test_warnings_default_empty(self):
        result = ExtractionResult(transactions=[])
        assert result.warnings == []

    def test_statement_period_optional(self):
        result = ExtractionResult(transactions=[])
        assert result.statement_period_start is None
        assert result.statement_period_end is None
