"""
Mocked route_node and grade_node tests (#10).

No real API calls — the LLM is fully mocked via unittest.mock.
Tests confirm:
1. route_node correctly reads RouteDecision output and returns the route.
2. route_node falls back to 'rag' on any parse/API failure (not a crash).
3. grade_node correctly scores an answer that contains the expected number.
4. grade_node correctly scores 0 when the expected number is missing.
5. grade_condition correctly emits 'end', 'retry', or 'end_low_confidence'.
"""
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from finsight.models import Category, FinSightState, Transaction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_transactions(amounts: list[float]) -> list[Transaction]:
    """Build minimal Transaction objects with the given amounts."""
    return [
        Transaction(
            date=date(2026, 6, 1),
            description=f"Txn {i}",
            amount=a,
            category=Category.FOOD,
        )
        for i, a in enumerate(amounts)
    ]


# ---------------------------------------------------------------------------
# route_node tests
# ---------------------------------------------------------------------------

class TestRouteNode:
    def test_routes_to_rag_on_decision(self):
        """route_node should return {'route': 'rag'} when LLM says rag."""
        from finsight.agents.supervisor import RouteDecision

        mock_llm = MagicMock()
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = RouteDecision(
            destination="rag", reasoning="it's a factual question"
        )
        mock_llm.with_structured_output.return_value = mock_chain

        with patch("finsight.agents.supervisor.get_llm", return_value=mock_llm):
            with patch("finsight.agents.supervisor.llm_call_with_retry", side_effect=lambda fn, *a, **kw: fn(*a)):
                from finsight.agents.supervisor import route_node
                state = FinSightState(
                    user_query="How much on food?",
                    redacted_query="How much on food?",
                )
                result = route_node(state)

        assert result == {"route": "rag"}

    def test_routes_to_analyze_on_decision(self):
        """route_node should return {'route': 'analyze'} when LLM says analyze."""
        from finsight.agents.supervisor import RouteDecision

        mock_llm = MagicMock()
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = RouteDecision(
            destination="analyze", reasoning="monthly summary request"
        )
        mock_llm.with_structured_output.return_value = mock_chain

        with patch("finsight.agents.supervisor.get_llm", return_value=mock_llm):
            with patch("finsight.agents.supervisor.llm_call_with_retry", side_effect=lambda fn, *a, **kw: fn(*a)):
                from finsight.agents.supervisor import route_node
                state = FinSightState(
                    user_query="Give me a monthly report",
                    redacted_query="Give me a monthly report",
                )
                result = route_node(state)

        assert result == {"route": "analyze"}

    def test_fallback_to_rag_on_api_failure(self):
        """
        If the LLM call raises ANY exception, route_node must default to
        'rag' instead of crashing — this is the #2 fallback requirement.
        """
        mock_llm = MagicMock()
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = RuntimeError("API timeout")
        mock_llm.with_structured_output.return_value = mock_chain

        with patch("finsight.agents.supervisor.get_llm", return_value=mock_llm):
            # llm_call_with_retry should propagate the exception after retries
            with patch(
                "finsight.agents.supervisor.llm_call_with_retry",
                side_effect=RuntimeError("API timeout"),
            ):
                from finsight.agents.supervisor import route_node
                state = FinSightState(
                    user_query="test",
                    redacted_query="test",
                )
                result = route_node(state)

        assert result == {"route": "rag"}


# ---------------------------------------------------------------------------
# grade_node / grade_condition tests
# ---------------------------------------------------------------------------

class TestGradeNode:
    def _base_state(self, answer: str, attempts: int = 1) -> FinSightState:
        return FinSightState(
            user_query="How much on food?",
            redacted_query="How much on food?",
            transactions=_make_transactions([450.0, 380.0, 510.0]),  # total = 1340
            answer=answer,
            attempts=attempts,
        )

    def test_passes_when_answer_contains_correct_number(self):
        """Answer containing 1340 should get score 1.0."""
        from finsight.agents.supervisor import grade_node

        state = self._base_state("You spent ₹1340.00 on food delivery.")

        with patch("finsight.agents.supervisor.plan_retrieval") as mock_plan, \
             patch("finsight.agents.supervisor.filter_transactions", return_value=state.transactions):
            from finsight.agents.rag_agent import RetrievalPlan
            mock_plan.return_value = RetrievalPlan(
                needs_data=True, category_filter=Category.FOOD, reasoning="food question"
            )
            result = grade_node(state)

        assert result["answer_score"] == 1.0

    def test_fails_when_answer_lacks_correct_number(self):
        """Answer without the expected total should get score 0.0."""
        from finsight.agents.supervisor import grade_node

        state = self._base_state("I couldn't find any food transactions.")

        with patch("finsight.agents.supervisor.plan_retrieval") as mock_plan, \
             patch("finsight.agents.supervisor.filter_transactions", return_value=state.transactions):
            from finsight.agents.rag_agent import RetrievalPlan
            mock_plan.return_value = RetrievalPlan(
                needs_data=True, category_filter=Category.FOOD, reasoning="food question"
            )
            result = grade_node(state)

        assert result["answer_score"] == 0.0

    def test_within_tolerance_passes(self):
        """A number within 5% of 1340 (e.g. 1341) should pass."""
        from finsight.agents.supervisor import grade_node

        state = self._base_state("Your food spend was approximately ₹1341.")

        with patch("finsight.agents.supervisor.plan_retrieval") as mock_plan, \
             patch("finsight.agents.supervisor.filter_transactions", return_value=state.transactions):
            from finsight.agents.rag_agent import RetrievalPlan
            mock_plan.return_value = RetrievalPlan(
                needs_data=True, category_filter=Category.FOOD, reasoning="food"
            )
            result = grade_node(state)

        assert result["answer_score"] == 1.0


class TestGradeCondition:
    def test_end_when_score_passes(self):
        from finsight.agents.supervisor import grade_condition, GRADE_PASS_THRESHOLD
        state = FinSightState(
            user_query="q", answer_score=GRADE_PASS_THRESHOLD, attempts=1
        )
        assert grade_condition(state) == "end"

    def test_retry_when_score_fails_and_attempts_remaining(self):
        from finsight.agents.supervisor import grade_condition, MAX_ATTEMPTS
        state = FinSightState(
            user_query="q", answer_score=0.0, attempts=1
        )
        # attempts (1) < MAX_ATTEMPTS (3) → should retry
        assert grade_condition(state) == "retry"

    def test_end_low_confidence_when_attempts_exhausted(self):
        from finsight.agents.supervisor import grade_condition, MAX_ATTEMPTS
        state = FinSightState(
            user_query="q", answer_score=0.0, attempts=MAX_ATTEMPTS
        )
        assert grade_condition(state) == "end_low_confidence"
