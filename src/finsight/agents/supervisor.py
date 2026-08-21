"""
Supervisor agent — hardened.

Ties RAG and analysis together under one stateful LangGraph. Extraction is
treated as a one-time preprocessing step (done once when a statement is
uploaded, in main.py) rather than something re-run on every query.

Graph shape (with self-correction loop, #3):

    guardrails -> route -> [rag | analyze]
                                |
                           grade_node  (rag path only)
                                |
                   ┌────────────────────────────┐
                   │ score OK → END             │
                   │ attempts < MAX → rag       │ (regenerate)
                   │ attempts == MAX → END      │ (low_confidence=True)
                   └────────────────────────────┘
                   analyze → END directly (no numeric grading)

route_node (#2): on parse failure defaults to "rag" instead of crashing.
"""
import re
from datetime import date
from typing import Literal

from langgraph.graph import StateGraph, END
from pydantic import BaseModel

from finsight.config import get_llm, llm_call_with_retry
from finsight.guardrails import apply_guardrails
from finsight.logging_setup import get_logger
from finsight.models import FinSightState
from finsight.agents.rag_agent import answer_question, filter_transactions, plan_retrieval
from finsight.agents.analysis_agent import generate_report

logger = get_logger(__name__)

MAX_ATTEMPTS = 3
GRADE_PASS_THRESHOLD = 0.8   # score at or above this → accept answer
NUMERIC_TOLERANCE = 0.05     # 5% relative tolerance for number matching


class RouteDecision(BaseModel):
    destination: Literal["rag", "analyze"]
    reasoning: str


ROUTER_PROMPT = """Classify the user's finance question into exactly one route:
- "rag": a specific factual question about their transactions
  (e.g. "how much did I spend on X", "what was my biggest purchase").
- "analyze": a request for an overall summary, report, or health check
  (e.g. "how am I doing this month", "give me my monthly report",
  "any spending anomalies?").
"""


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def guardrail_node(state: FinSightState) -> dict:
    safe_text, metadata = apply_guardrails(state.user_query)
    logger.info(f"guardrails passed pii={metadata['pii_types_found']} injection_suspected={metadata['injection_suspected']}")
    return {"redacted_query": safe_text}


def route_node(state: FinSightState) -> dict:
    """
    Routes to 'rag' or 'analyze'. (#2) On any failure — parse error,
    API timeout, unexpected schema — defaults to 'rag' instead of crashing.
    """
    try:
        llm = get_llm(task="routing")
        # Use json_mode for portability across Groq's model lineup — mirrors
        # the same fix applied in rag_agent.plan_retrieval.
        schema_str = RouteDecision.model_json_schema()
        system_prompt = (
            f"{ROUTER_PROMPT}\n\n"
            f"You MUST respond with valid JSON matching EXACTLY this schema "
            f"(use these exact field names, no extras):\n{schema_str}"
        )
        router = llm.with_structured_output(RouteDecision, method="json_mode")
        decision: RouteDecision = llm_call_with_retry(
            router.invoke,
            [("system", system_prompt), ("human", state.redacted_query)],
            task="routing",
        )
        route = decision.destination
        logger.info(f"routed to={route!r} reasoning={decision.reasoning!r}")
    except Exception as exc:
        logger.warning(
            f"route_node failed ({type(exc).__name__}: {exc}), defaulting to 'rag'"
        )
        route = "rag"
    return {"route": route}


def route_condition(state: FinSightState) -> str:
    return state.route


def rag_node(state: FinSightState) -> dict:
    attempt_num = state.attempts + 1
    logger.info(f"rag_node attempt={attempt_num}/{MAX_ATTEMPTS}")
    answer = answer_question(
        state.redacted_query, state.transactions, chat_history=state.chat_history
    )
    updated_history = state.chat_history + [
        {"role": "user", "content": state.user_query},
        {"role": "assistant", "content": answer},
    ]
    return {
        "answer": answer,
        "chat_history": updated_history,
        "attempts": attempt_num,
    }


def analyze_node(state: FinSightState) -> dict:
    if state.transactions:
        months = [t.date.strftime("%Y-%m") for t in state.transactions]
        month = max(set(months), key=months.count)
    else:
        month = date.today().strftime("%Y-%m")
    report = generate_report(month, state.transactions)
    answer = f"Monthly report generated for {month}."
    updated_history = state.chat_history + [
        {"role": "user", "content": state.user_query},
        {"role": "assistant", "content": answer},
    ]
    return {"report": report, "answer": answer, "chat_history": updated_history}


# ---------------------------------------------------------------------------
# Programmatic grader (#3)
# ---------------------------------------------------------------------------

def _extract_numbers_from_text(text: str) -> list[float]:
    """Pull out every number-looking token from an answer string."""
    # Matches integers or decimals, with optional thousands separators
    raw = re.findall(r"\b[\d,]+(?:\.\d+)?\b", text)
    results = []
    for r in raw:
        try:
            results.append(float(r.replace(",", "")))
        except ValueError:
            pass
    return results


def _numbers_match(expected: float, candidates: list[float], tol: float = NUMERIC_TOLERANCE) -> bool:
    """Return True if any candidate is within `tol` relative error of expected."""
    if expected == 0:
        return any(abs(c) < 0.01 for c in candidates)
    return any(abs(c - expected) / abs(expected) <= tol for c in candidates)


def grade_node(state: FinSightState) -> dict:
    """
    Programmatic answer grader — no LLM call needed.

    1. Re-runs the same filter logic used by rag_node to get the relevant
       transactions for this query.
    2. Computes the expected total in pure Python (the ground truth).
    3. Checks whether the LLM's answer contains that number (±5% tolerance).
    4. Updates answer_score, best_answer, and optionally low_confidence.
    """
    answer = state.answer or ""

    # Re-derive the retrieval plan to know what the expected number is.
    # We already computed this inside rag_node but don't persist it to state;
    # re-running it is cheap (no LLM, no I/O) and keeps state lean.
    try:
        plan = plan_retrieval(state.redacted_query, chat_history=state.chat_history[:-2] or None)
        relevant = filter_transactions(state.transactions, plan)
    except Exception as exc:
        logger.warning(f"grade_node: plan_retrieval failed ({exc}), skipping grade")
        return {"answer_score": None, "best_answer": answer}

    spends = [t for t in relevant if t.amount > 0]
    expected_total = sum(t.amount for t in spends)

    candidates = _extract_numbers_from_text(answer)
    passed = _numbers_match(expected_total, candidates) if expected_total > 0 else True

    score = 1.0 if passed else 0.0
    logger.info(
        f"grade_node attempts={state.attempts} expected={expected_total:.2f} "
        f"candidates={candidates} score={score}"
    )

    # Track the best answer seen so far across retries
    prev_score = state.answer_score if state.answer_score is not None else -1.0
    best_answer = answer if score >= prev_score else (state.best_answer or answer)

    return {
        "answer_score": score,
        "best_answer": best_answer,
    }


def grade_condition(state: FinSightState) -> str:
    """
    Conditional edge after grade_node:
    - score >= threshold → END (accept answer)
    - attempts < MAX and score low → loop back to rag (regenerate)
    - attempts == MAX → END with low_confidence=True (best effort)
    """
    score = state.answer_score if state.answer_score is not None else 0.0

    if score >= GRADE_PASS_THRESHOLD:
        logger.info(f"grade passed score={score:.2f}, routing to END")
        return "end"

    if state.attempts < MAX_ATTEMPTS:
        logger.info(
            f"grade failed score={score:.2f} attempts={state.attempts}/{MAX_ATTEMPTS}, "
            f"retrying rag"
        )
        return "retry"

    # Exhausted retries — mark low confidence so the UI can warn the user
    logger.warning(
        f"grade exhausted all {MAX_ATTEMPTS} attempts score={score:.2f}, "
        f"returning best_answer with low_confidence=True"
    )
    return "end_low_confidence"


def _finalize_low_confidence(state: FinSightState) -> dict:
    """Pseudo-node: sets low_confidence flag and surfaces best_answer."""
    return {
        "low_confidence": True,
        "answer": state.best_answer or state.answer,
    }


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(FinSightState)

    graph.add_node("guardrails", guardrail_node)
    graph.add_node("route", route_node)
    graph.add_node("rag", rag_node)
    graph.add_node("analyze", analyze_node)
    graph.add_node("grade", grade_node)
    graph.add_node("finalize_low_confidence", _finalize_low_confidence)

    graph.set_entry_point("guardrails")
    graph.add_edge("guardrails", "route")
    graph.add_conditional_edges(
        "route", route_condition, {"rag": "rag", "analyze": "analyze"}
    )
    graph.add_edge("rag", "grade")
    graph.add_conditional_edges(
        "grade",
        grade_condition,
        {
            "end": END,
            "retry": "rag",
            "end_low_confidence": "finalize_low_confidence",
        },
    )
    graph.add_edge("finalize_low_confidence", END)
    graph.add_edge("analyze", END)

    return graph.compile()


if __name__ == "__main__":
    # Phase 5 checkpoint: run `python -m finsight.agents.supervisor` and
    # confirm the first query routes to rag (a real number in the answer)
    # and the second routes to analyze (a MonthlyReport comes back).
    from pathlib import Path
    from finsight.agents.extraction_agent import extract_from_csv

    sample_path = Path(__file__).parent.parent.parent.parent / "data" / "sample_statement.csv"
    extraction = extract_from_csv(sample_path)

    app = build_graph()

    print("--- RAG route ---")
    result = app.invoke(
        FinSightState(
            user_query="How much did I spend on food delivery?",
            transactions=extraction.transactions,
        )
    )
    print(result["answer"])
    print(f"low_confidence={result.get('low_confidence', False)}")

    print("\n--- Analyze route ---")
    result = app.invoke(
        FinSightState(
            user_query="How am I doing this month? Any spending anomalies?",
            transactions=extraction.transactions,
        )
    )
    print(result["answer"])
    if result.get("report"):
        print(result["report"].model_dump_json(indent=2))
