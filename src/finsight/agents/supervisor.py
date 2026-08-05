"""
Supervisor agent — Phase 5.

Ties RAG and analysis together under one stateful LangGraph. Extraction is
treated as a one-time preprocessing step (done once when a statement is
uploaded, in main.py) rather than something re-run on every query.

Graph shape:

    guardrails -> route -> [rag | analyze] -> END

route is an LLM classifier that reads the query and decides which
specialist should handle it. Uses the gateway's "routing" task model
(fast) — classifying intent doesn't need the strongest model.
"""
from datetime import date
from typing import Literal

from langgraph.graph import StateGraph, END
from pydantic import BaseModel

from finsight.config import get_llm
from finsight.guardrails import apply_guardrails
from finsight.models import FinSightState
from finsight.agents.rag_agent import answer_question
from finsight.agents.analysis_agent import generate_report


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


def guardrail_node(state: FinSightState) -> dict:
    safe_text, _metadata = apply_guardrails(state.user_query)
    return {"redacted_query": safe_text}


def route_node(state: FinSightState) -> dict:
    llm = get_llm(task="routing")
    router = llm.with_structured_output(RouteDecision)
    decision: RouteDecision = router.invoke(
        [("system", ROUTER_PROMPT), ("human", state.redacted_query)]
    )
    return {"route": decision.destination}


def route_condition(state: FinSightState) -> str:
    return state.route


def rag_node(state: FinSightState) -> dict:
    answer = answer_question(
        state.redacted_query, state.transactions, chat_history=state.chat_history
    )
    updated_history = state.chat_history + [
        {"role": "user", "content": state.user_query},
        {"role": "assistant", "content": answer},
    ]
    return {"answer": answer, "chat_history": updated_history}


def analyze_node(state: FinSightState) -> dict:
    if state.transactions:
        # Use the most common year-month across the actual transaction
        # dates, not today's date — a statement from June shouldn't be
        # labeled with whatever month it happens to be run in.
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


def build_graph():
    graph = StateGraph(FinSightState)

    graph.add_node("guardrails", guardrail_node)
    graph.add_node("route", route_node)
    graph.add_node("rag", rag_node)
    graph.add_node("analyze", analyze_node)

    graph.set_entry_point("guardrails")
    graph.add_edge("guardrails", "route")
    graph.add_conditional_edges(
        "route", route_condition, {"rag": "rag", "analyze": "analyze"}
    )
    graph.add_edge("rag", END)
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
