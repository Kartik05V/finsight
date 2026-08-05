"""
CLI entry point. Run with: python -m finsight.main [path/to/statement.csv]
Defaults to data/sample_statement.csv if no path is given.
"""
import sys
from pathlib import Path

from finsight.agents.extraction_agent import extract_from_csv
from finsight.agents.supervisor import build_graph
from finsight.models import FinSightState

DEFAULT_STATEMENT = Path(__file__).parent.parent.parent / "data" / "sample_statement.csv"


def load_transactions(csv_path: str | Path):
    print(f"Extracting transactions from {csv_path} ...")
    result = extract_from_csv(csv_path)
    if result.warnings:
        print(f"({len(result.warnings)} rows had warnings — see result.warnings)")
    print(f"Loaded {len(result.transactions)} transactions.\n")
    return result.transactions


def run(query: str, transactions, chat_history=None):
    app = build_graph()
    state = FinSightState(
        user_query=query, transactions=transactions, chat_history=chat_history or []
    )
    result = app.invoke(state)
    return result


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_STATEMENT
    transactions = load_transactions(csv_path)
    chat_history: list[dict] = []

    print("FinSight — ask about your finances (try a factual question, then")
    print('"how am I doing this month?"). Follow-up questions remember')
    print("earlier context in this session. Ctrl+C to quit.\n")

    while True:
        try:
            query = input("> ")
        except (KeyboardInterrupt, EOFError):
            print("\nbye")
            break
        if not query.strip():
            continue

        try:
            result = run(query, transactions, chat_history=chat_history)
        except ValueError as e:
            print(f"[blocked] {e}")
            continue

        print(result.get("answer"))
        if result.get("report"):
            print(result["report"].model_dump_json(indent=2))
        print()

        chat_history = result.get("chat_history", chat_history)


if __name__ == "__main__":
    main()
