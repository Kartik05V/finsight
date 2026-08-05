"""
Phase 7 eval runner. Run with: python -m evals.run_evals

Simple pass/fail on whether expected substrings appear in the answer.
Swap for an LLM-as-judge scorer once your questions get more open-ended.
"""
import json
from pathlib import Path

from finsight.agents.rag_agent import answer_question
from finsight.agents.extraction_agent import extract_from_csv

QUESTIONS_PATH = Path(__file__).parent / "eval_questions.json"
SAMPLE_STATEMENT_PATH = Path(__file__).parent.parent / "data" / "sample_statement.csv"


def load_questions():
    return json.loads(QUESTIONS_PATH.read_text())


def run():
    questions = load_questions()
    result = extract_from_csv(SAMPLE_STATEMENT_PATH)
    transactions = result.transactions

    passed = 0
    scored = 0
    for q in questions:
        answer = answer_question(q["question"], transactions)

        if not q["expected_answer_contains"]:
            print(f"[REVIEW] {q['question']}\n  -> {answer}\n")
            continue

        scored += 1
        # Strip thousands-separator commas before matching — the LLM
        # correctly formats large numbers as "28,976.50", which broke a
        # literal substring match against "28976.5" even though the
        # answer was numerically correct. This isn't a workaround for a
        # wrong answer, it's fixing a real false-negative in the test.
        normalized_answer = answer.replace(",", "")
        ok = any(expected in normalized_answer for expected in q["expected_answer_contains"])
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        print(f"[{status}] {q['question']}\n  -> {answer}\n")

    print(f"Score: {passed}/{scored} ({100 * passed / scored:.0f}%)")


if __name__ == "__main__":
    run()
