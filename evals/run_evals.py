"""
Richer eval runner (#5).

Changes from the original:
- Scoring broken into three categories: extraction accuracy, RAG correctness,
  analysis quality (open-ended review).
- Per-question latency tracked with time.perf_counter(); aggregate mean and
  p95 reported at the end.
- Adversarial cases handled: injection attempts, ambiguous queries, malformed
  references — each has its own expected outcome type so the runner knows
  whether to check for a block, a graceful answer, or a substring.

Run with: python -m evals.run_evals
"""
import json
import statistics
import time
from pathlib import Path

from finsight.agents.extraction_agent import extract_from_csv
from finsight.agents.rag_agent import answer_question
from finsight.guardrails import apply_guardrails

QUESTIONS_PATH = Path(__file__).parent / "eval_questions.json"
SAMPLE_STATEMENT_PATH = Path(__file__).parent.parent / "data" / "sample_statement.csv"


def load_questions():
    return json.loads(QUESTIONS_PATH.read_text())


def _run_rag_question(q: dict, transactions) -> tuple[str, float]:
    """Returns (answer, latency_seconds)."""
    t0 = time.perf_counter()
    answer = answer_question(q["question"], transactions)
    latency = time.perf_counter() - t0
    return answer, latency


def _check_rag(answer: str, q: dict) -> bool:
    """Substring match with comma-stripping (handles formatted numbers)."""
    if not q.get("expected_answer_contains"):
        return True  # open-ended — skip auto-check
    normalized = answer.replace(",", "")
    return any(exp in normalized for exp in q["expected_answer_contains"])


def run():
    questions = load_questions()

    # --- Extraction accuracy (programmatic, no LLM) ------------------------
    print("\n" + "=" * 60)
    print("CATEGORY 1: Extraction accuracy")
    print("=" * 60)
    result = extract_from_csv(SAMPLE_STATEMENT_PATH)
    transactions = result.transactions
    ext_pass = len(transactions) > 0
    print(f"  Transactions extracted : {len(transactions)}")
    print(f"  Extraction warnings    : {len(result.warnings)}")
    for w in result.warnings:
        print(f"    [WARN] {w}")
    total_spend = sum(t.amount for t in transactions if t.amount > 0)
    print(f"  Total spend (Python)   : Rs.{total_spend:,.2f}")
    expected_total = 28976.5  # from sample_statement.csv ground truth
    ext_match = abs(total_spend - expected_total) / expected_total < 0.01
    print(f"  Total matches expected : {'[PASS]' if ext_match else '[FAIL]'} "
          f"(expected Rs.{expected_total:,.2f})")

    # --- RAG correctness ---------------------------------------------------
    print("\n" + "=" * 60)
    print("CATEGORY 2: RAG correctness")
    print("=" * 60)

    rag_questions = [q for q in questions if q.get("category", "rag") == "rag"
                     or "category" not in q and q.get("expected_answer_contains")]
    adversarial_questions = [q for q in questions if q.get("category") == "adversarial"]
    review_questions = [q for q in questions if not q.get("expected_answer_contains")
                        and q.get("category") != "adversarial"]

    rag_passed = 0
    rag_latencies: list[float] = []

    for q in rag_questions:
        answer, latency = _run_rag_question(q, transactions)
        rag_latencies.append(latency)
        ok = _check_rag(answer, q)
        if ok:
            rag_passed += 1
        status = "[PASS]" if ok else "[FAIL]"
        print(f"  [{status}] ({latency:.2f}s) {q['question']}")
        print(f"         >> {answer[:120].strip()}")

    rag_score = rag_passed / len(rag_questions) if rag_questions else 0
    print(f"\n  RAG score: {rag_passed}/{len(rag_questions)} ({rag_score * 100:.0f}%)")

    # Latency stats
    if rag_latencies:
        mean_lat = statistics.mean(rag_latencies)
        p95_lat = sorted(rag_latencies)[int(len(rag_latencies) * 0.95)]
        print(f"  Latency  : mean={mean_lat:.2f}s  p95={p95_lat:.2f}s")

    # --- Adversarial cases -------------------------------------------------
    print("\n" + "=" * 60)
    print("CATEGORY 3: Adversarial / edge cases")
    print("=" * 60)

    adv_passed = 0
    for q in adversarial_questions:
        expected_behavior = q.get("expected_behavior", "answer")
        t0 = time.perf_counter()

        if expected_behavior == "block":
            # Should raise a guardrails ValueError
            try:
                apply_guardrails(q["question"])
                ok = False
                answer = "[NOT BLOCKED - expected block]"
            except ValueError as e:
                ok = True
                answer = f"[BLOCKED: {e}]"
        elif expected_behavior == "soft_flag":
            # Should return metadata with injection_suspected=True but not block
            try:
                _, meta = apply_guardrails(q["question"])
                ok = meta.get("injection_suspected", False)
                answer = f"[FLAGGED={ok}]"
            except ValueError:
                ok = False
                answer = "[HARD BLOCKED - expected only soft flag]"
        else:
            answer, _ = _run_rag_question(q, transactions)
            ok = _check_rag(answer, q)

        latency = time.perf_counter() - t0
        if ok:
            adv_passed += 1
        status = "[PASS]" if ok else "[FAIL]"
        print(f"  [{status}] ({latency:.2f}s) {q['question']}")
        print(f"         >> {str(answer)[:120].strip()}")

    adv_score = adv_passed / len(adversarial_questions) if adversarial_questions else 0
    if adversarial_questions:
        print(f"\n  Adversarial score: {adv_passed}/{len(adversarial_questions)} ({adv_score * 100:.0f}%)")

    # --- Analysis quality (manual review) ----------------------------------
    if review_questions:
        print("\n" + "=" * 60)
        print("CATEGORY 4: Analysis quality (manual review)")
        print("=" * 60)
        for q in review_questions:
            answer, latency = _run_rag_question(q, transactions)
            print(f"  [REVIEW] ({latency:.2f}s) {q['question']}")
            print(f"         >> {answer[:200].strip()}")
            if q.get("note"):
                print(f"         note: {q['note']}")

    # --- Summary -----------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Extraction accuracy : {'[PASS]' if ext_match else '[FAIL]'}")
    print(f"  RAG correctness     : {rag_passed}/{len(rag_questions)} ({rag_score * 100:.0f}%)")
    if adversarial_questions:
        print(f"  Adversarial         : {adv_passed}/{len(adversarial_questions)} ({adv_score * 100:.0f}%)")
    print(f"  Analysis quality    : {len(review_questions)} question(s) for manual review above")


if __name__ == "__main__":
    run()
