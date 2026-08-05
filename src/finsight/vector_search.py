"""
Optional extension — Phase 4 upgrade: real vector-store RAG.

Your main pipeline (rag_agent.py) uses "vectorless RAG" — filtering
transactions by category/month with plain Python. This module adds real
embedding-based semantic search on top, for questions that don't map
cleanly to a fixed category.

Uses local HuggingFace embeddings (all-MiniLM-L6-v2) — runs on your own
machine, no additional API key needed. Only cost is a one-time ~90MB
model download the first time you run it.

Not wired into the main supervisor graph by default, so trying it can't
break the working pipeline. To use:

    uv pip install -e ".[vectorstore]"
    python -m finsight.vector_search
"""
from pathlib import Path

from finsight.models import Transaction


def build_vector_store(transactions: list[Transaction]):
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    docs = [
        Document(
            page_content=f"{t.description} ({t.merchant or 'unknown merchant'}) "
            f"- {t.category.value} - amount {t.amount} on {t.date}",
            metadata={
                "date": str(t.date),
                "amount": t.amount,
                "category": t.category.value,
                "merchant": t.merchant or "",
            },
        )
        for t in transactions
    ]

    store = Chroma.from_documents(docs, embedding=embeddings)
    return store


def semantic_search(store, query: str, k: int = 5) -> list[dict]:
    results = store.similarity_search(query, k=k)
    return [{"content": r.page_content, **r.metadata} for r in results]


if __name__ == "__main__":
    from finsight.agents.extraction_agent import extract_from_csv

    sample_path = Path(__file__).parent.parent.parent / "data" / "sample_statement.csv"

    print("Extracting transactions (needs your LLM API key)...")
    result = extract_from_csv(sample_path)

    print("Building local vector store (downloads embedding model on first run)...")
    store = build_vector_store(result.transactions)

    print("\n--- Semantic search demo ---")
    for query in ["coffee or cafe purchases", "food delivery apps", "monthly bills"]:
        print(f"\nQuery: {query!r}")
        for match in semantic_search(store, query, k=3):
            print(f"  {match['content']}")
