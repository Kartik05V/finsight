"""
Phase 8 — Streamlit front end.
Run with: streamlit run app.py

Changes:
- Persistence (#7): transactions and chat history are saved to SQLite
  (via persistence.py) so state survives page reloads within a session.
- Low-confidence banner (#3): if the self-correction loop exhausted all
  retries without passing the numeric grade, shows a warning banner
  so the user knows to double-check the number.
"""
import uuid

import streamlit as st
import pandas as pd

from finsight.agents.extraction_agent import extract_from_csv
from finsight.agents.supervisor import build_graph
from finsight.models import FinSightState
from finsight.persistence import (
    load_transactions,
    save_transactions,
    load_chat_history,
    save_message,
)

st.set_page_config(page_title="FinSight", page_icon="💰")
st.title("FinSight — agentic finance analyst")

# Each browser session gets a stable UUID stored in session_state.
# This is the key that persistence.py uses to scope all DB reads/writes.
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

session_id = st.session_state.session_id

# --- Restore transactions from DB on first load ---------------------------
if "transactions" not in st.session_state:
    restored = load_transactions(session_id)
    st.session_state.transactions = restored if restored else None

if "messages" not in st.session_state:
    st.session_state.messages = []

# Restore chat history from DB (used to hydrate the graph state)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history(session_id)

# --- Upload ------------------------------------------------------------------
uploaded = st.file_uploader("Upload a statement CSV", type="csv")

if uploaded is not None and st.session_state.transactions is None:
    with st.spinner("Extracting transactions..."):
        import tempfile, os
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name
        result = extract_from_csv(tmp_path)
        os.unlink(tmp_path)
        st.session_state.transactions = result.transactions

        # Persist for this session so a page reload doesn't wipe everything
        save_transactions(session_id, result.transactions)

        if result.warnings:
            st.warning(
                f"⚠️ Extraction warnings ({len(result.warnings)}):\n"
                + "\n".join(f"- {w}" for w in result.warnings)
            )

    st.success(f"Extracted {len(st.session_state.transactions)} transactions.")

if st.session_state.transactions:
    df = pd.DataFrame([t.model_dump() for t in st.session_state.transactions])
    with st.expander("View extracted transactions"):
        st.dataframe(df)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    query = st.chat_input("Ask about your spending, or ask for a monthly report")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                app = build_graph()
                try:
                    result = app.invoke(
                        FinSightState(
                            user_query=query,
                            transactions=st.session_state.transactions,
                            chat_history=st.session_state.chat_history,
                        )
                    )
                    answer = result.get("answer") or ""
                    low_confidence = result.get("low_confidence", False)

                    st.write(answer)

                    # Low-confidence banner — shown when the self-correction
                    # loop exhausted all retry attempts without passing the
                    # programmatic numeric grade (#3).
                    if low_confidence:
                        st.warning(
                            "⚠️ **Low confidence**: this answer didn't pass the "
                            "numeric accuracy check after all retry attempts. "
                            "Please double-check the figure against your statement."
                        )

                    if result.get("report"):
                        st.json(result["report"].model_dump())

                    # Persist the new messages to DB
                    save_message(session_id, "user", query)
                    save_message(session_id, "assistant", answer)

                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                    st.session_state.chat_history = result.get(
                        "chat_history", st.session_state.chat_history
                    )
                except ValueError as e:
                    st.error(f"Blocked by guardrails: {e}")
else:
    st.info("Upload a CSV to get started — or use data/sample_statement.csv")
