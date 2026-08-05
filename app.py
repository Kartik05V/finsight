"""
Phase 8 — Streamlit front end.
Run with: streamlit run app.py
"""
import streamlit as st
import pandas as pd

from finsight.agents.extraction_agent import extract_from_csv
from finsight.agents.supervisor import build_graph
from finsight.models import FinSightState

st.set_page_config(page_title="FinSight", page_icon="💰")
st.title("FinSight — agentic finance analyst")

if "transactions" not in st.session_state:
    st.session_state.transactions = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded = st.file_uploader("Upload a statement CSV", type="csv")

if uploaded is not None and st.session_state.transactions is None:
    with st.spinner("Extracting transactions..."):
        tmp_path = "/tmp/uploaded_statement.csv"
        with open(tmp_path, "wb") as f:
            f.write(uploaded.getbuffer())
        result = extract_from_csv(tmp_path)
        st.session_state.transactions = result.transactions
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
                    st.write(result.get("answer"))
                    if result.get("report"):
                        st.json(result["report"].model_dump())
                    st.session_state.messages.append(
                        {"role": "assistant", "content": result.get("answer")}
                    )
                    st.session_state.chat_history = result.get(
                        "chat_history", st.session_state.chat_history
                    )
                except ValueError as e:
                    st.error(f"Blocked by guardrails: {e}")
else:
    st.info("Upload a CSV to get started — or use data/sample_statement.csv")
