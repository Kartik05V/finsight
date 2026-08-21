import os
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from pydantic import BaseModel
from typing import Optional

class RetrievalPlan(BaseModel):
    needs_data: bool
    category_filter: Optional[str] = None
    month_filter: Optional[str] = None
    reasoning: str

PLANNER_PROMPT = """/no_think
Given a user's question about their finances, decide whether
you need to look up their transaction data, and if so, what filters to apply.

category_filter must be one of exactly these values (or null): food, groceries,
transport, rent, utilities, entertainment, shopping, health, travel,
subscriptions, other. Do not invent a category not in this list — pick the
closest match, or leave it null if the question isn't about one category.

month_filter must be null unless the user explicitly names a month or says
"this month" / "last month". When you DO set it, it must be formatted
EXACTLY as "YYYY-MM" (e.g. "2026-06") — never a natural language date like
"June 2026" or "06/2026".

If the question can be answered without data (e.g. "what is a good savings
rate?"), set needs_data to false.

If the question is ambiguous or lacks context (e.g. "what about those?"), set needs_data to false.
"""

llm = ChatGroq(model='qwen/qwen3.6-27b', api_key=os.getenv('GROQ_API_KEY'))
schema_str = RetrievalPlan.model_json_schema()
system_prompt = f"{PLANNER_PROMPT}\n\nYou MUST respond with valid JSON matching EXACTLY this schema (use these exact field names, no extras):\n{schema_str}"

planner = llm.with_structured_output(RetrievalPlan, method='json_mode')

print("Testing...")
for i in range(2):
    try:
        print(f"Attempt {i+1}...")
        result = planner.invoke([("system", system_prompt), ("human", "What about those?")])
        print(f"  OK: {result}")
    except Exception as e:
        print(f"  FAIL: {type(e).__name__} - {e}")
