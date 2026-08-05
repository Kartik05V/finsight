"""
Central place for config so you never hardcode a model name, key, or
provider somewhere buried in an agent file.

Also implements a lightweight LLM gateway: different tasks route to
different models (fast/cheap for structured or repetitive work, stronger
for reasoning-heavy work), plus basic rate limiting so you don't blow
through Groq's free-tier requests-per-minute cap.
"""
import os
import time
from dotenv import load_dotenv

load_dotenv()

# Groq is the default because it's genuinely free (generous rate limits,
# no card required) and its OpenAI-compatible API works as a drop-in with
# LangChain's ChatGroq class, including .with_structured_output().
# Get a key at https://console.groq.com/keys
FINSIGHT_PROVIDER = os.getenv("FINSIGHT_PROVIDER") or "groq"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

DEFAULT_MODEL = os.getenv("FINSIGHT_MODEL") or (
    # llama-3.3-70b-versatile is Groq's solid free general-purpose model.
    "llama-3.3-70b-versatile" if FINSIGHT_PROVIDER == "groq" else "gpt-4o-mini"
)

# --- LLM Gateway: task-based model routing -------------------------------
#
# Not every call needs the same model. Structured/repetitive tasks
# (parsing a statement row, classifying which agent should handle a
# query) work fine on a smaller, faster model. Reasoning-heavy tasks
# (summing filtered transactions correctly, writing sensible financial
# insights) get the stronger model. This is the "LLM gateway" pattern
# from your course — routing requests to the right-sized model instead
# of using one model for everything.

FAST_MODEL = os.getenv("FINSIGHT_FAST_MODEL") or (
    "llama-3.1-8b-instant" if FINSIGHT_PROVIDER == "groq" else "gpt-4o-mini"
)
STRONG_MODEL = os.getenv("FINSIGHT_STRONG_MODEL") or DEFAULT_MODEL

TASK_MODEL_MAP = {
    "extraction": FAST_MODEL,            # repetitive structured parsing of statement rows
    "extraction_fallback": STRONG_MODEL,  # used if the fast model fails this schema
    "routing": FAST_MODEL,                # simple classification (which agent handles this?)
    "rag": STRONG_MODEL,                  # needs to filter + sum data correctly
    "analysis": STRONG_MODEL,             # narrative insight generation, more judgment
    "default": DEFAULT_MODEL,
}

# --- Basic rate limiting ---------------------------------------------------
#
# Groq's free tier caps requests-per-minute. This is a deliberately simple
# throttle (minimum gap between calls), not a full token-bucket limiter —
# enough to demonstrate the concept and avoid 429s during a demo, not a
# production-grade rate limiter.

MIN_SECONDS_BETWEEN_CALLS = float(os.getenv("FINSIGHT_MIN_CALL_INTERVAL", "1.0"))
_last_call_at = {"ts": 0.0}


def _throttle():
    elapsed = time.time() - _last_call_at["ts"]
    if elapsed < MIN_SECONDS_BETWEEN_CALLS:
        time.sleep(MIN_SECONDS_BETWEEN_CALLS - elapsed)
    _last_call_at["ts"] = time.time()


def get_llm(task: str = "default", temperature: float = 0):
    """
    Single factory every agent calls instead of importing ChatOpenAI/ChatGroq
    directly. `task` picks the model via TASK_MODEL_MAP (the gateway
    routing); swap providers by changing FINSIGHT_PROVIDER in .env — no
    code changes needed anywhere else in the project.
    """
    model = TASK_MODEL_MAP.get(task, DEFAULT_MODEL)
    _throttle()
    print(f"[gateway] task={task!r} -> model={model!r}")

    if FINSIGHT_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, temperature=temperature, api_key=GROQ_API_KEY)

    if FINSIGHT_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temperature, api_key=OPENAI_API_KEY)

    raise ValueError(f"Unknown FINSIGHT_PROVIDER: {FINSIGHT_PROVIDER}")
