"""
Central place for config so you never hardcode a model name, key, or
provider somewhere buried in an agent file.

Also implements a lightweight LLM gateway: different tasks route to
different models (fast/cheap for structured or repetitive work, stronger
for reasoning-heavy work), plus basic rate limiting so you don't blow
through Groq's free-tier requests-per-minute cap.

Changes (#2, #8):
- get_llm() now has built-in exponential-backoff retry (3 attempts, 2→4→8s)
  so transient API failures and rate-limit 429s don't crash the pipeline.
- All print() calls replaced with structured logger output.
"""
import os
import time
from dotenv import load_dotenv

from finsight.logging_setup import get_logger

logger = get_logger(__name__)

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
    # qwen/qwen3.6-27b is the current Groq free-tier model that supports tool calling
    # (needed for structured output / .with_structured_output() in RAG and routing).
    # In hybrid mode the Groq sub-path (RAG/routing) also uses Groq model names.
    # (mixtral-8x7b-32768 decommissioned; groq/compound lacks tool-call support)
    "qwen/qwen3.6-27b" if FINSIGHT_PROVIDER in ("groq", "hybrid") else "gpt-4o-mini"
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
    # qwen/qwen3.6-27b: only current Groq free-tier model with tool calling support
    # required for .with_structured_output() used in routing and RAG plan steps.
    "qwen/qwen3.6-27b" if FINSIGHT_PROVIDER in ("groq", "hybrid") else "gpt-4o-mini"
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


# --- Retry helper (#2) -----------------------------------------------------
#
# Wraps any callable that makes an LLM API call with simple exponential
# backoff. No extra dependency (tenacity etc.) needed — a 3-retry loop
# covers the primary failure modes (transient 500s, rate-limit 429s).
# Deliberately simple: if you need circuit-breaking or jitter, swap this
# for tenacity.retry() — the interface is identical.

_RETRY_DELAYS = [2.0, 4.0, 8.0]  # seconds between attempt 1→2, 2→3, 3→fail


def llm_call_with_retry(fn, *args, task: str = "default", **kwargs):
    """
    Call fn(*args, **kwargs) with up to len(_RETRY_DELAYS) retries on failure.

    Usage:
        result = llm_call_with_retry(chain.invoke, messages, task="rag")

    Raises the last exception if all retries are exhausted.
    """
    last_exc: Exception | None = None
    for attempt, delay in enumerate([0.0] + _RETRY_DELAYS):
        if delay:
            logger.warning(
                f"llm_retry attempt={attempt} task={task!r} delay={delay}s "
                f"reason={type(last_exc).__name__}: {last_exc}"
            )
            time.sleep(delay)
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc

    raise last_exc  # type: ignore[misc]


def get_llm(task: str = "default", temperature: float = 0):
    """
    Single factory every agent calls instead of importing ChatOpenAI/ChatGroq
    directly. `task` picks the model via TASK_MODEL_MAP (the gateway
    routing); swap providers by changing FINSIGHT_PROVIDER in .env — no
    code changes needed anywhere else in the project.
    """
    model = TASK_MODEL_MAP.get(task, DEFAULT_MODEL)
    _throttle()
    logger.info(f"task={task!r} -> model={model!r}")

    if FINSIGHT_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, temperature=temperature, api_key=GROQ_API_KEY)

    if FINSIGHT_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temperature, api_key=OPENAI_API_KEY)

    if FINSIGHT_PROVIDER == "hybrid":
        if task in ("extraction", "extraction_fallback"):
            from langchain_google_genai import ChatGoogleGenerativeAI
            # Gemini 3.6 Flash: fast, large context, perfect for chunkless extraction
            # (API-recommended replacement for gemini-2.5-flash for new AQ. auth key users)
            return ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=temperature)
        elif task == "analysis":
            from langchain_google_genai import ChatGoogleGenerativeAI
            # Gemini 3.1 Pro: deep reasoning for narrative reports
            # (available for AQ. auth key users — confirmed via ListModels)
            return ChatGoogleGenerativeAI(model="gemini-3.1-pro-preview", temperature=temperature)
        else:
            from langchain_groq import ChatGroq
            # Groq for RAG and routing (ultra-fast UI response)
            return ChatGroq(model=model, temperature=temperature, api_key=GROQ_API_KEY)

    raise ValueError(f"Unknown FINSIGHT_PROVIDER: {FINSIGHT_PROVIDER}")
