"""
Central config and LLM gateway. Swap providers via FINSIGHT_PROVIDER in .env.
Different tasks route to different model sizes; includes rate limiting and retry.
"""
import os
import time
from dotenv import load_dotenv

from finsight.logging_setup import get_logger

logger = get_logger(__name__)

load_dotenv()

# Default provider is groq (free tier, no card needed). Get a key at https://console.groq.com/keys
FINSIGHT_PROVIDER = os.getenv("FINSIGHT_PROVIDER") or "groq"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# qwen/qwen3.6-27b: only Groq free-tier model with tool-calling support
DEFAULT_MODEL = os.getenv("FINSIGHT_MODEL") or (
    "qwen/qwen3.6-27b" if FINSIGHT_PROVIDER in ("groq", "hybrid") else "gpt-4o-mini"
)

# Fast model for cheap/repetitive tasks; strong model for reasoning-heavy tasks
FAST_MODEL = os.getenv("FINSIGHT_FAST_MODEL") or (
    "qwen/qwen3.6-27b" if FINSIGHT_PROVIDER in ("groq", "hybrid") else "gpt-4o-mini"
)
STRONG_MODEL = os.getenv("FINSIGHT_STRONG_MODEL") or DEFAULT_MODEL

TASK_MODEL_MAP = {
    "extraction": FAST_MODEL,            # structured parsing
    "extraction_fallback": STRONG_MODEL,  # fallback if fast model fails schema
    "routing": FAST_MODEL,                # simple classification
    "rag": STRONG_MODEL,                  # filtering + summing numbers
    "analysis": STRONG_MODEL,             # narrative insight generation
    "default": DEFAULT_MODEL,
}

# Basic throttle to avoid Groq free-tier rate limit (429s)
MIN_SECONDS_BETWEEN_CALLS = float(os.getenv("FINSIGHT_MIN_CALL_INTERVAL", "1.0"))
_last_call_at = {"ts": 0.0}


def _throttle():
    elapsed = time.time() - _last_call_at["ts"]
    if elapsed < MIN_SECONDS_BETWEEN_CALLS:
        time.sleep(MIN_SECONDS_BETWEEN_CALLS - elapsed)
    _last_call_at["ts"] = time.time()


# Exponential backoff retry: 3 attempts with 2→4→8s delays
_RETRY_DELAYS = [2.0, 4.0, 8.0]


def llm_call_with_retry(fn, *args, task: str = "default", **kwargs):
    """Call fn(*args, **kwargs) with up to 3 retries on failure."""
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
    Factory all agents use to get an LLM. Routes to the right model/provider
    based on task type. Change FINSIGHT_PROVIDER in .env to switch providers.
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
            # Gemini Flash: large context window, ideal for chunkless extraction
            return ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=temperature)
        elif task == "analysis":
            from langchain_google_genai import ChatGoogleGenerativeAI
            # Gemini Pro: deep reasoning for narrative reports
            return ChatGoogleGenerativeAI(model="gemini-3.1-pro-preview", temperature=temperature)
        else:
            from langchain_groq import ChatGroq
            # Groq for RAG and routing (low latency)
            return ChatGroq(model=model, temperature=temperature, api_key=GROQ_API_KEY)

    raise ValueError(f"Unknown FINSIGHT_PROVIDER: {FINSIGHT_PROVIDER}")
