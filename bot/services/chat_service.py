import time
from typing import NamedTuple, Optional

from bot.dataset import search
from bot.services.llm_service import LLMService


def _friendly_llm_failure_message(exc: Exception) -> str:
    """Plain-language explanation when the LLM cannot run (bad/missing key, limits, etc.)."""
    raw = str(exc)
    lower = raw.lower()

    if isinstance(exc, ValueError) and "openai_api_key" in lower:
        return (
            "The assistant cannot call OpenAI because OPENAI_API_KEY is not set. "
            "Set that environment variable to a valid key and restart the server."
        )

    _AuthErr = _RateErr = _StatusErr = None
    try:
        from openai import APIStatusError, AuthenticationError, RateLimitError

        _AuthErr = AuthenticationError
        _RateErr = RateLimitError
        _StatusErr = APIStatusError
    except ImportError:
        pass

    if _AuthErr is not None and isinstance(exc, _AuthErr):
        return (
            "The OpenAI API key is invalid or not authorized. "
            "Update OPENAI_API_KEY with a working key from your OpenAI account and try again."
        )

    if _RateErr is not None and isinstance(exc, _RateErr):
        return (
            "OpenAI rate limits were hit. Wait a bit and try again."
        )

    if _StatusErr is not None and isinstance(exc, _StatusErr):
        code = getattr(exc, "status_code", None)
        if code == 401:
            return (
                "The OpenAI API key was rejected (unauthorized). "
                "Check OPENAI_API_KEY and try again."
            )
        if code == 429:
            return (
                "OpenAI rate limits were hit. Wait a bit and try again."
            )

    if "401" in raw or "invalid_api_key" in lower or "incorrect api key" in lower:
        return (
            "The OpenAI API key did not work. Verify OPENAI_API_KEY and try again."
        )
    if "429" in raw or ("rate" in lower and "limit" in lower):
        return "OpenAI rate limits were hit. Wait a bit and try again."

    return (
        "The language model could not generate a reply right now. "
        "Here is evidence retrieved from the dataset instead:"
    )


class ChatAnswer(NamedTuple):
    text: str
    latency_ms: float
    outcome: str
    error_message: Optional[str]


class ChatService:
    @staticmethod
    def answer_question(question: str, temperature: float = 0.2) -> ChatAnswer:
        t0 = time.perf_counter()

        def done(text: str, outcome: str, error_message: Optional[str] = None) -> ChatAnswer:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return ChatAnswer(text=text, latency_ms=elapsed_ms, outcome=outcome, error_message=error_message)

        try:
            results = search(question, top_k=3)
        except Exception as exc:
            return done(
                "The knowledge search failed. Please try again shortly.",
                "error",
                str(exc),
            )

        if not results:
            return done(
                "I could not find relevant information in the dataset.",
                "no_results",
            )

        context_parts = []
        for result in results:
            context_parts.append(
                f"Result {result['rank']}: "
                f"Team={result['team']}, Season={result['season']}, "
                f"Conference={result['conference']}, Seed={result['seed']}, "
                f"Region={result['region']}. Details: {result['text']}"
            )
        context = "\n".join(context_parts)

        try:
            text = LLMService().generate_grounded_answer(
                question,
                context,
                temperature=temperature,
            )
            return done(text.strip(), "success")
        except Exception as exc:
            header = _friendly_llm_failure_message(exc)
            lines = [header, "", "Retrieved rows from the dataset:"]
            for r in results:
                lines.append(f"- {r['team']} ({r['season']}): {r['text']}")
            return done("\n".join(lines), "llm_fallback", str(exc))