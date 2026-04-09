import re
import time
from typing import NamedTuple, Optional

from bot.dataset import search
from bot.services.llm_service import LLMService


def extract_teams(question: str):
    patterns = [
        r"compare (.+?) vs (.+)",
        r"compare (.+?) and (.+)",
        r"(.+?) vs (.+)",
    ]

    question_lower = question.lower()

    for pattern in patterns:
        match = re.search(pattern, question_lower)
        if match:
            team1 = match.group(1).strip()
            team2 = match.group(2).strip()
            return team1, team2

    return None, None


def _friendly_llm_failure_message(exc: Exception) -> str:
    raw = str(exc)
    lower = raw.lower()

    if isinstance(exc, ValueError) and (
        "gemini_api_key" in lower or "google_api_key" in lower
    ):
        return (
            "The assistant cannot call Gemini because GEMINI_API_KEY is not set. "
            "Set that environment variable and restart the server."
        )

    if "api key" in lower and ("invalid" in lower or "rejected" in lower):
        return (
            "The Gemini API key is invalid or not authorized. "
            "Update GEMINI_API_KEY and try again."
        )

    if "503" in raw or "quota" in lower or "rate" in lower or "unavailable" in lower:
        return (
            "Gemini is temporarily unavailable or rate-limited. "
            "Here is evidence retrieved from the dataset instead:"
        )

    return (
        "The language model could not generate a reply right now. "
        "Here is evidence retrieved from the dataset instead:"
    )


class ChatAnswer(NamedTuple):
    text: str
    latency_ms: float
    outcome: str
    error_message: Optional[str] = None
    token_used: str = "N/A"
    response_time: str = "N/A"


class ChatService:
    @staticmethod
    def answer_question(question: str, temperature: float = 0.2) -> ChatAnswer:
        t0 = time.perf_counter()

        def done(
            text: str,
            outcome: str,
            *,
            error_message: Optional[str] = None,
            token_used: str = "N/A",
            response_time: Optional[str] = None,
        ) -> ChatAnswer:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            rt = response_time if response_time is not None else f"{int(round(elapsed_ms))}ms"
            return ChatAnswer(
                text=text,
                latency_ms=elapsed_ms,
                outcome=outcome,
                error_message=error_message,
                token_used=token_used,
                response_time=rt,
            )

        team1, team2 = extract_teams(question)

        if team1 and team2:
            try:
                results_team1 = search(team1, top_k=3)
                results_team2 = search(team2, top_k=3)
            except Exception as exc:
                return done(
                    "The knowledge search failed. Please try again shortly.",
                    "error",
                    error_message=str(exc),
                )

            if not results_team1 or not results_team2:
                return done(
                    "I could not find sufficient data for one or both teams.",
                    "no_results",
                )

            context_team1 = "\n".join([
                f"{r['team']} ({r['season']}): {r['text']}"
                for r in results_team1
            ])
            context_team2 = "\n".join([
                f"{r['team']} ({r['season']}): {r['text']}"
                for r in results_team2
            ])

            try:
                payload = LLMService().generate_comparison_answer(
                    team1,
                    team2,
                    context_team1,
                    context_team2,
                    temperature=temperature,
                )
                return done(
                    payload["text"].strip(),
                    "success",
                    token_used=payload.get("token_used", "N/A"),
                    response_time=payload.get("response_time"),
                )
            except Exception as exc:
                lines = [f"Comparison data for {team1.upper()} vs {team2.upper()}:"]
                for r in results_team1:
                    lines.append(f"- {team1.capitalize()} ({r['season']}): {r['text']}")
                for r in results_team2:
                    lines.append(f"- {team2.capitalize()} ({r['season']}): {r['text']}")
                return done(
                    "\n".join(lines),
                    "llm_fallback",
                    error_message=str(exc),
                )

        try:
            results = search(question, top_k=3)
        except Exception as exc:
            return done(
                "The knowledge search failed. Please try again shortly.",
                "error",
                error_message=str(exc),
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
            payload = LLMService().generate_grounded_answer(
                question,
                context,
                temperature=temperature,
            )
            return done(
                payload["text"].strip(),
                "success",
                token_used=payload.get("token_used", "N/A"),
                response_time=payload.get("response_time"),
            )
        except Exception as exc:
            header = _friendly_llm_failure_message(exc)
            lines = [header, "", "Retrieved rows from the dataset:"]
            for r in results:
                lines.append(f"- {r['team']} ({r['season']}): {r['text']}")
            return done(
                "\n".join(lines),
                "llm_fallback",
                error_message=str(exc),
            )
