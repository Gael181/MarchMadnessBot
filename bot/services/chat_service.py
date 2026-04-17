import re
import time
from typing import NamedTuple, Optional

from bot.dataset import search
from bot.services.llm_service import LLMService

def route_dataset(question: str) -> str:
    q = question.lower()

    if any (k in q for k in [
        "upset", "seed", "12 vs 5", "11 vs 6", "historical trend", "how often", "most common upset"
    ]):
        return "tournament"
    
    return "teams"

def extract_teams(question: str):
    patterns = [
        r"compare\s+([A-Za-z][A-Za-z0-9 &\-\.\']{0,40}?)\s+vs\s+([A-Za-z][A-Za-z0-9 &\-\.\']{0,40}?)",
        r"compare\s+([A-Za-z][A-Za-z0-9 &\-\.\']{0,40}?)\s+and\s+([A-Za-z][A-Za-z0-9 &\-\.\']{0,40}?)",
        r"([A-Za-z][A-Za-z0-9 &\-\.\']{0,40}?)\s+vs\s+([A-Za-z][A-Za-z0-9 &\-\.\']{0,40}?)",
    ]

    for pattern in patterns:
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            team1 = match.group(1).strip()
            team2 = match.group(2).strip()

            if re.fullmatch(r"\d+", team1) and re.fullmatch(r"\d+", team2):
                return None, None

            return team1, team2

    return None, None

def is_trend_query(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in [
        "trend", "how often", "most common", "frequency", "upset"
    ])


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

        # Team Comparison Block -------------------------------------------------------------

        team1, team2 = extract_teams(question)

        if team1 and team2:
            try:
                results_team1 = search(team1, top_k=3, dataset="teams")
                results_team2 = search(team2, top_k=3, dataset="teams")
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
            
        # --------------------------------------------------------------------------------------

        dataset = route_dataset(question)
        is_trend = is_trend_query(question)
        top_k = 8 if dataset == "tournament" else 3

        try:
            results = search(question, top_k=top_k, dataset=dataset)
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
            if dataset == "tournament":
                context_parts.append(f"{result['text']}")
            else:
                context_parts.append(
                f"Result {result['rank']}: "
                f"Team={result['team']}, Season={result['season']}, "
                f"Conference={result['conference']}, Seed={result['seed']}, "
                f"Region={result['region']}. Details: {result['text']}"
            )
        context = "\n".join(context_parts)

        try:
            llm = LLMService()

            if dataset == "tournament" and is_trend:
                payload = llm.generate_trend_answer(
                    question,
                    context,
                    temperature=0.7,
                )
                outcome = "trend_success"
            else:
                payload = LLMService().generate_grounded_answer(
                    question,
                    context,
                    temperature=temperature,
                )
                outcome = "success"
            
            return done(
                payload["text"].strip(),
                outcome,
                token_used=payload.get("token_used", "N/A"),
                response_time=payload.get("response_time"),
            )
        except Exception as exc:
            header = _friendly_llm_failure_message(exc)
            is_trend = is_trend_query(question)
            dataset = route_dataset(question)

            if is_trend and dataset == "tournament":
                lines = [header, "", "Upset trend analysis (dataset-driven fallback):"]
                seed_counts = {}

                for r in results:
                    text = r["text"].lower()

                    if "12" in text and "5" in text:
                        seed_counts["12 vs 5"] = seed_counts.get("12 vs 5", 0) + 1
                    if "11" in text and "6" in text:
                        seed_counts["11 vs 6"] = seed_counts.get("11 vs 6", 0) + 1
                    if "10" in text and "7" in text:
                        seed_counts["10 vs 7"] = seed_counts.get("10 vs 7", 0)+ 1

                lines.append("Observed seed matchup patterns (from retrieved samples):")

                if seed_counts:
                    for k, v in seed_counts.items():
                        lines.append(f"- {k}: {v} occurences")
                else:
                    lines.append("- No clear upset patterns found in retrieved samples")

                lines.append("\nSample Games:")
                for r in results:
                    lines.append(f"{r['text'][:150]}")
                
                lines.append(
                    "\n Note: This is a sample-based estimate, not full dataset statistics."
                )
            else:
                lines = [header, "", "Retrieved rows from the dataset:"]

                for r in results:
                    lines.append(f"- {r['team']} ({r['season']}): {r['text']}")

            return done(
                "\n".join(lines),
                "llm_fallback",
                error_message=str(exc),
            )
