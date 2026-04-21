import re
import time
from typing import NamedTuple, Optional, Tuple

from bot.dataset import search, initialize, STORE, is_Upset
from bot.services.llm_service import LLMService
from bot.services.prompts import QuestionIntent, PromptSelector

def route_dataset(question: str) -> str:
    q = question.lower()

    if any (k in q for k in [
        "upset", "seed", "12 vs 5", "11 vs 6",
        "historical trend", "how often", "most common upset"
    ]):
        return "tournament"
    
    return "teams"

def extract_teams(question: str):
    STOP = r"(?=\s*(?:$|[?!.,;]|\bin\b|\bfor\b|\bfrom\b|\bduring\b|\bover\b|\bseason\b))"
    patterns = [
        rf"compare\s+([A-Za-z][A-Za-z &\-\.\']*?)\s+vs\.?\s+([A-Za-z][A-Za-z &\-\.\']*?){STOP}",
        rf"compare\s+([A-Za-z][A-Za-z &\-\.\']*?)\s+and\s+([A-Za-z][A-Za-z &\-\.\']*?){STOP}",
        rf"([A-Za-z][A-Za-z &\-\.\']*?)\s+vs\.?\s+([A-Za-z][A-Za-z &\-\.\']*?){STOP}",
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

def extract_seeds(question: str) -> Tuple[Optional[int], Optional[int]]:
    q = question.lower()
    
    primary = (
        r"#?(\d{1,2})\s*[-]?\s*(?:seed|seeds)?\s*"
        r"(?:vs\.?|versus|against|v\.)\s*"
        r"#?(\d{1,2})\s*[-]?\s*(?:seed|seeds)?"
    )
    m = re.search(primary, q)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if 1 <= a <= 16 and 1 <= b <= 16 and a!= b:
            return a, b
        
    verbish = (
        r"(?:a\s+)?#?(\d{1,2})(?:\s*[-]?\s*seeds?)?\s+"
        r"(?:upsets?|beats?|over|defeats?|knock(?:s|ed)?\s*(?:off)?)\s+"
        r"(?:a\s+)?#?(\d{1,2})(?:\s*[-]?\s*seeds?)?"
    )
    m = re.search(verbish, q)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if 1 <= a <= 16 and 1 <= b <= 16 and a != b:
            return a, b
        
    seed_mentions = re.findall(r"#?(\d{1,2})[\s-]*seeds?\b", q)
    valid = [int(s) for s in seed_mentions if 1 <= int(s) <= 16]
    if len(valid) >= 2 and valid[0] != valid[1]:
        return valid[0], valid[1]
 
    return None, None

def is_trend_query(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in [
        "trend", "how often", "most common", "frequency", "upset"
    ])

def is_team_stats_query(question: str) -> bool:
    """Detect queries focused on team statistics and records."""
    q = question.lower()
    return any(k in q for k in [
        "stats", "record", "how many", "what is the record",
        "wins", "losses", "win-loss", "tournament record",
        "conference record", "seed history"
    ])

def is_prediction_query(question: str) -> bool:
    """Detect hypothetical prediction queries."""
    q = question.lower()
    return any(k in q for k in [
        "would", "if ", "how would", "could ", "match up",
        "matchup", "vs if", "hypothetical", "what if",
        "who would win"
    ])

def is_rules_query(question: str) -> bool:
    """Detect tournament rules and structure explanation queries."""
    q = question.lower()
    return any(k in q for k in [
        "what is", "what does", "mean", "explain", "rule",
        "how does", "seeding", "seed mean", "tournament structure",
        "bracket", "what's a"
    ])

def detect_intent(question: str) -> Tuple[QuestionIntent, dict]:
    """
    Detect the primary intent of a user question.
    
    Returns a tuple of (QuestionIntent, metadata_dict) where metadata includes:
    - For SEED_MATCHUP: "seed_a", "seed_b"
    - For TEAM_COMPARISON: "team1", "team2"
    - For TEAM_STATS: "team"
    - For PREDICTION: "team1", "team2"
    - For others: {}
    
    Decision tree:
    1. Seed matchup (specific seed numbers)
    2. Team comparison (two teams mentioned)
    3. Trend analysis (trend/upset keywords + tournament context)
    4. Team stats (stats/record keywords)
    5. Prediction (hypothetical keywords)
    6. Rules explanation (explanation keywords)
    7. Fallback to factual lookup
    """
    metadata = {}
    
    # Check for seed matchup first (most specific)
    seed_a, seed_b = extract_seeds(question)
    if seed_a is not None and seed_b is not None:
        metadata = {"seed_a": seed_a, "seed_b": seed_b}
        return QuestionIntent.SEED_MATCHUP, metadata
    
    # Check for team comparison (two teams)
    team1, team2 = extract_teams(question)
    if team1 and team2:
        # If it looks like a prediction, classify as prediction instead
        if is_prediction_query(question):
            metadata = {"team1": team1, "team2": team2}
            return QuestionIntent.PREDICTION, metadata
        # Otherwise it's a comparison
        metadata = {"team1": team1, "team2": team2}
        return QuestionIntent.TEAM_COMPARISON, metadata
    
    # Check for trend analysis (specific keywords + tournament dataset)
    if is_trend_query(question):
        dataset = route_dataset(question)
        if dataset == "tournament":
            return QuestionIntent.TREND_ANALYSIS, metadata
    
    # Check for team stats (one team mentioned)
    if is_team_stats_query(question) and team1:
        metadata = {"team": team1}
        return QuestionIntent.TEAM_STATS, metadata
    
    # Check for prediction (but no specific teams extracted yet)
    if is_prediction_query(question):
        return QuestionIntent.PREDICTION, metadata
    
    # Check for rules/explanation
    if is_rules_query(question):
        return QuestionIntent.RULES_EXPLANATION, metadata
    
    # Default fallback
    return QuestionIntent.FACTUAL_LOOKUP, metadata

def _filter_tournament_by_seeds(seed_a: int, seed_b: int):
    initialize("tournament")
    df = STORE["tournament"]["df"]

    seed1 = pd.to_numeric(df["seed"], errors="coerce")
    seed2 = pd.to_numeric(df["seed.1"], errors="coerce")

    mask = (
        ((seed1 == seed_a) & (seed2 == seed_b))
        | ((seed1 == seed_b) & (seed2 == seed_a))
    )
    matching = df[mask]
    upsets = matching[matching.apply(is_Upset, axis=1)]
    return matching, upsets

import pandas as pd
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
        "Here is evidence retrieved from the dataset instead:\n"
        +"error = "+lower
    )


class ChatAnswer(NamedTuple):
    text: str
    latency_ms: float
    outcome: str
    error_message: Optional[str] = None
    token_used: str = "N/A"
    response_time: str = "N/A"
    question_intent: str = ""


class ChatService:
    @staticmethod
    def answer_question(question: str, temperature: float = 0.2) -> ChatAnswer:
        t0 = time.perf_counter()
        
        # Detect intent early for logging
        intent, intent_metadata = detect_intent(question)
        intent_str = intent.value

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
                question_intent=intent_str,
            )

        # Seed Matchup Block ---------------------------------------------------------------------

        seed_a, seed_b = extract_seeds(question)
        if seed_a is not None and seed_b is not None:
            try:
                matching, upsets = _filter_tournament_by_seeds(seed_a, seed_b)
            except Exception as exc:
                return done(
                    "The knowledge search failed. Please try again shortly.",
                    "error",
                    error_message=str(exc),
                )
            
            total = len(matching)
            if total == 0:
                return done(
                    f"No {seed_a} vs {seed_b} seed games were found in the "
                    f"tournament dataset (1985-present).",
                    "no_results",
                )
            
            upset_count = len(upsets)
            pct = (upset_count / total) * 100.0
            favorite = min(seed_a, seed_b)
            underdog = max(seed_a, seed_b)

            sample_games = matching.head(10)["text_chunk"].tolist()
            context_lines = [
                f"Matchup: #{favorite} seed vs #{underdog} seed",
                f"Total games in dataset: {total}",
                f"Upsets (the #{underdog} seed winning): {upset_count} "
                f"({pct:.1f}% of games)",
                "",
                "Sample games (up to 10):",
            ]
            context_lines.extend(f"- {t}" for t in sample_games)
            context = "\n".join(context_lines)

            try:
                payload = LLMService().generate_answer(
                    PromptSelector.get_prompt(QuestionIntent.SEED_MATCHUP),
                    {
                        "seed_a": seed_a,
                        "seed_b": seed_b,
                        "context": context,
                    },
                    temperature_override=0.7,
                )
                return done(
                    payload["text"].strip(),
                    "trend_success",
                    token_used=payload.get("token_used", "N/A"),
                    response_time=payload.get("response_time"),
                )
            except Exception as exc:
                header = _friendly_llm_failure_message(exc)
                lines = [
                    header,
                    "",
                    f"#{favorite} vs #{underdog} seed matchups "
                    f"(NCAA Tournament, 1985-present):",
                    f"- Games played: {total}",
                    f"- #{underdog} seed wins (upsets): {upset_count} "
                    f"({pct:.1f}%)",
                    f"- #{favorite} seed wins: {total - upset_count} "
                    f"({100.0 - pct:.1f}%)",
                    "",
                    "Sample games:",
                ]
                lines.extend(f"- {t}" for t in sample_games)
                return done(
                    "\n".join(lines),
                    "llm_fallback",
                    error_message=str(exc),
                )

        # ---------------------------------------------------------------------------------------

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
                payload = LLMService().generate_answer(
                    PromptSelector.get_prompt(QuestionIntent.TEAM_COMPARISON),
                    {
                        "team1": team1,
                        "team2": team2,
                        "context1": context_team1,
                        "context2": context_team2,
                    },
                    temperature_override=temperature,
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
                    lines.append(f"- {r['text']}")
                for r in results_team2:
                    lines.append(f"- {r['text']}")
                return done(
                    "\n".join(lines),
                    "llm_fallback",
                    error_message=str(exc),
                )
            
        # --------------------------------------------------------------------------------------

        # Default Block: Use intent detection to select appropriate prompt ---------------------

        intent, intent_metadata = detect_intent(question)
        dataset = route_dataset(question)
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
            prompt_template = PromptSelector.get_prompt(intent)
            
            # Build template params based on intent
            template_params = {"context": context, "question": question}
            
            # Add any metadata-driven params
            if intent == QuestionIntent.TEAM_STATS and "team" in intent_metadata:
                template_params["team"] = intent_metadata["team"]
            elif intent == QuestionIntent.PREDICTION and "team1" in intent_metadata:
                template_params["team1"] = intent_metadata["team1"]
                template_params["team2"] = intent_metadata.get("team2", "")
            elif intent == QuestionIntent.TREND_ANALYSIS:
                template_params["question"] = question
            
            # Use lower temperature for stats/factual; higher for creative/predictive
            temp_override = None
            if intent in [QuestionIntent.TEAM_STATS, QuestionIntent.RULES_EXPLANATION]:
                temp_override = 0.5
            elif intent == QuestionIntent.TREND_ANALYSIS:
                temp_override = 0.7
            elif intent == QuestionIntent.PREDICTION:
                temp_override = 0.8
            
            payload = llm.generate_answer(
                prompt_template,
                template_params,
                temperature_override=temp_override or temperature,
            )
            
            # Map intent to outcome for logging
            outcome_map = {
                QuestionIntent.FACTUAL_LOOKUP: "success",
                QuestionIntent.TEAM_COMPARISON: "success",
                QuestionIntent.SEED_MATCHUP: "trend_success",
                QuestionIntent.TREND_ANALYSIS: "trend_success",
                QuestionIntent.TEAM_STATS: "success",
                QuestionIntent.PREDICTION: "success",
                QuestionIntent.RULES_EXPLANATION: "success",
            }
            outcome = outcome_map.get(intent, "success")
            
            return done(
                payload["text"].strip(),
                outcome,
                token_used=payload.get("token_used", "N/A"),
                response_time=payload.get("response_time"),
            )
        except Exception as exc:
            header = _friendly_llm_failure_message(exc)

            if intent == QuestionIntent.TREND_ANALYSIS and dataset == "tournament":
                lines = [header, "", "Upset trend analysis (dataset-driven fallback):"]
                df = STORE["tournament"]["df"]
                seed_counts = {("12", "5"): 0, ("11", "6"): 0, ("10", "7"): 0}

                for r in results:
                    m = re.search(
                        r"\(Seed\s+(\d{1,2})\)\s+\d+\s*-\s*[^\(]+\(Seed\s+(\d{1,2})\)",
                        r["text"],
                    )
                    if not m:
                        continue
                    s1, s2 = int(m.group(1)), int(m.group(2))
                    pair = tuple(sorted((s1, s2), reverse=True))
                    key = (str(pair[0]), str(pair[1]))
                    if key in seed_counts:
                        seed_counts[key] += 1

                lines.append("Seed matchups observed in retrieved sample:")
                any_found = False
                for (hi, lo), count in seed_counts.items():
                    if count:
                        lines.append(f"- {hi} vs {lo}: {count} game(s)")
                        any_found = True
                if not any_found:
                    lines.append("- No classic upset matchups in retrieved samples")
 
                lines.append("\nSample Games:")
                for r in results:
                    lines.append(f"{r['text'][:150]}")
 
                lines.append(
                    "\nNote: This is a sample-based estimate, not full dataset statistics."
                )
            else:
                lines = [header, "", "Retrieved rows from the dataset:"]

                for r in results:
                    lines.append(f"- {r['text']}")

            return done(
                "\n".join(lines),
                "llm_fallback",
                error_message=str(exc),
            )
