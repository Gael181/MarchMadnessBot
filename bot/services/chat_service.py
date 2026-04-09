from bot.dataset import search
from bot.services.llm_service import LLMService
import re

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

class ChatService:    
    @staticmethod
    def answer_question(question: str, temperature: float = 0.2) -> str:
        team1, team2 = extract_teams(question)

        if team1 and team2:
            results_team1 = search(team1, top_k=3)
            results_team2 = search(team2, top_k=3)

            if not results_team1 or not results_team2:
                return {
                    'text': "I could not find sufficient data for one or both teams.",
                    'token_used': 'N/A',
                    'response_time': 'N/A',
                }

            context_team1 = "\n".join([
                f"{r['team']} ({r['season']}): {r['text']}"
                for r in results_team1
            ])
            context_team2 = "\n".join([
                f"{r['team']} ({r['season']}): {r['text']}"
                for r in results_team2
            ])

            try:
                return LLMService().generate_comparison_answer(
                    team1,
                    team2,
                    context_team1,
                    context_team2,
                    temperature=temperature,
                )
            except Exception as e:
                print(f"LLM comparison failed: {e}")

                lines = [f"Comparison data for {team1.upper()} vs {team2.upper()}:"]
                for r in results_team1:
                    lines.append(f"- {team1.capitalize()} ({r['season']}): {r['text']}")
                for r in results_team2:
                    lines.append(f"- {team2.capitalize()} ({r['season']}): {r['text']}")
                return "\n".join(lines)
        
        results = search(question, top_k=3)

        if not results:
            return "I could not find relevant information in the dataset."

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
            return LLMService().generate_grounded_answer(
                question,
                context,
                temperature=temperature,
            )
        except Exception:
            lines = [
                "LLM generation is unavailable right now, so here is the retrieved dataset evidence:"
            ]
            for r in results:
                lines.append(f"- {r['team']} ({r['season']}): {r['text']}")
            return "\n".join(lines)

