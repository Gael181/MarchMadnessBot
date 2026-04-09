from logging import exception

from bot.dataset import search
from bot.services.llm_service import LLMService


class ChatService:
    @staticmethod
    def answer_question(question: str, temperature: float = 0.2) -> dict:
        results = search(question, top_k=3)

        if not results:
            return {
                'text': "I could not find relevant information in the dataset.",
                'token_used': 'N/A',
                'response_time': 'N/A',
            }

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
        except Exception as e:
            print(e)
            lines = [
                "LLM generation is unavailable right now, so here is the retrieved dataset evidence:"
            ]
            for r in results:
                lines.append(f"- {r['team']} ({r['season']}): {r['text']}")
            return {
                'text': "\n".join(lines),
                'token_used': 'N/A',
                'response_time': 'N/A',
            }