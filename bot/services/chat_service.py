from bot.dataset import search


class ChatService:
    @staticmethod
    def answer_question(question: str) -> str:
        results = search(question, top_k=3)

        if not results:
            return "I could not find relevant information in the dataset."

        # Build a natural-language summary
        team_names = list({r["team"] for r in results if r["team"]})
        team_str = ", ".join(team_names)

        summary_lines = []

        if team_str:
            summary_lines.append(f"{team_str} appears in the dataset with the following performance:")

        for r in results:
            summary_lines.append(
                f"- {r['team']} in {r['season']} "
                f"had offensive efficiency {r['text'].split('Offensive Efficiency: ')[1].split(',')[0]} "
                f"and net rating {r['text'].split('Net Rating: ')[1].split(',')[0]}."
            )

        summary_lines.append("")
        summary_lines.append("These observations are based on the most relevant rows retrieved from the dataset.")

        return "\n".join(summary_lines)