import os


class LLMService:
    def __init__(self):
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set.")

        self.client = OpenAI(api_key=api_key)

    def generate_grounded_answer(self, question: str, context: str, temperature: float = 0.2) -> str:
        prompt = f"""
        You are a college basketball assistant.

        Rules:
        1. Use the dataset context as your primary evidence.
        2. If the dataset context supports the answer, explicitly mention the supporting evidence.
        3. Do not invent statistics or seasons not present in the dataset context.
        4. If the dataset context is insufficient, say so clearly.
        5. If the question is general and outside the dataset, you may answer generally, but explicitly say that the answer is not grounded in the dataset.

        User question:
        {question}

        Dataset context:
        {context}
        """

        response = self.client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
            temperature=temperature,
        )

        return response.output_text.strip()