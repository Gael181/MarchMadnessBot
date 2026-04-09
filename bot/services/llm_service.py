import os, dotenv
import time
from openai import OpenAI


class LLMService:
    def __init__(self):
        dotenv.load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set.")

        self.client = OpenAI(api_key=api_key)

    def generate_grounded_answer(self, question: str, context: str, temperature: float = 0.2) -> dict:
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

        start_time = time.perf_counter()
        response = self.client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
            temperature=temperature,
        )
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        usage = getattr(response, 'usage', None)
        token_used = None
        if usage is not None:
            token_used = getattr(usage, 'total_tokens', None)
            if token_used is None:
                token_used = getattr(usage, 'prompt_tokens', None)

        return {
            'text': response.output_text.strip(),
            'token_used': str(token_used) if token_used is not None else 'N/A',
            'response_time': f'{elapsed_ms}ms',
        }