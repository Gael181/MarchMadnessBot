import os
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from google import genai


class LLMService:
    def __init__(self):
        gemini_key = os.getenv("GEMINI_API_KEY")
        google_key = os.getenv("GOOGLE_API_KEY")

        if gemini_key and google_key:
            print("Both GOOGLE_API_KEY and GEMINI_API_KEY are set. Using GEMINI_API_KEY.")

        api_key = gemini_key or google_key
        if not api_key:
            raise ValueError("Set GEMINI_API_KEY (preferred) or GOOGLE_API_KEY.")

        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-2.5-flash"

    def _generate_with_retry(self, prompt: str, temperature: float, max_attempts: int = 4):
        delay = 1.0

        last_exception = None
        for attempt in range(max_attempts):
            try:
                return self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config={
                        "temperature": temperature,
                    },
                )
            except Exception as exc:
                last_exception = exc
                message = str(exc).lower()

                is_retryable = (
                    "503" in str(exc)
                    or "unavailable" in message
                    or "high demand" in message
                    or "overloaded" in message
                )

                if not is_retryable or attempt == max_attempts - 1:
                    raise

                time.sleep(delay)
                delay *= 2

        raise last_exception

    def generate_grounded_answer(
        self,
        question: str,
        context: str,
        temperature: float = 1.0,
    ) -> dict:
        prompt = f"""
You are a college basketball assistant.

Rules:
1. Use the dataset context as your primary evidence.
2. Explicitly reference supporting evidence from the dataset when available.
3. Do not invent statistics or seasons not present in the dataset context.
4. If the dataset context is insufficient, say so clearly.
5. If the question is general and outside the dataset, you may answer generally, but say that the answer is not grounded in the dataset.

User question:
{question}

Dataset context:
{context}
"""

        start_time = time.perf_counter()
        response = self._generate_with_retry(prompt, temperature=temperature)
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        return {
            "text": (response.text or "").strip(),
            "token_used": "N/A",
            "response_time": f"{elapsed_ms}ms",
        }

    def generate_comparison_answer(
        self,
        team1: str,
        team2: str,
        context1: str,
        context2: str,
        temperature: float = 1.0,
    ) -> dict:
        prompt = f"""
You are a college basketball analyst.

Compare the following two teams using ONLY the dataset context.

Rules:
1. Use only the provided context.
2. Do not invent stats.
3. Clearly compare strengths and weaknesses.
4. Mention specific stats when possible.
5. End with a short conclusion.

Team 1: {team1}
Context:
{context1}

Team 2: {team2}
Context:
{context2}

Provide:
- Key statistical comparison
- Tournament-related insights if available
- Final comparison summary
"""

        start_time = time.perf_counter()
        response = self._generate_with_retry(prompt, temperature=temperature)
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        return {
            "text": (response.text or "").strip(),
            "token_used": "N/A",
            "response_time": f"{elapsed_ms}ms",
        }