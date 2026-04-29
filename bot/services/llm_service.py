import os
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from google import genai
from bot.services.prompts import PromptTemplate

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

    def generate_answer(
        self,
        prompt_template: PromptTemplate,
        template_params: dict,
        temperature_override: float = None,
    ) -> dict:
        """
        Generate an answer using a prompt template with context interpolation.
        
        Args:
            prompt_template: PromptTemplate instance to use
            template_params: Dict of parameters to interpolate into the template
                (e.g., {"question": "...", "context": "..."})
            temperature_override: Optional temperature override; uses template default if None
        
        Returns:
            Dict with "text", "token_used", "response_time" keys
        """
        # Build the final prompt
        prompt_str = prompt_template.build(**template_params)
        
        # Use override temperature if provided, otherwise use template default
        temperature = temperature_override if temperature_override is not None else prompt_template.get_temperature()
        
        start_time = time.perf_counter()
        response = self._generate_with_retry(prompt_str, temperature=temperature)
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        
        return {
            "text": (response.text or "").strip(),
            "token_used": "N/A",
            "response_time": f"{elapsed_ms}ms",
        }

    def generate_trend_answer(
        self,
        prompt_template: PromptTemplate,
        template_params: dict,
        temperature_override: float = None,
    ) -> dict:
        return self.generate_answer(
            prompt_template,
            template_params,
            temperature_override=temperature_override,
        )

    def generate_comparison_answer(
        self,
        prompt_template: PromptTemplate,
        template_params: dict,
        temperature_override: float = None,
    ) -> dict:
        return self.generate_answer(
            prompt_template,
            template_params,
            temperature_override=temperature_override,
        )

    def generate_grounded_answer(
        self,
        prompt_template: PromptTemplate,
        template_params: dict,
        temperature_override: float = None,
    ) -> dict:
        return self.generate_answer(
            prompt_template,
            template_params,
            temperature_override=temperature_override,
        )