"""LLM factory (Groq via LlamaIndex)."""
from llama_index.llms.groq import Groq

from app.config import Settings
from app.core.exceptions import LLMNotConfiguredError


def get_llm(settings: Settings) -> Groq:
    if not settings.groq_api_key:
        raise LLMNotConfiguredError(
            "GROQ_API_KEY is not set. Create a free key at https://console.groq.com/keys "
            "and add it to your .env file."
        )
    return Groq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )
