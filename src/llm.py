"""
Builds the chat model every tool uses, so switching provider is a .env edit.

LLM_PROVIDER:
  openrouter  — default; OpenAI-compatible gateway (model: OPENROUTER_MODEL)
  groq        — free tier, 200k tokens/day (too little for a full ablation)
  openai      — OpenAI API

API keys are read from os.environ when a client is built — never stored in
config, never logged. ChatOpenAI wraps the official `openai` SDK; base_url
points it at OpenRouter.
"""

import os

from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from config.settings import LLM_MODEL, LLM_PROVIDER, OPENROUTER_MODEL

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Attribution headers — requests show up under this project in the OpenRouter dashboard
OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/Abdul-Ahad-Barrawas/support-agent",
    "X-Title": "Support Ticket Triage Agent",
}


def get_api_key(name: str) -> str:
    """Read an API key from the environment, failing loudly if it's absent."""
    try:
        key = os.environ[name]
    except KeyError:
        key = ""
    if not key.strip():
        raise RuntimeError(
            f"{name} is not set. Export it in your shell, or add it to a .env file "
            f"in the project root (see .env.example)."
        )
    return key


def model_name(provider: str = LLM_PROVIDER) -> str:
    return OPENROUTER_MODEL if provider == "openrouter" else LLM_MODEL


def get_llm(temperature: float = 0.0, provider: str = LLM_PROVIDER,
            model: str | None = None) -> BaseChatModel:
    model = model or model_name(provider)

    if provider == "openrouter":
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=get_api_key("OPENROUTER_API_KEY"),
            base_url=OPENROUTER_BASE_URL,
            default_headers=OPENROUTER_HEADERS,
            # Only route to providers that honour response_format, or
            # structured output can silently come back as free text
            extra_body={"provider": {"require_parameters": True}},
        )
    if provider == "groq":
        return ChatGroq(model=model, temperature=temperature,
                        api_key=get_api_key("GROQ_API_KEY"))
    if provider == "openai":
        return ChatOpenAI(model=model, temperature=temperature,
                          api_key=get_api_key("OPENAI_API_KEY"))
    raise ValueError(f"Unknown LLM_PROVIDER {provider!r} — use openrouter, groq or openai")
