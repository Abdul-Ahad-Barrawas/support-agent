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
import time

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from config.settings import (
    LLM_MAX_RETRIES,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_TIMEOUT,
    OPENROUTER_MODEL,
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Attribution headers — requests show up under this project in the OpenRouter dashboard
OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/Abdul-Ahad-Barrawas/support-agent",
    "X-Title": "Support Ticket Triage Agent",
}


class _DeadlineStream(httpx.SyncByteStream):
    def __init__(self, stream: httpx.SyncByteStream, deadline: float, request: httpx.Request):
        self._stream, self._deadline, self._request = stream, deadline, request

    def __iter__(self):
        for chunk in self._stream:
            if time.monotonic() > self._deadline:
                raise httpx.ReadTimeout("LLM request exceeded its total time limit",
                                        request=self._request)
            yield chunk

    def close(self):
        self._stream.close()


class DeadlineTransport(httpx.BaseTransport):
    """
    Caps the TOTAL time of each HTTP request.

    httpx's timeout only limits the gap between bytes, and OpenRouter keeps a
    slow request alive by trickling whitespace — so without this a stuck
    upstream provider hangs the call indefinitely (seen: 40+ min). Raising
    httpx.ReadTimeout lets the SDK treat it as an ordinary timeout and retry.
    """

    def __init__(self, total_timeout: float, transport: httpx.BaseTransport | None = None):
        self.total_timeout = total_timeout
        self._transport = transport or httpx.HTTPTransport()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        deadline = time.monotonic() + self.total_timeout
        response = self._transport.handle_request(request)
        response.stream = _DeadlineStream(response.stream, deadline, request)
        return response

    def close(self):
        self._transport.close()


def _http_client() -> httpx.Client:
    return httpx.Client(transport=DeadlineTransport(LLM_TIMEOUT), timeout=LLM_TIMEOUT)


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
    limits = {"timeout": LLM_TIMEOUT, "max_retries": LLM_MAX_RETRIES, "http_client": _http_client()}

    if provider == "openrouter":
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            **limits,
            api_key=get_api_key("OPENROUTER_API_KEY"),
            base_url=OPENROUTER_BASE_URL,
            default_headers=OPENROUTER_HEADERS,
            # Only route to providers that honour response_format, or
            # structured output can silently come back as free text
            extra_body={"provider": {"require_parameters": True}},
        )
    if provider == "groq":
        return ChatGroq(model=model, temperature=temperature, **limits,
                        api_key=get_api_key("GROQ_API_KEY"))
    if provider == "openai":
        return ChatOpenAI(model=model, temperature=temperature, **limits,
                          api_key=get_api_key("OPENAI_API_KEY"))
    raise ValueError(f"Unknown LLM_PROVIDER {provider!r} — use openrouter, groq or openai")
