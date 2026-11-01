import pytest
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from src.llm import OPENROUTER_BASE_URL, OPENROUTER_HEADERS, get_api_key, get_llm, model_name

FAKE_KEY = "test-key-not-real"


@pytest.fixture(autouse=True)
def fake_keys(monkeypatch):
    for name in ["OPENROUTER_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY"]:
        monkeypatch.setenv(name, FAKE_KEY)


def test_openrouter_client_config():
    llm = get_llm(provider="openrouter")

    assert isinstance(llm, ChatOpenAI)
    assert llm.openai_api_base == OPENROUTER_BASE_URL
    assert llm.model_name == model_name("openrouter")
    assert llm.default_headers == OPENROUTER_HEADERS
    assert llm.extra_body == {"provider": {"require_parameters": True}}


def test_key_is_read_from_environment_at_call_time(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "rotated-key")

    assert get_llm(provider="openrouter").openai_api_key.get_secret_value() == "rotated-key"


@pytest.mark.parametrize("value", [None, "", "   "])
def test_missing_key_fails_fast_with_clear_error(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("OPENROUTER_API_KEY")
    else:
        monkeypatch.setenv("OPENROUTER_API_KEY", value)

    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY is not set"):
        get_llm(provider="openrouter")


def test_error_message_never_contains_a_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY")

    with pytest.raises(RuntimeError) as err:
        get_api_key("GROQ_API_KEY")
    assert FAKE_KEY not in str(err.value)


def test_model_override():
    assert get_llm(provider="openrouter", model="openai/gpt-4o-mini").model_name == "openai/gpt-4o-mini"


def test_groq():
    llm = get_llm(0.3, provider="groq")

    assert isinstance(llm, ChatGroq)
    assert llm.temperature == 0.3


def test_openai():
    llm = get_llm(provider="openai", model="gpt-4o-mini")

    assert isinstance(llm, ChatOpenAI)
    assert llm.openai_api_base in (None, "https://api.openai.com/v1")


def test_unknown_provider_rejected():
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        get_llm(provider="typo")


def test_structured_output_methods_build_on_openrouter():
    # The exact calls the tools make at import time must work on the OpenAI client too
    from src.tools.intent_classifier import IntentResult
    from src.tools.priority_scorer import PriorityResult
    from src.tools.response_drafter import DraftResult

    llm = get_llm(provider="openrouter")
    llm.with_structured_output(IntentResult)
    llm.with_structured_output(PriorityResult, method="json_schema")
    llm.with_structured_output(DraftResult, method="json_schema")


@pytest.mark.parametrize("provider", ["openrouter", "groq", "openai"])
def test_every_provider_has_a_request_timeout_and_retries(provider):
    from config.settings import LLM_MAX_RETRIES, LLM_TIMEOUT

    llm = get_llm(provider=provider, model="m")

    assert llm.request_timeout == LLM_TIMEOUT
    assert llm.max_retries == LLM_MAX_RETRIES


# --- total-time limit (OpenRouter trickles whitespace, so httpx's inactivity timeout never fires) ---

import time  # noqa: E402

import httpx  # noqa: E402
import openai  # noqa: E402

from src.llm import DeadlineTransport  # noqa: E402


def trickling_transport(chunk_every: float, chunks: int, body_end: bytes = b""):
    """Server that keeps the connection alive with whitespace, like OpenRouter while waiting."""
    def handler(request):
        def body():
            for _ in range(chunks):
                time.sleep(chunk_every)
                yield b" "
            yield body_end
        return httpx.Response(200, headers={"content-type": "application/json"},
                              stream=httpx.ByteStream(b"") if chunks == 0 else _Iter(body()))
    return httpx.MockTransport(handler)


class _Iter(httpx.SyncByteStream):
    def __init__(self, it):
        self._it = it

    def __iter__(self):
        yield from self._it


def test_deadline_aborts_a_request_that_keeps_trickling():
    client = httpx.Client(transport=DeadlineTransport(0.3, trickling_transport(0.05, 1000)))

    start = time.monotonic()
    with pytest.raises(httpx.ReadTimeout, match="total time limit"):
        client.get("https://openrouter.test/")
    assert time.monotonic() - start < 1.0


def test_deadline_passes_fast_responses_through_intact():
    client = httpx.Client(transport=DeadlineTransport(5.0, trickling_transport(0.0, 3, b'{"ok": true}')))

    assert client.get("https://openrouter.test/").json() == {"ok": True}


def test_openai_sdk_turns_deadline_into_a_timeout_error():
    # The full path the tools use: SDK → our transport → hung "provider"
    sdk = openai.OpenAI(
        api_key=FAKE_KEY, base_url="https://openrouter.test/v1", max_retries=0,
        http_client=httpx.Client(transport=DeadlineTransport(0.3, trickling_transport(0.05, 1000))),
    )

    start = time.monotonic()
    with pytest.raises(openai.APITimeoutError):
        sdk.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])
    assert time.monotonic() - start < 1.0


@pytest.mark.parametrize("provider", ["openrouter", "groq", "openai"])
def test_every_provider_uses_the_deadline_transport(provider):
    llm = get_llm(provider=provider, model="m")

    assert isinstance(llm.http_client._transport, DeadlineTransport)
