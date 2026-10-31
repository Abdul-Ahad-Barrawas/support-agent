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
    from src.tools.priority_scorer import SentimentResult
    from src.tools.response_drafter import DraftResult

    llm = get_llm(provider="openrouter")
    llm.with_structured_output(IntentResult)
    llm.with_structured_output(SentimentResult, method="json_mode")
    llm.with_structured_output(DraftResult, method="json_schema")
