import pytest

from rag.config import load_config
from rag.llm_client import LLMClient, LLMUnavailableError


def test_client_unavailable_without_provider(monkeypatch):
    monkeypatch.delenv("LEGALEASE_LLM_PROVIDER", raising=False)
    client = LLMClient(load_config())

    assert not client.is_available()

    with pytest.raises(LLMUnavailableError):
        client.generate("system", "user")


def test_openai_requires_api_key(monkeypatch):
    monkeypatch.setenv("LEGALEASE_LLM_PROVIDER", "openai")
    monkeypatch.delenv("LEGALEASE_LLM_API_KEY", raising=False)

    assert not LLMClient(load_config()).is_available()


def test_openai_available_with_api_key(monkeypatch):
    monkeypatch.setenv("LEGALEASE_LLM_PROVIDER", "openai")
    monkeypatch.setenv("LEGALEASE_LLM_API_KEY", "test-key")

    client = LLMClient(load_config())

    assert client.is_available()
    assert client.provider == "openai"


def test_ollama_available_without_key(monkeypatch):
    monkeypatch.setenv("LEGALEASE_LLM_PROVIDER", "ollama")
    monkeypatch.delenv("LEGALEASE_LLM_API_KEY", raising=False)

    assert LLMClient(load_config()).is_available()
