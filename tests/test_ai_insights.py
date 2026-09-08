"""AI-generated executive summary — the actual LLM calls aren't unit-testable
(need real, free-tier-but-still-real API keys, and network access CI
shouldn't depend on), so these tests cover exactly what IS deterministic:
the provider fallback chain (Gemini -> Groq -> Anthropic) falls through
correctly on missing keys or failed calls, and the feature degrades
gracefully with a clear message when nothing is configured, rather than
crashing the request that asked for it.
"""
from __future__ import annotations

import sys
import types

import pytest

from src.ai_insights import generate_narrative_summary


def _fake_gemini_module(text="ok", raise_error=None):
    class FakeResponse:
        def __init__(self):
            self.text = text

    class FakeModels:
        def generate_content(self, **kwargs):
            if raise_error:
                raise raise_error
            return FakeResponse()

    class FakeClient:
        def __init__(self, api_key):
            self.models = FakeModels()

    class FakeTypes:
        class GenerateContentConfig:
            def __init__(self, **kwargs):
                pass

    fake_genai = types.SimpleNamespace(Client=FakeClient, types=FakeTypes)
    fake_google = types.SimpleNamespace(genai=fake_genai)
    return fake_google, fake_genai, FakeTypes


def _fake_groq_module(text="ok", raise_error=None):
    class FakeMessage:
        content = text

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]

    class FakeCompletions:
        def create(self, **kwargs):
            if raise_error:
                raise raise_error
            return FakeResponse()

    class FakeChat:
        completions = FakeCompletions()

    class FakeGroq:
        def __init__(self, api_key):
            self.chat = FakeChat()

    return types.SimpleNamespace(Groq=FakeGroq)


def _fake_anthropic_module(text="ok", raise_error=None):
    class FakeTextBlock:
        type = "text"

        def __init__(self, t):
            self.text = t

    class FakeResponse:
        def __init__(self, t):
            self.content = [FakeTextBlock(t)]

    class FakeMessages:
        def create(self, **kwargs):
            if raise_error:
                raise raise_error
            return FakeResponse(text)

    class FakeAnthropic:
        def __init__(self, api_key):
            self.messages = FakeMessages()

    return types.SimpleNamespace(Anthropic=FakeAnthropic)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Guarantee tests run as if no provider keys are set in the environment,
    regardless of the actual machine/CI they run on."""
    for var in ("GEMINI_API_KEY", "GROQ_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def test_no_provider_configured_returns_clear_error():
    result = generate_narrative_summary({"summary": {}})
    assert "error" in result
    assert "GEMINI_API_KEY" in result["error"]
    assert len(result["attempts"]) == 3  # all three tried, all three missing keys


def test_gemini_succeeds_first_no_fallback_needed(monkeypatch):
    fake_google, fake_genai, fake_types = _fake_gemini_module(text="Paid Search is under-funded.")
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)

    result = generate_narrative_summary(
        {"summary": {}}, api_keys={"gemini": "fake-gemini-key"}, providers=["gemini", "groq", "anthropic"]
    )
    assert "error" not in result
    assert result["provider"] == "gemini"
    assert "Paid Search" in result["summary"]
    assert "fell_back_from" not in result


def test_falls_back_to_groq_when_gemini_key_missing(monkeypatch):
    fake_groq = _fake_groq_module(text="Groq wrote this summary.")
    monkeypatch.setitem(sys.modules, "groq", fake_groq)

    # no gemini key provided at all -> should skip straight to groq
    result = generate_narrative_summary(
        {"summary": {}}, api_keys={"groq": "fake-groq-key"}, providers=["gemini", "groq", "anthropic"]
    )
    assert "error" not in result
    assert result["provider"] == "groq"
    assert result["fell_back_from"] == ["gemini"]


def test_falls_back_to_groq_when_gemini_call_fails(monkeypatch):
    fake_google, fake_genai, fake_types = _fake_gemini_module(raise_error=RuntimeError("gemini rate limited"))
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)

    fake_groq = _fake_groq_module(text="Fallback worked.")
    monkeypatch.setitem(sys.modules, "groq", fake_groq)

    result = generate_narrative_summary(
        {"summary": {}},
        api_keys={"gemini": "fake-key", "groq": "fake-key"},
        providers=["gemini", "groq", "anthropic"],
    )
    assert "error" not in result
    assert result["provider"] == "groq"
    assert result["fell_back_from"] == ["gemini"]


def test_falls_back_to_anthropic_when_gemini_and_groq_both_fail(monkeypatch):
    fake_google, fake_genai, fake_types = _fake_gemini_module(raise_error=RuntimeError("down"))
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)

    fake_groq = _fake_groq_module(raise_error=RuntimeError("rate limited"))
    monkeypatch.setitem(sys.modules, "groq", fake_groq)

    fake_anthropic = _fake_anthropic_module(text="Anthropic saved the day.")
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)

    result = generate_narrative_summary(
        {"summary": {}},
        api_keys={"gemini": "k1", "groq": "k2", "anthropic": "k3"},
        providers=["gemini", "groq", "anthropic"],
    )
    assert "error" not in result
    assert result["provider"] == "anthropic"
    assert set(result["fell_back_from"]) == {"gemini", "groq"}


def test_all_providers_fail_returns_error_with_attempts(monkeypatch):
    fake_google, fake_genai, fake_types = _fake_gemini_module(raise_error=RuntimeError("e1"))
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)

    fake_groq = _fake_groq_module(raise_error=RuntimeError("e2"))
    monkeypatch.setitem(sys.modules, "groq", fake_groq)

    fake_anthropic = _fake_anthropic_module(raise_error=RuntimeError("e3"))
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)

    result = generate_narrative_summary(
        {"summary": {}},
        api_keys={"gemini": "k1", "groq": "k2", "anthropic": "k3"},
        providers=["gemini", "groq", "anthropic"],
    )
    assert "error" in result
    assert len(result["attempts"]) == 3
    assert all("error" in a for a in result["attempts"])


def test_restricting_providers_list_skips_others(monkeypatch):
    fake_groq = _fake_groq_module(text="Only groq ran.")
    monkeypatch.setitem(sys.modules, "groq", fake_groq)

    result = generate_narrative_summary(
        {"summary": {}}, api_keys={"groq": "fake-key"}, providers=["groq"]
    )
    assert "error" not in result
    assert result["provider"] == "groq"


def test_empty_response_treated_as_failure(monkeypatch):
    fake_groq = _fake_groq_module(text="")  # empty string response
    monkeypatch.setitem(sys.modules, "groq", fake_groq)

    result = generate_narrative_summary(
        {"summary": {}}, api_keys={"groq": "fake-key"}, providers=["groq"]
    )
    assert "error" in result
