"""AI-generated executive summary — the actual LLM call isn't unit-testable
(needs a real, paid API key, and network access CI shouldn't depend on), so
these tests cover exactly what IS deterministic: the feature is fully
optional and fails gracefully with a clear message when no key is
configured, rather than crashing the request that asked for it.
"""
from __future__ import annotations

import os

import pytest

from src.ai_insights import generate_narrative_summary


@pytest.fixture(autouse=True)
def no_api_key(monkeypatch):
    """Guarantee these tests run as if no key is configured, regardless of
    the actual environment this suite happens to run in."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def test_missing_api_key_returns_error_not_exception():
    result = generate_narrative_summary({"summary": {"total_journeys": 100}})
    assert "error" in result
    assert "ANTHROPIC_API_KEY" in result["error"]


def test_missing_api_key_does_not_attempt_network_call():
    # If this tried to actually call the API without a key, it would raise
    # an SDK auth error instead of returning our own clean error dict.
    result = generate_narrative_summary({"summary": {}})
    assert "summary" not in result
    assert isinstance(result["error"], str)


def test_successful_call_returns_summary_and_model(monkeypatch):
    # Mock the Anthropic client entirely -- no real network call, no real
    # key needed, deterministic. Confirms the request/response plumbing
    # (condensing the pipeline result, extracting text blocks) is correct.
    import types

    class FakeTextBlock:
        type = "text"
        text = "Paid Search is under-funded relative to its modeled credit."

    class FakeResponse:
        content = [FakeTextBlock()]

    class FakeMessages:
        def create(self, **kwargs):
            assert "system" in kwargs and "messages" in kwargs
            return FakeResponse()

    class FakeAnthropic:
        def __init__(self, api_key):
            self.messages = FakeMessages()

    fake_module = types.SimpleNamespace(Anthropic=FakeAnthropic)
    monkeypatch.setitem(__import__("sys").modules, "anthropic", fake_module)

    result = generate_narrative_summary(
        {"summary": {"total_journeys": 100}, "roi": []}, api_key="sk-ant-fake-test-key"
    )
    assert "error" not in result
    assert "Paid Search" in result["summary"]
    assert result["model"] == "claude-sonnet-5"


def test_api_failure_returns_error_dict_not_exception(monkeypatch):
    import types

    class FakeMessages:
        def create(self, **kwargs):
            raise RuntimeError("simulated API failure")

    class FakeAnthropic:
        def __init__(self, api_key):
            self.messages = FakeMessages()

    fake_module = types.SimpleNamespace(Anthropic=FakeAnthropic)
    monkeypatch.setitem(__import__("sys").modules, "anthropic", fake_module)

    result = generate_narrative_summary({"summary": {}}, api_key="sk-ant-fake-test-key")
    assert "error" in result
    assert "simulated API failure" in result["error"]
