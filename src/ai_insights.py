"""AI-powered executive summary — the one piece of this project that's
explicitly generative, not analytical. Everything else in src/ computes a
number; this takes the numbers everything else already computed and writes
the paragraph a human analyst would have to write by hand otherwise (compare
to the manually-written summaries in docs/06_business_recommendation_template.md).

Design choices, all deliberate:
- **Opt-in, not automatic.** The rest of this project works completely
  without it — the dashboard's static "Biggest divergence" callout is the
  free, always-on version of what this does.
- **Free-first provider chain, not a single paid API.** Tries providers in
  this order, using whichever key is actually configured:
    1. **Gemini** (Google AI Studio) — free tier, no card, current-gen
       model, best writing quality of the free options (see docs/08 for the
       comparison this ordering is based on)
    2. **Groq** — free tier, no card, fallback if Gemini's key is missing or
       its call fails (rate limit, outage, etc.) — different infrastructure
       entirely, so a Gemini-side outage doesn't take this feature down too
    3. **Anthropic** (Claude) — paid, last resort, kept for parity with how
       this feature was originally built and for anyone who'd rather pay
       for Claude's writing quality specifically
  Every provider gets the *same* constrained system prompt and the *same*
  condensed input, so switching providers doesn't change what's being asked
  of the model, only who answers.
- **Server-side API keys only, via environment variables.** Never a form
  field a visitor types a key into, never a key this code stores or
  forwards anywhere else — reads `GEMINI_API_KEY`, `GROQ_API_KEY`, and
  `ANTHROPIC_API_KEY` from the environment the Flask process runs in, and
  simply skips whichever ones are unset.
- **The prompt is constrained to what the pipeline already computed** — this
  is a summarization/writing task over real numbers, not a request for the
  model to invent an analysis. Every provider is explicitly instructed to
  keep the exact caveats this project states everywhere else (directional
  not causal, estimated spend), so the AI-generated text doesn't accidentally
  overclaim what a human-reviewed version of the same summary wouldn't.
"""
from __future__ import annotations

import json
import os

SYSTEM_PROMPT = """You are writing the executive summary for a marketing attribution analysis, in the style of docs/06_business_recommendation_template.md in this project: direct, quantified, and honest about what the analysis does NOT prove.

Rules:
- Open with the single most actionable finding (which channel is most over/under-funded relative to its modeled credit), stated in one sentence with real numbers.
- Then 2-3 sentences of supporting detail (which models agree, what the ROI table shows).
- Then exactly one sentence of explicit caveat: this is correlational attribution, not a causal/incrementality-tested result, and dollar figures may be estimated rather than audited.
- Under 150 words total. No preamble, no "Based on the data provided". Write it as the summary itself.
- Never invent a number that isn't in the JSON you're given."""

# (env var, human name) for each provider, in fallback order.
_PROVIDER_ORDER = [
    ("GEMINI_API_KEY", "gemini"),
    ("GROQ_API_KEY", "groq"),
    ("ANTHROPIC_API_KEY", "anthropic"),
]


def _condense(pipeline_result: dict) -> dict:
    """Trim the result to what the summary actually needs -- keeps the
    prompt small and avoids sending the full Sankey/credit_long payload."""
    return {
        "summary": pipeline_result.get("summary"),
        "divergence": pipeline_result.get("divergence"),
        "credit_matrix": pipeline_result.get("credit_matrix"),
        "roi": pipeline_result.get("roi"),
        "roi_model_used": pipeline_result.get("roi_model_used"),
        "unallocated_pool": pipeline_result.get("unallocated_pool"),
        "source": pipeline_result.get("source"),
    }


def _call_gemini(condensed: dict, api_key: str, model: str = "gemini-3.6-flash") -> dict:
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return {"error": "The 'google-genai' package is not installed (pip install google-genai)."}

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=json.dumps(condensed, default=str),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                # max_output_tokens caps thinking + the visible answer
                # COMBINED, not the answer alone. Measured directly against
                # the live API: thinking_level="LOW" (the floor -- Gemini 3.x
                # has no MINIMAL/off option) still consumes ~570-600 tokens
                # on its own before writing a single word of the actual
                # answer. 1500 leaves roughly 900 tokens of headroom for the
                # ~150-word answer after thinking, confirmed sufficient.
                max_output_tokens=1500,
                thinking_config=types.ThinkingConfig(thinking_level="LOW"),
            ),
        )
        text = (response.text or "").strip()
        if not text:
            return {"error": "Gemini returned an empty response."}
        return {"summary": text, "model": model, "provider": "gemini"}
    except Exception as e:  # pragma: no cover -- network/API errors
        return {"error": f"Gemini call failed: {e}"}


def _call_groq(condensed: dict, api_key: str, model: str = "openai/gpt-oss-120b") -> dict:
    try:
        from groq import Groq
    except ImportError:
        return {"error": "The 'groq' package is not installed (pip install groq)."}

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            max_completion_tokens=600,
            # gpt-oss models are reasoning models -- chain-of-thought goes to
            # a separate `.reasoning` field, not `.content`, but with
            # reasoning_effort unset the model can spend the whole token
            # budget reasoning before writing anything to `.content` at all,
            # confirmed against the live API (empty content, model_tokens
            # fully consumed). "low" is enough for this task: a direct
            # restatement of numbers already computed, not something that
            # benefits from extended reasoning.
            reasoning_effort="low",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(condensed, default=str)},
            ],
        )
        text = (response.choices[0].message.content or "").strip()
        if not text:
            return {"error": "Groq returned an empty response."}
        return {"summary": text, "model": model, "provider": "groq"}
    except Exception as e:  # pragma: no cover -- network/API errors
        return {"error": f"Groq call failed: {e}"}


def _call_anthropic(condensed: dict, api_key: str, model: str = "claude-sonnet-5") -> dict:
    try:
        import anthropic
    except ImportError:
        return {"error": "The 'anthropic' package is not installed (pip install anthropic)."}

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=400,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": json.dumps(condensed, default=str)}],
        )
        text = "".join(block.text for block in response.content if block.type == "text").strip()
        if not text:
            return {"error": "Anthropic returned an empty response."}
        return {"summary": text, "model": model, "provider": "anthropic"}
    except Exception as e:  # pragma: no cover -- network/API errors
        return {"error": f"Anthropic call failed: {e}"}


_CALLERS = {"gemini": _call_gemini, "groq": _call_groq, "anthropic": _call_anthropic}


def generate_narrative_summary(
    pipeline_result: dict,
    api_keys: dict[str, str] | None = None,
    providers: list[str] | None = None,
) -> dict:
    """Generate a short executive-summary paragraph from an already-computed
    src.pipeline.run_pipeline() result, trying providers in order (Gemini ->
    Groq -> Anthropic by default) and falling through to the next one on any
    failure -- a missing key, a rate limit, an outage. Returns
    {"summary": str, "model": str, "provider": str} on success, or
    {"error": str, "attempts": [...]} if every configured provider failed
    (or none were configured at all).

    `api_keys`: override dict {"gemini": "...", "groq": "...", "anthropic": "..."}
    -- for tests/explicit use. Normally left None, in which case each
    provider's key is read from its environment variable.
    `providers`: override the fallback order/subset (e.g. ["groq"] to force
    just one). Defaults to the full Gemini -> Groq -> Anthropic chain.
    """
    condensed = _condense(pipeline_result)
    order = providers or [name for _, name in _PROVIDER_ORDER]
    env_by_name = {name: env for env, name in _PROVIDER_ORDER}

    attempts = []
    for name in order:
        key = (api_keys or {}).get(name) or os.environ.get(env_by_name.get(name, ""), "")
        if not key:
            attempts.append({"provider": name, "error": f"No {env_by_name.get(name, name.upper() + '_API_KEY')} configured."})
            continue

        result = _CALLERS[name](condensed, key)
        if "error" not in result:
            if attempts:
                result["fell_back_from"] = [a["provider"] for a in attempts]
            return result
        attempts.append({"provider": name, "error": result["error"]})

    return {
        "error": (
            "No AI provider available. Set one of GEMINI_API_KEY (free, recommended), "
            "GROQ_API_KEY (free), or ANTHROPIC_API_KEY (paid) as an environment variable. "
            "See docs/08_advanced_analytics.md."
        ),
        "attempts": attempts,
    }
