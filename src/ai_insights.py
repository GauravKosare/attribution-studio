"""AI-powered executive summary — the one piece of this project that's
explicitly generative, not analytical. Everything else in src/ computes a
number; this takes the numbers everything else already computed and writes
the paragraph a human analyst would have to write by hand otherwise (compare
to the manually-written summaries in docs/06_business_recommendation_template.md).

Design choices, all deliberate:
- **Opt-in, not automatic.** Costs money per call (a real API, someone's real
  key) and the rest of this project works completely without it — the
  dashboard's static "Biggest divergence" callout is the free, always-on
  version of what this does.
- **Server-side API key only, via environment variable.** Never a form field
  a visitor types a key into, never a key this code stores or forwards
  anywhere else — it reads `ANTHROPIC_API_KEY` from the environment the
  Flask process runs in (the same pattern as every other credential in this
  project) and does nothing if it's unset, rather than prompting for one.
- **The prompt is constrained to what the pipeline already computed** — this
  is a summarization/writing task over real numbers, not a request for the
  model to invent an analysis. It's explicitly instructed to keep the exact
  caveats (directional not causal, estimated spend, etc.) this project
  states everywhere else, so the AI-generated text doesn't accidentally
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


def generate_narrative_summary(pipeline_result: dict, api_key: str | None = None, model: str = "claude-sonnet-5") -> dict:
    """Generate a short executive-summary paragraph from an already-computed
    src.pipeline.run_pipeline() result. Returns {"summary": str, "model": str}
    on success, or {"error": str} if no API key is configured or the call
    fails -- callers (the Flask endpoint, the dashboard) must handle the
    error case gracefully since this feature is optional by design.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return {
            "error": "No ANTHROPIC_API_KEY configured. This feature is optional -- "
            "set the environment variable to enable it. See docs/08_advanced_analytics.md."
        }

    try:
        import anthropic
    except ImportError:
        return {"error": "The 'anthropic' package is not installed (pip install anthropic)."}

    # Trim the result to what the summary actually needs -- keeps the prompt
    # small and avoids sending the full Sankey/credit_long payload unnecessarily.
    condensed = {
        "summary": pipeline_result.get("summary"),
        "divergence": pipeline_result.get("divergence"),
        "credit_matrix": pipeline_result.get("credit_matrix"),
        "roi": pipeline_result.get("roi"),
        "roi_model_used": pipeline_result.get("roi_model_used"),
        "unallocated_pool": pipeline_result.get("unallocated_pool"),
        "source": pipeline_result.get("source"),
    }

    try:
        client = anthropic.Anthropic(api_key=key)
        response = client.messages.create(
            model=model,
            max_tokens=400,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": json.dumps(condensed, default=str)}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        return {"summary": text.strip(), "model": model}
    except Exception as e:  # pragma: no cover -- network/API errors, not unit-testable
        return {"error": f"AI summary generation failed: {e}"}
