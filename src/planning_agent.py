import ast
import json
import os

from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)


def planning_agent(topic: str, model: str = "google:gemini-3.1-flash-lite") -> list[str]:
    """
    This function is a placeholder for the planning agent implementation. It is intended to be used in the context of an agentic AI system, where the agent is responsible for making decisions and planning actions based on its environment and goals.
    """

    prompt = f"""
You are a planning agent responsible for organizing a research workflow using multiple intelligent agents.

🧠 Available agents:
- Research agent: MUST begin with a broad **web search using Tavily** to identify only **relevant** and **authoritative** items (e.g., high-impact venues, seminal works, surveys, or recent comprehensive sources). The output of this step MUST capture for each candidate: title, authors, year, venue/source, URL, and (if available) DOI.
- Research agent: AFTER the Tavily step, perform a **targeted arXiv search** ONLY for the candidates discovered in the web step (match by title/author/DOI). If an arXiv preprint/version exists, record its arXiv URL and version info. Do NOT run a generic arXiv search detached from the Tavily results.
- Writer agent: drafts based on research findings.
- Editor agent: reviews, reflects on, and improves drafts.

🎯 Produce a clear step-by-step research plan **as a valid Python list of strings** (no markdown, no explanations).
Each step must be atomic, actionable, and assigned to one of the agents.
Maximum of 7 steps.

🚫 DO NOT include steps like "create CSV", "set up repo", "install packages".
✅ Focus on meaningful research tasks (search, extract, rank, draft, revise).
✅ The FIRST step MUST be exactly:
"Research agent: Use Tavily to perform a broad web search and collect top relevant items (title, authors, year, venue/source, URL, DOI if available)."
✅ The SECOND step MUST be exactly:
"Research agent: For each collected item, search on arXiv to find matching preprints/versions and record arXiv URLs (if they exist)."

🔚 The FINAL step MUST instruct the writer agent to generate a comprehensive Markdown report that:
- Uses all findings and outputs from previous steps
- Includes inline citations (e.g., [1], (Wikipedia/arXiv))
- Includes a References section with clickable links for all citations
- Preserves earlier sources
- Is detailed and self-contained

Topic: "{topic}"
"""

    # Strip provider prefix if present (e.g., "google:gemini-..." -> "gemini-...")
    model_name = model.split(":")[-1]

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.choices[0].message.content.strip()
    print(f"Raw response from planning agent: {raw}")

    steps = _coerce_to_list(raw)
    steps = _ensure_contract(steps)

    print(f"Planning agent steps: {steps}")
    return steps


def _coerce_to_list(raw_str: str) -> list[str]:
    # Try JSON first
    try:
        obj = json.loads(raw_str)
        if isinstance(obj, list) and all(isinstance(item, str) for item in obj):
            return obj[:7]
    except json.JSONDecodeError:
        pass

    # Try Python literal evaluation
    try:
        obj = ast.literal_eval(raw_str)
        if isinstance(obj, list) and all(isinstance(item, str) for item in obj):
            return obj[:7]
    except Exception:
        pass

    # Try extracting from a code fence if present
    if raw_str.startswith("```") and raw_str.endswith("```"):
        inner = raw_str.strip("`").strip()
        try:
            obj = ast.literal_eval(inner)
            if isinstance(obj, list) and all(isinstance(item, str) for item in obj):
                return obj[:7]
        except Exception:
            pass

    return []


def _ensure_contract(steps: list[str]) -> list[str]:
    required_first = (
        "Research agent: Use Tavily to perform a broad web search and collect top relevant items "
        "(title, authors, year, venue/source, URL, DOI if available)."
    )
    required_second = (
        "Research agent: For each collected item, search on arXiv to find matching preprints/versions "
        "and record arXiv URLs (if they exist)."
    )
    final_required = (
        "Writer agent: Generate the final comprehensive Markdown report with inline citations "
        "and a complete References section with clickable links."
    )

    if not steps:
        return [
            required_first,
            required_second,
            "Research agent: Synthesize and rank findings by relevance, recency, and authority; deduplicate by title/DOI.",
            "Writer agent: Draft a structured outline based on the ranked evidence.",
            "Editor agent: Review for coherence, coverage, and citation completeness; request fixes.",
            final_required,
        ]

    steps = [s for s in steps if isinstance(s, str)]

    # Enforce first step
    if not steps or steps[0] != required_first:
        steps = [required_first] + steps

    # Enforce second step
    if len(steps) < 2 or steps[1] != required_second:
        remaining = [s for s in steps[1:] if "arxiv" not in s.lower() or "For each collected item" in s]
        steps = [steps[0], required_second] + remaining

    # Enforce final step
    if final_required not in steps:
        steps.append(final_required)

    return steps[:7]
