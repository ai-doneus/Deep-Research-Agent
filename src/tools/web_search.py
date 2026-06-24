"""
Example tool definition for the Research node.

Wraps Tavily search behind a plain function so it can be passed to the
LLM as a tool and also referenced by description in
context_engineering/retrieve.py for tool-retrieval.
"""
from __future__ import annotations

import os
from langchain_core.tools import tool
from tavily import TavilyClient

from src.ctx.retrieve import ToolSpec

@tool()
def web_search(query: str, max_results: int = 5) -> str:
  """Search the web via Tavily and return a condensed text summary.

  Description matters here: this string is what
  context_engineering.retrieve.retrieve_relevant_tools matches against,
  so keep it specific about what the tool does and when to use it.

  Use this tool when the research brief requires facts, news, data, or any
  information that benefits from live web results rather than model knowledge.
  Prefer specific, targeted queries over broad ones - one focused search beats
  three vague ones.

  The docstring above this line is what retrieve_relevant_tools() matches
  against, so keep it specific about what the tool does and when to use it.

  """
  api_key = os.environ.get("TAVILY_API_KEY")
  if not api_key:
    raise RuntimeError("TAVILY_API_KEY is not set - check your .env file")

  client = TavilyClient(api_key=api_key)
  results = client.search(query=query, max_results=max_results)

  lines = []
  for r in results.get("results", []):
    lines.append(f"- {r.get('title')}: {r.get('content', '')[:300]}")
  return "\n".join(lines) if lines else "No results found."

# ToolSpec wraps the function for retrieve.py's keyword scorer.
# The description here is what gets matched against the research brief -
# make it specific enough that it scores high on the right tasks.
WEB_SEARCH_SPEC = ToolSpec(
  name="web_search",
  description=(
    "Search the web for current information, news, facts, data, statistics, "
    "recent events, or any topic requiring live results beyond model knowledge."
  ),
  fn=web_search,
)
 
# Registry: all tools available to the research node.
# retrieve_relevant_tools() picks from this list at runtime.
ALL_TOOLS: list[ToolSpec] = [
    WEB_SEARCH_SPEC,
]
