"""
Scope node: turn an ambiguous user request into a durable research brief.

This is the only node that talks to the user directly. Everything it
produces gets compressed into `research_brief` - a short, written-once
artifact that Research and Write read from instead of replaying the
original conversation. This is the first and cheapest context-engineering
win in the pipeline: pay the clarification cost once, then offload it.

phase 3 :
  - Uses raw Anthropic client so cache_control blocks reach the API.
    (LangChain's ChatAnthropic strips unknown keys from message dicts.)
  - SCOPE_SYSTEM_PROMPT is static -> marked ephemeral -> cached after
    the first call. Scope runs once per pipeline run, so caching matters
    most when the pipeline is called many times in a session.
  - Writes to `researched_brief` (the correct state key, matching state.py).
    The original stub wrote to `research_brief` — typo fixed here.

"""
from __future__ import annotations

from langchain_anthropic import ChatAnthropic

from src.agent.state import AgentState
from src.ctx.cache import build_cached_system, make_client

SCOPE_SYSTEM_PROMPT = """You turn a user's research request into a tight \
research brief: scope, key questions to answer, and what "done" looks \
like. 
Ask at most one clarifying question if the request is genuinely \
ambiguous; otherwise produce the brief directly. 
Keep the brief under 200 words \
- it will be the only context \
downstream agents see about \
the user's intent."""
 
MODEL = "claude-sonnet-4-6"
 
# Static — never interpolate per-request data here or the cache misses.
# SCOPE_SYSTEM_PROMPT = (
  # "You turn a user's research request into a tight research brief: "
  # "scope, key questions to answer, and what 'done' looks like. "
  # "Ask at most one clarifying question if the request is genuinely "
  # "ambiguous; otherwise produce the brief directly. "
  # "Keep the brief under 200 words - it will be the only context "
  # "downstream agents see about the user's intent."
# )
 
# No tools in scope, but build_cached_system requires two blocks.
# Use an empty second block so the cache structure stays consistent
# across all three nodes.
_NO_TOOLS = "No tools available in this node."
 
 

 
def scope_node(state: AgentState) -> dict:
  client = make_client()

  # Static prefix — cached after the first call
  system_blocks = build_cached_system(SCOPE_SYSTEM_PROMPT, _NO_TOOLS)

  # Mutable: the actual user message(s) — never cached
  user_content = state["messages"][-1].content if state["messages"] else ""

  response = client.messages.create(
    model=MODEL,
    max_tokens=512,
    system=system_blocks,
    messages=[{"role": "user", "content": user_content}],
  )

  brief = "".join(
    block.text for block in response.content
    if hasattr(block, "type") and block.type == "text"
  )

  return {
    "messages": [{"role": "assistant", "content": brief}],
    "researched_brief": brief,  # correct key — matches state.py
  }