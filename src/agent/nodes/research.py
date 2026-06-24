"""
Research node: runs tool calls against the brief and produces a condensed
research_summary for Write.

This node is where most of the context-engineering patterns from the
project notes actually fire:
  - offload:  long tool outputs get written to memory/ instead of staying
              in the message history (see context_engineering/offload.py)
  - reduce:   raw_research_notes get summarized before Write ever sees them
              (see context_engineering/reduce.py)
  - isolate:  research can be split across sub-agents that each get a
              narrow slice of context, never the full history
              (see context_engineering/isolate.py)

  Phase 2 — Tool loop
    - retrieve_relevant_tools() picks which tools match the brief (retrieve.py)
    - model.bind_tools() makes those tools callable by the model
    - The loop runs until the model stops emitting tool_calls:
        invoke -> tool_calls? -> execute each -> append result -> repeat
    - Each tool result is offloaded to disk via write_note() (offload.py)
    - raw_research_notes accumulates in state; Write never reads it directly
 
  Phase 3 — Prompt cache
    - Uses raw Anthropic client (not ChatAnthropic) so cache_control blocks
      are forwarded to the API (LangChain strips unknown keys)
    - RESEARCH_SYSTEM_PROMPT + TOOL_DESCRIPTIONS are static strings ->
      both get cache_control: ephemeral -> cached after first call
    - Only the brief (mutable per-run) goes in as a fresh user message
    - Tool call/result turns are appended as fresh messages after the cache
      breakpoint, which is correct: only the stable prefix should be cached
 
Context engineering patterns active in this node:
  offload  -> write_note() per tool result
  reduce   -> summarize_notes() before returning to state
  retrieve -> retrieve_relevant_tools() picks tools dynamically
  cache    -> build_cached_system() marks static prefix as ephemeral
"""

from __future__ import annotations
 
import json
 
from src.agent.state import AgentState
from src.ctx.cache import build_cached_system, make_client
from src.ctx.offload import write_note
from src.ctx.reduce import summarize_notes
from src.ctx.retrieve import retrieve_relevant_tools
from src.tools.web_search import ALL_TOOLS
 
MODEL = "claude-sonnet-4-6"
MAX_TOOL_ROUNDS = 6  # hard ceiling - prevents runaway loops
 
# ---------------------------------------------------------------------------
# Static strings for the cache prefix.
# NEVER f-string these or interpolate per-request data into them.
# The whole point is byte-for-byte identity across calls so the cache hits.
# ---------------------------------------------------------------------------
RESEARCH_SYSTEM_PROMPT = (
  "You are a {research sub-agent}++. "
  "You receive a research brief, not the full conversation. "
  "Investigate the brief's open questions by calling tools as needed. "
  "Be thorough: run multiple targeted searches rather than one broad one. "
  "Do not write a final answer - that is the Write node's job. "
  "When you have gathered enough information, stop calling tools and "
  "output a concise summary of your findings."
)
 
# One-line descriptions of every tool in ALL_TOOLS, used as the second
# cached block. Build once at import time from the ToolSpec registry so
# it never drifts out of sync with the actual tool list.
TOOL_DESCRIPTIONS = "Available tools:\n" + "\n".join(
  f"  - {spec.name}: {spec.description}" for spec in ALL_TOOLS
)
 
 
def _tool_map() -> dict:
  """name -> callable, built from ALL_TOOLS registry."""
  return {spec.name: spec.fn for spec in ALL_TOOLS}
 
 
def _to_anthropic_tool(spec) -> dict:
  """Convert a LangChain @tool into the raw Anthropic tool schema dict."""
  lc_tool = spec.fn
  return {
    "name": lc_tool.name,
    "description": lc_tool.description,
    "input_schema": lc_tool.args_schema.model_json_schema()
    if hasattr(lc_tool, "args_schema") and lc_tool.args_schema
    else {"type": "object", "properties": {}},
  }
 
 
def research_node(state: AgentState) -> dict:
  brief = state.get("researched_brief", state.get("research_brief", ""))
 
  # ------------------------------------------------------------------
  # Phase 2: retrieve — pick only the tools relevant to this brief
  # ------------------------------------------------------------------
  relevant_specs = retrieve_relevant_tools(brief, ALL_TOOLS, top_k=3)
  tools_schema = [_to_anthropic_tool(s) for s in relevant_specs]
  tool_lookup = {spec.name: spec.fn for spec in relevant_specs}

  # ------------------------------------------------------------------
  # Phase 3: cache — static system prefix marked ephemeral
  # ------------------------------------------------------------------
  client = make_client()
  system_blocks = build_cached_system(RESEARCH_SYSTEM_PROMPT, TOOL_DESCRIPTIONS)

  # Mutable per-run messages (NOT cached — changes every invocation)
  messages: list[dict] = [{"role": "user", "content": brief}]

  raw_notes: list[str] = list(state.get("raw_research_notes", []))
  note_paths: list[str] = []

  # ------------------------------------------------------------------
  # Phase 2: agentic tool loop
  # ------------------------------------------------------------------
  for round_num in range(MAX_TOOL_ROUNDS):
    response = client.messages.create(
      model=MODEL,
      max_tokens=4096,
      system=system_blocks,
      tools=tools_schema,
      messages=messages,
    )

    # Append the assistant turn to the running message list
    # so the next round has full context of what was said + done.
    messages.append({"role": "assistant", "content": response.content})

    # No tool calls -> model is done researching
    if response.stop_reason != "tool_use":
      # Collect any final text the model emitted
      for block in response.content:
        if hasattr(block, "type") and block.type == "text" and block.text.strip():
          raw_notes.append(block.text)
      break

    # Execute every tool call the model requested this round
    tool_results = []
    for block in response.content:
      if not (hasattr(block, "type") and block.type == "tool_use"):
        continue

      tool_name = block.name
      tool_input = block.input

      fn = tool_lookup.get(tool_name)
      if fn is None:
        result_text = f"[error] unknown tool: {tool_name}"
      else:
        try:
          result_text = fn.invoke(tool_input)
        except Exception as exc:
          result_text = f"[error] {tool_name} raised: {exc}"

      # Offload raw tool output to disk — Phase 2 ctx pattern
      note_path = write_note(
        content=f"# Tool: {tool_name}\nInput: {json.dumps(tool_input)}\n\n{result_text}",
        label=f"tool_{tool_name}_round{round_num + 1}",
      )
      note_paths.append(note_path)
      raw_notes.append(result_text)

      tool_results.append({
        "type": "tool_result",
        "tool_use_id": block.id,
        "content": result_text,
      })

    # Append tool results as a user turn so the model sees them next round
    messages.append({"role": "user", "content": tool_results})

  # ------------------------------------------------------------------
  # Phase 2: reduce — collapse notes before Write sees them
  # ------------------------------------------------------------------
  summary = summarize_notes(raw_notes)

  return {
    "raw_research_notes": raw_notes,
    "research_summary": summary,
    "messages": [{"role": "assistant", "content": f"Research complete. Notes saved to: {note_paths}"}],
  }
 
