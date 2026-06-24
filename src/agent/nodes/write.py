"""
Write node: produces the final deliverable from research_brief +
research_summary only - never from raw_research_notes or full message
history. This is the payoff of the reduce/offload steps upstream: Write
runs on a small, high-signal context instead of the full research trace.
"""
from __future__ import annotations

from langchain_anthropic import ChatAnthropic

from src.agent.state import AgentState
from src.ctx.cache import make_client

WRITE_SYSTEM_PROMPT = """You write the final report for the user. 
You are \
given a research brief and a condensed research summary - not the raw \
research trace. 
Produce a clear, well-structured answer that satisfies \
the brief.
Use headers and bullet points where they improve readability.
"""

MODEL = "claude-sonnet-4-6"

def write_node(state: AgentState) -> dict:
  client = make_client()

  brief = state.get("researched_brief", "")
  summary = state.get("research_summary", "")

  # Cache structure for Write:
  #   block 0 (ephemeral): WRITE_SYSTEM_PROMPT  — static across all runs
  #   block 1 (ephemeral): the brief            — stable within one run;
  #                                               if Write is retried the
  #                                               brief won't change, so
  #                                               caching it is safe
  # research_summary goes in as fresh user content — it's the mutable part.
  system_blocks = [
    {
      "type": "text",
      "text": WRITE_SYSTEM_PROMPT,
      "cache_control": {"type": "ephemeral"},
    },
    {
      "type": "text",
      "text": f"Research brief:\n{brief}",
      "cache_control": {"type": "ephemeral"},
    },
  ]

  user_message = f"Research summary:\n{summary}\n\nWrite the final report."

  response = client.messages.create(
    model=MODEL,
    max_tokens=2048,
    system=system_blocks,
    messages=[{"role": "user", "content": user_message}],
  )

  report = "".join(
    block.text for block in response.content
    if hasattr(block, "type") and block.type == "text"
  )

  return {
    "final_report": report,
    "messages": [{"role": "assistant", "content": report}],
  }
