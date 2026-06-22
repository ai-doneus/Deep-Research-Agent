"""
Write node: produces the final deliverable from research_brief +
research_summary only - never from raw_research_notes or full message
history. This is the payoff of the reduce/offload steps upstream: Write
runs on a small, high-signal context instead of the full research trace.
"""
from __future__ import annotations

from langchain_anthropic import ChatAnthropic

from src.agent.state import AgentState

WRITE_SYSTEM_PROMPT = """You write the final report for the user. You are \
given a research brief and a condensed research summary - not the raw \
research trace. Produce a clear, well-structured answer that satisfies \
the brief."""

def write_node(state: AgentState) -> dict:
  model = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

  prompt = (
    f"Research brief:\n{state['research_brief']}\n\n"
    f"Research summary:\n{state['research_summary']}\n\n"
    "Write the final report."
  )
  response = model.invoke([{"role": "user", "content": prompt}])

  return {
    "final_report": response.content,
    "messages": [response],
  }