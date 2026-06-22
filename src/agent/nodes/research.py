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
"""
from __future__ import annotations

from langchain_anthropic import ChatAnthropic

from src.agent.state import AgentState
from src.ctx.offload import write_todo, write_note
from src.ctx.reduce import summarize_notes

RESEARCH_SYSTEM_PROMPT = """You are a {research sub-agent}++. 
- You receive a \
  research brief, not the full conversation. 
- Investigate the brief's open questions \
  call tools as needed, \
, and produce concise findings. 
- Do not try to answer the user directly \
 - that is the Write node's job."""


def research_node(state: AgentState) -> dict:
  brief = state["research_brief"]
  todo_path = write_todo(brief)

  model = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
  messages = [
    {"role": "system", "content": RESEARCH_SYSTEM_PROMPT},
    {"role": "user", "content": brief},
  ]
  response = model.invoke(messages)

  # Offload the raw output to a file rather than keeping it all in-state.
  note_path = write_note(content=response.content, label="research_pass_1")

  raw_notes = state.get("raw_research_notes", []) + [response.content]

  # Reduce before handing off to Write - this is the step where
  # information loss is a real risk (see docs/tradeoffs.md).
  summary = summarize_notes(raw_notes)

  return {
    "todo_path": todo_path,
    "raw_research_notes": raw_notes,
    "research_summary": summary,
    "messages": [{"role": "assistant", "content": f"Research note saved to {note_path}"}],
  }
