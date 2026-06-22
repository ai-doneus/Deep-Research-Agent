"""
Scope node: turn an ambiguous user request into a durable research brief.

This is the only node that talks to the user directly. Everything it
produces gets compressed into `research_brief` - a short, written-once
artifact that Research and Write read from instead of replaying the
original conversation. This is the first and cheapest context-engineering
win in the pipeline: pay the clarification cost once, then offload it.
"""
from __future__ import annotations

from langchain_anthropic import ChatAnthropic

from src.agent.state import AgentState

SCOPE_SYSTEM_PROMPT = """You turn a user's research request into a tight \
research brief: scope, key questions to answer, and what "done" looks \
like. Ask at most one clarifying question if the request is genuinely \
ambiguous; otherwise produce the brief directly. Keep the brief under \
200 words - it will be the only context downstream agents see about \
the user's intent."""


def scope_node(state: AgentState) -> dict:
  model = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

  messages = [{"role": "system", "content": SCOPE_SYSTEM_PROMPT}, *state["messages"]]
  response = model.invoke(messages)

  return {
    "messages": [response],
    "research_brief": response.content,
  }