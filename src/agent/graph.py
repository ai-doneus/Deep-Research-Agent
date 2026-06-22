"""
Graph definition: Scope -> Research -> Write.

This wires the three nodes into a linear LangGraph pipeline. Kept linear
on purpose for the base version - the natural extension point is adding a
conditional edge from Research back to itself (loop until the TODO list
is empty) or fanning Research out into N isolated sub-agents.
"""
from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from src.agent.state import AgentState
from src.agent.nodes.scope import scope_node
from src.agent.nodes.research import research_node
from src.agent.nodes.write import write_node


def build_graph():
  graph = StateGraph(AgentState)

  graph.add_node("scope", scope_node)
  graph.add_node("research", research_node)
  graph.add_node("write", write_node)

  graph.add_edge(START, "scope")
  graph.add_edge("scope", "research")
  graph.add_edge("research", "write")
  graph.add_edge("write", END)
  # graph.add_conditional_edges(
  #   "agent",
  #   should_continue,
  #   {
  #     "tools": "tools",
  #     "END": END
  #   }
  # )
  # workflow.set_entry_point("agent")
  return graph.compile()


# Importable compiled graph, e.g.:
#   from src.agent.graph import agent
#   agent.invoke({"messages": [{"role": "user", "content": "..."}]})
agent = build_graph()