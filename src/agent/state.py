"""
Shared State schema passed between graph nodes.

the brief is one thing that must traverse/survive the whole pipeline
evertything else is target to offloading or reduceing before it reaches the next node
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages

class AgentState(TypedDict, total=False):
  # Full Conversation so far Subject to 'reduce' 
  #  summarize or prune before being haded to the next node
  # context_engineering.reduce.py
  messages: Annotated[list,add_messages]

  # Output of the Scope node , 
  # compressed and durable artifact that steers Research and Write
  # written once read many times
  researched_brief:str

  # Path to the on-disk TODO file used for task tracking
  todo_path:str

  # Raw research notes / tool outputs before reduction
  #  Large and Disposable - never pass this directly into write
  raw_research_notes:list[str]

  # condensed research notes after reduce
  # this is what write reads form actually 
  research_summary: str

  # Final Deliverable produced by the write node
  final_report:str
