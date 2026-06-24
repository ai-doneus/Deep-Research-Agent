"""
Isolate: give sub-agents a narrow slice of context instead of the full
shared history, so they can't see (or accidentally rely on) state outside
their assigned task.

Per the project notes: multi-agents that share full context can make
conflicting decisions (Cognition/Walden Yan). The mitigation used here
follows open-deep-research's approach - sub-agents are kept low-risk by
giving them research/lookup tasks, not decisions that affect shared state.
"""
from __future__ import annotations
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class SubAgentContext:
  """The minimal, isolated context handed to a sub-agent.

  Note there is no reference to the parent's full message history here -
  only what the sub-agent actually needs to do its job.
  """

  task: str
  research_brief: str
  extra_notes: list[str] = field(default_factory=list)


def make_subagent_context(
    task: str, research_brief: str, extra_notes: Optional[list[str]]
) -> SubAgentContext:
  """Build an isolated context for a single sub-agent task.

  Use this at every research/agent handoff instead of passing the full
  parent state - keeps sub-agents from making decisions based on context
  they shouldn't have, and keeps failures local to one sub-agent.
  """
  return SubAgentContext(
    task=task,
    research_brief=research_brief,
    extra_notes=extra_notes or [],
  )

def format_for_prompt(s_ctx: SubAgentContext)->str:
  """
  Serialises Sub agent ctx to mutable mut_ctx str
  this is what goes into  build_cache_messages(mutable_ctx)
  Plain text so the model can read naturally - no json - no special syntax
  Order matters 
    task first : highest priority
    brief next : why/ what done
    prior notes : supporting ctx
  """
  parts = [
    f"### Your Task: \n{s_ctx.task}",
    f"#####################",
    f"### Research Brief :\n\t {s_ctx.research_brief}"
  ]

  if s_ctx.extra_notes:
    notes_block = "\n\n---\n\n".join(s_ctx.extra_notes)
    parts.append(f"### Prior Findings (context only - do not repeat)\n{notes_block}")
  
  return "\n\n".join(parts)