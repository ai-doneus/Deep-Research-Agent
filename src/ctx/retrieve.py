"""
Retrieve: pull relevant context into the prompt on demand, instead of
keeping everything resident in context at all times.

Base version implements tool-description-based tool retrieval (pick the
right tool by matching its description against the task, rather than
listing every available tool in every prompt). 

 TODO: Document retrieval + re-ranking is a documented extension point, not implemented here yet.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ToolSpec:
  name: str
  description: str
  fn: callable


def retrieve_relevant_tools(
  task: str,
  tools: list[ToolSpec],
  top_k: int = 3
) -> list[ToolSpec]:
  """Return the top_k tools whose description best matches the task.

  Base implementation: simple keyword overlap between the task text and
  each tool's description. This is intentionally naive - 
  
  TODO: swap in an embedding-similarity or re-ranking step here as the tool count grows
  past what keyword matching can handle.
  """
  task_words = set(task.lower().split())

  def score(tool: ToolSpec) -> int:
    desc_words = set(tool.description.lower().split())
    return len(task_words & desc_words)

  ranked = sorted(tools, key=score, reverse=True)
  return ranked[:top_k]
