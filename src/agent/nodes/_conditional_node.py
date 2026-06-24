

from src.agent.state import AgentState


def _increment_loop_count(state: AgentState) -> dict:
  """Lightweight pass-through that increments the loop counter.

  Sits between research_node and the conditional edge so the counter
  is always up to date when should_continue() reads it.
  Keeping this logic here (not in research_node) means research_node
  doesn't need to know it might be called in a loop.
  """
  return {"research_loop_count": state.get("research_loop_count", 0) + 1}
 
