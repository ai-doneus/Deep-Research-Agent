
from agent.state import AgentState

def should_continue(state: AgentState) -> str:
  messages = state["messages"]
  last_message = messages[-1]
  
  # If the last message has tool calls, we go to the tool node
  if last_message.tool_calls:
      return "tools"
  # Otherwise, we end the loop
  return "END"
