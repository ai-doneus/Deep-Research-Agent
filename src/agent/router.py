
from agent.state import AgentState
from ctx.offload import is_todo_done

MAX_RESEARCH_LOOPS = 3

def should_continue(state: AgentState) -> str:
  messages = state["messages"]
  last_message = messages[-1]
  
  # If the last message has tool calls, we go to the tool node
  if last_message.tool_calls:
      return "tools"
  # Otherwise, we end the loop
  return "END"

def should_continue_loop(state: AgentState) -> str:
  """Return next node name based on todo completion state:

  Langgraph passes this return value to the conditional edge routing map
  in graph.py  
  research -> loop back into research node
  write -> proceed to deliverable node"""
  todo_path = state.get('todo_path',"")
  
  if not todo_path:
     return "write";

  loop_count = state.get("research_loop_count",0);
  
  if loop_count >= MAX_RESEARCH_LOOPS:
    return "write"
  
  if is_todo_done(todo_path):
    return "write"

  return "research"

     


    
