"""
Entrypoint for the agentic AI pipeline.
 
Usage:
    python main.py                        # prompts for input interactively
    python main.py "your research query"  # pass query directly as CLI arg
 
Pipeline: Scope -> Research -> Write
  - Scope   : turns the user query into a tight research brief
  - Research : investigates the brief (tool calls go here in Phase 2)
  - Write   : produces the final report from the brief + research summary
"""
from __future__ import annotations
 
import sys
import textwrap
from config import _check_env

def _get_query() -> str:
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:])
    print("Agentic AI — research pipeline")
    print("─" * 40)
    query = input("What do you want to research? ").strip()
    if not query:
        print("[error] empty query, exiting.")
        sys.exit(1)
    return query
 
 
def _print_section(title: str, content: str) -> None:
    width = 60
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")
    wrapped = textwrap.fill(content.strip(), width=width - 2, initial_indent="  ", subsequent_indent="  ")
    print(wrapped)
 
 
def main() -> None:
    _check_env()
    query = _get_query()
 
    # Import here so env is loaded before langchain/anthropic initialise
    from src.agent.graph import agent
 
    print(f"\n[running] query: {query!r}")
    print("[scope]    generating research brief …")
 
    initial_state = {
        "messages": [{"role": "user", "content": query}]
    }

    final_state = agent.invoke(initial_state)
 
    # scope_node writes to "research_brief" (note: state.py has a typo
    # "researched_brief" — tracked in Phase 1 bug list, not fixed here)
    brief = final_state.get("research_brief")
    summary = final_state.get("research_summary", "")
    report = final_state.get("final_report", "")
    todo_path = final_state.get("todo_path", "")
 
    if brief:
        _print_section("Research brief (scope node output)", brief)
 
    if todo_path:
        print(f"\n[offload]  todo written → {todo_path}")
 
    if summary:
        _print_section("Research summary (reduce node output)", summary)
 
    if report:
        _print_section("Final report", report)
    else:
        print("\n[warn] final_report is empty — check write_node output.")
 
    print(f"\n{'─' * 60}\n[done]\n")
 
 
if __name__ == "__main__":
    main()