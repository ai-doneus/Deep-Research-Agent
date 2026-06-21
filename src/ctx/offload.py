"""
Offload: move context out of the prompt and into the file system.

Patterns implemented here (see project notes / README references):
  - todo.md for task tracking (Manus-style planning)
  - long_term/ notes for token-heavy research output, so raw content lives
    on disk and only a pointer (the file path) needs to stay in state.

The point of offloading is that nothing here needs to round-trip through
the LLM's context window - it's plain file I/O.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

MEMORY_DIR = Path(__file__).resolve().parents[2] / "memory"
TODO_PATH = MEMORY_DIR / "todo.md"
LONG_TERM_DIR = MEMORY_DIR / "long_term"


def write_todo(brief: str) -> str:
  """Write/refresh todo.md from the research brief. Returns the file path.

  Status values follow Manus's pending / in_progress / completed model.
  This base version writes a single top-level task; extend this to parse
  the brief into multiple tracked subtasks as the agent grows.
  """
  TODO_PATH.parent.mkdir(parents=True, exist_ok=True)
  content = (
    f"# TODO\n\n"
    f"- [ ] (pending) Research brief: {brief.strip()[:120]}...\n"
  )
  TODO_PATH.write_text(content, encoding="utf-8")
  return str(TODO_PATH)


def write_note(content: str, label: str = "note") -> str:
  """Persist a research note / tool output to long_term/ and return its path.

  Use this instead of appending large tool outputs directly to
  state['messages']. Only the returned path needs to live in context.
  """
  LONG_TERM_DIR.mkdir(parents=True, exist_ok=True)
  timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
  filename = f"{timestamp}_{label}_{uuid.uuid4().hex[:8]}.md"
  path = LONG_TERM_DIR / filename
  path.write_text(content, encoding="utf-8")
  return str(path)


def read_note(path: str) -> str:
  """Read back a previously offloaded note by path."""
  return Path(path).read_text(encoding="utf-8")
