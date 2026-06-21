"""
Reduce: shrink context before it's passed to the next node, via
summarization or pruning.

This is the highest-risk pattern in the pipeline - see docs/tradeoffs.md
for a concrete case where summarizing a tool output dropped a detail
Write later needed. Cognition and Manus both flag this same failure mode
in the source material this project is based on.
"""
from __future__ import annotations

from langchain_anthropic import ChatAnthropic

SUMMARIZE_SYSTEM_PROMPT = """
Task Name : Summarize 
Description : 
- Summarize the following research notes into \
 a dense, factual brief for a downstream writing agent. \ 
Rules :
- Preserve every \
  concrete fact, number, name, and caveat - cut only redundant phrasing and \
  narration. 
- If you are unsure whether a detail matters, keep it.
"""


def summarize_notes(notes: list[str]) -> str:
  """Collapse a list of raw research notes into one condensed summary.

  Deliberately instructed to err on the side of keeping detail, per the
  tradeoff documented in docs/tradeoffs.md - aggressive summarization
  is the easiest way to silently lose information the Write node needs.
  """
  if not notes:
    return ""

  combined = "\n\n---\n\n".join(notes)
  model = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
  response = model.invoke(
    [
      {"role": "system", "content": SUMMARIZE_SYSTEM_PROMPT},
      {"role": "user", "content": combined},
    ]
  )
  return response.content


def prune_messages(messages: list, keep_last_n: int = 10) -> list:
  """Drop all but the most recent N messages from a history.

  Simple recency-based pruning. A better version would prune by
  relevance to the current research_brief rather than recency alone -
  left as a known limitation, see docs/tradeoffs.md.
  """
  if len(messages) <= keep_last_n:
    return messages
  return messages[-keep_last_n:]
