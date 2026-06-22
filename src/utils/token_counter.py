"""
Helper to measure context size before/after applying offload/reduce, so
the effect of context engineering is provable rather than asserted.
"""
from __future__ import annotations

import tiktoken

_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
  """Approximate token count for a string (cl100k_base encoding).

  Anthropic models use their own tokenizer, so treat this as an
  order-of-magnitude estimate for comparing before/after sizes, not an
  exact billed-token count.
  """
  return len(_ENCODING.encode(text))


def count_messages_tokens(messages: list[dict]) -> int:
  """Sum approximate token count across a list of {"content": ...} messages."""
  return sum(count_tokens(str(m.get("content", ""))) for m in messages)
