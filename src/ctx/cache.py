"""
Cache : structure prompts so te static part 
  ( insutructions , tool descriptions)
  can  be cached, 
and the only mutable part 
  (recent observations) 
  is sent fresh in each call

Anthropics prompt caching makes 
  cached input tokens substancially cheaper 
  than fresh ones
  but only if cached prefix is byte-to-byte identical across calls
  - so the system prompt and tool description 
  must never be string-interpolated with pre-call data

 it makes the split between 
 "stable cached prefix" and 
 "fresh per-call content" 
 explicit at the call site, 
 so callers can't accidentally cache something mutable.
"""

from __future__ import annotations

import anthropic

def build_cached_messages(
  system_prompt: str,
  tool_descriptions: str,
  mutable_ctx: str
) -> list[dict]:
  """ Assemble messages list with stable
       cacheable prefix and a mutable suffix

    `system_prompt` and `tool_descriptions` must be static strings 
    - do not interpolate per request data into them
    or the cache will always miss
    `mutable_ctx` is where recent observation/the current turn input goes
  """
  return [
    {
      "role":"system",
      "content": system_prompt,
      "cache_control": {"type": "ephemeral"}
    },
    {
      "role":"system",
      "content": tool_descriptions,
      "cache_control": {"type": "ephemeral"}
    },
    {
      "role":"user",
      "content": mutable_ctx,
    },
  ]

def build_cached_system(system_prompt: str, tool_descriptions: str) -> list[dict]:
    """
    Build the system content blocks with cache_control breakpoints.
 
    Returns a list of content blocks suitable for the `system` param of
    the raw Anthropic messages API. Two blocks, each marked ephemeral so
    the API can cache them independently.
 
    Args:
      system_prompt:  Static instruction string. NEVER interpolate
                      per-request data into this - it breaks caching.
      tool_descriptions:  Static string describing available tools
                          (names + one-line descriptions). Also static.
    """
    return [
        {
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": tool_descriptions,
            "cache_control": {"type": "ephemeral"},
        },
    ]


def make_client() -> anthropic.Anthropic:
  """Return a raw Anthropic client. API key is read from ANTHROPIC_API_KEY env var."""
  return anthropic.Anthropic()
