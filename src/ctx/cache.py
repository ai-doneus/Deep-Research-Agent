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
"""

from __future__ import annotations

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
      "cache_control": {"type":" ephemeral"}
    },
    {
      "role":"system",
      "content": tool_descriptions,
      "cache_control": {"type":" ephemeral"}
    },
    {
      "role":"user",
      "content": mutable_ctx,
    },
  ]