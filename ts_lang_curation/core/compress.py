# core/compress.py — text rewrite/summarize via the team's Qwen vLLM API (OpenAI-compatible).
# Config via env vars (same names as the team doc):
#   OPENAI_BASE_URL (required), OPENAI_API_KEY (default EMPTY), OPENAI_MODEL (or server discovery).
import os
from functools import lru_cache

@lru_cache(maxsize=4)
def _client_for(base_url, api_key):
    from openai import OpenAI
    return OpenAI(base_url=base_url, api_key=api_key, timeout=60.0)


def _client():
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not base_url:
        raise RuntimeError("OPENAI_BASE_URL is required for generated annotations")
    return _client_for(base_url, os.environ.get("OPENAI_API_KEY", "EMPTY"))


@lru_cache(maxsize=4)
def _deployed_model(base_url):
    """The team swaps the vLLM deployment without notice (qwen -> gemma -> qwen, 2026-06);
    ask the server what it serves instead of hardcoding a name."""
    models = _client().models.list().data
    if not models:
        raise RuntimeError(f"no deployed model reported by {base_url}")
    return models[0].id


def complete(system, user, max_tokens=800, temperature=0.3, model=None):
    """Low-level chat call. Returns the assistant text. Model-aware:
    - Qwen/a3b: keep the system role + disable thinking (else content comes back empty);
    - Gemma (and other no-system models): fold system into the user turn, no extra kwargs."""
    mid = model or os.environ.get("OPENAI_MODEL") or _deployed_model(os.environ.get("OPENAI_BASE_URL"))
    kwargs = {"model": mid, "temperature": temperature, "max_tokens": max_tokens}
    if "gemma" in mid.lower():
        kwargs["messages"] = [{"role": "user", "content": f"{system}\n\n{user}"}]
    else:                                              # Qwen / a3b and similar
        kwargs["messages"] = [{"role": "system", "content": system},
                              {"role": "user", "content": user}]
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    content = _client().chat.completions.create(**kwargs).choices[0].message.content
    return (content or "").strip()


_SUMMARIZE_SYS = (
    "You compress source text for a time-series world-knowledge dataset. Rewrite it into at "
    "most {n} tokens, preserving the critical world knowledge (event, direction/magnitude, "
    "timing, named entities) and dropping boilerplate, disclaimers, and navigation. "
    "Output only the summary.")


def summarize(text, max_tokens=800):
    """Long text -> ~max_tokens summary. Returns (summary, is_generated_tag).
    On any failure, returns (text, 'real') so the caller still gets usable text."""
    try:
        out = complete(_SUMMARIZE_SYS.format(n=max_tokens), text, max_tokens=max_tokens)
        return (out, "derived_compressed") if out else (text, "real")
    except Exception:
        return text, "real"
