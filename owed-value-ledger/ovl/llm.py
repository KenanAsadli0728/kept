"""Thin, optional LLM wrapper. Returns None whenever the LLM is unavailable so callers fall back offline."""
from . import config

_client = None


def available() -> bool:
    return config.LLM_ENABLED


def complete(prompt: str, system: str = "", max_tokens: int = 300):
    """Return model text, or None (no key / SDK missing / API error). Tests monkeypatch this function."""
    if not available():
        return None
    global _client
    try:
        if _client is None:
            import anthropic
            _client = anthropic.Anthropic()
        msg = _client.messages.create(model=config.LLM_MODEL, max_tokens=max_tokens, system=system,
                                      messages=[{"role": "user", "content": prompt}])
        return msg.content[0].text
    except Exception:
        return None
