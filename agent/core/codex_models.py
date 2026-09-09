"""Model-id helpers for the Codex app-server runtime.

``codex/<model>`` is intentionally separate from the OpenAI-compatible
LiteLLM providers. Codex owns its authentication session and may use either a
ChatGPT subscription or an OpenAI API key, depending on how ``codex login`` was
completed. ML Intern never reads or forwards Codex's cached credentials.
"""

CODEX_MODEL_PREFIX = "codex/"
CODEX_DEFAULT_MODEL_ID = f"{CODEX_MODEL_PREFIX}default"


def is_codex_model_id(model_id: str | None) -> bool:
    """Return ``True`` for a well-formed Codex runtime model id."""
    if not model_id or any(char.isspace() for char in model_id):
        return False
    return model_id.startswith(CODEX_MODEL_PREFIX) and bool(
        model_id.removeprefix(CODEX_MODEL_PREFIX)
    )


def codex_model_name(model_id: str) -> str | None:
    """Return the model passed to Codex, or ``None`` for its current default."""
    if not is_codex_model_id(model_id):
        raise ValueError(f"Unsupported Codex model id: {model_id}")
    name = model_id.removeprefix(CODEX_MODEL_PREFIX)
    return None if name == "default" else name
