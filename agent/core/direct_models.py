"""Helpers for direct OpenAI-compatible cloud model ids."""

DIRECT_MODEL_PROVIDERS: dict[str, dict[str, str]] = {
    "atlas/": {
        "base_url_env": "ATLASCLOUD_BASE_URL",
        "base_url_default": "https://api.atlascloud.ai/v1",
        "api_key_env": "ATLASCLOUD_API_KEY",
    },
}

DIRECT_MODEL_PREFIXES = tuple(DIRECT_MODEL_PROVIDERS)


def direct_model_provider(model_id: str) -> dict[str, str] | None:
    """Return provider config for a direct model id, if supported."""
    for prefix, config in DIRECT_MODEL_PROVIDERS.items():
        if model_id.startswith(prefix):
            return config
    return None


def direct_model_name(model_id: str) -> str | None:
    """Return the upstream model name with the direct provider prefix removed."""
    for prefix in DIRECT_MODEL_PREFIXES:
        if model_id.startswith(prefix):
            name = model_id[len(prefix) :]
            return name or None
    return None


def is_direct_model_id(model_id: str) -> bool:
    """Return True for non-empty, whitespace-free direct model ids."""
    if not model_id or any(char.isspace() for char in model_id):
        return False
    return direct_model_name(model_id) is not None
