"""Helpers for CLI OpenAI-compatible model ids.

Covers both local inference servers (Ollama, vLLM, LM Studio, llama.cpp)
and the generic ``openai-compat/`` prefix, which points at any other
OpenAI-compatible endpoint — typically a self-hosted or corporate LLM
gateway. They share the same plumbing: a configurable base URL, an API
key, and the model suffix forwarded verbatim to the endpoint.
"""

LOCAL_MODEL_PROVIDERS: dict[str, dict[str, str]] = {
    "ollama/": {
        "base_url_env": "OLLAMA_BASE_URL",
        "base_url_default": "http://localhost:11434",
        "api_key_env": "OLLAMA_API_KEY",
    },
    "vllm/": {
        "base_url_env": "VLLM_BASE_URL",
        "base_url_default": "http://localhost:8000",
        "api_key_env": "VLLM_API_KEY",
    },
    "lm_studio/": {
        "base_url_env": "LMSTUDIO_BASE_URL",
        "base_url_default": "http://127.0.0.1:1234",
        "api_key_env": "LMSTUDIO_API_KEY",
    },
    "llamacpp/": {
        "base_url_env": "LLAMACPP_BASE_URL",
        "base_url_default": "http://localhost:8080",
        "api_key_env": "LLAMACPP_API_KEY",
    },
}

OPENAI_COMPAT_PREFIX = "openai-compat/"

# Generic escape hatch for self-hosted / corporate OpenAI-compatible gateways.
# No default base URL: there is no sensible guess, so the user has to say where
# the gateway lives.
OPENAI_COMPAT_PROVIDER: dict[str, str] = {
    "base_url_env": "OPENAI_COMPAT_BASE_URL",
    "base_url_default": "",
    "api_key_env": "OPENAI_COMPAT_API_KEY",
}

# Every endpoint we talk to directly rather than through HF Router. Same
# plumbing for all of them; they differ in whether they're a localhost server
# or a remote gateway, which matters for reasoning-effort handling.
DIRECT_ENDPOINT_PROVIDERS: dict[str, dict[str, str]] = {
    **LOCAL_MODEL_PROVIDERS,
    OPENAI_COMPAT_PREFIX: OPENAI_COMPAT_PROVIDER,
}

LOCAL_MODEL_PREFIXES = tuple(LOCAL_MODEL_PROVIDERS)
DIRECT_ENDPOINT_PREFIXES = tuple(DIRECT_ENDPOINT_PROVIDERS)
LOCAL_MODEL_BASE_URL_ENV = "LOCAL_LLM_BASE_URL"
LOCAL_MODEL_API_KEY_ENV = "LOCAL_LLM_API_KEY"
LOCAL_MODEL_API_KEY_DEFAULT = "sk-local-no-key-required"


def endpoint_provider(model_id: str) -> dict[str, str] | None:
    """Return provider config for a directly-addressed endpoint model id."""
    for prefix, config in DIRECT_ENDPOINT_PROVIDERS.items():
        if model_id.startswith(prefix):
            return config
    return None


def endpoint_model_name(model_id: str) -> str | None:
    """Return the backend model name with the endpoint prefix removed.

    Only the prefix is stripped: gateways namespace models by vendor, so
    ``openai-compat/vertex/claude-opus-5`` keeps ``vertex/claude-opus-5``.
    """
    for prefix in DIRECT_ENDPOINT_PREFIXES:
        if model_id.startswith(prefix):
            name = model_id[len(prefix) :]
            return name or None
    return None


def is_direct_endpoint_model_id(model_id: str) -> bool:
    """Return True for any id we send straight to a configured endpoint.

    These skip the HF Router entirely, so they don't need an ``HF_TOKEN`` and
    aren't looked up in the router catalog.
    """
    if not model_id or any(char.isspace() for char in model_id):
        return False
    return endpoint_model_name(model_id) is not None


def is_local_model_id(model_id: str) -> bool:
    """Return True for a localhost inference server id (not a gateway)."""
    return is_direct_endpoint_model_id(model_id) and model_id.startswith(
        LOCAL_MODEL_PREFIXES
    )


def is_openai_compat_model_id(model_id: str) -> bool:
    """Return True for a self-hosted / corporate gateway id."""
    return is_direct_endpoint_model_id(model_id) and model_id.startswith(
        OPENAI_COMPAT_PREFIX
    )
