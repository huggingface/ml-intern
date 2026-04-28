"""Shared helpers for local OpenAI-compatible model ids."""

LOCAL_MODEL_PREFIXES = ("ollama/", "vllm/", "llamacpp/", "local://")


def is_local_model_id(model_id: str) -> bool:
    return any(
        model_id.startswith(prefix) and len(model_id) > len(prefix)
        for prefix in LOCAL_MODEL_PREFIXES
    )
