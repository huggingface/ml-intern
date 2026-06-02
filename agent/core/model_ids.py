"""Canonical model ids for HF Router inference."""

HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"

CLAUDE_OPUS_46_MODEL_ID = "anthropic/claude-opus-4.6:fal-ai"
CLAUDE_OPUS_47_MODEL_ID = "anthropic/claude-opus-4.7:fal-ai"
CLAUDE_OPUS_48_MODEL_ID = "anthropic/claude-opus-4.8:fal-ai"
CLAUDE_SONNET_46_MODEL_ID = "anthropic/claude-sonnet-4-6:fal-ai"
GPT_55_MODEL_ID = "openai/gpt-5.5:fal-ai"
KIMI_K26_MODEL_ID = "moonshotai/Kimi-K2.6"

DEFAULT_MODEL_ID = CLAUDE_OPUS_48_MODEL_ID

PREMIUM_MODEL_IDS = {
    CLAUDE_OPUS_46_MODEL_ID,
    CLAUDE_OPUS_47_MODEL_ID,
    CLAUDE_OPUS_48_MODEL_ID,
    CLAUDE_SONNET_46_MODEL_ID,
    GPT_55_MODEL_ID,
}


def strip_huggingface_model_prefix(model_id: str | None) -> str | None:
    """Return model ids without LiteLLM's optional ``huggingface/`` prefix."""
    if not model_id:
        return model_id
    return model_id.removeprefix("huggingface/")


def is_premium_model_id(model_id: str | None) -> bool:
    normalized = strip_huggingface_model_prefix(model_id)
    return bool(normalized and normalized in PREMIUM_MODEL_IDS)


def is_native_provider_model_id(model_id: str | None) -> bool:
    """Return True for non-router native provider ids that must be rejected."""
    if not model_id:
        return False
    stripped = strip_huggingface_model_prefix(model_id) or model_id
    if stripped.startswith("bedrock/"):
        return True
    if stripped.startswith("anthropic/") and ":" not in stripped:
        return True
    if stripped.startswith("openai/") and not stripped.startswith("openai/gpt-oss"):
        return ":" not in stripped
    return False
