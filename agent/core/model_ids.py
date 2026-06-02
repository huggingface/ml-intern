"""Canonical model ids and legacy normalization for HF Router inference."""

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

LEGACY_MODEL_ID_MAP = {
    "bedrock/us.anthropic.claude-opus-4-6-v1": CLAUDE_OPUS_46_MODEL_ID,
    "bedrock/us.anthropic.claude-opus-4-8": CLAUDE_OPUS_48_MODEL_ID,
    "bedrock/us.anthropic.claude-sonnet-4-6": CLAUDE_SONNET_46_MODEL_ID,
    "anthropic/claude-opus-4-6": CLAUDE_OPUS_46_MODEL_ID,
    "anthropic/claude-opus-4-7": CLAUDE_OPUS_47_MODEL_ID,
    "anthropic/claude-opus-4-8": CLAUDE_OPUS_48_MODEL_ID,
    "anthropic/claude-sonnet-4-6": CLAUDE_SONNET_46_MODEL_ID,
    "openai/gpt-5.4": "openai/gpt-5.4:fal-ai",
    "openai/gpt-5.5": GPT_55_MODEL_ID,
}


def normalize_legacy_model_id(model_id: str | None) -> str | None:
    """Map known pre-router-only ids to their HF Router equivalents."""
    if not model_id:
        return model_id
    stripped = model_id.removeprefix("huggingface/")
    return LEGACY_MODEL_ID_MAP.get(stripped, stripped)


def is_premium_model_id(model_id: str | None) -> bool:
    normalized = normalize_legacy_model_id(model_id)
    return bool(normalized and normalized in PREMIUM_MODEL_IDS)


def is_legacy_native_model_id(model_id: str | None) -> bool:
    """Return True for old native provider ids that new selections must reject."""
    if not model_id:
        return False
    stripped = model_id.removeprefix("huggingface/")
    if stripped.startswith("bedrock/"):
        return True
    if stripped.startswith("anthropic/") and ":" not in stripped:
        return True
    if stripped.startswith("openai/") and not stripped.startswith("openai/gpt-oss"):
        return ":" not in stripped
    return False
