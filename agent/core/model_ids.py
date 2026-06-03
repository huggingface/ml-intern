"""Canonical model ids for HF Router inference."""

HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"

# Keep these as verbatim HF Router ids; version punctuation differs by model.
CLAUDE_OPUS_48_MODEL_ID = "anthropic/claude-opus-4.8:fal-ai"
GPT_55_MODEL_ID = "openai/gpt-5.5:fal-ai"
KIMI_K26_MODEL_ID = "moonshotai/Kimi-K2.6"
MINIMAX_M27_MODEL_ID = "MiniMaxAI/MiniMax-M2.7"
GLM_51_MODEL_ID = "zai-org/GLM-5.1"
DEEPSEEK_V4_PRO_MODEL_ID = "deepseek-ai/DeepSeek-V4-Pro:deepinfra"

DEFAULT_MODEL_ID = KIMI_K26_MODEL_ID

PREMIUM_MODEL_IDS = {
    CLAUDE_OPUS_48_MODEL_ID,
    GPT_55_MODEL_ID,
}

PRO_ONLY_PREMIUM_MODEL_IDS = {
    CLAUDE_OPUS_48_MODEL_ID,
    GPT_55_MODEL_ID,
}

KNOWN_ROUTER_MODEL_IDS = PREMIUM_MODEL_IDS | {
    KIMI_K26_MODEL_ID,
    MINIMAX_M27_MODEL_ID,
    GLM_51_MODEL_ID,
    DEEPSEEK_V4_PRO_MODEL_ID,
}


def strip_huggingface_model_prefix(model_id: str | None) -> str | None:
    """Return model ids without LiteLLM's optional ``huggingface/`` prefix."""
    if not model_id:
        return model_id
    return model_id.removeprefix("huggingface/")


def is_premium_model_id(model_id: str | None) -> bool:
    normalized = strip_huggingface_model_prefix(model_id)
    return bool(normalized and normalized in PREMIUM_MODEL_IDS)


def is_pro_only_premium_model_id(model_id: str | None) -> bool:
    normalized = strip_huggingface_model_prefix(model_id)
    return bool(normalized and normalized in PRO_ONLY_PREMIUM_MODEL_IDS)


def is_known_router_model_id(model_id: str | None) -> bool:
    normalized = strip_huggingface_model_prefix(model_id)
    return bool(normalized and normalized in KNOWN_ROUTER_MODEL_IDS)
