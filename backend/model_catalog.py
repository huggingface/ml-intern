"""Model catalog and validation helpers for agent API routes."""

import os
from typing import Any

from ml_intern.local_models import is_local_model_id


def local_models_enabled() -> bool:
    return os.environ.get("ENABLE_LOCAL_MODELS", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def get_available_models() -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = [
        {
            "id": "moonshotai/Kimi-K2.6",
            "label": "Kimi K2.6",
            "provider": "huggingface",
            "tier": "free",
            "recommended": True,
        },
        {
            "id": "bedrock/us.anthropic.claude-opus-4-6-v1",
            "label": "Claude Opus 4.6",
            "provider": "anthropic",
            "tier": "pro",
            "recommended": True,
        },
        {
            "id": "MiniMaxAI/MiniMax-M2.7",
            "label": "MiniMax M2.7",
            "provider": "huggingface",
            "tier": "free",
        },
        {
            "id": "zai-org/GLM-5.1",
            "label": "GLM 5.1",
            "provider": "huggingface",
            "tier": "free",
        },
    ]

    if local_models_enabled():
        models.extend(
            [
                {
                    "id": "ollama/llama3.1",
                    "label": "Llama 3.1 (Ollama)",
                    "provider": "local",
                    "tier": "free",
                },
                {
                    "id": "vllm/Qwen3.5-2B",
                    "label": "Qwen3.5-2B (vLLM)",
                    "provider": "local",
                    "tier": "free",
                },
                {
                    "id": "llamacpp/unsloth/Qwen3.5-2B",
                    "label": "Qwen3.5-2B (llama.cpp)",
                    "provider": "local",
                    "tier": "free",
                    "recommended": True,
                },
            ]
        )

    return models


def available_model_ids() -> set[str]:
    return {m["id"] for m in get_available_models()}


def is_custom_local_model_id(model_id: str) -> bool:
    if not local_models_enabled():
        return False
    if not isinstance(model_id, str):
        return False
    if not model_id or any(char.isspace() for char in model_id):
        return False
    return is_local_model_id(model_id)


def is_valid_model_id(model_id: str) -> bool:
    return model_id in available_model_ids() or is_custom_local_model_id(model_id)


def is_anthropic_model(model_id: str) -> bool:
    return model_id.startswith(("anthropic/", "bedrock/")) and "anthropic" in model_id
