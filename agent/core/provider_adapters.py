"""Provider adapters for runtime params and model-name validation."""

import os
from dataclasses import dataclass
from typing import Any, ClassVar


class UnsupportedEffortError(ValueError):
    """The requested effort isn't valid for this provider's API surface.

    Raised synchronously before any network call so the probe cascade can
    skip levels the provider can't accept (e.g. ``max`` on HF router).
    """


def _has_model_suffix(model_name: str, prefix: str) -> bool:
    if not model_name.startswith(prefix):
        return False
    tail = model_name[len(prefix) :].split(":", 1)[0]
    return bool(tail) and all(tail.split("/"))


def _normalize_openai_api_base(api_base: str) -> str:
    base = api_base.rstrip("/")
    if base.endswith("/v1"):
        return base
    return f"{base}/v1"


def _all_adapter_prefixes() -> tuple[str, ...]:
    prefixes: list[str] = []
    for adapter in ADAPTERS:
        prefixes.extend(adapter.prefixes)
    return tuple(dict.fromkeys(prefixes))


def _is_hf_model_name(model_name: str) -> bool:
    if model_name.startswith(_all_adapter_prefixes()):
        return False
    bare = model_name.removeprefix("huggingface/").split(":", 1)[0]
    parts = bare.split("/")
    return len(parts) >= 2 and all(parts)


@dataclass(frozen=True)
class ProviderAdapter:
    provider_id: str
    prefixes: tuple[str, ...] = ()

    def matches(self, model_name: str) -> bool:
        return bool(self.prefixes) and model_name.startswith(self.prefixes)

    def build_params(
        self,
        model_name: str,
        *,
        session_hf_token: str | None = None,
        reasoning_effort: str | None = None,
        strict: bool = False,
    ) -> dict:
        raise NotImplementedError

    def allows_model_name(self, model_name: str) -> bool:
        return self.matches(model_name)


@dataclass(frozen=True)
class AnthropicAdapter(ProviderAdapter):
    """Anthropic models via native API (thinking + output_config.effort)."""

    prefixes: tuple[str, ...] = ("anthropic/",)
    _EFFORTS: ClassVar[frozenset[str]] = frozenset(
        {"low", "medium", "high", "xhigh", "max"}
    )

    def allows_model_name(self, model_name: str) -> bool:
        return _has_model_suffix(model_name, "anthropic/")

    def build_params(
        self,
        model_name: str,
        *,
        session_hf_token: str | None = None,
        reasoning_effort: str | None = None,
        strict: bool = False,
    ) -> dict:
        params: dict[str, Any] = {"model": model_name}
        if reasoning_effort:
            level = "low" if reasoning_effort == "minimal" else reasoning_effort
            if level not in self._EFFORTS:
                if strict:
                    raise UnsupportedEffortError(
                        f"Anthropic doesn't accept effort={level!r}"
                    )
            else:
                params["thinking"] = {"type": "adaptive"}
                params["output_config"] = {"effort": level}
        return params


@dataclass(frozen=True)
class OpenAIAdapter(ProviderAdapter):
    """OpenAI models via native API (reasoning_effort top-level kwarg)."""

    prefixes: tuple[str, ...] = ("openai/",)
    _EFFORTS: ClassVar[frozenset[str]] = frozenset({"minimal", "low", "medium", "high"})

    def allows_model_name(self, model_name: str) -> bool:
        return _has_model_suffix(model_name, "openai/")

    def build_params(
        self,
        model_name: str,
        *,
        session_hf_token: str | None = None,
        reasoning_effort: str | None = None,
        strict: bool = False,
    ) -> dict:
        params: dict[str, Any] = {"model": model_name}
        if reasoning_effort:
            if reasoning_effort not in self._EFFORTS:
                if strict:
                    raise UnsupportedEffortError(
                        f"OpenAI doesn't accept effort={reasoning_effort!r}"
                    )
            else:
                params["reasoning_effort"] = reasoning_effort
        return params


@dataclass(frozen=True)
class OpenAICompatAdapter(ProviderAdapter):
    api_base_url: str = ""
    api_key_env: str = ""
    default_api_key: str = ""
    supports_reasoning_effort: bool = True
    use_raw_model_name: bool = False

    def resolved_api_base(self) -> str:
        return _normalize_openai_api_base(self.api_base_url)

    def resolved_api_key(self) -> str | None:
        if self.api_key_env:
            return os.environ.get(self.api_key_env, self.default_api_key)
        return self.default_api_key or None

    def allows_model_name(self, model_name: str) -> bool:
        return bool(self.prefixes) and _has_model_suffix(model_name, self.prefixes[0])

    def build_params(
        self,
        model_name: str,
        *,
        session_hf_token: str | None = None,
        reasoning_effort: str | None = None,
        strict: bool = False,
    ) -> dict:
        del session_hf_token

        model_id = model_name.removeprefix(self.prefixes[0])
        params: dict[str, Any] = {
            "model": model_name if self.use_raw_model_name else f"openai/{model_id}",
            "api_base": self.resolved_api_base(),
            "api_key": self.resolved_api_key(),
        }

        if reasoning_effort:
            if not self.supports_reasoning_effort:
                if strict:
                    raise UnsupportedEffortError(
                        f"{self.provider_id} doesn't accept effort={reasoning_effort!r}"
                    )
            else:
                params["reasoning_effort"] = reasoning_effort

        return params


@dataclass(frozen=True)
class OllamaAdapter(OpenAICompatAdapter):
    prefixes: tuple[str, ...] = ("ollama/",)
    api_key_env: str = "OLLAMA_API_KEY"
    default_api_key: str = "ollama"
    supports_reasoning_effort: bool = False

    def resolved_api_base(self) -> str:
        return _normalize_openai_api_base(
            os.environ.get("OLLAMA_API_BASE", "http://localhost:11434/v1")
        )


@dataclass(frozen=True)
class LmStudioAdapter(OpenAICompatAdapter):
    prefixes: tuple[str, ...] = ("lm_studio/",)
    api_key_env: str = "LMSTUDIO_API_KEY"
    default_api_key: str = "lm-studio"
    supports_reasoning_effort: bool = False
    use_raw_model_name: bool = True

    def resolved_api_base(self) -> str:
        return _normalize_openai_api_base(
            os.environ.get("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
        )


@dataclass(frozen=True)
class VllmAdapter(OpenAICompatAdapter):
    prefixes: tuple[str, ...] = ("vllm/",)
    api_key_env: str = "VLLM_API_KEY"
    default_api_key: str = "vllm"
    supports_reasoning_effort: bool = False

    def resolved_api_base(self) -> str:
        return _normalize_openai_api_base(
            os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
        )


@dataclass(frozen=True)
class OpenRouterAdapter(OpenAICompatAdapter):
    prefixes: tuple[str, ...] = ("openrouter/",)
    api_base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"


@dataclass(frozen=True)
class OpenCodeZenAdapter(OpenAICompatAdapter):
    prefixes: tuple[str, ...] = ("opencode/",)
    api_base_url: str = "https://opencode.ai/zen/v1"
    api_key_env: str = "OPENCODE_ZEN_API_KEY"


@dataclass(frozen=True)
class OpenCodeGoAdapter(OpenAICompatAdapter):
    prefixes: tuple[str, ...] = ("opencode-go/",)
    api_base_url: str = "https://opencode.ai/zen/go/v1"
    api_key_env: str = "OPENCODE_GO_API_KEY"


@dataclass(frozen=True)
class GenericOpenAICompatAdapter(OpenAICompatAdapter):
    prefixes: tuple[str, ...] = ("openai-compat/",)
    api_key_env: str = "OPENAI_COMPAT_API_KEY"

    def resolved_api_base(self) -> str:
        api_base = os.environ.get("OPENAI_COMPAT_BASE_URL", "")
        if not api_base:
            raise ValueError("OPENAI_COMPAT_BASE_URL is required for openai-compat/")
        return _normalize_openai_api_base(api_base)


@dataclass(frozen=True)
class HfRouterAdapter(ProviderAdapter):
    """HuggingFace router — OpenAI-compat endpoint with HF token chain."""

    _EFFORTS: ClassVar[frozenset[str]] = frozenset({"low", "medium", "high"})

    def matches(self, model_name: str) -> bool:
        return _is_hf_model_name(model_name)

    def allows_model_name(self, model_name: str) -> bool:
        return _is_hf_model_name(model_name)

    def build_params(
        self,
        model_name: str,
        *,
        session_hf_token: str | None = None,
        reasoning_effort: str | None = None,
        strict: bool = False,
    ) -> dict:
        hf_model = model_name.removeprefix("huggingface/")
        inference_token = os.environ.get("INFERENCE_TOKEN")
        api_key = inference_token or session_hf_token or os.environ.get("HF_TOKEN")

        params: dict[str, Any] = {
            "model": f"openai/{hf_model}",
            "api_base": "https://router.huggingface.co/v1",
            "api_key": api_key,
        }

        if inference_token:
            bill_to = os.environ.get("HF_BILL_TO", "smolagents")
            params["extra_headers"] = {"X-HF-Bill-To": bill_to}

        if reasoning_effort:
            hf_level = "low" if reasoning_effort == "minimal" else reasoning_effort
            if hf_level not in self._EFFORTS:
                if strict:
                    raise UnsupportedEffortError(
                        f"HF router doesn't accept effort={hf_level!r}"
                    )
            else:
                params["extra_body"] = {"reasoning_effort": hf_level}

        return params


ADAPTERS: tuple[ProviderAdapter, ...] = (
    AnthropicAdapter(provider_id="anthropic"),
    OpenAIAdapter(provider_id="openai"),
    OllamaAdapter(provider_id="ollama"),
    LmStudioAdapter(provider_id="lm_studio"),
    VllmAdapter(provider_id="vllm"),
    OpenRouterAdapter(provider_id="openrouter"),
    OpenCodeZenAdapter(provider_id="opencode_zen"),
    OpenCodeGoAdapter(provider_id="opencode_go"),
    GenericOpenAICompatAdapter(provider_id="openai_compat"),
    HfRouterAdapter(provider_id="huggingface"),
)


def resolve_adapter(model_name: str) -> ProviderAdapter | None:
    for adapter in ADAPTERS:
        if adapter.matches(model_name):
            return adapter
    return None


def is_valid_model_name(model_name: str) -> bool:
    adapter = resolve_adapter(model_name)
    return adapter is not None and adapter.allows_model_name(model_name)
