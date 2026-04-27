from agent.core.llm_params import UnsupportedEffortError, _resolve_llm_params


def test_openai_xhigh_effort_is_forwarded():
    params = _resolve_llm_params(
        "openai/gpt-5.5",
        reasoning_effort="xhigh",
        strict=True,
    )

    assert params["model"] == "openai/gpt-5.5"
    assert params["reasoning_effort"] == "xhigh"


def test_openai_max_effort_is_still_rejected():
    try:
        _resolve_llm_params(
            "openai/gpt-5.4",
            reasoning_effort="max",
            strict=True,
        )
    except UnsupportedEffortError as exc:
        assert "OpenAI doesn't accept effort='max'" in str(exc)
    else:
        raise AssertionError("Expected UnsupportedEffortError for max effort")


def test_gemini_effort_is_forwarded_for_google_ai_studio():
    params = _resolve_llm_params(
        "google/gemini-3.1-pro-preview",
        reasoning_effort="high",
        strict=True,
    )

    assert params == {
        "model": "gemini/gemini-3.1-pro-preview",
        "reasoning_effort": "high",
    }


def test_gemini_accepts_preview_model_ids_without_a_catalog_entry():
    params = _resolve_llm_params(
        "google/gemini-2.5-flash-lite-preview-09-2025",
        reasoning_effort="minimal",
        strict=True,
    )

    assert params["model"] == "gemini/gemini-2.5-flash-lite-preview-09-2025"
    assert params["reasoning_effort"] == "minimal"


def test_gemini_rejects_efforts_litellm_cannot_map():
    try:
        _resolve_llm_params(
            "google/gemini-2.5-pro",
            reasoning_effort="xhigh",
            strict=True,
        )
    except UnsupportedEffortError as exc:
        assert "Gemini doesn't accept effort='xhigh'" in str(exc)
    else:
        raise AssertionError("Expected UnsupportedEffortError for xhigh effort")


def test_vertex_ai_effort_is_forwarded():
    params = _resolve_llm_params(
        "google-geap/gemini-3-flash-preview",
        reasoning_effort="medium",
        strict=True,
    )

    assert params == {
        "model": "vertex_ai/gemini-3-flash-preview",
        "reasoning_effort": "medium",
    }


def test_vertex_ai_accepts_preview_model_ids_without_a_catalog_entry():
    params = _resolve_llm_params(
        "google-geap/gemini-2.5-flash-preview-09-2025",
        reasoning_effort="low",
        strict=True,
    )

    assert params["model"] == "vertex_ai/gemini-2.5-flash-preview-09-2025"
    assert params["reasoning_effort"] == "low"


def test_vertex_ai_rejects_efforts_litellm_cannot_map():
    try:
        _resolve_llm_params(
            "google-geap/gemini-2.5-pro",
            reasoning_effort="max",
            strict=True,
        )
    except UnsupportedEffortError as exc:
        assert "Vertex AI doesn't accept effort='max'" in str(exc)
    else:
        raise AssertionError("Expected UnsupportedEffortError for max effort")
