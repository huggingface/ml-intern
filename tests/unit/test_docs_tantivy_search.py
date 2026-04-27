import pytest

from agent.tools import docs_tools


@pytest.mark.asyncio
async def test_search_docs_returns_best_passage_with_source_lines():
    docs_tools._index_cache.clear()
    docs = [
        {
            "title": "SFT Trainer",
            "url": "https://huggingface.co/docs/trl/sft_trainer",
            "md_url": "https://huggingface.co/docs/trl/sft_trainer.md",
            "section": "trl",
            "glimpse": "",
            "content": (
                "# SFT Trainer\n\n"
                "Overview of supervised fine tuning.\n\n"
                "## Dataset processing\n\n"
                "Set dataset_text_field on SFTConfig when your dataset stores text "
                "in a custom column."
            ),
        },
        {
            "title": "DPO Trainer",
            "url": "https://huggingface.co/docs/trl/dpo_trainer",
            "md_url": "https://huggingface.co/docs/trl/dpo_trainer.md",
            "section": "trl",
            "glimpse": "",
            "content": "# DPO Trainer\n\nPreference optimization reference.",
        },
    ]

    results, note = await docs_tools._search_docs(
        "trl", docs, "dataset_text_field SFTConfig", 3
    )

    assert note is None
    assert results[0]["title"] == "SFT Trainer / Dataset processing"
    assert results[0]["line_start"] == "5"
    assert results[0]["line_end"] == "7"
    assert "dataset_text_field" in results[0]["glimpse"]


@pytest.mark.asyncio
async def test_explore_hf_docs_allows_public_docs_without_token(monkeypatch):
    async def fake_get_docs(hf_token, endpoint):
        assert hf_token == ""
        assert endpoint == "trl"
        return [
            {
                "title": "SFT Trainer",
                "url": "https://huggingface.co/docs/trl/sft_trainer",
                "md_url": "https://huggingface.co/docs/trl/sft_trainer.md",
                "section": "trl",
                "glimpse": "Use SFTConfig with dataset_text_field.",
                "content": "# SFT Trainer\n\nUse SFTConfig with dataset_text_field.",
            }
        ]

    monkeypatch.setattr(docs_tools, "_get_docs", fake_get_docs)

    text, ok = await docs_tools.explore_hf_docs_handler(
        {"endpoint": "trl", "query": "dataset_text_field", "max_results": 1},
        session=None,
    )

    assert ok is True
    assert "SFT Trainer" in text
    assert "Lines:" in text


@pytest.mark.asyncio
async def test_search_openapi_uses_tantivy_index(monkeypatch):
    docs_tools._openapi_cache = None
    docs_tools._openapi_index_cache = None

    async def fake_fetch_openapi_spec():
        return {
            "servers": [{"url": "https://huggingface.co"}],
            "paths": {
                "/api/repos/create": {
                    "post": {
                        "operationId": "createRepo",
                        "summary": "Create a repository",
                        "description": "Create model, dataset, or Space repositories.",
                        "tags": ["Repo"],
                        "parameters": [
                            {"name": "name", "in": "query", "schema": {"type": "string"}}
                        ],
                        "responses": {"200": {"description": "Created"}},
                    }
                },
                "/api/models/{repo_id}": {
                    "get": {
                        "operationId": "modelInfo",
                        "summary": "Get model info",
                        "description": "Retrieve model metadata.",
                        "tags": ["Model"],
                        "parameters": [],
                        "responses": {"200": {"description": "OK"}},
                    }
                },
            },
        }

    monkeypatch.setattr(docs_tools, "_fetch_openapi_spec", fake_fetch_openapi_spec)

    results, note = await docs_tools._search_openapi("create repository", "Repo", 5)

    assert note is None
    assert len(results) == 1
    assert results[0]["path"] == "/api/repos/create"
    assert results[0]["operationId"] == "createRepo"
