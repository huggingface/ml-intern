from agent.tools import github_find_examples


class _FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code


def test_github_get_retries_public_request_when_token_is_rejected(monkeypatch):
    seen_auth = []

    def fake_get(url, headers, **kwargs):
        seen_auth.append(headers.get("Authorization"))
        return _FakeResponse(401 if headers.get("Authorization") else 200)

    monkeypatch.setattr(github_find_examples.requests, "get", fake_get)

    response = github_find_examples._github_get("https://api.github.com/repos/x/y", "bad")

    assert response.status_code == 200
    assert seen_auth == ["Bearer bad", None]


def test_excerpt_around_query_skips_unrelated_prefix():
    content = "license header\n" * 80 + "trainer = GRPOTrainer(args=config)\n"

    excerpt = github_find_examples._excerpt_around_query(
        content, "grpo trainer", max_chars=160
    )

    assert excerpt.startswith("...\n")
    assert "GRPOTrainer" in excerpt
    assert len(excerpt) < 220


def test_search_example_snippets_finds_content_only_match(monkeypatch):
    files = [
        {
            "path": "examples/scripts/sft.py",
            "ref": "abc123",
            "size": 240,
            "branch": "main",
            "url": "https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py",
        },
        {
            "path": "examples/scripts/dpo.py",
            "ref": "def456",
            "size": 120,
            "branch": "main",
            "url": "https://github.com/huggingface/trl/blob/main/examples/scripts/dpo.py",
        },
    ]

    def fake_fetch_file_content_cached(org, repo, file, token):
        if file["path"].endswith("sft.py"):
            return (
                "from trl import SFTConfig, SFTTrainer\n\n"
                "config = SFTConfig(dataset_text_field='text', packing=True)\n"
                "trainer = SFTTrainer(args=config)\n"
            )
        return "from trl import DPOTrainer\n"

    monkeypatch.setattr(
        github_find_examples,
        "_fetch_file_content_cached",
        fake_fetch_file_content_cached,
    )

    hits = github_find_examples._search_example_snippets(
        "dataset_text_field packing",
        "huggingface",
        "trl",
        files,
        "token",
        limit=3,
    )

    assert hits[0]["path"] == "examples/scripts/sft.py"
    assert hits[0]["line_start"] == "1"
    assert hits[0]["line_end"] == "4"
    assert "dataset_text_field" in hits[0]["content"]


def test_find_examples_promotes_snippet_hit_from_file_content(monkeypatch):
    files = [
        {
            "path": "examples/scripts/sft.py",
            "ref": "abc123",
            "size": 240,
            "branch": "main",
            "url": "https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py",
        }
    ]

    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setattr(
        github_find_examples,
        "_get_repo_tree",
        lambda org, repo, token: (files, ""),
    )
    monkeypatch.setattr(
        github_find_examples,
        "_search_example_snippets",
        lambda keyword, org, repo, files, token, limit: [
            {
                "path": "examples/scripts/sft.py",
                "url": "https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py",
                "ref": "abc123",
                "size": "240",
                "heading": "config = SFTConfig(...)",
                "content": "config = SFTConfig(dataset_text_field='text', packing=True)",
                "line_start": "3",
                "line_end": "3",
                "score": 1.5,
            }
        ],
    )

    result = github_find_examples.find_examples(
        keyword="dataset_text_field",
        repo="trl",
        org="huggingface",
        max_results=1,
        min_score=95,
    )

    assert not result.get("isError", False)
    assert "Best indexed code snippets" in result["formatted"]
    assert "'line_start': 3, 'line_end': 3" in result["formatted"]


def test_find_examples_allows_public_repo_without_github_token(monkeypatch):
    files = [
        {
            "path": "examples/scripts/sft.py",
            "ref": "abc123",
            "size": 240,
            "branch": "main",
            "url": "https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py",
        }
    ]

    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    def fake_get_repo_tree(org, repo, token):
        assert token == ""
        return files, ""

    monkeypatch.setattr(github_find_examples, "_get_repo_tree", fake_get_repo_tree)
    monkeypatch.setattr(
        github_find_examples,
        "_search_example_snippets",
        lambda keyword, org, repo, files, token, limit: [],
    )

    result = github_find_examples.find_examples(repo="trl", org="huggingface")

    assert not result.get("isError", False)
    assert "examples/scripts/sft.py" in result["formatted"]
