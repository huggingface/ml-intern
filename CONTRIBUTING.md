# Contributing to ml-intern

Thanks for your interest in contributing. This document covers development setup, code conventions, the PR workflow, and the review standards we apply.

## Table of Contents

- [Development Setup](#development-setup)
- [Before You Open a PR](#before-you-open-a-pr)
- [Branch and Commit Conventions](#branch-and-commit-conventions)
- [Code Style](#code-style)
- [Testing](#testing)
- [PR Types and Guidelines](#pr-types-and-guidelines)
- [Review Standards](#review-standards)
- [FAQ](#faq)

---

## Development Setup

```bash
git clone https://github.com/YOUR_USERNAME/ml-intern.git
cd ml-intern
uv sync
uv tool install -e .
```

Requires Python 3.11+ and [uv](https://github.com/astral-sh/uv).

Create a `.env` file in the project root with your API keys (see [README](./README.md) for the full list). At minimum you need `HF_TOKEN` to run the agent.

---

## Before You Open a PR

1. **For bug fixes** — reproduce the issue locally and confirm your fix resolves it.
2. **For new features or tools** — open an Issue first and describe what you want to add. Wait for a maintainer signal before investing significant implementation time. A PR without prior discussion is more likely to be declined on design grounds.
3. **For documentation changes** — no prior discussion required.

Read [REVIEW.md](./REVIEW.md) to understand how PRs are evaluated. The review is automated via Claude and uses the severity levels defined there (P0 / P1 / P2).

---

## Branch and Commit Conventions

Branch names follow a `type/description` pattern, matching existing branches in this repo:

| Change type | Prefix | Example |
|---|---|---|
| Bug fix | `fix/` | `fix/cli-model-loose-validation` |
| Feature | `feat/` | `feat/anthropic-prompt-caching` |
| Documentation | `docs/` | `docs/update-readme` |
| CI/tooling | `ci/` | `ci/review-md` |

Commit messages should be short and imperative (`fix: handle empty tool call list`, not `Fixed the thing`). No strict format is enforced, but be specific.

---

## Code Style

- **Python 3.11+** with type annotations throughout.
- Use `from __future__ import annotations` at the top of new files.
- Every public function needs a docstring. Document parameters and return type. See any file in `agent/tools/` for examples.
- Match the async patterns already in the codebase — handlers are `async def` and follow the `ToolSpec` pattern in `agent/core/tools.py`.
- Keep new tool definitions consistent with existing ones: a `TOOL_SPEC` constant plus a separate `handler` function, both imported and registered in `agent/core/tools.py`.

There is no auto-formatter configured in CI yet. Match the style of the file you're editing.

---

## Testing

Tests live in `tests/unit/`. The test runner is `pytest` with `asyncio_mode = auto` (see `pyproject.toml`).

```bash
# Install dev dependencies
uv sync --extra dev

# Run all unit tests
pytest tests/unit/

# Run a single file
pytest tests/unit/test_llm_params.py
```

**What to test:**

- New pure functions (no network, no filesystem) should have unit tests.
- Tool handlers that hit external services do not need unit tests, but include a brief note in the PR description about how you manually verified the behavior.
- If you're modifying the agent loop, doom-loop detector, or context manager, run the existing tests and confirm nothing regresses.

PRs that add meaningful new behavior without any test coverage are harder to merge. A small focused test is better than no test.

---

## PR Types and Guidelines

### Bug Fixes

Describe the bug in the PR body: what breaks, how to reproduce it, and what the fix does. Reference the relevant issue if one exists. No need to write a test if the behavior is difficult to unit-test, but explain how you verified the fix.

### Documentation

Typos, README improvements, docstring additions — welcome without prior discussion. Keep the scope small; one fix per PR is easier to review.

### New Tools

New tools must follow the `ToolSpec` pattern in `agent/core/tools.py`. Each tool needs:

- A `TOOL_SPEC` constant (name, description, JSON-schema parameters).
- An async handler function with a docstring.
- Registration in `create_builtin_tools()`.
- A note in the PR body about how you tested it end-to-end.

Open an Issue before starting implementation. The tool should serve ml-intern's core mission (researching papers, training models, shipping ML code in the HF ecosystem).

### Configuration Changes

Changes to `configs/cli_agent_config.json` or `configs/frontend_agent_config.json` should explain *why* in the PR description — what behavior changes, and why the new values are better. See PR #118 for an example.

### CI / Infrastructure

Changes to `.github/workflows/` or build tooling follow the same code review process. Be explicit about what the change enables or fixes.

---

## Review Standards

PRs are automatically reviewed by Claude using the rules in [REVIEW.md](./REVIEW.md). The key points:

- **P0** — blocks merge. Must be addressed before the PR can land.
- **P1** — worth fixing, not blocking. You can defer to a follow-up issue at your discretion.
- **P2** — informational. No action required.

The review philosophy is *rigor over speed*. Every behavior claim in a finding will cite a `file:line` reference. If you believe a P0 finding is incorrect, point to the code or test that contradicts it — the reviewer will reconsider. If you believe a P1 or P2 is not worth addressing, say so and move on.

Maintainers may add their own review on top of the automated one.

---

## FAQ

**My PR has been open for a few days with no response. What should I do?**

Leave a comment asking for feedback. The team is small and reviews sometimes get delayed.

**Can I submit a draft PR to get early feedback?**

Yes. The automated review only runs on non-draft PRs (see `.github/workflows/claude-review.yml`), so a draft is a good way to share work-in-progress without triggering a full review.

**My change touches `uv.lock`. Do I need to do anything special?**

Lock file changes are reviewed for provenance and correctness of the accompanying `pyproject.toml` change. Make sure the package version in `pyproject.toml` matches what you intended and include a brief note on why the new dependency is needed.

**I want to contribute but don't know where to start.**

Check the open Issues. Documentation improvements and small bug fixes are low-risk starting points and help you get familiar with the codebase before tackling larger features.
