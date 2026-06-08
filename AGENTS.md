# Agent Notes

## Local Dev Servers

- Frontend: from `frontend/`, run `npm ci` if dependencies are missing, then `npm run dev`.
- Backend: from `backend/`, run `uv run uvicorn main:app --host ::1 --port 7860`.
- Frontend URL: http://localhost:5173/
- Backend health check: `curl -g http://[::1]:7860/api`
- Frontend proxy health check: `curl http://localhost:5173/api`

Notes:

- Vite proxies `/api` and `/auth` to `http://localhost:7860`.
- If `127.0.0.1:7860` is already owned by another local process, binding the backend to `::1` lets the Vite proxy resolve `localhost` cleanly.
- Prefer `npm ci` over `npm install` for setup, since `npm install` may rewrite `frontend/package-lock.json` metadata depending on npm version.
- Non-local LLM calls use `https://router.huggingface.co/v1` with the active Hugging Face user's token. Web sessions default to Kimi K2.6 for free users and Claude Opus 4.8 for Pro users; the CLI default is Claude Opus 4.8. For local development, set `HF_TOKEN` and optionally `ML_INTERN_DEFAULT_MODEL_ID`.
- When debugging a web app issue tied to a session ID, inspect the session data in `smolagents/ml-intern-sessions` for additional context.

## Development Checks

- Before every commit, run `uv run ruff check .` and `uv run ruff format --check .`.
- If formatting fails, run `uv run ruff format .`, then re-run the Ruff checks before committing.

## Git Workflow

- Before creating any new branch or worktree, switch to `main` and pull the latest changes.

## GitHub CLI

- Always use the `gh` CLI for GitHub operations such as opening, editing, inspecting, or commenting on PRs and issues.
- For multiline PR descriptions, prefer `gh pr edit <number> --body-file <file>` over inline `--body` so shell quoting, `$` env-var names, backticks, and newlines are preserved correctly.
- If `gh` reports an invalid token or auth failure, retry the command with `GH_TOKEN` and `GITHUB_TOKEN` unset, for example `env -u GH_TOKEN -u GITHUB_TOKEN gh pr create ...`, so `gh` can use the stored login token instead of a stale environment token.
- In Codex, sandboxed `gh` auth checks can report a valid keyring login as invalid when GitHub network access is restricted. Before telling the user to re-authenticate, retry with both env tokens unset and GitHub network access enabled.

## GitHub PRs

- Open code changes as GitHub PRs first. Do not push code changes directly to the Hugging Face Space deployment branch or Space remote before the PR has been opened, reviewed, and merged, unless the user explicitly asks to bypass the PR flow.
- After implementing a plan, run the required checks, commit the changes, open a GitHub PR, then start the backend and frontend local dev servers for testing.

## Hugging Face Space Deploys

- The Space remote is `space` and points to `https://huggingface.co/spaces/smolagents/ml-intern`.
- Deploy GitHub `main` to the Space from the local `space-main` branch by merging `origin/main` into `space-main` with a single merge commit, then pushing `space-main:main` to the `space` remote.
- Keep the Space-only README frontmatter on `space-main`; `.gitattributes` should contain `README.md merge=ours` and the local repo config should include `merge.ours.driver=true`.
- Local dev commonly uses a personal `HF_TOKEN`, but the deployed Space uses HF OAuth tokens. When adding Hub features, make sure the Space README `hf_oauth_scopes` frontmatter and the backend OAuth request in `backend/routes/auth.py` include the scopes required by the Hub APIs being called. A feature can work locally with a broad PAT and still fail in production with 403s if OAuth scopes are missing; after changing scopes, users may need to log out and log in again to receive a fresh token.
- Recommended deploy flow:

```bash
git pull --ff-only origin main
git switch space-main
git config merge.ours.driver true
git merge --no-ff origin/main -m "Deploy $(date +%Y-%m-%d)" \
  -m "Co-authored-by: OpenAI Codex <codex@openai.com>"
git push space space-main:main
git switch main
```
