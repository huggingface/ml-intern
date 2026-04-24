# Review instructions

These rules override the default review guidance. Treat them as the highest-priority
instruction block for any review of this repo. If something here contradicts a more
generic review habit, follow these.

## Default bias: merge

The goal of review is to catch things that would break production or leak cost,
not to gate every PR on a round trip. **Default bias is to merge.** Only 🔴
Important findings read as "fix before merge." 🟡 Nit and 🟣 Pre-existing are
informational — the author may defer them to a follow-up issue or a "fix-it" pass
at their discretion, and the review should not frame them as required changes.

If the author pushes back on a 🟡 or 🟣 without fixing it, accept the pushback —
do not re-flag it on subsequent commits. If Claude and the author repeatedly
disagree on the same class of finding, the signal is that REVIEW.md is missing a
rule; note it once in the PR summary as `suggest-rule: <short description>` and
stop.

## What "Important" means here

Reserve 🔴 Important for findings that would break production behavior, leak data or
cost, or break a rollback. For this repo that means:

- **LLM routing breakage** — changes that break LiteLLM calls or Bedrock inference
  profile routing (`bedrock/us.anthropic.claude-*` ids, `anthropic/`, `openai/`,
  HF router). Includes wrong `thinking` / `output_config` shape, wrong
  `reasoning_effort` cascade, and dropped prompt-cache markers on system prompt or
  tools.
- **Effort-probe regression** — changes to `agent/core/llm_params.py` or the effort
  probe cascade that silently drop thinking on specific models.
- **Auth / quota regression** — any change to the sandbox bearer-token guard, the
  Opus daily cap, or the HF-org gate that fails open, leaks Opus access to
  non-allowlisted orgs, or bypasses the daily cap. Fail-closed defaults are required.
- **Injection / SSRF** — unsanitized input flowing into `subprocess`, `bash -c`,
  URL fetches, or HTML render paths. Note: `bash -c "$user_input"` is still
  injection-vulnerable even inside `asyncio.create_subprocess_exec` — flag that as
  cosmetic if the PR claims it as a fix.
- **Agent-loop correctness** — broken streaming, lost `thinking_blocks` across tool
  turns, broken Ctrl-C handling, lost messages across compaction, session
  persistence that drops state on resume.
- **Backend/frontend contract drift** — FastAPI route signature changes without the
  matching React client update (or vice versa).

Everything else — style, naming, refactor suggestions, docstring polish, test
organization — is 🟡 Nit at most.

## Nit cap

Report at most **5** 🟡 Nits per review. If you found more, say "plus N similar
items" in the summary. If everything you found is a Nit, open the summary with
"No blocking issues."

## Re-review convergence

If this PR has already received a Claude review (there is a prior review comment
by the `claude` bot), suppress new Nit findings and post only 🔴 Important ones.
Do not re-post Nits that were already flagged on earlier commits. If the author
pushed a fix for a previously flagged issue, acknowledge it in one line rather
than re-flagging.

## Do not report

Anything in these paths — skip entirely:

- `frontend/node_modules/**`, `**/*.lock`, `uv.lock`, `package-lock.json`
- `hf_agent.egg-info/**`, `.ruff_cache/**`, `.pytest_cache/**`, `.venv/**`
- `session_logs/**`, `reports/**`
- Anything under a `gen/` or `generated/` path

Anything CI already enforces — skip entirely:

- Lint, formatting, import order (ruff covers it)
- Basic type errors (mypy / pyright covers it if it runs in CI)
- Spelling (out of scope unless the typo is in a user-facing string)

Anything speculative — do not post:

- "This might be slow" without a concrete complexity claim tied to a specific
  input size
- "Consider adding a test" without naming the specific behavior that is
  untested and would regress silently
- Hypothetical race conditions without a concrete interleaving

## Always check

- New provider / routing paths (`anthropic/`, `openai/`, `bedrock/`, any new
  prefix) are added to the `startswith` tuple in
  `agent/core/model_switcher.py::_print_hf_routing_info` so they bypass the HF
  router catalog lookup.
- New LLM calls pass through `agent/core/llm_params.py` so effort and caching
  are applied uniformly. Inline `litellm.acompletion` calls that bypass it are
  🔴 Important.
- New tools classified as destructive (writes to jobs, sandbox, filesystem)
  require approval; missing `approval_required` semantics is 🔴 Important.
- New backend routes that mutate state require the bearer-token / auth guard.
  Public routes that leak user input into logs are 🔴 Important.
- Changes to `agent/prompts/system_prompt_v*.yaml` — diff against the previous
  version and call out any **dropped rules** explicitly; an unintentionally
  removed guardrail is 🔴 Important.
- Changes to prompt-cache markers — the cache breakpoint on the system prompt
  and the tool block must stay intact. Breaking the cache silently is 🔴
  Important (cost regression).

## Verification bar

Every behavior claim in a finding must cite `file:line`. "This breaks X" is not
actionable without a line reference. If you cannot cite a line, do not post
the finding.

For routing / effort / caching claims specifically: cite both the call site and
the function in `llm_params.py` or `effort_probe.py` that handles it, so the
author can verify the chain end-to-end.

## Summary shape

Open the review body with a single-line tally:

- `3 important, 2 nits` if both, or
- `No blocking issues — 2 nits` if no Important, or
- `LGTM` if nothing at all.

Then a **What I checked** bullet list — one line per major area you examined,
regardless of whether you found anything. This gives the author visible coverage
even on a clean review, so "LGTM" carries weight instead of looking like a skim.
Example:

> What I checked:
> - LiteLLM/Bedrock routing in `llm_params.py`, `effort_probe.py`
> - Auth guard on the new `/sandbox/upload` route
> - Prompt-cache markers on the modified system prompt
> - Frontend message-bubble contract against the new `event_queue` shape

Then one paragraph of context at most. Everything else belongs in inline
comments.
