#!/bin/bash
#SBATCH --job-name=autoresearch
#SBATCH --partition=hopper-prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=248000
#SBATCH --gres=gpu:h100:1
#SBATCH --time=16:00:00
#SBATCH --output=/fsx/aksel/autoresearch/logs/ar_%j.out
#SBATCH --error=/fsx/aksel/autoresearch/logs/ar_%j.err
#SBATCH --export=ALL

# Usage: sbatch run.sbatch [hours] [agent_config] [mode] [resume_run_id]
# Requires: ANTHROPIC_API_KEY in environment
#
# Modes:
#   vanilla  — run with just program.md (default)
#   guided   — append community findings + structured research loop
#
# Examples:
#   cd /fsx/aksel/autoresearch && sbatch slurm/run.sbatch 8 claude-opus-4-6
#   cd /fsx/aksel/autoresearch && sbatch slurm/run.sbatch 8 claude-opus-4-6 guided
#   cd /fsx/aksel/autoresearch && sbatch slurm/run.sbatch 1 claude-opus-4-6 guided 22070536  # resume
#
# Git history persists at /fsx/aksel/autoresearch/runs/run_<jobid>/repo/
# After job:
#   cd runs/run_<jobid>/repo && git log --oneline
#   cat runs/run_<jobid>/repo/results.tsv

set -euo pipefail

NUM_HOURS="${1:-8}"
AGENT_CONFIG="${2:-claude-opus-4-6}"
MODE="${3:-vanilla}"
RESUME_RUN="${4:-}"

# ── Paths ──
BASE_DIR="/fsx/aksel/autoresearch"
SHARED_CACHE="${BASE_DIR}/cache"   # persistent autoresearch data cache (shared across runs)
if [ -n "$RESUME_RUN" ]; then
    RUN_DIR="${BASE_DIR}/runs/run_${RESUME_RUN}"
    if [ ! -d "${RUN_DIR}/repo/.git" ]; then
        echo "ERROR: Cannot resume — ${RUN_DIR}/repo does not exist"
        exit 1
    fi
    echo "RESUMING run_${RESUME_RUN}"
else
    RUN_DIR="${BASE_DIR}/runs/run_${SLURM_JOB_ID}"
fi
REGISTRY="registry.hpc-cluster-hopper.hpc.internal.huggingface.tech/library"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-${REGISTRY}/posttrainbench:latest}"

mkdir -p "${BASE_DIR}/logs" "${SHARED_CACHE}" "${RUN_DIR}"

echo "Autoresearch | Hours: $NUM_HOURS | Config: $AGENT_CONFIG | Mode: $MODE | Job: $SLURM_JOB_ID"
echo "Run dir: $RUN_DIR (persistent on /fsx)"
echo "Shared cache: $SHARED_CACHE"
echo "Container: $CONTAINER_IMAGE"
echo "Node: $(hostname)"
nvidia-smi -L

# ── Fresh-run setup (clone + seed prompt) ──
if [ -z "$RESUME_RUN" ]; then
    git clone https://github.com/karpathy/autoresearch.git "${RUN_DIR}/repo" 2>/dev/null || {
        echo "WARN: git clone failed on host, will clone inside container"
    }
fi

# Build prompt from program.md if available, else defer to container
if [ -f "${RUN_DIR}/repo/program.md" ]; then
    PROMPT="$(cat "${RUN_DIR}/repo/program.md")

You have ${NUM_HOURS} hours of GPU time total. Check 'bash timer.sh' periodically
and wind down ~15 minutes before the deadline. The timer reports remaining time
as 'H:MM' on its own line."
else
    PROMPT="Read program.md and follow its instructions. You have ${NUM_HOURS} hours of GPU time total."
fi

if [ -n "$RESUME_RUN" ] && [ -f "${RUN_DIR}/repo/results.tsv" ]; then
    RESUME_CONTEXT="RESUMING A PREVIOUS RUN. The repo, branch, and results.tsv are intact. Pick up where you left off.

Your previous results.tsv:
$(cat "${RUN_DIR}/repo/results.tsv")

Current git log:
$(cd "${RUN_DIR}/repo" && git log --oneline -10)

Continue the experiment loop from here. Do NOT re-run the baseline or redo experiments already in results.tsv."

    PROMPT="${RESUME_CONTEXT}

${PROMPT}"
fi

if [ "$MODE" = "guided" ]; then
    GUIDED_FILE="${BASE_DIR}/slurm/guided_prompt.txt"
    if [ -f "$GUIDED_FILE" ]; then
        PROMPT="${PROMPT}

$(cat "$GUIDED_FILE")"
        echo "Mode: guided — appending community findings + research loop"
    else
        echo "WARNING: guided mode requested but ${GUIDED_FILE} not found, running vanilla"
    fi
fi

echo "$PROMPT" > "${RUN_DIR}/prompt.txt"

# ── Generate a wall-clock timer.sh for --resume reprompting ──
# The agent polls this with `bash timer.sh`. We set the deadline now based on
# NUM_HOURS so the reprompt loop knows when to stop firing new prompts.
#
# Retry credit: rate-limit / transient errors can eat minutes of backoff.
# agent_loop.py writes the total backoff seconds to $RETRY_CREDIT_FILE;
# timer.sh adds that credit to REMAINING_SEC so retry time doesn't count
# against the agent's productive budget. Capped by SLURM wall clock.
JOB_START_EPOCH=$(date +%s)
JOB_END_EPOCH=$((JOB_START_EPOCH + NUM_HOURS * 3600))
RETRY_CREDIT_FILE="/workspace/retry_credit"
cat > "${RUN_DIR}/timer.sh" <<EOF
#!/bin/bash
# Auto-generated timer for autoresearch run ${SLURM_JOB_ID}
# Prints remaining time as 'H:MM' on its own line, or 'expired'.
NOW=\$(date +%s)
REMAINING_SEC=\$((${JOB_END_EPOCH} - NOW))
# Credit retry-backoff time so rate-limit waits don't shrink the budget.
CREDIT_FILE="${RETRY_CREDIT_FILE}"
if [ -f "\$CREDIT_FILE" ]; then
    CREDIT=\$(cat "\$CREDIT_FILE" 2>/dev/null || echo 0)
    CREDIT=\${CREDIT:-0}
    REMAINING_SEC=\$((REMAINING_SEC + CREDIT))
fi
if [ \$REMAINING_SEC -le 0 ]; then
    echo "Remaining time:"
    echo "expired"
    exit 0
fi
HOURS=\$((REMAINING_SEC / 3600))
MINS=\$(( (REMAINING_SEC % 3600) / 60 ))
echo "Remaining time:"
printf '%d:%02d\n' \$HOURS \$MINS
EOF
chmod +x "${RUN_DIR}/timer.sh"

echo "================================"
echo "========= STARTING ============="
echo "================================"

START_TIME=$(date +%s)

# ── Run inside container ──
# Mounts:
#   /workspace      ← run dir (repo, prompt.txt, timer.sh, session_logs)
#   /root/.cache/autoresearch ← shared data cache (downloaded ONCE, reused)
#   /root/.cache/huggingface  ← HF cache (for any model downloads)
srun \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${RUN_DIR}:/workspace,${SHARED_CACHE}:/root/.cache/autoresearch,${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface" \
    --container-workdir="/workspace" \
    --no-container-mount-home \
    --container-writable \
    bash -c '
        set -euo pipefail
        export PATH="/root/.local/bin:$PATH"

        # ── Clone autoresearch if missing ──
        if [ ! -d /workspace/repo/.git ]; then
            git clone https://github.com/karpathy/autoresearch.git /workspace/repo
        fi

        # ── Setup autoresearch deps (in its own .venv, reused across runs) ──
        cd /workspace/repo
        ls pyproject.toml  # sanity check
        echo "3.11" > .python-version
        uv sync

        # ── Prepare data (tokenizer + shards). Fast no-op if cache is warm. ──
        uv run prepare.py

        # ── Git setup for experiment tracking ──
        git config user.name "autoresearch"
        git config user.email "autoresearch@cluster"
        if [ -z "'"${RESUME_RUN}"'" ]; then
            git checkout -b "autoresearch/run_'"${SLURM_JOB_ID}"'" || true
        fi

        # ── Copy timer.sh into the repo so the agent can call it as `bash timer.sh` ──
        cp /workspace/timer.sh ./timer.sh
        chmod +x ./timer.sh

        # ── Install ml-intern (headless agent) into system Python ──
        export ANTHROPIC_API_KEY="'"${ANTHROPIC_API_KEY:-}"'"
        export OPENAI_API_KEY="'"${OPENAI_API_KEY:-}"'"
        export HF_TOKEN="'"${HF_TOKEN:-}"'"
        # agent_loop.py appends rate-limit / transient retry backoff seconds
        # here; timer.sh reads the file and extends the deadline. Matches the
        # path baked into the generated timer.sh above.
        export RETRY_CREDIT_FILE="/workspace/retry_credit"
        unset VIRTUAL_ENV

        git clone --depth 1 --branch posttrain-bench \
            https://github.com/huggingface/ml-intern.git /tmp/hf_agent
        cd /tmp/hf_agent
        uv pip install --system -e .

        # ── Run agent in the autoresearch repo ──
        cd /workspace/repo
        PROMPT=$(cat /workspace/prompt.txt)

        MODEL_FLAG=""
        if [ -n "'"${AGENT_CONFIG}"'" ]; then
            MODEL_FLAG="--model anthropic/'"${AGENT_CONFIG}"'"
        fi

        # --resume keeps the agent session alive across turn_complete events
        # as long as the timer has >= 15 minutes left. Program.md says "NEVER
        # STOP" but agents sometimes hit finish_reason=stop; the reprompt loop
        # nudges them to continue iterating until the wall clock runs out.
        python -m agent.main \
            --max-iterations -1 \
            --no-stream \
            --resume timer.sh \
            --resume-min-minutes 15 \
            $MODEL_FLAG \
            "$PROMPT"
    ' > "${RUN_DIR}/agent_out.txt" 2>&1 || true

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
printf '%02d:%02d:%02d\n' $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60)) > "${RUN_DIR}/time_taken.txt"

echo "============================================"
echo "=== DONE ($(cat ${RUN_DIR}/time_taken.txt)) ==="
echo "============================================"
echo "Git history: ${RUN_DIR}/repo"
echo "Results TSV: ${RUN_DIR}/repo/results.tsv"
echo ""
if [ -f "${RUN_DIR}/repo/results.tsv" ]; then
    echo "=== results.tsv ==="
    cat "${RUN_DIR}/repo/results.tsv"
fi
