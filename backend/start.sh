#!/bin/bash
# Entrypoint for HF Spaces dev mode compatibility.
# Dev mode spawns CMD multiple times simultaneously on restart.
# Only the first instance can bind port 7860 — the rest must exit
# with code 0 so the dev mode daemon doesn't mark the app as crashed.

# Run the retirement-only server; the original backend remains available in
# main.py for rollback but is not started by the hosted Space.
uvicorn retired_main:app --host 0.0.0.0 --port 7860
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    # Check if this was a port-in-use failure (another instance already running)
    echo "uvicorn exited with code $EXIT_CODE, exiting gracefully."
    exit 0
fi
