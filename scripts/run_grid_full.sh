#!/usr/bin/env bash
# End-to-end pipeline for the full grid (configs/grid_full.yaml):
#   download data -> LLM score-pool -> selection/training grid -> report.
#
# Usage:
#   scripts/run_grid_full.sh                 # start a new run
#   scripts/run_grid_full.sh --resume RUN_ID # continue an interrupted run
set -euo pipefail

cd "$(dirname "$0")/.."
CONFIG=configs/grid_full.yaml

RESUME=""
if [[ "${1:-}" == "--resume" ]]; then
    RESUME="${2:?usage: $0 --resume RUN_ID}"
elif [[ $# -gt 0 ]]; then
    echo "usage: $0 [--resume RUN_ID]" >&2
    exit 1
fi

if [[ ! -f .env ]] || ! grep -q OPENROUTER_API_KEY .env; then
    echo "OPENROUTER_API_KEY not found in .env (needed for score-pool)" >&2
    exit 1
fi

uv sync

if [[ ! -d data/raw ]] || [[ -z "$(ls -A data/raw 2>/dev/null)" ]]; then
    uv run uq-pet download-data
fi

# Cached + resumable; a no-op once every pool sentence is scored.
uv run uq-pet score-pool --config "$CONFIG"

if [[ -n "$RESUME" ]]; then
    uv run uq-pet run --resume "$RESUME"
    uv run uq-pet report --run-id "$RESUME"
else
    uv run uq-pet run --config "$CONFIG"
    uv run uq-pet report
fi
