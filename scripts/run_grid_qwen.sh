#!/usr/bin/env bash
# End-to-end pipeline for the local Qwen grid (configs/grid_qwen.yaml):
#   download data -> in-process mlx scoring -> selection/training grid -> report.
#
# No API key needed: the model runs locally via mlx_lm. The first run downloads
# ~4.5 GB of Qwen3-8B-4bit weights from Hugging Face. The scoring pass is long
# (~8,000 generations) but Ctrl-C-safe: it resumes from the cache.
#
# Usage:
#   scripts/run_grid_qwen.sh                 # start a new run
#   scripts/run_grid_qwen.sh --resume RUN_ID # continue an interrupted run
set -euo pipefail

cd "$(dirname "$0")/.."
CONFIG=configs/grid_qwen.yaml

RESUME=""
if [[ "${1:-}" == "--resume" ]]; then
    RESUME="${2:?usage: $0 --resume RUN_ID}"
elif [[ $# -gt 0 ]]; then
    echo "usage: $0 [--resume RUN_ID]" >&2
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
