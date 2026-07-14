# uq-pet

Is **LLM uncertainty quantification a good criterion for choosing training data**, compared to random choice? This repo runs that experiment on the [PET dataset](https://github.com/patriziobellan86/PETv1.1) (process-extraction NER, 417 sentences, 15 BIO tags).

## Experiment design

```text
PET NER dataset
├── 80% experiment pool (333 sentences)
│     │  LLM repeated sampling (llama-3-8b, K=5, temp 0.7) → uncertainty score per sentence
│     ├── top-N% most uncertain  → fine-tune transformer → evaluate
│     └── random N%              → fine-tune transformer → evaluate
└── 20% held-out test (84 sentences) ──────────────────────→ used for both evaluations
```

- **Budgets**: N ∈ {10%, 25%, 50%} + a 100% full-pool reference.
- **Uncertainty metrics** (the metric is itself an experimental variable, see
  `src/uq_pet/uncertainty.py`). Two families:
  - *black-box* — disagreement between the K sampled tag sequences, works with
    any API backend: `sequence_entropy`, `mean_token_entropy`,
    `max_token_entropy`, `variation_ratio`, `jaccard_distance`;
  - *white-box* — the model's own per-token predictive entropy, needs the
    in-process mlx backend which caches `token_entropies`:
    `predictive_entropy`.
- **Trained model**: `distilbert-base-cased` token classifier, fixed recipe for
  every cell (the selected data is the only variable), 5 seeds per cell.
- **Evaluation**: entity-level micro F1 (seqeval), per-type F1, token accuracy.

## Running

Experiments are defined as YAML files in `configs/` (`grid_full.yaml` is the
API sweep; `grid_qwen.yaml` scores with a local Qwen3-8B via mlx and adds the
white-box strategy; `smoke.yaml` is a 2-cell sanity run). API scoring
(`llm.backend: openrouter`) requires `OPENROUTER_API_KEY` in `.env`; local
scoring (`llm.backend: mlx`) needs no key but downloads the model weights on
first run.

```bash
uv sync
uv run pytest                     # offline unit tests (metrics, selection, parsing, config)

# 0. Download the PET dataset → data/raw/
uv run uq-pet download-data

# 1. LLM sampling over the pool (~1,665 calls; cached + resumable in data/processed/llm_scores/)
uv run uq-pet score-pool --config configs/grid_full.yaml

# 2. Selection → training → evaluation grid (65 cells) → results/<run_id>/
uv run uq-pet run --config configs/grid_full.yaml
uv run uq-pet run --resume <run_id>          # continue an interrupted run

# 3. Figures + summary table → results/<run_id>/figures/
uv run uq-pet report                         # defaults to the latest run
uv run uq-pet report --run-id <run_id>
```

`scripts/run_grid_full.sh` and `scripts/run_grid_qwen.sh` chain the full
pipeline (download → score-pool → run → report) for the respective config,
with a `--resume RUN_ID` passthrough.

Each run directory `results/<run_id>/` holds a `config.yaml` snapshot,
per-cell `records.jsonl`, an aggregated `metrics.json`, `run.log`, and the
report's `figures/`. Uncertainty metrics are recomputed from the cached LLM
samples, so budgets, metrics and repeats can be swept without new API calls.

## Layout

| Path | Purpose |
| ---- | ------- |
| `configs/` | YAML experiment definitions, one per run/sweep |
| `data/raw/` | downloaded PET jsonl (gitignored, never edited by hand) |
| `data/processed/llm_scores/` | cached LLM samples, shared across runs (gitignored) |
| `src/uq_pet/config.py` | tags, prompts, project paths, config dataclasses + YAML loader |
| `src/uq_pet/data.py` | PET download/loading, 80/20 pool/test split (seed 3407) |
| `src/uq_pet/llm_scoring.py` | prompt building, sampling, parsing, JSONL cache |
| `src/uq_pet/uncertainty.py` | pluggable uncertainty-metric registry |
| `src/uq_pet/selection.py` | top-uncertainty / random selection strategies |
| `src/uq_pet/train.py`, `src/uq_pet/evaluate.py` | fine-tuning (manual torch loop, MPS) + seqeval metrics |
| `src/uq_pet/experiment.py` | grid orchestration, per-run dirs, resumable records |
| `src/uq_pet/reporting.py` | learning curves, summary table, UQ-vs-error diagnostic |
| `scripts/` | runnable entry points wrapping the `uq-pet` CLI |
| `results/<run_id>/` | one directory per run (gitignored) |
| `notebooks/pipeline.ipynb` | end-to-end walkthrough (cache-aware: reuses the score cache) |
| `tests/` | offline unit tests |

## Reading the results

`results/<run_id>/figures/learning_curves.png` is the decision plot: if an uncertainty
metric's F1-vs-budget curve sits above the random curve beyond the ±std bands
consistently across budgets, uncertainty selection wins. Check
`uncertainty_vs_error.png` (does the metric track LLM difficulty at all?) and
the mean-sentence-length column in `summary.md` (unnormalized metrics favor
long sentences, which buys more tokens per budget — a known confound).
