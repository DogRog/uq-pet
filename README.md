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
  `uq_pet/uncertainty.py`): `sequence_entropy`, `mean_token_entropy`,
  `max_token_entropy`, `variation_ratio`, `jaccard_distance`.
- **Trained model**: `distilbert-base-cased` token classifier, fixed recipe for
  every cell (the selected data is the only variable), 5 seeds per cell.
- **Evaluation**: entity-level micro F1 (seqeval), per-type F1, token accuracy.

## Running

Requires `OPENROUTER_API_KEY` in `.env`.

```bash
uv sync

# 1. LLM sampling over the pool (~1,665 calls; cached + resumable in results/llm_scores/)
uv run python -m uq_pet.cli score-pool

# 2. Selection → training → evaluation grid (65 runs; resumable via results/experiment_runs.jsonl)
uv run python -m uq_pet.cli run

# 3. Figures + summary table → results/figures/
uv run python -m uq_pet.cli report
```

Uncertainty metrics are recomputed from the cached LLM samples, so budgets,
metrics and repeats can be swept without new API calls.

## Layout

| Path | Purpose |
| ---- | ------- |
| `uq_pet/config.py` | tags, prompts, paths, experiment dataclasses |
| `uq_pet/data.py` | PET loading, 80/20 pool/test split (seed 3407) |
| `uq_pet/llm_scoring.py` | prompt building, sampling, parsing, JSONL cache |
| `uq_pet/uncertainty.py` | pluggable uncertainty-metric registry |
| `uq_pet/selection.py` | top-uncertainty / random selection strategies |
| `uq_pet/train.py`, `uq_pet/evaluate.py` | fine-tuning (manual torch loop, MPS) + seqeval metrics |
| `uq_pet/experiment.py` | grid orchestration, resumable run log |
| `uq_pet/reporting.py` | learning curves, summary table, UQ-vs-error diagnostic |
| `pipeline.ipynb` | end-to-end pipeline walkthrough (cache-aware: reuses `results/` caches) |

## Reading the results

`results/figures/learning_curves.png` is the decision plot: if an uncertainty
metric's F1-vs-budget curve sits above the random curve beyond the ±std bands
consistently across budgets, uncertainty selection wins. Check
`uncertainty_vs_error.png` (does the metric track LLM difficulty at all?) and
the mean-sentence-length column in `summary.md` (unnormalized metrics favor
long sentences, which buys more tokens per budget — a known confound).
