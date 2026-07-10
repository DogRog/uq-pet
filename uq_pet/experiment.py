"""Grid orchestration: (budget x strategy x seed) -> train -> evaluate -> record.

Each run appends one JSON record to results/experiment_runs.jsonl; completed
cells are skipped on rerun, so the grid is resumable.
"""

import json
from datetime import datetime, timezone

from .config import RUNS_PATH, ExperimentConfig
from .data import sentence_key, split_pool_test
from .evaluate import evaluate_model_on
from .llm_scoring import load_cache
from .selection import select
from .train import train_token_classifier
from .uncertainty import METRICS


def cell_id(budget: int, strategy: str, seed: int) -> str:
    return f"budget={budget}|strategy={strategy}|seed={seed}"


def load_completed_runs() -> dict[str, dict]:
    completed = {}
    if RUNS_PATH.exists():
        with open(RUNS_PATH) as f:
            for line in f:
                record = json.loads(line)
                completed[cell_id(record["budget_pct"], record["strategy"], record["seed"])] = record
    return completed


def run_grid(cfg: ExperimentConfig) -> None:
    pool, test = split_pool_test(seed=cfg.llm.seed)
    pool_examples = list(pool)
    test_examples = list(test)
    by_key = {sentence_key(ex): ex for ex in pool_examples}
    all_keys = list(by_key)

    # Precompute uncertainty scores per metric from the LLM sample cache.
    metric_names = {
        s.split(":", 1)[1] for s in cfg.strategies if s.startswith("uncertainty:")
    }
    scores_by_metric: dict[str, dict[str, float]] = {}
    if metric_names:
        cache = load_cache(cfg.llm.cache_path())
        missing = [k for k in all_keys if k not in cache]
        if missing:
            raise RuntimeError(
                f"{len(missing)} pool sentences missing from LLM cache "
                f"{cfg.llm.cache_path()}; run `score-pool` first."
            )
        for name in metric_names:
            metric = METRICS[name]
            scores_by_metric[name] = {
                k: metric(cache[k]["parsed_samples"]) for k in all_keys
            }

    completed = load_completed_runs()
    RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Full pool (100%) is strategy-independent: run it once per seed as "full".
    grid = []
    for budget in cfg.budgets:
        strategies = ["full"] if budget >= 100 else cfg.strategies
        for strategy in strategies:
            for seed in range(cfg.repeats):
                grid.append((budget, strategy, seed))

    print(f"Grid: {len(grid)} cells ({len(completed)} already recorded)")
    with open(RUNS_PATH, "a") as runs_file:
        for i, (budget, strategy, seed) in enumerate(grid, start=1):
            cid = cell_id(budget, strategy, seed)
            if cid in completed:
                print(f"[{i}/{len(grid)}] skip (done): {cid}")
                continue

            n = round(len(all_keys) * budget / 100)
            if strategy == "full":
                selected_keys = all_keys
            else:
                metric_name = strategy.split(":", 1)[1] if ":" in strategy else None
                selected_keys = select(
                    strategy, all_keys,
                    scores_by_metric.get(metric_name), n, seed,
                )

            print(f"[{i}/{len(grid)}] {cid} -> {len(selected_keys)} sentences")
            selected = [by_key[k] for k in selected_keys]
            model, tokenizer = train_token_classifier(selected, cfg.train, seed)
            metrics = evaluate_model_on(model, tokenizer, test_examples, cfg.train)
            del model

            mean_len = sum(len(by_key[k]["tokens"]) for k in selected_keys) / len(selected_keys)
            record = {
                "budget_pct": budget,
                "n_selected": len(selected_keys),
                "strategy": strategy,
                "metric": strategy.split(":", 1)[1] if ":" in strategy else None,
                "seed": seed,
                "selected_keys": selected_keys,
                "selected_mean_tokens": mean_len,
                "metrics": metrics,
                "train_config": vars(cfg.train),
                "llm_config": {
                    "model": cfg.llm.model,
                    "num_samples": cfg.llm.num_samples,
                    "temperature": cfg.llm.temperature,
                    "seed": cfg.llm.seed,
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            runs_file.write(json.dumps(record) + "\n")
            runs_file.flush()
            print(f"    entity_f1={metrics['entity_f1']:.4f} "
                  f"token_acc={metrics['token_accuracy']:.4f}")

    print(f"Done. Records in {RUNS_PATH}")
