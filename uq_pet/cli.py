"""Command-line entry points: score-pool | run | report."""

import argparse
import asyncio

from .config import ExperimentConfig, LLMScoreConfig, TrainConfig


def cmd_score_pool(args) -> None:
    from .data import split_pool_test
    from .llm_scoring import score_pool

    cfg = LLMScoreConfig(
        model=args.model,
        num_samples=args.k,
        temperature=args.temperature,
        seed=args.seed,
        max_concurrency=args.concurrency,
    )
    pool, _ = split_pool_test(seed=args.seed)
    cache = asyncio.run(score_pool(cfg, pool, limit=args.limit))
    print(f"Cache now holds {len(cache)} sentences at {cfg.cache_path()}")


def cmd_run(args) -> None:
    from .experiment import run_grid

    cfg = ExperimentConfig(
        budgets=args.budgets,
        strategies=args.strategies.split(","),
        repeats=args.repeats,
        llm=LLMScoreConfig(model=args.model, num_samples=args.k,
                           temperature=args.temperature, seed=args.seed),
        train=TrainConfig(checkpoint=args.checkpoint, epochs=args.epochs),
    )
    run_grid(cfg)


def cmd_report(args) -> None:
    from .reporting import build_report

    build_report()


def main() -> None:
    parser = argparse.ArgumentParser(prog="uq_pet", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_score = sub.add_parser("score-pool", help="LLM repeated-sampling pass over the pool")
    p_score.add_argument("--model", default="meta-llama/llama-3-8b-instruct")
    p_score.add_argument("--k", type=int, default=5)
    p_score.add_argument("--temperature", type=float, default=0.7)
    p_score.add_argument("--seed", type=int, default=3407)
    p_score.add_argument("--concurrency", type=int, default=8)
    p_score.add_argument("--limit", type=int, default=None,
                         help="score only the first N pool sentences (smoke test)")
    p_score.set_defaults(func=cmd_score_pool)

    p_run = sub.add_parser("run", help="run the selection/training grid")
    p_run.add_argument("--budgets", type=int, nargs="+", default=[10, 25, 50, 100])
    p_run.add_argument("--strategies", default=(
        "random,uncertainty:mean_token_entropy,"
        "uncertainty:sequence_entropy,uncertainty:jaccard_distance"
    ))
    p_run.add_argument("--repeats", type=int, default=5)
    p_run.add_argument("--model", default="meta-llama/llama-3-8b-instruct")
    p_run.add_argument("--k", type=int, default=5)
    p_run.add_argument("--temperature", type=float, default=0.7)
    p_run.add_argument("--seed", type=int, default=3407)
    p_run.add_argument("--checkpoint", default="distilbert-base-cased")
    p_run.add_argument("--epochs", type=int, default=20)
    p_run.set_defaults(func=cmd_run)

    p_report = sub.add_parser("report", help="figures + summary table from experiment runs")
    p_report.set_defaults(func=cmd_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
