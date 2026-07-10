"""Training-data selection strategies.

Strategy strings: "random" or "uncertainty:<metric_name>" where the metric is
registered in uq_pet.uncertainty.METRICS.
"""

import random


def select_top_uncertainty(scores: dict[str, float], n: int, seed: int) -> list[str]:
    """Top-n keys by uncertainty, most uncertain first.

    A seeded shuffle before the stable sort breaks ties randomly but
    reproducibly (sequence entropy saturates at log2(K), so ties are common).
    """
    keys = list(scores)
    random.Random(seed).shuffle(keys)
    keys.sort(key=lambda k: scores[k], reverse=True)
    return keys[:n]


def select_random(keys: list[str], n: int, seed: int) -> list[str]:
    return random.Random(seed).sample(list(keys), n)


def select(strategy: str, keys: list[str], scores: dict[str, float] | None,
           n: int, seed: int) -> list[str]:
    if strategy == "random":
        return select_random(keys, n, seed)
    if strategy.startswith("uncertainty:"):
        if scores is None:
            raise ValueError(f"Strategy '{strategy}' needs uncertainty scores.")
        # Ties are broken with a fixed seed so the selected subset is identical
        # across repeats; only the training seed varies for uncertainty cells.
        return select_top_uncertainty(scores, n, seed=0)
    raise ValueError(f"Unknown strategy '{strategy}'.")
