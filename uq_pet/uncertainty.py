"""Pluggable uncertainty metrics computed over repeated LLM samples.

Each metric takes `parsed_samples` — a list of K tag sequences (one per LLM
sample, all the same length) — and returns a scalar where higher = more
uncertain. Metrics register themselves in METRICS so the experiment can treat
the metric as a variable (`uncertainty:<name>` strategy strings).
"""

import math
from collections import Counter
from typing import Callable, Protocol

from datasets import Dataset

METRICS: dict[str, Callable[[list[list[str]]], float]] = {}


def register(name: str):
    def decorator(fn):
        METRICS[name] = fn
        return fn
    return decorator


def shannon_entropy(samples: list) -> float:
    """Shannon entropy (bits) over the frequencies of a list of hashable items."""
    if not samples:
        return 0.0
    counts = Counter(samples)
    total = len(samples)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def token_entropies(parsed_samples: list[list[str]]) -> list[float]:
    """Per-position entropy of the tags the K samples assigned to each token."""
    if not parsed_samples:
        return []
    length = len(parsed_samples[0])
    return [
        shannon_entropy([sample[i] for sample in parsed_samples])
        for i in range(length)
    ]


def majority_vote(parsed_samples: list[list[str]]) -> list[str]:
    """Most frequent tag per position across the K samples."""
    if not parsed_samples:
        return []
    length = len(parsed_samples[0])
    return [
        Counter(sample[i] for sample in parsed_samples).most_common(1)[0][0]
        for i in range(length)
    ]


@register("sequence_entropy")
def sequence_entropy(parsed_samples: list[list[str]]) -> float:
    """Entropy over exact full-sequence matches. Saturates at log2(K)."""
    return shannon_entropy([tuple(sample) for sample in parsed_samples])


@register("mean_token_entropy")
def mean_token_entropy(parsed_samples: list[list[str]]) -> float:
    entropies = token_entropies(parsed_samples)
    return sum(entropies) / len(entropies) if entropies else 0.0


@register("max_token_entropy")
def max_token_entropy(parsed_samples: list[list[str]]) -> float:
    entropies = token_entropies(parsed_samples)
    return max(entropies) if entropies else 0.0


@register("variation_ratio")
def variation_ratio(parsed_samples: list[list[str]]) -> float:
    """Mean over tokens of (1 - majority tag fraction)."""
    if not parsed_samples or not parsed_samples[0]:
        return 0.0
    k = len(parsed_samples)
    length = len(parsed_samples[0])
    ratios = []
    for i in range(length):
        top_count = Counter(sample[i] for sample in parsed_samples).most_common(1)[0][1]
        ratios.append(1.0 - top_count / k)
    return sum(ratios) / length


@register("jaccard_distance")
def jaccard_distance(parsed_samples: list[list[str]]) -> float:
    """Average pairwise Jaccard distance over (position, tag) sets."""
    if not parsed_samples or len(parsed_samples) < 2:
        return 0.0
    distances = []
    for i in range(len(parsed_samples)):
        set_i = set(enumerate(parsed_samples[i]))
        for j in range(i + 1, len(parsed_samples)):
            set_j = set(enumerate(parsed_samples[j]))
            union = len(set_i | set_j)
            jaccard_index = len(set_i & set_j) / union if union > 0 else 1.0
            distances.append(1.0 - jaccard_index)
    return sum(distances) / len(distances)


class UncertaintyScorer(Protocol):
    """Maps the experiment pool to a per-sentence uncertainty score."""

    def score(self, pool: Dataset) -> dict[str, float]: ...


class LLMSampleScorer:
    """Scores sentences from cached LLM samples using a registered metric."""

    def __init__(self, cache: dict[str, dict], metric_name: str):
        if metric_name not in METRICS:
            raise KeyError(f"Unknown metric '{metric_name}'. Available: {sorted(METRICS)}")
        self.cache = cache
        self.metric = METRICS[metric_name]

    def score(self, pool: Dataset) -> dict[str, float]:
        from .data import sentence_key

        scores = {}
        for example in pool:
            key = sentence_key(example)
            if key not in self.cache:
                raise KeyError(f"No cached LLM samples for '{key}'; run score-pool first.")
            scores[key] = self.metric(self.cache[key]["parsed_samples"])
        return scores


class ModelScorer:
    """Stretch goal: score sentences with a trained model's own uncertainty
    (e.g. logit entropy or MC dropout). Interface placeholder only."""

    def __init__(self, checkpoint: str, **kwargs):
        raise NotImplementedError("Model-based scoring is not implemented yet.")

    def score(self, pool: Dataset) -> dict[str, float]:
        raise NotImplementedError
