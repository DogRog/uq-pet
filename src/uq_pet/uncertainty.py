"""Pluggable uncertainty metrics computed over repeated LLM samples.

Every metric returns a scalar where higher = more uncertain, but they split by
what they may observe:
- black-box metrics take `parsed_samples` — a list of K tag sequences (one per
  LLM sample) — and measure disagreement between samples;
- white-box metrics take the full cache record and read the model's internal
  signal (`token_entropies`, only present in caches from a local backend).

Metrics register themselves in METRICS with their box type so the experiment
can treat the metric as a variable (`uncertainty:<name>` strategy strings) and
`compute_metric` dispatches the right input.
"""

import math
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Protocol

from datasets import Dataset


@dataclass(frozen=True)
class Metric:
    fn: Callable[..., float]
    box: str  # "black": fn(parsed_samples) | "white": fn(record)


METRICS: dict[str, Metric] = {}


def register(name: str, box: str = "black"):
    def decorator(fn):
        METRICS[name] = Metric(fn, box)
        return fn
    return decorator


class WhiteboxDataMissingError(ValueError):
    """A white-box metric was asked to score a record without model internals."""


def compute_metric(name: str, record: dict) -> float:
    """Score one cache record with a registered metric, black- or white-box."""
    if name not in METRICS:
        raise KeyError(f"Unknown metric '{name}'. Available: {sorted(METRICS)}")
    metric = METRICS[name]
    if metric.box == "white":
        return metric.fn(record)
    return metric.fn(record["parsed_samples"])


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


@register("predictive_entropy", box="white")
def predictive_entropy(record: dict) -> float:
    """Mean token predictive entropy (bits) from the model's own per-step
    distributions: mean over generated tokens per sample, mean over K samples."""
    token_entropies = record.get("token_entropies")
    if not token_entropies:
        raise WhiteboxDataMissingError(
            f"Record '{record.get('key')}' has no token_entropies; white-box "
            "metrics need a cache produced by a local backend (llm.backend: mlx). "
            "Re-run score-pool with an mlx config."
        )
    per_sample = [sum(ents) / len(ents) for ents in token_entropies if ents]
    return sum(per_sample) / len(per_sample) if per_sample else 0.0


class UncertaintyScorer(Protocol):
    """Maps the experiment pool to a per-sentence uncertainty score."""

    def score(self, pool: Dataset) -> dict[str, float]: ...


class LLMSampleScorer:
    """Scores sentences from cached LLM samples using a registered metric."""

    def __init__(self, cache: dict[str, dict], metric_name: str):
        if metric_name not in METRICS:
            raise KeyError(f"Unknown metric '{metric_name}'. Available: {sorted(METRICS)}")
        self.cache = cache
        self.metric_name = metric_name

    def score(self, pool: Dataset) -> dict[str, float]:
        from .data import sentence_key

        scores = {}
        for example in pool:
            key = sentence_key(example)
            if key not in self.cache:
                raise KeyError(f"No cached LLM samples for '{key}'; run score-pool first.")
            scores[key] = compute_metric(self.metric_name, self.cache[key])
        return scores


class ModelScorer:
    """Stretch goal: score sentences with a trained model's own uncertainty
    (e.g. logit entropy or MC dropout). Interface placeholder only."""

    def __init__(self, checkpoint: str, **kwargs):
        raise NotImplementedError("Model-based scoring is not implemented yet.")

    def score(self, pool: Dataset) -> dict[str, float]:
        raise NotImplementedError
