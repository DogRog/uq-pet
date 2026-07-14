import math

import pytest

from uq_pet.uncertainty import (
    METRICS,
    WhiteboxDataMissingError,
    compute_metric,
    jaccard_distance,
    majority_vote,
    max_token_entropy,
    mean_token_entropy,
    predictive_entropy,
    sequence_entropy,
    shannon_entropy,
    variation_ratio,
)


def test_shannon_entropy_uniform_and_constant():
    assert shannon_entropy(["a", "b"]) == 1.0
    assert shannon_entropy(["a", "a", "a"]) == 0.0
    assert shannon_entropy([]) == 0.0


def test_sequence_entropy_identical_is_zero():
    samples = [["O", "B-Actor"]] * 5
    assert sequence_entropy(samples) == 0.0


def test_sequence_entropy_all_distinct_is_log2k():
    samples = [["O"], ["B-Actor"], ["I-Actor"], ["B-Activity"]]
    assert sequence_entropy(samples) == math.log2(4)


def test_mean_token_entropy():
    # position 0 agrees (entropy 0), position 1 splits 50/50 (entropy 1)
    samples = [["O", "B-Actor"], ["O", "I-Actor"]]
    assert mean_token_entropy(samples) == 0.5
    assert mean_token_entropy([]) == 0.0


def test_max_token_entropy():
    samples = [["O", "B-Actor"], ["O", "I-Actor"]]
    assert max_token_entropy(samples) == 1.0
    assert max_token_entropy([]) == 0.0


def test_variation_ratio():
    # position 0: majority 2/2 -> 0.0; position 1: majority 1/2 -> 0.5
    samples = [["O", "B-Actor"], ["O", "I-Actor"]]
    assert variation_ratio(samples) == 0.25
    assert variation_ratio([]) == 0.0


def test_jaccard_distance_identical_zero():
    samples = [["O", "B-Actor"], ["O", "B-Actor"]]
    assert jaccard_distance(samples) == 0.0
    assert jaccard_distance([["O"]]) == 0.0  # fewer than 2 samples


def test_jaccard_distance_disjoint_one():
    samples = [["O", "O"], ["B-Actor", "I-Actor"]]
    assert jaccard_distance(samples) == 1.0


def test_majority_vote():
    samples = [["O", "B-Actor"], ["O", "B-Actor"], ["B-Actor", "O"]]
    assert majority_vote(samples) == ["O", "B-Actor"]
    assert majority_vote([]) == []


def test_metrics_registry_names_and_box_types():
    assert set(METRICS) == {
        "sequence_entropy",
        "mean_token_entropy",
        "max_token_entropy",
        "variation_ratio",
        "jaccard_distance",
        "predictive_entropy",
    }
    assert METRICS["predictive_entropy"].box == "white"
    black = set(METRICS) - {"predictive_entropy"}
    assert all(METRICS[name].box == "black" for name in black)


def test_predictive_entropy_means_over_tokens_then_samples():
    record = {"token_entropies": [[1.0, 3.0], [2.0, 2.0]]}
    assert predictive_entropy(record) == 2.0


def test_predictive_entropy_missing_data_raises():
    with pytest.raises(WhiteboxDataMissingError):
        predictive_entropy({"key": "doc-0", "parsed_samples": [["O"]]})
    with pytest.raises(WhiteboxDataMissingError):
        predictive_entropy({"token_entropies": []})


def test_compute_metric_dispatches_by_box_type():
    record = {
        "parsed_samples": [["O", "B-Actor"], ["O", "I-Actor"]],
        "token_entropies": [[1.0], [3.0]],
    }
    assert compute_metric("mean_token_entropy", record) == 0.5
    assert compute_metric("predictive_entropy", record) == 2.0
    with pytest.raises(KeyError):
        compute_metric("nope", record)
