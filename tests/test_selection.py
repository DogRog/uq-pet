import pytest

from uq_pet.uncertainty import select, select_random, select_top_uncertainty

KEYS = [f"doc::{i}" for i in range(20)]


def test_select_random_deterministic_with_seed():
    assert select_random(KEYS, 5, seed=7) == select_random(KEYS, 5, seed=7)


def test_select_random_returns_n_unique_keys():
    picked = select_random(KEYS, 8, seed=1)
    assert len(picked) == 8
    assert len(set(picked)) == 8
    assert set(picked) <= set(KEYS)


def test_select_top_uncertainty_ordering():
    scores = {"a": 0.1, "b": 0.9, "c": 0.5}
    assert select_top_uncertainty(scores, 2, seed=0) == ["b", "c"]


def test_select_top_uncertainty_tiebreak_reproducible():
    scores = {k: 1.0 for k in KEYS}
    first = select_top_uncertainty(scores, 5, seed=3)
    assert first == select_top_uncertainty(scores, 5, seed=3)
    assert len(set(first)) == 5


def test_select_dispatch_random_and_uncertainty():
    scores = {k: float(i) for i, k in enumerate(KEYS)}
    assert len(select("random", KEYS, None, 4, seed=0)) == 4
    top = select("uncertainty:mean_token_entropy", KEYS, scores, 3, seed=0)
    assert top == KEYS[-1:-4:-1]  # highest scores first


def test_select_unknown_strategy_raises():
    with pytest.raises(ValueError, match="Unknown strategy"):
        select("cleverness", KEYS, None, 3, seed=0)


def test_select_uncertainty_without_scores_raises():
    with pytest.raises(ValueError, match="needs uncertainty scores"):
        select("uncertainty:sequence_entropy", KEYS, None, 3, seed=0)
