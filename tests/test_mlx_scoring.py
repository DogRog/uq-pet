import pytest

mx = pytest.importorskip("mlx.core")

from uq_pet.mlx_scoring import entropy_bits_from_logprobs


def test_entropy_uniform_distribution():
    # Uniform over 4 outcomes -> 2 bits.
    logprobs = mx.log(mx.array([0.25, 0.25, 0.25, 0.25]))
    assert entropy_bits_from_logprobs(logprobs) == pytest.approx(2.0, abs=1e-5)


def test_entropy_near_one_hot_is_near_zero():
    probs = mx.array([1.0 - 3e-9, 1e-9, 1e-9, 1e-9])
    assert entropy_bits_from_logprobs(mx.log(probs)) == pytest.approx(0.0, abs=1e-6)
