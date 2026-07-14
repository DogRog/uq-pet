import math

import pytest
import torch

from uq_pet.hf_scoring import entropy_bits_from_logits


def test_entropy_uniform_distribution():
    # Equal logits -> uniform over 4 outcomes -> 2 bits.
    assert entropy_bits_from_logits(torch.zeros(4)) == pytest.approx(2.0, abs=1e-5)


def test_entropy_near_one_hot_is_near_zero():
    logits = torch.tensor([30.0, 0.0, 0.0, 0.0])
    assert entropy_bits_from_logits(logits) == pytest.approx(0.0, abs=1e-6)


def test_entropy_matches_manual_two_outcomes():
    # p = (0.75, 0.25) -> H = 0.75*log2(4/3) + 0.25*log2(4)
    logits = torch.log(torch.tensor([0.75, 0.25]))
    expected = 0.75 * math.log2(4 / 3) + 0.25 * 2
    assert entropy_bits_from_logits(logits) == pytest.approx(expected, abs=1e-6)
