"""In-process (white-box) LLM sampling via mlx-lm on Apple Silicon.

Unlike the API backend, local generation exposes the full log-softmax over the
vocabulary at every step, so alongside each sampled text we record the
per-token predictive entropy (bits) that white-box uncertainty metrics need.
All mlx imports live in this module so the API path never touches them.
"""

import math

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler


def entropy_bits_from_logprobs(logprobs: mx.array) -> float:
    """Shannon entropy (bits) of a log-softmax distribution over the vocab."""
    return float(-(mx.exp(logprobs) * logprobs).sum().item()) / math.log(2)


class MLXGenerator:
    """Loads a local model once and samples completions with per-token entropies."""

    def __init__(self, model_name: str):
        self.model, self.tokenizer = load(model_name)

    def sample(self, prompt: str, temperature: float, max_tokens: int,
               seed: int) -> tuple[str, list[float]]:
        """One sampled completion; returns (text, per-generated-token entropies)."""
        messages = [{"role": "user", "content": prompt}]
        try:
            # Qwen3 defaults to thinking mode; <think> blocks would break NER
            # parsing and pollute the entropy signal with reasoning tokens.
            text_prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
                enable_thinking=False,
            )
        except TypeError:
            text_prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
            )
        mx.random.seed(seed)
        sampler = make_sampler(temp=temperature)
        texts: list[str] = []
        entropies: list[float] = []
        for response in stream_generate(self.model, self.tokenizer, text_prompt,
                                        max_tokens=max_tokens, sampler=sampler):
            texts.append(response.text)
            entropies.append(round(entropy_bits_from_logprobs(response.logprobs), 4))
        return "".join(texts).strip(), entropies
