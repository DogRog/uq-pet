"""In-process (white-box) LLM sampling via transformers, for CUDA servers.

The Linux/GPU counterpart of mlx_scoring: local generation exposes the raw
logits at every step, so alongside each sampled text we record the per-token
predictive entropy (bits) that white-box uncertainty metrics need. Runs on
CUDA when available (falls back to MPS/CPU via train.get_device).
"""

import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .train import get_device


def entropy_bits_from_logits(logits: torch.Tensor) -> float:
    """Shannon entropy (bits) of the softmax distribution given raw logits."""
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    return float(-(logprobs.exp() * logprobs).sum().item()) / math.log(2)


class HFGenerator:
    """Loads a local HF model once and samples completions with per-token entropies."""

    def __init__(self, model_name: str):
        self.device = get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto").to(self.device)
        self.model.eval()

    @torch.no_grad()
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
        inputs = self.tokenizer(text_prompt, return_tensors="pt").to(self.device)
        torch.manual_seed(seed)
        output = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_tokens,
            return_dict_in_generate=True,
            output_logits=True,  # raw logits, matching mlx's un-tempered logprobs
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated = output.sequences[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        entropies = [round(entropy_bits_from_logits(step[0]), 4) for step in output.logits]
        return text.strip(), entropies
