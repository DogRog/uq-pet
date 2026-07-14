"""In-process (white-box) LLM sampling via transformers, for CUDA servers.

The Linux/GPU counterpart of mlx_scoring: local generation exposes the raw
logits at every step, so alongside each sampled text we record the per-token
predictive entropy (bits) that white-box uncertainty metrics need. Runs on
CUDA when available (falls back to MPS/CPU via train.get_device).

All samples for one sentence share a prompt, so they are generated in a
single batched `generate` call: the prompt is prefilled once and the samples
decode in parallel, instead of one memory-bandwidth-bound sequence at a time.
"""

import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .train import get_device


def entropy_bits(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy (bits) of the softmax distributions along the last axis."""
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    return -(logprobs.exp() * logprobs).sum(dim=-1) / math.log(2)


def generated_length(tokens: torch.Tensor, stop_ids: list[int]) -> int:
    """Tokens up to and including the first stop token; anything after it is
    padding emitted while other rows of the batch were still generating."""
    stops = torch.isin(tokens, torch.tensor(stop_ids, device=tokens.device)).nonzero()
    return int(stops[0]) + 1 if stops.numel() else tokens.shape[0]


class HFGenerator:
    """Loads a local HF model once and samples completions with per-token entropies."""

    def __init__(self, model_name: str):
        self.device = get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto").to(self.device)
        self.model.eval()

    def _stop_ids(self) -> list[int]:
        eos = self.model.generation_config.eos_token_id
        if eos is None:
            eos = self.tokenizer.eos_token_id
        return eos if isinstance(eos, list) else [eos]

    @torch.no_grad()
    def sample_batch(self, prompt: str, temperature: float, max_tokens: int,
                     seeds: list[int]) -> tuple[list[str], list[list[float]]]:
        """len(seeds) sampled completions of one prompt as a single batched
        generate call; returns (texts, per-generated-token entropy lists).

        Rows of a batched generate draw from one shared RNG stream, so
        per-sample seeding is impossible: the batch is seeded once with
        seeds[0], which is deterministic per sentence, so an interrupted and
        resumed pass still reproduces the same draws.
        """
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
        input_ids = inputs["input_ids"].repeat(len(seeds), 1)
        torch.manual_seed(seeds[0])
        output = self.model.generate(
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"].repeat(len(seeds), 1),
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_tokens,
            return_dict_in_generate=True,
            output_logits=True,  # raw logits, matching mlx's un-tempered logprobs
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated = output.sequences[:, input_ids.shape[1]:].cpu()
        # One batched entropy kernel per step and a single device-to-host
        # transfer — never a per-token .item() sync.
        step_entropies = torch.stack([entropy_bits(step) for step in output.logits], dim=1).cpu()

        stop_ids = self._stop_ids()
        texts, entropies = [], []
        for row_tokens, row_entropies in zip(generated, step_entropies):
            length = generated_length(row_tokens, stop_ids)
            text = self.tokenizer.decode(row_tokens[:length], skip_special_tokens=True)
            texts.append(text.strip())
            entropies.append([round(float(e), 4) for e in row_entropies[:length]])
        return texts, entropies
