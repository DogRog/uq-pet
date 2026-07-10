"""LLM repeated-sampling pass over the experiment pool, with JSONL caching.

Cache format (`results/llm_scores/<model>_k<K>_t<temp>_seed<seed>.jsonl`):
line 0 is a header record describing the config; every other line is one
sentence record with the raw responses and parsed tag sequences. Metrics are
never stored — they are recomputed from `parsed_samples`, so budgets, metrics
and repeats can be swept without new API calls. Existing keys are skipped on
rerun, so an interrupted pass resumes where it left off.
"""

import asyncio
import hashlib
import json
import os
from pathlib import Path

from datasets import Dataset
from dotenv import load_dotenv
from openai import AsyncOpenAI

from .config import (
    DATASET_RULES,
    ENTITY_DEFINITIONS,
    FEW_SHOT_EXAMPLE_INDEX,
    NER_DATASET_URL,
    NER_TAGS,
    LLMScoreConfig,
)
from .data import sentence_key, tag_ids_to_labels

load_dotenv()


def build_ner_prompt(tokens: list, example_tokens: list, example_tags: list) -> str:
    tags_str = ", ".join(f"'{tag}'" for tag in NER_TAGS)
    example_pairs = [{"token": tok, "tag": tag} for tok, tag in zip(example_tokens, example_tags)]
    example_output = json.dumps(example_pairs, indent=2)

    return (
        "You are a strict Named Entity Recognition (NER) system for Process Extraction.\n"
        "Assign exactly one tag to each token in the sentence.\n\n"
        f"ENTITY DEFINITIONS:\n{ENTITY_DEFINITIONS}\n\n"
        f"DATASET RULES:\n{DATASET_RULES}\n"
        f"- Tags available: [{tags_str}]\n\n"
        "=== EXAMPLE ===\n"
        f"Tokens: {example_tokens}\n"
        f"Output:\n{example_output}\n"
        "=== END OF EXAMPLE ===\n\n"
        f"Tokens to tag:\n{tokens}\n\n"
        "Output MUST be a valid JSON array of objects. Do not output anything except the JSON array."
    )


def parse_ner_output(output_str: str, tokens: list) -> list[str]:
    """Parse the model's JSON output into a fixed-length list of tags matching `tokens`."""
    expected_length = len(tokens)
    cleaned_str = output_str.strip()

    if cleaned_str.startswith("```"):
        cleaned_str = cleaned_str.replace("```json", "").replace("```", "").strip()

    extracted_tags = []
    try:
        parsed = json.loads(cleaned_str)
        if isinstance(parsed, list):
            extracted_tags = [item.get("tag", "O") for item in parsed if isinstance(item, dict)]
    except Exception:
        pass

    if len(extracted_tags) < expected_length:
        extracted_tags.extend(["O"] * (expected_length - len(extracted_tags)))
    else:
        extracted_tags = extracted_tags[:expected_length]

    return [tag if tag in NER_TAGS else "O" for tag in extracted_tags]


def prompt_fingerprint(few_shot_tokens: list, few_shot_tags: list) -> str:
    """Hash of the prompt template + few-shot example, to detect stale caches."""
    template = build_ner_prompt(["<TOKENS>"], few_shot_tokens, few_shot_tags)
    return hashlib.sha256(template.encode()).hexdigest()[:16]


def make_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )


async def get_single_sample(client: AsyncOpenAI, prompt: str, cfg: LLMScoreConfig,
                            semaphore: asyncio.Semaphore) -> str:
    async with semaphore:
        for attempt in range(cfg.max_retries):
            try:
                response = await client.chat.completions.create(
                    model=cfg.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=cfg.temperature,
                    max_tokens=cfg.max_tokens,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt == cfg.max_retries - 1:
                    print(f"API error after {cfg.max_retries} attempts: {e}")
                    return ""
                await asyncio.sleep(2 ** attempt)
    return ""


def load_cache(cache_path: Path, expected_header: dict | None = None) -> dict[str, dict]:
    """Load cached sentence records keyed by sentence key; validate the header."""
    cache = {}
    if not cache_path.exists():
        return cache
    with open(cache_path) as f:
        for line_no, line in enumerate(f):
            record = json.loads(line)
            if line_no == 0 and "header" in record:
                if expected_header is not None:
                    mismatched = {
                        k: (record["header"].get(k), v)
                        for k, v in expected_header.items()
                        if record["header"].get(k) != v
                    }
                    if mismatched:
                        raise ValueError(
                            f"Cache {cache_path} header mismatch (cached, expected): {mismatched}"
                        )
                continue
            cache[record["key"]] = record
    return cache


async def score_pool(cfg: LLMScoreConfig, pool: Dataset, limit: int | None = None) -> dict[str, dict]:
    """Sample the LLM K times for every pool sentence, appending results to the cache.

    Returns the full cache (existing + newly scored records).
    """
    few_shot_example = pool[FEW_SHOT_EXAMPLE_INDEX]
    few_shot_tokens = few_shot_example["tokens"]
    few_shot_tags = tag_ids_to_labels(few_shot_example["ner-tags"])

    header = {
        "model": cfg.model,
        "num_samples": cfg.num_samples,
        "temperature": cfg.temperature,
        "seed": cfg.seed,
        "prompt_fingerprint": prompt_fingerprint(few_shot_tokens, few_shot_tags),
        "dataset_url": NER_DATASET_URL,
    }

    cache_path = cfg.cache_path()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = load_cache(cache_path, expected_header=header)
    if not cache_path.exists():
        with open(cache_path, "w") as f:
            f.write(json.dumps({"header": header}) + "\n")

    examples = list(pool)
    if limit is not None:
        examples = examples[:limit]
    pending = [ex for ex in examples if sentence_key(ex) not in cache]
    print(f"Scoring pool: {len(pending)} to score, {len(cache)} cached ({cache_path})")

    client = make_client()
    semaphore = asyncio.Semaphore(cfg.max_concurrency)
    write_lock = asyncio.Lock()
    done_count = 0

    # Sentences run concurrently; the semaphore caps total in-flight API calls.
    async def score_sentence(example, f):
        nonlocal done_count
        tokens = example["tokens"]
        prompt = build_ner_prompt(tokens, few_shot_tokens, few_shot_tags)
        raw = await asyncio.gather(*[
            get_single_sample(client, prompt, cfg, semaphore)
            for _ in range(cfg.num_samples)
        ])
        record = {
            "key": sentence_key(example),
            "document": example["document name"],
            "sentence_id": example["sentence-ID"],
            "tokens": tokens,
            "gt_tags": tag_ids_to_labels(example["ner-tags"]),
            "raw_responses": list(raw),
            "parsed_samples": [parse_ner_output(r, tokens) for r in raw],
        }
        async with write_lock:
            f.write(json.dumps(record) + "\n")
            f.flush()
            cache[record["key"]] = record
            done_count += 1
            print(f"Scored {done_count}/{len(pending)} ({record['key']})")

    with open(cache_path, "a") as f:
        await asyncio.gather(*[score_sentence(ex, f) for ex in pending])

    return cache
