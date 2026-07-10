"""Prediction and evaluation for trained token classifiers."""

import torch
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

from .config import NER_TAGS, TrainConfig
from .data import tag_ids_to_labels
from .train import encode_batch, get_device


@torch.no_grad()
def predict_tags(model, tokenizer, examples: list[dict],
                 cfg: TrainConfig, batch_size: int = 32) -> list[list[str]]:
    """Predict one tag per word (first-subword logits) for each example."""
    device = get_device()
    model.eval()
    predictions = []

    examples = list(examples)
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        batch_tokens = [ex["tokens"] for ex in batch]
        encoding, _ = encode_batch(tokenizer, batch_tokens, None, cfg.max_length)
        encoding = {k: v.to(device) for k, v in encoding.items()}
        logits = model(**encoding).logits.argmax(dim=-1).cpu()

        for i, tokens in enumerate(batch_tokens):
            word_ids = tokenizer(
                [tokens], is_split_into_words=True, truncation=True,
                max_length=cfg.max_length,
            ).word_ids(batch_index=0)
            tags = ["O"] * len(tokens)  # words truncated away default to O
            previous_word = None
            for position, word_id in enumerate(word_ids):
                if word_id is not None and word_id != previous_word:
                    tags[word_id] = NER_TAGS[logits[i, position].item()]
                previous_word = word_id
            predictions.append(tags)
    return predictions


def evaluate(predictions: list[list[str]], gold: list[list[str]]) -> dict:
    """Entity-level seqeval micro F1 (primary) + per-type F1 + token accuracy."""
    f1 = float(f1_score(gold, predictions, average="micro", zero_division=0))
    precision = float(precision_score(gold, predictions, average="micro", zero_division=0))
    recall = float(recall_score(gold, predictions, average="micro", zero_division=0))
    report = classification_report(gold, predictions, output_dict=True, zero_division=0)
    per_type_f1 = {
        entity_type: float(metrics["f1-score"])
        for entity_type, metrics in report.items()
        if entity_type not in ("micro avg", "macro avg", "weighted avg")
    }

    total = sum(len(seq) for seq in gold)
    correct = sum(
        1 for pred_seq, gold_seq in zip(predictions, gold)
        for p, g in zip(pred_seq, gold_seq) if p == g
    )

    return {
        "entity_f1": f1,
        "entity_precision": precision,
        "entity_recall": recall,
        "per_type_f1": per_type_f1,
        "token_accuracy": correct / total if total else 0.0,
    }


def evaluate_model_on(model, tokenizer, examples: list[dict], cfg: TrainConfig) -> dict:
    predictions = predict_tags(model, tokenizer, examples, cfg)
    gold = [tag_ids_to_labels(ex["ner-tags"]) for ex in examples]
    return evaluate(predictions, gold)
