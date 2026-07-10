"""Fine-tune a token-classification transformer on a set of PET sentences.

Compact manual torch loop (no HF Trainer: transformers v5 Trainer requires
accelerate, and the manual loop gives exact device/seed control). Every grid
cell uses the same fixed recipe from TrainConfig so the selected data is the
only experimental variable.
"""

import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from .config import NER_TAGS, TrainConfig


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def encode_batch(tokenizer, batch_tokens: list[list[str]], batch_tags: list[list[int]] | None,
                 max_length: int):
    """Tokenize pre-split words; label first subword per word, -100 elsewhere."""
    encoding = tokenizer(
        batch_tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    if batch_tags is None:
        return encoding, None

    labels = []
    for i, tags in enumerate(batch_tags):
        word_ids = encoding.word_ids(batch_index=i)
        seq_labels = []
        previous_word = None
        for word_id in word_ids:
            if word_id is None or word_id == previous_word:
                seq_labels.append(-100)
            else:
                seq_labels.append(tags[word_id])
            previous_word = word_id
        labels.append(seq_labels)
    return encoding, torch.tensor(labels)


def train_token_classifier(examples: list[dict], cfg: TrainConfig, seed: int):
    """Train on `examples` (dicts with 'tokens' and integer 'ner-tags').

    Returns (model, tokenizer); model is left on the training device in eval mode.
    """
    set_seed(seed)
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(cfg.checkpoint)
    model = AutoModelForTokenClassification.from_pretrained(
        cfg.checkpoint,
        num_labels=len(NER_TAGS),
        id2label=dict(enumerate(NER_TAGS)),
        label2id={tag: i for i, tag in enumerate(NER_TAGS)},
    ).to(device)

    def collate(batch):
        encoding, labels = encode_batch(
            tokenizer,
            [ex["tokens"] for ex in batch],
            [ex["ner-tags"] for ex in batch],
            cfg.max_length,
        )
        return encoding, labels

    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(list(examples), batch_size=cfg.batch_size, shuffle=True,
                        generator=generator, collate_fn=collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                                  weight_decay=cfg.weight_decay)
    total_steps = len(loader) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(cfg.warmup_fraction * total_steps),
        num_training_steps=total_steps,
    )

    model.train()
    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        for encoding, labels in loader:
            encoding = {k: v.to(device) for k, v in encoding.items()}
            labels = labels.to(device)
            outputs = model(**encoding, labels=labels)
            outputs.loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            epoch_loss += outputs.loss.item()
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch + 1}/{cfg.epochs} loss={epoch_loss / len(loader):.4f}")

    model.eval()
    return model, tokenizer
