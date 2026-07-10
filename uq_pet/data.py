"""PET NER dataset loading and pool/test splitting."""

from datasets import ClassLabel, Dataset, Features, Sequence, Value, load_dataset

from .config import NER_DATASET_URL, NER_TAGS, SEED, TEST_SIZE


def load_pet_ner() -> Dataset:
    """Download the PET entities dataset (417 sentence-level examples)."""
    features = Features({
        "document name": Value("string"),
        "sentence-ID": Value("int8"),
        "tokens": Sequence(Value("string")),
        "tokens-IDs": Sequence(Value("int8")),
        "ner-tags": Sequence(ClassLabel(names=NER_TAGS)),
    })
    dataset = load_dataset("json", data_files={"full": NER_DATASET_URL}, features=features)
    return dataset["full"]


def split_pool_test(seed: int = SEED, test_size: float = TEST_SIZE) -> tuple[Dataset, Dataset]:
    """80/20 split: (experiment pool, held-out test)."""
    splits = load_pet_ner().train_test_split(test_size=test_size, seed=seed)
    return splits["train"], splits["test"]


def sentence_key(example: dict) -> str:
    """Unique sentence identifier; sentence-ID alone repeats across documents."""
    return f"{example['document name']}::{example['sentence-ID']}"


def tag_ids_to_labels(tag_ids: list) -> list[str]:
    return [NER_TAGS[tid] for tid in tag_ids]
