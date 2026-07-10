"""Shared constants and experiment configuration dataclasses."""

from dataclasses import dataclass, field
from pathlib import Path

SEED = 3407
TEST_SIZE = 0.2
FEW_SHOT_EXAMPLE_INDEX = 12  # index into the pool split used as the prompt example

RESULTS_DIR = Path("results")
LLM_SCORES_DIR = RESULTS_DIR / "llm_scores"
RUNS_PATH = RESULTS_DIR / "experiment_runs.jsonl"
FIGURES_DIR = RESULTS_DIR / "figures"

NER_DATASET_URL = (
    "https://raw.githubusercontent.com/patriziobellan86/PETv1.1/master/"
    "PETv1.1-entities.jsonl"
)

NER_TAGS = [
    "O", "B-Actor", "I-Actor", "B-Activity", "I-Activity",
    "B-Activity Data", "I-Activity Data", "B-Further Specification",
    "I-Further Specification", "B-XOR Gateway", "I-XOR Gateway",
    "B-Condition Specification", "I-Condition Specification",
    "B-AND Gateway", "I-AND Gateway"
]

ENTITY_DEFINITIONS = """\
- Actor: The person, system, or role performing the action.
- Activity: The task or action being executed.
- Activity Data: The object, document, or data manipulated by the activity.
- Further Specification: Additional context, tools, or locations (e.g., 'via email').
- XOR Gateway: Words indicating an exclusive branching point (e.g., 'If', 'otherwise').
- Condition Specification: The condition required to take a branch (e.g., 'the claim is valid').
- AND Gateway: Words indicating parallel execution (e.g., 'in parallel').
- O: Tokens outside of any process entity."""

DATASET_RULES = """\
- Determiners ('The', 'a', 'an') MUST be included in the entity if they precede it.
- Multi-word entities must start with 'B-' (Beginning) and continue with 'I-' (Inside).
- Single-word entities get the 'B-' tag."""


@dataclass
class LLMScoreConfig:
    """Configuration for the LLM repeated-sampling pass over the pool."""

    model: str = "meta-llama/llama-3-8b-instruct"
    num_samples: int = 5
    temperature: float = 0.7
    max_tokens: int = 1500
    seed: int = SEED
    max_concurrency: int = 8
    max_retries: int = 3

    def cache_path(self) -> Path:
        safe_model = self.model.replace("/", "_")
        name = f"{safe_model}_k{self.num_samples}_t{self.temperature}_seed{self.seed}.jsonl"
        return LLM_SCORES_DIR / name


@dataclass
class TrainConfig:
    """Fixed fine-tuning recipe used for every grid cell."""

    checkpoint: str = "distilbert-base-cased"
    epochs: int = 20
    batch_size: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_fraction: float = 0.1
    max_length: int = 256


@dataclass
class ExperimentConfig:
    """Full grid: budgets x strategies x seeds."""

    budgets: list[int] = field(default_factory=lambda: [10, 25, 50, 100])
    strategies: list[str] = field(default_factory=lambda: [
        "random",
        "uncertainty:mean_token_entropy",
        "uncertainty:sequence_entropy",
        "uncertainty:jaccard_distance",
    ])
    repeats: int = 5
    llm: LLMScoreConfig = field(default_factory=LLMScoreConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
