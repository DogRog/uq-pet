import pytest

from uq_pet.config import (
    ExperimentConfig,
    LLMScoreConfig,
    config_to_yaml,
    load_config,
    project_root,
)


def test_load_config_round_trip(tmp_path):
    cfg = ExperimentConfig(budgets=[5, 15], repeats=2)
    cfg.llm.temperature = 0.3
    cfg.train.epochs = 7
    path = tmp_path / "cfg.yaml"
    path.write_text(config_to_yaml(cfg))
    assert load_config(path) == cfg


def test_load_config_missing_sections_use_defaults(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("budgets: [10]\n")
    cfg = load_config(path)
    assert cfg.budgets == [10]
    assert cfg.llm == LLMScoreConfig()
    assert cfg.repeats == ExperimentConfig().repeats


def test_load_config_unknown_key_raises(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("budgets: [10]\nbanana: true\n")
    with pytest.raises(TypeError):
        load_config(path)


def test_cache_path_under_processed_llm_scores():
    path = LLMScoreConfig().cache_path()
    assert path.parts[-3:-1] == ("processed", "llm_scores")
    assert path.name == "meta-llama_llama-3-8b-instruct_k5_t0.7_seed3407.jsonl"


def test_project_root_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("UQ_PET_ROOT", str(tmp_path))
    assert project_root() == tmp_path.resolve()
    monkeypatch.delenv("UQ_PET_ROOT")
    assert (project_root() / "pyproject.toml").exists()
