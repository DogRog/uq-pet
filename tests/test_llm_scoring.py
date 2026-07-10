import json

from uq_pet.llm_scoring import build_ner_prompt, parse_ner_output, prompt_fingerprint

TOKENS = ["The", "clerk", "checks", "the", "form"]


def _output(tags: list[str]) -> str:
    return json.dumps([{"token": t, "tag": tag} for t, tag in zip(TOKENS, tags)])


def test_parse_ner_output_valid_json():
    tags = ["B-Actor", "I-Actor", "B-Activity", "B-Activity Data", "I-Activity Data"]
    assert parse_ner_output(_output(tags), TOKENS) == tags


def test_parse_ner_output_strips_code_fence():
    tags = ["O", "O", "B-Activity", "O", "O"]
    fenced = f"```json\n{_output(tags)}\n```"
    assert parse_ner_output(fenced, TOKENS) == tags


def test_parse_ner_output_pads_short_with_o():
    short = json.dumps([{"token": "The", "tag": "B-Actor"}])
    assert parse_ner_output(short, TOKENS) == ["B-Actor", "O", "O", "O", "O"]


def test_parse_ner_output_truncates_long():
    long = json.dumps([{"token": "x", "tag": "O"}] * 10)
    assert len(parse_ner_output(long, TOKENS)) == len(TOKENS)


def test_parse_ner_output_invalid_tag_maps_to_o():
    tags = ["B-Actor", "NOT-A-TAG", "O", "O", "O"]
    assert parse_ner_output(_output(tags), TOKENS) == ["B-Actor", "O", "O", "O", "O"]


def test_parse_ner_output_garbage_returns_all_o():
    assert parse_ner_output("sorry, I cannot help", TOKENS) == ["O"] * len(TOKENS)


def test_build_ner_prompt_contains_tokens_and_example():
    example_tokens = ["Alice", "approves"]
    example_tags = ["B-Actor", "B-Activity"]
    prompt = build_ner_prompt(TOKENS, example_tokens, example_tags)
    assert str(TOKENS) in prompt
    assert str(example_tokens) in prompt
    assert "B-Activity" in prompt


def test_prompt_fingerprint_stable_and_sensitive():
    fp = prompt_fingerprint(["Alice"], ["B-Actor"])
    assert fp == prompt_fingerprint(["Alice"], ["B-Actor"])
    assert fp != prompt_fingerprint(["Bob"], ["B-Actor"])
