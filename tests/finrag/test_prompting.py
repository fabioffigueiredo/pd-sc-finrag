import pytest
from finrag.models import MockLLM
from finrag.prompting import (
    build_extraction_prompt, parse_json_response, validate_extraction,
    extract, FinancialExtraction,
)

VALID = ('{"empresa":"Acme","evento":"queda de receita",'
         '"sentimento":"negativo","risco":"alto","horizonte":"curto"}')


def test_parse_strips_code_fences():
    raw = "```json\n" + VALID + "\n```"
    assert parse_json_response(raw)["empresa"] == "Acme"


def test_parse_finds_object_in_noise():
    raw = "Claro! Aqui está:\n" + VALID + "\nEspero ter ajudado."
    assert parse_json_response(raw)["sentimento"] == "negativo"


def test_parse_raises_on_garbage():
    with pytest.raises(ValueError):
        parse_json_response("sem json aqui")


def test_validate_rejects_bad_sentiment():
    bad = {"empresa": "x", "evento": "y", "sentimento": "otimo",
           "risco": "z", "horizonte": "w"}
    with pytest.raises(Exception):
        validate_extraction(bad)


@pytest.mark.parametrize("tech", ["zero_shot", "few_shot", "cot", "meta"])
def test_prompt_techniques_are_distinct_and_contain_text(tech):
    p = build_extraction_prompt("Acme teve queda de receita.", tech)
    assert "Acme" in p and len(p) > 30


def test_few_shot_has_examples_zero_shot_does_not():
    assert "Exemplo" in build_extraction_prompt("t", "few_shot")
    assert "Exemplo" not in build_extraction_prompt("t", "zero_shot")


def test_extract_end_to_end_with_mock():
    llm = MockLLM(VALID)
    out = extract("Acme teve queda de receita.", llm, "few_shot")
    assert isinstance(out, FinancialExtraction)
    assert out.sentimento == "negativo"
