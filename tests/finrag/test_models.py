from finrag.models import MockLLM, get_llm


def test_mockllm_returns_scripted_sequence():
    llm = MockLLM(["a", "b"])
    assert llm.generate("x") == "a"
    assert llm.generate("x") == "b"
    assert llm.generate("x") == "b"  # repete a última


def test_mockllm_single_string():
    llm = MockLLM("ok")
    assert llm.generate("qualquer") == "ok"


def test_get_llm_mock_backend():
    llm = get_llm("mock", scripted="oi")
    assert llm.generate("p") == "oi"


def test_get_llm_groq_without_key_falls_back_to_mock(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    llm = get_llm("groq")
    # sem chave, get_llm devolve um MockLLM para reprodutibilidade
    assert isinstance(llm, MockLLM)
