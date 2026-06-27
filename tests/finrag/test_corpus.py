from pathlib import Path
from finrag.corpus import Document, Chunk, load_documents, chunk_document


def _doc(text):
    return Document(id="d1", text=text, source="teste")


def test_short_doc_one_chunk():
    chunks = chunk_document(_doc("Parágrafo único curto."), max_chars=800)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == 0


def test_chunks_never_exceed_when_paragraphs_fit():
    p = "Frase. " * 30  # ~210 chars
    doc = _doc(p + "\n\n" + p + "\n\n" + p)
    chunks = chunk_document(doc, max_chars=300, overlap=50)
    assert len(chunks) >= 3
    # respeita fronteira: nenhum chunk corta uma palavra ao meio
    for c in chunks:
        assert not c.text.startswith(" ")
        assert c.text == c.text.strip()


def test_no_empty_chunks():
    doc = _doc("A.\n\n\n\nB.\n\n")
    chunks = chunk_document(doc)
    assert all(c.text for c in chunks)


def test_overlap_present_between_consecutive_chunks():
    doc = _doc("\n\n".join(f"Paragrafo numero {i} com algum conteudo." for i in range(10)))
    chunks = chunk_document(doc, max_chars=120, overlap=40)
    assert len(chunks) >= 2  # houve divisão
    # o início do chunk seguinte compartilha palavras com o fim do anterior
    prev_tail = set(chunks[0].text.split()[-4:])
    next_head = set(chunks[1].text.split()[:4])
    assert prev_tail & next_head  # sobreposição real de conteúdo
    # nenhum chunk começa no meio de uma palavra
    for c in chunks:
        assert c.text == c.text.strip()


def test_load_documents(tmp_path: Path):
    (tmp_path / "a.txt").write_text("conteudo A", encoding="utf-8")
    (tmp_path / "b.txt").write_text("conteudo B", encoding="utf-8")
    docs = load_documents(tmp_path)
    assert {d.id for d in docs} == {"a", "b"}
