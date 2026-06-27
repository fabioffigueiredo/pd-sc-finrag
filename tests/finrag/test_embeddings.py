import numpy as np
from finrag.corpus import Chunk
from finrag.embeddings import SemanticIndex, bm25_search


def _chunks():
    textos = ["o gato dorme no sofá", "ações da bolsa caíram hoje",
              "o cachorro late no quintal", "juros sobem e mercado reage"]
    return [Chunk(doc_id="d", chunk_id=i, text=t, source="s")
            for i, t in enumerate(textos)]


def _fake_embed(texts):
    # embedding determinístico por bag-of-chars (offline, sem baixar modelo)
    vecs = []
    for t in texts:
        v = np.zeros(32, dtype="float32")
        for ch in t.lower():
            v[ord(ch) % 32] += 1.0
        vecs.append(v)
    return np.vstack(vecs)


def test_semantic_search_orders_by_similarity():
    idx = SemanticIndex(embed_fn=_fake_embed)
    idx.build(_chunks())
    res = idx.search("ações e juros no mercado", k=2)
    assert len(res) == 2
    # scores em ordem decrescente
    assert res[0][1] >= res[1][1]
    # o top-1 deve ser um dos chunks de mercado/finanças
    assert "mercado" in res[0][0].text or "ações" in res[0][0].text or "juros" in res[0][0].text


def test_semantic_search_is_deterministic():
    idx = SemanticIndex(embed_fn=_fake_embed)
    idx.build(_chunks())
    a = [c.chunk_id for c, _ in idx.search("juros", k=3)]
    b = [c.chunk_id for c, _ in idx.search("juros", k=3)]
    assert a == b


def test_save_and_load(tmp_path):
    idx = SemanticIndex(embed_fn=_fake_embed)
    idx.build(_chunks())
    idx.save(tmp_path)
    idx2 = SemanticIndex(embed_fn=_fake_embed)
    idx2.load(tmp_path)
    assert idx2.search("gato", k=1)[0][0].text == idx.search("gato", k=1)[0][0].text


def test_bm25_keyword_match():
    res = bm25_search("cachorro quintal", _chunks(), k=1)
    assert "cachorro" in res[0][0].text
