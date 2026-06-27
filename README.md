# FinRAG — Assistente Cognitivo de Mercado (Projeto 2)

> **Disciplina:** Sistemas Cognitivos com Large Language Models
> **Programa:** Pós-Graduação em Machine Learning, Deep Learning e IA — INFNET
> **Autor:** Fabio Ferreira Figueiredo

Sistema cognitivo que recebe perguntas em linguagem natural, recupera trechos
relevantes de uma base de conhecimento por **busca vetorial** e gera respostas
**fundamentadas, estruturadas e auditáveis** usando LLMs — com a opção de execução
**100% local/privada**. É a evolução do Projeto 1 (FinNLP), agora centrada em
**LLMs + RAG**.

> **Compliance.** Projeto **acadêmico**. O cliente — "a Gestora" — é **fictício e
> genérico**; nenhuma instituição financeira real é citada. O corpus é sintético ou
> público/citável. Nenhuma chave, token ou dado sensível está no repositório.

---

## Cobertura das competências (rubricas)

| # | Competência | Onde é atendida |
|---|---|---|
| 1 | NLP com LLMs e modelos pré-treinados | `src/finrag/models.py` · Notebook Seção 1 |
| 2 | Prompt engineering e saídas controladas (JSON) | `src/finrag/prompting.py` · Notebook Seção 2 |
| 3 | Embeddings semânticos e busca vetorial | `src/finrag/embeddings.py` · Notebook Seção 3 |
| 4 | Inferência local/privada (GPT4All) vs remota | `src/finrag/models.py` · Notebook Seção 1 |
| 5 | Pipeline RAG + segurança (prompt injection) | `src/finrag/rag.py` · `guardrails.py` · Notebook Seção 4 |

---

## Estrutura

```
src/finrag/
├── models.py        # LLMClient: GroqClient (remoto) + GPT4AllClient (local) + MockLLM
├── prompting.py     # 4 técnicas de prompt + parsing/validação JSON (pydantic)
├── corpus.py        # carregamento de docs + chunking consciente (parágrafo/sentença + overlap)
├── embeddings.py    # SemanticIndex (sentence-transformers + FAISS) + bm25_search
├── guardrails.py    # detecção/sanitização de prompt injection
└── rag.py           # pipeline retrieve → guardrail → augment → generate
scripts/
├── prepare_corpus.py     # gera sintéticos + doc envenenado e indexa o corpus (FAISS)
├── build_notebook.py     # monta o notebook FinRAG_Pipeline.ipynb
└── gerar_relatorio_pdf.py# gera o relatório técnico em PDF
notebooks/FinRAG_Pipeline.ipynb   # vitrine integrada (Run-All), 5 seções
data/finrag/{synthetic,raw,index}/# corpus e índice FAISS persistido
tests/finrag/                     # 31 testes (offline, mockados)
reports/finrag/                   # relatório técnico (PDF)
```

---

## Como reproduzir

> Ambiente: **Python 3.12** (CPU-only). Gerenciador recomendado: **uv**.

```bash
# 1. Ambiente virtual + dependências
uv venv .venv --python 3.12
uv pip install -r requirements.txt    # numpy fica pinado em 1.26.4

# 2. Chave da Groq (modelo remoto) — gratuita em https://console.groq.com/keys
cp .env.example .env
#   edite .env e preencha GROQ_API_KEY=gsk_...   (o .env é ignorado pelo git)

# 3. Modelo local (GPT4All) — baixa ~1.9 GB no primeiro uso
python -c "from gpt4all import GPT4All; GPT4All('Llama-3.2-3B-Instruct-Q4_0.gguf')"

# 4. Preparar o corpus e indexar (gera data/finrag/index/)
PYTHONPATH=src python scripts/prepare_corpus.py

# 5. Executar o notebook de ponta a ponta
PYTHONPATH=src jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=600 notebooks/FinRAG_Pipeline.ipynb

# (opcional) rodar os testes
PYTHONPATH=src python -m pytest tests/finrag/ -v
```

> **Sem chave da Groq?** O `get_llm("groq")` cai automaticamente para um `MockLLM`
> determinístico — o notebook ainda executa, mas com saída simulada no lugar da
> geração real. Para a avaliação completa, recomenda-se preencher a chave (free tier).

---

## Como consultar (uso programático)

```python
from dotenv import load_dotenv; load_dotenv()
from finrag.models import get_llm
from finrag.embeddings import SemanticIndex
from finrag.rag import answer

idx = SemanticIndex(); idx.load("data/finrag/index")
llm = get_llm("groq")          # ou get_llm("local") para inferência privada

res = answer("Como a Gestora se protege do risco de liquidez?", idx, llm, k=3)
print(res.answer)              # resposta fundamentada
print([(c.source, c.chunk_id) for c in res.contexts])  # trechos citados
print([c.source for c in res.blocked])                 # trechos bloqueados pelo guardrail
```

---

## Limitações conhecidas

- **Modelo local (3B) em CPU** é lento e exibe artefatos (repetição) — adequado para
  privacidade, não para qualidade máxima.
- **Busca densa pura**: uma busca **híbrida** (densa + BM25) seria a evolução natural
  para consultas curtas/ambíguas.
- **Guardrail por heurística/regex**: cobre injeções conhecidas; um atacante criativo
  pode contorná-lo — defesa em profundidade (validação de saída, limites de escopo)
  seria o próximo passo.
- **Corpus sintético**: criado para o exercício; resultados não representam um fundo real.
