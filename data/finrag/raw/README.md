# data/finrag/raw — Fontes reais

Esta pasta destina-se a documentos `.txt` de domínio público que complementam o corpus sintético do FinRAG.

## Como usar

1. Coloque aqui qualquer arquivo `.txt` de fonte aberta relevante (ex.: atas do COPOM, relatórios do FMI, Banco Mundial, OCDE).
2. Rode `PYTHONPATH=src .venv/bin/python scripts/prepare_corpus.py` novamente — o pipeline reindexará automaticamente todos os arquivos encontrados nesta pasta.
3. O pipeline funciona apenas com os documentos sintéticos caso esta pasta esteja vazia.

## Proveniência e licença

| Arquivo | Fonte | URL | Licença |
|---------|-------|-----|---------|
| *(nenhum arquivo adicionado ainda)* | — | — | — |

Ao adicionar um arquivo, preencha a tabela acima com a fonte original, a URL de acesso e a licença de uso (ex.: CC BY 4.0, domínio público).

## Restrições

- Não adicione documentos proprietários, confidenciais ou que não tenham licença compatível com uso acadêmico.
- Não inclua dados de clientes ou informações de instituições reais sem autorização explícita.
