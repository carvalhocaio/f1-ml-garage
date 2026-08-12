# f1-ml-garage

Laboratório pessoal de Machine Learning clássico aplicado a dados reais de
Fórmula 1, via [FastF1](https://docs.fastf1.dev/). O objetivo não é prever
resultados com a maior acurácia possível, e sim implementar e aplicar,
módulo a módulo, o ferramental de ML supervisionado/não-supervisionado
(regressão, SVM, árvores, ensembles, clustering, PCA/UMAP, GMM) sobre um
domínio real e divertido de acompanhar — voltas, telemetria, estratégia de
pneus e resultados de corrida.

Cada módulo tem um documento correspondente em [`docs/`](docs/) com o
raciocínio por trás das escolhas de implementação.

## Estrutura

\```
src/f1_ml_garage/
└── data/         # normalização do schema do FastF1 e carregamento de
                   # sessões com cache local (docs/00-data.md)
\```

## Setup

Requer Python 3.12+ e [`uv`](https://docs.astral.sh/uv/).

\```bash
uv sync --all-groups
\```

## Testes

Os testes são organizados por marker:

- `unit` — testes rápidos, sem dependência externa
- `integration` — bate na API real do FastF1 (rede); pulado por padrão
- `slow` — testes de treino/avaliação de modelos com datasets maiores

\```bash
make test              # unit (padrão, sem rede)
make test-integration  # inclui integration, bate na API real do FastF1
make test-cov          # unit com relatório de cobertura
\```

## Lint e formatação

\```bash
make lint           # ruff check
make lint-fix       # ruff check --fix
make format         # ruff format
make format-check   # ruff format --check
make check          # lint + format-check
\```

## Documentação

- [`00-data.md`](docs/00-data.md) — pipeline de dados (FastF1)
