# f1-ml-garage

Laboratório pessoal de Machine Learning clássico aplicado a dados reais de
Fórmula 1, via [FastF1](https://docs.fastf1.dev/). O objetivo não é prever
resultados com a maior acurácia possível, e sim implementar e aplicar,
módulo a módulo, o ferramental de ML supervisionado/não-supervisionado
(regressão, SVM, árvores, clustering, PCA, GMM) sobre um domínio real e
divertido de acompanhar — voltas, telemetria, estratégia de pneus e
resultados de corrida.

Cada módulo tem um documento correspondente em [`docs/`](docs/) com o
raciocínio por trás das escolhas de implementação — incluindo bugs reais
encontrados e hipóteses testadas (confirmadas e não confirmadas).

## Estrutura

```
src/f1_ml_garage/
├── data/                    # normalização + carregamento (única parte
│   │                        # que fala com a rede FastF1)
│   ├── timeutils.py         # conversão Timedelta -> segundos, compartilhada
│   ├── laps.py              # normalização de voltas + filtros de qualidade
│   ├── results.py           # normalização de classificação final (inclui dnf)
│   ├── telemetry.py         # normalização de telemetria por amostra
│   └── session.py           # carregamento via FastF1, com cache
├── features/                # feature engineering, sem tocar rede
│   ├── pace.py               # features do modelo de ritmo (Módulo 2)
│   ├── dnf.py                 # features do modelo de DNF (Módulo 2)
│   ├── telemetry_summary.py    # telemetria -> vetor de features por volta
│   ├── tyre.py                  # features do SVM de composto (Módulo 2)
│   └── driving_style.py          # features de clustering de estilo (Módulo 4)
└── models/                  # pipelines e avaliação
    ├── pace.py               # regressão linear de tempo de volta
    ├── dnf.py                 # árvore, logística, Random Forest, XGBoost,
    │                          # stacking — 5 modelos pro mesmo problema
    ├── evaluation.py          # avaliação de classificadores, compartilhada
    ├── tyre.py                 # SVM de composto
    └── clustering.py            # PCA + k-means + GMM/EM
```

## Setup

Requer Python 3.12+ e [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --all-groups
```

## Testes

Os testes são organizados por marker:

- `unit` — testes rápidos, sem dependência externa
- `integration` — bate na API real do FastF1 (rede); pulado por padrão
- `slow` — testes de treino/avaliação de modelos com datasets maiores

```bash
make test              # unit (padrão, sem rede)
make test-integration  # inclui integration, bate na API real do FastF1
make test-cov          # unit com relatório de cobertura
```

## Lint e formatação

```bash
make lint           # ruff check
make lint-fix       # ruff check --fix
make format         # ruff format
make format-check   # ruff format --check
make check          # lint + format-check
```

## Documentação

- [`00-data.md`](docs/00-data.md) — pipeline de dados (FastF1): voltas,
  resultados, telemetria, carregamento e cache
- [`01-pace-model.md`](docs/01-pace-model.md) — Módulo 2: regressão linear
  de ritmo de corrida (2 bugs de colinearidade encontrados e corrigidos)
- [`02-dnf-model.md`](docs/02-dnf-model.md) — Módulo 2: 5 classificadores
  de abandono (árvore, logística, Random Forest, XGBoost, stacking) — bug
  real de dados no alvo `dnf` corrigido, dados desbalanceados e ajuste de
  limiar de decisão em números reais, bias-variance via capacidade de
  ensemble, feature engineering testada e não confirmada
- [`03-tyre-model.md`](docs/03-tyre-model.md) — Módulo 2: SVM classificando
  composto de pneu via telemetria
- [`04-driving-style-clustering.md`](docs/04-driving-style-clustering.md) —
  Módulo 4: PCA + k-means + GMM/EM explorando estilo de pilotagem, 6
  iterações de hipóteses testadas contra dado real
