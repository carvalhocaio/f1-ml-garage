"""Árvore de decisão para prever DNF (Módulo 2 — aprendizado supervisionado).

Dois pontos do currículo se encontram aqui, não só "árvore de decisão":
dados desbalanceados (DNF é raro — normalmente <20% das largadas) e
bias-variance (uma árvore sem limite de profundidade decora o treino fácil,
principalmente com poucas features e pouca amostra).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

DEFAULT_N_SPLITS = 5

# Profundidade rasa de propósito: com ~10-15 features (grid_position +
# times one-hot) e algumas centenas de linhas por temporada, uma árvore
# funda decora ruído em vez de aprender padrão — é o mesmo trade-off
# bias-variance do resto do currículo, só que a "regularização" de uma
# árvore é a profundidade, não um termo de penalização.
DEFAULT_MAX_DEPTH = 4

# Primeira técnica de dado desbalanceado a tentar, antes de qualquer coisa
# mais sofisticada (SMOTE, undersampling): pesar as classes pelo inverso da
# frequência. `class_weight="balanced"` faz isso automaticamente a partir
# do `y` de treino, sem precisar calcular a proporção na mão.
DEFAULT_CLASS_WEIGHT = "balanced"

# Com classe rara e uma árvore rasa, algum fold pode acabar com um modelo
# que não prevê NENHUM positivo (comum na versão sem peso de classe) —
# precision vira 0/0, matematicamente indefinida. `zero_division=0` deixa
# esse caso explícito (vira 0.0) em vez de um warning do sklearn por baixo
# dos panos: um modelo que não arrisca nenhuma previsão positiva não
# merece crédito nenhum de precision.
_SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
}


def build_dnf_pipeline(
    max_depth: int = DEFAULT_MAX_DEPTH,
    class_weight: str | None = DEFAULT_CLASS_WEIGHT,
) -> Pipeline:
    return Pipeline(
        [
            (
                "model",
                DecisionTreeClassifier(
                    max_depth=max_depth,
                    class_weight=class_weight,
                    random_state=0,
                ),
            )
        ]
    )


def evaluate_dnf_model(
    features: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
    n_splits: int = DEFAULT_N_SPLITS,
    max_depth: int = DEFAULT_MAX_DEPTH,
    class_weight: str | None = DEFAULT_CLASS_WEIGHT,
) -> dict[str, float]:
    """Avalia o classificador de DNF com `StratifiedGroupKFold`.

    Precisa das duas coisas ao mesmo tempo, não uma ou outra:
    `GroupKFold` sozinho (agrupado por piloto) não garante que cada fold
    tenha uma proporção de DNF parecida — com uma classe já rara, um fold
    "ruim" pode ficar com 0 ou 1 DNF só, e a métrica desse fold vira
    ruído. `StratifiedKFold` sozinho não evita vazar o mesmo piloto entre
    treino e teste. `StratifiedGroupKFold` faz as duas coisas.

    Reporta accuracy, precision, recall e F1 (classe DNF=True) — não só
    accuracy: com DNF raro, "sempre prever que termina" já acerta a
    maioria das vezes e teria accuracy alta parecendo um modelo bom, sem
    detectar DNF nenhum. `recall` é a métrica que mostraria isso quebrado.
    """
    pipeline = build_dnf_pipeline(max_depth=max_depth, class_weight=class_weight)
    cv = StratifiedGroupKFold(n_splits=n_splits)
    scores = cross_validate(
        pipeline,
        features,
        target,
        groups=groups,
        cv=cv,
        scoring=_SCORING,
    )

    return {
        "accuracy": float(np.mean(scores["test_accuracy"])),
        "precision": float(np.mean(scores["test_precision"])),
        "recall": float(np.mean(scores["test_recall"])),
        "f1": float(np.mean(scores["test_f1"])),
        "dnf_rate": float(target.mean()),
    }
