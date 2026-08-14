"""Avaliação de classificadores binários, compartilhada entre modelos de DNF.

`StratifiedGroupKFold` + as métricas de dados desbalanceados (accuracy,
precision, recall, f1, com `zero_division` explícito) não dependem de qual
classificador está por trás do pipeline - extraído aqui para reusar entre
árvore de decisão e regressão logística sem duplicar a lógica de CV.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline

DEFAULT_N_SPLITS = 5

# Com classe rara, algum fold pode acabar com um modelo que não prevê
# NENHUM positivo — precision vira 0/0, matematicamente indefinida.
# `zero_division=0` deixa esse caso explícito (vira 0.0) em vez de um
# warning do sklearn por baixo dos panos: um modelo que não arrisca nenhuma
# previsão positiva não merece crédito nenhum de precision.
_SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
}


def evaluate_classifier(
    pipeline: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
    n_splits: int = DEFAULT_N_SPLITS,
) -> dict[str, float]:
    """Avalia um classificador binário com `StratifiedGroupKFold`.

    Precisa das duas coisas ao mesmo tempo, não uma ou outra: `GroupKFold`
    sozinho (agrupado por piloto) não garante que cada fold tenha uma
    proporção de classe positiva parecida — com uma classe já rara, um
    fold "ruim" pode ficar com 0 ou 1 positivo só, e a métrica desse fold
    vira ruído. `StratifiedKFold` sozinho não evita vazar o mesmo piloto
    entre treino e teste. `StratifiedGroupKFold` faz as duas coisas.

    Reporta accuracy, precision, recall e F1 (classe positiva) — não só
    accuracy: com classe rara, "sempre prever a maioria" já acerta a
    maioria das vezes e teria accuracy alta parecendo um modelo bom, sem
    detectar a classe rara nenhuma vez. `recall` é a métrica que mostraria
    isso quebrado.
    """
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


_MULTICLASS_SCORING = {
    "accuracy": "accuracy",
    "precision_macro": make_scorer(precision_score, average="macro", zero_division=0),
    "recall_macro": make_scorer(recall_score, average="macro", zero_division=0),
    "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
}


def evaluate_multiclass_classifier(
    pipeline: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
    n_splits: int = DEFAULT_N_SPLITS,
) -> dict[str, float]:
    """Avalia um classificador multiclasse com `StratifiedGroupKFold`.

    Precision/recall/f1 usam `average="macro"` — cada classe pesa igual no
    resultado final, independente de quantas voltas ela tem. `"weighted"`
    (a média ponderada pelo tamanho de cada classe) esconderia o mesmo
    problema que accuracy sozinha esconde em `evaluate_classifier`: se
    "medium" for raro, `"weighted"` quase ignoraria o desempenho nele;
    `"macro"` não deixa.
    """
    cv = StratifiedGroupKFold(n_splits=n_splits)
    scores = cross_validate(
        pipeline,
        features,
        target,
        groups=groups,
        cv=cv,
        scoring=_MULTICLASS_SCORING,
    )

    return {
        "accuracy": float(np.mean(scores["test_accuracy"])),
        "precision_macro": float(np.mean(scores["test_precision_macro"])),
        "recall_macro": float(np.mean(scores["test_recall_macro"])),
        "f1_macro": float(np.mean(scores["test_f1_macro"])),
    }
