"""Avaliação de classificadores binários, compartilhada entre modelos de DNF.

`StratifiedGroupKFold` + as métricas de dados desbalanceados (accuracy,
precision, recall, f1, com `zero_division` explícito) não dependem de qual
classificador está por trás do pipeline - extraído aqui para reusar entre
árvore de decisão e regressão logística sem duplicar a lógica de CV.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    StratifiedGroupKFold,
    cross_val_predict,
    cross_validate,
)
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


def find_best_threshold(probabilities: np.ndarray, target: pd.Series) -> float:
    """Escolhe o limiar de decisão (probabilidade -> classe) que maximiza
    F1, varrendo a curva precision-recall.

    Todo classificador binário do sklearn/XGBoost usa 0.5 como corte
    padrão pra transformar probabilidade em "sim"/"não" — um valor
    arbitrário, não otimizado pra nada em particular. Quando accuracy não
    é a métrica que importa (ex.: recall, pra não deixar passar DNF),
    ajustar esse corte diretamente costuma valer mais que só reponderar o
    treino (`class_weight`/`scale_pos_weight`) — foi o que descobrimos
    comparando árvore/logística/Random Forest/XGBoost no mesmo problema
    (`docs/02-dnf-model.md`, seção de ensembles): mais capacidade de
    modelo, preso no corte de 0.5, só empurrava recall pra baixo.
    """
    precision, recall, thresholds = precision_recall_curve(target, probabilities)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-12)
    # precision_recall_curve devolve um ponto a mais em precision/recall
    # do que em thresholds (o último ponto é precision=1, recall=0, sem
    # limiar correspondente) — descarta esse último ponto antes do argmax.
    best_index = int(np.argmax(f1_scores[:-1]))
    return float(thresholds[best_index])


def evaluate_classifier_with_tuned_threshold(
    pipeline: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
    n_splits: int = DEFAULT_N_SPLITS,
) -> dict[str, float]:
    """Mesma avaliação de `evaluate_classifier`, mas escolhendo o limiar
    de decisão que maximiza F1 em vez de usar o padrão (0.5).

    Usa `cross_val_predict` (não `cross_validate`) pra pegar a
    probabilidade prevista de cada linha quando ela estava no fold de
    TESTE — nunca a probabilidade de uma previsão vazada do próprio
    treino.

    Limitação honesta, não escondida: o limiar é escolhido usando essas
    MESMAS probabilidades fora-da-dobra que depois viram a métrica
    reportada — um viés otimista pequeno (o limiar "conhece" um pouco do
    resultado que está sendo medido, já que os dois vêm da mesma rodada
    de CV). Não é uma validação totalmente independente (isso exigiria
    escolher o limiar dentro de cada fold de treino, um CV aninhado). Mas
    é o padrão comum na prática pra esse tipo de ajuste, e bem mais
    honesto que usar 0.5 sem questionar.
    """
    cv = StratifiedGroupKFold(n_splits=n_splits)
    probabilities = cross_val_predict(
        pipeline,
        features,
        target,
        groups=groups,
        cv=cv,
        method="predict_proba",
    )[:, 1]

    threshold = find_best_threshold(probabilities, target)
    predictions = probabilities >= threshold

    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
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
