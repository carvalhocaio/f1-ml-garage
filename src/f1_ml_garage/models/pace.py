"""Regressão linear de tempo de volta (Módulo 2 — aprendizado supervisionado).

O modelo em si é deliberadamente simples (`sklearn.linear_model.LinearRegression`
puro, sem regularização) — o ponto do Módulo 2 aqui não é maximizar
acurácia, é entender o efeito de composto/idade de pneu no ritmo, e fazer a
avaliação corretamente. Regularização (Ridge/Lasso) e modelos não-lineares
entram como comparação depois, não como primeira tentativa.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline

DEFAULT_N_SPLITS = 5


def build_pace_pipeline() -> Pipeline:
    """Pipeline do modelo. Um único passo por enquanto (`LinearRegression`),
    mas fica em `Pipeline` desde já - quando Ridge/Lasso entrarem em cena,
    um passo de escala de features (`StandardScaler`) vem antes, e trocar
    de `LinearRegression` para `Ridge` não deve mudar mais nada ao redor.
    """
    return Pipeline([("model", LinearRegression())])


def evaluate_pace_model(
    features: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
    n_splits: int = DEFAULT_N_SPLITS,
) -> dict[str, float]:
    """Avalia o modelo de ritmo com `GroupKFold` agrupado por piloto.

    Por que `GroupKFold` e não `KFold` comum: um `KFold` aleatório pode
    colocar a volta 12 do stint de um piloto no treino e a volta 13 (quase
    idêntica, mesmo stint, mesmo composto, 1 volta de diferença de idade de
    pneu) no teste. O modelo "acerta" por memorizar o vizinho, não por
    aprender a relação composto/idade -> tempo. Agrupar por piloto garante
    que nenhum piloto apareça simultaneamente em treino e teste em nenhum
    fold - o modelo é avaliado na sua capacidade de generalizar para
    pilotos que não viu, não de interpolar entre voltar vizinhas.

    Retorna MAE (segundos) e R², média e desvio padrão entre os folds.
    """
    pipeline = build_pace_pipeline()
    cv = GroupKFold(n_splits=n_splits)
    scores = cross_validate(
        pipeline,
        features,
        target,
        groups=groups,
        cv=cv,
        scoring=("neg_mean_absolute_error", "r2"),
    )

    mae_scores = -scores["test_neg_mean_absolute_error"]
    r2_scores = scores["test_r2"]

    return {
        "mae_s": float(np.mean(mae_scores)),
        "mae_s_std": float(np.std(mae_scores)),
        "r2": float(np.mean(r2_scores)),
        "r2_std": float(np.std(r2_scores)),
    }
