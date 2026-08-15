"""Classificadores de DNF (Módulo 2 — aprendizado supervisionado).

Dois modelos pro mesmo problema — árvore de decisão e regressão logística
— pra comparar diretamente. Dois pontos do currículo se encontram aqui além
disso: dados desbalanceados (DNF é raro) e bias-variance (uma árvore sem
limite de profundidade decora o treino fácil, com poucas features e pouca
amostra).
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Profundidade rasa de propósito: com ~10-15 features (grid_position +
# times one-hot) e algumas centenas de linhas por temporada, uma árvore
# funda decora ruído em vez de aprender padrão — é o mesmo trade-off
# bias-variance do resto do currículo, só que a "regularização" de uma
# árvore é a profundidade, não um termo de penalização.
DEFAULT_MAX_DEPTH = 4

# Primeira técnica de dado desbalanceado a tentar, antes de qualquer coisa
# mais sofisticada (SMOTE, undersampling): pesar as classes pelo inverso da
# frequência. `class_weight="balanced"` faz isso automaticamente a partir
# do `y` de treino, sem precisar calcular a proporção na mão. Vale pros
# dois modelos, não só a árvore.
DEFAULT_CLASS_WEIGHT = "balanced"

DEFAULT_C = 1.0
DEFAULT_N_ESTIMATORS = 200


def build_dnf_tree_pipeline(
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


def build_dnf_logistic_pipeline(
    class_weight: str | None = DEFAULT_CLASS_WEIGHT,
    C: float = DEFAULT_C,  # noqa: N803 -- nome padrão do sklearn (LogisticRegression)
) -> Pipeline:
    """Regressão logística pro mesmo problema de DNF.

    `fit_intercept=False` pelo mesmo motivo do modelo de ritmo
    (`features/pace.py`/`models/pace.py`): `team` entra como one-hot
    COMPLETO (todas as equipes, nenhuma referência dropada) — com
    intercepto, a soma das colunas de time seria sempre 1, colinear com o
    intercepto. Sem ele, cada time carrega seu próprio coeficiente (a base
    de log-odds daquele time), do mesmo jeito que cada composto carrega o
    seu em `pace.py`. Mesmo bug, mesma correção, aplicada de propósito
    desta vez em vez de descoberta depois.

    `grid_position` é escalado (`StandardScaler`) antes do modelo — é a
    única feature contínua, numa escala bem diferente das dummies de
    `team` (1-20 vs 0/1); sem escalar, a penalização L2 (`C`) trataria
    `grid_position` de forma desproporcional só pela escala, não pelo
    efeito real. As dummies de `team` passam sem escala
    (`remainder="passthrough"`), preservando a lógica de "cada time tem seu
    próprio baseline" que o `fit_intercept=False` pressupõe.
    """
    preprocessing = ColumnTransformer(
        transformers=[("scale_grid", StandardScaler(), ["grid_position"])],
        remainder="passthrough",
    )
    return Pipeline(
        [
            ("preprocessing", preprocessing),
            (
                "model",
                LogisticRegression(
                    class_weight=class_weight,
                    C=C,
                    fit_intercept=False,
                    random_state=0,
                    max_iter=1000,
                ),
            ),
        ]
    )


def build_dnf_random_forest_pipeline(
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    max_depth: int = DEFAULT_MAX_DEPTH,
    class_weight: str | None = DEFAULT_CLASS_WEIGHT,
) -> Pipeline:
    """Random Forest: bagging de árvores de decisão.

    Cada árvore treina numa amostra bootstrap diferente do dado (com
    reposição) e enxerga só um subconjunto aleatório de features em cada
    corte — a previsão final é o voto das árvores. Bagging reduz
    VARIÂNCIA (a instabilidade de uma árvore única, que muda bastante com
    pequenas mudanças no dado de treino) sem mudar viés — não conserta
    uma árvore sistematicamente errada, só estabiliza uma árvore instável
    combinando várias.

    `max_depth` continua controlando bias-variance de cada árvore
    individual, mesmo raciocínio de `build_dnf_tree_pipeline` — mantido
    igual (não relaxado) de propósito, pra manter a comparação entre os
    dois modelos justa (mesma profundidade máxima por árvore).
    """
    return Pipeline(
        [
            (
                "model",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    class_weight=class_weight,
                    random_state=0,
                ),
            )
        ]
    )


def compute_scale_pos_weight(target: pd.Series) -> float:
    """Proporção negativos/positivos — o jeito do XGBoost lidar com classe
    desbalanceada (`scale_pos_weight`), diferente do `class_weight=
    "balanced"` que os outros modelos desta suíte usam.

    XGBoost não recalcula isso sozinho a partir do `y` de treino como o
    sklearn faz com `class_weight="balanced"` — precisa ser calculado
    explicitamente e passado. Calculado aqui a partir do dataset completo
    antes da validação cruzada, não por fold — simplificação deliberada:
    proporção de classes é estatística estável (não informação individual
    por exemplo), diferente do tipo de vazamento que de fato importaria
    evitar.
    """
    positive = target.sum()
    negative = len(target) - positive
    return float(negative / positive) if positive > 0 else 1.0


def build_dnf_boosting_pipeline(
    scale_pos_weight: float = 1.0,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> Pipeline:
    """Gradient boosting (XGBoost): árvores em sequência, cada uma
    corrigindo o erro das anteriores.

    Diferença conceitual de bagging (Random Forest): árvores não são
    independentes — cada árvore nova é treinada sobre o RESÍDUO (erro) das
    árvores anteriores, não numa amostra bootstrap independente. Isso
    reduz VIÉS (o conjunto consegue capturar padrões que uma única árvore
    rasa não capturaria, adicionando árvore após árvore), ao custo de mais
    risco de aumentar variância se `n_estimators` for grande demais ou as
    árvores muito profundas — por isso `max_depth` raso importa MAIS aqui
    que no Random Forest (uma árvore profunda numa sequência de boosting
    decora resíduo, não só ruído aleatório de uma amostra bootstrap).

    `scale_pos_weight` (não `class_weight`) é como o XGBoost lida com
    classe rara — ver `compute_scale_pos_weight`. Precisa ser calculado e
    passado explicitamente, diferente de `"balanced"` nos outros modelos.
    """
    return Pipeline(
        [
            (
                "model",
                XGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    scale_pos_weight=scale_pos_weight,
                    random_state=0,
                    eval_metric="logloss",
                ),
            )
        ]
    )
