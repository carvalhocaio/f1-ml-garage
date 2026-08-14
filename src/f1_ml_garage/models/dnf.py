"""Classificadores de DNF (Módulo 2 — aprendizado supervisionado).

Dois modelos pro mesmo problema — árvore de decisão e regressão logística
— pra comparar diretamente. Dois pontos do currículo se encontram aqui além
disso: dados desbalanceados (DNF é raro) e bias-variance (uma árvore sem
limite de profundidade decora o treino fácil, com poucas features e pouca
amostra).
"""

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

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
