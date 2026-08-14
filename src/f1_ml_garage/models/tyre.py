"""SVM para classificar composto de pneu a partir de telemetria (Módulo 2).

SVM é sensível à escala das features por construção — a margem depende de
distância euclidiana entre pontos, e uma feature em escala maior (RPM, na
casa dos milhares) dominaria uma em escala bem menor (fração de frenagem,
0-1) sem relação com o efeito real. `StandardScaler` aqui não é opcional
como era pra árvore/regressão logística com dummies — é parte de fazer o
algoritmo funcionar direito, aplicado a TODAS as features (diferente de
`dnf.py`, aqui não tem dummy 0/1 misturado com contínua).
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DEFAULT_KERNEL = "rbf"
DEFAULT_C = 1.0
DEFAULT_CLASS_WEIGHT = "balanced"


def build_tyre_svm_pipeline(
    kernel: str = DEFAULT_KERNEL,
    C: float = DEFAULT_C,  # noqa: N803 -- nome padrão do sklearn (SVC)
    class_weight: str | None = DEFAULT_CLASS_WEIGHT,
) -> Pipeline:
    """Pipeline do SVM.

    `kernel="rbf"` (não-linear) é o padrão do scikit-learn e um bom
    primeiro teste — se um kernel linear (`kernel="linear"`) tiver
    desempenho parecido, é sinal de que a fronteira entre compostos é
    aproximadamente linear no espaço dessas features, sem precisar da
    flexibilidade extra do RBF.

    `class_weight="balanced"` pelo mesmo motivo de sempre: nada garante
    que os 3 compostos aparecem em proporções parecidas numa corrida (viu
    isso na prática com "medium" ausente no Bahrain 2024 — `docs/02-dnf-
    model.md`).
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                SVC(kernel=kernel, C=C, class_weight=class_weight, random_state=0),
            ),
        ]
    )
