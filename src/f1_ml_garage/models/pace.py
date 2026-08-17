"""Regressão linear de tempo de volta (Módulo 2 — aprendizado supervisionado).

Três variantes: `LinearRegression` pura (sem regularização, a original),
`Ridge` (L2) e `Lasso` (L1) — todas avaliadas pela mesma
`evaluate_pace_model`, que recebe o pipeline como parâmetro (não fixa um
modelo por dentro) pra reusar a lógica de CV entre as três sem duplicar,
mesmo padrão de `evaluate_classifier` (`models/evaluation.py`).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_N_SPLITS = 5
DEFAULT_ALPHA = 1.0


def build_pace_pipeline() -> Pipeline:
    """Pipeline do modelo original: `LinearRegression` pura, sem
    regularização — o ponto do Módulo 2 aqui não é maximizar acurácia, é
    entender o efeito de composto/idade de pneu no ritmo, e fazer a
    avaliação corretamente.

    `fit_intercept=False` é proposital, não default: `build_pace_features`
    gera uma dummy pra CADA composto (não droppa nenhuma como referência).
    Com intercepto, isso seria colinearidade perfeita — mas mais sutil
    ainda, dropar uma referência fixa e manter o intercepto só funciona se
    essa referência aparecer nos dados; numa corrida onde ela nunca é usada
    (ex.: Bahrain 2024 sem laps de "medium"), a colinearidade reaparece via
    os dados, não o encoding (ver `features/pace.py`). Sem intercepto
    compartilhado, cada composto carrega seu próprio coeficiente — sempre
    identificável, mesmo quando algum composto está ausente do subconjunto
    (só o coeficiente DAQUELE composto ausente fica sem sentido, o que é
    esperado: não dá pra estimar efeito de um composto nunca observado).
    """
    return Pipeline([("model", LinearRegression(fit_intercept=False))])


def _build_regularized_pipeline(model) -> Pipeline:
    """Monta um pipeline de regressão regularizada (Ridge ou Lasso) com
    TODAS as features escaladas — diferente do resto do módulo (que deixa
    dummies de composto sem escala, pra manter coeficientes diretamente em
    segundos em `build_pace_pipeline`), aqui isso não é opcional.

    A penalização de Ridge/Lasso soma o valor (L1) ou quadrado (L2) de
    TODOS os coeficientes. Dummies de composto sem escala precisam de
    coeficientes enormes (~85-90, a própria ordem de grandeza do tempo de
    volta) comparados a `tyre_life`/`lap_number` sem escala (~0.01-0.05) —
    a penalização esmagaria as duas variáveis contínuas sistematicamente,
    não por serem menos importantes, só por estarem numa escala menor.
    Confirmado na prática antes de fixar este design: com dummies sem
    escala, `alpha=1.0` (um padrão razoável de qualquer forma) zerava
    `tyre_life`/`lap_number` por completo.

    Custo dessa escolha: os coeficientes de Ridge/Lasso saem em unidades
    padronizadas pra TODAS as features, não mais diretamente em segundos
    como em `build_pace_pipeline` (OLS) — `pace_coefficients` ainda
    funciona tecnicamente, mas a leitura "X segundos por unidade" só vale
    pro modelo sem regularização.
    """
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def build_pace_ridge_pipeline(alpha: float = DEFAULT_ALPHA) -> Pipeline:
    """Ridge (regularização L2) pro modelo de ritmo.

    Encolhe os coeficientes em direção a zero proporcionalmente ao
    quadrado de cada um — reduz variância (sensibilidade a ruído no
    treino) ao custo de um pouco de viés, o trade-off bias-variance de
    novo, agora via penalização em vez de profundidade de árvore
    (`models/dnf.py`) ou `n_estimators`.

    `fit_intercept=True`, DIFERENTE de `build_pace_pipeline` — e não é só
    uma troca de padrão. `LinearRegression` usava `fit_intercept=False` +
    3 dummies de composto de propósito, porque mínimos quadrados puro
    precisa da colinearidade resolvida manualmente (ver
    `features/pace.py`). Ridge não tem esse problema: a penalização L2
    resolve colinearidade automaticamente (soma um valor positivo à
    diagonal da matriz normal, tornando-a invertível mesmo com
    redundância) — é uma propriedade conhecida de Ridge, não um acaso.
    Testado nos dois formatos antes de fixar este design:
    `fit_intercept=False` com todas as features escaladas (inclusive os
    dummies) QUEBRA o modelo — padronizar os dummies remove exatamente a
    propriedade de "valor bruto 1" que fazia eles funcionarem como um
    intercepto disfarçado, e sem intercepto real o modelo não consegue
    mais representar nenhum baseline diferente de zero.

    `alpha` maior = mais regularização (coeficientes mais perto de zero);
    `alpha=0` equivaleria a uma regressão linear comum (com intercepto).
    """
    return _build_regularized_pipeline(Ridge(alpha=alpha, fit_intercept=True))


def build_pace_lasso_pipeline(alpha: float = DEFAULT_ALPHA) -> Pipeline:
    """Lasso (regularização L1) pro modelo de ritmo.

    Mesma ideia de Ridge (`fit_intercept=True`, mesmo motivo — ver
    docstring de `build_pace_ridge_pipeline`), mas a penalização (valor
    absoluto, não quadrado) tem uma propriedade diferente: pode zerar
    coeficientes por completo, não só encolher — funciona como seleção de
    features implícita. Com só 5 features aqui (`tyre_life`, `lap_number`,
    3 dummies de composto), zerar alguma seria um sinal de que ela não
    carrega informação independente das outras.
    """
    return _build_regularized_pipeline(Lasso(alpha=alpha, fit_intercept=True))


def evaluate_pace_model(
    pipeline: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
    n_splits: int = DEFAULT_N_SPLITS,
) -> dict[str, float]:
    """Avalia um modelo de ritmo (`build_pace_pipeline`,
    `build_pace_ridge_pipeline` ou `build_pace_lasso_pipeline`) com
    `GroupKFold` agrupado por piloto.

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


def fit_pace_model(
    pipeline: Pipeline, features: pd.DataFrame, target: pd.Series
) -> Pipeline:
    """Ajusta o pipeline em todos os dados fornecidos, sem hold-out.

    Não serve para medir performance - isso é `evaluate_pace_model` com
    `GroupKFold`. Serve para inspecionar os coeficientes ajustados, que é o
    ponto real do Módulo 2: entender o efeito de cada feature no ritmo, não
    só o quão bem o modelo genereliza.
    """
    pipeline.fit(features, target)
    return pipeline


def pace_coefficients(pipeline: Pipeline, feature_names: pd.Index) -> pd.Series:
    """Extrai os coeficientes do modelo linear ajustado, indexados pelo
    nome da feature — mais o intercepto, quando o modelo tem um
    (`build_pace_ridge_pipeline`/`build_pace_lasso_pipeline`; `
    build_pace_pipeline` não, por design — ver sua docstring).

    Sem intercepto (OLS, `build_pace_pipeline`), não há "referência" — o
    coeficiente de cada dummy de composto já é o efeito absoluto daquele
    composto, não uma diferença em relação a outro. `tyre_life` e
    `lap_number` continuam com um único coeficiente compartilhado entre
    todos os compostos (o modelo assume mesma taxa de degradação e mesmo
    efeito de combustível/evolução de pista pra qualquer composto — uma
    simplificação; interações composto×degradação ficam pra uma iteração
    futura).

    Interpretação (alvo é o delta de ritmo em segundos, ver
    `compute_driver_delta_target`): cada coeficiente é quantos segundos por
    unidade daquela feature a volta fica mais lenta (positivo) ou mais
    rápida (negativo), mantendo as outras features constantes. Essa leitura
    direta em segundos só vale pro modelo SEM regularização
    (`build_pace_pipeline`) — em Ridge/Lasso, TODAS as features são
    escaladas antes do modelo (`_build_regularized_pipeline`) e há um
    intercepto de verdade, então os coeficientes saem em unidades
    padronizadas, comparáveis entre si mas não diretamente em "segundos
    por unidade bruta", e os dummies de composto viram CONTRASTES em
    relação ao intercepto, não mais valores absolutos.
    """
    model = pipeline.named_steps["model"]
    coefficients = pd.Series(model.coef_, index=feature_names, name="coef_s")
    if getattr(model, "fit_intercept", False):
        coefficients["intercept"] = model.intercept_
    return coefficients
