"""Features para o modelo de ritmo de corrida (Módulo 2, supervisionado).

Feature engineering aqui é deliberadamente mínimo — só o necessário pra dar
munição real ao primeiro modelo (regressão linear de tempo de volta). Isso
não é o Módulo 5 (feature engineering/seleção de features) completo; é o
que o Módulo 2 precisa agora, e vai crescer junto com os modelos.
"""

import pandas as pd

from f1_ml_garage.data.laps import filter_accurate_laps

# Compostos fixos, não inferidos dos dados: se um subconjunto de treino não
# tiver voltas de "hard" (comum em corridas sem 3 stints), `pd.get_dummies`
# sozinho geraria uma matriz de features sem a coluna `compound_hard` — e o
# modelo quebraria ao prever num conjunto que tenha essa coluna. Fixar as
# categorias garante o mesmo shape de X sempre, treino ou predição.
COMPOUND_CATEGORIES = ("soft", "medium", "hard")


def select_green_flag_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Restringe a voltas cronometradas confiáveis (`filter_accurate_laps`)
    E sob bandeira verde limpa (`track_status == "1"`, sem nenhuma outra
    flag concatenada na mesma volta — amarela, safety car, VSC).

    Esse é o subconjunto onde tempo de volta reflete só ritmo de carro/pneu,
    e não um incidente de pista. Fora daqui, `lap_time_s` mistura dois
    fenômenos causais diferentes na mesma variável.
    """
    accurate = filter_accurate_laps(laps)
    return accurate.loc[accurate["track_status"] == "1"].reset_index(drop=True)


def build_pace_features(
    laps: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Monta a matriz de features (X), o alvo (y) e os grupos (piloto).

    Espera receber `laps` já filtrado — ver `select_green_flag_laps`. Não
    filtra de novo aqui, pra manter a função pura e testável com qualquer
    subconjunto controlado nos testes, sem acoplar às duas regras de
    negócio ao mesmo tempo.

    `groups` (o piloto de cada volta) não entra em X: existe pra uso em
    `GroupKFold` na avaliação do modelo, evitando que voltas do mesmo
    piloto apareçam em treino e teste ao mesmo tempo.
    """
    compound = pd.Categorical(laps["compound"], categories=COMPOUND_CATEGORIES)
    compound_dummies = pd.get_dummies(compound, prefix="compound")

    features = pd.concat(
        [laps[["tyre_life"]].reset_index(drop=True), compound_dummies],
        axis=1,
    )
    target = laps["lap_time_s"].reset_index(drop=True)
    groups = laps["driver"].reset_index(drop=True)

    return features, target, groups
