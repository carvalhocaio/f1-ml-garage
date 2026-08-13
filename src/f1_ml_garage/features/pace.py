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

    `lap_number` entra como proxy de dois efeitos sistemáticos que não
    temos direto no schema: carga de combustível (carro fica mais leve e
    mais rápido ao longo da corrida) e evolução de pista (asfalto fica mais
    rápido conforme mais borracha é depositada). Não é colinear com
    `tyre_life` apesar de ambos crescerem "com o tempo": `tyre_life` zera a
    cada pit stop, `lap_number` não — dá pra separar o efeito de pneu novo
    numa volta 30 (tyre_life baixo, lap_number alto) do de pneu novo numa
    volta 3 (os dois baixos).
    """
    compound = pd.Categorical(laps["compound"], categories=COMPOUND_CATEGORIES)
    compound_dummies = pd.get_dummies(compound, prefix="compound")

    features = pd.concat(
        [
            laps[["tyre_life", "lap_number"]].reset_index(drop=True),
            compound_dummies,
        ],
        axis=1,
    )
    target = laps["lap_time_s"].reset_index(drop=True)
    groups = laps["driver"].reset_index(drop=True)

    return features, target, groups


def compute_driver_delta_target(laps: pd.DataFrame) -> pd.Series:
    """Alvo alternativo: delta de tempo de volta em relação à média do
    próprio piloto na sessão (`lap_time_s - média(lap_time_s) por driver`).

    O primeiro modelo, treinado sobre `lap_time_s` bruto, teve R² ≈ 0.02
    numa corrida real (ver `docs/01-pace-model.md`) — a diferença de ritmo
    baseline entre carros/pilotos (facilmente 1-2s/volta em 2024) domina
    tanto a variância que o efeito de composto/idade de pneu (frações de
    segundo) vira ruído por comparação. Centralizar por piloto remove essa
    baseline da variável de resposta, isolando o que queremos medir: o
    efeito de composto/idade de pneu *dentro* do desempenho de cada piloto.

    Não vaza informação entre pilotos: `groupby("driver").transform("mean")`
    usa só as voltas do próprio piloto. E como a avaliação agrupa por
    piloto (`GroupKFold`), todas as voltas de um piloto já ficam inteiramente
    de um lado do split — a média usada pra centralizar nunca mistura dado
    de treino com dado de teste.
    """
    driver_mean = laps.groupby("driver")["lap_time_s"].transform("mean")
    return (laps["lap_time_s"] - driver_mean).reset_index(drop=True)
