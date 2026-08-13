import numpy as np
import pandas as pd
import pytest

from f1_ml_garage.models.pace import (
    evaluate_pace_model,
    fit_pace_model,
    pace_coefficients,
)

N_DRIVERS = 6
LAPS_PER_DRIVER = 12
COMPOUND_OFFSET_S = {"soft": -0.5, "medium": 0.0, "hard": 0.5}
TYRE_DEGRADATION_S_PER_LAP = 0.03
BASE_LAP_TIME_S = 90.0


def _synthetic_pace_dataset() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Gera um dataset sintético onde `lap_time_s` é uma função linear
    EXATA (sem ruído) de `tyre_life` e `compound`, espalhada por vários
    pilotos. Serve de oráculo: se o modelo e a avaliação estão corretos,
    `evaluate_pace_model` tem que recuperar essa relação quase
    perfeitamente (R² ≈ 1, MAE ≈ 0) — o mesmo espírito das validações
    contra oráculo do `cotton-math-lab`, adaptado pra "relação conhecida
    por construção" em vez de "biblioteca de referência".
    """
    compounds = ["soft", "medium", "hard"]
    rows = []
    for driver_idx in range(N_DRIVERS):
        driver = f"D{driver_idx}"
        for lap in range(1, LAPS_PER_DRIVER + 1):
            compound = compounds[lap % len(compounds)]
            tyre_life = float(lap)
            lap_time = (
                BASE_LAP_TIME_S
                + TYRE_DEGRADATION_S_PER_LAP * tyre_life
                + COMPOUND_OFFSET_S[compound]
            )
            rows.append(
                {
                    "driver": driver,
                    "lap_time_s": lap_time,
                    "compound": compound,
                    "tyre_life": tyre_life,
                }
            )

    laps = pd.DataFrame(rows)

    compound_dummies = pd.get_dummies(
        pd.Categorical(laps["compound"], categories=compounds), prefix="compound"
    )
    features = pd.concat([laps[["tyre_life"]], compound_dummies], axis=1)
    target = laps["lap_time_s"]
    groups = laps["driver"]
    return features, target, groups


@pytest.mark.unit
def test_recovers_known_linear_relationship_without_noise():
    features, target, groups = _synthetic_pace_dataset()

    result = evaluate_pace_model(features, target, groups, n_splits=3)

    assert result["r2"] == pytest.approx(1.0, abs=1e-6)
    assert result["mae_s"] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.unit
def test_returns_expected_metric_keys():
    features, target, groups = _synthetic_pace_dataset()
    result = evaluate_pace_model(features, target, groups, n_splits=3)

    assert set(result.keys()) == {"mae_s", "mae_s_std", "r2", "r2_std"}
    assert all(isinstance(v, float) for v in result.values())


@pytest.mark.unit
def test_raises_when_more_splits_than_groups():
    """GroupKFold exige n_splits <= número de grupos distintos; queremos
    que esse erro se propague claramente, não seja mascarado."""
    features, target, groups = _synthetic_pace_dataset()

    with pytest.raises(ValueError):
        evaluate_pace_model(features, target, groups, n_splits=N_DRIVERS + 1)


@pytest.mark.unit
def test_worse_model_on_shuffled_target_has_lower_r2():
    """Sanity check na direção oposta: embaralhar o alvo quebra a relação
    real, e o modelo tem que performar visivelmente pior — garante que o
    teste anterior não passaria "por acidente" com qualquer entrada."""
    features, target, groups = _synthetic_pace_dataset()
    shuffled_target = target.sample(frac=1, random_state=42).reset_index(drop=True)

    real = evaluate_pace_model(features, target, groups, n_splits=3)
    shuffled = evaluate_pace_model(features, shuffled_target, groups, n_splits=3)

    assert real["r2"] > shuffled["r2"]
    assert not np.isclose(shuffled["r2"], 1.0, atol=1e-3)


LAP_NUMBER_COEF_S = 0.02
TYRE_LIFE_COEF_S = 0.01
SOFT_OFFSET_S = -0.3
HARD_OFFSET_S = 0.4


def _known_effects_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Dataset minimalista, já no formato de X/y (sem passar por
    `build_pace_features`), com efeitos conhecidos por construção e sem
    ruído — mesmo espírito oráculo de `_synthetic_pace_dataset`, mas focado
    em validar a EXTRAÇÃO de coeficientes, não a avaliação por CV.
    """
    lap_numbers = list(range(1, 21)) * 3
    compounds = (["soft"] * 20) + (["medium"] * 20) + (["hard"] * 20)
    tyre_lives = [float((i % 8) + 1) for i in range(60)]
    offsets = {"soft": SOFT_OFFSET_S, "medium": 0.0, "hard": HARD_OFFSET_S}

    target = pd.Series(
        [
            LAP_NUMBER_COEF_S * ln + TYRE_LIFE_COEF_S * tl + offsets[c]
            for ln, tl, c in zip(lap_numbers, tyre_lives, compounds, strict=True)
        ]
    )
    features = pd.DataFrame(
        {
            "tyre_life": tyre_lives,
            "lap_number": [float(ln) for ln in lap_numbers],
            "compound_soft": [1.0 if c == "soft" else 0.0 for c in compounds],
            "compound_hard": [1.0 if c == "hard" else 0.0 for c in compounds],
        }
    )
    return features, target


@pytest.mark.unit
def test_pace_coefficients_recovers_known_effects():
    features, target = _known_effects_dataset()

    pipeline = fit_pace_model(features, target)
    coefficients = pace_coefficients(pipeline, features.columns)

    assert coefficients["lap_number"] == pytest.approx(LAP_NUMBER_COEF_S, abs=1e-9)
    assert coefficients["tyre_life"] == pytest.approx(TYRE_LIFE_COEF_S, abs=1e-9)
    assert coefficients["compound_soft"] == pytest.approx(SOFT_OFFSET_S, abs=1e-9)
    assert coefficients["compound_hard"] == pytest.approx(HARD_OFFSET_S, abs=1e-9)
    assert coefficients["intercept"] == pytest.approx(0.0, abs=1e-9)
