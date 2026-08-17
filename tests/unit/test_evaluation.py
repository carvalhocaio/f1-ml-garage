import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from f1_ml_garage.models.dnf import (
    build_dnf_boosting_pipeline,
    build_dnf_random_forest_pipeline,
    build_dnf_tree_pipeline,
    compute_scale_pos_weight,
)
from f1_ml_garage.models.evaluation import (
    compute_calibration,
    compute_roc_auc,
    evaluate_classifier,
    evaluate_classifier_with_tuned_threshold,
    find_best_threshold,
)


@pytest.mark.unit
def test_find_best_threshold_matches_a_fine_grid_search():
    """Verificação tipo oráculo: o limiar escolhido tem que dar um F1 pelo
    menos tão bom quanto o melhor limiar encontrado varrendo uma grade
    fina de candidatos na mão — se `find_best_threshold` estiver errado,
    a busca em grade encontraria algo melhor."""
    rng = np.random.default_rng(0)
    probabilities = rng.random(500)
    target = pd.Series(rng.random(500) < probabilities)

    threshold = find_best_threshold(probabilities, target)
    f1_at_threshold = f1_score(target, probabilities >= threshold)

    grid = np.linspace(0.01, 0.99, 99)
    best_grid_f1 = max(f1_score(target, probabilities >= t) for t in grid)

    assert f1_at_threshold >= best_grid_f1 - 1e-6


@pytest.mark.unit
def test_find_best_threshold_returns_value_in_valid_range():
    rng = np.random.default_rng(1)
    probabilities = rng.random(200)
    target = pd.Series(rng.random(200) < 0.3)

    threshold = find_best_threshold(probabilities, target)

    assert 0.0 <= threshold <= 1.0


N_DRIVERS = 20
RACES_PER_DRIVER = 6
TEAMS = ("Red Bull Racing", "Mercedes", "Ferrari", "McLaren", "TeamX")
HIGH_RISK_TEAM = "TeamX"
HIGH_RISK_GRID = 18.0


def _separable_dnf_dataset() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Mesmo dataset sintético separável de `test_dnf_model.py`: regra
    determinística (sem ruído) — DNF se o grid é >= 18 OU o time é
    "TeamX". Serve de oráculo pras métricas novas (ROC-AUC, calibração)
    aqui: num padrão perfeitamente separável, ROC-AUC deveria ser 1.0 e
    Brier score deveria ser ~0.
    """
    rows = []
    for driver_idx in range(N_DRIVERS):
        team = TEAMS[driver_idx % len(TEAMS)]
        for race in range(RACES_PER_DRIVER):
            grid_position = float(((driver_idx + race) % 20) + 1)
            dnf = (grid_position >= HIGH_RISK_GRID) or (team == HIGH_RISK_TEAM)
            rows.append(
                {
                    "driver": f"D{driver_idx}",
                    "team": team,
                    "grid_position": grid_position,
                    "dnf": dnf,
                }
            )

    data = pd.DataFrame(rows)
    features = pd.concat(
        [data[["grid_position"]], pd.get_dummies(data["team"], prefix="team")],
        axis=1,
    )
    return features, data["dnf"], data["driver"]


def _noisy_imbalanced_dataset() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Mesmo dataset sintético de `test_dnf_model.py` — ruído real,
    desbalanceamento real (~11% de classe positiva)."""
    rng = np.random.default_rng(42)
    n = 200
    grid_position = rng.integers(1, 21, size=n).astype(float)
    teams = np.array(["Red Bull Racing", "Mercedes", "Ferrari", "McLaren", "Haas"])
    team = rng.choice(teams, size=n)
    dnf_probability = np.where(grid_position >= 15, 0.35, 0.05)
    dnf = pd.Series(rng.random(n) < dnf_probability)

    features = pd.concat(
        [
            pd.Series(grid_position, name="grid_position"),
            pd.get_dummies(pd.Series(team), prefix="team"),
        ],
        axis=1,
    )
    groups = pd.Series([f"D{i}" for i in range(n)])
    return features, dnf, groups


@pytest.mark.unit
def test_evaluate_classifier_with_tuned_threshold_returns_expected_keys():
    features, target, groups = _noisy_imbalanced_dataset()
    result = evaluate_classifier_with_tuned_threshold(
        build_dnf_random_forest_pipeline(), features, target, groups, n_splits=5
    )

    assert set(result.keys()) == {
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "dnf_rate",
    }


@pytest.mark.unit
def test_tuned_threshold_improves_f1_over_default_cutoff():
    """A descoberta real documentada em `docs/02-dnf-model.md` (seção de
    ensembles): presos no corte padrão de 0.5, modelos com mais
    capacidade (Random Forest) inclinam pra classe majoritária com classe
    rara. Ajustar o limiar deveria recuperar F1 sem precisar mudar o
    modelo."""
    features, target, groups = _noisy_imbalanced_dataset()
    pipeline = build_dnf_random_forest_pipeline()

    default_cutoff = evaluate_classifier(pipeline, features, target, groups, n_splits=5)
    tuned = evaluate_classifier_with_tuned_threshold(
        pipeline, features, target, groups, n_splits=5
    )

    assert tuned["f1"] >= default_cutoff["f1"]


@pytest.mark.unit
def test_tuned_threshold_shifts_away_from_default_for_boosting():
    """No XGBoost especificamente (o caso mais extremo documentado — F1
    caiu de 0.300 pra 0.206 preso no limiar padrão), o limiar ajustado
    deveria se afastar bastante de 0.5, não só ficar perto."""
    features, target, groups = _noisy_imbalanced_dataset()
    scale_pos_weight = compute_scale_pos_weight(target)
    pipeline = build_dnf_boosting_pipeline(scale_pos_weight=scale_pos_weight)

    tuned = evaluate_classifier_with_tuned_threshold(
        pipeline, features, target, groups, n_splits=5
    )

    assert tuned["threshold"] < 0.3


@pytest.mark.unit
def test_roc_auc_is_perfect_for_separable_pattern():
    features, target, groups = _separable_dnf_dataset()
    roc_auc = compute_roc_auc(
        build_dnf_tree_pipeline(), features, target, groups, n_splits=4
    )

    assert roc_auc == pytest.approx(1.0, abs=1e-6)


@pytest.mark.unit
def test_roc_auc_is_above_chance_for_noisy_data():
    """0.5 = não faz melhor que sorteio. O padrão de grid_position/team
    injetado no dataset ruidoso é real, então o ROC-AUC tem que ficar
    visivelmente acima de 0.5, mesmo sem ser perfeito."""
    features, target, groups = _noisy_imbalanced_dataset()
    roc_auc = compute_roc_auc(
        build_dnf_tree_pipeline(), features, target, groups, n_splits=5
    )

    assert 0.5 < roc_auc < 1.0


@pytest.mark.unit
def test_compute_calibration_returns_expected_keys():
    features, target, groups = _noisy_imbalanced_dataset()
    result = compute_calibration(
        build_dnf_tree_pipeline(), features, target, groups, n_splits=5, n_bins=3
    )

    assert set(result.keys()) == {
        "brier_score",
        "fraction_of_positives",
        "mean_predicted_value",
    }
    assert 0.0 <= result["brier_score"] <= 1.0
    assert len(result["fraction_of_positives"]) == len(result["mean_predicted_value"])


@pytest.mark.unit
def test_compute_calibration_brier_score_near_zero_for_separable_pattern():
    """Padrão perfeitamente separável -> probabilidades previstas deveriam
    bater bem perto do resultado real (0 ou 1), Brier score baixo."""
    features, target, groups = _separable_dnf_dataset()
    result = compute_calibration(
        build_dnf_tree_pipeline(), features, target, groups, n_splits=4, n_bins=3
    )

    assert result["brier_score"] < 0.05
