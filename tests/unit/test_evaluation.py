import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from f1_ml_garage.models.dnf import (
    build_dnf_boosting_pipeline,
    build_dnf_random_forest_pipeline,
    compute_scale_pos_weight,
)
from f1_ml_garage.models.evaluation import (
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
