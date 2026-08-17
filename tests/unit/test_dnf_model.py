import numpy as np
import pandas as pd
import pytest

from f1_ml_garage.models.dnf import (
    build_dnf_boosting_pipeline,
    build_dnf_logistic_pipeline,
    build_dnf_random_forest_pipeline,
    build_dnf_stacking_pipeline,
    build_dnf_tree_pipeline,
    compute_scale_pos_weight,
)
from f1_ml_garage.models.evaluation import (
    evaluate_classifier,
    evaluate_classifier_with_tuned_threshold,
)

N_DRIVERS = 20
RACES_PER_DRIVER = 6
TEAMS = ("Red Bull Racing", "Mercedes", "Ferrari", "McLaren", "TeamX")
HIGH_RISK_TEAM = "TeamX"
HIGH_RISK_GRID = 18.0


def _separable_dnf_dataset() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Dataset sintético com uma regra determinística (sem ruído) por
    construção: DNF se o grid é >= 18 OU o time é "TeamX". Serve de
    oráculo — uma árvore rasa o suficiente pra representar um OR de duas
    condições (profundidade 2 já basta) tem que recuperar isso quase
    perfeitamente. Regressão logística, sendo uma fronteira suave (não
    cortes exatos), não tem obrigação de chegar a 1.0 no mesmo padrão —
    ver `test_logistic_recovers_known_separable_pattern`.
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


@pytest.mark.unit
def test_tree_recovers_known_separable_pattern():
    features, target, groups = _separable_dnf_dataset()

    result = evaluate_classifier(
        build_dnf_tree_pipeline(), features, target, groups, n_splits=4
    )

    assert result["recall"] == pytest.approx(1.0, abs=1e-6)
    assert result["precision"] == pytest.approx(1.0, abs=1e-6)
    assert result["f1"] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.unit
def test_logistic_recovers_known_separable_pattern():
    """Regressão logística ajusta uma fronteira suave (sigmoide), não
    cortes exatos como a árvore — não tem por que chegar a 1.0 num padrão
    de regra dura (OR de duas condições), mas tem que capturar a maior
    parte dele."""
    features, target, groups = _separable_dnf_dataset()

    result = evaluate_classifier(
        build_dnf_logistic_pipeline(), features, target, groups, n_splits=4
    )

    assert result["recall"] > 0.85
    assert result["precision"] > 0.6
    assert result["f1"] > 0.75


@pytest.mark.unit
def test_returns_expected_metric_keys():
    features, target, groups = _separable_dnf_dataset()
    result = evaluate_classifier(
        build_dnf_tree_pipeline(), features, target, groups, n_splits=4
    )

    assert set(result.keys()) == {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "dnf_rate",
    }


@pytest.mark.unit
def test_random_forest_recovers_known_separable_pattern():
    """Bagging (várias árvores em amostras bootstrap diferentes, voto
    majoritário) deveria recuperar o padrão quase tão bem quanto uma
    árvore única — não perfeito, o "quase" vem do próprio bagging
    suavizando decisões perto da fronteira."""
    features, target, groups = _separable_dnf_dataset()

    result = evaluate_classifier(
        build_dnf_random_forest_pipeline(), features, target, groups, n_splits=4
    )

    assert result["recall"] > 0.85
    assert result["precision"] > 0.9
    assert result["f1"] > 0.85


@pytest.mark.unit
def test_boosting_recovers_known_separable_pattern():
    """Boosting (árvores em sequência, cada uma corrigindo o resíduo da
    anterior) tem capacidade de representar o padrão quase perfeitamente
    com estimadores suficientes — diferente da regressão logística
    (fronteira suave), não tem por que ficar abaixo de 1.0 aqui."""
    features, target, groups = _separable_dnf_dataset()
    scale_pos_weight = compute_scale_pos_weight(target)

    result = evaluate_classifier(
        build_dnf_boosting_pipeline(scale_pos_weight=scale_pos_weight),
        features,
        target,
        groups,
        n_splits=4,
    )

    assert result["recall"] > 0.95
    assert result["precision"] > 0.95
    assert result["f1"] > 0.95


@pytest.mark.unit
def test_compute_scale_pos_weight_matches_negative_over_positive_ratio():
    target = pd.Series([True, True, False, False, False, False])
    # 2 positivos, 4 negativos -> 4/2 = 2.0
    assert compute_scale_pos_weight(target) == pytest.approx(2.0)


def _noisy_imbalanced_dataset() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Dataset sintético com ruído e desbalanceamento real (~11% DNF): grid
    alto AUMENTA a chance de DNF, não garante — diferente do dataset
    separável acima, de propósito, pra expor a diferença entre
    `class_weight="balanced"` e sem peso nenhum.
    """
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
def test_class_weight_balanced_improves_recall_on_imbalanced_data():
    """A lição de "dados desbalanceados" do currículo, em números: sem
    peso de classe, o modelo prefere prever "não abandona" quase sempre —
    accuracy parece boa, mas recall (a métrica que importa pra um evento
    raro) desmorona. `class_weight="balanced"` troca uma fatia de accuracy
    por muito mais recall — o trade-off certo quando o custo de não
    detectar um DNF é maior que o de um falso positivo.
    """
    features, target, groups = _noisy_imbalanced_dataset()

    balanced = evaluate_classifier(
        build_dnf_tree_pipeline(class_weight="balanced"),
        features,
        target,
        groups,
        n_splits=5,
    )
    unweighted = evaluate_classifier(
        build_dnf_tree_pipeline(class_weight=None),
        features,
        target,
        groups,
        n_splits=5,
    )

    assert balanced["recall"] > 0.5
    assert unweighted["recall"] < 0.3
    assert balanced["recall"] > unweighted["recall"]


@pytest.mark.unit
def test_scale_pos_weight_improves_recall_on_imbalanced_data():
    """Mesma lição do teste acima, agora pro XGBoost — que não tem
    `class_weight="balanced"`, usa `scale_pos_weight` no lugar
    (`compute_scale_pos_weight`)."""
    features, target, groups = _noisy_imbalanced_dataset()
    scale_pos_weight = compute_scale_pos_weight(target)

    weighted = evaluate_classifier(
        build_dnf_boosting_pipeline(scale_pos_weight=scale_pos_weight),
        features,
        target,
        groups,
        n_splits=5,
    )
    unweighted = evaluate_classifier(
        build_dnf_boosting_pipeline(scale_pos_weight=1.0),
        features,
        target,
        groups,
        n_splits=5,
    )

    assert weighted["recall"] > unweighted["recall"]


@pytest.mark.unit
def test_stacking_recovers_known_separable_pattern():
    """Stacking combina árvore/logística/Random Forest/XGBoost enxuto via
    um meta-modelo — num padrão perfeitamente separável, deveria recuperar
    tão bem quanto os modelos base individuais recuperam sozinhos."""
    features, target, groups = _separable_dnf_dataset()
    scale_pos_weight = compute_scale_pos_weight(target)

    result = evaluate_classifier(
        build_dnf_stacking_pipeline(scale_pos_weight=scale_pos_weight),
        features,
        target,
        groups,
        n_splits=4,
    )

    assert result["recall"] == pytest.approx(1.0, abs=1e-6)
    assert result["precision"] == pytest.approx(1.0, abs=1e-6)
    assert result["f1"] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.unit
def test_stacking_with_tuned_threshold_beats_zero_recall_default():
    """No limiar padrão (0.5), o stacking também zera recall no cenário
    ruidoso/desbalanceado — mesma lição de sempre (árvore, logística, RF,
    XGBoost, todos precisaram de ajuste de limiar aqui). Com o limiar
    ajustado, tem que recuperar sinal de verdade."""
    features, target, groups = _noisy_imbalanced_dataset()
    scale_pos_weight = compute_scale_pos_weight(target)
    pipeline = build_dnf_stacking_pipeline(scale_pos_weight=scale_pos_weight)

    default_cutoff = evaluate_classifier(pipeline, features, target, groups, n_splits=5)
    tuned = evaluate_classifier_with_tuned_threshold(
        pipeline, features, target, groups, n_splits=5
    )

    assert default_cutoff["f1"] == pytest.approx(0.0, abs=1e-6)
    assert tuned["f1"] > 0.3
