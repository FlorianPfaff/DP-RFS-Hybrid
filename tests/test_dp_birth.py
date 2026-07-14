import numpy as np
import pytest

from dp_rfs_hybrid import DirichletProcessBirthModel, GaussianState


def make_birth_model(
    clutter_intensity: float = 1e-4,
    birth_rate: float | None = None,
) -> DirichletProcessBirthModel:
    measurement = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )
    return DirichletProcessBirthModel(
        alpha=6.0,
        base_state=GaussianState(
            mean=np.zeros(4),
            covariance=np.diag([100.0, 100.0, 25.0, 25.0]),
        ),
        measurement_matrix=measurement,
        measurement_noise=np.diag([0.25, 0.25]),
        clutter_intensity=clutter_intensity,
        birth_probability=0.8,
        birth_rate=birth_rate,
        odds_threshold=10.0,
    )


def test_new_measurement_creates_birth_atom() -> None:
    model = make_birth_model()

    decision = model.process(np.array([12.0, -3.0]))

    assert decision.accepted
    assert decision.branch == "new"
    assert decision.birth_density is not None
    assert decision.birth_intensity is not None
    assert len(model.atoms) == 1
    assert model.atoms[0].count == 1.0


def test_nearby_measurement_reuses_existing_atom() -> None:
    model = make_birth_model()
    model.process(np.array([12.0, -3.0]))

    decision = model.process(np.array([12.1, -2.9]))

    assert decision.accepted
    assert decision.branch == "existing"
    assert len(model.atoms) == 1
    assert model.atoms[0].count == 2.0


def test_measurement_can_be_rejected_as_clutter() -> None:
    model = make_birth_model(clutter_intensity=10.0)

    decision = model.process(np.array([12.0, -3.0]))

    assert not decision.accepted
    assert decision.branch == "clutter"
    assert len(model.atoms) == 0


def test_birth_rate_scales_odds_without_changing_predictive_density() -> None:
    measurement = np.array([12.0, -3.0])
    low_rate = make_birth_model(birth_rate=1.0)
    high_rate = make_birth_model(birth_rate=6.0)

    low_decision = low_rate.decide(measurement)
    high_decision = high_rate.decide(measurement)

    assert low_decision.birth_density == pytest.approx(high_decision.birth_density)
    assert high_decision.birth_intensity == pytest.approx(6.0 * low_decision.birth_intensity)
    assert high_decision.odds == pytest.approx(6.0 * low_decision.odds)
    assert not low_decision.accepted
    assert high_decision.accepted


def test_default_birth_rate_preserves_legacy_scale() -> None:
    model = make_birth_model()

    assert model.birth_rate == pytest.approx(model.alpha * model.birth_probability)


def test_nearby_confirmed_births_reuse_one_atom() -> None:
    model = make_birth_model()
    first_state = GaussianState(
        mean=np.array([8.0, 2.0, 0.0, 0.0]),
        covariance=np.diag([0.25, 0.25, 1.0, 1.0]),
    )
    second_state = GaussianState(
        mean=np.array([8.2, 2.2, 0.0, 0.0]),
        covariance=np.diag([0.25, 0.25, 1.0, 1.0]),
    )

    first_atom = model.learn_confirmed_state(first_state)
    second_atom = model.learn_confirmed_state(second_state)

    assert first_atom == 0
    assert second_atom == 0
    assert len(model.atoms) == 1
    assert model.atoms[0].count == pytest.approx(2.0)
    assert model.atoms[0].state.mean == pytest.approx(np.array([8.1, 2.1, 0.0, 0.0]))
    assert model.atoms[0].state.covariance[0, 0] > first_state.covariance[0, 0]


def test_distant_confirmed_births_create_separate_atoms() -> None:
    model = make_birth_model()
    first_state = GaussianState(
        mean=np.array([8.0, 2.0, 0.0, 0.0]),
        covariance=np.diag([0.25, 0.25, 1.0, 1.0]),
    )
    second_state = GaussianState(
        mean=np.array([30.0, 30.0, 0.0, 0.0]),
        covariance=np.diag([0.25, 0.25, 1.0, 1.0]),
    )

    first_atom = model.learn_confirmed_state(first_state)
    second_atom = model.learn_confirmed_state(second_state)

    assert first_atom == 0
    assert second_atom == 1
    assert len(model.atoms) == 2
    assert [atom.count for atom in model.atoms] == pytest.approx([1.0, 1.0])


def test_confirmed_birth_reclustering_can_be_disabled_for_ablation() -> None:
    model = make_birth_model()
    model.recluster_confirmed_states = False
    first_state = GaussianState(
        mean=np.array([8.0, 2.0, 0.0, 0.0]),
        covariance=np.diag([0.25, 0.25, 1.0, 1.0]),
    )
    second_state = GaussianState(
        mean=np.array([8.2, 2.2, 0.0, 0.0]),
        covariance=np.diag([0.25, 0.25, 1.0, 1.0]),
    )

    model.learn_confirmed_state(first_state)
    model.learn_confirmed_state(second_state)

    assert len(model.atoms) == 2
