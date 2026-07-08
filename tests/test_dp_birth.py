import numpy as np

from dp_rfs_hybrid import DirichletProcessBirthModel, GaussianState


def make_birth_model(clutter_intensity: float = 1e-4) -> DirichletProcessBirthModel:
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
        odds_threshold=10.0,
    )


def test_new_measurement_creates_birth_atom() -> None:
    model = make_birth_model()

    decision = model.process(np.array([12.0, -3.0]))

    assert decision.accepted
    assert decision.branch == "new"
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


def test_preview_birth_does_not_update_atoms() -> None:
    model = make_birth_model()

    decision = model.preview(np.array([12.0, -3.0]))

    assert decision.accepted
    assert decision.state is not None
    assert len(model.atoms) == 0


def test_confirmed_state_learning_creates_birth_atom() -> None:
    model = make_birth_model()
    decision = model.preview(np.array([12.0, -3.0]))

    update = model.learn_from_state(decision.state)

    assert update.branch == "new"
    assert update.atom_index == 0
    assert len(model.atoms) == 1
    assert model.atoms[0].count == 1.0
