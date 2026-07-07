import numpy as np
import pytest

from dp_rfs_hybrid import DirichletProcessClutterModel, FixedGaussianMixtureClutterModel


def make_clutter_model() -> DirichletProcessClutterModel:
    return DirichletProcessClutterModel(
        alpha=1.0,
        base_mean=np.zeros(2),
        base_covariance=np.diag([25.0, 25.0]),
        rate=5.0,
        prune_below_count=0.01,
    )


def make_fixed_mixture_model() -> FixedGaussianMixtureClutterModel:
    return FixedGaussianMixtureClutterModel(
        weights=np.array([2.0, 1.0]),
        means=np.array([[0.0, 0.0], [10.0, 5.0]]),
        covariances=np.stack([np.eye(2), np.eye(2) * 4.0]),
        rate=3.0,
    )


def test_update_creates_clutter_atom_from_fractional_observation() -> None:
    model = make_clutter_model()

    update = model.update(np.array([10.0, -2.0]), responsibility=0.5)

    assert update.branch == "new"
    assert update.responsibility == 0.5
    assert len(model.atoms) == 1
    assert np.allclose(model.atoms[0].mean, np.array([10.0, -2.0]))
    assert model.atoms[0].count == pytest.approx(0.5)


def test_nearby_clutter_reuses_existing_atom_and_increases_density() -> None:
    model = make_clutter_model()
    model.update(np.array([10.0, -2.0]), responsibility=1.0)
    density_before = model.density(np.array([10.1, -1.9]))

    update = model.update(np.array([10.1, -1.9]), responsibility=0.75)
    density_after = model.density(np.array([10.1, -1.9]))

    assert update.branch == "existing"
    assert len(model.atoms) == 1
    assert model.atoms[0].count == pytest.approx(1.75)
    assert density_after > density_before


def test_zero_responsibility_leaves_atoms_unchanged() -> None:
    model = make_clutter_model()

    update = model.update(np.array([4.0, 5.0]), responsibility=0.0)

    assert update.branch == "ignored"
    assert len(model.atoms) == 0


def test_intensity_is_rate_times_density() -> None:
    model = make_clutter_model()
    z = np.array([1.0, 2.0])

    assert model.intensity(z) == pytest.approx(model.rate * model.density(z))


def test_responsibility_must_be_probability() -> None:
    model = make_clutter_model()

    with pytest.raises(ValueError, match="responsibility"):
        model.update(np.array([0.0, 0.0]), responsibility=1.5)


def test_fixed_gaussian_mixture_clutter_intensity_is_rate_times_density() -> None:
    model = make_fixed_mixture_model()
    z = np.array([0.0, 0.0])

    assert model.intensity(z) == pytest.approx(model.rate * model.density(z))
    assert model.density(z) > model.density(np.array([30.0, 30.0]))


def test_fixed_gaussian_mixture_update_is_noop_with_diagnostics() -> None:
    model = make_fixed_mixture_model()
    density_before = model.density(np.array([10.0, 5.0]))

    update = model.update(np.array([10.0, 5.0]), responsibility=0.25)

    assert update.branch == "fixed"
    assert update.responsibility == 0.25
    assert model.density(np.array([10.0, 5.0])) == pytest.approx(density_before)


def test_fixed_gaussian_mixture_validates_weights() -> None:
    with pytest.raises(ValueError, match="weight"):
        FixedGaussianMixtureClutterModel(
            weights=np.array([0.0, 0.0]),
            means=np.array([[0.0, 0.0], [1.0, 1.0]]),
            covariances=np.stack([np.eye(2), np.eye(2)]),
            rate=1.0,
        )
