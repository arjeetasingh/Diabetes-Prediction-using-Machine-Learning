import numpy as np
import pytest

from diabetes_ml.conformal import MondrianConformal


def test_toy_threshold_value():
    y = np.array([0] * 9 + [1] * 9)
    p1 = np.concatenate([np.linspace(0.05, 0.45, 9), np.linspace(0.55, 0.95, 9)])
    c = MondrianConformal(alpha=0.2).fit(y, p1)
    # a clean separation: confident labels everywhere away from the middle
    assert c.decisions([0.05])[0] == "low"
    assert c.decisions([0.95])[0] == "high"


def test_alpha_validation():
    with pytest.raises(ValueError):
        MondrianConformal(alpha=0)
    with pytest.raises(ValueError):
        MondrianConformal(alpha=1)


def test_requires_fit():
    with pytest.raises(RuntimeError):
        MondrianConformal().predict_sets([0.5])


def test_coverage_guarantee_on_synthetic_data():
    """Per-class coverage must be about 1 - alpha on fresh data, even for an imbalanced problem."""
    rng = np.random.default_rng(0)

    def draw(n):
        y = (rng.random(n) < 0.35).astype(int)
        logit = -1.0 + 2.2 * y + rng.normal(0, 1.2, n)
        return y, 1 / (1 + np.exp(-logit))

    alpha = 0.10
    cov0, cov1 = [], []
    for _ in range(200):
        y_cal, p_cal = draw(600)
        y_te, p_te = draw(2000)
        c = MondrianConformal(alpha).fit(y_cal, p_cal)
        s = c.predict_sets(p_te)
        cov0.append(s[y_te == 0, 0].mean())
        cov1.append(s[y_te == 1, 1].mean())
    assert np.mean(cov0) >= 1 - alpha - 0.01
    assert np.mean(cov1) >= 1 - alpha - 0.01
    assert np.mean(cov0) <= 1 - alpha + 0.03  # not wastefully conservative


def test_decisions_are_consistent_with_sets():
    rng = np.random.default_rng(1)
    y = (rng.random(400) < 0.4).astype(int)
    p = np.clip(0.5 + (y - 0.5) * 0.5 + rng.normal(0, 0.2, 400), 0.01, 0.99)
    c = MondrianConformal(0.1).fit(y, p)
    sets = c.predict_sets(p)
    dec = c.decisions(p)
    assert ((dec == "high") == (sets[:, 1] & ~sets[:, 0])).all()
    assert ((dec == "low") == (sets[:, 0] & ~sets[:, 1])).all()
