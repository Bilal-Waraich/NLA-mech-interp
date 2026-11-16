"""Inference primitives against scipy or hand-computed references on small inputs."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import PermutationMethod, bootstrap, false_discovery_control, mannwhitneyu

from medinterp.eval.stats import (benjamini_hochberg, cluster_bootstrap, cluster_bootstrap_difference,
                                  complete_separation, grouped_mannwhitney, holm)


def test_bootstrap_of_constant_has_zero_width() -> None:
    r = cluster_bootstrap(lambda i: np.full(10, 3.0)[i].mean(), np.arange(10), 200, 0.95, seed=0)
    assert r.estimate == r.ci_low == r.ci_high == 3.0


def test_singleton_groups_agree_with_scipy_percentile_bootstrap() -> None:
    x = np.random.default_rng(0).exponential(size=300)
    ours = cluster_bootstrap(lambda i: x[i].mean(), np.arange(300), 4000, 0.95, seed=1)
    ref = bootstrap((x,), np.mean, n_resamples=4000, method="percentile", random_state=2).confidence_interval
    assert ours.ci_low == pytest.approx(ref.low, abs=0.02)
    assert ours.ci_high == pytest.approx(ref.high, abs=0.02)


def test_whole_groups_are_resampled() -> None:
    """Copying every value five times inside its own group must give exactly the same interval as the
    values alone: the resamples draw the same groups, and a group's mean is its value."""
    x = np.random.default_rng(3).normal(size=50)
    single = cluster_bootstrap(lambda i: x[i].mean(), np.arange(50), 500, 0.95, seed=7)
    xr, gr = np.repeat(x, 5), np.repeat(np.arange(50), 5)
    grouped = cluster_bootstrap(lambda i: xr[i].mean(), gr, 500, 0.95, seed=7)
    assert grouped.ci_low == pytest.approx(single.ci_low, abs=1e-12)
    assert grouped.ci_high == pytest.approx(single.ci_high, abs=1e-12)
    # Treating the copies as independent images would make the interval far too narrow.
    naive = cluster_bootstrap(lambda i: xr[i].mean(), np.arange(250), 500, 0.95, seed=7)
    assert (naive.ci_high - naive.ci_low) < 0.6 * (grouped.ci_high - grouped.ci_low)


def test_paired_difference_cancels_shared_noise() -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(size=400) * 10
    a, b = base + 0.5, base                       # identical apart from a constant shift
    r = cluster_bootstrap_difference(lambda i: a[i].mean(), lambda i: b[i].mean(), np.arange(400),
                                     300, 0.95, seed=0)
    assert r.ci_low == pytest.approx(0.5) and r.ci_high == pytest.approx(0.5)


def test_vector_statistic() -> None:
    x = np.random.default_rng(0).normal(size=(100, 3))
    r = cluster_bootstrap(lambda i: x[i].mean(0), np.arange(100), 200, 0.9, seed=0)
    assert r.estimate.shape == r.ci_low.shape == (3,) and (r.ci_low <= r.ci_high).all()


def test_benjamini_hochberg_matches_scipy_and_hand_values() -> None:
    p = np.array([0.01, 0.04, 0.03, 0.005, 0.2, 0.5])
    adj, rej = benjamini_hochberg(p, 0.05)
    np.testing.assert_allclose(adj, false_discovery_control(p, method="bh"))
    # Hand: sorted 0.005,0.01,0.03,0.04,0.2,0.5 -> *6/rank = .03,.03,.06,.06,.24,.5 -> step-up min.
    np.testing.assert_allclose(adj, [0.03, 0.06, 0.06, 0.03, 0.24, 0.5])
    assert rej.tolist() == [True, False, False, True, False, False]


def test_benjamini_hochberg_skips_nan() -> None:
    adj, rej = benjamini_hochberg(np.array([0.01, np.nan, 0.02]), 0.05)
    np.testing.assert_allclose(adj[[0, 2]], [0.02, 0.02])
    assert np.isnan(adj[1]) and not rej[1]


def test_holm_hand_values() -> None:
    p = np.array([0.01, 0.04, 0.03, 0.005])
    adj, rej = holm(p, 0.05)
    # Sorted .005,.01,.03,.04 times 4,3,2,1 = .02,.03,.06,.04 -> running max .02,.03,.06,.06.
    np.testing.assert_allclose(adj, [0.03, 0.06, 0.06, 0.02])
    assert rej.tolist() == [True, False, False, True]
    statsmodels = pytest.importorskip("statsmodels.stats.multitest")
    np.testing.assert_allclose(adj, statsmodels.multipletests(p, method="holm")[1])


def test_complete_separation_five_vs_five() -> None:
    r = complete_separation([0.92, 0.93, 0.95, 0.91, 0.94], [0.84, 0.85, 0.83, 0.88, 0.86], margin=0.02)
    assert r.separated and r.gap == pytest.approx(0.03)
    assert r.p_value == pytest.approx(1 / 252)
    ref = mannwhitneyu([0.92, 0.93, 0.95, 0.91, 0.94], [0.84, 0.85, 0.83, 0.88, 0.86],
                       alternative="greater", method="exact").pvalue
    assert r.p_value == pytest.approx(ref)


def test_complete_separation_margin_and_direction() -> None:
    assert not complete_separation([0.90, 0.91], [0.89], margin=0.02).separated
    assert not complete_separation([1, 5], [2, 3]).separated
    r = complete_separation([0.1, 0.2], [0.5, 0.6], higher_is_better=False)
    assert r.separated and r.gap == pytest.approx(0.3)


def test_complete_separation_overlap_matches_scipy() -> None:
    a, b = [3.0, 5.0, 6.0, 8.0], [1.0, 4.0, 5.0, 2.0, 7.0]       # includes a tie
    r = complete_separation(a, b)
    ref = mannwhitneyu(a, b, alternative="greater", method=PermutationMethod(n_resamples=np.inf)).pvalue
    assert r.p_value == pytest.approx(ref)


def test_grouped_mannwhitney_reduces_groups_first() -> None:
    rng = np.random.default_rng(0)
    vals = rng.normal(size=(60, 2))
    pos = np.repeat([True, False], 30)
    groups = np.repeat(np.arange(20), 3)                        # three images per patient
    p, auc = grouped_mannwhitney(vals, pos, groups)
    red = vals.reshape(20, 3, 2).mean(1)
    ref = mannwhitneyu(red[:10], red[10:], alternative="two-sided", method="asymptotic", axis=0)
    np.testing.assert_allclose(p, ref.pvalue)
    np.testing.assert_allclose(auc, ref.statistic / 100)


def test_grouped_mannwhitney_constant_and_ties() -> None:
    vals = np.zeros((40, 2))
    pos = np.arange(40) < 20
    vals[:, 1] = np.where(pos, 1.0, 0.0) * (np.arange(40) % 2)  # heavy ties
    p, auc = grouped_mannwhitney(vals, pos, np.arange(40))
    assert p[0] == 1.0 and auc[0] == 0.5
    ref = mannwhitneyu(vals[pos, 1], vals[~pos, 1], method="asymptotic")
    assert p[1] == pytest.approx(ref.pvalue) and auc[1] == pytest.approx(0.75)
