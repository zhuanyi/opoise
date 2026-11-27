import numpy as np
import pytest

from opoise import (AssetBounds, BetaBounds, Bounds, Budget, FactorExposureBounds, GroupBounds,
                              GroupRiskCap, LinearBounds, RiskCap, RiskTerm, Turnover, solve)
from helpers import (B, BENCH, BETA, IDS, ONE, SECTOR, SIGMA, W_MV, as_dict, problem, row, te, vec, vol)

BUDGET = Budget(name="budget")
MINVAR = [RiskTerm(aversion=1.0)]
MINACT = [RiskTerm(active=True, aversion=1.0)]


def solved(p):
    r = solve(p)
    assert r.status == "optimal", r.message
    return r


def test_te_cap_binds_on_two_fund_segment():
    c = 0.5 * te(W_MV)
    r = solved(problem(MINVAR, [BUDGET, RiskCap(name="te", active=True, limit=c)]))
    np.testing.assert_allclose(vec(r.weights), 0.5 * W_MV + 0.5 * BENCH, atol=1e-5)
    t = row(r, "te")
    assert t.value == pytest.approx(c, abs=1e-6) and t.upper == c and t.lower is None and t.binding
    assert r.diagnostics.binding == ["budget", "te"] and row(r, "budget").binding


def test_te_cap_slack_when_loose():
    r = solved(problem(MINVAR, [BUDGET, RiskCap(name="te", active=True, limit=2 * te(W_MV))]))
    np.testing.assert_allclose(vec(r.weights), W_MV, atol=1e-5)
    assert not row(r, "te").binding and row(r, "te").slack == pytest.approx(te(W_MV), abs=1e-5)


def test_total_risk_cap_on_min_active_risk():
    cap = 0.5 * (vol(W_MV) + vol(BENCH))
    assert vol(W_MV) < cap < vol(BENCH)
    d = W_MV - BENCH
    roots = np.roots([d @ SIGMA @ d, 2 * BENCH @ SIGMA @ d, BENCH @ SIGMA @ BENCH - cap ** 2]).real
    theta = min(t for t in roots if 0 <= t <= 1)
    r = solved(problem(MINACT, [BUDGET, RiskCap(name="vol", limit=cap)]))
    np.testing.assert_allclose(vec(r.weights), BENCH + theta * d, atol=1e-5)
    assert row(r, "vol").binding and r.diagnostics.portfolio.total_risk == pytest.approx(cap, abs=1e-6)


def test_beta_lower_bound_matches_two_constraint_closed_form():
    assert BETA @ W_MV < 0.95
    A = np.column_stack([ONE, BETA])
    SiA = np.linalg.solve(SIGMA, A)
    w = SiA @ np.linalg.solve(A.T @ SiA, np.array([1.0, 1.0]))
    r = solved(problem(MINVAR, [BUDGET, BetaBounds(name="beta", lower=1.0, upper=1.2)]))
    np.testing.assert_allclose(vec(r.weights), w, atol=1e-5)
    assert r.diagnostics.beta == pytest.approx(1.0, abs=1e-6)
    assert row(r, "beta").binding and row(r, "beta").lower == 1.0 and row(r, "beta").upper == 1.2


def test_turnover_binds_then_releases():
    w0 = as_dict(W_MV)
    full = float(np.abs(BENCH - W_MV).sum())
    assert full > 0.5
    tight = solved(problem(MINACT, [BUDGET, Turnover(name="turnover", limit=0.5)], initial_holdings=w0))
    assert float(np.abs(vec(tight.weights) - W_MV).sum()) == pytest.approx(0.5, abs=1e-6)
    assert tight.diagnostics.turnover == pytest.approx(0.5, abs=1e-6) and row(tight, "turnover").binding
    assert 0.0 < te(vec(tight.weights)) < te(W_MV)
    loose = solved(problem(MINACT, [BUDGET, Turnover(name="turnover", limit=full + 0.1)], initial_holdings=w0))
    np.testing.assert_allclose(vec(loose.weights), BENCH, atol=1e-5)
    assert not row(loose, "turnover").binding


def test_long_only_min_variance_satisfies_kkt():
    assert (W_MV < 0).any()
    r = solved(problem(MINVAR, [BUDGET, AssetBounds(name="long_only", lower=0.0)]))
    w = vec(r.weights)
    g = SIGMA @ w
    on = w > 1e-5
    assert w.min() > -1e-7 and 0 < on.sum() < len(w)
    np.testing.assert_allclose(g[on], g[on].mean(), atol=1e-6)
    assert (g[~on] >= g[on].mean() - 1e-6).all()
    rows = [x for x in r.diagnostics.constraints if x.constraint == "long_only"]
    assert [x.row for x in rows] == [f"long_only[{i}]" for i in IDS]
    assert {x.row for x in rows if x.binding} == {f"long_only[{i}]" for i, v in zip(IDS, w) if v <= 1e-5}


def test_asset_bounds_per_asset_override_and_active():
    ab = AssetBounds(name="ab", lower=-0.05, upper=0.10, per_asset={IDS[0]: Bounds(lower=0.0, upper=0.02)})
    w = vec(solved(problem(MINVAR, [BUDGET, ab])).weights)
    assert w[1:].min() >= -0.05 - 1e-7 and w[1:].max() <= 0.10 + 1e-7 and -1e-7 <= w[0] <= 0.02 + 1e-7
    act = AssetBounds(name="act", active=True, lower=-0.02, upper=0.02)
    r = solved(problem(MINVAR, [BUDGET, act]))
    assert np.abs(vec(r.weights) - BENCH).max() <= 0.02 + 1e-7
    assert row(r, f"act[{IDS[0]}]").lower == -0.02


def test_sector_weight_bounds():
    base = [BUDGET, AssetBounds(name="long_only", lower=0.0)]
    free = solved(problem(MINVAR, base))
    assert vec(free.weights)[SECTOR == "S0"].sum() > 0.35
    # S1 >= 0 is redundant with long_only; it is there to pin the row label order
    g = GroupBounds(name="sectors", attribute="sector", bounds={"S0": Bounds(upper=0.30), "S1": Bounds(lower=0.0)})
    r = solved(problem(MINVAR, base + [g]))
    assert vec(r.weights)[SECTOR == "S0"].sum() == pytest.approx(0.30, abs=1e-6) and row(r, "sectors[S0]").binding
    assert [x.row for x in r.diagnostics.constraints if x.constraint == "sectors"] == ["sectors[S0]", "sectors[S1]"]


def test_active_sector_bound():
    m = SECTOR == "S2"
    assert (W_MV - BENCH)[m].sum() < 0.05
    g = GroupBounds(name="act", attribute="sector", active=True, bounds={"S2": Bounds(lower=0.05)})
    r = solved(problem(MINVAR, [BUDGET, g]))
    assert (vec(r.weights) - BENCH)[m].sum() == pytest.approx(0.05, abs=1e-6) and row(r, "act[S2]").binding


def test_group_standalone_risk_cap():
    x0 = np.where(SECTOR == "S0", W_MV, 0.0)
    assert np.sqrt(x0 @ SIGMA @ x0) > 0.06
    r = solved(problem(MINVAR, [BUDGET, GroupRiskCap(name="srisk", attribute="sector", caps={"S0": 0.05})]))
    x = np.where(SECTOR == "S0", vec(r.weights), 0.0)
    assert float(np.sqrt(x @ SIGMA @ x)) == pytest.approx(0.05, abs=1e-6) and row(r, "srisk[S0]").binding


def test_factor_exposure_bounds():
    fx = FactorExposureBounds(name="fx", bounds={"STYLE1": Bounds(lower=0.0, upper=0.0), "MARKET": Bounds(lower=1.0)})
    r = solved(problem(MINVAR, [BUDGET, fx]))
    y = B.T @ vec(r.weights)
    assert y[1] == pytest.approx(0.0, abs=1e-6) and y[0] >= 1.0 - 1e-6
    assert row(r, "fx[STYLE1]").binding and row(r, "fx[MARKET]").binding and row(r, "fx[MARKET]").upper is None
    assert [x.row for x in r.diagnostics.constraints][1:] == ["fx[STYLE1]", "fx[MARKET]"]
    afx = FactorExposureBounds(name="afx", active=True, bounds={"MARKET": Bounds(lower=0.0, upper=0.0)})
    ra = solved(problem(MINVAR, [BUDGET, afx]))
    assert float(B[:, 0] @ (vec(ra.weights) - BENCH)) == pytest.approx(0.0, abs=1e-6)


def test_linear_bounds():
    assert W_MV[0] + W_MV[1] < 0.3
    lb = LinearBounds(name="pair", coefficients={IDS[0]: 1.0, IDS[1]: 1.0}, lower=0.3)
    r = solved(problem(MINVAR, [BUDGET, lb]))
    w = vec(r.weights)
    assert w[0] + w[1] == pytest.approx(0.3, abs=1e-6) and row(r, "pair").binding
