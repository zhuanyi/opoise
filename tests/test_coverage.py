"""Compile/solve coverage of constraint variants the other files do not exercise (review finding 10)."""
import numpy as np
import pytest

from opoise import (AssetBounds, BetaBounds, Bounds, Budget, FactorExposureBounds, GroupBounds,
                              GroupRiskCap, LinearBounds, RiskCap, RiskTerm, SolverSettings, Turnover,
                              compile_problem, risk_report, solve)
from helpers import BENCH, BETA, D, IDS, ONE, SIGMA, W_MV, as_dict, problem, row, vec

MINVAR = [RiskTerm(aversion=1.0)]
MINACT = [RiskTerm(active=True, aversion=1.0)]
BUDGET = Budget(name="budget")


def solved(p):
    r = solve(p)
    assert r.status == "optimal", r.message
    return r


def test_every_constraint_variant_compiles_and_solves():
    cons = [BUDGET,
            AssetBounds(name="box", lower=-0.2, upper=0.3),
            AssetBounds(name="act_one", active=True, per_asset={IDS[1]: Bounds(lower=-0.01)}),
            Turnover(name="turnover", limit=3.0),
            GroupBounds(name="act_sector", attribute="sector", active=True, bounds={"S0": Bounds(upper=0.1)}),
            GroupRiskCap(name="g_factor", attribute="sector", measure="factor", active=True, caps={"S1": 0.2}),
            GroupRiskCap(name="g_spec", attribute="sector", measure="specific", caps={"S2": 0.2}),
            BetaBounds(name="beta_cap", upper=1.1),
            RiskCap(name="f_cap", measure="factor", limit=0.5),
            RiskCap(name="s_te", measure="specific", active=True, limit=0.5),
            FactorExposureBounds(name="act_fx", active=True, bounds={"STYLE1": Bounds(upper=0.5)}),
            LinearBounds(name="pin", coefficients={IDS[2]: 1.0}, lower=0.05, upper=0.05)]
    p = problem(MINVAR, cons, initial_holdings=as_dict(BENCH))
    c = compile_problem(p)
    assert c.cvx.is_dcp() and not c.cvx.is_qp()
    assert [x.constraint for x in c.rows] == [k.name for k in cons] and set(c.constraint_map) == {k.name for k in cons}
    assert len(c.rows[1].labels) == len(IDS) and c.rows[2].labels == [f"act_one[{IDS[1]}]"]
    assert c.rows[5].labels == ["g_factor[S1]"] and c.rows[5].convex and not c.rows[4].convex
    r = solved(p)
    assert not any(x.violated for x in r.diagnostics.constraints)
    assert r.weights[IDS[2]] == pytest.approx(0.05, abs=1e-6) and row(r, "pin").binding
    assert row(r, "pin").lower == row(r, "pin").upper == 0.05


def test_dollar_neutral_budget_closed_form():
    # min (w-b)'S(w-b) s.t. sum(w) = 0  ->  w = b - w_mv (since sum(b) = 1)
    r = solved(problem(MINACT, [Budget(name="neutral", total=0.0)]))
    np.testing.assert_allclose(vec(r.weights), BENCH - W_MV, atol=1e-5)
    assert sum(r.weights.values()) == pytest.approx(0.0, abs=1e-7) and row(r, "neutral").binding


@pytest.mark.parametrize("measure", ["factor", "specific"])
def test_factor_and_specific_risk_caps_bind(measure):
    free = risk_report(problem(MINVAR), as_dict(W_MV))
    # floor = least attainable risk of this kind under the budget (0, or the inverse-variance portfolio's)
    floor = 0.0 if measure == "factor" else float(np.sqrt(1.0 / (1.0 / D).sum()))
    cap = 0.5 * (floor + (free.factor_risk if measure == "factor" else free.specific_risk))
    r = solved(problem(MINVAR, [BUDGET, RiskCap(name="cap", measure=measure, limit=cap)]))
    got = r.diagnostics.portfolio.factor_risk if measure == "factor" else r.diagnostics.portfolio.specific_risk
    assert got == pytest.approx(cap, abs=1e-6) and row(r, "cap").binding


def test_beta_upper_bound_and_zero_turnover():
    assert BETA @ W_MV > 0.5
    A = np.column_stack([ONE, BETA])
    SiA = np.linalg.solve(SIGMA, A)
    w = SiA @ np.linalg.solve(A.T @ SiA, np.array([1.0, 0.5]))
    r = solved(problem(MINVAR, [BUDGET, BetaBounds(name="beta", upper=0.5)]))
    np.testing.assert_allclose(vec(r.weights), w, atol=1e-5)
    assert row(r, "beta").binding and row(r, "beta").lower is None
    frozen = solved(problem(MINVAR, [BUDGET, Turnover(name="t", limit=0.0)], initial_holdings=as_dict(BENCH)))
    np.testing.assert_allclose(vec(frozen.weights), BENCH, atol=1e-5)
    assert frozen.diagnostics.turnover == pytest.approx(0.0, abs=1e-5)


def test_scs_with_options_and_verbose(capfd):
    s = SolverSettings(solver="SCS", verbose=True, solver_options={"eps_abs": 1e-9, "eps_rel": 1e-9, "max_iters": 100000})
    r = solve(problem(MINVAR, [BUDGET], settings=s))
    assert r.status in ("optimal", "optimal_inaccurate") and r.solver == "SCS"
    np.testing.assert_allclose(vec(r.weights), W_MV, atol=1e-4)
    assert "SCS" in capfd.readouterr().out
