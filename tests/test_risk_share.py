"""Variant B: sector share-of-risk cap (Euler contribution), solved by sequential convex approximation.

The constraint is non-convex, so these tests pin what is guaranteed: every returned point satisfies the caps (each
convex subproblem is a conservative inner approximation), the convergence flag is truthful, and shares are the true
Euler shares computed from the final weights (never the approximation).
"""
import numpy as np
import pytest
from pydantic import ValidationError

from opoise import (Asset, AssetBounds, Budget, FactorRiskModel, GroupRiskShareCap, OpoiseError, Problem, RiskTerm,
                    SolverCapabilityError, SolverSettings, Universe, compile_problem, load_workspace, save_workspace,
                    solve)
from helpers import BENCH, IDS, SECTOR, SIGMA, TIGHT, W_MV, problem, row, vec

BUDGET = Budget(name="budget")
MINVAR = [RiskTerm(aversion=1.0)]
FREE_S0 = 0.460  # share of S0 in the unconstrained minimum-variance portfolio (asserted in test_cap_is_active)


def share(w, label, active=False):
    x = w - BENCH if active else w
    m = SECTOR == label
    return float(np.where(m, x, 0.0) @ SIGMA @ x / (x @ SIGMA @ x))


def solved(p):
    r = solve(p)
    assert r.status == "optimal", r.message
    assert r.sca is not None and r.sca.converged, r.message
    return r


def toy(cap_b):
    """Two assets, no factor risk, specific variances 0.04 and 0.01: shares are d_i w_i^2 / sum(d_j w_j^2)."""
    rm = FactorRiskModel.from_arrays(["X0", "X1"], ["F"], np.zeros((2, 1)), np.eye(1), [0.04, 0.01])
    u = Universe(assets=[Asset(id="X0", attributes={"grp": "A"}), Asset(id="X1", attributes={"grp": "B"})])
    cons = [BUDGET, GroupRiskShareCap(name="share", attribute="grp", caps={"B": cap_b})]
    return Problem(universe=u, risk_model=rm, objective=MINVAR, constraints=cons,
                   settings=SolverSettings(solver_options=TIGHT))


def test_analytic_binding_case():
    # unconstrained optimum is w = (0.2, 0.8) with B-share 0.8; share_B <= 0.5 <=> sqrt(.01)|w1| <= sqrt(.04)|w0|,
    # so the cap binds at w1 = 2 w0, i.e. w = (1/3, 2/3), variance 0.04/9 + 0.01*4/9
    r = solved(toy(0.5))
    np.testing.assert_allclose([r.weights["X0"], r.weights["X1"]], [1 / 3, 2 / 3], atol=1e-5)
    assert r.sca.shares == {"share": pytest.approx({"A": 0.5, "B": 0.5}, abs=1e-6)}
    assert r.objective_value == pytest.approx(0.04 / 9 + 0.04 / 9, abs=1e-8)
    assert row(r, "share[B]").binding and row(r, "share[B]").upper == 0.5 and row(r, "share[B]").lower is None
    assert 2 <= r.sca.iterations <= 50 and r.sca.max_weight_change <= 1e-6


def test_analytic_loose_cap_needs_one_iteration():
    r = solved(toy(0.9))
    np.testing.assert_allclose([r.weights["X0"], r.weights["X1"]], [0.2, 0.8], atol=1e-5)
    assert r.sca.iterations == 1 and not row(r, "share[B]").binding
    assert r.sca.shares["share"]["B"] == pytest.approx(0.8, abs=1e-6)


def test_cap_is_active_in_the_data():
    free = share(W_MV, "S0")
    assert free == pytest.approx(FREE_S0, abs=1e-3) and free > 0.30
    assert share(W_MV, "S1") > 0.25 and share(W_MV, "S0", active=True) > 0.30


@pytest.mark.parametrize("caps", [{"S0": 0.30}, {"S0": 0.20}, {"S0": 0.30, "S1": 0.25}])
def test_cap_respected_shares_sum_to_one_and_match_independent_computation(caps):
    r = solved(problem(MINVAR, [BUDGET, GroupRiskShareCap(name="share", attribute="sector", caps=caps)]))
    w = vec(r.weights)
    s = r.sca.shares["share"]
    assert list(s) == ["S0", "S1", "S2", "S3"]
    assert sum(s.values()) == pytest.approx(1.0, abs=1e-9)
    for lab in s:
        assert s[lab] == pytest.approx(share(w, lab), abs=1e-9)
    for lab, c in caps.items():
        assert s[lab] <= c + 1e-6 and row(r, f"share[{lab}]").binding
        assert row(r, f"share[{lab}]").value == pytest.approx(s[lab], abs=1e-12)
    assert not any(x.violated for x in r.diagnostics.constraints)
    free = solve(problem(MINVAR, [BUDGET]))
    assert r.objective_value > free.objective_value + 1e-6  # a binding cap costs variance


def test_active_share_cap_uses_active_weights():
    g = GroupRiskShareCap(name="share", attribute="sector", active=True, caps={"S0": 0.15})
    r = solved(problem(MINVAR, [BUDGET, g]))
    w = vec(r.weights)
    assert share(w, "S0", active=True) <= 0.15 + 1e-6 and row(r, "share[S0]").binding
    assert sum(r.sca.shares["share"].values()) == pytest.approx(1.0, abs=1e-9)
    assert r.sca.shares["share"]["S0"] == pytest.approx(share(w, "S0", active=True), abs=1e-9)


def test_max_iterations_is_reported_not_hidden():
    cap = {"S0": 0.20}
    g = GroupRiskShareCap(name="share", attribute="sector", caps=cap)
    full = solved(problem(MINVAR, [BUDGET, g]))
    assert full.sca.iterations > 3
    objs = []
    for n in (1, 2, 3):
        s = SolverSettings(solver_options=TIGHT, sca_max_iterations=n)
        r = solve(problem(MINVAR, [BUDGET, g], settings=s))
        assert r.status == "optimal_inaccurate" and r.sca.converged is False and r.sca.iterations == n
        assert f"did not converge in {n} iterations" in r.message
        assert r.sca.max_weight_change > s.sca_tol
        assert r.sca.shares["share"]["S0"] <= 0.20 + 1e-6  # every iterate satisfies the cap
        assert row(r, "share[S0]").value == pytest.approx(r.sca.shares["share"]["S0"], abs=1e-12)
        objs.append(r.objective_value)
    assert objs[0] >= objs[1] - 1e-9 >= objs[2] - 2e-9 and objs[2] >= full.objective_value - 1e-9


def test_other_constraints_still_hold_and_rows_keep_problem_order():
    cons = [BUDGET, AssetBounds(name="long_only", lower=0.0),
            GroupRiskShareCap(name="share", attribute="sector", caps={"S0": 0.25})]
    r = solved(problem(MINVAR, cons))
    w = vec(r.weights)
    assert w.min() > -1e-7 and share(w, "S0") <= 0.25 + 1e-6
    names = [x.constraint for x in r.diagnostics.constraints]
    assert names[0] == "budget" and names[-1] == "share" and names.count("long_only") == len(IDS)
    assert not any(x.violated for x in r.diagnostics.constraints)


def test_infeasible_without_the_share_caps_is_reported_as_such():
    cons = [BUDGET, AssetBounds(name="box", upper=0.01),
            GroupRiskShareCap(name="share", attribute="sector", caps={"S0": 0.25})]
    r = solve(problem(MINVAR, cons))
    assert r.status == "infeasible" and r.sca is None and r.weights is None
    assert r.message.startswith("without the share caps: infeasible")
    assert r.infeasibility.conflict_set == ["budget", "box"]


def test_validation_and_contract():
    g = GroupRiskShareCap(name="share", attribute="sector", caps={"S9": 0.3})
    with pytest.raises(ValidationError, match="constraint 'share': unknown group 'S9'"):
        problem(MINVAR, [g])
    ga = GroupRiskShareCap(name="share", attribute="sector", active=True, caps={"S0": 0.3})
    with pytest.raises(ValidationError, match="constraint 'share' requires a benchmark"):
        problem(MINVAR, [ga], benchmark=None)
    with pytest.raises(ValidationError):
        GroupRiskShareCap(name="share", attribute="sector", caps={"S0": 0.0})
    with pytest.raises(ValidationError):
        GroupRiskShareCap(name="share", attribute="sector", caps={})
    for bad in ({"sca_max_iterations": 0}, {"sca_tol": 0.0}):
        with pytest.raises(ValidationError):
            SolverSettings(**bad)
    assert SolverSettings().sca_max_iterations == 50 and SolverSettings().sca_tol == 1e-6
    p = problem(MINVAR, [BUDGET, GroupRiskShareCap(name="share", attribute="sector", caps={"S0": 0.3})])
    with pytest.raises(OpoiseError, match="linearisation point"):
        compile_problem(p)
    c = compile_problem(p, w_ref=W_MV)
    assert c.cvx.is_dcp() and c.rows[1].labels == ["share[S0]"] and c.rows[1].convex
    assert list(c.rows[1].upper) == [0.0] and not np.isfinite(c.rows[1].lower).any()


def test_osqp_is_refused_because_the_subproblems_have_quadratic_constraints():
    p = problem(MINVAR, [BUDGET, GroupRiskShareCap(name="share", attribute="sector", caps={"S0": 0.3})],
                settings=SolverSettings(solver="OSQP"))
    with pytest.raises(SolverCapabilityError, match="OSQP solves QPs only"):
        solve(p)


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
def test_workspace_round_trip(tmp_path, suffix):
    p = problem(MINVAR, [BUDGET, GroupRiskShareCap(name="share", attribute="sector", active=True, caps={"S0": 0.3})],
                settings=SolverSettings(solver_options=TIGHT, sca_max_iterations=7, sca_tol=1e-7))
    save_workspace(p, tmp_path / f"w{suffix}")
    assert load_workspace(tmp_path / f"w{suffix}") == p
