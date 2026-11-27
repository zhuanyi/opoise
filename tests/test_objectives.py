import os
import subprocess
import sys

import cvxpy as cp
import numpy as np
import pytest

from opoise import (AssetBounds, Budget, Problem, RiskCap, RiskTerm, SolverCapabilityError,
                              SolverNotAvailableError, SolverSettings, Universe,
                              compile_problem, solve)
from helpers import B, BENCH, D, DATA, IDS, SIGMA, TIGHT, W_MV, problem, row, te, vec

BUDGET = Budget(name="budget")


def solved(p):
    r = solve(p)
    assert r.status == "optimal", r.message
    return r


def test_min_variance_closed_form():
    r = solved(problem([RiskTerm(aversion=1.0)], [BUDGET]))
    np.testing.assert_allclose(vec(r.weights), W_MV, atol=1e-5)
    assert r.objective_value == pytest.approx(float(W_MV @ SIGMA @ W_MV), abs=1e-7)
    assert r.solver == "CLARABEL" and r.solve_seconds >= 0.0 and list(r.weights) == IDS
    assert r.diagnostics.portfolio.total_risk == pytest.approx(float(np.sqrt(W_MV @ SIGMA @ W_MV)), abs=1e-6)


def test_aversion_scales_objective_not_solution():
    r = solved(problem([RiskTerm(aversion=7.5)], [BUDGET]))
    np.testing.assert_allclose(vec(r.weights), W_MV, atol=1e-5)
    assert r.objective_value == pytest.approx(7.5 * float(W_MV @ SIGMA @ W_MV), rel=1e-5)


def test_factor_plus_specific_terms_equal_total():
    r = solved(problem([RiskTerm(measure="factor", aversion=1.0), RiskTerm(measure="specific", aversion=1.0)],
                       [BUDGET]))
    np.testing.assert_allclose(vec(r.weights), W_MV, atol=1e-5)


def test_specific_only_is_inverse_variance():
    r = solved(problem([RiskTerm(measure="specific", aversion=1.0)], [BUDGET]))
    np.testing.assert_allclose(vec(r.weights), (1 / D) / (1 / D).sum(), atol=1e-5)


def test_factor_only_zeroes_factor_exposure():
    # the box keeps the optimal face bounded (factor variance alone is singular)
    box = AssetBounds(name="box", lower=-1.0, upper=1.0)
    r = solved(problem([RiskTerm(measure="factor", aversion=1.0)], [BUDGET, box]))
    assert r.objective_value == pytest.approx(0.0, abs=1e-8)
    np.testing.assert_allclose(B.T @ vec(r.weights), 0.0, atol=1e-5)
    assert sum(r.weights.values()) == pytest.approx(1.0, abs=1e-7)


def test_min_active_risk_returns_benchmark():
    r = solved(problem([RiskTerm(active=True, aversion=1.0)], [BUDGET]))
    np.testing.assert_allclose(vec(r.weights), BENCH, atol=1e-5)
    assert te(vec(r.weights)) < 1e-5 and r.diagnostics.beta == pytest.approx(1.0, abs=1e-5)
    assert r.diagnostics.active.total_risk < 1e-5


def test_risk_model_superset_is_sliced():
    sub = Universe(assets=DATA.universe.assets[:5])
    r = solved(Problem(universe=sub, risk_model=DATA.risk_model, objective=[RiskTerm(aversion=1.0)],
                       constraints=[BUDGET], settings=SolverSettings(solver_options=TIGHT)))
    w = np.linalg.solve(SIGMA[:5, :5], np.ones(5))
    np.testing.assert_allclose([r.weights[i] for i in IDS[:5]], w / w.sum(), atol=1e-5)
    assert list(r.weights) == IDS[:5] and r.diagnostics.active is None and r.diagnostics.beta is None


def test_trades_are_weights_minus_initial():
    w0 = {IDS[0]: 0.6, IDS[1]: 0.4}
    r = solved(problem([RiskTerm(aversion=1.0)], [BUDGET], initial_holdings=w0))
    np.testing.assert_allclose(vec(r.trades), vec(r.weights) - vec(w0), atol=1e-12)
    assert r.diagnostics.turnover == pytest.approx(float(np.abs(vec(r.trades)).sum()), abs=1e-12)


def test_compile_problem_is_inspectable():
    cp_ = compile_problem(problem([RiskTerm(aversion=1.0)], [BUDGET, RiskCap(name="te", active=True, limit=0.05)]))
    assert isinstance(cp_.cvx, cp.Problem) and cp_.cvx.is_dcp() and not cp_.cvx.is_qp()
    assert [r.constraint for r in cp_.rows] == ["budget", "te"] and cp_.rows[1].labels == ["te"]
    assert set(cp_.constraint_map) == {"budget", "te"}
    feas = compile_problem(problem([RiskTerm(aversion=1.0)], [BUDGET]), constraint_names=[], feasibility=True)
    assert feas.rows == [] and feas.cvx.objective.value == 0


def test_osqp_path_and_capability_error():
    s = SolverSettings(solver="OSQP", solver_options={"eps_abs": 1e-9, "eps_rel": 1e-9, "max_iter": 200000})
    r = solve(problem([RiskTerm(aversion=1.0)], [BUDGET], settings=s))
    assert r.status in ("optimal", "optimal_inaccurate") and r.solver == "OSQP"
    np.testing.assert_allclose(vec(r.weights), W_MV, atol=1e-4)
    with pytest.raises(SolverCapabilityError, match="OSQP"):
        solve(problem([RiskTerm(aversion=1.0)], [BUDGET, RiskCap(name="te", active=True, limit=0.05)],
                      settings=SolverSettings(solver="OSQP")))


def test_missing_solver():
    if "MOSEK" in cp.installed_solvers():
        pytest.skip("MOSEK is installed here")
    with pytest.raises(SolverNotAvailableError, match="MOSEK"):
        solve(problem([RiskTerm(aversion=1.0)], [BUDGET], settings=SolverSettings(solver="MOSEK")))


@pytest.mark.parametrize("solver", ["CLARABEL", "OSQP"])  # Clarabel raises TypeError, cvxpy wraps OSQP's as SolverError
def test_rejected_solver_option_is_an_error_status(solver):
    s = SolverSettings(solver=solver, solver_options={"no_such_option": 1})
    r = solve(problem([RiskTerm(aversion=1.0)], [BUDGET], settings=s))
    assert r.status == "error" and "no_such_option" in r.message and r.weights is None and r.objective_value is None


def test_status_mapping_is_total():
    from opoise.optimize import map_status
    cases = {"optimal": "optimal", "optimal_inaccurate": "optimal_inaccurate", "infeasible": "infeasible",
             "infeasible_inaccurate": "infeasible", "infeasible_or_unbounded": "infeasible",
             "unbounded": "unbounded", "unbounded_inaccurate": "unbounded", "user_limit": "error",
             "solver_error": "error", "something new": "error"}
    assert {k: map_status(k) for k in cases} == cases


def test_binding_tolerance_and_violation_flags():
    cons = [BUDGET, RiskCap(name="te", active=True, limit=2 * te(W_MV))]
    r = solved(problem([RiskTerm(aversion=1.0)], cons))
    assert not row(r, "te").binding and row(r, "budget").binding
    assert not any(x.violated for x in r.diagnostics.constraints)
    wide = solved(problem([RiskTerm(aversion=1.0)], cons, settings=SolverSettings(binding_tol=1.0, solver_options=TIGHT)))
    assert row(wide, "te").binding and wide.diagnostics.binding == ["budget", "te"]


def test_package_import_does_not_load_cvxpy():
    code = "import sys, opoise as p; a = 'cvxpy' in sys.modules; s = p.solve; print(a, 'cvxpy' in sys.modules)"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env)
    assert out.stdout.split() == ["False", "True"]
