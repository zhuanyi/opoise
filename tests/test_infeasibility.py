from opoise import (AssetBounds, BetaBounds, Bounds, Budget, FactorExposureBounds, RiskCap, RiskTerm,
                              SolverSettings, Turnover, solve)
from helpers import DATA, W_MV, problem, vol

MINVAR = [RiskTerm(aversion=1.0)]


def beta_conflict():
    return [FactorExposureBounds(name="f_wide", bounds={f: Bounds(lower=-10.0, upper=10.0)
                                                         for f in DATA.risk_model.factor_ids}),
            Turnover(name="turnover_loose", limit=100.0),
            Budget(name="budget"),
            AssetBounds(name="long_only", lower=0.0, upper=1.0),
            BetaBounds(name="beta_min", lower=5.0)]


def test_infeasible_beta_names_the_conflict():
    r = solve(problem(MINVAR, beta_conflict()))
    assert r.status == "infeasible" and r.weights is None and r.trades is None and r.diagnostics is None
    rep = r.infeasibility
    assert rep.conflict_set == ["budget", "long_only", "beta_min"]
    assert rep.method == "deletion_filter" and rep.solves == 6
    assert "budget, long_only, beta_min" in rep.message and "beta_min" in r.message


def test_infeasible_risk_cap():
    cons = [AssetBounds(name="wide", lower=-1.0, upper=1.0), Budget(name="budget"),
            RiskCap(name="vol_cap", limit=0.5 * vol(W_MV))]
    r = solve(problem(MINVAR, cons))
    assert r.status == "infeasible" and r.infeasibility.conflict_set == ["budget", "vol_cap"]
    assert r.infeasibility.solves == 4


def test_diagnosis_can_be_disabled():
    r = solve(problem(MINVAR, beta_conflict(), settings=SolverSettings(diagnose_infeasibility=False)))
    assert r.status == "infeasible" and r.infeasibility is None


def test_inconclusive_subset_solve_is_reported(monkeypatch):
    import opoise.infeasibility as inf
    real, calls = inf.run, []

    def flaky(cvx, settings):  # the second subset test (drop f_wide) "fails" inside the solver
        calls.append(1)
        return ("solver_error", 0.0, "injected") if len(calls) == 2 else real(cvx, settings)

    monkeypatch.setattr(inf, "run", flaky)
    rep = solve(problem(MINVAR, beta_conflict())).infeasibility
    assert rep.conflict_set == ["f_wide", "budget", "long_only", "beta_min"] and rep.solves == 6
    assert "may not be minimal" in rep.message


def test_feasible_problem_has_no_report():
    r = solve(problem(MINVAR, [Budget(name="budget")]))
    assert r.status == "optimal" and r.infeasibility is None
