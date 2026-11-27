"""Result models returned by a solve, and the risk and constraint reports.

The models here are frozen pydantic models with extra fields forbidden. They
carry plain data only: weights and trades as dicts keyed by asset id, risk
numbers as standard deviations in the units of the risk model, and one report
row per compiled constraint row. Result models allow non-finite numbers only so
a report can represent a slack of infinity; in practice every value comes from
a finite computation.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

_CONFIG = ConfigDict(extra="forbid", frozen=True)


class RiskReport(BaseModel):
    """Risk of one portfolio, decomposed into Euler contributions.

    All risk numbers are standard deviations in the units of the risk model,
    so an annual covariance gives an annual volatility. Contributions sum to
    the total risk whenever it is positive; for a portfolio with zero risk
    every contribution is zero.

    Attributes
    ----------
    total_risk : float
        Standard deviation of the portfolio (or active) return, the square
        root of the variance.
    factor_risk : float
        Standard deviation explained by the factor model.
    specific_risk : float
        Standard deviation of the specific (idiosyncratic) part.
    factor_exposures : dict of str to float
        Factor exposures of the portfolio, one per factor, in factor order.
    by_factor : dict of str to float
        Euler contribution of each factor to the total risk, in factor order;
        together with specific_contribution it sums to the total risk.
    specific_contribution : float
        Euler contribution of the specific part to the total risk.
    by_asset : dict of str to float
        Euler contribution of each asset to the total risk, in universe order;
        sums to the total risk.
    by_group : dict of str to dict of str to float
        Contribution per grouping attribute, then per label. Every attribute
        set on at least one asset appears, in order of first appearance, and
        its labels in order of first appearance; assets without the attribute
        are reported under UNASSIGNED.

    Examples
    --------
    >>> from opoise import Problem, RiskTerm, risk_report, synthetic_data
    >>> data = synthetic_data()
    >>> problem = Problem(universe=data.universe, risk_model=data.risk_model,
    ...                   objective=[RiskTerm(aversion=1.0)])
    >>> report = risk_report(problem, data.benchmark)
    >>> round(report.total_risk, 6) > 0
    True
    """

    model_config = _CONFIG

    total_risk: float
    factor_risk: float
    specific_risk: float
    factor_exposures: dict[str, float]
    by_factor: dict[str, float]
    specific_contribution: float
    by_asset: dict[str, float]
    by_group: dict[str, dict[str, float]]


class ConstraintRowReport(BaseModel):
    """Value, bounds, slack and flags for one compiled constraint row.

    Attributes
    ----------
    constraint : str
        Name of the constraint object that produced the row.
    row : str
        Row label: the constraint name for a scalar row, or name[key] for one
        element of a vector row.
    value : float
        Value the row's expression took at the solution.
    lower : float or None
        Lower bound, or None when the row has no lower bound.
    upper : float or None
        Upper bound, or None when the row has no upper bound.
    slack : float
        Smallest distance to a finite bound; negative means the row is
        violated. Zero or near zero means binding.
    binding : bool
        True when the absolute slack is within the binding tolerance scaled by
        the size of the bound on the side that gave the slack.
    violated : bool
        True when the slack is below minus that tolerance, which can happen on
        a solve that only reached reduced accuracy.
    """

    model_config = _CONFIG

    constraint: str
    row: str
    value: float
    lower: float | None
    upper: float | None
    slack: float
    binding: bool
    violated: bool


class Diagnostics(BaseModel):
    """Everything measured on a solved portfolio besides its weights.

    Attributes
    ----------
    portfolio : RiskReport
        Absolute risk decomposition of the weights.
    active : RiskReport or None
        Active risk decomposition against the benchmark, or None when the
        problem has no benchmark.
    beta : float or None
        Predicted portfolio beta against the benchmark, or None when the
        problem has no benchmark.
    turnover : float
        Two-way turnover, the sum of absolute trades against the initial
        holdings; one-way turnover is half of it.
    constraints : list of ConstraintRowReport
        One entry per compiled row, in problem order.
    binding : list of str
        Row labels of the constraints that are binding.
    """

    model_config = _CONFIG

    portfolio: RiskReport
    active: RiskReport | None
    beta: float | None
    turnover: float
    constraints: list[ConstraintRowReport]
    binding: list[str]


class InfeasibilityReport(BaseModel):
    """A conflict set found by the deletion filter of an infeasible problem.

    Attributes
    ----------
    conflict_set : list of str
        Constraint names that cannot hold together; each member is needed for
        this conflict, and other conflicts may exist.
    method : str
        Always "deletion_filter".
    solves : int
        Number of feasibility solves spent finding the set.
    message : str
        Human-readable summary, including a warning when a subset solve was
        inconclusive and the set may not be minimal.
    """

    model_config = _CONFIG

    conflict_set: list[str]
    method: Literal["deletion_filter"] = "deletion_filter"
    solves: int
    message: str


class ScaReport(BaseModel):
    """Convergence and share report of a sequential convex approximation.

    A share-of-risk cap is not convex, so it is solved by a sequence of convex
    subproblems; this report says whether that sequence settled and what the
    final shares are. The shares are the true Euler shares of the returned
    weights, never the approximation the last subproblem used.

    Attributes
    ----------
    converged : bool
        True when the last iteration changed no weight by more than the
        solver setting sca_tol.
    iterations : int
        Number of convex subproblems solved.
    max_weight_change : float
        Largest absolute weight change of the last iteration.
    shares : dict of str to dict of str to float
        True share of the total (or active) risk per constraint name, then per
        group label in order of first appearance; every inner dict sums to 1
        when the portfolio has positive risk.
    """

    model_config = _CONFIG

    converged: bool
    iterations: int
    max_weight_change: float
    shares: dict[str, dict[str, float]]


Status = Literal["optimal", "optimal_inaccurate", "infeasible", "unbounded", "error"]
"""Solver outcome of a solve: only optimal and optimal_inaccurate carry a portfolio."""


class Result(BaseModel):
    """Outcome of one solve: status, timing, weights and measurements.

    For every status other than optimal and optimal_inaccurate the fields
    objective_value, weights, trades and diagnostics are None, because no
    portfolio was returned.

    Attributes
    ----------
    status : Status
        "optimal", "optimal_inaccurate", "infeasible", "unbounded" or "error".
        Reduced accuracy and solver failures are statuses, never exceptions.
    solver : str
        Solver that ran, as named in the solver settings.
    solve_seconds : float
        Wall-clock seconds spent inside the solver; compilation, diagnostics
        and infeasibility diagnosis are excluded.
    objective_value : float or None
        Optimal objective value, or None when there is no solution.
    weights : dict of str to float or None
        Portfolio weights in universe order, fractions of NAV.
    trades : dict of str to float or None
        Weight change per asset against the initial holdings, so the sum of
        absolute trades is the two-way turnover.
    diagnostics : Diagnostics or None
        Risk decomposition, beta, turnover and the constraint report.
    infeasibility : InfeasibilityReport or None
        Conflict set when the problem is infeasible and diagnosis was enabled.
    sca : ScaReport or None
        Convergence and share report when the sequential approximation ran to
        a result (status optimal or optimal_inaccurate); None for a problem
        without a share cap and on the failure paths, where the solve returns
        the base result or a status error without one.
    message : str
        Empty on success, otherwise an explanation of a non-optimal status.
    """

    model_config = _CONFIG

    status: Status
    solver: str
    solve_seconds: float
    objective_value: float | None = None
    weights: dict[str, float] | None = None
    trades: dict[str, float] | None = None
    diagnostics: Diagnostics | None = None
    infeasibility: InfeasibilityReport | None = None
    sca: ScaReport | None = None
    message: str = ""
