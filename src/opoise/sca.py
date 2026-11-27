"""Sequential convex approximation for sector share-of-risk caps.

A GroupRiskShareCap bounds a group's Euler share of the total (or active)
risk, x_g-transpose Sigma x / x-transpose Sigma x <= c_g, which is an
indefinite quadratic form and therefore not convex. This module solves it by
the convex-concave procedure (Lipp and Boyd 2016): each iteration replaces the
concave part of the constraint with its tangent at the current weights, which
is an inner approximation, so every iterate satisfies the true caps and the
subproblems stay DCP. The result is a local optimum of the non-convex
problem; the ScaReport says whether the iteration converged.

opoise.optimize imports this module only when a problem carries such a cap, so
importing opoise still does not import cvxpy.
"""
from __future__ import annotations

import numpy as np

from .compiler import CompiledProblem, compile_problem, run
from .diagnostics import risk_report
from .errors import OpoiseError, SolverCapabilityError
from .models import Problem
from .result import ConstraintRowReport, Diagnostics, Result, ScaReport

_OSQP_REFUSAL = (
    "OSQP solves QPs only; the share-of-risk subproblems have quadratic constraints; "
    "use CLARABEL, SCS or MOSEK"
)
"""Message of the capability error raised when the solver is OSQP."""


def solve_sca(problem: Problem) -> Result:
    """Solve a problem whose share-of-risk caps make it non-convex.

    The problem without its share caps is solved first, which gives the
    starting point and detects infeasibility of everything else. Then each
    iteration linearises the caps around the previous weights and solves the
    convex subproblem; the iteration stops when no weight moves by more than
    the solver setting sca_tol.

    Parameters
    ----------
    problem : Problem
        A validated problem with at least one GroupRiskShareCap constraint.

    Returns
    -------
    Result
        The last iterate measured exactly like a convex solve, with the share
        rows carrying the true shares and sca set to a ScaReport. A base
        problem that is not optimal comes back unchanged except for its
        message; a subproblem that fails after iteration one comes back with
        status error.

    Raises
    ------
    SolverCapabilityError
        If the solver is OSQP, whose subproblems would carry a convex
        quadratic constraint and are not QPs.
    OpoiseError
        If a constraint other than GroupRiskShareCap sets requires_sca, which
        the SCA linearisation does not support ("... the SCA solver only
        linearises group_risk_share_cap").

    Notes
    -----
    The method is local: the answer depends on the starting point (the
    unconstrained minimum of the problem without the caps) and is not
    guaranteed to be the global optimum. A cap is only reported as converged
    when the last iteration moved no weight by more than sca_tol; running out
    of iterations gives status optimal_inaccurate with converged False, never
    a silent optimum.
    """
    from .constraints.risk import GroupRiskShareCap
    from .optimize import evaluate_rows, map_status, solve

    settings = problem.settings
    if settings.solver == "OSQP":
        raise SolverCapabilityError(_OSQP_REFUSAL)
    for constraint in problem.constraints:
        if constraint.requires_sca and not isinstance(constraint, GroupRiskShareCap):
            raise OpoiseError(
                f"constraint '{constraint.name}' of type '{constraint.type}' sets requires_sca, "
                "but the SCA solver only linearises group_risk_share_cap"
            )
    cap_of = {
        constraint.name: constraint
        for constraint in problem.constraints
        if constraint.requires_sca
    }
    base = problem.model_copy(
        update={
            "constraints": [
                constraint for constraint in problem.constraints if not constraint.requires_sca
            ]
        }
    )
    # shortcut: model_copy skips re-validation; the constraints were validated with the problem
    start = solve(base)
    if start.status not in ("optimal", "optimal_inaccurate"):
        return start.model_copy(update={"message": "without the share caps: " + start.message})
    ids = problem.universe.ids()
    previous = np.array([start.weights[asset_id] for asset_id in ids], dtype=float)
    elapsed = start.solve_seconds
    compiled: CompiledProblem | None = None
    raw = ""
    iterations = 0
    change = np.inf
    converged = False
    for iteration in range(1, settings.sca_max_iterations + 1):
        compiled = compile_problem(problem, w_ref=previous)
        raw, seconds, error = run(compiled.cvx, settings)
        elapsed += seconds
        if raw == "solver_error":
            return Result(status="error", solver=settings.solver, solve_seconds=elapsed, message=error)
        status = map_status(raw)
        if status not in ("optimal", "optimal_inaccurate"):
            return Result(
                status="error",
                solver=settings.solver,
                solve_seconds=elapsed,
                message=(
                    f"sequential approximation: iteration {iteration} subproblem has status {raw}; "
                    "the share caps may be unattainable, or the starting point is too far from the "
                    "feasible region (not a proof of infeasibility)"
                ),
            )
        weights_vector = np.asarray(compiled.context.w.value, dtype=float).reshape(-1)
        change = float(np.max(np.abs(weights_vector - previous)))
        previous = weights_vector
        iterations = iteration
        if change <= settings.sca_tol:
            converged = True
            break
    weights = {asset_id: float(previous[position]) for position, asset_id in enumerate(ids)}
    trades = {
        asset_id: float(previous[position] - compiled.context.w0[position])
        for position, asset_id in enumerate(ids)
    }
    shares = _true_shares(problem, weights, cap_of)
    reports = _row_reports(compiled, cap_of, shares, settings.binding_tol, evaluate_rows)
    diagnostics = Diagnostics(
        portfolio=risk_report(problem, weights),
        active=risk_report(problem, weights, active=True) if problem.benchmark is not None else None,
        beta=float(compiled.context.beta @ previous) if compiled.context.beta is not None else None,
        turnover=float(sum(abs(trade) for trade in trades.values())),
        constraints=reports,
        binding=[report.row for report in reports if report.binding],
    )
    last_status = map_status(raw)
    if converged:
        status = last_status
        message = "solver reported reduced accuracy" if status == "optimal_inaccurate" else ""
    else:
        status = "optimal_inaccurate"
        message = (
            f"sequential approximation did not converge in {iterations} iterations (last max weight "
            f"change {change:.3g} > sca_tol {settings.sca_tol:g}); weights satisfy the share caps but "
            "may be sub-optimal"
        )
    return Result(
        status=status,
        solver=settings.solver,
        solve_seconds=elapsed,
        objective_value=float(compiled.cvx.value),
        weights=weights,
        trades=trades,
        diagnostics=diagnostics,
        sca=ScaReport(converged=converged, iterations=iterations, max_weight_change=change, shares=shares),
        message=message,
    )


def _true_shares(problem: Problem, weights: dict[str, float], cap_of: dict) -> dict[str, dict[str, float]]:
    """Return the true Euler share of every label, per share-cap constraint.

    Parameters
    ----------
    problem : Problem
        The problem that was solved.
    weights : dict of str to float
        Final weights in universe order.
    cap_of : dict
        Share-cap constraints keyed by constraint name.

    Returns
    -------
    dict of str to dict of str to float
        Share per constraint name, then per group label in order of first
        appearance (UNASSIGNED included); 0.0 for every label when the
        portfolio has zero risk.
    """
    shares: dict[str, dict[str, float]] = {}
    for name, constraint in cap_of.items():
        report = risk_report(problem, weights, active=constraint.active)
        total = report.total_risk
        contributions = report.by_group[constraint.attribute]
        shares[name] = {
            label: (float(contribution) / total if total > 0.0 else 0.0)
            for label, contribution in contributions.items()
        }
    return shares


def _row_reports(
    compiled: CompiledProblem,
    cap_of: dict,
    shares: dict[str, dict[str, float]],
    binding_tol: float,
    evaluate_rows,
) -> list[ConstraintRowReport]:
    """Measure every compiled row, using true shares for the share rows.

    Parameters
    ----------
    compiled : CompiledProblem
        The last compiled subproblem.
    cap_of : dict
        Share-cap constraints keyed by constraint name.
    shares : dict of str to dict of str to float
        True share per constraint name and label.
    binding_tol : float
        Base tolerance for a row to count as binding.
    evaluate_rows : callable
        The row evaluator of opoise.optimize, used for every other row.

    Returns
    -------
    list of ConstraintRowReport
        One report per row element, in problem order and label order.
    """
    reports: list[ConstraintRowReport] = []
    for row in compiled.rows:
        constraint = cap_of.get(row.constraint)
        if constraint is None:
            reports.extend(evaluate_rows([row], binding_tol))
            continue
        for label in row.labels:
            group = label[len(row.constraint) + 1 : -1]
            value = shares[row.constraint][group]
            upper = float(constraint.caps[group])
            slack = upper - value
            tolerance = binding_tol * max(1.0, abs(upper))
            reports.append(
                ConstraintRowReport(
                    constraint=row.constraint,
                    row=label,
                    value=value,
                    lower=None,
                    upper=upper,
                    slack=float(slack),
                    binding=bool(abs(slack) <= tolerance),
                    violated=bool(slack < -tolerance),
                )
            )
    return reports
