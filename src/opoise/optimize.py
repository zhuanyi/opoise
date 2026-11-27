"""Solve a problem and turn the solver outcome into a Result.

solve() is the public entry point: it compiles the declarative problem, calls
the solver through compiler.run, maps the raw solver status onto the Status
literal and, when a portfolio came back, measures it with the risk analytics.
Solver quirks are outcomes, not exceptions: apart from an unavailable solver
and OSQP being asked for a cone problem, everything becomes a status.

The module is called optimize, not solve, because PEP 562 lazy exports would
otherwise rebind the package attribute solve to this module and break callers.
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np

from .compiler import Row, compile_problem, run
from .diagnostics import risk_report
from .errors import OpoiseError, SolverCapabilityError, SolverNotAvailableError
from .infeasibility import find_conflict
from .models import Problem
from .result import ConstraintRowReport, Diagnostics, Result, Status

_STATUS = {
    "optimal": "optimal",
    "optimal_inaccurate": "optimal_inaccurate",
    "infeasible": "infeasible",
    "infeasible_inaccurate": "infeasible",
    "infeasible_or_unbounded": "infeasible",
    "unbounded": "unbounded",
    "unbounded_inaccurate": "unbounded",
}
"""Raw solver status to result status; every other raw status maps to error."""


def map_status(raw: str) -> Status:
    """Map a raw solver status onto the five result statuses.

    Anything the mapping does not name, including a cvxpy failure such as
    "solver_error" or a limit such as "user_limit", becomes "error".

    Parameters
    ----------
    raw : str
        Status string reported by the solver (or by run()).

    Returns
    -------
    Status
        "optimal", "optimal_inaccurate", "infeasible", "unbounded" or
        "error".
    """
    return _STATUS.get(raw, "error")


def evaluate_rows(rows: list[Row], binding_tol: float) -> list[ConstraintRowReport]:
    """Measure every row at the solution and flag binding and violated rows.

    The slack of an element is its smallest distance to a finite bound,
    negative when the element is outside it. The tolerance is binding_tol
    times max(1, the size of the bound that gave the slack), so it is absolute
    for unit-scale bounds and relative above them.

    Parameters
    ----------
    rows : list of Row
        Rows of a compiled problem, with solver values on their expressions.
    binding_tol : float
        Base tolerance for a row to count as binding.

    Returns
    -------
    list of ConstraintRowReport
        One report per row element, in row order then label order.

    Raises
    ------
    OpoiseError
        If a row element has no finite bound at all, which no builder
        produces.
    """
    reports: list[ConstraintRowReport] = []
    for row in rows:
        values = np.asarray(row.expr.value, dtype=float).reshape(-1)
        for position, label in enumerate(row.labels):
            value = float(values[position])
            lower = float(row.lower[position])
            upper = float(row.upper[position])
            sides: list[tuple[float, float]] = []
            if np.isfinite(lower):
                sides.append((value - lower, lower))
            if np.isfinite(upper):
                sides.append((upper - value, upper))
            if not sides:
                raise OpoiseError(f"internal: row '{label}' has no finite bound")
            slack, bound = min(sides, key=lambda side: side[0])
            tolerance = binding_tol * max(1.0, abs(bound))
            reports.append(
                ConstraintRowReport(
                    constraint=row.constraint,
                    row=label,
                    value=value,
                    lower=lower if np.isfinite(lower) else None,
                    upper=upper if np.isfinite(upper) else None,
                    slack=float(slack),
                    binding=bool(abs(slack) <= tolerance),
                    violated=bool(slack < -tolerance),
                )
            )
    return reports


def solve(problem: Problem) -> Result:
    """Solve a problem and return its weights, diagnostics and status.

    The solver is chosen by the problem settings. A solver that is not
    installed, or OSQP asked for a problem with second-order cone rows, raises
    because they are configuration errors; an infeasible, unbounded or failed
    solve is a status with a message and no portfolio.

    Parameters
    ----------
    problem : Problem
        The validated problem to solve.

    Returns
    -------
    Result
        Status, solver, solve seconds and, for the two optimal statuses, the
        objective value, weights, trades and diagnostics. solve_seconds counts
        the solver call only, not compilation or measurement.

    Raises
    ------
    SolverNotAvailableError
        If the requested solver is not installed ("solver {name} is not
        installed (installed: [...])").
    SolverCapabilityError
        If the solver is OSQP and the problem has cone rows ("OSQP solves QPs
        only; ..."), or if the problem has a share-of-risk cap and the solver
        is OSQP ("OSQP solves QPs only; the share-of-risk subproblems ...").

    Notes
    -----
    A problem with a GroupRiskShareCap constraint is not convex; it goes
    through opoise.sca, which iterates convex subproblems and reports whether
    it converged. The result then also carries a ScaReport.

    Examples
    --------
    >>> from opoise import Budget, Problem, RiskTerm, solve, synthetic_data
    >>> data = synthetic_data()
    >>> result = solve(Problem(universe=data.universe, risk_model=data.risk_model,
    ...                        objective=[RiskTerm(aversion=1.0)],
    ...                        constraints=[Budget(name="budget")]))
    >>> result.status, len(result.weights)
    ('optimal', 20)
    """
    settings = problem.settings
    installed = cp.installed_solvers()
    if settings.solver not in installed:
        raise SolverNotAvailableError(
            f"solver {settings.solver} is not installed (installed: {sorted(installed)})"
        )
    if any(constraint.type == "group_risk_share_cap" for constraint in problem.constraints):
        from .sca import solve_sca

        return solve_sca(problem)
    compiled = compile_problem(problem)
    if settings.solver == "OSQP" and not compiled.cvx.is_qp():
        raise SolverCapabilityError(
            "OSQP solves QPs only; this problem has cone rows (risk_cap or group_risk_cap); "
            "use CLARABEL, SCS or MOSEK"
        )
    raw, seconds, error = run(compiled.cvx, settings)
    if raw == "solver_error":
        return Result(status="error", solver=settings.solver, solve_seconds=seconds, message=error)
    status = map_status(raw)
    if status in ("optimal", "optimal_inaccurate"):
        context = compiled.context
        weights_vector = np.asarray(context.w.value, dtype=float).reshape(-1)
        weights = {
            asset_id: float(weights_vector[position])
            for position, asset_id in enumerate(context.ids)
        }
        trades = {
            asset_id: float(weights_vector[position] - context.w0[position])
            for position, asset_id in enumerate(context.ids)
        }
        reports = evaluate_rows(compiled.rows, settings.binding_tol)
        diagnostics = Diagnostics(
            portfolio=risk_report(problem, weights),
            active=(
                risk_report(problem, weights, active=True)
                if problem.benchmark is not None
                else None
            ),
            beta=float(context.beta @ weights_vector) if context.beta is not None else None,
            turnover=float(sum(abs(trade) for trade in trades.values())),
            constraints=reports,
            binding=[report.row for report in reports if report.binding],
        )
        return Result(
            status=status,
            solver=settings.solver,
            solve_seconds=seconds,
            objective_value=float(compiled.cvx.value),
            weights=weights,
            trades=trades,
            diagnostics=diagnostics,
            message="solver reported reduced accuracy" if status == "optimal_inaccurate" else "",
        )
    if status == "infeasible":
        if settings.diagnose_infeasibility:
            report = find_conflict(problem)
            return Result(
                status="infeasible",
                solver=settings.solver,
                solve_seconds=seconds,
                infeasibility=report,
                message=report.message,
            )
        return Result(
            status="infeasible",
            solver=settings.solver,
            solve_seconds=seconds,
            message="infeasible (diagnosis disabled)",
        )
    return Result(
        status=status, solver=settings.solver, solve_seconds=seconds, message=f"solver status {raw}"
    )
