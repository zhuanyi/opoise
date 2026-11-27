"""Compile a Problem into a cvxpy problem, and call the solver.

Each constraint builds its own Row (a scalar or a vector of expressions with
element-wise bounds) and the compiler emits cvxpy constraints from it:
equalities, lower and upper inequalities, one vector constraint per non-empty
index set. Each objective term builds its own cvxpy expression; the compiler
checks that it is a scalar and convex (or affine) and minimises the sum. The
built-in terms are aversion times a squared norm of the risk parts, so built-in
problems are quadratic or second-order cone programs.

run() is the only place the package calls a solver; optimize.solve and the
infeasibility diagnosis both go through it.
"""
from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cvxpy as cp
import numpy as np

from .context import CompileContext, Row
from .errors import OpoiseError
from .models import Problem, SolverSettings

if TYPE_CHECKING:
    from .constraints import ConstraintBase
    from .objectives import ObjectiveBase


def _curvature(expr: cp.Expression) -> str:
    """Return the curvature of an expression as one short word."""
    if expr.is_affine():
        return "affine"
    if expr.is_convex():
        return "convex"
    if expr.is_concave():
        return "concave"
    return "neither convex nor concave"


def _check_row(constraint: ConstraintBase, row: Row) -> None:
    """Check that a constraint built the row the compiler needs.

    Parameters
    ----------
    constraint : ConstraintBase
        The constraint whose build() produced the row.
    row : Row
        The row to check.

    Raises
    ------
    OpoiseError
        If the row names another constraint, if its labels and expression do
        not share their length, if the expression shape is not one-dimensional
        with one element per label ("constraint '<name>' of type '<tag>':
        build() returned an expression of shape ..."), or if the expression is
        neither affine nor a convex row with upper bounds only ("constraint
        '<name>' of type '<tag>': build() returned a non-DCP row (...)...").
    """
    tag = constraint.type
    if row.constraint != constraint.name:
        raise OpoiseError(
            f"constraint '{constraint.name}' of type '{tag}': "
            f"build() returned a row for constraint '{row.constraint}'"
        )
    shape = getattr(row.expr, "shape", None)
    if shape != (len(row.labels),):
        raise OpoiseError(
            f"constraint '{constraint.name}' of type '{tag}': build() returned an expression "
            f"of shape {shape} for {len(row.labels)} labels; "
            "it must be 1-D with one element per label"
        )
    if len({len(row.labels), len(row.lower), len(row.upper)}) != 1:
        raise OpoiseError(
            f"constraint '{constraint.name}' of type '{tag}': build() returned "
            f"{len(row.labels)} labels, {len(row.lower)} lower bounds and "
            f"{len(row.upper)} upper bounds"
        )
    curvature = _curvature(row.expr)
    if curvature != "affine" and not (row.convex and curvature == "convex"):
        raise OpoiseError(
            f"constraint '{constraint.name}' of type '{tag}': build() returned a non-DCP row "
            f"({curvature}); a row must be affine, or convex with upper bounds only"
        )


def _objective_term(index: int, term: ObjectiveBase, context: CompileContext) -> cp.Expression:
    """Build one objective term and check that Minimize can take it.

    Parameters
    ----------
    index : int
        Position of the term in the objective, for the error message.
    term : ObjectiveBase
        The term to build; any registered objective type.
    context : CompileContext
        Compilation context of the problem.

    Returns
    -------
    cvxpy.Expression
        The term's convex expression.

    Raises
    ------
    OpoiseError
        If build() returns something that is not a cvxpy expression
        ("objective term <i> of type '<tag>': build() must return a cvxpy
        expression, got <type>"), a non-scalar expression ("objective term <i>
        of type '<tag>': build() must return a scalar expression, got shape
        <shape>"), or a non-convex expression ("objective term <i> of type
        '<tag>': build() returned a non-convex expression (<curvature>);
        Minimize needs a convex term").
    """
    expression = term.build(context)
    if not isinstance(expression, cp.Expression):
        raise OpoiseError(
            f"objective term {index} of type '{term.type}': build() must return a "
            f"cvxpy expression, got {type(expression).__name__}"
        )
    if not expression.is_scalar():
        raise OpoiseError(
            f"objective term {index} of type '{term.type}': build() must return a "
            f"scalar expression, got shape {expression.shape}"
        )
    curvature = _curvature(expression)
    if curvature not in ("affine", "convex"):
        raise OpoiseError(
            f"objective term {index} of type '{term.type}': build() returned a non-convex "
            f"expression ({curvature}); Minimize needs a convex term"
        )
    return expression


@dataclass
class CompiledProblem:
    """A cvxpy problem together with the rows it was built from.

    Attributes
    ----------
    cvx : cvxpy.Problem
        The assembled problem, ready for a solver.
    rows : list of Row
        Logical rows in problem order (a subset when constraint names were
        given); the result report and the row evaluation read them back.
    constraint_map : dict of str to list of cvxpy.Constraint
        Emitted cvxpy constraints per constraint name.
    context : CompileContext
        The context holding the weight variable and the sliced data.
    """

    cvx: cp.Problem
    rows: list[Row]
    constraint_map: dict[str, list[cp.Constraint]]
    context: CompileContext


def emit(row: Row) -> list[cp.Constraint]:
    """Turn one Row into the cvxpy constraints that express its bounds.

    Elements that are equal on both sides become equalities, the rest become
    lower and upper inequalities; each kind is emitted as one vector
    constraint over the elements of that kind.

    Parameters
    ----------
    row : Row
        The row to emit.

    Returns
    -------
    list of cvxpy.Constraint
        One constraint per non-empty index set; empty when every element is
        unbounded (which no builder produces).

    Raises
    ------
    OpoiseError
        If the row is convex and carries a lower bound, which would not be a
        convex constraint ("constraint '<name>': convex row with a lower
        bound").
    """
    lower = np.asarray(row.lower, dtype=float)
    upper = np.asarray(row.upper, dtype=float)
    has_lower = np.isfinite(lower)
    has_upper = np.isfinite(upper)
    if row.convex and bool(has_lower.any()):
        raise OpoiseError(f"constraint '{row.constraint}': convex row with a lower bound")
    equality = has_lower & has_upper & (lower == upper)
    parts = (
        (equality, "equality"),
        (has_lower & ~equality, "lower"),
        (has_upper & ~equality, "upper"),
    )
    emitted: list[cp.Constraint] = []
    for mask, kind in parts:
        positions = np.flatnonzero(mask)
        if positions.size == 0:
            continue
        selected = row.expr[positions]
        if kind == "equality":
            emitted.append(selected == lower[positions])
        elif kind == "lower":
            emitted.append(selected >= lower[positions])
        else:
            emitted.append(selected <= upper[positions])
    return emitted


def compile_problem(
    problem: Problem,
    constraint_names: Sequence[str] | None = None,
    feasibility: bool = False,
    w_ref: np.ndarray | None = None,
) -> CompiledProblem:
    """Compile a problem into a cvxpy problem.

    Parameters
    ----------
    problem : Problem
        The validated problem to compile.
    constraint_names : sequence of str or None
        Restrict the compiled constraints to these names (in problem order);
        None compiles every constraint of the problem.
    feasibility : bool
        Minimise the zero objective instead of the problem objective, which
        turns the problem into a pure feasibility question.
    w_ref : numpy.ndarray or None
        Linearisation point, in universe order, for a share-of-risk cap; the
        sequential approximation passes its previous iterate and a call
        without one fails on such a constraint.

    Returns
    -------
    CompiledProblem
        The cvxpy problem, its rows in problem order, the emitted constraints
        per name and the compilation context.

    Raises
    ------
    ValueError
        If constraint_names names a constraint the problem does not have.
    OpoiseError
        If a constraint builds a malformed or non-DCP row ("constraint
        '<name>' of type '<tag>': ..."), an objective term does not build a
        scalar convex cvxpy expression ("objective term <i> of type '<tag>':
        ..."), a share-of-risk
        cap has no linearisation point ("internal: group_risk_share_cap needs
        a linearisation point; call solve()"), or the compiled problem is not
        DCP ("internal: compiled problem is not DCP").
    """
    context = CompileContext(problem, w_ref)
    selected = problem.constraints
    if constraint_names is not None:
        wanted = set(constraint_names)
        unknown = sorted(wanted - {constraint.name for constraint in problem.constraints})
        if unknown:
            raise ValueError(f"unknown constraint name(s): {', '.join(unknown)}")
        selected = [constraint for constraint in problem.constraints if constraint.name in wanted]
    rows: list[Row] = []
    constraint_map: dict[str, list[cp.Constraint]] = {}
    constraints: list[cp.Constraint] = []
    for constraint in selected:
        row = constraint.build(context)
        _check_row(constraint, row)
        emitted = emit(row)
        rows.append(row)
        constraint_map[constraint.name] = emitted
        constraints.extend(emitted)
    if feasibility:
        objective = cp.Minimize(0)
    else:
        objective = cp.Minimize(
            sum(_objective_term(index, term, context) for index, term in enumerate(problem.objective))
        )
    cvx = cp.Problem(objective, constraints)
    if not cvx.is_dcp():
        raise OpoiseError("internal: compiled problem is not DCP")
    return CompiledProblem(
        cvx=cvx,
        rows=rows,
        constraint_map=constraint_map,
        context=context,
    )


def run(cvx: cp.Problem, settings: SolverSettings) -> tuple[str, float, str]:
    """Call the solver on a compiled problem and report its raw outcome.

    The returned status is the solver's own status string, so a status the
    result mapping does not know (for instance "user_limit") stays visible.
    Solver failures never propagate as exceptions: a cvxpy SolverError and an
    option the solver rejects both come back as the status "solver_error" with
    the solver's message.

    Parameters
    ----------
    cvx : cvxpy.Problem
        The compiled problem to solve.
    settings : SolverSettings
        Solver name, options and verbose flag.

    Returns
    -------
    tuple
        (raw status, seconds spent in the solver call, message). The message
        is empty unless the status is "solver_error".

    Raises
    ------
    TypeError
        From the solver call when it rejects an argument with no
        solver_options set; such a call is a bug, not an input error.
    ValueError
        Same, for a solver that reports a bad call that way.
    """
    started = time.perf_counter()
    try:
        cvx.solve(solver=settings.solver, verbose=settings.verbose, **settings.solver_options)
    except cp.error.SolverError as error:
        return "solver_error", time.perf_counter() - started, str(error)
    except (TypeError, ValueError) as error:
        if not settings.solver_options:
            raise
        rejected = sorted(settings.solver_options)
        return (
            "solver_error",
            time.perf_counter() - started,
            f"{settings.solver} rejected solver_options {rejected}: {error}",
        )
    return cvx.status, time.perf_counter() - started, ""
