"""Compile a Problem into a cvxpy problem, and call the solver.

The compiler turns each declarative constraint into one Row (a scalar or a
vector of expressions with element-wise bounds) and then emits cvxpy
constraints from it: equalities, lower and upper inequalities, one vector
constraint per non-empty index set. The objective is a sum of aversion times a
squared norm of the risk parts, so every problem is a quadratic or a
second-order cone program, never anything a solver would reject as non-DCP.

run() is the only place the package calls a solver; optimize.solve and the
infeasibility diagnosis both go through it.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import cvxpy as cp
import numpy as np

from .constraints import (
    AssetBounds,
    BetaBounds,
    Budget,
    FactorExposureBounds,
    GroupBounds,
    GroupRiskCap,
    GroupRiskShareCap,
    LinearBounds,
    RiskCap,
    Turnover,
)
from .context import CompileContext, Row
from .errors import OpoiseError
from .models import Problem, SolverSettings


def budget(constraint: Budget, ctx: CompileContext) -> Row:
    """Build the row of a Budget constraint: the sum of the weights.

    Parameters
    ----------
    constraint : Budget
        The budget constraint.
    ctx : CompileContext
        Compilation context of the problem.

    Returns
    -------
    Row
        One scalar row fixing the sum of the weights to the requested total.
    """
    expression = cp.hstack([cp.sum(ctx.w)])
    total = float(constraint.total)
    return Row(
        constraint=constraint.name,
        labels=[constraint.name],
        expr=expression,
        lower=np.array([total]),
        upper=np.array([total]),
    )


def asset_bounds(constraint: AssetBounds, ctx: CompileContext) -> Row:
    """Build the row of an AssetBounds constraint: per-asset weight limits.

    A per_asset entry replaces both default sides for that asset. Only assets
    with at least one finite bound after the overrides get a row element, in
    universe order.

    Parameters
    ----------
    constraint : AssetBounds
        The bounds constraint.
    ctx : CompileContext
        Compilation context of the problem.

    Returns
    -------
    Row
        One vector row with an element per bounded asset.
    """
    lower = np.full(ctx.n, -np.inf)
    upper = np.full(ctx.n, np.inf)
    for asset_id in ctx.ids:
        override = constraint.per_asset.get(asset_id)
        low = override.lower if override is not None else constraint.lower
        high = override.upper if override is not None else constraint.upper
        position = ctx.index[asset_id]
        if low is not None:
            lower[position] = low
        if high is not None:
            upper[position] = high
    selected = np.flatnonzero(np.isfinite(lower) | np.isfinite(upper))
    return Row(
        constraint=constraint.name,
        labels=[f"{constraint.name}[{ctx.ids[position]}]" for position in selected],
        expr=ctx.x(constraint.active)[selected],
        lower=lower[selected],
        upper=upper[selected],
    )


def risk_cap(constraint: RiskCap, ctx: CompileContext) -> Row:
    """Build the row of a RiskCap constraint: a norm of the risk parts.

    The cap is a 2-norm of the risk parts of the weights or active weights, so
    it is a second-order cone and only carries an upper bound. With active set
    it is a tracking-error cap.

    Parameters
    ----------
    constraint : RiskCap
        The risk cap constraint.
    ctx : CompileContext
        Compilation context of the problem.

    Returns
    -------
    Row
        One convex scalar row bounded above by the limit.
    """
    expression = cp.hstack([cp.norm(ctx.risk_parts(ctx.x(constraint.active), constraint.measure), 2)])
    return Row(
        constraint=constraint.name,
        labels=[constraint.name],
        expr=expression,
        lower=np.array([-np.inf]),
        upper=np.array([float(constraint.limit)]),
        convex=True,
    )


def turnover(constraint: Turnover, ctx: CompileContext) -> Row:
    """Build the row of a Turnover constraint: two-way turnover against holdings.

    Turnover is the sum of absolute trades against the initial holdings, so
    the row is an L1 norm of the trade vector and only carries an upper bound.
    One-way turnover is half of the reported number.

    Parameters
    ----------
    constraint : Turnover
        The turnover constraint.
    ctx : CompileContext
        Compilation context of the problem.

    Returns
    -------
    Row
        One convex scalar row bounded above by the limit.
    """
    expression = cp.hstack([cp.norm1(ctx.w - ctx.w0)])
    return Row(
        constraint=constraint.name,
        labels=[constraint.name],
        expr=expression,
        lower=np.array([-np.inf]),
        upper=np.array([float(constraint.limit)]),
        convex=True,
    )


def group_bounds(constraint: GroupBounds, ctx: CompileContext) -> Row:
    """Build the row of a GroupBounds constraint: summed weights per group.

    Each element is the sum of the weights (or active weights) of the assets
    carrying one attribute value, so the row is a matrix of 0/1 group masks
    applied to the weight vector. Labels follow the order of the bounds dict.

    Parameters
    ----------
    constraint : GroupBounds
        The group bounds constraint.
    ctx : CompileContext
        Compilation context of the problem.

    Returns
    -------
    Row
        One affine vector row with an element per bounded group.
    """
    labels = list(constraint.bounds)
    masks = np.vstack([ctx.group_mask(constraint.attribute, label) for label in labels])
    lower = np.array(
        [-np.inf if constraint.bounds[label].lower is None else float(constraint.bounds[label].lower) for label in labels],
        dtype=float,
    )
    upper = np.array(
        [np.inf if constraint.bounds[label].upper is None else float(constraint.bounds[label].upper) for label in labels],
        dtype=float,
    )
    return Row(
        constraint=constraint.name,
        labels=[f"{constraint.name}[{label}]" for label in labels],
        expr=masks @ ctx.x(constraint.active),
        lower=lower,
        upper=upper,
    )


def group_risk_cap(constraint: GroupRiskCap, ctx: CompileContext) -> Row:
    """Build the row of a GroupRiskCap constraint: stand-alone group risk.

    Each element is the 2-norm of the risk parts of the group sub-portfolio,
    that is the group's own risk with every other asset set to zero. It is a
    second-order cone and only carries an upper bound. Labels follow the order
    of the caps dict.

    Parameters
    ----------
    constraint : GroupRiskCap
        The group risk cap constraint.
    ctx : CompileContext
        Compilation context of the problem.

    Returns
    -------
    Row
        One convex vector row bounded above by the caps.
    """
    labels = list(constraint.caps)
    weights = ctx.x(constraint.active)
    expressions = [
        cp.norm(
            ctx.risk_parts(cp.multiply(ctx.group_mask(constraint.attribute, label), weights), constraint.measure),
            2,
        )
        for label in labels
    ]
    return Row(
        constraint=constraint.name,
        labels=[f"{constraint.name}[{label}]" for label in labels],
        expr=cp.hstack(expressions),
        lower=np.full(len(labels), -np.inf),
        upper=np.array([float(constraint.caps[label]) for label in labels]),
        convex=True,
    )


def group_risk_share_cap(constraint: GroupRiskShareCap, ctx: CompileContext) -> Row:
    """Build the row of a GroupRiskShareCap: a tangent to the share cap.

    The true constraint, "the group's share of total risk is at most cap", is
    convex minus convex. Its concave part is replaced by the tangent at the
    linearisation point ctx.w_ref, which gives an inner approximation: every
    point the row admits satisfies the true cap, so each sequential
    subproblem stays DCP and its solution is feasible for the real problem.

    Parameters
    ----------
    constraint : GroupRiskShareCap
        The share-of-risk cap constraint.
    ctx : CompileContext
        Compilation context, holding the linearisation point w_ref.

    Returns
    -------
    Row
        One convex vector row bounded above by zero, one element per cap.

    Raises
    ------
    OpoiseError
        If the context has no linearisation point ("internal:
        group_risk_share_cap needs a linearisation point; call solve()").
    """
    if ctx.w_ref is None:
        raise OpoiseError("internal: group_risk_share_cap needs a linearisation point; call solve()")
    weights = ctx.x(constraint.active)
    reference = ctx.w_ref - ctx.b if constraint.active else ctx.w_ref
    labels = list(constraint.caps)
    expressions = []
    for label in labels:
        cap = float(constraint.caps[label])
        mask = ctx.group_mask(constraint.attribute, label)
        outside = 1.0 - mask
        complement = outside * reference
        sigma_complement = ctx.sigma_times(complement)
        sigma_reference = ctx.sigma_times(reference)
        constant = -0.25 * float(complement @ sigma_complement) - cap * float(reference @ sigma_reference)
        slope = -0.5 * outside * sigma_complement - 2.0 * cap * sigma_reference
        expressions.append(
            0.25 * cp.sum_squares(ctx.risk_parts(cp.multiply(1.0 + mask, weights), "total"))
            + constant
            + slope @ (weights - reference)
        )
    return Row(
        constraint=constraint.name,
        labels=[f"{constraint.name}[{label}]" for label in labels],
        expr=cp.hstack(expressions),
        lower=np.full(len(labels), -np.inf),
        upper=np.zeros(len(labels)),
        convex=True,
    )


def beta_bounds(constraint: BetaBounds, ctx: CompileContext) -> Row:
    """Build the row of a BetaBounds constraint: the predicted portfolio beta.

    The predicted beta of every asset is data (a vector computed from the
    benchmark), so the portfolio beta is an affine function of the weights.
    The row always uses the absolute weights, never the active ones.

    Parameters
    ----------
    constraint : BetaBounds
        The beta bounds constraint.
    ctx : CompileContext
        Compilation context of the problem.

    Returns
    -------
    Row
        One affine scalar row with the requested bounds.
    """
    expression = cp.hstack([ctx.beta @ ctx.w])
    lower = -np.inf if constraint.lower is None else float(constraint.lower)
    upper = np.inf if constraint.upper is None else float(constraint.upper)
    return Row(
        constraint=constraint.name,
        labels=[constraint.name],
        expr=expression,
        lower=np.array([lower]),
        upper=np.array([upper]),
    )


def factor_exposure_bounds(constraint: FactorExposureBounds, ctx: CompileContext) -> Row:
    """Build the row of a FactorExposureBounds constraint: factor exposures.

    Each element is one column of the exposure matrix applied to the weights
    (or active weights), so the row is affine. Labels follow the order of the
    bounds dict; the factor columns come from the risk model.

    Parameters
    ----------
    constraint : FactorExposureBounds
        The factor exposure bounds constraint.
    ctx : CompileContext
        Compilation context of the problem.

    Returns
    -------
    Row
        One affine vector row with an element per bounded factor.
    """
    column_of = {factor_id: position for position, factor_id in enumerate(ctx.problem.risk_model.factor_ids)}
    labels = list(constraint.bounds)
    columns = [column_of[label] for label in labels]
    lower = np.array(
        [-np.inf if constraint.bounds[label].lower is None else float(constraint.bounds[label].lower) for label in labels],
        dtype=float,
    )
    upper = np.array(
        [np.inf if constraint.bounds[label].upper is None else float(constraint.bounds[label].upper) for label in labels],
        dtype=float,
    )
    return Row(
        constraint=constraint.name,
        labels=[f"{constraint.name}[{label}]" for label in labels],
        expr=ctx.B[:, columns].T @ ctx.x(constraint.active),
        lower=lower,
        upper=upper,
    )


def linear_bounds(constraint: LinearBounds, ctx: CompileContext) -> Row:
    """Build the row of a LinearBounds constraint: a linear combination.

    The coefficients are placed in universe order (assets without a
    coefficient are zero), so the row is one affine scalar on the weights.
    The row always uses the absolute weights, never the active ones.

    Parameters
    ----------
    constraint : LinearBounds
        The linear bounds constraint.
    ctx : CompileContext
        Compilation context of the problem.

    Returns
    -------
    Row
        One affine scalar row with the requested bounds.
    """
    coefficients = np.zeros(ctx.n)
    for asset_id, value in constraint.coefficients.items():
        coefficients[ctx.index[asset_id]] = float(value)
    expression = cp.hstack([coefficients @ ctx.w])
    lower = -np.inf if constraint.lower is None else float(constraint.lower)
    upper = np.inf if constraint.upper is None else float(constraint.upper)
    return Row(
        constraint=constraint.name,
        labels=[constraint.name],
        expr=expression,
        lower=np.array([lower]),
        upper=np.array([upper]),
    )


ROW_BUILDERS: dict[str, Callable[[Any, CompileContext], Row]] = {
    "budget": budget,
    "asset_bounds": asset_bounds,
    "risk_cap": risk_cap,
    "turnover": turnover,
    "group_bounds": group_bounds,
    "group_risk_cap": group_risk_cap,
    "group_risk_share_cap": group_risk_share_cap,
    "beta_bounds": beta_bounds,
    "factor_exposure_bounds": factor_exposure_bounds,
    "linear_bounds": linear_bounds,
}
"""Row builder per constraint type tag; a type without a builder is an error."""


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
        convex constraint ("internal: convex row with a lower bound").
    """
    lower = np.asarray(row.lower, dtype=float)
    upper = np.asarray(row.upper, dtype=float)
    has_lower = np.isfinite(lower)
    has_upper = np.isfinite(upper)
    if row.convex and bool(has_lower.any()):
        raise OpoiseError("internal: convex row with a lower bound")
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
        If a constraint type has no row builder ("internal: no row builder for
        '...'") or a share-of-risk cap has no linearisation point ("internal:
        group_risk_share_cap needs a linearisation point; call solve()").
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
        builder = ROW_BUILDERS.get(constraint.type)
        if builder is None:
            raise OpoiseError(f"internal: no row builder for '{constraint.type}'")
        row = builder(constraint, context)
        emitted = emit(row)
        rows.append(row)
        constraint_map[constraint.name] = emitted
        constraints.extend(emitted)
    if feasibility:
        objective = cp.Minimize(0)
    else:
        objective = cp.Minimize(
            sum(
                term.aversion * cp.sum_squares(context.risk_parts(context.x(term.active), term.measure))
                for term in problem.objective
            )
        )
    return CompiledProblem(
        cvx=cp.Problem(objective, constraints),
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
