"""Compilation context and the logical row the compiler emits.

A CompileContext holds everything the row builders need: the numpy data of a
problem sliced to its universe rows, the factor square root that turns a
variance into a squared norm, and the single cvxpy decision variable. A Row is
one logical constraint (a scalar or one vector) before it is split into cvxpy
constraints.

Only this module, opoise.compiler, opoise.optimize and the later modules that
call a solver import cvxpy, so 'import opoise' leaves it unloaded.
"""
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from .diagnostics import beta_vector
from .models import Problem


@dataclass
class Row:
    """One logical constraint row, before emission as cvxpy constraints.

    The expression has one element per label. Every element is compared with
    the matching element of the bounds, so a scalar row is a vector row of
    length one.

    Attributes
    ----------
    constraint : str
        Name of the constraint object that produced the row.
    labels : list of str
        Report labels, one per element: the constraint name for a scalar row,
        name[key] for a vector row.
    expr : cvxpy.Expression
        Affine or convex expression of shape (len(labels),).
    lower : numpy.ndarray
        Float64 lower bounds, -infinity where the element has no lower bound.
    upper : numpy.ndarray
        Float64 upper bounds, +infinity where the element has no upper bound.
    convex : bool
        True when the expression is convex rather than affine (a norm); such a
        row may only carry upper bounds.
    """

    constraint: str
    labels: list[str]
    expr: cp.Expression
    lower: np.ndarray
    upper: np.ndarray
    convex: bool = False


class CompileContext:
    """Numpy data, the risk factorisation and the decision variable of a problem.

    All arrays are float64 and follow the universe order of the problem; the
    risk model is sliced to the universe rows, so a model that covers more
    assets costs nothing extra.

    Attributes
    ----------
    problem : Problem
        The problem being compiled.
    ids : list of str
        Asset ids in universe order; defines the coordinate order of every
        array and of the weight vector.
    n : int
        Number of universe assets.
    index : dict of str to int
        Position of each asset id in the universe.
    B : numpy.ndarray
        Factor exposures on the universe rows, shape (n, k).
    F : numpy.ndarray
        Factor covariance, shape (k, k).
    d : numpy.ndarray
        Specific variances on the universe rows, shape (n,).
    LtBt : numpy.ndarray
        L-transpose times B-transpose, shape (k, n), where L is the factor
        square root (F = L L-transpose, negative eigenvalues clipped to zero).
        Its product with x is the factor block of the risk parts, whose squared
        norm is x-transpose B F B-transpose x.
    sqrt_d : numpy.ndarray
        Square roots of the specific variances, shape (n,).
    w0 : numpy.ndarray
        Initial holdings in universe order, zero where not declared.
    b : numpy.ndarray or None
        Benchmark weights in universe order, or None without a benchmark.
    beta : numpy.ndarray or None
        Predicted beta of each asset against the benchmark, from beta_vector,
        or None without a benchmark.
    w : cvxpy.Variable
        The weight variable, shape (n,), named "w".
    w_ref : numpy.ndarray or None
        Linearisation point of a share-of-risk cap, in universe order, or
        None when the compilation is not inside the sequential approximation.

    Parameters
    ----------
    problem : Problem
        A validated problem; the constructor assumes its validation passed.
    w_ref : numpy.ndarray or None
        Linearisation point for the share-of-risk row builder, in universe
        order; None everywhere except inside the sequential approximation.

    Examples
    --------
    >>> from opoise import Problem, RiskTerm, synthetic_data
    >>> from opoise.compiler import CompileContext
    >>> data = synthetic_data()
    >>> context = CompileContext(Problem(universe=data.universe, risk_model=data.risk_model,
    ...                                  objective=[RiskTerm(aversion=1.0)]))
    >>> context.n, context.w.shape
    (20, (20,))
    """

    problem: Problem
    ids: list[str]
    n: int
    index: dict[str, int]
    B: np.ndarray
    F: np.ndarray
    d: np.ndarray
    LtBt: np.ndarray
    sqrt_d: np.ndarray
    w0: np.ndarray
    b: np.ndarray | None
    beta: np.ndarray | None
    w: cp.Variable
    w_ref: np.ndarray | None

    def __init__(self, problem: Problem, w_ref: np.ndarray | None = None) -> None:
        """Slice the problem's data and create the weight variable.

        Parameters
        ----------
        problem : Problem
            The problem to compile; its universe defines the order.
        w_ref : numpy.ndarray or None
            Linearisation point of a share-of-risk cap, in universe order; the
            share-of-risk row builder reproduces it as a tangent and refuses to
            build a row without it.
        """
        self.problem = problem
        self.w_ref = None if w_ref is None else np.asarray(w_ref, dtype=float).reshape(-1)
        self.ids = problem.universe.ids()
        self.n = len(self.ids)
        self.index = {asset_id: position for position, asset_id in enumerate(self.ids)}
        exposures, factor_covariance, specific_variance = problem.risk_model.arrays()
        row_of = {asset_id: row for row, asset_id in enumerate(problem.risk_model.asset_ids)}
        rows = np.fromiter((row_of[asset_id] for asset_id in self.ids), dtype=int, count=self.n)
        self.B = exposures[rows]
        self.F = factor_covariance
        self.d = specific_variance[rows]
        eigenvalues, eigenvectors = np.linalg.eigh(self.F)
        factor_root = eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))
        self.LtBt = factor_root.T @ self.B.T
        self.sqrt_d = np.sqrt(self.d)
        holdings = problem.initial_holdings
        self.w0 = np.fromiter(
            (holdings.get(asset_id, 0.0) for asset_id in self.ids), dtype=float, count=self.n
        )
        if problem.benchmark is None:
            self.b = None
            self.beta = None
        else:
            self.b = np.fromiter(
                (problem.benchmark.get(asset_id, 0.0) for asset_id in self.ids),
                dtype=float,
                count=self.n,
            )
            self.beta = beta_vector(problem)
        self.w = cp.Variable(self.n, name="w")

    def x(self, active: bool) -> cp.Expression:
        """Return the weights or the active weights.

        Parameters
        ----------
        active : bool
            If true, return w - benchmark; a problem with an active term is
            validated to have a benchmark.

        Returns
        -------
        cvxpy.Expression
            Affine expression of shape (n,).
        """
        if active:
            return self.w - self.b
        return self.w

    def risk_parts(self, x: cp.Expression, measure: str) -> cp.Expression:
        """Return the vector whose squared norm is the requested variance.

        The parts of a vector x are [L-transpose B-transpose x; sqrt(d) times
        x], so the squared 2-norm of the parts is x-transpose Sigma x. Asking
        for one measure returns only the matching block.

        Parameters
        ----------
        x : cvxpy.Expression
            Weights or active weights, shape (n,).
        measure : str
            "total" (both blocks), "factor" (the L-transpose B-transpose block
            only) or "specific" (the sqrt(d) block only).

        Returns
        -------
        cvxpy.Expression
            Expression whose squared 2-norm is the factor, the specific or the
            total variance of x.

        Raises
        ------
        ValueError
            If measure is not one of the three names.
        """
        if measure == "total":
            return cp.hstack([self.LtBt @ x, cp.multiply(self.sqrt_d, x)])
        if measure == "factor":
            return self.LtBt @ x
        if measure == "specific":
            return cp.multiply(self.sqrt_d, x)
        raise ValueError(f"unknown measure '{measure}'")

    def sigma_times(self, v: np.ndarray) -> np.ndarray:
        """Return the risk covariance applied to a vector, in factor form.

        The product is B (F (B-transpose v)) + d times v, never an n-by-n
        matrix, so the cost is linear in the number of assets and factors.

        Parameters
        ----------
        v : numpy.ndarray
            Vector in universe order, shape (n,).

        Returns
        -------
        numpy.ndarray
            Sigma v, shape (n,).
        """
        return self.B @ (self.F @ (self.B.T @ v)) + self.d * v

    def group_mask(self, attribute: str, label: str) -> np.ndarray:
        """Return the 0/1 mask of the assets carrying one group label.

        Assets that do not have the attribute get 0, matching the report, which
        collects them under UNASSIGNED.

        Parameters
        ----------
        attribute : str
            Attribute name, e.g. "sector".
        label : str
            Attribute value selecting the group.

        Returns
        -------
        numpy.ndarray
            Float64 mask in universe order, 1.0 on the group.
        """
        return np.fromiter(
            (
                1.0 if asset.attributes.get(attribute) == label else 0.0
                for asset in self.problem.universe.assets
            ),
            dtype=float,
            count=self.n,
        )
