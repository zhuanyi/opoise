"""The logical row the compiler emits, before it becomes cvxpy constraints.

A Row is one logical constraint (a scalar or one vector) with its expression,
its bounds and its report labels. It carries no cvxpy import at module level,
so importing it never loads the solver stack.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


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
    expr: Any
    lower: np.ndarray
    upper: np.ndarray
    convex: bool = False
