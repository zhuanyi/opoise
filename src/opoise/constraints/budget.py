"""The BUDGET category: the total exposure of the portfolio."""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from ..registry import Category, register_category
from ..rows import Row
from .base import CONSTRAINTS, ConstraintBase

if TYPE_CHECKING:
    from ..context import CompileContext

CATEGORY = register_category(Category("BUDGET", "Total exposure of the portfolio.", "constraint"))
"""The category of this module."""


@CONSTRAINTS.register
class Budget(ConstraintBase):
    """Fix the sum of the weights (1-vector transpose times w equals total).

    Attributes
    ----------
    name : str
        Constraint name.
    type : str
        Constant tag "budget".
    total : float
        Required sum of weights; 1.0 is fully invested, 0.0 is dollar neutral.
    """

    category = CATEGORY
    type: Literal["budget"] = "budget"
    total: float = 1.0

    def build(self, ctx: CompileContext) -> Row:
        """Build the row of a Budget constraint: the sum of the weights.

        Parameters
        ----------
        ctx : CompileContext
            Compilation context of the problem.

        Returns
        -------
        Row
            One scalar row fixing the sum of the weights to the requested
            total.
        """
        import cvxpy as cp

        expression = cp.hstack([cp.sum(ctx.w)])
        total = float(self.total)
        return Row(
            constraint=self.name,
            labels=[self.name],
            expr=expression,
            lower=np.array([total]),
            upper=np.array([total]),
        )
