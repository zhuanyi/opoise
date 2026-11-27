"""The TURNOVER category: trading against the initial holdings."""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import Field

from ..registry import Category, register_category
from ..rows import Row
from .base import CONSTRAINTS, ConstraintBase

if TYPE_CHECKING:
    from ..context import CompileContext

CATEGORY = register_category(
    Category("TURNOVER", "Trading against the initial holdings.", "constraint")
)
"""The category of this module."""


@CONSTRAINTS.register
class Turnover(ConstraintBase):
    """Cap the two-way turnover of the weights against the initial holdings.

    Attributes
    ----------
    name : str
        Constraint name.
    type : str
        Constant tag "turnover".
    limit : float
        Maximum two-way turnover (one-way turnover is half of it); at least 0.
    """

    category = CATEGORY
    type: Literal["turnover"] = "turnover"
    limit: float = Field(ge=0)

    def build(self, ctx: CompileContext) -> Row:
        """Build the row of a Turnover constraint: turnover against holdings.

        Turnover is the sum of absolute trades against the initial holdings,
        so the row is an L1 norm of the trade vector and only carries an upper
        bound. One-way turnover is half of the reported number.

        Parameters
        ----------
        ctx : CompileContext
            Compilation context of the problem.

        Returns
        -------
        Row
            One convex scalar row bounded above by the limit.
        """
        import cvxpy as cp

        expression = cp.hstack([cp.norm1(ctx.w - ctx.w0)])
        return Row(
            constraint=self.name,
            labels=[self.name],
            expr=expression,
            lower=np.array([-np.inf]),
            upper=np.array([float(self.limit)]),
            convex=True,
        )
