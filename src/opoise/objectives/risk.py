"""The RISK category: variance terms of the objective.

A term is aversion times the variance of the weights or active weights,
written as a squared norm of the risk parts. Variance, not standard deviation,
is the primitive, there is no factor of one half, and every term is convex.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import Field

from ..registry import Category, register_category
from .base import OBJECTIVES, ObjectiveBase

if TYPE_CHECKING:
    import cvxpy as cp

    from ..context import CompileContext

CATEGORY = register_category(Category("RISK", "Variance terms of the portfolio risk.", "objective"))
"""The category of this module."""


@OBJECTIVES.register
class RiskTerm(ObjectiveBase):
    """A weighted variance term in the objective.

    Attributes
    ----------
    type : str
        Constant tag "risk".
    measure : str
        "total", "factor" or "specific" variance of the weights or active
        weights.
    active : bool
        If true, the variance is measured on w - benchmark (active risk), which
        requires a benchmark on the problem.
    aversion : float
        Positive multiplier on the variance: the term is aversion times the
        variance.
    """

    category = CATEGORY
    type: Literal["risk"] = "risk"
    measure: Literal["total", "factor", "specific"] = "total"
    active: bool = False
    aversion: float = Field(gt=0)

    def build(self, ctx: CompileContext) -> cp.Expression:
        """Build the term: aversion times the squared norm of the risk parts.

        The measure picks the factor block, the specific block or both, so
        total variance is the squared norm of the stacked blocks. The term is
        convex, minimised as aversion times the variance with no factor of one
        half.

        Parameters
        ----------
        ctx : CompileContext
            Compilation context of the problem.

        Returns
        -------
        cvxpy.Expression
            Convex scalar expression.
        """
        import cvxpy as cp

        return self.aversion * cp.sum_squares(ctx.risk_parts(ctx.x(self.active), self.measure))
