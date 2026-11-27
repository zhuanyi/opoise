"""The LINEAR category: a bounded linear combination of the weights."""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import Field, model_validator

from ..registry import Category, register_category
from ..rows import Row
from .base import CONSTRAINTS, ConstraintBase, _check_bounds

if TYPE_CHECKING:
    from ..context import CompileContext

CATEGORY = register_category(
    Category("LINEAR", "A linear combination of the weights.", "constraint")
)
"""The category of this module."""


@CONSTRAINTS.register
class LinearBounds(ConstraintBase):
    """Lower and upper bounds on a linear combination of weights.

    Immutability is shallow: fields cannot be reassigned, but the contents of
    dict fields are not frozen, so do not mutate them after construction.

    Attributes
    ----------
    name : str
        Constraint name.
    type : str
        Constant tag "linear_bounds".
    coefficients : dict of str to float
        Coefficient per asset id; at least one entry, missing assets are 0.
    lower : float or None
        Lower bound on the combination.
    upper : float or None
        Upper bound on the combination.

    Raises
    ------
    pydantic.ValidationError
        If both limits are None or lower > upper.
    """

    category = CATEGORY
    type: Literal["linear_bounds"] = "linear_bounds"
    coefficients: dict[str, float] = Field(min_length=1)
    lower: float | None = None
    upper: float | None = None

    @model_validator(mode="after")
    def _validate(self) -> LinearBounds:
        _check_bounds(self.lower, self.upper)
        return self

    def build(self, ctx: CompileContext) -> Row:
        """Build the row of a LinearBounds constraint: a linear combination.

        The coefficients are placed in universe order (assets without a
        coefficient are zero), so the row is one affine scalar on the weights.
        The row always uses the absolute weights, never the active ones.

        Parameters
        ----------
        ctx : CompileContext
            Compilation context of the problem.

        Returns
        -------
        Row
            One affine scalar row with the requested bounds.
        """
        import cvxpy as cp

        coefficients = np.zeros(ctx.n)
        for asset_id, value in self.coefficients.items():
            coefficients[ctx.index[asset_id]] = float(value)
        expression = cp.hstack([coefficients @ ctx.w])
        lower = -np.inf if self.lower is None else float(self.lower)
        upper = np.inf if self.upper is None else float(self.upper)
        return Row(
            constraint=self.name,
            labels=[self.name],
            expr=expression,
            lower=np.array([lower]),
            upper=np.array([upper]),
        )
