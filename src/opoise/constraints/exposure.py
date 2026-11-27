"""The EXPOSURE category: factor and benchmark exposures of the portfolio."""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import Field, model_validator

from ..registry import Category, register_category
from ..rows import Row
from .base import CONSTRAINTS, Bounds, ConstraintBase, _check_bounds

if TYPE_CHECKING:
    from ..context import CompileContext

CATEGORY = register_category(
    Category("EXPOSURE", "Factor and benchmark exposures of the portfolio.", "constraint")
)
"""The category of this module."""


@CONSTRAINTS.register
class BetaBounds(ConstraintBase):
    """Lower and upper bounds on the portfolio predicted beta.

    Attributes
    ----------
    name : str
        Constraint name.
    type : str
        Constant tag "beta_bounds".
    lower : float or None
        Lower bound on the predicted beta.
    upper : float or None
        Upper bound on the predicted beta.

    Raises
    ------
    pydantic.ValidationError
        If both limits are None or lower > upper.
    """

    category = CATEGORY
    type: Literal["beta_bounds"] = "beta_bounds"
    lower: float | None = None
    upper: float | None = None

    @model_validator(mode="after")
    def _validate(self) -> BetaBounds:
        _check_bounds(self.lower, self.upper)
        return self

    def build(self, ctx: CompileContext) -> Row:
        """Build the row of a BetaBounds constraint: the predicted beta.

        The predicted beta of every asset is data (a vector computed from the
        benchmark), so the portfolio beta is an affine function of the
        weights. The row always uses the absolute weights, never the active
        ones.

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

        expression = cp.hstack([ctx.beta @ ctx.w])
        lower = -np.inf if self.lower is None else float(self.lower)
        upper = np.inf if self.upper is None else float(self.upper)
        return Row(
            constraint=self.name,
            labels=[self.name],
            expr=expression,
            lower=np.array([lower]),
            upper=np.array([upper]),
        )


@CONSTRAINTS.register
class FactorExposureBounds(ConstraintBase):
    """Lower and upper bounds on factor exposures.

    Immutability is shallow: fields cannot be reassigned, but the contents of
    dict fields are not frozen, so do not mutate them after construction.

    Attributes
    ----------
    name : str
        Constraint name.
    type : str
        Constant tag "factor_exposure_bounds".
    bounds : dict of str to Bounds
        Bound per factor id; at least one entry.
    active : bool
        If true, exposures are measured on the active weights.
    """

    category = CATEGORY
    type: Literal["factor_exposure_bounds"] = "factor_exposure_bounds"
    bounds: dict[str, Bounds] = Field(min_length=1)
    active: bool = False

    def build(self, ctx: CompileContext) -> Row:
        """Build the row of a FactorExposureBounds constraint.

        Each element is one column of the exposure matrix applied to the
        weights (or active weights), so the row is affine. Labels follow the
        order of the bounds dict; the factor columns come from the risk model.

        Parameters
        ----------
        ctx : CompileContext
            Compilation context of the problem.

        Returns
        -------
        Row
            One affine vector row with an element per bounded factor.
        """
        column_of = {factor_id: position for position, factor_id in enumerate(ctx.problem.risk_model.factor_ids)}
        labels = list(self.bounds)
        columns = [column_of[label] for label in labels]
        lower = np.array(
            [-np.inf if self.bounds[label].lower is None else float(self.bounds[label].lower) for label in labels],
            dtype=float,
        )
        upper = np.array(
            [np.inf if self.bounds[label].upper is None else float(self.bounds[label].upper) for label in labels],
            dtype=float,
        )
        return Row(
            constraint=self.name,
            labels=[f"{self.name}[{label}]" for label in labels],
            expr=ctx.B[:, columns].T @ ctx.x(self.active),
            lower=lower,
            upper=upper,
        )
