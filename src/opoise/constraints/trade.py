"""The TRADE category: limits on the trades against the initial holdings."""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import Field, model_validator

from ..registry import Category, register_category
from ..rows import Row
from .base import CONSTRAINTS, Bounds, ConstraintBase

if TYPE_CHECKING:
    from ..context import CompileContext

CATEGORY = register_category(
    Category("TRADE", "Limits on the trades against the initial holdings.", "constraint")
)
"""The category of this module."""


@CONSTRAINTS.register
class TradeBounds(ConstraintBase):
    """Per-asset lower and upper bounds on the trade against the holdings.

    A trade is the weight minus the initial holding, w - w0, so the bounds are
    element-wise on that difference and no benchmark is involved.

    Immutability is shallow: fields cannot be reassigned, but the contents of
    dict fields are not frozen, so do not mutate them after construction.

    Attributes
    ----------
    name : str
        Constraint name.
    type : str
        Constant tag "trade_bounds".
    lower : float or None
        Default lower bound for assets without an override.
    upper : float or None
        Default upper bound for assets without an override.
    per_asset : dict of str to Bounds
        Overrides keyed by asset id; an entry replaces both default sides for
        that asset.

    Raises
    ------
    pydantic.ValidationError
        If lower > upper, or if lower, upper and per_asset are all empty
        ("at least one of lower/upper/per_asset must be set").
    """

    category = CATEGORY
    type: Literal["trade_bounds"] = "trade_bounds"
    lower: float | None = None
    upper: float | None = None
    per_asset: dict[str, Bounds] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate(self) -> TradeBounds:
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError(f"lower > upper ({self.lower} > {self.upper})")
        if self.lower is None and self.upper is None and not self.per_asset:
            raise ValueError("at least one of lower/upper/per_asset must be set")
        return self

    def build(self, ctx: CompileContext) -> Row:
        """Build the row of a TradeBounds constraint: the trades of each asset.

        A per_asset entry replaces both default sides for that asset. Only
        assets with at least one finite bound after the overrides get a row
        element, in universe order.

        Parameters
        ----------
        ctx : CompileContext
            Compilation context of the problem.

        Returns
        -------
        Row
            One affine vector row with an element per bounded asset.
        """
        trade = ctx.w - ctx.w0
        lower = np.full(ctx.n, -np.inf)
        upper = np.full(ctx.n, np.inf)
        for asset_id in ctx.ids:
            override = self.per_asset.get(asset_id)
            low = override.lower if override is not None else self.lower
            high = override.upper if override is not None else self.upper
            position = ctx.index[asset_id]
            if low is not None:
                lower[position] = low
            if high is not None:
                upper[position] = high
        selected = np.flatnonzero(np.isfinite(lower) | np.isfinite(upper))
        return Row(
            constraint=self.name,
            labels=[f"{self.name}[{ctx.ids[position]}]" for position in selected],
            expr=trade[selected],
            lower=lower[selected],
            upper=upper[selected],
        )
