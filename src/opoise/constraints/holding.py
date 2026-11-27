"""The HOLDING category: per-asset and per-group weight limits."""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import Field, model_validator

from ..registry import Category, register_category
from ..rows import Row
from .base import CONSTRAINTS, Bounds, ConstraintBase

if TYPE_CHECKING:
    from ..context import CompileContext

CATEGORY = register_category(Category("HOLDING", "Per-asset and per-group weight limits.", "constraint"))
"""The category of this module."""


@CONSTRAINTS.register
class AssetBounds(ConstraintBase):
    """Per-asset lower and upper weight bounds, absolute or active.

    Immutability is shallow: fields cannot be reassigned, but the contents of
    dict fields are not frozen, so do not mutate them after construction.

    Attributes
    ----------
    name : str
        Constraint name.
    type : str
        Constant tag "asset_bounds".
    lower : float or None
        Default lower bound for assets without an override.
    upper : float or None
        Default upper bound for assets without an override.
    per_asset : dict of str to Bounds
        Overrides keyed by asset id; an entry replaces both default sides for
        that asset.
    active : bool
        If true, the bounds apply to w - benchmark instead of w.

    Raises
    ------
    pydantic.ValidationError
        If lower > upper, or if lower, upper and per_asset are all empty
        ("at least one of lower/upper/per_asset must be set").
    """

    category = CATEGORY
    type: Literal["asset_bounds"] = "asset_bounds"
    lower: float | None = None
    upper: float | None = None
    per_asset: dict[str, Bounds] = Field(default_factory=dict)
    active: bool = False

    @model_validator(mode="after")
    def _validate(self) -> AssetBounds:
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError(f"lower > upper ({self.lower} > {self.upper})")
        if self.lower is None and self.upper is None and not self.per_asset:
            raise ValueError("at least one of lower/upper/per_asset must be set")
        return self

    def build(self, ctx: CompileContext) -> Row:
        """Build the row of an AssetBounds constraint: per-asset weight limits.

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
            One vector row with an element per bounded asset.
        """
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
            expr=ctx.x(self.active)[selected],
            lower=lower[selected],
            upper=upper[selected],
        )


@CONSTRAINTS.register
class GroupBounds(ConstraintBase):
    """Lower and upper bounds on the summed weight of each group.

    Immutability is shallow: fields cannot be reassigned, but the contents of
    dict fields are not frozen, so do not mutate them after construction.

    Attributes
    ----------
    name : str
        Constraint name.
    type : str
        Constant tag "group_bounds".
    attribute : str
        Asset attribute that defines the groups, e.g. "sector".
    bounds : dict of str to Bounds
        Bound per group label; at least one entry.
    active : bool
        If true, the bounds apply to the active weights w - benchmark.
    """

    category = CATEGORY
    type: Literal["group_bounds"] = "group_bounds"
    attribute: str = Field(min_length=1)
    bounds: dict[str, Bounds] = Field(min_length=1)
    active: bool = False

    def build(self, ctx: CompileContext) -> Row:
        """Build the row of a GroupBounds constraint: summed weights per group.

        Each element is the sum of the weights (or active weights) of the
        assets carrying one attribute value, so the row is a matrix of 0/1
        group masks applied to the weight vector. Labels follow the order of
        the bounds dict.

        Parameters
        ----------
        ctx : CompileContext
            Compilation context of the problem.

        Returns
        -------
        Row
            One affine vector row with an element per bounded group.
        """
        labels = list(self.bounds)
        masks = np.vstack([ctx.group_mask(self.attribute, label) for label in labels])
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
            expr=masks @ ctx.x(self.active),
            lower=lower,
            upper=upper,
        )
