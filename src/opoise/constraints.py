"""Declarative constraint models.

Every constraint is a frozen pydantic model with a name and a constant type
tag; a Constraint field accepts any of them and picks the class from type.
Weights are fractions of NAV and the unit of every bound is the unit of the
quantity it bounds (a weight, a beta, a risk).
"""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

NAME = r"^[A-Za-z0-9_.:-]{1,64}$"
"""Pattern every constraint name must match: 1-64 of A-Z a-z 0-9 _ . : -."""

_CONFIG = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


def _check_bounds(lower: float | None, upper: float | None) -> None:
    if lower is None and upper is None:
        raise ValueError("at least one of lower/upper must be set")
    if lower is not None and upper is not None and lower > upper:
        raise ValueError(f"lower > upper ({lower} > {upper})")


class Bounds(BaseModel):
    """A lower limit, an upper limit, or both.

    Attributes
    ----------
    lower : float or None
        Lower limit; None means unbounded below.
    upper : float or None
        Upper limit; None means unbounded above.

    Raises
    ------
    pydantic.ValidationError
        If both limits are None ("at least one of lower/upper must be set")
        or if lower > upper ("lower > upper (...)").
    """

    model_config = _CONFIG

    lower: float | None = None
    upper: float | None = None

    @model_validator(mode="after")
    def _validate(self) -> Bounds:
        _check_bounds(self.lower, self.upper)
        return self


class ConstraintBase(BaseModel):
    """Common base of every constraint: a validated name.

    Attributes
    ----------
    name : str
        Unique name within a problem; 1-64 characters from A-Z a-z 0-9 _ . : -.
        Row labels in the result are derived from it (name or name[key]).
    """

    model_config = _CONFIG

    name: str = Field(pattern=NAME)


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

    type: Literal["budget"] = "budget"
    total: float = 1.0


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

    type: Literal["asset_bounds"] = "asset_bounds"
    lower: float | None = None
    upper: float | None = None
    per_asset: dict[str, Bounds] = {}
    active: bool = False

    @model_validator(mode="after")
    def _validate(self) -> AssetBounds:
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError(f"lower > upper ({self.lower} > {self.upper})")
        if self.lower is None and self.upper is None and not self.per_asset:
            raise ValueError("at least one of lower/upper/per_asset must be set")
        return self


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

    type: Literal["turnover"] = "turnover"
    limit: float = Field(ge=0)


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

    type: Literal["group_bounds"] = "group_bounds"
    attribute: str = Field(min_length=1)
    bounds: dict[str, Bounds] = Field(min_length=1)
    active: bool = False


class GroupRiskCap(ConstraintBase):
    """Cap the stand-alone risk of each group sub-portfolio (variant A).

    The cap is the 2-norm of the group's risk parts, at most caps[g]: a
    second-order cone on the group's own risk. It ignores the group's
    covariance with the rest of the portfolio; a cap on the group's
    contribution to total risk is the share-of-risk variant.

    Immutability is shallow: fields cannot be reassigned, but the contents of
    dict fields are not frozen, so do not mutate them after construction.

    Attributes
    ----------
    name : str
        Constraint name.
    type : str
        Constant tag "group_risk_cap".
    attribute : str
        Asset attribute that defines the groups.
    caps : dict of str to float
        Risk cap per group label, strictly positive, in risk (not variance) units.
    measure : str
        "total", "factor" or "specific" risk of the group sub-portfolio.
    active : bool
        If true, the caps apply to the active weights.
    """

    type: Literal["group_risk_cap"] = "group_risk_cap"
    attribute: str = Field(min_length=1)
    caps: dict[str, Annotated[float, Field(gt=0)]] = Field(min_length=1)
    measure: Literal["total", "factor", "specific"] = "total"
    active: bool = False


class GroupRiskShareCap(ConstraintBase):
    """Cap each group's share of total (or active) risk (variant B).

    The cap is on the Euler contribution of the group divided by the total
    risk, which is not a convex constraint: it is solved by sequential convex
    approximation (see opoise.sca), so the result is a local optimum and the
    report says whether the iteration converged. Shares are not bounded to
    [0, 1], so a cap must only be positive; a cap on every label whose caps sum
    to less than 1 can only be met at a zero portfolio. A group's share is
    undefined when the portfolio has zero risk; it is then reported as 0.

    Immutability is shallow: fields cannot be reassigned, but the contents of
    dict fields are not frozen, so do not mutate them after construction.

    Attributes
    ----------
    name : str
        Constraint name.
    type : str
        Constant tag "group_risk_share_cap".
    attribute : str
        Asset attribute that defines the groups.
    caps : dict of str to float
        Share cap per group label, strictly positive, as a fraction of the
        total (or active) risk.
    active : bool
        If true, the shares are those of the active weights, against the
        benchmark.

    Raises
    ------
    pydantic.ValidationError
        If caps is empty, a cap is not positive, or the attribute is set on no
        asset (checked when the constraint is put in a problem).
    """

    # shortcut: no measure field; factor/specific shares are not requested, add one when they are
    type: Literal["group_risk_share_cap"] = "group_risk_share_cap"
    attribute: str = Field(min_length=1)
    caps: dict[str, Annotated[float, Field(gt=0)]] = Field(min_length=1)
    active: bool = False


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

    type: Literal["beta_bounds"] = "beta_bounds"
    lower: float | None = None
    upper: float | None = None

    @model_validator(mode="after")
    def _validate(self) -> BetaBounds:
        _check_bounds(self.lower, self.upper)
        return self


class RiskCap(ConstraintBase):
    """Cap total, factor or specific risk (a tracking-error cap when active).

    Attributes
    ----------
    name : str
        Constraint name.
    type : str
        Constant tag "risk_cap".
    measure : str
        "total", "factor" or "specific" risk of the weights or active weights.
    active : bool
        If true, risk is measured on w - benchmark (tracking error).
    limit : float
        Upper bound on that risk, strictly positive, in risk units.
    """

    type: Literal["risk_cap"] = "risk_cap"
    measure: Literal["total", "factor", "specific"] = "total"
    active: bool = False
    limit: float = Field(gt=0)


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

    type: Literal["factor_exposure_bounds"] = "factor_exposure_bounds"
    bounds: dict[str, Bounds] = Field(min_length=1)
    active: bool = False


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

    type: Literal["linear_bounds"] = "linear_bounds"
    coefficients: dict[str, float] = Field(min_length=1)
    lower: float | None = None
    upper: float | None = None

    @model_validator(mode="after")
    def _validate(self) -> LinearBounds:
        _check_bounds(self.lower, self.upper)
        return self


Constraint = Annotated[
    (
        Budget
        | AssetBounds
        | Turnover
        | GroupBounds
        | GroupRiskCap
        | GroupRiskShareCap
        | BetaBounds
        | RiskCap
        | FactorExposureBounds
        | LinearBounds
    ),
    Field(discriminator="type"),
]
"""Any constraint; the class is selected by its type tag."""
