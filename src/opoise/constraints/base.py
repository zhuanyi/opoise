"""Base types shared by every constraint category.

Every constraint is a frozen pydantic model with a name, a category and a
constant type tag; build() turns the model and a compilation context into one
row. Weights are fractions of NAV and the unit of every bound is the unit of
the quantity it bounds (a weight, a beta, a risk).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..registry import Category, Registry

if TYPE_CHECKING:
    from ..context import CompileContext
    from ..rows import Row

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
    """Common base of every constraint: a category, a name and build().

    Attributes
    ----------
    name : str
        Unique name within a problem; 1-64 characters from A-Z a-z 0-9 _ . : -.
        Row labels in the result are derived from it (name or name[key]).

    Notes
    -----
    Every subclass sets category to a registered Category, declares its type
    tag in the class body and overrides build(). requires_sca marks a
    constraint whose row needs a linearisation point, which sends the problem
    through the sequential approximation in opoise.sca; only
    GroupRiskShareCap is supported today.
    """

    model_config = _CONFIG

    category: ClassVar[Category]
    """The category the class registers under; set in every subclass."""

    requires_sca: ClassVar[bool] = False
    """True when the constraint is solved by sequential convex approximation.

    Only GroupRiskShareCap is supported: opoise.sca linearises that type alone
    and rejects any other type that sets this flag.
    """

    name: str = Field(pattern=NAME)

    def build(self, ctx: CompileContext) -> Row:
        """Build the constraint's row for a compilation context.

        Parameters
        ----------
        ctx : CompileContext
            Compilation context of the problem.

        Returns
        -------
        Row
            One row with the constraint's labels, expression and bounds.

        Raises
        ------
        NotImplementedError
            Always; every registered constraint overrides this method.
        """
        raise NotImplementedError


CONSTRAINTS = Registry(ConstraintBase, "constraint")
"""The process-global registry of constraint types."""
