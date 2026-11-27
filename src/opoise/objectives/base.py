"""Base types shared by every objective category.

Every objective term is a frozen pydantic model with a category and a constant
type tag; build() turns the model and a compilation context into a convex
cvxpy expression. The compiler minimises the sum of the terms, and the built-in
term is aversion times a variance, with no factor of one half.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel, ConfigDict

from ..registry import Category, Registry

if TYPE_CHECKING:
    import cvxpy as cp

    from ..context import CompileContext

_CONFIG = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ObjectiveBase(BaseModel):
    """Common base of every objective term: a category and build().

    Notes
    -----
    Every subclass sets category to a registered Category, declares its type
    tag in the class body and overrides build(). The compiler minimises the sum
    of the built expressions, so every term must build a convex expression.
    """

    model_config = _CONFIG

    category: ClassVar[Category]
    """The category the class registers under; set in every subclass."""

    def build(self, ctx: CompileContext) -> cp.Expression:
        """Build the term's expression for a compilation context.

        Parameters
        ----------
        ctx : CompileContext
            Compilation context of the problem.

        Returns
        -------
        cvxpy.Expression
            Convex expression the compiler adds to the objective.

        Raises
        ------
        NotImplementedError
            Always; every registered term overrides this method.
        """
        raise NotImplementedError


OBJECTIVES = Registry(ObjectiveBase, "objective")
"""The process-global registry of objective types."""
