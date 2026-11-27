"""Constraint models, one module per category.

Each category lives in its own module in this folder. Importing the package
imports every module here in sorted order, and each module registers its
category and its classes, so a new built-in category is one file and nothing
else. A project registers its own category and classes from its own code; see
register_constraint.
"""
from __future__ import annotations

import importlib
import pkgutil

from pydantic import SerializeAsAny

from ..registry import Category, categories, register_category, unregister_category
from .base import CONSTRAINTS, NAME, Bounds, ConstraintBase
from .budget import Budget
from .exposure import BetaBounds, FactorExposureBounds
from .holding import AssetBounds, GroupBounds
from .linear import LinearBounds
from .risk import GroupRiskCap, GroupRiskShareCap, RiskCap
from .trade import TradeBounds
from .turnover import Turnover


def register_constraint(cls: type | None = None, *, replace: bool = False):
    """Register a constraint class under its type tag.

    Works as a decorator on the class or as a direct call, and returns the
    class either way.

    Parameters
    ----------
    cls : type or None
        The class to register; None returns a decorator.
    replace : bool
        Replace a class already registered under the same tag.

    Returns
    -------
    type or callable
        The class, or a decorator when cls is None.

    Raises
    ------
    OpoiseError
        If the class fails a registration check.
    """
    if cls is None:
        return lambda registered: CONSTRAINTS.register(registered, replace=replace)
    return CONSTRAINTS.register(cls, replace=replace)


def unregister_constraint(type_tag: str, *, missing_ok: bool = False) -> None:
    """Remove a constraint type by tag.

    Parameters
    ----------
    type_tag : str
        The type tag to remove.
    missing_ok : bool
        Return quietly instead of raising when the tag is unknown.

    Raises
    ------
    OpoiseError
        If the tag is unknown and missing_ok is False.
    """
    CONSTRAINTS.unregister(type_tag, missing_ok=missing_ok)


def constraint_types(category: str | Category | None = None) -> dict[str, type]:
    """Return the registered constraint types, keyed by tag and sorted by tag.

    Parameters
    ----------
    category : str or Category or None
        Keep only the types of this category; None keeps all.

    Returns
    -------
    dict of str to type
        A new dict in sorted order.
    """
    return CONSTRAINTS.types(category)


Constraint = SerializeAsAny[ConstraintBase]
"""Any registered constraint, validated as the base class.

The class is chosen from the type tag before validation, so a validated
problem can be built from any registered type, including one declared in a
project. SerializeAsAny keeps the subclass fields in a dump, which the
workspace format and model_dump_json rely on.
"""


def _discover() -> None:
    """Import every module of the package in sorted order, except base and _ ones."""
    for info in sorted(pkgutil.iter_modules(__path__), key=lambda module: module.name):
        if info.name == "base" or info.name.startswith("_"):
            continue
        importlib.import_module(f"{__name__}.{info.name}")


_discover()

__all__ = [
    "CONSTRAINTS",
    "NAME",
    "AssetBounds",
    "BetaBounds",
    "Bounds",
    "Budget",
    "Category",
    "Constraint",
    "ConstraintBase",
    "FactorExposureBounds",
    "GroupBounds",
    "GroupRiskCap",
    "GroupRiskShareCap",
    "LinearBounds",
    "RiskCap",
    "TradeBounds",
    "Turnover",
    "categories",
    "constraint_types",
    "register_category",
    "register_constraint",
    "unregister_category",
    "unregister_constraint",
]
