"""Objective terms.

An objective is a non-empty list of terms; there is one kind, a weighted
variance (RiskTerm). Variance, not standard deviation, is the primitive: a
term contributes aversion times x-transpose Sigma x, with no factor of one
half.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

_CONFIG = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class RiskTerm(BaseModel):
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

    model_config = _CONFIG

    type: Literal["risk"] = "risk"
    measure: Literal["total", "factor", "specific"] = "total"
    active: bool = False
    aversion: float = Field(gt=0)


# shortcut: ObjectiveTerm is a plain alias while RiskTerm is the only term
# type; revisit when another term type exists.
ObjectiveTerm = RiskTerm
# One term of the objective; RiskTerm is the only term type.
