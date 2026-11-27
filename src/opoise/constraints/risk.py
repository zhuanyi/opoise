"""The RISK category: risk limits on the portfolio.

A risk cap is a norm of the risk parts and therefore a second-order cone. The
share-of-risk cap is convex minus convex; it is solved by the sequential
approximation in opoise.sca, which marks it with requires_sca.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, ClassVar, Literal

import numpy as np
from pydantic import Field

from ..errors import OpoiseError
from ..registry import Category, register_category
from ..rows import Row
from .base import CONSTRAINTS, ConstraintBase

if TYPE_CHECKING:
    from ..context import CompileContext

CATEGORY = register_category(Category("RISK", "Risk limits on the portfolio.", "constraint"))
"""The category of this module."""


@CONSTRAINTS.register
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

    category = CATEGORY
    type: Literal["risk_cap"] = "risk_cap"
    measure: Literal["total", "factor", "specific"] = "total"
    active: bool = False
    limit: float = Field(gt=0)

    def build(self, ctx: CompileContext) -> Row:
        """Build the row of a RiskCap constraint: a norm of the risk parts.

        The cap is a 2-norm of the risk parts of the weights or active weights,
        so it is a second-order cone and only carries an upper bound. With
        active set it is a tracking-error cap.

        Parameters
        ----------
        ctx : CompileContext
            Compilation context of the problem.

        Returns
        -------
        Row
            One convex scalar row bounded above by the limit.
        """
        import cvxpy as cp

        expression = cp.hstack([cp.norm(ctx.risk_parts(ctx.x(self.active), self.measure), 2)])
        return Row(
            constraint=self.name,
            labels=[self.name],
            expr=expression,
            lower=np.array([-np.inf]),
            upper=np.array([float(self.limit)]),
            convex=True,
        )


@CONSTRAINTS.register
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

    category = CATEGORY
    type: Literal["group_risk_cap"] = "group_risk_cap"
    attribute: str = Field(min_length=1)
    caps: dict[str, Annotated[float, Field(gt=0)]] = Field(min_length=1)
    measure: Literal["total", "factor", "specific"] = "total"
    active: bool = False

    def build(self, ctx: CompileContext) -> Row:
        """Build the row of a GroupRiskCap constraint: stand-alone group risk.

        Each element is the 2-norm of the risk parts of the group
        sub-portfolio, that is the group's own risk with every other asset set
        to zero. It is a second-order cone and only carries an upper bound.
        Labels follow the order of the caps dict.

        Parameters
        ----------
        ctx : CompileContext
            Compilation context of the problem.

        Returns
        -------
        Row
            One convex vector row bounded above by the caps.
        """
        import cvxpy as cp

        labels = list(self.caps)
        weights = ctx.x(self.active)
        expressions = [
            cp.norm(
                ctx.risk_parts(cp.multiply(ctx.group_mask(self.attribute, label), weights), self.measure),
                2,
            )
            for label in labels
        ]
        return Row(
            constraint=self.name,
            labels=[f"{self.name}[{label}]" for label in labels],
            expr=cp.hstack(expressions),
            lower=np.full(len(labels), -np.inf),
            upper=np.array([float(self.caps[label]) for label in labels]),
            convex=True,
        )


@CONSTRAINTS.register
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

    category = CATEGORY
    requires_sca: ClassVar[bool] = True
    # shortcut: no measure field; factor/specific shares are not requested, add one when they are
    type: Literal["group_risk_share_cap"] = "group_risk_share_cap"
    attribute: str = Field(min_length=1)
    caps: dict[str, Annotated[float, Field(gt=0)]] = Field(min_length=1)
    active: bool = False

    def build(self, ctx: CompileContext) -> Row:
        """Build the row of a GroupRiskShareCap: a tangent to the share cap.

        The true constraint, "the group's share of total risk is at most cap",
        is convex minus convex. Its concave part is replaced by the tangent at
        the linearisation point ctx.w_ref, which gives an inner approximation:
        every point the row admits satisfies the true cap, so each sequential
        subproblem stays DCP and its solution is feasible for the real problem.

        Parameters
        ----------
        ctx : CompileContext
            Compilation context, holding the linearisation point w_ref.

        Returns
        -------
        Row
            One convex vector row bounded above by zero, one element per cap.

        Raises
        ------
        OpoiseError
            If the context has no linearisation point ("internal:
            group_risk_share_cap needs a linearisation point; call solve()").
        """
        import cvxpy as cp

        if ctx.w_ref is None:
            raise OpoiseError("internal: group_risk_share_cap needs a linearisation point; call solve()")
        weights = ctx.x(self.active)
        reference = ctx.w_ref - ctx.b if self.active else ctx.w_ref
        labels = list(self.caps)
        expressions = []
        for label in labels:
            cap = float(self.caps[label])
            mask = ctx.group_mask(self.attribute, label)
            outside = 1.0 - mask
            complement = outside * reference
            sigma_complement = ctx.sigma_times(complement)
            sigma_reference = ctx.sigma_times(reference)
            constant = -0.25 * float(complement @ sigma_complement) - cap * float(reference @ sigma_reference)
            slope = -0.5 * outside * sigma_complement - 2.0 * cap * sigma_reference
            expressions.append(
                0.25 * cp.sum_squares(ctx.risk_parts(cp.multiply(1.0 + mask, weights), "total"))
                + constant
                + slope @ (weights - reference)
            )
        return Row(
            constraint=self.name,
            labels=[f"{self.name}[{label}]" for label in labels],
            expr=cp.hstack(expressions),
            lower=np.full(len(labels), -np.inf),
            upper=np.zeros(len(labels)),
            convex=True,
        )
