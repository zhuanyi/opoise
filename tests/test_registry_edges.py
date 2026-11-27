"""Edge cases around constraint dispatch, row checks and the SCA guard."""
from typing import Literal
from unittest.mock import patch

import cvxpy as cp
import numpy as np
import pytest
from pydantic import ValidationError

import opoise
from opoise import (
    Asset,
    Budget,
    ConstraintBase,
    FactorRiskModel,
    Problem,
    RiskTerm,
    SolverSettings,
    Universe,
    categories,
    compile_problem,
    constraint_types,
    register_constraint,
    solve,
    unregister_category,
    unregister_constraint,
)
from opoise.constraints import linear
from opoise.registry import Registry
from opoise.rows import Row

TIGHT = {"tol_gap_abs": 1e-9, "tol_gap_rel": 1e-9, "tol_feas": 1e-9}


@pytest.fixture
def registry():
    """Undo every type and category this test registers."""
    before = set(constraint_types())
    cats = {(c.kind, c.name) for c in categories()}
    yield
    for tag in set(constraint_types()) - before:
        unregister_constraint(tag, missing_ok=True)
    for kind, name in {(c.kind, c.name) for c in categories()} - cats:
        unregister_category(name, kind)


def tiny():
    universe = Universe(assets=[Asset(id="A"), Asset(id="B")])
    risk = FactorRiskModel(
        asset_ids=["A", "B"],
        factor_ids=["F"],
        exposures=[[0.0], [0.0]],
        factor_covariance=[[0.01]],
        specific_variance=[0.04, 0.01],
    )
    return Problem(
        universe=universe,
        risk_model=risk,
        objective=[RiskTerm(aversion=1.0)],
        constraints=[Budget(name="budget")],
        settings=SolverSettings(solver_options=TIGHT),
    )


def test_untagged_constraint_dict_rejected(registry):
    with pytest.raises(ValidationError, match="has no 'type' tag"):
        tiny().add_constraint({"name": "x"})
    with pytest.raises(ValidationError, match="bare ConstraintBase"):
        tiny().add_constraint(ConstraintBase(name="y"))
    doc = {**tiny().model_dump(), "constraints": [{"name": "x"}]}
    with pytest.raises(ValidationError, match="has no 'type' tag"):
        Problem.model_validate(doc)


def test_tuple_of_constraints_accepted(registry):
    doc = {**tiny().model_dump(), "constraints": ({"type": "budget", "name": "b"},)}
    problem = Problem.model_validate(doc)
    assert len(problem.constraints) == 1
    assert isinstance(problem.constraints[0], Budget)


def test_requires_sca_outside_share_cap_rejected(registry):
    @register_constraint
    class ScaLinear(ConstraintBase):
        category = linear.CATEGORY
        type: Literal["sca_linear"] = "sca_linear"
        requires_sca = True

        def build(self, ctx):
            return Row(
                constraint=self.name,
                labels=[self.name],
                expr=cp.hstack([cp.sum(ctx.w)]),
                lower=np.array([-np.inf]),
                upper=np.array([1.0]),
            )

    problem = tiny().add_constraint(ScaLinear(name="sca"))
    with pytest.raises(opoise.OpoiseError, match="only linearises group_risk_share_cap"):
        solve(problem)


def test_row_must_be_one_dimensional(registry):
    @register_constraint
    class ScalarRow(ConstraintBase):
        category = linear.CATEGORY
        type: Literal["scalar_row"] = "scalar_row"

        def build(self, ctx):
            return Row(
                constraint=self.name,
                labels=[self.name],
                expr=cp.sum(ctx.w),
                lower=np.array([-np.inf]),
                upper=np.array([1.0]),
            )

    with pytest.raises(opoise.OpoiseError, match=r"shape \(\)"):
        solve(tiny().add_constraint(ScalarRow(name="scalar")))

    @register_constraint
    class ColumnRow(ConstraintBase):
        category = linear.CATEGORY
        type: Literal["column_row"] = "column_row"

        def build(self, ctx):
            return Row(
                constraint=self.name,
                labels=[f"{self.name}[{i}]" for i in range(ctx.n)],
                expr=cp.reshape(ctx.w, (ctx.n, 1), order="C"),
                lower=np.full(ctx.n, -np.inf),
                upper=np.ones(ctx.n),
            )

    with pytest.raises(opoise.OpoiseError, match=r"shape \(2, 1\)"):
        solve(tiny().add_constraint(ColumnRow(name="column")))


def test_compiled_problem_must_be_dcp(registry):
    problem = tiny()
    with (
        patch.object(cp.Problem, "is_dcp", return_value=False),
        pytest.raises(opoise.OpoiseError, match="internal: compiled problem is not DCP"),
    ):
        compile_problem(problem)


def test_objective_registry_names_a_constraint_category(registry):
    reg = Registry(ConstraintBase, "objective")

    class WrongKind(ConstraintBase):
        category = linear.CATEGORY
        type: Literal["wrong_kind"] = "wrong_kind"

        def build(self, ctx):
            raise NotImplementedError

    with pytest.raises(opoise.OpoiseError, match="is a constraint category"):
        reg.register(WrongKind)
