"""Objective-term edge cases: build() must return a scalar convex cvxpy expression."""
from typing import Literal

import cvxpy as cp
import pytest

import opoise
from opoise import (Budget, ObjectiveBase, Problem, SolverSettings, objective_types, register_objective,
                    solve, unregister_objective)
from opoise.objectives import risk as objective_risk
from test_registry import TIGHT, tiny


@pytest.fixture
def objectives():
    before = set(objective_types()); yield
    for tag in set(objective_types()) - before: unregister_objective(tag, missing_ok=True)


def problem_with(term):
    base = tiny()
    return Problem(universe=base.universe, risk_model=base.risk_model, objective=[term],
                   constraints=[Budget(name="budget")], settings=SolverSettings(solver_options=TIGHT))


def test_concave_objective_term_rejected(objectives):
    @register_objective
    class ConcaveTerm(ObjectiveBase):
        category = objective_risk.CATEGORY
        type: Literal["concave_term"] = "concave_term"
        def build(self, ctx):
            return -cp.sum_squares(ctx.w)

    with pytest.raises(opoise.OpoiseError, match="non-convex"):
        solve(problem_with(ConcaveTerm()))


def test_vector_objective_term_rejected(objectives):
    @register_objective
    class VectorTerm(ObjectiveBase):
        category = objective_risk.CATEGORY
        type: Literal["vector_term"] = "vector_term"
        def build(self, ctx):
            return cp.hstack([ctx.w[0], ctx.w[0]])

    with pytest.raises(opoise.OpoiseError, match="scalar expression"):
        solve(problem_with(VectorTerm()))


def test_float_objective_term_rejected(objectives):
    @register_objective
    class FloatTerm(ObjectiveBase):
        category = objective_risk.CATEGORY
        type: Literal["float_term"] = "float_term"
        def build(self, ctx):
            return 1.0

    with pytest.raises(opoise.OpoiseError, match="cvxpy expression"):
        solve(problem_with(FloatTerm()))
