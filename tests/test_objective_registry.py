"""Objective registry: categories, dispatch and a custom term solved end to end."""
from typing import Literal
import pytest
from pydantic import ValidationError
from opoise import (Budget, ObjectiveBase, Problem, SolverSettings, categories, objective_types,
                    register_objective, solve, unregister_objective)
from opoise.objectives import risk as objective_risk
from test_registry import TIGHT, tiny

@pytest.fixture
def objectives():
    before = set(objective_types()); yield
    for tag in set(objective_types()) - before: unregister_objective(tag, missing_ok=True)


def test_unknown_objective_type_lists_registered_tags():
    data = {"universe": {"assets": [{"id": "A"}]},
            "risk_model": {"asset_ids": ["A"], "factor_ids": ["F"], "exposures": [[0.0]],
                           "factor_covariance": [[0.01]], "specific_variance": [0.04]},
            "objective": [{"type": "nope"}]}
    with pytest.raises(ValidationError) as info:
        Problem.model_validate(data)
    assert "unknown objective type 'nope'" in str(info.value) and "risk" in str(info.value)
    assert {c.name for c in categories("objective")} == {"RISK"} and set(objective_types()) == {"risk"}

def test_custom_objective_solves_end_to_end(objectives):
    @register_objective
    class HalfRisk(ObjectiveBase):
        category = objective_risk.CATEGORY
        type: Literal["half_risk"] = "half_risk"
        def build(self, ctx):
            import cvxpy as cp
            return 0.5 * cp.sum_squares(ctx.risk_parts(ctx.w, "total"))
    base = tiny()
    problem = Problem(universe=base.universe, risk_model=base.risk_model, objective=[HalfRisk()],
                      constraints=[Budget(name="budget")], settings=SolverSettings(solver_options=TIGHT))
    result = solve(problem)
    assert result.status == "optimal", result.message
    assert result.weights["A"] == pytest.approx(0.2, abs=1e-6)
    assert result.objective_value == pytest.approx(0.004, abs=1e-9)
