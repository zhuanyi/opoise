"""Constraint registries, category discovery, TradeBounds and custom types."""
from typing import Literal
import numpy as np
import pytest
from pydantic import ValidationError
import opoise
from opoise import (Asset, Budget, Bounds, Category, ConstraintBase, FactorRiskModel, Problem, RiskTerm,
                    SolverSettings, TradeBounds, Universe, categories, constraint_types, load_workspace,
                    register_category, register_constraint, save_workspace, solve, unregister_category,
                    unregister_constraint)
from opoise.constraints import holding

TIGHT = {"tol_gap_abs": 1e-9, "tol_gap_rel": 1e-9, "tol_feas": 1e-9}
@pytest.fixture
def registry():
    before = set(constraint_types())
    cats = {(c.kind, c.name) for c in categories()}
    yield
    for tag in set(constraint_types()) - before:
        unregister_constraint(tag, missing_ok=True)
    for kind, name in {(c.kind, c.name) for c in categories()} - cats:
        unregister_category(name, kind)

def tiny(initial_holdings=None):
    universe = Universe(assets=[Asset(id="A"), Asset(id="B")])
    risk = FactorRiskModel(asset_ids=["A", "B"], factor_ids=["F"], exposures=[[0.0], [0.0]],
                           factor_covariance=[[0.01]], specific_variance=[0.04, 0.01])
    return Problem(universe=universe, risk_model=risk, initial_holdings=initial_holdings or {},
                   objective=[RiskTerm(aversion=1.0)], constraints=[Budget(name="budget")],
                   settings=SolverSettings(solver_options=TIGHT))

def floor_class(cat):
    class AFloor(ConstraintBase):
        category = cat
        type: Literal["a_floor"] = "a_floor"
        floor: float
        def build(self, ctx):
            import cvxpy as cp
            from opoise.rows import Row
            a = np.zeros(ctx.n)
            a[ctx.index["A"]] = 1.0
            return Row(constraint=self.name, labels=[self.name], expr=cp.hstack([a @ ctx.w]),
                       lower=np.array([float(self.floor)]), upper=np.array([np.inf]))
    return AFloor

def test_discovered_categories_and_types():
    assert {c.name for c in categories("constraint")} == {
        "BUDGET", "HOLDING", "TRADE", "TURNOVER", "RISK", "EXPOSURE", "LINEAR"}
    assert set(constraint_types()) == {
        "budget", "asset_bounds", "trade_bounds", "turnover", "group_bounds", "group_risk_cap",
        "group_risk_share_cap", "beta_bounds", "risk_cap", "factor_exposure_bounds", "linear_bounds"}
    assert set(constraint_types("HOLDING")) == {"asset_bounds", "group_bounds"}

def test_register_unregister_and_duplicate(registry):
    @register_constraint
    class Note(ConstraintBase):
        category = holding.CATEGORY
        type: Literal["note"] = "note"
        def build(self, ctx):
            raise NotImplementedError
    assert constraint_types()["note"] is Note
    unregister_constraint("note")
    assert "note" not in constraint_types()
    with pytest.raises(opoise.OpoiseError, match="unknown constraint type 'note'"):
        unregister_constraint("note")
    class First(ConstraintBase):
        category = holding.CATEGORY
        type: Literal["dup"] = "dup"
        def build(self, ctx):
            raise NotImplementedError
    class Second(ConstraintBase):
        category = holding.CATEGORY
        type: Literal["dup"] = "dup"
        def build(self, ctx):
            raise NotImplementedError
    register_constraint(First)
    with pytest.raises(opoise.OpoiseError, match="already registered"):
        register_constraint(Second)
    register_constraint(Second, replace=True)
    assert constraint_types()["dup"] is Second

def test_category_must_be_registered_and_hold_its_kind(registry):
    winter = Category("WINTER", "a category declared by the test", "constraint")
    with pytest.raises(opoise.OpoiseError, match="is not registered"):
        register_constraint(floor_class(winter))
    register_category(winter)
    register_constraint(floor_class(winter))
    with pytest.raises(opoise.OpoiseError, match="has registered types"):
        unregister_category("WINTER", "constraint")
    unregister_constraint("a_floor")
    unregister_category("WINTER", "constraint")
    wrong = Category("SIM_COST", "an objective-only category", "objective")
    register_category(wrong)
    with pytest.raises(opoise.OpoiseError, match="objective category"):
        register_constraint(floor_class(wrong))

def test_custom_constraint_solves_and_round_trips(registry, tmp_path):
    custom = register_category(Category("CUSTOM_HOLDING", "declared by the test", "constraint"))
    AFloor = register_constraint(floor_class(custom))
    problem = tiny().add_constraint(AFloor(name="a_floor", floor=0.25))
    result = solve(problem)
    assert result.status == "optimal", result.message
    assert result.weights["A"] == pytest.approx(0.25, abs=1e-6)
    assert result.weights["B"] == pytest.approx(0.75, abs=1e-6)
    assert result.objective_value == pytest.approx(0.008125, abs=1e-9)
    row = next(r for r in result.diagnostics.constraints if r.row == "a_floor")
    assert row.binding and row.value == pytest.approx(0.25, abs=1e-6)
    text = problem.model_dump_json()
    assert '"type":"a_floor"' in text and '"floor":0.25' in text
    loaded = Problem.model_validate_json(text)
    assert isinstance(loaded.constraints[-1], AFloor) and loaded == problem
    path = tmp_path / "problem.json"
    save_workspace(problem, path)
    assert load_workspace(path) == problem

def test_non_dcp_custom_constraint_names_itself(registry):
    @register_constraint
    class ConcaveCap(ConstraintBase):
        category = holding.CATEGORY
        type: Literal["concave_cap"] = "concave_cap"
        def build(self, ctx):
            import cvxpy as cp
            from opoise.rows import Row
            return Row(constraint=self.name, labels=[self.name],
                       expr=cp.hstack([-cp.sum_squares(ctx.w)]),
                       lower=np.array([-np.inf]), upper=np.array([0.0]))
    with pytest.raises(opoise.OpoiseError) as info:
        solve(tiny().add_constraint(ConcaveCap(name="concave")))
    text = str(info.value)
    assert "constraint 'concave'" in text and "concave_cap" in text and "non-DCP" in text

@pytest.mark.parametrize("limit, weight, cost, slack", [
    (0.1, 0.4, 0.0100, 0.0),
    (0.4, 0.2, 0.0080, 0.1)])
def test_trade_bounds_bind_and_slack(limit, weight, cost, slack):
    result = solve(tiny({"A": 0.5, "B": 0.5}).add_constraint(
        TradeBounds(name="trade", lower=-limit, upper=limit)))
    assert result.status == "optimal", result.message
    assert result.weights["A"] == pytest.approx(weight, abs=1e-6)
    assert result.weights["B"] == pytest.approx(1.0 - weight, abs=1e-6)
    assert result.objective_value == pytest.approx(cost, abs=1e-9)
    rows = [r for r in result.diagnostics.constraints if r.constraint == "trade"]
    assert [r.row for r in rows] == ["trade[A]", "trade[B]"]
    assert [r.value for r in rows] == pytest.approx([weight - 0.5, 0.5 - weight], abs=1e-6)
    assert all(abs(r.slack - slack) <= 1e-6 for r in rows)
    assert all(r.binding is (slack == 0.0) for r in rows)

def test_validation_and_unknown_type_messages():
    with pytest.raises(ValidationError, match="constraint 'trade': unknown asset 'Z'"):
        tiny().add_constraint(TradeBounds(name="trade", per_asset={"Z": Bounds(upper=0.1)}))
    with pytest.raises(ValidationError) as info:
        tiny().add_constraint({"type": "nope", "name": "x"})
    message = str(info.value)
    assert "unknown constraint type 'nope'" in message
    assert "budget" in message and "trade_bounds" in message
