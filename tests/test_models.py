import numpy as np
import pydantic
import pytest

from opoise import (Asset, AssetBounds, BetaBounds, Bounds, Budget, FactorExposureBounds,
                              FactorRiskModel, GroupBounds, GroupRiskCap, LinearBounds, Problem, RiskCap,
                              RiskTerm, Turnover, Universe)
from helpers import DATA, IDS


def rm(**over):
    base = dict(asset_ids=["A", "B"], factor_ids=["F1"], exposures=[[1.0], [0.5]],
                factor_covariance=[[0.04]], specific_variance=[0.01, 0.02])
    base.update(over)
    return FactorRiskModel(**base)


def test_risk_model_ok_and_from_arrays():
    m = rm()
    B, F, d = m.arrays()
    assert B.shape == (2, 1) and F.tolist() == [[0.04]] and d.tolist() == [0.01, 0.02]
    m2 = FactorRiskModel.from_arrays(["A", "B"], ["F1"], np.array([[1.0], [0.5]]),
                                     np.array([[0.04]]), np.array([0.01, 0.02]))
    assert m2 == m and isinstance(m2.exposures[0][0], float)


TWO = dict(factor_ids=["F1", "F2"], exposures=[[1.0, 0.0], [0.5, 0.0]])


@pytest.mark.parametrize("over,msg", [
    ({"exposures": [[1.0]]}, "exposures: expected 2x1"),
    ({"factor_covariance": [[0.04, 0.0]]}, "factor_covariance: expected 1x1"),
    ({"specific_variance": [0.01]}, "specific_variance: expected 2 values"),
    ({"specific_variance": [0.01, -0.02]}, "specific_variance must be >= 0"),
    ({**TWO, "factor_covariance": [[0.04, 0.01], [0.0, 0.04]]}, "factor_covariance is not symmetric"),
    ({**TWO, "factor_covariance": [[0.04, 0.05], [0.05, 0.04]]},
     "factor_covariance is not positive semidefinite"),
    ({"specific_variance": [0.01, float("nan")]}, "finite"),
    ({"asset_ids": ["A", "A"]}, "duplicate asset id"),
    ({"factor_ids": ["F1", "F1"], "exposures": [[1.0, 0.0], [0.5, 0.0]],
      "factor_covariance": [[0.04, 0.0], [0.0, 0.04]]}, "duplicate factor id"),
])
def test_risk_model_rejects(over, msg):
    with pytest.raises(pydantic.ValidationError, match=msg):
        rm(**over)


def test_universe_and_bounds_validation():
    with pytest.raises(pydantic.ValidationError, match="duplicate asset id"):
        Universe(assets=[Asset(id="X"), Asset(id="X")])
    with pytest.raises(pydantic.ValidationError, match="lower > upper"):
        Bounds(lower=1.0, upper=0.0)
    with pytest.raises(pydantic.ValidationError, match="at least one of lower/upper"):
        Bounds()
    with pytest.raises(pydantic.ValidationError, match="at least one of lower/upper"):
        BetaBounds(name="beta")
    with pytest.raises(pydantic.ValidationError, match="lower > upper"):
        AssetBounds(name="ab", lower=0.2, upper=0.1)
    with pytest.raises(pydantic.ValidationError, match="String should match pattern"):
        Budget(name="bad name")
    with pytest.raises(pydantic.ValidationError, match="Extra inputs are not permitted"):
        Budget(name="b", foo=1)
    with pytest.raises(pydantic.ValidationError):
        RiskTerm(aversion=0.0)
    with pytest.raises(pydantic.ValidationError):
        Turnover(name="t", limit=-0.1)
    with pytest.raises(pydantic.ValidationError):
        RiskCap(name="r", limit=0.0)
    assert Budget(name="b").type == "budget" and Budget(name="b").total == 1.0
    with pytest.raises(pydantic.ValidationError, match="Extra inputs are not permitted"):
        Budget(name="b", soft={"penalty": 2.0})  # soft constraints are not supported, so the extra field is rejected
    with pytest.raises(pydantic.ValidationError, match="String should match pattern"):
        Asset(id="A[1]")  # row labels are name[id], so ids exclude brackets
    with pytest.raises(pydantic.ValidationError, match="'UNASSIGNED' is reserved"):
        Asset(id="X", attributes={"sector": "UNASSIGNED"})


def base_kwargs():
    return dict(universe=DATA.universe, risk_model=DATA.risk_model, benchmark=DATA.benchmark,
                objective=[RiskTerm(aversion=1.0)])


@pytest.mark.parametrize("kwargs,msg", [
    ({"initial_holdings": {"ZZZ": 0.1}}, "initial_holdings: unknown asset 'ZZZ'"),
    ({"benchmark": {"ZZZ": 0.1}}, "benchmark: unknown asset 'ZZZ'"),
    ({"benchmark": {}}, "benchmark: predicted variance is zero"),
    ({"benchmark": {IDS[0]: 0.0}}, "benchmark: predicted variance is zero"),
    ({"constraints": [Budget(name="b"), Budget(name="b")]}, "duplicate constraint name 'b'"),
    ({"benchmark": None, "objective": [RiskTerm(active=True, aversion=1.0)]}, "requires a benchmark"),
    ({"benchmark": None, "constraints": [BetaBounds(name="beta", lower=0.9)]}, "requires a benchmark"),
    ({"benchmark": None, "constraints": [RiskCap(name="te", active=True, limit=0.05)]},
     "requires a benchmark"),
    ({"constraints": [GroupBounds(name="g", attribute="sector", bounds={"NOPE": Bounds(upper=0.1)})]},
     "unknown group 'NOPE'"),
    ({"constraints": [GroupRiskCap(name="g", attribute="sector", caps={"NOPE": 0.1})]},
     "unknown group 'NOPE'"),
    ({"constraints": [GroupBounds(name="g", attribute="country", bounds={"US": Bounds(upper=0.1)})]},
     "attribute 'country' is not set on any asset"),
    ({"constraints": [FactorExposureBounds(name="f", bounds={"NOPE": Bounds(upper=0.1)})]},
     "unknown factor 'NOPE'"),
    ({"constraints": [AssetBounds(name="a", per_asset={"ZZZ": Bounds(upper=0.1)})]}, "unknown asset 'ZZZ'"),
    ({"constraints": [LinearBounds(name="l", coefficients={"ZZZ": 1.0}, upper=0.1)]}, "unknown asset 'ZZZ'"),
    ({"objective": []}, "at least 1"),
])
def test_problem_rejects(kwargs, msg):
    kw = base_kwargs()
    kw.update(kwargs)
    with pytest.raises(pydantic.ValidationError, match=msg):
        Problem(**kw)


def test_risk_model_must_cover_universe_but_may_be_superset():
    uni = Universe(assets=list(DATA.universe.assets) + [Asset(id="EXTRA")])
    with pytest.raises(pydantic.ValidationError, match="risk_model: missing assets: EXTRA"):
        Problem(universe=uni, risk_model=DATA.risk_model, objective=[RiskTerm(aversion=1.0)])
    sub = Universe(assets=DATA.universe.assets[:5])
    p = Problem(universe=sub, risk_model=DATA.risk_model, objective=[RiskTerm(aversion=1.0)])
    assert [a.id for a in p.universe.assets] == IDS[:5] and p.initial_holdings == {} and p.benchmark is None


def test_constraint_union_parses_from_dicts():
    p = Problem(**{**base_kwargs(), "constraints": [
        {"type": "budget", "name": "budget"},
        {"type": "risk_cap", "name": "te", "active": True, "limit": 0.05},
        {"type": "group_bounds", "name": "s", "attribute": "sector", "bounds": {"S0": {"upper": 0.3}}}]})
    assert [type(c).__name__ for c in p.constraints] == ["Budget", "RiskCap", "GroupBounds"]
    assert p.settings.solver == "CLARABEL" and p.settings.binding_tol == 1e-5
