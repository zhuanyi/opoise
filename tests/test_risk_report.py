import numpy as np
import pytest

from opoise import Asset, Problem, RiskTerm, Universe, predicted_beta, risk_report
from helpers import B, BENCH, BETA, DATA, F, IDS, SIGMA, W_MV, as_dict, problem

P = problem([RiskTerm(aversion=1.0)])
RNG_W = np.random.default_rng(123).normal(0.0, 0.1, len(IDS))


@pytest.mark.parametrize("w", [W_MV, RNG_W], ids=["w_mv", "random"])
@pytest.mark.parametrize("active", [False, True])
def test_decomposition_sums_to_total(w, active):
    x = w - BENCH if active else w
    rep = risk_report(P, as_dict(w), active=active)
    total = float(np.sqrt(x @ SIGMA @ x))
    y = B.T @ x
    assert rep.total_risk == pytest.approx(total, rel=1e-9)
    assert rep.factor_risk == pytest.approx(float(np.sqrt(y @ F @ y)), rel=1e-9)
    assert rep.total_risk ** 2 == pytest.approx(rep.factor_risk ** 2 + rep.specific_risk ** 2, rel=1e-9)
    assert list(rep.by_asset) == IDS
    assert sum(rep.by_asset.values()) == pytest.approx(total, rel=1e-9)
    assert list(rep.by_factor) == DATA.risk_model.factor_ids
    assert sum(rep.by_factor.values()) + rep.specific_contribution == pytest.approx(total, rel=1e-9)
    assert list(rep.by_group) == ["sector"] and set(rep.by_group["sector"]) == {"S0", "S1", "S2", "S3"}
    assert sum(rep.by_group["sector"].values()) == pytest.approx(total, rel=1e-9)
    np.testing.assert_allclose([rep.factor_exposures[f] for f in DATA.risk_model.factor_ids], y, atol=1e-12)
    i = 3
    assert rep.by_asset[IDS[i]] == pytest.approx(x[i] * (SIGMA @ x)[i] / total, rel=1e-9)


def test_zero_and_single_asset_portfolios():
    rep = risk_report(P, DATA.benchmark, active=True)
    assert rep.total_risk == 0.0 and all(v == 0.0 for v in rep.by_asset.values())
    assert rep.specific_contribution == 0.0 and all(v == 0.0 for v in rep.by_factor.values())
    one = risk_report(P, {IDS[0]: 1.0})
    assert one.total_risk == pytest.approx(float(np.sqrt(SIGMA[0, 0])), rel=1e-12)
    assert one.by_asset[IDS[0]] == pytest.approx(one.total_risk, rel=1e-12)
    assert all(v == 0.0 for k, v in one.by_asset.items() if k != IDS[0])


def test_unassigned_group_and_errors():
    assets = [Asset(id=a.id, attributes={} if k == 0 else a.attributes) for k, a in enumerate(DATA.universe.assets)]
    p = Problem(universe=Universe(assets=assets), risk_model=DATA.risk_model, objective=[RiskTerm(aversion=1.0)])
    rep = risk_report(p, as_dict(W_MV))
    assert "UNASSIGNED" in rep.by_group["sector"]
    assert rep.by_group["sector"]["UNASSIGNED"] == pytest.approx(rep.by_asset[IDS[0]], rel=1e-12)
    with pytest.raises(ValueError, match="unknown asset 'ZZZ'"):
        risk_report(P, {"ZZZ": 1.0})
    with pytest.raises(ValueError, match="requires a benchmark"):
        risk_report(p, as_dict(W_MV), active=True)
    with pytest.raises(ValueError, match="requires a benchmark"):
        predicted_beta(p, as_dict(W_MV))


def test_predicted_beta():
    assert predicted_beta(P, DATA.benchmark) == pytest.approx(1.0, rel=1e-12)
    assert predicted_beta(P, as_dict(W_MV)) == pytest.approx(float(BETA @ W_MV), rel=1e-12)
