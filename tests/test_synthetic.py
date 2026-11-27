import numpy as np
import pytest

from opoise import synthetic_data


def test_deterministic_shapes_and_validity():
    a = synthetic_data(n_assets=30, n_factors=4, n_sectors=3, seed=7)
    b = synthetic_data(n_assets=30, n_factors=4, n_sectors=3, seed=7)
    assert a.risk_model == b.risk_model and a.benchmark == b.benchmark and a.universe == b.universe
    B, F, d = a.risk_model.arrays()
    assert B.shape == (30, 4) and F.shape == (4, 4) and d.shape == (30,)
    assert a.risk_model.factor_ids == ["MARKET", "STYLE1", "STYLE2", "STYLE3"]
    assert a.risk_model.asset_ids == [f"A{i:03d}" for i in range(30)]
    assert [x.id for x in a.universe.assets] == a.risk_model.asset_ids
    assert np.array_equal(F, F.T) and np.linalg.eigvalsh(F).min() > 0 and (d > 0).all()
    assert abs(sum(a.benchmark.values()) - 1.0) < 1e-12 and min(a.benchmark.values()) > 0
    assert [x.attributes["sector"] for x in a.universe.assets][:4] == ["S0", "S1", "S2", "S0"]
    assert synthetic_data(n_assets=30, n_factors=4, n_sectors=3, seed=8).risk_model != a.risk_model


@pytest.mark.parametrize("kw", [{"n_assets": 1}, {"n_factors": 0}, {"n_sectors": 0},
                                {"n_assets": 3, "n_sectors": 4}])
def test_bad_sizes(kw):
    with pytest.raises(ValueError):
        synthetic_data(**kw)
