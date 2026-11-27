"""Synthetic problem data.

The generator produces a complete, valid problem instance from a seed: a
universe with sector attributes, a factor risk model with a dominant market
factor and style factors, and a long-only benchmark that sums to one. It is
deterministic for a given seed and size, so tests and examples are stable.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .models import Asset, FactorRiskModel, Universe


class SyntheticData(NamedTuple):
    """A generated problem instance.

    Attributes
    ----------
    universe : Universe
        Assets A000, A001, ... with a sector attribute S0, S1, ... assigned by
        position modulo the sector count.
    risk_model : FactorRiskModel
        Market plus style factors, a valid covariance and positive specific
        variance for every asset.
    benchmark : dict of str to float
        Long-only benchmark weights summing to one.
    """

    universe: Universe
    risk_model: FactorRiskModel
    benchmark: dict[str, float]


def synthetic_data(n_assets: int = 20, n_factors: int = 3, n_sectors: int = 4, seed: int = 0) -> SyntheticData:
    """Generate a deterministic synthetic problem instance.

    Parameters
    ----------
    n_assets : int
        Number of assets, at least 2.
    n_factors : int
        Number of factors, at least 1; the first is a market factor.
    n_sectors : int
        Number of sector labels, from 1 to n_assets.
    seed : int
        Seed of the random stream.

    Returns
    -------
    SyntheticData
        Universe, risk model and benchmark.

    Raises
    ------
    ValueError
        If n_assets < 2, n_factors < 1 or n_sectors is outside 1..n_assets.
    """
    if n_assets < 2:
        raise ValueError(f"n_assets must be >= 2 (got {n_assets})")
    if n_factors < 1:
        raise ValueError(f"n_factors must be >= 1 (got {n_factors})")
    if not 1 <= n_sectors <= n_assets:
        raise ValueError(f"n_sectors must be between 1 and n_assets={n_assets} (got {n_sectors})")
    rng = np.random.default_rng(seed)
    n, k = n_assets, n_factors
    B = np.empty((n, k))
    B[:, 0] = rng.normal(1.0, 0.25, n)
    B[:, 1:] = rng.normal(0.0, 1.0, (n, k - 1))
    G = rng.normal(0.0, 1.0, (k, k))
    C = G @ G.T + k * np.eye(k)
    s = np.sqrt(np.diag(C))
    C = C / np.outer(s, s)
    vols = np.r_[0.16, np.full(k - 1, 0.04)]
    F = np.outer(vols, vols) * C
    F = (F + F.T) / 2
    d = rng.uniform(0.15, 0.35, n) ** 2
    m = rng.lognormal(0.0, 1.0, n)
    bench = m / m.sum()
    assets = [
        Asset(id=f"A{index:03d}", attributes={"sector": f"S{index % n_sectors}"})
        for index in range(n)
    ]
    factor_ids = ["MARKET"] + [f"STYLE{index}" for index in range(1, k)]
    risk_model = FactorRiskModel.from_arrays([asset.id for asset in assets], factor_ids, B, F, d)
    return SyntheticData(
        universe=Universe(assets=assets),
        risk_model=risk_model,
        benchmark={asset.id: float(weight) for asset, weight in zip(assets, bench)},
    )
