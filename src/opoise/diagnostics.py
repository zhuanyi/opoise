"""Risk analytics computed in numpy from a problem and a weight vector.

These functions need no solver and no cvxpy: they evaluate the factor risk
model directly, so a portfolio can be analysed whether or not it came from a
solve. Every computation stays in factor form (never forming the n by n
covariance matrix), so cost is linear in the number of assets for a fixed
number of factors.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .models import UNASSIGNED, Problem
from .result import RiskReport

__all__ = ["UNASSIGNED", "beta_vector", "predicted_beta", "risk_report"]


def _universe_arrays(problem: Problem) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the risk model arrays on the universe rows, in universe order."""
    exposures, factor_covariance, specific_variance = problem.risk_model.arrays()
    row_of = {asset_id: row for row, asset_id in enumerate(problem.risk_model.asset_ids)}
    rows = np.fromiter(
        (row_of[asset_id] for asset_id in problem.universe.ids()), dtype=int, count=len(problem.universe.assets)
    )
    return exposures[rows], factor_covariance, specific_variance[rows]


def _weight_vector(problem: Problem, weights: Mapping[str, float]) -> np.ndarray:
    """Turn a weight mapping into a float vector in universe order (missing = 0)."""
    ids = problem.universe.ids()
    id_set = set(ids)
    for asset_id in weights:
        if asset_id not in id_set:
            raise ValueError(f"weights: unknown asset '{asset_id}'")
    return np.fromiter((weights.get(asset_id, 0.0) for asset_id in ids), dtype=float, count=len(ids))


def _benchmark_vector(problem: Problem) -> np.ndarray:
    """Return the benchmark weights in universe order (missing = 0)."""
    benchmark = problem.benchmark or {}
    return np.fromiter(
        (benchmark.get(asset_id, 0.0) for asset_id in problem.universe.ids()),
        dtype=float,
        count=len(problem.universe.assets),
    )


def risk_report(
    problem: Problem, weights: Mapping[str, float], active: bool = False
) -> RiskReport:
    """Decompose the risk of a portfolio into Euler contributions.

    The total risk is the standard deviation of the portfolio return, or of
    the active return when active is True. Asset contributions
    x_i (Sigma x)_i / sigma, factor contributions y_f (F y)_f / sigma (with
    y = B-transpose x) and the specific contribution (sum of d_i x_i^2) /
    sigma each sum to the total risk; group contributions sum their assets,
    with assets lacking the attribute under UNASSIGNED. All of it is computed
    in factor form, so the n by n covariance matrix is never formed.

    Parameters
    ----------
    problem : Problem
        Supplies the universe order, the risk model and the benchmark.
    weights : mapping of str to float
        Portfolio weights as fractions of NAV, keyed by asset id; assets
        missing from the mapping have weight zero.
    active : bool
        Measure the active weights w - b against the benchmark instead of w.
        Requires the problem to have a benchmark.

    Returns
    -------
    RiskReport
        Total, factor and specific risk, exposures, Euler contributions by
        factor, by asset and by group, and the specific contribution.

    Raises
    ------
    ValueError
        If a key of weights is not an asset of the universe ("weights: unknown
        asset '<id>'"), or if active is True and the problem has no benchmark
        ("active risk report requires a benchmark").

    Examples
    --------
    >>> from opoise import Problem, RiskTerm, risk_report, synthetic_data
    >>> data = synthetic_data()
    >>> problem = Problem(universe=data.universe, risk_model=data.risk_model,
    ...                   objective=[RiskTerm(aversion=1.0)])
    >>> report = risk_report(problem, data.benchmark)
    >>> round(sum(report.by_asset.values()) - report.total_risk, 12) == 0.0
    True
    """
    w = _weight_vector(problem, weights)
    if active:
        if problem.benchmark is None:
            raise ValueError("active risk report requires a benchmark")
        x = w - _benchmark_vector(problem)
    else:
        x = w
    exposures, factor_covariance, specific_variance = _universe_arrays(problem)
    factor_ids = problem.risk_model.factor_ids
    ids = problem.universe.ids()
    y = exposures.T @ x
    factor_loading = factor_covariance @ y
    factor_variance = float(y @ factor_loading)
    specific_variance_total = float(np.sum(specific_variance * x * x))
    variance = factor_variance + specific_variance_total
    total_risk = float(np.sqrt(variance)) if variance > 0.0 else 0.0
    if total_risk > 0.0:
        contributions = x * (exposures @ factor_loading + specific_variance * x) / total_risk
        by_factor_values = y * factor_loading / total_risk
        specific_contribution = specific_variance_total / total_risk
    else:
        contributions = np.zeros_like(x)
        by_factor_values = np.zeros_like(y)
        specific_contribution = 0.0
    by_asset = {asset_id: float(value) for asset_id, value in zip(ids, contributions)}
    by_factor = {factor_id: float(value) for factor_id, value in zip(factor_ids, by_factor_values)}
    by_group: dict[str, dict[str, float]] = {}
    attribute_names: list[str] = []
    for asset in problem.universe.assets:
        for attribute in asset.attributes:
            if attribute not in attribute_names:
                attribute_names.append(attribute)
    for attribute in attribute_names:
        labels: dict[str, float] = {}
        for asset, value in zip(problem.universe.assets, contributions):
            label = asset.attributes.get(attribute, UNASSIGNED)
            labels[label] = labels.get(label, 0.0) + float(value)
        by_group[attribute] = labels
    return RiskReport(
        total_risk=total_risk,
        factor_risk=float(np.sqrt(max(factor_variance, 0.0))),
        specific_risk=float(np.sqrt(specific_variance_total)),
        factor_exposures={factor_id: float(value) for factor_id, value in zip(factor_ids, y)},
        by_factor=by_factor,
        specific_contribution=float(specific_contribution),
        by_asset=by_asset,
        by_group=by_group,
    )


def beta_vector(problem: Problem) -> np.ndarray:
    """Return each asset's predicted beta against the benchmark.

    Beta is Sigma b / (b-transpose Sigma b) with b the benchmark weights, so
    the benchmark's own portfolio beta is exactly 1. Computed in factor form
    as B (F (B-transpose b)) + d * b over the benchmark variance.

    Parameters
    ----------
    problem : Problem
        Supplies the universe order, the risk model and the benchmark.

    Returns
    -------
    numpy.ndarray
        Float64 betas in universe order, shape (n_assets,).

    Raises
    ------
    ValueError
        If the problem has no benchmark ("predicted beta requires a
        benchmark") or its predicted variance is not positive ("predicted
        beta is undefined: benchmark has zero predicted variance").
    """
    if problem.benchmark is None:
        raise ValueError("predicted beta requires a benchmark")
    benchmark = _benchmark_vector(problem)
    exposures, factor_covariance, specific_variance = _universe_arrays(problem)
    covariance_times_benchmark = (
        exposures @ (factor_covariance @ (exposures.T @ benchmark)) + specific_variance * benchmark
    )
    benchmark_variance = float(benchmark @ covariance_times_benchmark)
    if benchmark_variance <= 0.0:
        raise ValueError("predicted beta is undefined: benchmark has zero predicted variance")
    return covariance_times_benchmark / benchmark_variance


def predicted_beta(problem: Problem, weights: Mapping[str, float]) -> float:
    """Return the predicted beta of a portfolio against the benchmark.

    Parameters
    ----------
    problem : Problem
        Supplies the universe order, the risk model and the benchmark.
    weights : mapping of str to float
        Portfolio weights as fractions of NAV, keyed by asset id; assets
        missing from the mapping have weight zero.

    Returns
    -------
    float
        Beta of the portfolio; the benchmark itself gives 1.

    Raises
    ------
    ValueError
        If a key of weights is not an asset of the universe ("weights: unknown
        asset '<id>'"), or if the problem has no benchmark ("predicted beta
        requires a benchmark").
    """
    return float(beta_vector(problem) @ _weight_vector(problem, weights))
