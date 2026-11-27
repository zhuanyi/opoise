"""Core problem models: assets, universe, risk model, settings and problem.

All models are frozen pydantic models with extra="forbid" and allow_inf_nan=False,
so a typo or a non-finite number fails at construction instead of silently
changing the optimisation. Cross-field errors are raised as ValueError inside
after-validators, so callers see pydantic.ValidationError with the message
quoted in the Raises sections.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .constraints import (
    AssetBounds,
    BetaBounds,
    Constraint,
    FactorExposureBounds,
    GroupBounds,
    GroupRiskCap,
    GroupRiskShareCap,
    LinearBounds,
)
from .objectives import ObjectiveTerm

UNASSIGNED = "UNASSIGNED"
"""Report label for assets that do not carry a grouped attribute; reserved."""

_CONFIG = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


def _duplicates(values: list[str]) -> list[str]:
    seen: set[str] = set()
    dup: list[str] = []
    for value in values:
        if value in seen and value not in dup:
            dup.append(value)
        seen.add(value)
    return dup


class Asset(BaseModel):
    """One investable asset: an id and free-form string attributes.

    Attributes
    ----------
    id : str
        Unique asset id, 1-64 characters excluding whitespace and brackets
        (row labels are written as name[id]).
    attributes : dict of str to str
        Grouping attributes such as sector or country. A value may not be
        "UNASSIGNED", which is reserved as a report label.

    Raises
    ------
    pydantic.ValidationError
        If the id is malformed or an attribute value is "UNASSIGNED"
        ("attribute '<key>': value 'UNASSIGNED' is reserved").
    """

    model_config = _CONFIG

    id: str = Field(pattern=r"^[^\s\[\]]{1,64}$")
    attributes: dict[str, str] = {}

    @model_validator(mode="after")
    def _validate(self) -> Asset:
        for key, value in self.attributes.items():
            if value == UNASSIGNED:
                raise ValueError(f"attribute '{key}': value 'UNASSIGNED' is reserved")
        return self


class Universe(BaseModel):
    """The ordered list of assets a problem invests in.

    Attributes
    ----------
    assets : list of Asset
        At least one asset; the order defines the weight order everywhere.

    Raises
    ------
    pydantic.ValidationError
        If the list is empty or an asset id repeats ("duplicate asset id(s): ...").
    """

    model_config = _CONFIG

    assets: list[Asset] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate(self) -> Universe:
        dup = _duplicates([asset.id for asset in self.assets])
        if dup:
            raise ValueError(f"duplicate asset id(s): {', '.join(dup[:5])}")
        return self

    def ids(self) -> list[str]:
        """Return the asset ids in universe order.

        Returns
        -------
        list of str
            Asset ids in universe order.
        """
        return [asset.id for asset in self.assets]


class FactorRiskModel(BaseModel):
    """A factor risk model: exposures, factor covariance, specific variance.

    The covariance of the asset returns is Sigma = B F B-transpose + diag(d)
    on the rows of the model; risk is the standard deviation in the units of
    the covariance. The model may cover more assets than a problem's universe.

    Attributes
    ----------
    asset_ids : list of str
        Row labels, at least one, unique.
    factor_ids : list of str
        Column and factor-covariance labels, at least one, unique.
    exposures : list of list of float
        B, shape (n_assets, n_factors).
    factor_covariance : list of list of float
        F, shape (n_factors, n_factors), symmetric and positive semidefinite.
    specific_variance : list of float
        d, one non-negative value per asset.

    Raises
    ------
    pydantic.ValidationError
        On duplicate ids, a wrong shape ("exposures: expected {n}x{k}",
        "factor_covariance: expected {k}x{k}", "specific_variance: expected
        {n} values"), a negative specific variance ("specific_variance must be
        >= 0"), an asymmetric factor covariance ("factor_covariance is not
        symmetric") or an indefinite one ("factor_covariance is not positive
        semidefinite (min eigenvalue ...)").
    """

    model_config = _CONFIG

    asset_ids: list[str] = Field(min_length=1)
    factor_ids: list[str] = Field(min_length=1)
    exposures: list[list[float]]
    factor_covariance: list[list[float]]
    specific_variance: list[float]

    @model_validator(mode="after")
    def _validate(self) -> FactorRiskModel:
        n, k = len(self.asset_ids), len(self.factor_ids)
        dup = _duplicates(self.asset_ids)
        if dup:
            raise ValueError(f"duplicate asset id(s): {', '.join(dup[:5])}")
        dup = _duplicates(self.factor_ids)
        if dup:
            raise ValueError(f"duplicate factor id(s): {', '.join(dup[:5])}")
        if len(self.exposures) != n or any(len(row) != k for row in self.exposures):
            raise ValueError(f"exposures: expected {n}x{k}")
        if len(self.factor_covariance) != k or any(len(row) != k for row in self.factor_covariance):
            raise ValueError(f"factor_covariance: expected {k}x{k}")
        if len(self.specific_variance) != n:
            raise ValueError(f"specific_variance: expected {n} values")
        if any(value < 0 for value in self.specific_variance):
            raise ValueError("specific_variance must be >= 0")
        covariance = np.asarray(self.factor_covariance, dtype=float)
        scale = max(1.0, float(np.abs(covariance).max()))
        if float(np.abs(covariance - covariance.T).max()) > 1e-10 * scale:
            raise ValueError("factor_covariance is not symmetric")
        eigenvalues = np.linalg.eigvalsh((covariance + covariance.T) / 2)
        if float(eigenvalues.min()) < -1e-10 * max(1.0, float(eigenvalues.max())):
            raise ValueError(
                "factor_covariance is not positive semidefinite "
                f"(min eigenvalue {float(eigenvalues.min()):.3g})"
            )
        return self

    @classmethod
    def from_arrays(
        cls,
        asset_ids: list[str],
        factor_ids: list[str],
        exposures: np.ndarray,
        factor_covariance: np.ndarray,
        specific_variance: np.ndarray,
    ) -> FactorRiskModel:
        """Build a model from numpy arrays, storing plain Python floats.

        Parameters
        ----------
        asset_ids : list of str
            Row labels in array order.
        factor_ids : list of str
            Column and factor-covariance labels in array order.
        exposures : numpy.ndarray
            B, shape (n_assets, n_factors).
        factor_covariance : numpy.ndarray
            F, shape (n_factors, n_factors).
        specific_variance : numpy.ndarray
            d, shape (n_assets,).

        Returns
        -------
        FactorRiskModel
            The validated model with list-of-float fields.
        """
        return cls(
            asset_ids=list(asset_ids),
            factor_ids=list(factor_ids),
            exposures=np.asarray(exposures, dtype=float).tolist(),
            factor_covariance=np.asarray(factor_covariance, dtype=float).tolist(),
            specific_variance=np.asarray(specific_variance, dtype=float).tolist(),
        )

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return fresh float64 copies of the model arrays.

        Returns
        -------
        tuple of numpy.ndarray
            (B, F, d) with shapes (n_assets, n_factors), (n_factors,
            n_factors) and (n_assets,). Every call returns new arrays.
        """
        return (
            np.array(self.exposures, dtype=float),
            np.array(self.factor_covariance, dtype=float),
            np.array(self.specific_variance, dtype=float),
        )


class SolverSettings(BaseModel):
    """Solver choice and tolerances.

    Attributes
    ----------
    solver : str
        "CLARABEL" (default), "OSQP", "SCS" or "MOSEK".
    solver_options : dict
        Passed unchanged to the solver, e.g. tol_gap_abs / tol_feas for
        Clarabel, eps_abs / max_iter for OSQP, eps_abs / max_iters for SCS.
    verbose : bool
        Ask the solver for its log.
    binding_tol : float
        Absolute tolerance for a row to count as binding when the bound is at
        most 1 in size, relative above it; lower it for very small bounds.
    diagnose_infeasibility : bool
        Search for a minimal conflicting constraint set when a problem is
        infeasible.
    sca_max_iterations : int
        Maximum number of subproblems for the sequential convex approximation
        of share-of-risk caps; running out turns the result into
        optimal_inaccurate with converged False.
    sca_tol : float
        Largest maximum weight change between two share-of-risk iterations
        that counts as converged.
    """

    model_config = _CONFIG

    solver: Literal["CLARABEL", "OSQP", "SCS", "MOSEK"] = "CLARABEL"
    solver_options: dict[str, bool | int | float | str] = {}
    verbose: bool = False
    binding_tol: float = Field(default=1e-5, gt=0)
    diagnose_infeasibility: bool = True
    sca_max_iterations: int = Field(default=50, ge=1)
    sca_tol: float = Field(default=1e-6, gt=0)


class Problem(BaseModel):
    """One declarative portfolio construction problem.

    Immutability is shallow: fields cannot be reassigned, but the contents of
    list and dict fields are not frozen, so do not mutate them after
    construction.

    Attributes
    ----------
    universe : Universe
        Assets and their attributes; defines weight order.
    risk_model : FactorRiskModel
        Must cover every universe asset (it may cover more).
    initial_holdings : dict of str to float
        Weights held before trading, fractions of NAV; missing assets are 0.
    benchmark : dict of str to float or None
        Benchmark weights on the universe, required by active terms and
        benchmark-relative constraints; its predicted variance must be positive.
    objective : list of ObjectiveTerm
        At least one term.
    constraints : list of Constraint
        Any of the constraint models; names must be unique.
    settings : SolverSettings
        Solver choice, options and tolerances.

    Raises
    ------
    pydantic.ValidationError
        On a risk model that misses universe assets ("risk_model: missing
        assets: ..."), unknown ids in holdings, benchmark, per-asset bounds,
        linear coefficients or constraints ("...: unknown asset '<id>'"), a
        benchmark with zero predicted variance ("benchmark: predicted variance
        is zero ..."), an active objective term or benchmark-relative
        constraint without a benchmark ("... requires a benchmark"), duplicate
        constraint names ("duplicate constraint name '<name>'"), an attribute
        set on no asset ("constraint '<name>': attribute '<attr>' is not set
        on any asset") and unknown group or factor labels ("...: unknown group
        '<g>'", "...: unknown factor '<f>'").
    """

    model_config = _CONFIG

    universe: Universe
    risk_model: FactorRiskModel
    initial_holdings: dict[str, float] = {}
    benchmark: dict[str, float] | None = None
    objective: list[ObjectiveTerm] = Field(min_length=1)
    constraints: list[Constraint] = []
    settings: SolverSettings = SolverSettings()

    @model_validator(mode="after")
    def _validate(self) -> Problem:
        ids = self.universe.ids()
        id_set = set(ids)
        row_of = {asset_id: row for row, asset_id in enumerate(self.risk_model.asset_ids)}
        missing = [asset_id for asset_id in ids if asset_id not in row_of]
        if missing:
            message = f"risk_model: missing assets: {', '.join(missing[:5])}"
            if len(missing) > 5:
                message += f" (+{len(missing) - 5} more)"
            raise ValueError(message)
        for asset_id in self.initial_holdings:
            if asset_id not in id_set:
                raise ValueError(f"initial_holdings: unknown asset '{asset_id}'")
        if self.benchmark is not None:
            for asset_id in self.benchmark:
                if asset_id not in id_set:
                    raise ValueError(f"benchmark: unknown asset '{asset_id}'")
            exposures, factor_covariance, specific_variance = self.risk_model.arrays()
            rows = np.fromiter((row_of[asset_id] for asset_id in ids), dtype=int, count=len(ids))
            benchmark = np.fromiter(
                (self.benchmark.get(asset_id, 0.0) for asset_id in ids), dtype=float, count=len(ids)
            )
            loadings = exposures[rows].T @ benchmark
            variance = float(
                loadings @ factor_covariance @ loadings + np.sum(specific_variance[rows] * benchmark * benchmark)
            )
            if variance <= 0.0:
                raise ValueError("benchmark: predicted variance is zero (empty, all-zero or riskless benchmark)")
        for index, term in enumerate(self.objective):
            if term.active and self.benchmark is None:
                raise ValueError(f"objective term {index} (active risk) requires a benchmark")
        factor_ids = set(self.risk_model.factor_ids)
        seen: set[str] = set()
        for constraint in self.constraints:
            if constraint.name in seen:
                raise ValueError(f"duplicate constraint name '{constraint.name}'")
            seen.add(constraint.name)
            if self.benchmark is None and (isinstance(constraint, BetaBounds) or getattr(constraint, "active", False)):
                raise ValueError(f"constraint '{constraint.name}' requires a benchmark")
            if isinstance(constraint, (GroupBounds, GroupRiskCap, GroupRiskShareCap)):
                labels = {
                    asset.attributes[constraint.attribute]
                    for asset in self.universe.assets
                    if constraint.attribute in asset.attributes
                }
                if not labels:
                    raise ValueError(
                        f"constraint '{constraint.name}': attribute '{constraint.attribute}' is not set on any asset"
                    )
                keys = constraint.bounds if isinstance(constraint, GroupBounds) else constraint.caps
                for label in keys:
                    if label not in labels:
                        raise ValueError(f"constraint '{constraint.name}': unknown group '{label}'")
            if isinstance(constraint, FactorExposureBounds):
                for factor_id in constraint.bounds:
                    if factor_id not in factor_ids:
                        raise ValueError(f"constraint '{constraint.name}': unknown factor '{factor_id}'")
            if isinstance(constraint, AssetBounds):
                for asset_id in constraint.per_asset:
                    if asset_id not in id_set:
                        raise ValueError(f"constraint '{constraint.name}': unknown asset '{asset_id}'")
            if isinstance(constraint, LinearBounds):
                for asset_id in constraint.coefficients:
                    if asset_id not in id_set:
                        raise ValueError(f"constraint '{constraint.name}': unknown asset '{asset_id}'")
        return self
