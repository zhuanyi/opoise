"""Risk model providers: the protocol a risk data source implements.

The package defines the protocol only; the package never fetches data. An adapter
maps the provider's ids and units onto FactorRiskModel and is validated like
any other risk model, so a bad model fails at construction.
"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Protocol, runtime_checkable

from .models import FactorRiskModel


@runtime_checkable
class RiskModelProvider(Protocol):
    """Supply a factor risk model for a set of assets at a point in time.

    Implement this protocol to connect the engine to a risk data source. The
    adapter returns declarative data, so data access stays outside the
    package: map the source's asset ids to universe ids (no whitespace or
    brackets), supply exposures, factor covariance and specific variance in one
    consistent unit (state the annualisation), and let FactorRiskModel validate
    the result. It is checked structurally: any object with a matching
    risk_model method is accepted.
    """

    def risk_model(self, asset_ids: Sequence[str], as_of: date) -> FactorRiskModel:
        """Return the risk model covering the assets as of the given date.

        Parameters
        ----------
        asset_ids : sequence of str
            Universe ids to cover; the model may cover more assets.
        as_of : datetime.date
            Observation date of the model.

        Returns
        -------
        FactorRiskModel
            The validated model, with units consistent with the problem.
        """
        ...
