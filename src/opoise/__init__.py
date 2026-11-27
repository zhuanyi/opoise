"""Opoise: convex portfolio construction engine on cvxpy.

The package exposes declarative problem models (universe, factor risk model,
objectives, constraints, solver settings), JSON/YAML workspace files with a
loader protocol, a risk model provider protocol, a synthetic data generator,
risk analytics and a solve entry point. Importing the package does not import
cvxpy: solve, compile_problem and CompiledProblem are loaded on first use.
"""
from .constraints import (
    AssetBounds,
    BetaBounds,
    Bounds,
    Budget,
    Constraint,
    FactorExposureBounds,
    GroupBounds,
    GroupRiskCap,
    GroupRiskShareCap,
    LinearBounds,
    RiskCap,
    Turnover,
)
from .diagnostics import predicted_beta, risk_report
from .errors import (
    OpoiseError,
    SolverCapabilityError,
    SolverNotAvailableError,
    WorkspaceFormatError,
)
from .models import Asset, FactorRiskModel, Problem, SolverSettings, Universe
from .objectives import ObjectiveTerm, RiskTerm
from .providers import RiskModelProvider
from .result import (
    ConstraintRowReport,
    Diagnostics,
    InfeasibilityReport,
    Result,
    RiskReport,
    ScaReport,
)
from .synthetic import SyntheticData, synthetic_data
from .workspace import (
    NativeWorkspaceLoader,
    WorkspaceLoader,
    load_workspace,
    save_workspace,
)

__version__ = "0.1.0"

_LAZY = {"solve": "optimize", "compile_problem": "compiler", "CompiledProblem": "compiler"}
"""Public names that need cvxpy, and the module that defines them."""


def __getattr__(name: str):
    """Import a cvxpy-backed public name on first access and cache it.

    Parameters
    ----------
    name : str
        Attribute requested from the package.

    Returns
    -------
    object
        The named object, imported from its module and cached in the package
        globals, so a second access does not repeat the import.

    Raises
    ------
    AttributeError
        If the package has no such attribute ("module 'opoise' has no
        attribute '...'").
    """
    if name in _LAZY:
        import importlib

        value = getattr(importlib.import_module(f".{_LAZY[name]}", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Asset",
    "AssetBounds",
    "BetaBounds",
    "Bounds",
    "Budget",
    "CompiledProblem",
    "Constraint",
    "ConstraintRowReport",
    "Diagnostics",
    "FactorExposureBounds",
    "FactorRiskModel",
    "GroupBounds",
    "GroupRiskCap",
    "GroupRiskShareCap",
    "InfeasibilityReport",
    "LinearBounds",
    "NativeWorkspaceLoader",
    "ObjectiveTerm",
    "OpoiseError",
    "Problem",
    "Result",
    "RiskCap",
    "RiskModelProvider",
    "RiskReport",
    "RiskTerm",
    "ScaReport",
    "SolverCapabilityError",
    "SolverNotAvailableError",
    "SolverSettings",
    "SyntheticData",
    "Turnover",
    "Universe",
    "WorkspaceFormatError",
    "WorkspaceLoader",
    "__version__",
    "compile_problem",
    "load_workspace",
    "predicted_beta",
    "risk_report",
    "save_workspace",
    "solve",
    "synthetic_data",
]
