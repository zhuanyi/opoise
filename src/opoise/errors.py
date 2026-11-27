"""Exception types raised by Opoise.

All errors raised by the package itself derive from OpoiseError, so a caller
can catch one base class. Invalid inputs are not raised as these types: they
surface as pydantic.ValidationError at model construction.
"""
from __future__ import annotations


class OpoiseError(Exception):
    """Base class for every error Opoise raises itself."""


class SolverNotAvailableError(OpoiseError):
    """Raised when the requested solver is not installed.

    The message names the solver and the installed solvers, e.g.
    "solver MOSEK is not installed (installed: [...])."
    """


class SolverCapabilityError(OpoiseError):
    """Raised when a solver is installed but cannot handle the problem class.

    The only case is OSQP asked to solve a problem with
    second-order cone rows (risk caps); OSQP solves quadratic programs only.
    """


class WorkspaceFormatError(OpoiseError):
    """Raised when a workspace file cannot be read, parsed or recognised."""
