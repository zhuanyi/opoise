"""Workspace files: problems as portable JSON or YAML documents.

A native workspace document holds the format marker, a schema version and the
validated problem, so a problem definition can live in version control next to
the code that solves it. The native loader reads the .json, .yaml and .yml
suffixes; the WorkspaceLoader protocol is how a project plugs in a workspace
format of its own. Loading a workspace never imports cvxpy.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import yaml

from .errors import WorkspaceFormatError
from .models import Problem

WORKSPACE_FORMAT = "opoise-workspace"
"""Value of the format key carried by every native workspace document."""

SCHEMA_VERSION = 1
"""Value of the schema_version key carried by every native workspace document."""

_SUFFIXES = frozenset({".json", ".yaml", ".yml"})


@runtime_checkable
class WorkspaceLoader(Protocol):
    """Read a problem from one file format.

    Implement this protocol to load a workspace format the package does not
    know, then pass the instance to load_workspace. It is checked structurally:
    any object with a matching can_load and load is accepted, no subclassing
    is needed.
    """

    def can_load(self, path: Path) -> bool:
        """Return True when this loader handles the file.

        Parameters
        ----------
        path : pathlib.Path
            Candidate workspace file.

        Returns
        -------
        bool
            True when load can read this file.
        """
        ...

    def load(self, path: Path) -> Problem:
        """Read the file and return the validated problem.

        Parameters
        ----------
        path : pathlib.Path
            Workspace file this loader accepted.

        Returns
        -------
        Problem
            The problem stored in the file.

        Raises
        ------
        WorkspaceFormatError
            If the file cannot be parsed or recognised.
        """
        ...


class NativeWorkspaceLoader:
    """Load the native JSON and YAML workspace format.

    A file is accepted when its suffix is .json, .yaml or .yml, in any case.
    JSON is parsed with json.loads and YAML with yaml.safe_load, so a YAML file
    cannot construct arbitrary Python objects.
    """

    def can_load(self, path: Path) -> bool:
        """Return True for the .json, .yaml and .yml suffixes, in any case.

        Parameters
        ----------
        path : pathlib.Path
            Candidate workspace file.

        Returns
        -------
        bool
            True when the suffix is a native workspace suffix.
        """
        return path.suffix.lower() in _SUFFIXES

    def load(self, path: Path) -> Problem:
        """Read a native workspace file and return its problem.

        Parameters
        ----------
        path : pathlib.Path
            Native workspace file.

        Returns
        -------
        Problem
            The validated problem stored under the problem key.

        Raises
        ------
        WorkspaceFormatError
            If the file cannot be parsed ("<path>: cannot parse: <error>") or
            is not a native document with the current schema version
            ("<path>: expected format 'opoise-workspace' with schema_version 1").
        pydantic.ValidationError
            If the stored problem is invalid; it propagates unchanged.
        """
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            try:
                data = json.loads(text)
            except json.JSONDecodeError as error:
                raise WorkspaceFormatError(f"{path}: cannot parse: {error}") from error
        else:
            try:
                data = yaml.safe_load(text)
            except yaml.YAMLError as error:
                raise WorkspaceFormatError(f"{path}: cannot parse: {error}") from error
        if (
            not isinstance(data, dict)
            or data.get("format") != WORKSPACE_FORMAT
            or data.get("schema_version") != SCHEMA_VERSION
        ):
            raise WorkspaceFormatError(
                f"{path}: expected format '{WORKSPACE_FORMAT}' with schema_version {SCHEMA_VERSION}"
            )
        return Problem.model_validate(data.get("problem"))


def save_workspace(problem: Problem, path: Path) -> None:
    """Write a problem to a native workspace file.

    The document is a mapping with the keys format, schema_version and problem,
    in that order; problem is the JSON-mode dump of the model. JSON is written
    with two-space indentation and a trailing newline, YAML without sorting
    keys, so the file diffs cleanly in version control.

    Parameters
    ----------
    problem : Problem
        The validated problem to write.
    path : pathlib.Path
        Destination file; the suffix selects the format.

    Raises
    ------
    WorkspaceFormatError
        If the suffix is not .json, .yaml or .yml
        ("<path>: unsupported suffix '<suffix>' (use .json, .yaml or .yml)").
    """
    suffix = path.suffix
    if suffix.lower() not in _SUFFIXES:
        raise WorkspaceFormatError(f"{path}: unsupported suffix '{suffix}' (use .json, .yaml or .yml)")
    document = {
        "format": WORKSPACE_FORMAT,
        "schema_version": SCHEMA_VERSION,
        "problem": problem.model_dump(mode="json"),
    }
    if suffix.lower() == ".json":
        path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    else:
        path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def load_workspace(path: Path, loaders: Sequence[WorkspaceLoader] | None = None) -> Problem:
    """Load a problem with the first loader that accepts the file.

    Parameters
    ----------
    path : pathlib.Path
        Workspace file to read.
    loaders : sequence of WorkspaceLoader or None
        Loaders tried in order; the default is the native loader alone.

    Returns
    -------
    Problem
        The problem returned by the first loader whose can_load is True.

    Raises
    ------
    WorkspaceFormatError
        If no loader accepts the file ("<path>: no loader accepts this file").
    """
    candidates = [NativeWorkspaceLoader()] if loaders is None else list(loaders)
    for loader in candidates:
        if loader.can_load(path):
            return loader.load(path)
    raise WorkspaceFormatError(f"{path}: no loader accepts this file")
