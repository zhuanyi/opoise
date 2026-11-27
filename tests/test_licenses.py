"""Fails if any distribution in the venv has not been licence-reviewed (the set is the uv.lock closure of all default groups).

ALLOWED is a name allowlist. The licence strings record the review (PyPI metadata, 2026-11-26); they are not
compared with installed metadata. Run it with `uv run pytest` (the project venv): under another interpreter it sees other packages.
"""
from importlib.metadata import distributions

ALLOWED = {
    # runtime closure of the pinned cvxpy stack
    "annotated-types": "MIT", "cffi": "MIT-0", "clarabel": "Apache-2.0", "cloudpickle": "BSD-3-Clause",
    "cvxpy": "Apache-2.0", "highspy": "MIT", "jinja2": "BSD-3-Clause", "joblib": "BSD-3-Clause",
    "markupsafe": "BSD-3-Clause", "numpy": "BSD-3-Clause", "osqp": "Apache-2.0",
    "packaging": "Apache-2.0 OR BSD-2-Clause", "pycparser": "BSD-3-Clause", "pydantic": "MIT", "pydantic-core": "MIT",
    "pyyaml": "MIT", "qdldl": "Apache-2.0", "scipy": "BSD-3-Clause", "scs": "MIT", "setuptools": "MIT",
    "sparsediffpy": "Apache-2.0", "typing-extensions": "PSF-2.0", "typing-inspection": "MIT",
    # this project (editable install by uv sync)
    "opoise": "Apache-2.0",
    # dev group: tests and lint
    "iniconfig": "MIT", "pluggy": "MIT", "pygments": "BSD-2-Clause", "pytest": "MIT", "ruff": "MIT",
    # docs group: mkdocs, mkdocstrings and their closure. PATHSPEC IS MPL-2.0 (file-level copyleft): it is a build-time
    # dependency of mkdocs only, never imported by the package and never shipped; kept only if open decision O8 is yes.
    "click": "BSD-3-Clause", "ghp-import": "Apache-2.0", "griffelib": "ISC", "markdown": "BSD-3-Clause",
    "mergedeep": "MIT", "mkdocs": "BSD-2-Clause", "mkdocs-autorefs": "ISC", "mkdocs-get-deps": "MIT",
    "mkdocstrings": "ISC", "mkdocstrings-python": "ISC", "pathspec": "MPL-2.0 (docs build tool only, see O8)",
    "platformdirs": "MIT", "pymdown-extensions": "MIT", "python-dateutil": "BSD-3-Clause", "pyyaml-env-tag": "MIT",
    "tzdata": "Apache-2.0", "watchdog": "Apache-2.0",
    # platform/Python-version conditional (not installed on Linux Python 3.14, listed so other hosts do not fail)
    "colorama": "BSD-3-Clause", "exceptiongroup": "MIT", "tomli": "MIT",
}


def norm(name):
    return name.lower().replace("_", "-").replace(".", "-")


def test_only_reviewed_permissive_distributions_installed():
    names = {norm(d.metadata["Name"]) for d in distributions()}
    unknown = sorted(names - set(ALLOWED))
    assert not unknown, f"unreviewed distributions in the venv: {unknown}"
