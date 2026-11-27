# Opoise

Opoise is a convex portfolio construction engine on
[cvxpy](https://www.cvxpy.org/): you declare a universe, a factor risk model,
an objective and constraints, and get back weights with a risk decomposition
and a constraint report. The risk model is an input, never estimated here:
exposures $B$, a factor covariance $F$ and a specific variance $d$ define
$\Sigma = B F B^{\top} + \operatorname{diag}(d)$. Apache-2.0; version 0.1 (risk
objectives, basic constraints, diagnostics).

## Install

From a checkout, with uv:

```sh
uv venv
uv sync --locked
```

Then run everything through `uv run` (or `uv pip install .` to install the
package into the current environment). The tested platform is Python 3.14.4 on
Linux aarch64 with the versions pinned in `pyproject.toml` and `uv.lock` only.

## Quickstart

```python
from opoise import AssetBounds, Budget, Problem, RiskTerm, solve, synthetic_data
data = synthetic_data()
problem = Problem(universe=data.universe, risk_model=data.risk_model, benchmark=data.benchmark,
                  objective=[RiskTerm(aversion=1.0)],
                  constraints=[Budget(name="budget"), AssetBounds(name="long_only", lower=0.0)])
result = solve(problem)
print(result.status, round(result.diagnostics.portfolio.total_risk, 4), result.diagnostics.binding)
```

## Documentation

- [Concepts](concepts.md): the data model, the factor form of the risk model and
  the conventions every quantity follows.
- [Formulation](formulation.md): the notation, the Euler risk decomposition, and
  every objective term and constraint with its formula and convexity class.
- [User guide](user-guide.md): workspace files, reading a result, building and
  solving a problem, a constraint cookbook, infeasibility and sector
  share-of-risk caps, with runnable examples.
- [API reference](reference.md): every public module, generated from the
  docstrings by mkdocstrings.
- [Changelog](changelog.md): release notes in Keep a Changelog format, kept as
  `CHANGELOG.md` in the repository root and copied to the docs so it renders
  there as well.

Render the site locally with `uv run mkdocs build --strict`; the strict build
fails on any warning, so a broken link or a missing nav page cannot go unnoticed.

## Licence

Apache-2.0. See `LICENSE` and `NOTICE` in the repository root.
