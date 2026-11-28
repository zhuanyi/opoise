# Opoise

[![CI](https://github.com/zhuanyi/opoise/actions/workflows/ci.yml/badge.svg)](https://github.com/zhuanyi/opoise/actions/workflows/ci.yml)

Opoise is a convex portfolio construction engine built on [cvxpy](https://www.cvxpy.org/): you describe a portfolio problem declaratively (universe, factor risk model, objective, constraints), and the engine compiles it to a DCP conic program, solves it with an open-source solver and returns weights with a risk decomposition and a constraint report. The risk model is an input, never estimated here: exposures `B`, a factor covariance `F` and specific variance `d`, with `Σ = B F Bᵀ + diag(d)`. Apache-2.0.

Status: version 0.2 (risk objectives, the constraint library including share-of-risk caps, workspace files, diagnostics).

Tested on Python 3.14.4 (Linux aarch64) with the versions pinned in `pyproject.toml` / `uv.lock` only.

## Install

From GitHub:

```sh
pip install "opoise @ git+https://github.com/zhuanyi/opoise.git"
# or with uv
uv pip install "git+https://github.com/zhuanyi/opoise.git"
```

From a checkout of this repository:

```sh
uv pip install .
# or, into the active environment
pip install .
```

## Development

The project uses uv with a plain project venv:

```sh
uv venv
uv sync --locked
```

Every command then runs through `uv run` (for example `uv run pytest -q`).

## Quickstart

```python
from opoise import AssetBounds, Budget, Problem, RiskCap, RiskTerm, Turnover, solve, synthetic_data

data = synthetic_data(n_assets=40, n_factors=5, n_sectors=4, seed=7)
problem = Problem(
    universe=data.universe,
    risk_model=data.risk_model,
    benchmark=data.benchmark,
    objective=[RiskTerm(aversion=2.0)],
    constraints=[
        Budget(name="budget"),
        AssetBounds(name="long_only", lower=0.0, upper=0.10),
        RiskCap(name="te", active=True, limit=0.03),
        Turnover(name="turnover", limit=0.40),
    ],
    initial_holdings=dict(data.benchmark),
)
result = solve(problem)
print(result.status, result.solver, round(result.objective_value, 6))
top = sorted(result.weights.items(), key=lambda item: -item[1])[:3]
print([(asset_id, round(weight, 4)) for asset_id, weight in top])
print(round(sum(result.weights.values()), 9), round(result.diagnostics.turnover, 4))
print(round(result.diagnostics.portfolio.total_risk, 6),
      round(result.diagnostics.active.total_risk, 6))
print(result.diagnostics.binding)
print({label: round(value, 4)
       for label, value in result.diagnostics.portfolio.by_group["sector"].items()})
```

The example builds a 40-asset problem from the synthetic generator: minimise
variance, stay fully invested, long only with a 10% per-asset cap, keep
tracking error at or below 3% against the benchmark, and trade at most 40%
two-way turnover from the benchmark holdings. The benchmark is also needed
because `RiskCap(active=True)` is a tracking-error cap on the active weights
`w − b`. The last three `print` calls show total and active risk as standard
deviations, the binding row labels, and each sector's contribution to total
risk (the sector contributions sum to the total).

## Conventions

- The objective is always minimised. A risk term costs
  `aversion · xᵀΣx`, with no factor of one half, where `x = w` or `w − b`
  for `active=True`.
- Weights are fractions of NAV. Turnover is two-way, `Σ|w − w0|`.
- Cash is not special: add a `CASH` asset with zero exposures and zero
  specific variance when you want explicit cash.
- Risk numbers come back as standard deviations in the units of the risk
  model; an annual covariance gives annual volatility.
- `binding_tol` assumes unit-scale bounds (weights, betas and risks are at
  most 1 in size): it is absolute for |bound| ≤ 1 and relative above. For very
  small bounds (for example benchmark weights around 1e-4 at n ≥ 1,000) lower
  it, for example to 1e-7.

## Documentation

The documentation lives in the repository and is readable as plain Markdown:

- [`docs/`](docs/) - the site source: concepts, formulation, the user guide,
  the generated API reference and the changelog. Start at
  [`docs/index.md`](docs/index.md).
- [`CHANGELOG.md`](CHANGELOG.md) - release notes in Keep a Changelog format,
  also rendered as [`docs/changelog.md`](docs/changelog.md).
- [Extending Opoise](docs/user-guide.md#extending-opoise) - risk model
  providers, file loaders and solver options, in the user guide.

Docstrings are the source of truth for the API and use the numpy convention;
`docs/reference.md` renders them with mkdocstrings. Build the site locally:

```sh
uv run mkdocs build --strict
```

The strict build turns every warning (a broken link, a malformed docstring
section, a page missing from the nav) into a failure. The rendered site is
written to `site/`, which is gitignored; nothing hosts it yet.

## Testing

Development: `uv run pytest -q`.

## Licence

Apache-2.0. See `LICENSE` and `NOTICE`.
