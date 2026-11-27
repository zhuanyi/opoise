# User guide

## Workspace files

A problem is data: a universe, a factor risk model, objective terms, constraints
and solver settings. Opoise writes that data to a **workspace file** so a
booking, a review or a production run can share one validated definition instead
of a script. The native format comes in two spellings, JSON and YAML, chosen by
the file suffix (.json, .yaml or .yml, in any case):

```python
import tempfile
from pathlib import Path

from opoise import Budget, Problem, RiskTerm, load_workspace, save_workspace, synthetic_data

data = synthetic_data()
problem = Problem(universe=data.universe, risk_model=data.risk_model, benchmark=data.benchmark,
                  objective=[RiskTerm(aversion=2.0)], constraints=[Budget(name="budget")])
path = Path(tempfile.mkdtemp()) / "problem.json"
save_workspace(problem, path)
print(load_workspace(path) == problem)
```

A native document is a mapping with three keys, in this order: `format` (always
`opoise-workspace`), `schema_version` (currently 1) and `problem`, the problem
model dumped to JSON values. JSON is written with two-space indentation and a
trailing newline, YAML without sorting keys, so the file diffs cleanly in
version control. Loading validates the document and then the problem exactly as
if it had been constructed in Python: a typo, an unknown constraint type or a
missing benchmark fails at load time, not at solve time.

The loader rejects a file it cannot parse (`cannot parse` in the message) and a
file that is not a native document at the current schema version (`expected
format` in the message); both raise `WorkspaceFormatError` from
`opoise`. A problem that fails validation raises `pydantic.ValidationError`
unchanged, so the error message names the offending field.

### Loading a format of your own

`load_workspace` accepts any list of loaders and uses the first one whose
`can_load` returns True, so a project can add its own format without touching
the package. A loader is anything with those two methods; `WorkspaceLoader` is
the protocol it is checked against:

```python
import tempfile
from pathlib import Path

from opoise import Problem, RiskTerm, WorkspaceLoader, load_workspace, synthetic_data


class MyFormatLoader:
    """Read a project-specific problem format."""

    def can_load(self, path: Path) -> bool:
        return path.suffix == ".mine"

    def load(self, path: Path) -> Problem:
        data = synthetic_data()
        return Problem(universe=data.universe, risk_model=data.risk_model, benchmark=data.benchmark,
                       objective=[RiskTerm(aversion=1.0)])


loader = MyFormatLoader()
print(isinstance(loader, WorkspaceLoader))
path = Path(tempfile.mkdtemp()) / "problem.mine"
path.write_text("our own format")
print(load_workspace(path, loaders=[loader]).objective[0].aversion)
```

A loader for another file format maps every element of that format to an engine
model and raises one error that lists the elements it cannot map, rather than
dropping them, so every gap in the translation is visible.

## Reading a result

```python
import numpy as np

from opoise import Problem, RiskTerm, predicted_beta, risk_report, synthetic_data

data = synthetic_data()
problem = Problem(universe=data.universe, risk_model=data.risk_model, benchmark=data.benchmark,
                  objective=[RiskTerm(aversion=1.0)])
B, F, d = data.risk_model.arrays()
w = np.linalg.solve(B @ F @ B.T + np.diag(d), np.ones(len(data.universe.assets)))
w = w / w.sum()
weights = {asset.id: float(value) for asset, value in zip(data.universe.assets, w)}

report = risk_report(problem, weights)
print(round(report.total_risk, 6), round(report.factor_risk, 6), round(report.specific_risk, 6))
print(round(sum(report.by_asset.values()), 6) == round(report.total_risk, 6))
print(sorted(report.by_group)[0], round(sum(report.by_group["sector"].values()), 6))
print(round(predicted_beta(problem, weights), 6))
```

A solve returns a `Result`: the `status`, the `solver` and `solve_seconds`, and,
when the status is `optimal` or `optimal_inaccurate`, the `objective_value`,
the `weights`, the `trades` against the initial holdings and a `Diagnostics`
object. Any other status leaves those four fields `None` and explains itself in
`message`: infeasible, unbounded and solver failures are outcomes, not
exceptions.

`Diagnostics` carries the absolute risk report `portfolio`, the `active` risk
report against the benchmark (or `None` without one), the predicted `beta`, the
two-way `turnover` (the sum of absolute trades), and `constraints`: one
`ConstraintRowReport` per compiled row, with the row label, the value taken, the
bounds, the `slack` and the `binding` and `violated` flags. The `binding` list
repeats the labels of the binding rows, so a report can state at a glance which
limits shaped the portfolio.

A `RiskReport` splits the risk three ways: by asset, by factor plus a specific
contribution, and by group for every grouping attribute of the universe (assets
without the attribute appear under `UNASSIGNED`). The contributions at each
level sum to `total_risk`, which is the standard deviation in the risk model's
units; for a portfolio with zero risk all contributions are zero. `factor_risk`
and `specific_risk` are standard deviations, so they add in quadrature, not
linearly: `total_risk**2 == factor_risk**2 + specific_risk**2`. Use
`risk_report` on any weights, not only on solved ones, to inspect an existing
portfolio, and `predicted_beta` for its beta against the benchmark.

## Building and solving a problem

`solve` takes a validated `Problem` and returns a `Result`. Everything else is
derived: the compiler slices the risk model to the universe, builds the weight
variable, turns each constraint into a row and each objective term into a
squared norm, and the solver only sees a quadratic or second-order cone
problem.

```python
from opoise import AssetBounds, Budget, Problem, RiskCap, RiskTerm, solve, synthetic_data

data = synthetic_data()
problem = Problem(
    universe=data.universe,
    risk_model=data.risk_model,
    benchmark=data.benchmark,
    objective=[RiskTerm(aversion=1.0)],
    constraints=[
        Budget(name="budget"),
        AssetBounds(name="long_only", lower=0.0),
        RiskCap(name="te", active=True, limit=0.03),
    ],
)
result = solve(problem)
print(result.status, result.solver, result.objective_value is not None)
print(round(result.diagnostics.portfolio.total_risk, 6), round(result.diagnostics.active.total_risk, 6))
print(result.diagnostics.binding)
```

`diagnostics.binding` lists the row labels that sit on a bound. The tracking-error
cap appears there when it shapes the portfolio and is absent when another
constraint is the binding one.

### Choosing a solver

The default is Clarabel, which solves every problem the package builds. `OSQP` is faster
on the QP subset (no `risk_cap`, `group_risk_cap` or `group_risk_share_cap`
row) and `solve` raises `SolverCapabilityError` rather than letting it fail
obscurely on a cone problem. `SCS` handles the same classes with looser
accuracy, and `MOSEK` is optional and licensed by the user: install it with
`uv pip install 'opoise[mosek]'`. Asking for a solver that is not installed
raises `SolverNotAvailableError`.

```python
from opoise import Budget, Problem, RiskTerm, SolverSettings, solve, synthetic_data

data = synthetic_data()
settings = SolverSettings(
    solver="OSQP",
    solver_options={"eps_abs": 1e-9, "eps_rel": 1e-9, "max_iter": 200000},
)
problem = Problem(
    universe=data.universe,
    risk_model=data.risk_model,
    objective=[RiskTerm(aversion=1.0)],
    constraints=[Budget(name="budget")],
    settings=settings,
)
result = solve(problem)
print(result.status, result.solver, result.solve_seconds >= 0.0)
```

`solver_options` go to the solver unchanged. Clarabel rejects an unknown option
with a `TypeError`, OSQP with a `SolverError` or a `ValueError` depending on
whether the problem was already solved; `solve` turns all three into a
`Result` with status `error` and the solver's message, never an exception.

Clarabel's default tolerances (1e-8) can leave a binding row about 1e-6 away
from its bound. When that matters, pass tighter `tol_gap_abs`, `tol_gap_rel`
and `tol_feas` values (for example 1e-9) in `solver_options`.

### Reading the status

A solve ends in one of five statuses:

- `optimal` and `optimal_inaccurate`: a portfolio came back. Only the second
  one means the solver stopped short of its target accuracy; the message says
  so, and a row may then show as violated.
- `infeasible`: no weight vector meets every constraint; the result carries a
  conflict set naming the constraints responsible, unless diagnosis is turned
  off.
- `unbounded`: the objective can be made arbitrarily small, usually a missing
  budget or bounds.
- `error`: the solver failed or rejected an option; the message carries its
  own text.

Except for the two optimal statuses, `objective_value`, `weights`, `trades` and
`diagnostics` are `None`:

```python
from opoise import AssetBounds, Budget, Problem, RiskTerm, solve, synthetic_data

data = synthetic_data()
problem = Problem(
    universe=data.universe,
    risk_model=data.risk_model,
    objective=[RiskTerm(aversion=1.0)],
    constraints=[Budget(name="budget"), AssetBounds(name="floor", lower=0.9)],
)
result = solve(problem)
print(result.status, result.weights, result.message)
```

`solve_seconds` times the solver call alone, so compilation, measurement and
the infeasibility diagnosis are excluded from it.

## Constraint cookbook

Every constraint is a frozen model with a unique `name`, and that name becomes
the prefix of the row labels in the report. Each block below is complete: it
builds a problem on `synthetic_data()`, solves it and prints the quantity the
constraint acts on.

### Every weight sums to a total (`Budget`)

```python
from opoise import Budget, Problem, RiskTerm, solve, synthetic_data

data = synthetic_data()
problem = Problem(universe=data.universe, risk_model=data.risk_model,
                  objective=[RiskTerm(aversion=1.0)],
                  constraints=[Budget(name="budget", total=1.0)])
result = solve(problem)
print(result.status, round(sum(result.weights.values()), 9))
```

`total=0.0` asks for a dollar-neutral portfolio; the row value is the sum of
the weights.

### Per-asset limits (`AssetBounds`)

```python
from opoise import AssetBounds, Budget, Problem, RiskTerm, solve, synthetic_data

data = synthetic_data()
problem = Problem(universe=data.universe, risk_model=data.risk_model,
                  objective=[RiskTerm(aversion=1.0)],
                  constraints=[Budget(name="budget"),
                               AssetBounds(name="long_only", lower=0.0, upper=0.10)])
result = solve(problem)
w = result.weights.values()
print(result.status, min(w) >= -1e-9, max(w) <= 0.10 + 1e-9)
```

A `per_asset` entry overrides both default sides for one asset, and
`active=True` applies the limits to `w - benchmark` instead of `w`.

### Bounds on the trade (`TradeBounds`)

```python
from opoise import Budget, Problem, RiskTerm, TradeBounds, solve, synthetic_data

data = synthetic_data()
problem = Problem(universe=data.universe, risk_model=data.risk_model, benchmark=data.benchmark,
                  objective=[RiskTerm(aversion=1.0)],
                  constraints=[Budget(name="budget"),
                               TradeBounds(name="trade", lower=-0.05, upper=0.05)],
                  initial_holdings=dict(data.benchmark))
result = solve(problem)
print(result.status, max(abs(value) for value in result.trades.values()) <= 0.05 + 1e-9)
```

The bound applies to the trade `w - w0` against the initial holdings, so no
benchmark is involved; a `per_asset` entry replaces both default sides for one
asset.

### Turnover against the initial holdings (`Turnover`)

```python
from opoise import Budget, Problem, RiskTerm, Turnover, solve, synthetic_data

data = synthetic_data()
problem = Problem(universe=data.universe, risk_model=data.risk_model, benchmark=data.benchmark,
                  objective=[RiskTerm(aversion=1.0)],
                  constraints=[Budget(name="budget"), Turnover(name="turnover", limit=0.5)],
                  initial_holdings=dict(data.benchmark))
result = solve(problem)
print(result.status, round(result.diagnostics.turnover, 6))
```

The limit is **two-way** turnover, so one-way turnover is half of it.

### Group weight bounds (`GroupBounds`)

```python
from opoise import Budget, Bounds, GroupBounds, Problem, RiskTerm, solve, synthetic_data

data = synthetic_data()
sector = {asset.id: asset.attributes["sector"] for asset in data.universe.assets}
problem = Problem(universe=data.universe, risk_model=data.risk_model,
                  objective=[RiskTerm(aversion=1.0)],
                  constraints=[Budget(name="budget"),
                               GroupBounds(name="sectors", attribute="sector",
                                           bounds={"S0": Bounds(upper=0.30)})])
result = solve(problem)
print(result.status, round(sum(value for asset_id, value in result.weights.items()
                               if sector[asset_id] == "S0"), 6))
```

The bounds apply to the sum of the weights of the assets carrying the label;
`active=True` bounds the group's active weight instead.

### Group stand-alone risk caps (`GroupRiskCap`)

```python
import numpy as np

from opoise import Budget, GroupRiskCap, Problem, RiskTerm, solve, synthetic_data

data = synthetic_data()
B, F, d = data.risk_model.arrays()
mask = np.array([asset.attributes["sector"] == "S0" for asset in data.universe.assets], dtype=float)
problem = Problem(universe=data.universe, risk_model=data.risk_model,
                  objective=[RiskTerm(aversion=1.0)],
                  constraints=[Budget(name="budget"),
                               GroupRiskCap(name="srisk", attribute="sector", caps={"S0": 0.05})])
result = solve(problem)
x = mask * np.array([result.weights[asset.id] for asset in data.universe.assets])
print(result.status, round(float(np.sqrt(x @ (B @ F @ B.T + np.diag(d)) @ x)), 6))
```

This is the **stand-alone** risk of the group sub-portfolio, not the group's
contribution to total risk. `measure` picks total, factor or specific risk and
`active=True` measures it on `w - benchmark`.

### Beta bounds (`BetaBounds`)

```python
from opoise import BetaBounds, Budget, Problem, RiskTerm, solve, synthetic_data

data = synthetic_data()
problem = Problem(universe=data.universe, risk_model=data.risk_model, benchmark=data.benchmark,
                  objective=[RiskTerm(aversion=1.0)],
                  constraints=[Budget(name="budget"),
                               BetaBounds(name="beta", lower=0.8, upper=1.2)])
result = solve(problem)
print(result.status, round(result.diagnostics.beta, 6))
```

The bounds are on the portfolio beta $\beta^{\top} w$, with the per-asset
predicted beta computed from the benchmark; the constraint requires a
benchmark.

### Risk caps and tracking error (`RiskCap`)

```python
from opoise import Budget, Problem, RiskCap, RiskTerm, solve, synthetic_data

data = synthetic_data()
problem = Problem(universe=data.universe, risk_model=data.risk_model, benchmark=data.benchmark,
                  objective=[RiskTerm(aversion=1.0)],
                  constraints=[Budget(name="budget"), RiskCap(name="te", active=True, limit=0.03)])
result = solve(problem)
print(result.status, round(result.diagnostics.active.total_risk, 6))
```

`active=True` turns the cap into a tracking-error cap; `measure` picks total,
factor or specific risk. Caps are upper bounds only, because a floor on risk is
not convex.

### Factor exposure bounds (`FactorExposureBounds`)

```python
import numpy as np

from opoise import Budget, Bounds, FactorExposureBounds, Problem, RiskTerm, solve, synthetic_data

data = synthetic_data()
problem = Problem(universe=data.universe, risk_model=data.risk_model,
                  objective=[RiskTerm(aversion=1.0)],
                  constraints=[Budget(name="budget"),
                               FactorExposureBounds(name="fx",
                                                    bounds={"MARKET": Bounds(lower=1.0, upper=1.0)})])
result = solve(problem)
B, _, _ = data.risk_model.arrays()
print(result.status, round(float(B[:, 0] @ np.array([result.weights[asset.id]
                                                       for asset in data.universe.assets])), 6))
```

One bound per factor id; `active=True` measures the exposure of the active
weights.

### A general linear bound (`LinearBounds`)

```python
from opoise import Budget, LinearBounds, Problem, RiskTerm, solve, synthetic_data

data = synthetic_data()
first, second = data.universe.assets[0].id, data.universe.assets[1].id
problem = Problem(universe=data.universe, risk_model=data.risk_model,
                  objective=[RiskTerm(aversion=1.0)],
                  constraints=[Budget(name="budget"),
                               LinearBounds(name="pair", coefficients={first: 1.0, second: 1.0},
                                            lower=0.30)])
result = solve(problem)
print(result.status, round(result.weights[first] + result.weights[second], 6))
```

The coefficients name assets and default to 0, so the row bounds any basket,
funding split or hand-made exposure.

## When a problem is infeasible

An infeasible problem returns status `infeasible` with no weights, and, unless
`diagnose_infeasibility=False`, a report naming a conflict set:

```python
from opoise import AssetBounds, BetaBounds, Budget, Problem, RiskTerm, solve, synthetic_data

data = synthetic_data()
problem = Problem(universe=data.universe, risk_model=data.risk_model,
                  benchmark=data.benchmark,
                  objective=[RiskTerm(aversion=1.0)],
                  constraints=[Budget(name="budget"),
                               AssetBounds(name="long_only", lower=0.0),
                               BetaBounds(name="beta_min", lower=5.0)])
result = solve(problem)
print(result.status)
print(result.infeasibility.conflict_set)
print(result.infeasibility.solves)
print(result.message)
```

The set is minimal with respect to the subset solves the filter completed:
dropping any member makes the rest feasible. It names one conflict only, so a
second, independent conflict elsewhere in the problem is not listed. When a
subset solve is inconclusive (a solver error, or reduced accuracy) the
constraint is kept in the set and the message warns that the set may not be
minimal. Set `SolverSettings(diagnose_infeasibility=False)` to skip the extra
solves; the result then carries `infeasibility=None` and the message
`"infeasible (diagnosis disabled)"`.

The diagnosis costs one feasibility solve per constraint plus one, so on a
large problem with many constraints turn it off when you only need the status.

## Sector share-of-risk caps

Two constraints limit how much risk one sector carries, and they mean different
things.

`GroupRiskCap` (variant A) caps the **stand-alone** risk of the sector
sub-portfolio: the sector's own covariance with every other asset weighted
zero. It is a second-order cone, so it solves in one convex pass to a global
optimum, and it is the right tool when you want to limit what the sector would
lose on its own.

`GroupRiskShareCap` (variant B) caps the sector's **share of the portfolio's
total or active risk**: its Euler contribution divided by the total risk. This
is the quantity risk attribution reports, but it is not convex, so `solve`
runs a sequential convex approximation instead of one solve. Every iterate
satisfies the true caps within solver tolerance. Subproblems that return status
`optimal_inaccurate` are accepted, so a cap can be missed by an amount of the
order of the solver tolerance. The answer is a local optimum that depends on
the starting point, which is the minimum of the problem without the share caps.
The first subproblem can report `error` even when the non-convex problem is
feasible, because the inner approximation at the starting point can be too
tight, so that status is not a proof of infeasibility.

```python
from opoise import Budget, GroupRiskShareCap, Problem, RiskTerm, solve, synthetic_data

data = synthetic_data()
problem = Problem(universe=data.universe, risk_model=data.risk_model,
                  benchmark=data.benchmark,
                  objective=[RiskTerm(aversion=1.0)],
                  constraints=[Budget(name="budget"),
                               GroupRiskShareCap(name="share", attribute="sector",
                                                 caps={"S0": 0.30})])
result = solve(problem)
print(result.status, result.sca.converged, result.sca.iterations)
print({label: round(value, 6) for label, value in result.sca.shares["share"].items()})
print(result.diagnostics.binding)
```

Read the outcome from three places:

- `result.sca` is a `ScaReport` whenever the sequential approximation ran to
  a result (`status` `optimal` or `optimal_inaccurate`), and `None` otherwise:
  for a problem without a share cap, and on the failure paths, namely when
  the solve without the share caps is not optimal (that result is returned
  with a `"without the share caps: "` message prefix), when a subproblem
  hits a solver error, and when an iteration subproblem is not optimal (the
  result then has status `error`). `converged` is true only when the last
  iteration moved no weight by more than `sca_tol`; that is a stopping rule,
  not a certificate that the weights are optimal. `max_weight_change` is that
  movement and `iterations` counts the convex subproblems solved (the starting
  solve of the problem without the share caps is not one of them).
- `result.sca.shares[constraint_name]` holds the **true** share of every label
  of the attribute, in order of first appearance (`UNASSIGNED` included), and
  the values sum to 1 when the total risk is positive
  ($x^{\top} \Sigma x > 0$); when it is zero, every share is reported as 0.
  The row report for `name[label]` carries the same
  number as its value, the cap as its upper bound and no lower bound.
- `result.status` is `optimal` when the iteration converged on an exact
  optimum. Running out of `SolverSettings(sca_max_iterations=...)` gives
  `optimal_inaccurate` with `converged` false and a message naming the last
  weight change: the weights still satisfy the caps within solver tolerance,
  but they may be sub-optimal, and the result never claims otherwise.

`sca_max_iterations` (default 50) and `sca_tol` (default 1e-6) trade
accuracy for time: a loose cap needs a single iteration, a binding one typically
five to ten. Options besides CLARABEL, SCS or MOSEK do not apply here: OSQP is
refused with `SolverCapabilityError` because every subproblem has a convex
quadratic constraint and is not a QP.

## Extending Opoise

The engine has no knowledge of where its data comes from or how a file is
written. A project supplies its own sources and formats by implementing one of
the small protocols below; the package only consumes the declarative models
that come out.

### Risk model providers

`opoise.providers.RiskModelProvider` has one method,
`risk_model(asset_ids, as_of) -> FactorRiskModel`, and is matched structurally:
any object with that method is accepted. Return the model for exactly the
requested asset ids.

- Universe ids may not contain whitespace or brackets, because report rows are
  written as `name[id]`; translate the source's identifiers on the way in.
- Give exposures, factor covariance and specific variance in one consistent
  unit, and record the annualisation wherever the model is built. Opoise never
  rescales: a variance in is a variance out.
- `FactorRiskModel` validates the result: a symmetric positive semidefinite
  factor covariance, a specific variance of at least zero and matching shapes.
  Adjust an adapter that fails these checks rather than bypassing them.

### File loaders

A loader has `can_load(path)` and `load(path) -> Problem`, and matches
`opoise.providers.WorkspaceLoader`; `load_workspace` takes the first loader
whose `can_load` returns True, and [Loading a format of your
own](#loading-a-format-of-your-own) shows a complete one. Translating a format
means mapping each element to an engine model (the universe and its grouping
attributes, the risk model, the objective terms and each constraint) and
raising one error that names every element left unmapped, so nothing is
dropped silently.

### Custom constraints and objectives

The registries are open: a project adds a type in its own code, names a
category, and the type then works in a problem, a JSON document and a workspace
file while it is registered. A new built-in category is one module under
`opoise/constraints/` or `opoise/objectives/`; the package imports every
module there in sorted order.

A constraint declares its category first. A built-in one can be reused, for
example `opoise.constraints.holding.CATEGORY`, or a new one declared:

```python
from typing import Literal

from opoise import Category, ConstraintBase, register_category, register_constraint

MY_RISK = register_category(Category("MY_RISK", "Custom risk limits.", "constraint"))


@register_constraint
class BasketCap(ConstraintBase):
    category = MY_RISK
    type: Literal["basket_cap"] = "basket_cap"
    assets: list[str]
    limit: float

    def build(self, ctx):
        import cvxpy as cp
        import numpy as np

        from opoise.rows import Row

        mask = np.zeros(ctx.n)
        for asset_id in self.assets:
            mask[ctx.index[asset_id]] = 1.0
        return Row(constraint=self.name, labels=[self.name], expr=cp.hstack([mask @ ctx.w]),
                   lower=np.array([-np.inf]), upper=np.array([float(self.limit)]))
```

Every subclass declares its type tag in the class body with
`type: Literal[...] = ...`, sets `category` to a registered category of the
same kind, and overrides `build(self, ctx)`; `@register_constraint` accepts the
class or is applied as a decorator, and `register_constraint(MyClass,
replace=True)` replaces a tag. A row carries its labels, an affine expression,
or a convex one with upper bounds only; the compiler reports a malformed or
non-DCP row with the constraint name and its type.

An objective term uses `ObjectiveBase` and `@register_objective`. The
compiler minimises the sum of the terms, so every term must build a convex
expression, and a non-convex one is reported with its position and type:

```python
from typing import Literal

from opoise import ObjectiveBase, register_objective
from opoise.objectives import risk as objective_risk


@register_objective
class HalfRisk(ObjectiveBase):
    category = objective_risk.CATEGORY
    type: Literal["half_risk"] = "half_risk"

    def build(self, ctx):
        import cvxpy as cp

        return 0.5 * cp.sum_squares(ctx.risk_parts(ctx.w, "total"))
```

A custom type is a frozen pydantic model like the built-ins, so `extra="forbid"`
and any field validator apply as usual. It is written to a workspace file and
read back while it is registered; loading a file whose tag is not registered
fails with a validation error that lists the registered tags.
`unregister_constraint` and `unregister_objective` remove a type, and a
document naming its tag no longer loads, though instances already built keep
working. Unregistering a built-in type is allowed and has the same effect.

### Holdings, benchmark and cash

- Weights are fractions of NAV and need not sum to one on input; a `Budget`
  constraint fixes the total.
- Explicit cash is an ordinary asset with zero exposures and zero specific
  variance, included in the weights. Implicit cash is the residual weight left
  unmodelled, with the budget sized to match.
- The benchmark is a weight map over the same universe with nonzero predicted
  variance, $b^{\top}\Sigma b > 0$. Predicted beta and active risk are
  undefined against a riskless or empty benchmark, so validation rejects one.

### Solvers

CLARABEL is the default and handles every problem the engine compiles; MOSEK is
optional and needs a licence of your own. [Choosing a
solver](#choosing-a-solver) gives the capability limits of each solver and
explains how options pass through.

