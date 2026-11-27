# Changelog

All notable changes to this project are documented in this file. The format is
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Declarative problem models: universe and assets with grouping attributes, a
  factor risk model (exposures, factor covariance, specific variance), solver
  settings and the problem container, all validated at construction.
- Constraint models: budget, asset bounds, turnover, group bounds, group
  stand-alone risk caps, beta bounds, risk caps, factor exposure bounds and
  generic linear bounds.
- Objective terms: weighted total, factor or specific variance, absolute or
  active.
- A deterministic synthetic data generator for examples and tests.
- Workspace files: `save_workspace` and `load_workspace` write and read a
  problem as a portable JSON or YAML document (format `opoise-workspace`,
  schema version 1), with a `WorkspaceLoader` protocol for formats of your
  own.
- A `RiskModelProvider` protocol for connecting a risk data source; the
  package still never fetches data.
- Result models: `Result` with its status, weights, trades, objective value,
  solve time and message, plus `Diagnostics`, `RiskReport`,
  `ConstraintRowReport` and `InfeasibilityReport`.
- Risk analytics in numpy: `risk_report` decomposes the total or active risk
  into Euler contributions by asset, by factor and by group (assets without the
  attribute under `UNASSIGNED`), and `predicted_beta` gives the beta of any
  weight vector against the benchmark, without a solver.
- Documentation: a `formulation` page with the notation, the factor form of
  the risk model and the derivation of the risk decomposition, and a
  "Reading a result" section in the user guide.
- Solving: `solve` compiles a problem, calls the chosen solver and returns a
  `Result` with the weights, the trades against the initial holdings, the
  objective value, the solve time, the risk decomposition and a constraint
  report with slack, binding and violated flags per row.
- The compiler: `CompileContext` slices the risk model to the universe and
  factorises the factor covariance so a variance is a squared norm;
  `compile_problem` builds a DCP cvxpy problem, exposing each logical
  constraint as a `Row` and turning its bounds into cvxpy constraints with
  `emit`; `run` is the single place a solver is called and reports a rejected
  solver option as an error status instead of an exception.
- Row builders for `Budget`, `AssetBounds` and `RiskCap`; the other constraint
  types compile in the next step.
- `solve`, `compile_problem` and `CompiledProblem` are lazy exports: importing
  `opoise` still does not import cvxpy.
- Documentation: the objective and the first three constraints on the
  formulation page, and a "Building and solving a problem" section in the user
  guide covering the solver choice, solver options and the statuses.
- Row builders for `Turnover`, `GroupBounds`, `GroupRiskCap`, `BetaBounds`,
  `FactorExposureBounds` and `LinearBounds`: every constraint type now
  compiles to a row and solves.
- Documentation: formulation entries for the six remaining constraints, with
  their formula, convexity class and compilation, and a "Constraint cookbook"
  section in the user guide with one runnable example per constraint type.
- Infeasibility diagnosis: an infeasible solve now runs a deletion filter over
  the constraint names, minimising zero on each subset, and returns an
  `InfeasibilityReport` with the conflict set, the number of feasibility
  solves and a message that names the constraints and warns when a subset
  solve was inconclusive. `SolverSettings(diagnose_infeasibility=False)`
  skips the extra solves.
- Documentation: a formulation section and a user-guide section on
  infeasibility diagnosis, and a README with install, development setup and a
  runnable quickstart. The user guide's "Extending Opoise" section covers risk
  model providers, file loaders and solver options.
- Sector share-of-risk caps (variant B): `GroupRiskShareCap` bounds a group's
  Euler share of the total or active risk. The constraint is not convex, so it
  compiles to a tangent at a linearisation point and solves by sequential
  convex approximation (Lipp and Boyd 2016) in the new `opoise.sca` module: the
  result carries an `ScaReport` with the convergence flag, the iteration count,
  the last weight change and the true shares, and running out of
  `SolverSettings.sca_max_iterations` reports `optimal_inaccurate` with
  `converged=False` instead of a silent optimum. `SolverSettings` gains
  `sca_max_iterations` and `sca_tol`; OSQP is refused because the subproblems
  have convex quadratic constraints.
- Documentation: a formulation entry deriving the convex-concave step, the
  inner-approximation and convergence properties and the compiled row, and a
  "Sector share-of-risk caps" section in the user guide on variant A versus B,
  the two solver settings and how to read `result.sca`.
- Documentation: the final home page (what the engine is, install with uv, a
  quickstart and links to every guide page and the changelog), a "Documentation"
  section in the README, and a `test_docs.py` gate that checks the pages and the
  mkdocs nav, the module coverage of the API reference, a formula and a
  `Convexity:` note for every objective term and constraint, and the changelog
  format and its links. The strict site build runs clean.
- Documentation: `result.sca` is now documented as a `ScaReport` only when the
  sequential approximation ran to a result; the user guide and the `Result`
  docstring say it is `None` for a problem without a share cap and on the
  failure paths (the solve without the share caps is not optimal, so that
  result is returned with a `"without the share caps: "` prefix; a subproblem
  returns a solver error; an iteration subproblem is not optimal and the
  result has status `error`).
- Documentation: the math review corrections. The shares sum to 1 only when
  total risk is positive; every iterate satisfies the true caps within solver
  tolerance because `optimal_inaccurate` results are accepted; the
  fixed-point/KKT statement names its constraint qualification and the
  gradient that can vanish where a cap binds; the convergence test is a
  stopping rule, not a proof; risks add in quadrature, not linearly; and the
  first subproblem can be infeasible for a feasible problem.
- Documentation: the extension notes for risk models, file loaders and solver
  options now live in the user guide's "Extending Opoise" section.
- A constraint registry: every constraint type is registered under its `type`
  tag and a category. A project declares a `Category` with
  `register_category`, subclasses `ConstraintBase` with a `build` method and
  decorates the class with `@register_constraint`; the new type then works in a
  problem, a JSON document and a workspace file while it is registered.
- Constraint categories, one module per category under `opoise.constraints`:
  `BUDGET`, `HOLDING`, `TRADE`, `TURNOVER`, `RISK`, `EXPOSURE` and `LINEAR`.
  The package imports every module in sorted order, so a new built-in category
  is one file. `categories()` and `constraint_types()` list what is
  registered.
- `TradeBounds`: per-asset lower and upper bounds on the trade against the
  initial holdings, w - w0, with per-asset overrides, compiled as one affine
  vector row.
- An objective registry: every objective term is registered under its `type`
  tag and a category. A project declares a `Category` with
  `register_category`, subclasses `ObjectiveBase` with a `build` method and
  decorates the class with `@register_objective`; the term then works in a
  problem, a JSON document and a workspace file while it is registered.
- Objective categories, one module per category under `opoise.objectives`:
  `RISK`. The package imports every module in sorted order, so a new built-in
  category is one file. `categories("objective")` and `objective_types()` list
  what is registered.
- Objective-term validation: the compiler raises `OpoiseError` naming the term
  index and type when a term's `build()` does not return a scalar cvxpy
  expression, or returns a non-convex one.

### Changed

- Constraints live in one module per category instead of one module;
  `opoise.constraints`, `opoise.context.Row` and every existing import path
  keep working.
- The compiler asks each constraint for its row through `build(ctx)` instead of
  dispatching on a builder table, and reports a malformed or non-DCP row with
  the constraint name and its type.
- A constraint declares whether it needs a linearisation point with
  `requires_sca`, which selects the sequential approximation without comparing
  type tags.
- Problem dumps keep the type tag and every subclass field of a constraint; an
  unknown tag raises a validation error that lists the registered tags.
  `Problem.add_constraint` and `Problem.remove_constraint` return a new
  validated problem.
- Constraint lists reject a dict without a `type` tag and a bare
  `ConstraintBase`, and accept a tuple of entries as well as a list. A row
  that is not one-dimensional with one element per label, or a
  `requires_sca` type other than `GroupRiskShareCap`, now fails with a
  message that names the constraint instead of a later crash.
- Objective terms live in one module per category instead of one module;
  `opoise.objectives` and every existing import path keep working.
- The compiler minimises the sum of the objective terms through `build(ctx)`
  instead of summing one fixed formula. A term that builds a non-convex
  expression fails with a message that names its position and type.
- The objective list dispatches tagged dicts through the registry like the
  constraint list: an unknown tag, a dict without a tag and a bare
  `ObjectiveBase` are rejected at validation, and a dump keeps the tag and
  every subclass field of each term.

### Fixed

- `docs/changelog.md` is a copy of the root file instead of a symlink, so it
  renders on GitHub; a test keeps the two identical.
- Documentation describes only the current release; forward-looking references
  were removed.
