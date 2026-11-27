# Formulation

This page fixes the notation and derives the risk decomposition that every
report uses, then documents each objective term and constraint with its
formula, its convexity class and how it compiles.

## Notation and factor form

A problem invests a universe of $n$ assets. The decision variable is the weight
vector $w \in \mathbb{R}^n$, with weights as fractions of net asset value and
the universe order as the coordinate order. A benchmark is a vector $b$ on the
same universe. Where a term is measured against the benchmark it uses the
**active weights**

$$
x = \begin{cases} w & \text{absolute} \\ w - b & \text{active} \end{cases}
$$

The risk model is an input: exposures $B \in \mathbb{R}^{n \times k}$, a
symmetric positive semidefinite factor covariance $F \in \mathbb{R}^{k \times k}$
and a non-negative specific variance $d \in \mathbb{R}^n$. They define the asset
covariance

$$
\Sigma = B F B^{\top} + \operatorname{diag}(d).
$$

The engine never forms $\Sigma$. With the eigen-factor $L$ such that
$F = L L^{\top}$ (negative eigenvalues clipped to zero), every variance is
written as a squared norm of an affine map:

$$
x^{\top} \Sigma x = \lVert L^{\top} B^{\top} x \rVert_2^2 + \lVert \sqrt{d} \circ x \rVert_2^2
= \lVert \operatorname{parts}(x) \rVert_2^2,
\qquad
\operatorname{parts}(x) = \begin{bmatrix} L^{\top} B^{\top} x \\ \sqrt{d} \circ x \end{bmatrix}.
$$

Risk is the standard deviation $\sigma(x) = \sqrt{x^{\top} \Sigma x}$ in the
units of the risk model, so an annual covariance gives an annual volatility.
Predicted beta against the benchmark is

$$
\beta = \frac{\Sigma b}{b^{\top} \Sigma b},
$$

a data vector; the portfolio beta is $\beta^{\top} w$ and the benchmark's own
beta is $1$. The denominator must be positive, which the problem validator
checks at construction.

## Risk decomposition

For $\sigma = \sqrt{x^{\top} \Sigma x} > 0$ the Euler decomposition splits the
total risk into additive contributions; for $\sigma = 0$ every contribution is
reported as $0$. Writing $y = B^{\top} x$ for the portfolio's factor exposures,
the three levels are

$$
\text{asset}_i = \frac{x_i (\Sigma x)_i}{\sigma},
\qquad
\text{factor}_f = \frac{y_f (F y)_f}{\sigma},
\qquad
\text{specific} = \frac{\sum_i d_i x_i^2}{\sigma}.
$$

Each level sums to the total risk. For assets, $\sum_i x_i (\Sigma x)_i = x^{\top} \Sigma x = \sigma^2$, so the sum is
$\sigma$. For factors and the specific part,
$x^{\top} \Sigma x = y^{\top} F y + \sum_i d_i x_i^2$, so the factor contributions plus the
specific contribution also sum to $\sigma$. The Euler contributions add, but
`factor_risk` and `specific_risk` are standard deviations and add in
quadrature, $\sigma^2 = factor\_risk^2 + specific\_risk^2$, not linearly. A
**group** contribution (a sector, a country) is the sum of its assets'
contributions, with assets that do not carry the attribute collected under
`UNASSIGNED`; it is the group's Euler
contribution and sums with the other groups of the same attribute to $\sigma$.

Every number is computed in numpy from the final weights, in factor form, by
`risk_report`; it does not depend on solver internals, so the same function
analyses any portfolio:

```python
import numpy as np

from opoise import Problem, RiskTerm, predicted_beta, risk_report, synthetic_data

data = synthetic_data()
problem = Problem(universe=data.universe, risk_model=data.risk_model, benchmark=data.benchmark,
                  objective=[RiskTerm(aversion=1.0)])
B, F, d = data.risk_model.arrays()
sigma = B @ F @ B.T + np.diag(d)
w = np.linalg.solve(sigma, np.ones(len(data.universe.assets)))
w = w / w.sum()
report = risk_report(problem, {asset.id: float(value) for asset, value in zip(data.universe.assets, w)})
print(round(sum(report.by_asset.values()) - report.total_risk, 12))
print(round(sum(report.by_factor.values()) + report.specific_contribution - report.total_risk, 12))
print(round(predicted_beta(problem, data.benchmark), 12))
```

The first two lines are zero up to rounding: the contributions add up to the
total risk at every level, and the benchmark's beta is one.

## Objective

The objective is a sum of terms, one positive aversion times one variance each;
minimising it is the whole of a problem's preference. Every term is convex, so
any sum of them stays convex, and the minimisation has no factor of one half.

### `RiskTerm`

Fields: `type` (constant "risk"), `measure` ("total", "factor" or "specific"),
`active`, `aversion` (strictly positive). With $x_t = w$ or $x_t = w - b$ when
`active`, and $\operatorname{parts}_t$ keeping the factor block, the specific
block or both according to `measure`:

$$
\min_w \sum_t a_t \lVert \operatorname{parts}_t(x_t) \rVert_2^2
= \sum_t a_t \, x_t^{\top} \Sigma_t x_t
$$

Convexity: convex QP. Each term is a positive multiple of a squared affine map,
so it is QP-representable, and OSQP is allowed as long as no cone row is
present.

A term compiles to `aversion * cp.sum_squares(risk_parts(x, measure))`; the
measure picks the block, so `total` is the factor block and the specific block
stacked, and the sum of a `factor` term and a `specific` term with the same
aversion equals one `total` term.

## Constraints

Every constraint is compiled to one row per element: equalities where both
bounds coincide, inequalities elsewhere, each emitted as a single vector
constraint. Rows are reported back with their value, bounds, slack and binding
flag.

### `Budget`

Fields: `total` (default 1.0).

$$
\mathbf{1}^{\top} w = T
$$

Convexity: affine equality, so the row is an LP constraint and any solver in
the settings accepts it.

It compiles to one scalar equality on the sum of the weights; `total=0` gives a
dollar-neutral problem.

### `AssetBounds`

Fields: `lower`, `upper`, `per_asset` (a map from asset id to `Bounds`),
`active`.

$$
l_i \le x_i \le u_i, \qquad x = w \ \text{or} \ x = w - b,
$$

with a `per_asset` entry replacing both default sides for that asset.

Convexity: affine, so an LP constraint. Each side is a separate inequality;
a missing default and no override leaves that asset out of the row, so only
bounded assets appear in the report.

It compiles to one vector row over the assets that have at least one finite
bound after the overrides, in universe order, labelled `name[asset]`.

### `Turnover`

Fields: `limit` (at least 0).

$$
\lVert w - w_0 \rVert_1 \le \tau
$$

with $w_0$ the initial holdings (missing assets are 0) and $\tau$ the limit.
Turnover is **two-way**: one-way turnover is half of $\lVert w - w_0 \rVert_1$.

Convexity: convex, LP. An L1 norm is not affine, but its epigraph is
($t_i \ge \pm(w_i - w_{0,i})$, $\sum_i t_i \le \tau$), which the solver builds
from the row.

It compiles to one convex scalar row, `cp.norm1(w - w0)`, bounded above by
`limit`; the reported value is the two-way turnover actually taken.

### `GroupBounds`

Fields: `attribute`, `bounds` (a map from label to `Bounds`), `active`.

$$
l_g \le m_g^{\top} x \le u_g
$$

with $m_g$ the 0/1 mask of the assets carrying label $g$ of the attribute.

Convexity: affine, LP. The masks are data, so each element is a fixed linear
functional of the weights; nothing about the group is estimated.

It compiles to one affine vector row, the stacked group masks applied to the
weights or active weights, one element per label in `bounds` order, labelled
`name[label]`; an asset without the attribute belongs to no group and, in the
risk report, to `UNASSIGNED`.

### `GroupRiskCap`

Fields: `attribute`, `caps` (per label, strictly positive), `measure`,
`active`.

$$
\lVert \operatorname{parts}_t(m_g \circ x) \rVert_2 \le c_g
$$

Convexity: second-order cone, convex. It caps the **stand-alone** risk of the
group sub-portfolio: the group's own covariance, with every other asset
weighted zero. It ignores the group's covariance with the rest of the
portfolio, so it is not the group's contribution to the portfolio's total
risk.

It compiles to one convex vector row whose elements are
`cp.norm(risk_parts(mask * x, measure), 2)`, one per label in `caps` order and
labelled `name[label]`. A problem containing one is not a QP, so OSQP refuses
it.

### `RiskCap`

Fields: `measure`, `active`, `limit` (strictly positive).

$$
\lVert \operatorname{parts}_t(x) \rVert_2 \le c
\quad \Longleftrightarrow \quad
x^{\top} \Sigma_t x \le c^2,
$$

and with `active=True` this is a tracking-error cap on $w - b$.

Convexity: second-order cone, convex, and an upper bound only. A floor, or an
equality (a tracking-error target), is reverse-convex and is not offered.

It compiles to one convex scalar row, `cp.norm(risk_parts(x, measure), 2)`
bounded above by `limit`; a problem containing one is not a QP, so OSQP refuses
it and `solve` raises `SolverCapabilityError`.

### `BetaBounds`

Fields: `lower`, `upper`.

$$
l \le \beta^{\top} w \le u,
\qquad
\beta = \frac{\Sigma b}{b^{\top} \Sigma b}
$$

Convexity: affine, LP. The predicted beta of each asset is data, computed once
from the benchmark, so the portfolio beta is a fixed linear functional of the
weights. It uses the absolute weights $w$, never the active weights.

It compiles to one affine scalar row; the problem validator requires a
benchmark with positive predicted variance, so the constraint always has its
$\beta$. Its value in the report is the portfolio beta.

### `FactorExposureBounds`

Fields: `bounds` (a map from factor id to `Bounds`), `active`.

$$
l_f \le (B^{\top} x)_f \le u_f
$$

Convexity: affine, LP. One column of the exposure matrix is fixed data, so each
element is a linear functional of the weights or active weights.

It compiles to one affine vector row over the requested factor columns of $B$,
in `bounds` order, labelled `name[factor]`; a factor id the risk model does
not have fails problem validation.

### `LinearBounds`

Fields: `coefficients` (a map from asset id to a coefficient), `lower`,
`upper`.

$$
l \le a^{\top} w \le u
$$

Convexity: affine, LP. It uses the absolute weights $w$; the coefficient vector
$a$ is data with 0 for every asset not named.

It compiles to one affine scalar row: the coefficients placed in universe
order and applied to the weights. It is the general form of a single
hand-made exposure, a funding split or a cap on one basket.

### `GroupRiskShareCap`

Fields: `attribute`, `caps` (per label, strictly positive), `active`.

$$
s_g(x) = \frac{x_g^{\top} \Sigma x}{x^{\top} \Sigma x} \le c_g,
\qquad x_g = m_g \circ x, \qquad x = w \text{ or } w - b
$$

Convexity: **non-convex**. With $P_g = \operatorname{diag}(m_g)$ the
constraint is $q_g(x) = x^{\top}\left(\tfrac{1}{2}(P_g\Sigma + \Sigma P_g) -
c_g\Sigma\right) x \le 0$, an indefinite quadratic form, so cvxpy rejects it
as not DCP and it goes through a sequential convex approximation instead.

The quantity $s_g$ is the group's Euler contribution of the risk decomposition
divided by the total risk, so over every label of the attribute (assets without
it under `UNASSIGNED`) the shares sum to 1 whenever $x^{\top}\Sigma x > 0$. A
share is not bounded to $[0, 1]$: offsetting positions can push one label above
1 and another below 0, which is why a cap only has to be positive. When the
portfolio has zero risk the share is undefined; it is reported as 0 and the
constraint counts as satisfied. Capping every label with caps that sum to less
than 1 can only be met at $x = 0$.

**The convex-concave step.** Write $\lVert u \rVert_\Sigma^2 = u^{\top}\Sigma u$
and $h = (I - P_g)x$. Polarisation gives

$$
x_g^{\top}\Sigma x = \tfrac{1}{4}\lVert (I + P_g)x \rVert_\Sigma^2 -
\tfrac{1}{4}\lVert h \rVert_\Sigma^2 ,
$$

so $q_g$ is a convex term minus a convex term:

$$
q_g(x) = \tfrac{1}{4}\lVert (I + P_g)x \rVert_\Sigma^2 -
\left[ \tfrac{1}{4}\lVert h \rVert_\Sigma^2 + c_g \lVert x \rVert_\Sigma^2 \right].
$$

Each $\lVert \cdot \rVert_\Sigma^2$ lies above its tangent at the previous
iterate $x_k$, so replacing the bracket by its tangent gives a convex
quadratic that dominates $q_g$, with equality at $x = x_k$:

$$
\tilde{q}_g(x; x_k) = \tfrac{1}{4}\lVert (I + P_g)x \rVert_\Sigma^2 +
\gamma_k + \nabla_k^{\top}(x - x_k),
\qquad \gamma_k = -\tfrac{1}{4} h_k^{\top}\Sigma h_k - c_g x_k^{\top}\Sigma x_k ,
$$

$$
\nabla_k = -\tfrac{1}{2}(I - P_g)\Sigma h_k - 2 c_g \Sigma x_k ,
\qquad h_k = (I - P_g)x_k .
$$

Three properties follow, and the tests pin all three: $\{\tilde{q}_g \le 0\}
\subset \{q_g \le 0\}$, an inner approximation, so every iterate satisfies
the true caps within solver tolerance (subproblems that return status
`optimal_inaccurate` are accepted, so a cap can be missed by an amount of the
order of the solver tolerance); from the second iteration on the previous
iterate is feasible for the new subproblem, so the objective does not increase
within solver tolerance; and, under a constraint qualification such as LICQ, a
fixed point is a KKT point of the non-convex problem (Lipp and Boyd 2016,
"Variations and extension of the convex-concave procedure", *Optimization and
Engineering* 17). The qualification is not automatic here: the gradient
$\nabla q_g = (P_g\Sigma + \Sigma P_g - 2 c_g \Sigma)x$ can vanish at a
nonzero portfolio, so the guarantee can fail exactly where a cap binds. The
result is then a local solution with a `converged` flag, not a certified
optimum, and the convergence test (largest weight change below `sca_tol`, an
absolute tolerance) is a stopping rule, not a proof of the KKT conditions. It
is a local method: the answer depends on the starting point and is not
guaranteed to be the global optimum.

It compiles to one convex vector row per cap set: the tangent above evaluated
at the previous iterate, bounded above by 0, labelled `name[label]`. The
compiler needs that linearisation point, so `compile_problem` takes
`w_ref` and a direct call without one fails; `solve` runs the iteration,
starting from the problem without the share caps, up to
`SolverSettings.sca_max_iterations` and stopping when no weight moves by more
than `sca_tol`. The first subproblem can report a non-optimal status even when
the non-convex problem is feasible, because the inner approximation at the
starting point can be too tight, so that status is not a proof of
infeasibility. The reported row value and `result.sca.shares` are the true
shares of the final weights, never the approximation. Every subproblem has a
convex quadratic constraint and is not a QP, so OSQP refuses the problem.

## Infeasibility diagnosis

When a solve returns `infeasible`, `solve` can explain why. With
`diagnose_infeasibility=True` (the default) it runs a deletion filter
(Chinneck and Dravnieks 1991) over the constraint names, in problem order. Each
test minimises zero subject to a subset of the constraints; a subset is
infeasible when the solver says so. Starting from every constraint, the filter
removes one constraint at a time and keeps it out when the rest are still
infeasible:

$$
K_0 = \{1, \dots, m\}, \qquad
K_{i} = \begin{cases} K_{i-1} \setminus \{i\} & \text{if } K_{i-1} \setminus \{i\} \text{ is infeasible} \\ K_{i-1} & \text{otherwise} \end{cases}
$$

The final `conflict_set` is the set that remains: every member is needed for
this conflict (removing any one of them makes the rest feasible), and the cost
is one feasibility solve per constraint plus one, reported as `solves`.

Limits worth knowing when reading the report:

- One conflict is reported. A problem can have several disjoint conflicts, and
  the filter finds the one its order runs into.
- Minimality holds only for conclusive subset solves. A subset that ends in a
  solver error or reduced accuracy is inconclusive; the constraint stays in
  the set and the message says the set may not be minimal.
- If the same constraints are jointly feasible with a zero objective (the
  reported infeasibility was numerical trouble in the full problem with its
  objective), the conflict set is empty and the message says so; try another
  solver or rescale.
