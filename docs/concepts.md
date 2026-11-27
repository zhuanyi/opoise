# Concepts

## Universe and assets

A problem invests a **universe**: an ordered list of assets, each with an id and
free-form string attributes such as a sector or a country. The order of the
universe is the order of every weight, every report row and every array. Ids
are unique, 1 to 64 characters and exclude whitespace and brackets, because
report rows are labelled `name[id]`.

Attributes define **groups**. A constraint that refers to a sector sums or caps
the assets carrying that sector label. When a report groups a portfolio by an
attribute, assets without the attribute appear under `UNASSIGNED`, which is
reserved and may not be used as an attribute value.

## The factor risk model

The risk model is an input, not something the engine estimates. It holds asset
exposures $B$ (assets by factors), a factor covariance $F$ (factors by
factors) and a specific variance $d$ per asset. They define the asset
covariance

$$
\Sigma = B F B^{\top} + \operatorname{diag}(d)
$$

on the rows of the model. Risk is the standard deviation in the units of
$\Sigma$: annualise all three inputs consistently (or none), and remember that
caps are on risk while objectives use variance.

The model may cover more assets than the universe; a problem uses the universe
rows in universe order. The engine never forms the $n \times n$ matrix
$\Sigma$: every risk quantity is computed in factor form, at cost linear in the
number of assets for a fixed number of factors.

## Weights, benchmark and active weights

Weights $w$ are fractions of net asset value. A benchmark $b$ lives on the same
universe, and **active weights** are $x = w - b$. An objective term or a
constraint with `active: true` is measured on $x$ instead of $w$; then the
problem needs a benchmark, and a benchmark with zero predicted variance is
rejected because active risk would be undefined.

## Conventions

- Everything is expressed as a minimisation.
- A variance term contributes $\text{aversion} \cdot x^{\top} \Sigma x$ with no
  factor of one half.
- Weights are fractions of NAV; a fully invested long-only portfolio sums to 1.
- Turnover is **two-way**: $\sum_i |w_i - w0_i|$ against the initial holdings;
  one-way turnover is half of it.
- Cash is an ordinary asset if you want one: add it to the universe.

## Declarative problems

A problem is a frozen, validated pydantic model with extra fields forbidden. A
typo fails at construction instead of being ignored, and an unknown constraint
type fails the same way. Immutability is shallow: fields cannot be reassigned,
but the contents of list and dict fields are not frozen, so do not mutate them
after construction. The same documents can be written to JSON or YAML
workspace files, so a problem definition is portable data rather than code.
