"""Find a minimal conflicting constraint set for an infeasible problem.

The diagnosis is the deletion filter of Chinneck and Dravnieks (1991): start
from every constraint, drop one at a time, and keep the constraint out when the
rest are still infeasible on their own. The result is one conflict set, minimal
with respect to the solves that were conclusive; other conflicts may exist.

Every test is a real feasibility solve of a subset, so the diagnosis costs one
solver call per constraint plus one. A subset solve that ends in numerical
trouble instead of a clear answer is reported as inconclusive and keeps the
constraint, which can leave the set larger than a true minimal conflict.
"""
from __future__ import annotations

from .compiler import compile_problem, run
from .models import Problem
from .result import InfeasibilityReport

_INFEASIBLE = frozenset({"infeasible", "infeasible_inaccurate"})
"""Raw solver statuses that count as a proven infeasible subset."""

_CONCLUSIVE = frozenset({"infeasible", "optimal"})
"""Raw solver statuses that settle the question either way."""


def find_conflict(problem: Problem) -> InfeasibilityReport:
    """Search for a minimal set of constraints that cannot hold together.

    The filter tests every constraint name in problem order against a zero
    objective. A subset is infeasible when the solver reports infeasible (or
    infeasible with reduced accuracy); any other answer than infeasible or
    optimal, such as a solver error or reduced accuracy, is inconclusive and
    keeps the constraint in the set, with a note in the message.

    Parameters
    ----------
    problem : Problem
        The problem whose constraints should be diagnosed. Its objective is
        ignored: every test minimises zero.

    Returns
    -------
    InfeasibilityReport
        The conflict set in problem order, the number of feasibility solves
        spent and a message naming the set. When the constraints are jointly
        feasible with a zero objective the set is empty and the message says
        the infeasible status came from numerical trouble.

    Examples
    --------
    >>> from opoise import AssetBounds, Budget, Problem, RiskTerm, synthetic_data
    >>> from opoise.infeasibility import find_conflict
    >>> data = synthetic_data()
    >>> problem = Problem(universe=data.universe, risk_model=data.risk_model,
    ...                   objective=[RiskTerm(aversion=1.0)],
    ...                   constraints=[Budget(name="budget"),
    ...                                AssetBounds(name="none", lower=2.0, upper=3.0)])
    >>> report = find_conflict(problem)
    >>> report.conflict_set
    ['budget', 'none']
    """
    names = [constraint.name for constraint in problem.constraints]

    def test(subset: list[str]) -> str:
        compiled = compile_problem(problem, subset, feasibility=True)
        raw, _, _ = run(compiled.cvx, problem.settings)
        return raw

    solves = 1
    if test(names) not in _INFEASIBLE:
        return InfeasibilityReport(
            conflict_set=[],
            solves=solves,
            message=(
                "all constraints are jointly feasible with a zero objective; "
                "the infeasible status came from numerical trouble in the full "
                "problem (try another solver or rescale)"
            ),
        )
    keep = list(names)
    uncertain = False
    for name in names:
        candidate = [other for other in keep if other != name]
        solves += 1
        raw = test(candidate)
        if raw in _INFEASIBLE:
            keep = candidate
        elif raw not in _CONCLUSIVE:
            uncertain = True
    tail = (
        "each one is needed for this conflict (other conflicts may exist)"
        if not uncertain
        else "the set may not be minimal because some subset solves were inconclusive"
    )
    return InfeasibilityReport(
        conflict_set=keep,
        solves=solves,
        message=f"infeasible: constraints [{', '.join(keep)}] cannot all hold together; {tail}",
    )
