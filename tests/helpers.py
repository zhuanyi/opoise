"""Shared synthetic fixtures for the acceptance tests (no real data)."""
import numpy as np

from opoise import Problem, SolverSettings, synthetic_data

DATA = synthetic_data(n_assets=20, n_factors=3, n_sectors=4, seed=0)
IDS = [a.id for a in DATA.universe.assets]
B, F, D = DATA.risk_model.arrays()
SIGMA = B @ F @ B.T + np.diag(D)
ONE = np.ones(len(IDS))
BENCH = np.array([DATA.benchmark.get(i, 0.0) for i in IDS])
W_MV = np.linalg.solve(SIGMA, ONE)
W_MV = W_MV / W_MV.sum()
BETA = SIGMA @ BENCH / (BENCH @ SIGMA @ BENCH)
SECTOR = np.array([a.attributes["sector"] for a in DATA.universe.assets])
# Clarabel defaults (1e-8) leave binding rows up to ~1e-6 off; 1e-9 gives >=100x margin on every assertion
# (measured). 1e-11 and tighter turn SOC solves into optimal_inaccurate.
TIGHT = {"tol_gap_abs": 1e-9, "tol_gap_rel": 1e-9, "tol_feas": 1e-9}


def vec(d):
    return np.array([d.get(i, 0.0) for i in IDS])


def as_dict(x):
    return {i: float(v) for i, v in zip(IDS, x)}


def vol(w):
    return float(np.sqrt(w @ SIGMA @ w))


def te(w):
    return vol(w - BENCH)


def problem(objective, constraints=(), **kw):
    kw.setdefault("benchmark", DATA.benchmark)
    kw.setdefault("settings", SolverSettings(solver_options=TIGHT))
    return Problem(universe=DATA.universe, risk_model=DATA.risk_model,
                   objective=list(objective), constraints=list(constraints), **kw)


def row(result, label):
    return next(r for r in result.diagnostics.constraints if r.row == label)
