import json

import pytest

from opoise import (AssetBounds, BetaBounds, Bounds, Budget, FactorExposureBounds, GroupBounds,
                              GroupRiskCap, LinearBounds, NativeWorkspaceLoader, RiskCap, RiskTerm,
                              SolverSettings, Turnover, WorkspaceFormatError, WorkspaceLoader,
                              load_workspace, save_workspace)
from helpers import IDS, problem


def full_problem():
    return problem(
        [RiskTerm(aversion=2.0), RiskTerm(measure="specific", active=True, aversion=0.5)],
        [Budget(name="budget"),
         AssetBounds(name="long_only", lower=0.0, upper=0.2, per_asset={IDS[0]: Bounds(upper=0.05)}),
         Turnover(name="turnover", limit=0.3),
         GroupBounds(name="sectors", attribute="sector", active=True, bounds={"S0": Bounds(upper=0.3)}),
         GroupRiskCap(name="sector_risk", attribute="sector", caps={"S1": 0.05}),
         BetaBounds(name="beta", lower=0.9, upper=1.1),
         RiskCap(name="te", active=True, limit=0.04),
         FactorExposureBounds(name="fx", bounds={"STYLE1": Bounds(lower=-0.1, upper=0.1)}),
         LinearBounds(name="lin", coefficients={IDS[1]: 1.0, IDS[2]: -1.0}, lower=0.0)],
        initial_holdings={IDS[0]: 0.5, IDS[3]: 0.5},
        settings=SolverSettings(solver="SCS", solver_options={"max_iters": 5000}))


@pytest.mark.parametrize("suffix", [".json", ".yaml", ".yml"])
def test_round_trip(tmp_path, suffix):
    p = full_problem()
    path = tmp_path / f"ws{suffix}"
    save_workspace(p, path)
    assert load_workspace(path) == p


def test_native_file_shape(tmp_path):
    path = tmp_path / "ws.json"
    save_workspace(full_problem(), path)
    raw = json.loads(path.read_text())
    assert list(raw) == ["format", "schema_version", "problem"]
    assert raw["format"] == "opoise-workspace" and raw["schema_version"] == 1
    assert [c["type"] for c in raw["problem"]["constraints"]][:3] == ["budget", "asset_bounds", "turnover"]


def test_bad_files(tmp_path):
    (tmp_path / "a.json").write_text("{not json")
    with pytest.raises(WorkspaceFormatError, match="a.json"):
        load_workspace(tmp_path / "a.json")
    (tmp_path / "b.json").write_text(json.dumps({"format": "other", "schema_version": 1, "problem": {}}))
    with pytest.raises(WorkspaceFormatError, match="format"):
        load_workspace(tmp_path / "b.json")
    (tmp_path / "c.custom").write_text("x")
    with pytest.raises(WorkspaceFormatError, match="no loader"):
        load_workspace(tmp_path / "c.custom")
    with pytest.raises(WorkspaceFormatError, match="suffix"):
        save_workspace(full_problem(), tmp_path / "d.txt")


def test_custom_loader_is_used(tmp_path):
    p = full_problem()

    class FakeCustomLoader:
        def can_load(self, path):
            return path.suffix == ".custom"

        def load(self, path):
            return p

    assert isinstance(FakeCustomLoader(), WorkspaceLoader) and isinstance(NativeWorkspaceLoader(), WorkspaceLoader)
    (tmp_path / "c.custom").write_text("x")
    assert load_workspace(tmp_path / "c.custom", loaders=[NativeWorkspaceLoader(), FakeCustomLoader()]) is p
