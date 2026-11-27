"""Documentation completeness gate (final work package). The strict site build checks that the docs render and link;
this checks that nothing was left out: every module is in the API reference, every constraint and objective term
has a formulation entry with a convexity note, the changelog follows Keep a Changelog, the pages exist."""
import importlib
import inspect
import pkgutil
import re
from pathlib import Path

import opoise
from opoise import constraints, objectives

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
PAGES = ["index.md", "concepts.md", "formulation.md", "user-guide.md", "reference.md", "changelog.md"]


def text(name):
    return (DOCS / name).read_text(encoding="utf-8")


def public_classes(module, base):
    modules = [module] + ([importlib.import_module(f"{module.__name__}.{m.name}")
                           for m in pkgutil.iter_modules(module.__path__)]
                          if hasattr(module, "__path__") else [])
    return sorted({c.__name__ for m in modules for _, c in inspect.getmembers(m, inspect.isclass)
                   if issubclass(c, base) and c is not base
                   and (c.__module__ == module.__name__ or c.__module__.startswith(module.__name__ + "."))})


def test_pages_exist_with_a_title_and_mkdocs_nav_lists_them():
    nav = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    for p in PAGES:
        t = text(p)
        assert re.match(r"# \S", t) and len(t.strip()) > 80, p
        assert p in nav, f"{p} missing from mkdocs.yml nav"


def test_every_module_is_in_the_api_reference():
    mods = [m.name for m in pkgutil.iter_modules(opoise.__path__)]
    assert len(mods) >= 14
    missing = [m for m in mods if f"::: opoise.{m}" not in text("reference.md")]
    assert not missing, f"modules missing from docs/reference.md: {missing}"


def test_every_constraint_and_objective_term_has_a_formulation_entry():
    from pydantic import BaseModel
    names = public_classes(constraints, BaseModel) + public_classes(objectives, BaseModel)
    names = [n for n in names if n not in ("Bounds", "ConstraintBase", "ObjectiveBase")]
    assert {"Budget", "GroupRiskShareCap", "RiskTerm", "GroupRiskCap"} <= set(names) and len(names) == 12
    body = text("formulation.md")
    sections = re.split(r"^### ", body, flags=re.M)[1:]
    for n in names:
        sec = next((s for s in sections if s.startswith(f"`{n}`")), None)
        assert sec is not None, f"docs/formulation.md has no '### `{n}`' section"
        assert re.search(r"\$\$.+\$\$", sec, flags=re.S), f"{n}: no display formula"
        assert re.search(r"Convexity:", sec), f"{n}: no 'Convexity:' note"


def test_changelog_follows_keep_a_changelog():
    t = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert t.startswith("# Changelog") and "keepachangelog.com" in t and "## [Unreleased]" in t
    assert re.search(r"^### Added", t, flags=re.M)
    assert "group_risk_share_cap" in t or "GroupRiskShareCap" in t
    assert (DOCS / "changelog.md").read_text(encoding="utf-8") == (ROOT / "CHANGELOG.md").read_text(encoding="utf-8"), (
        "docs/changelog.md must be a copy of CHANGELOG.md: edit the root file, then copy it to docs/"
    )

