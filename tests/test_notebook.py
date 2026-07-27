"""The analysis notebook, checked for dead references.

The notebook is the reporting surface -- it is what turns a run into the numbers a reader
sees -- and nothing tested it. It had drifted onto `mixed`, the retired rating-scale
Gaussian model, and called two functions that no longer exist: `stage_c.choice_diagnostics`
and `mixed.fit(..., with_item=True)`. It also filtered on `term == "slope"` where
`spread_model` emits `"lambda"`, and printed a SESOI in "spread points" after A2.9.2 moved
the scale to logits.

Executing it needs a full run's artifacts, which CI does not have. What CI *can* do is
resolve every module attribute the notebook names, which is the entire class of defect that
was actually present.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "stage0_analysis.ipynb"


@pytest.fixture(scope="module")
def nb() -> dict:
    return json.loads(NB_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def code_cells(nb) -> list[str]:
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    # NEGATIVE CONTROL for every absence-check below: an empty or unparsed notebook
    # satisfies "does not contain X" for every X.
    assert len(cells) >= 10, f"only {len(cells)} code cells -- the notebook did not load"
    assert any("spread_model" in c for c in cells), "cells loaded but look wrong"
    return cells


def test_every_code_cell_parses(code_cells):
    for i, src in enumerate(code_cells):
        try:
            ast.parse(src)
        except SyntaxError as exc:                       # pragma: no cover
            pytest.fail(f"code cell {i} does not parse: {exc}")


def test_every_module_attribute_the_notebook_names_exists(code_cells):
    """The dead-reference check. `stage_c.choice_diagnostics` was the live example."""
    import importlib

    # alias -> importable module path, mirroring the notebook's import cell
    aliases = {
        "plots": "src.analysis.plots",
        "spread_model": "src.analysis.spread_model",
        "stage_pw": "src.experiments.pass_a_pairwise",
        "stage_b": "src.experiments.pass_b",
        "stage_c": "src.experiments.pass_c",
        "validity": "src.readout.validity",
    }
    modules = {a: importlib.import_module(p) for a, p in aliases.items()}

    referenced: set[tuple[str, str]] = set()
    for src in code_cells:
        for node in ast.walk(ast.parse(src)):
            if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                    and node.value.id in modules):
                referenced.add((node.value.id, node.attr))

    assert referenced, "no module attributes found -- the walk is broken, not the notebook"
    missing = sorted(f"{a}.{attr}" for a, attr in referenced
                     if not hasattr(modules[a], attr))
    assert not missing, f"notebook references nonexistent attributes: {missing}"


def test_the_notebook_does_not_use_the_retired_model_or_its_vocabulary(nb):
    """A2.9.1 replaced the model and A2.9.2 the scale; the prose has to move with them."""
    whole = json.dumps(nb)
    retired = {
        "mixed.": "the retired rating-scale Gaussian model",
        "prepare_design": "mixed's design builder",
        "contrast_table": "mixed's contrast table (spread_model.contrasts now)",
        "with_item": "mixed's item random effect -- withdrawn by A3.8",
        "choice_diagnostics": "removed from pass_c",
        "artifact_agreement": "mixed's cross-check (structure_factor_agreement now)",
        'term == "slope"': "spread_model emits lambda, not slope",
        'term == "intercept"': "spread_model emits gamma, not intercept",
        "sesoi_raw_secondary": "retired by A2.9.2 -- meaningless on a logit scale",
        "spread points": "the DV is on a logit scale",
    }
    found = {k: why for k, why in retired.items() if k in whole}
    assert not found, f"notebook still references retired things: {found}"


def test_the_terms_the_notebook_filters_on_are_the_terms_the_model_emits(code_cells):
    """Pins the specific silent-empty-table failure: a wrong term yields no rows, not an error."""
    from src.analysis.spread_model import PLANNED

    joined = "\n".join(code_cells)
    assert 'term == "lambda"' in joined
    assert 'term == "gamma"' in joined
    assert 'term="lambda"' in joined and 'term="gamma"' in joined
    # and the primary contrast is referenced through the module, never retyped as a literal
    assert "spread_model.PRIMARY" in joined
    assert '"chose - yoked"' not in joined, "PRIMARY must not be duplicated as a literal"
    assert "chose - yoked" in PLANNED


def test_the_notebook_records_the_amendment_3_decisions(nb):
    """These are the interpretive load-bearing bits; a reader must not have to know them."""
    prose = "\n".join("".join(c["source"]) for c in nb["cells"]
                      if c["cell_type"] == "markdown")
    for ref in ("A3.1", "A3.3", "A3.6", "A3.7", "A3.8"):
        assert ref in prose, f"{ref} is not explained anywhere in the notebook"
    assert "inconclusive" in prose.lower()
    assert "min_detectable_effect" in json.dumps(nb), "the MDE replaced power at the SESOI"


def test_the_notebook_reads_power_rather_than_recomputing_it(code_cells):
    """A3.1's guarantee is that the MDE was computed before Pass C existed.

    Recomputing it here would produce a second number without that ordering, so the
    notebook must read the run's block. `power.analyze` must not be called.
    """
    joined = "\n".join(code_cells)
    assert 'results.get("power"' in joined
    assert "power_mod" not in joined and "power.analyze" not in joined


def test_no_forward_pass_or_generation_can_happen_in_the_notebook(code_cells):
    """It reports; it does not run models. The measurement path stays in src/."""
    joined = "\n".join(code_cells)
    for banned in (".generate(", "load_model", "Runner(", "run_pass_c", "run_pass_b",
                   "collect_comparisons"):
        assert banned not in joined, f"notebook would run models: {banned}"
