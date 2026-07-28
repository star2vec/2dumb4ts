"""Generate the status report from artifacts. Do not type these numbers.

    python scripts/status.py                 # print
    python scripts/status.py --write         # also write STATUS.md and commit-ready summaries

Four retractions have occurred (see RETRACTIONS.md) and three involved a number that
was transcribed or transferred by hand. The fix is not more care, it is making
reported numbers untypeable: everything below is read out of the parquet and JSON
artifacts, so a figure that appears here exists in a provenance-stamped file.

Also writes per-run summary JSON to `results/`, which IS tracked in git. `artifacts/`
is gitignored, so without this not one reported number is greppable in a repository
whose entire premise is provenance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

ART = REPO / "artifacts"
OUT = REPO / "results"


def _load_results() -> list[dict]:
    """Every results_*.json under artifacts/, with its stage and model."""
    rows = []
    for p in sorted(ART.rglob("results_*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        parts = p.relative_to(ART).parts
        d["_stage"] = parts[0] if parts else "?"
        d["_path"] = str(p.relative_to(REPO))
        d["_mtime"] = p.stat().st_mtime
        # Order on the timestamp RECORDED IN THE ARTIFACT, not the file's mtime. A git
        # checkout rewrites mtimes, so an mtime sort can silently invert on a fresh clone
        # -- on the run machine, which is where the reported numbers come from. mtime is
        # only a fallback for artifacts written before provenance carried created_utc.
        d["_when"] = _get(d, "provenance", "created_utc") or ""
        rows.append(d)
    return sorted(rows, key=lambda r: (r["_when"], r["_mtime"]), reverse=True)


def _newest_per_model(rows: list[dict]) -> list[dict]:
    """One result per (stage, model), the newest.

    `_load_results` sorts newest-first and every table de-duplicates, but the --write loop
    did not: it iterated ALL results newest-to-oldest into a filename keyed on
    (stage, model), so last-write-wins emitted the OLDEST. With stale artifact directories
    present -- three analysis results for gemma, two for llama -- the git-tracked results/
    JSONs would have carried 07-26 numbers, predating every fix since. That is the
    "silently reported superseded numbers" failure this file exists to prevent, reproduced
    inside the file itself. Found by the run machine while staging a transmit.

    A4.1 adds a second wrinkle: a model can now legitimately have TWO analysis results --
    the centered fit, which A4.1 commits as PRIMARY, and the non-centered robustness refit,
    which is newer. Newest-wins would silently promote the robustness check over the fit the
    amendment named primary, so the centered one is preferred explicitly. Artifacts written
    before A4.1 carry no `spread_parameterization` field and are centered, which is why the
    default is "centered" rather than "unknown".
    """
    def primary_first(r):
        param = r.get("spread_parameterization", "centered")
        return (0 if param == "centered" else 1,)

    seen, out = set(), []
    for r in sorted(rows, key=primary_first):
        key = (r.get("_stage"), r.get("model"))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def robustness_rows(rows: list[dict]) -> list[dict]:
    """Analysis results that are NOT the primary fit -- reported alongside, never instead.

    A4.1 commitment 4 requires BOTH fits reported for all three models. This helper existed
    and was never called, so the non-centered fit lived only in raw artifacts and appeared
    in neither STATUS.md nor the tracked results/. Present as data, absent from the report,
    which is not "reported". Found by the run machine while staging the transmit.
    """
    seen, out = set(), []
    for r in rows:
        if r.get("spread_parameterization", "centered") == "centered":
            continue
        key = (r.get("_stage"), r.get("model"))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def pass_c_robustness_table(results: list[dict]) -> pd.DataFrame:
    """A4.1's non-centered refit, reported beside the primary and never instead of it."""
    return _pass_c_rows(robustness_rows(results))


def _get(d: dict, *path, default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def absolute_table(results: list[dict]) -> pd.DataFrame:
    """Absolute-Likert instrument validation (A1.5)."""
    rows = []
    for r in results:
        g = r.get("gates")
        if not g:
            continue
        rows.append({
            "model": r.get("model"),
            "device": _get(r, "provenance", "device"),
            "median_rho": g.get("median_rho"),
            "sigma_coll": g.get("sigma_between"),
            "sigma_asc": g.get("sigma_between_ascending"),
            "icc_c1_asc": _get(g, "icc_all", "icc_c1"),
            "passed": g.get("passed"),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(subset=["model"]).sort_values("model")


def pairwise_table(results: list[dict]) -> pd.DataFrame:
    """Pairwise instrument under the Amendment 2 model."""
    rows = []
    for r in results:
        if r["_stage"] != "pass_a_pairwise":
            continue
        rg = r.get("reliability_gate") or {}
        th = r.get("theta") or {}
        ex = r.get("excess_consistency_slope") or {}
        betas = [b.get("beta_mean") for b in (r.get("beta") or []) if b.get("beta_mean") is not None]
        rows.append({
            "model": r.get("model"),
            "device": _get(r, "provenance", "device"),
            "outcome": r.get("outcome"),
            "sigma_item": th.get("sigma_item"),
            "post_sd": th.get("posterior_sd_median"),
            "sep_ratio": th.get("separation_ratio"),
            "reliab_emp": rg.get("empirical_reliability_spearman"),
            "reliab_model": rg.get("model_reliability"),
            "beta_min": min(betas) if betas else None,
            "beta_max": max(betas) if betas else None,
            "excess_slope": ex.get("slope"),
            "excess_flat": ex.get("flat"),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(subset=["model"]).sort_values("model")


def readout_mass_table(results: list[dict]) -> pd.DataFrame:
    """Per-template readout mass -- where prompt bugs surface first (A1.6)."""
    rows = []
    for r in results:
        for m in (r.get("readout_mass_by_template") or []):
            rows.append({"model": r.get("model"), "template": m.get("template"),
                         "frac_invalid": m.get("frac_invalid"),
                         "mass_median": m.get("mass_median")})
    if not rows:
        return pd.DataFrame()
    return (pd.DataFrame(rows).drop_duplicates(subset=["model", "template"])
            .pivot(index="model", columns="template", values="frac_invalid").round(3))


def pass_c_table(results: list[dict]) -> pd.DataFrame:
    """H1 per model, with the design figures that decide how to read it.

    status.py had no Pass C table at all: the reporting layer was never extended past the
    instrument, so the headline result was not in the generated record.
    """
    return _pass_c_rows(_newest_per_model(results))


def _pass_c_rows(source: list[dict]) -> pd.DataFrame:
    rows = []
    for r in source:
        pr, pw = r.get("primary"), r.get("power") or {}
        if not pr:
            continue
        rows.append({
            "model": r.get("model"),
            "device": _get(r, "provenance", "device"),
            "outcome": r.get("outcome"),
            "lambda_median": pr.get("median"),
            "hdi_low": pr.get("hdi_low"),
            "hdi_high": pr.get("hdi_high"),
            "P(<0)": pr.get("p_negative"),
            "sesoi": r.get("sesoi"),
            "mde/sesoi": pw.get("mde_over_sesoi"),
            "equiv_reachable": pw.get("equivalence_reachable"),
            "decision": pr.get("decision"),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("model")


def render(results: list[dict]) -> str:
    from datetime import datetime, timezone

    L = [
        "# Status", "",
        "**Generated by `scripts/status.py` — do not edit by hand.**",
        f"Read from {len(results)} artifact result file(s) at "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.", "",
        "Numbers here are read out of provenance-stamped artifacts. See RETRACTIONS.md "
        "for why this file is generated rather than written.", "",
    ]

    dev = {_get(r, "provenance", "device") for r in results} - {None}
    smoke = {bool(r.get("smoke")) for r in results}
    L += ["## Provenance", "",
          f"- devices present: {sorted(dev) or 'none'}",
          f"- smoke artifacts present: {any(smoke)}",
          "- `assert_reportable` requires CUDA + bf16 + pinned revision + clean tree; "
          "anything from a dev machine is non-reportable by construction.", ""]

    c = pass_c_table(results)
    if not c.empty:
        n_pass = int((c["decision"] == "pass").sum())
        directional = int((c["P(<0)"].fillna(0) >= 0.95).sum())
        entered = n_pass >= 1 and directional >= 2
        L += ["## Pass C -- H1 (primary: lambda_chose - lambda_yoked, predicted NEGATIVE)", "",
              c.round(4).to_markdown(index=False), "",
              f"- models with a full 9.2 pass: **{n_pass}**",
              f"- models directional at P(lambda<0) >= 0.95: **{directional}**",
              f"- **A3.7 project gate: Stage 1 {'ENTERED' if entered else 'NOT entered'}** "
              "(needs one full pass AND two directional)",
              "",
              "`inconclusive` is the MODAL outcome by design, not a surprise: A3.3 puts it "
              "at ~48% even when the effect is real and exactly at the SESOI, and ~70% "
              "under a true null. Read it against the MDE, not against zero.",
              "",
              "A credible effect in the WRONG (positive) direction scores `inconclusive` "
              "under 9.2 as written, since `pass` requires the HDI to exclude 0 in the "
              "PREDICTED direction and `fail` requires the whole HDI inside the ROPE. That "
              "is the rule as preregistered; it is reported here rather than amended.",
              ""]

    rb = pass_c_robustness_table(results)
    if not rb.empty:
        L += ["## Pass C robustness -- A4.1 non-centered template effect", "",
              rb.round(4).to_markdown(index=False), "",
              "A4.1 non-centred `u_template` after llama's centered fit showed 362 "
              "divergences in a `sd_template` funnel. The reparameterisation is pure -- "
              "same model, same posterior in expectation -- and was verified on synthetic "
              "data before it touched the real fit.",
              "",
              "**The centered fit above remains PRIMARY** (A4.1 commitment 2). This table "
              "is a robustness check and is reported whichever way it moves; that was "
              "fixed before the refit ran so it could not be chosen afterwards.", ""]

    a = absolute_table(results)
    if not a.empty:
        L += ["## Absolute-Likert instrument (A1.5 validation record)", "",
              a.round(4).to_markdown(index=False), "",
              "All models are expected to fail polarity validity here. That is the "
              "point of the record.", ""]

    p = pairwise_table(results)
    if not p.empty:
        L += ["## Pairwise instrument (Amendment 2 model)", "",
              p.round(4).to_markdown(index=False), "",
              "`reliab_emp` is the A2.2 gate (threshold 0.70). A large gap between "
              "`reliab_emp` and `reliab_model`, or `excess_flat = False`, indicates "
              "the order model is still misspecified.", ""]

    m = readout_mass_table(results)
    if not m.empty:
        L += ["## Readout mass: fraction invalid, by template (A1.6)", "",
              m.to_markdown(), "",
              "A template above ~0.1 has a prompt bug. This is the cheapest place to "
              "catch one.", ""]

    if not results:
        L += ["_No artifacts found. Run a pass first._", ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="write STATUS.md and tracked per-run summaries under results/")
    args = ap.parse_args()

    results = _load_results()
    text = render(results)
    print(text)

    if args.write:
        (REPO / "STATUS.md").write_text(text, encoding="utf-8")
        OUT.mkdir(exist_ok=True)
        for r in _newest_per_model(results):
            name = f"{r['_stage']}__{r.get('model','unknown')}.json"
            # Trim to the reported quantities; the full artifact stays under
            # artifacts/. This is what makes the numbers greppable in git.
            keep = {k: v for k, v in r.items()
                    if k in ("model", "outcome", "smoke", "config_hash", "provenance",
                             "gates", "theta", "beta", "reliability_gate",
                             "excess_consistency_slope", "operating_window",
                             "readout_mass", "readout_mass_by_template",
                             "order_invariance_reported", "test_retest",
                             # Pass C. The keep-list predated it, so the git-tracked
                             # summaries carried NO H1 result at all -- the entire payload
                             # of a Pass C snapshot was being dropped on the floor.
                             "primary", "contrasts", "sesoi", "power",
                             "structure_factor_agreement", "pairs",
                             "instrument_validation")}
            (OUT / name).write_text(json.dumps(keep, indent=2, default=str), encoding="utf-8")
        for r in robustness_rows(results):
            keep = {k: v for k, v in r.items()
                    if k in ("model", "outcome", "config_hash", "provenance", "primary",
                             "contrasts", "sesoi", "power", "spread_parameterization",
                             "spread_source_digest", "structure_factor_agreement")}
            (OUT / f"{r['_stage']}_robustness__{r.get('model','unknown')}.json").write_text(
                json.dumps(keep, indent=2, default=str), encoding="utf-8")

        kept = _newest_per_model(results)
        print(f"\nwrote STATUS.md and {len(kept)} summary file(s) to results/ "
              f"+ {len(robustness_rows(results))} robustness file(s) "
              f"(de-duplicated from {len(results)} artifact result files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
