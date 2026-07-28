"""One command runs Stage 0 end-to-end for one model.

    python -m src.experiments.run --config configs/stage0_gemma-2-2b.yaml

    absolute Pass A   -> instrument-validation record (A1.5). NEVER gates.
    pairwise Pass A   -> theta, beta, reliability gate (A2.2). HALTS here or nowhere.
    Pass B            -> pairs, from theta on disjoint template sets (audit T2.2)
    Pass C            -> the experiment (A2.9)
    spread model      -> H1

Every stage is skipped when its artifact exists under the current stage hash, so the
pipeline resumes from cached upstream output. Changing a number-affecting parameter
changes the hash and therefore the path, so a modified run cannot overwrite an earlier
one or be confused with it.

TWO THINGS THIS FILE USED TO GET WRONG, both from the Pass C audit:

  T2.1  It gated on the polarity criterion, which Amendment 2 retired. Every model
        would have been excluded on a rule that no longer exists. The absolute pass is
        now run and archived as a validation record and cannot halt anything.
  T2.2  It fed Pass B polarity-collapsed absolute ratings -- the retired instrument.
        Pass B now receives Bradley-Terry theta fitted on the disjoint template sets.

Power IS run, between Pass B and Pass C, because section 8 requires that ordering and
because a power figure computed after the outcomes is a number chosen with knowledge of
the answer. It reports the minimum detectable effect; section 8's "80% power at the SESOI"
was withdrawn by Amendment 3 A3.1 as unsatisfiable at any sample size.

Exit codes:  0 completed   2 halted by the reliability gate   1 error
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.analysis import spread_model
from src.analysis.bradley_terry import (
    anchor_ordering_check,
    convergence_summary,
    excess_consistency_slope,
    excess_slope_ppc_null,
    fit_bradley_terry,
    predicted_split_half,
    theta_item_scores,
)
from src.analysis.reliability import evaluate_gates
from src.config import RunConfig, load_config
from src.experiments import pass_a as stage_a
from src.experiments import pass_a_pairwise as stage_pw
from src.experiments import pass_b as stage_b
from src.experiments import pass_c as stage_c
from src.provenance import Provenance, assert_poolable, capture, read_parquet, write_parquet
from src.readout import validity
from src.readout.pairwise import collect_comparisons, load_anchors
from src.stimuli.build import load_items, load_templates

HALT = 2
STAGES = ("absolute", "pairwise", "pass_b", "pass_c", "analysis")


class Lazy:
    """Load the model only if a stage actually needs a forward pass."""

    def __init__(self, cfg: RunConfig):
        self.cfg, self._runner = cfg, None

    def __call__(self):
        if self._runner is None:
            from src.models.runner import describe_maps, load_runner

            print("loading model ...", flush=True)
            self._runner = load_runner(self.cfg)
            print(f"  device={self._runner.device} dtype={self._runner.model.dtype}")
            print(describe_maps(self._runner))
        return self._runner


def _echo(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}", flush=True)


def _instrument_validation(cfg: RunConfig, runner_factory, prov: Provenance,
                           results: dict) -> pd.DataFrame:
    """A1.5. Run the preregistered absolute instrument once, archive it, report it.

    It is EXPECTED to fail polarity validity. That is the point: the paper reports
    "we ran the preregistered instrument at full scale and these are the numbers"
    rather than resting the switch to a pairwise instrument on pilot runs. It gates
    nothing and is never read by a downstream stage.
    """
    _echo("Absolute Pass A -- instrument-validation record (A1.5). Gates nothing.")
    path = stage_a.artifact_path(cfg)
    if path.exists():
        print(f"cached: {path}")
        frame = read_parquet(path)
    else:
        frame = stage_a.run_pass_a(cfg, runner_factory(), prov)
        print(f"wrote: {path}")

    gate = evaluate_gates(cfg, frame)
    print(f"  median polarity rho = {gate.median_rho:+.3f}  "
          f"(retired threshold was {cfg.gates.validity_rho_min})")
    print(f"  sigma_between: collapsed {gate.sigma_between:.3f}, "
          f"ascending {gate.sigma_between_ascending:.3f}")
    print(f"  ICC(C,1) ascending = {gate.icc_all.get('icc_c1', float('nan')):.3f}")
    print(f"  digit mass: median {frame['digit_mass'].median():.4f}, "
          f"min {frame['digit_mass'].min():.4f}")
    if gate.median_rho < cfg.gates.validity_rho_min:
        print("  -> fails the retired polarity criterion, as expected. Recorded, not acted on.")

    archive = cfg.artifacts_dir / "instrument_validation" / cfg.model.name / cfg.hash("pass_a")
    archive.mkdir(parents=True, exist_ok=True)
    write_parquet(frame, archive / f"absolute_{cfg.hash('pass_a')}.parquet", prov)
    gate.validity.to_csv(archive / f"validity_{cfg.hash('pass_a')}.csv", index=False)

    results["instrument_validation"] = {
        "median_rho": gate.median_rho,
        "sigma_between_collapsed": gate.sigma_between,
        "sigma_between_ascending": gate.sigma_between_ascending,
        "icc_all": gate.icc_all,
        "per_template_validity": gate.validity.to_dict("records"),
        "archive": str(archive),
        "gates_nothing": True,
    }
    return frame


def run(cfg: RunConfig, *, stop_after: str | None = None, progressbar: bool = True) -> int:
    prov = capture(cfg)
    runner_factory = Lazy(cfg)
    out_dir = cfg.artifact_dir("analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {"model": cfg.model.name, "config_hash": cfg.hash(),
                     "provenance": prov.model_dump(), "smoke": cfg.smoke}

    _echo(f"Stage 0  |  {cfg.model.name}  |  config {cfg.hash()}")
    print(f"device={prov.device} ({prov.device_name})  dtype={prov.dtype}")
    print(f"revision={prov.model_revision} pinned={prov.model_revision_pinned}")
    print(f"git={prov.git_sha[:12]} dirty={prov.git_dirty}  seed={cfg.seed}")
    if cfg.smoke:
        print("SMOKE MODE: reduced stimuli. assert_reportable() rejects these artifacts.")

    absolute = _instrument_validation(cfg, runner_factory, prov, results)
    if stop_after == "absolute":
        return _finish(out_dir, cfg, results, "stopped-after-absolute")

    # ---- pairwise Pass A: the live instrument ----------------------------
    _echo("Pairwise Pass A -- anchor comparisons (A1.1)")
    pw_path = stage_pw.artifact_path(cfg)
    if pw_path.exists():
        print(f"cached: {pw_path}")
        comparisons = read_parquet(pw_path)
    else:
        comparisons = collect_comparisons(
            cfg, runner_factory(), load_items(cfg), load_anchors(cfg), load_templates(cfg),
            arms=("digits",), desc="pairwise",
            checkpoint=pw_path.parent / f"_checkpoint_{cfg.hash('pass_a')}.parquet",
        )
        comparisons = write_parquet(comparisons, pw_path, prov)
        print(f"wrote: {pw_path}")

    per_t = validity.summarize(comparisons, by=["arm", "template"])
    print("\nreadout mass by template (A1.6) -- where prompt bugs surface first:")
    print(per_t.to_string(index=False))
    results["readout_mass_by_template"] = per_t.to_dict("records")

    print("\norder invariance -- reported only, retired as a gate (A2.2):")
    inv = stage_pw.report_order_invariance(comparisons)
    print(inv.to_string(index=False))
    results["order_invariance_reported"] = inv.to_dict("records")

    _echo("Bradley-Terry with per-template order term (A2.1)")
    instrument = _instrument_fit(cfg, comparisons, progressbar=progressbar)
    results["beta"] = instrument["beta"]
    results["excess_consistency_slope"] = instrument["excess_consistency_slope"]

    # ---- the only gate ---------------------------------------------------
    _echo("Reliability gate (A2.2) -- the sole exclusion criterion")
    gate = instrument["reliability_gate"]
    print(f"empirical split-half of theta, {gate['split_a']} vs {gate['split_b']}: "
          f"Spearman {gate['empirical_reliability_spearman']:.3f} "
          f"(threshold {gate['threshold']})")
    # The full-data figure is NOT the comparator. It comes from a fit using all five
    # templates; the empirical figure is between two SHORTER fits, each noisier, so the
    # two differ even under perfect specification. `pass_a_pairwise.main()` printed the
    # length-matched prediction and this path did not, so the run the numbers come from
    # showed the misleading comparison. A2.4's recorded gap was computed this way.
    na, nb = len(gate["split_a"]), len(gate["split_b"])
    pred = predicted_split_half(gate["model_reliability"], na / (na + nb), nb / (na + nb))
    print(f"model-internal reliability {gate['model_reliability']:.3f} (all templates) "
          f"-- NOT the comparator; using it as the gate would admit models whose item "
          "rankings do not reproduce across paraphrases")
    print(f"length-matched prediction for a {na}/{nb} split = {pred:.3f}")
    print(f"  shortfall vs THAT: "
          f"{gate['empirical_reliability_spearman'] - pred:+.3f}  (A2.4's open discrepancy "
          "is this number, not the shortfall against the full-data figure)")
    gate["length_matched_prediction"] = pred
    print(f"  ->  {'PASS' if gate['passed'] else 'HALT'}")
    for r in gate["reasons"]:
        print(f"  - {r}")
    results["reliability_gate"] = gate

    if not gate["passed"]:
        return _finish(out_dir, cfg, results, "halted-reliability", HALT)
    if stop_after == "pairwise":
        return _finish(out_dir, cfg, results, "stopped-after-pairwise")

    # ---- Pass B, fed from theta (audit T2.2) -----------------------------
    _echo("Pass B -- pairs from theta on disjoint template sets")
    b_path = stage_b.artifact_path(cfg)
    if b_path.exists():
        print(f"cached: {b_path}")
        pairs = read_parquet(b_path)
    else:
        scores = theta_item_scores(cfg, comparisons, arm="digits")
        print(f"theta scores: selection sd {scores['score_selection'].std(ddof=1):.3f}, "
              f"analysis sd {scores['score_analysis'].std(ddof=1):.3f}")
        pairs = stage_b.run_pass_b(cfg, scores, prov, instrument["sigma_item"],
                                   instrument_key=instrument.get("cache_key"))
        print(f"wrote: {b_path}")
    stage_b.assert_reusable(cfg, pairs, instrument["sigma_item"],
                            instrument_key=instrument.get("cache_key"))
    print(f"match tolerance {cfg.pass_b.match_tolerance(instrument['sigma_item']):.4f} "
          f"= {cfg.pass_b.match_tolerance_sigma_fraction} x sigma_item "
          f"({instrument['sigma_item']:.4f})  [A2.9.2, per model]")

    diag = stage_b.pair_diagnostics(pairs)
    print(diag.to_string(index=False))
    print("matching quality:", {k: round(v, 4) for k, v in diag.attrs.items()})
    results["pairs"] = {"n": int(len(pairs)), "diagnostics": diag.to_dict("records")}

    # ---- power, on the realized design, BEFORE the experiment ------------
    # Section 8 requires this ordering and it is the whole point: computed here, the power
    # figure is produced by code that has not seen a single Pass C outcome. Computed after
    # Pass C it would be a number chosen with knowledge of the answer.
    _echo("Power on the realized design (A3.1/A3.4) -- computed before Pass C runs")
    results["power"] = _power(cfg, pairs, instrument)
    if stop_after == "pass_b":
        return _finish(out_dir, cfg, results, "stopped-after-pass-b")

    # ---- Pass C ----------------------------------------------------------
    _echo("Pass C -- the experiment (A2.9)")
    c_path = stage_c.artifact_path(cfg)
    if c_path.exists():
        print(f"cached: {c_path}")
        trials = read_parquet(c_path)
    else:
        trials = stage_c.run_pass_c(cfg, pairs, runner_factory(), prov)
        print(f"wrote: {c_path}")

    assert_poolable([comparisons, trials], context="pairwise Pass A + Pass C")
    print("\nreadout mass by condition:")
    print(validity.summarize(trials, by=["condition"]).to_string(index=False))
    if stop_after == "pass_c":
        return _finish(out_dir, cfg, results, "stopped-after-pass-c")

    # ---- H1 --------------------------------------------------------------
    _echo("Spread model -- H1 (A2.9.1)")
    design = spread_model.prepare(cfg, trials)
    print(f"|diff| centring: mean={design.diff_mean:.4f} sd={design.diff_sd:.4f}")
    idata = spread_model.fit(cfg, design, progressbar=progressbar)

    conv = spread_model.convergence(cfg, idata)
    bad = conv[~(conv["rhat_ok"] & conv["ess_ok"])]
    print(f"convergence: {len(bad)} parameter(s) failing")
    if len(bad):
        print(bad.to_string())

    # `instrument`, not `fit`. `fit` is a local inside `_instrument_fit` and has never been
    # in scope here: this was a NameError sitting in the one stage the pipeline had never
    # reached, because every test run so far stopped at or before Pass B. It would have
    # fired AFTER Pass C's 20,000 forward passes per model.
    sigma_item = float(instrument["sigma_item"])
    sesoi = cfg.analysis.sesoi_sigma_fraction * sigma_item
    print(f"\nSESOI = {cfg.analysis.sesoi_sigma_fraction} x sigma_item "
          f"({sigma_item:.3f}) = {sesoi:.4f} logits per SD of |diff|")
    contrasts = spread_model.contrasts(cfg, idata, sesoi)
    print(contrasts.to_string(index=False))
    # Keyed like the posterior. A4.1 commits the CENTERED fit as primary and requires
    # both fits to be reported, so the robustness refit must not overwrite it.
    contrasts.to_parquet(
        out_dir / f"contrasts_{cfg.hash()}-{spread_model.source_digest()}.parquet",
        index=False)
    idata.to_netcdf(str(spread_model.posterior_path(cfg, out_dir)))

    primary = contrasts[(contrasts["name"] == spread_model.PRIMARY)
                        & (contrasts["term"] == "lambda")]
    if not len(primary):
        # A selection that matches nothing must not read as a completed run. This exited 0
        # with outcome "primary-unavailable", so a rename of PRIMARY or of the `lambda`
        # term would have produced a successful-looking run carrying no primary test at
        # all -- the same silent-empty class as the notebook's `term == "slope"` filter.
        results["contrasts"] = contrasts.to_dict("records")
        results["sesoi"] = sesoi
        available = sorted(set(contrasts["name"])) if len(contrasts) else []
        _echo(f"PRIMARY TEST ({spread_model.PRIMARY}) x |diff|  ->  NOT COMPUTED")
        print(f"  no contrast named {spread_model.PRIMARY!r} with term 'lambda'.")
        print(f"  contrasts present: {available}")
        return _finish(out_dir, cfg, results, "primary-unavailable", HALT)

    decision = primary["decision"].iloc[0]
    _echo(f"PRIMARY TEST ({spread_model.PRIMARY}) x |diff|  ->  {decision.upper()}")
    row = primary.iloc[0]
    print(f"  median {row['median']:+.4f}   "
          f"{int(cfg.analysis.hdi_prob * 100)}% HDI "
          f"[{row['hdi_low']:+.4f}, {row['hdi_high']:+.4f}]   "
          f"P(<0) = {row['p_negative']:.3f}")

    agree = spread_model.structure_factor_agreement(contrasts)
    if agree:
        print(f"\nstructure factor estimated twice: {agree['edge_a']} = "
              f"{agree['estimate_a']:+.4f}, {agree['edge_b']} = {agree['estimate_b']:+.4f}, "
              f"discrepancy {agree['discrepancy']:+.4f}")
        print("  (agreement means a turn-presence effect is identified and subtractable)")

    results["contrasts"] = contrasts.to_dict("records")
    results["sesoi"] = sesoi
    results["structure_factor_agreement"] = agree
    results["spread_parameterization"] = spread_model.PARAMETERIZATION
    results["spread_source_digest"] = spread_model.source_digest()
    results["primary"] = primary.to_dict("records")[0] if len(primary) else {}
    return _finish(out_dir, cfg, results, f"primary-{decision}")


#: Modules whose source determines the instrument numbers. Their contents are digested
#: into the cache key, because the config hash covers PARAMETERS and these fits depend on
#: the model SPECIFICATION too. Editing the likelihood must invalidate the cache; a config
#: hash alone would happily serve a fit from a prior version of the model.
_INSTRUMENT_SOURCES = ("src/analysis/bradley_terry.py", "src/experiments/pass_a_pairwise.py")


def _instrument_fit(cfg: RunConfig, comparisons: pd.DataFrame, *, progressbar: bool) -> dict:
    """Fit the instrument and evaluate the gate, cached.

    This block is 27 Bayesian fits at production sampling -- one Bradley-Terry fit, the
    24 replicates of its posterior-predictive null, and the two reliability halves. On
    400 items that is 1.5-3 hours on ANY machine, because it is CPU-bound MCMC and not
    GPU work. It is also a pure function of the cached comparisons plus the pass_a
    parameters, so re-running it on every invocation was pure waste: a Pass C run would
    pay three hours to re-derive numbers it had already written down.

    Caching a FIT is more dangerous than caching forward passes, which are deterministic
    given model and prompt. So the key carries a digest of the fitting code as well.
    """
    import hashlib

    root = Path(__file__).resolve().parents[2]
    src_digest = hashlib.sha256()
    for rel in _INSTRUMENT_SOURCES:
        src_digest.update((root / rel).read_bytes())
    key = f"{cfg.hash('pass_a')}-{src_digest.hexdigest()[:8]}"
    path = cfg.artifacts_dir / "instrument_fit" / cfg.model.name / key / f"fit_{key}.json"

    if path.exists():
        rec = json.loads(path.read_text(encoding="utf-8"))
        print(f"cached: {path}")
        print(rec["printed"])
        return rec

    fit = fit_bradley_terry(cfg, comparisons, arm="digits", progressbar=progressbar)
    ex = excess_consistency_slope(comparisons, fit)
    null = excess_slope_ppc_null(cfg, comparisons, fit)
    z = ((ex["slope"] - null["null_mean"]) / null["null_sd"]) if null["null_sd"] > 0 else float("nan")
    gate = stage_pw.evaluate_reliability_gate(cfg, comparisons, load_templates(cfg))

    printed = "\n".join([
        fit.summary(),
        str(convergence_summary(cfg, fit.idata)),
        "\nposition bias by template (beta):",
        fit.beta.round(3).to_string(index=False),
        "\nanchor abilities (intended tier is a design annotation, never shown):",
        anchor_ordering_check(fit, load_anchors(cfg)).to_string(index=False),
        f"\nfit quality (A2.1): slope {ex['slope']:+.4f} "
        f"[{ex['ci_low']:+.4f}, {ex['ci_high']:+.4f}] vs null "
        f"{null['null_mean']:+.4f} (sd {null['null_sd']:.4f}) -> {z:+.2f} sd",
    ])
    print(printed)

    rec = {
        "sigma_item": fit.sigma_item,
        "beta": fit.beta.to_dict("records"),
        "excess_consistency_slope": {**ex, **null, "z_vs_null": z},
        "reliability_gate": {**gate, "model_reliability": fit.model_reliability},
        "printed": printed,
        "cache_key": key,
        "ppc_null_replicates": cfg.analysis.ppc_null_replicates,
        "sampling": {"chains": cfg.analysis.chains, "tune": cfg.analysis.tune,
                     "draws": cfg.analysis.draws},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote: {path}")
    return rec


def _power(cfg: RunConfig, pairs: pd.DataFrame, instrument: dict) -> dict:
    """Report the minimum detectable effect, not power at the SESOI (A3.1).

    Section 8's criterion is unsatisfiable: it asks for 80% power AT the SESOI while
    section 9.2's `pass` cell requires the median to EXCEED the SESOI, so power there
    converges to exactly 0.500 at any sample size. Amendment 3 withdrew it. The MDE is
    reported instead, along with whether the equivalence branch is reachable at all --
    because if it is not, a null can only be `inconclusive`, and that has to be known
    before the data rather than discovered while writing it up.
    """
    from src.analysis import power as power_mod

    sigma_item = instrument.get("sigma_item")
    if sigma_item is None:
        # A cached fit written before sigma_item was recorded. Refusing to guess: an
        # invented scale would silently rescale the SESOI, which is the one number
        # Amendment 3 was careful not to touch.
        print("SKIPPED: cached instrument fit predates the sigma_item field. Delete "
              f"the instrument_fit cache for {cfg.model.name} to recompute it.")
        return {"skipped": "sigma_item absent from cached instrument fit"}

    res = power_mod.analyze(cfg, pairs, pd.DataFrame(instrument["beta"]), sigma_item)
    print(res.summary())
    sens = power_mod.gamma_sensitivity(cfg, pairs, pd.DataFrame(instrument["beta"]), sigma_item)
    print("\ngamma sensitivity (gamma is the one quantity unknown before Pass C):")
    print(sens.round(4).to_string(index=False))

    return {
        "se": res.se, "sesoi": res.sesoi,
        "min_detectable_effect": res.min_detectable,
        "mde_over_sesoi": res.min_detectable / res.sesoi,
        "power_at_sesoi_not_a_criterion": res.power_at_sesoi,
        "equivalence_reachable": res.equivalence_reachable,
        "information_by_stratum": res.information.to_dict("records"),
        "gamma_sensitivity": sens.to_dict("records"),
        "assumptions": res.assumptions,
        "section_8_criterion": "withdrawn by Amendment 3 A3.1 as unsatisfiable",
    }


def _finish(out_dir: Path, cfg: RunConfig, results: dict, outcome: str, code: int = 0) -> int:
    results["outcome"] = outcome
    digest = results.get("spread_source_digest")
    path = out_dir / (f"results_{cfg.hash()}-{digest}.json" if digest
                      else f"results_{cfg.hash()}.json")
    path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nresults: {path}")
    if code == HALT:
        _echo("HALTED. Reported, not treated as evidence about H1 either way.")
    return code


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run Stage 0 end-to-end for one model.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--artifacts", default=None)
    ap.add_argument("--stop-after", choices=STAGES[:-1], default=None)
    ap.add_argument("--no-progress", action="store_true")
    # The run machine has to force cores=1 or PyMC's multiprocessing deadlocks on the
    # Windows spawn start method. It was doing that by wrapping run.py in a launcher,
    # because there was no hook -- and patching code in place on the run machine is
    # forbidden (it dirties the tree, which assert_reportable rejects). So: a flag.
    ap.add_argument("--sampler-cores", type=int, default=None,
                    help="force PyMC cores (use 1 on Windows to avoid a spawn deadlock)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, {"artifacts_dir": args.artifacts} if args.artifacts else None)
    if args.sampler_cores is not None:
        cfg = cfg.model_copy(update={"analysis": cfg.analysis.model_copy(
            update={"sampler_cores": args.sampler_cores})})
    return run(cfg, stop_after=args.stop_after, progressbar=not args.no_progress)


if __name__ == "__main__":
    sys.exit(main())
