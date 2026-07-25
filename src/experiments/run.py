"""One command runs Stage 0 end-to-end for one model.

    python -m src.experiments.run --config configs/stage0_qwen2.5-0.5b.yaml

Pass A -> validity gate -> sigma_between / SESOI -> Pass B -> power simulation
-> Pass C -> mixed model, planned contrasts, figures.

Every stage is skipped when its artifact already exists under the current config
hash, so the pipeline is re-runnable from cached upstream output. Changing any
number-affecting parameter changes the hash and therefore the paths, so a
modified run cannot silently overwrite or be confused with an earlier one.

Exit codes:  0 completed   2 halted by a preregistered exclusion   1 error
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import mixed, plots, power
from src.analysis.reliability import GateResult, evaluate_gates, item_scores
from src.config import RunConfig, load_config
from src.experiments import pass_a as stage_a
from src.experiments import pass_b as stage_b
from src.experiments import pass_c as stage_c
from src.provenance import (
    Provenance,
    assert_poolable,
    capture,
    read_parquet,
    write_parquet,
)
from src.stimuli.build import audit_templates, load_templates

HALT = 2


class Lazy:
    """Load the model only if a stage actually needs a forward pass."""

    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self._runner = None

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


def run(
    cfg: RunConfig,
    *,
    robustness: bool = False,
    make_plots: bool = True,
    stop_after: str | None = None,
) -> int:
    prov = capture(cfg)
    runner_factory = Lazy(cfg)

    _echo(f"Stage 0  |  {cfg.model.name}  |  config {cfg.hash()}")
    print(f"device={prov.device} ({prov.device_name})  dtype={prov.dtype}")
    print(f"revision={prov.model_revision} pinned={prov.model_revision_pinned}")
    print(f"torch={prov.torch_version} transformers={prov.transformers_version}")
    print(f"git={prov.git_sha} dirty={prov.git_dirty}  seed={cfg.seed}")
    if cfg.smoke:
        print(
            "SMOKE MODE: reduced stimuli, not a scientific run. "
            "assert_reportable() will reject these artifacts."
        )

    out_dir = cfg.artifact_dir("analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {
        "model": cfg.model.name,
        "config_hash": cfg.hash(),
        "provenance": prov.model_dump(),
        "smoke": cfg.smoke,
    }

    # ---- stimulus audit ---------------------------------------------------
    audit = audit_templates(load_templates(cfg))
    audit.to_csv(out_dir / f"stimulus_audit_{cfg.hash()}.csv", index=False)
    print("\nstimulus audit (turn/mention balance across conditions):")
    print(audit.groupby("condition")[["n_turns", "mentions_designated", "mentions_other"]]
          .agg(["min", "max"]).to_string())

    # ---- Pass A ----------------------------------------------------------
    _echo("Pass A -- item rating (400 items x 5 templates x 2 polarities)")
    a_path = stage_a.artifact_path(cfg)
    if a_path.exists():
        print(f"cached: {a_path}")
        pass_a_frame = read_parquet(a_path)
    else:
        pass_a_frame = stage_a.run_pass_a(cfg, runner_factory(), prov)
        print(f"wrote: {a_path}")
    print(f"{len(pass_a_frame)} ratings   "
          f"digit_mass min={pass_a_frame['digit_mass'].min():.4f} "
          f"median={pass_a_frame['digit_mass'].median():.4f}")

    # ---- gates -----------------------------------------------------------
    _echo("Exclusion criteria (preregistration.md section 5)")
    gate: GateResult = evaluate_gates(cfg, pass_a_frame)
    print(gate.summary())
    print("\nper-template polarity validity:")
    print(gate.validity.to_string(index=False))

    gate.validity.to_csv(out_dir / f"validity_{cfg.hash()}.csv", index=False)
    results["gates"] = {
        "passed": gate.passed,
        "median_rho": gate.median_rho,
        "surviving_templates": gate.surviving_templates,
        "icc_all": gate.icc_all,
        "icc_selection": gate.icc_selection,
        "icc_all_collapsed": gate.icc_all_collapsed,
        "sigma_between": gate.sigma_between,
        "sigma_between_ascending": gate.sigma_between_ascending,
        "sesoi_primary": gate.sesoi_primary,
        "sesoi_secondary": gate.sesoi_secondary,
        "icc_tripwire_hit": gate.icc_tripwire_hit,
        "reasons": gate.reasons,
    }

    if not gate.passed:
        results["outcome"] = "halted-by-exclusion-criteria"
        _write_results(out_dir, cfg, results)
        _echo("HALTED for this model. Reported, not treated as evidence either way.")
        return HALT

    sesoi = gate.sesoi_primary

    if stop_after == "pass_a":
        results["outcome"] = "stopped-after-pass-a"
        _write_results(out_dir, cfg, results)
        _echo("Stopped after Pass A as requested. Gate passed; Pass B/C not run.")
        return 0

    # ---- Pass B ----------------------------------------------------------
    _echo("Pass B -- pair construction from this model's own Pass A ratings")
    scores = item_scores(cfg, pass_a_frame)
    b_path = stage_b.artifact_path(cfg)
    if b_path.exists():
        print(f"cached: {b_path}")
        pairs = read_parquet(b_path)
    else:
        pairs = stage_b.run_pass_b(cfg, scores, prov)
        print(f"wrote: {b_path}")

    diag = stage_b.pair_diagnostics(pairs)
    print(diag.to_string(index=False))
    print("matching quality:", {k: round(v, 4) for k, v in diag.attrs.items()})
    diag.to_csv(out_dir / f"pair_diagnostics_{cfg.hash()}.csv", index=False)
    results["pairs"] = {
        "n": int(len(pairs)),
        "diagnostics": diag.to_dict("records"),
        "matching": {k: float(v) for k, v in diag.attrs.items()},
    }

    if stop_after == "pass_b":
        results["outcome"] = "stopped-after-pass-b"
        _write_results(out_dir, cfg, results)
        _echo("Stopped after Pass B as requested.")
        return 0

    # ---- power -----------------------------------------------------------
    _echo("Power simulation (after Pass A, before Pass C)")
    pw = power.simulate_power(
        cfg,
        sigma_between=gate.sigma_between,
        icc_c1=gate.icc_all["icc_c1"],
        diff_analysis=pairs["diff_analysis"].to_numpy(),
        sesoi=sesoi,
        n_templates=len(gate.surviving_templates),
    )
    print(pw.summary())
    print(pw.grid.to_string(index=False))
    pw.grid.to_csv(out_dir / f"power_{cfg.hash()}.csv", index=False)
    results["power"] = {
        "power_at_sesoi": pw.power_at_sesoi,
        "min_detectable": pw.min_detectable,
        "target": pw.target,
        "assumptions": pw.assumptions,
    }

    # ---- Pass C ----------------------------------------------------------
    _echo("Pass C -- 5 conditions x difficulty")
    c_path = stage_c.artifact_path(cfg)
    if c_path.exists():
        print(f"cached: {c_path}")
        pass_c_frame = read_parquet(c_path)
    else:
        pass_c_frame = stage_c.run_pass_c(cfg, pairs, runner_factory(), prov)
        print(f"wrote: {c_path}")

    # The pooling guard: artifacts from different devices or dtypes are not
    # comparable and must not be combined.
    assert_poolable([pass_a_frame, pass_c_frame], context="Pass A + Pass C")

    print(f"{len(pass_c_frame)} trials   "
          f"digit_mass min={pass_c_frame['digit_mass_min'].min():.4f}")
    print("\nspread by condition x difficulty:")
    print(
        pass_c_frame.pivot_table(
            index="condition", columns="difficulty", values="spread",
            aggfunc=["mean", "std", "count"],
        ).to_string()
    )

    choice_diag = stage_c.choice_diagnostics(pass_c_frame)
    print("\nchoice / position-bias diagnostics:")
    print(choice_diag.to_string(index=False))
    print(f"overall flip rate across option order: "
          f"{choice_diag.attrs.get('overall_flip_rate', float('nan')):.3f}")
    choice_diag.to_csv(out_dir / f"choice_diagnostics_{cfg.hash()}.csv", index=False)
    results["choice"] = {
        "overall_flip_rate": float(choice_diag.attrs.get("overall_flip_rate", np.nan)),
        "per_template": choice_diag.to_dict("records"),
    }

    # ---- model -----------------------------------------------------------
    _echo("Mixed-effects model and planned contrasts")
    design = mixed.prepare_design(cfg, pass_c_frame)
    print(f"|diff| centring: mean={design.diff_mean:.4f} sd={design.diff_sd:.4f} "
          "(interaction coefficient is spread points per SD of |diff|)")

    idata = mixed.fit(cfg, design, with_item=False, progressbar=True)
    conv = mixed.convergence(cfg, idata)
    bad = conv[~(conv["rhat_ok"] & conv["ess_ok"])]
    print(f"\nconvergence: {len(bad)} parameter(s) failing "
          f"R-hat<={cfg.analysis.rhat_max} or ESS>={cfg.analysis.ess_min}")
    if len(bad):
        print(bad.to_string())

    contrasts = mixed.contrast_table(cfg, idata, sesoi)
    print(f"\nplanned contrasts (SESOI = {sesoi:.4f}):")
    print(contrasts.to_string(index=False))
    contrasts.to_parquet(out_dir / f"contrasts_{cfg.hash()}.parquet", index=False)
    idata.to_netcdf(str(out_dir / f"posterior_{cfg.hash()}.nc"))

    primary = contrasts[
        (contrasts["name"] == mixed.PRIMARY) & (contrasts["term"] == "slope")
    ]
    decision = primary["decision"].iloc[0] if len(primary) else "unavailable"
    agreement = mixed.artifact_agreement(contrasts)

    results["contrasts"] = contrasts.to_dict("records")
    results["convergence_failures"] = int(len(bad))
    results["artifact_agreement"] = agreement
    results["primary"] = primary.to_dict("records")[0] if len(primary) else {}
    results["outcome"] = f"primary-{decision}"

    _echo(f"PRIMARY TEST ({mixed.PRIMARY}) x |diff|  ->  {decision.upper()}")
    if len(primary):
        row = primary.iloc[0]
        print(f"  median {row['median']:+.4f}   "
              f"{int(cfg.analysis.hdi_prob * 100)}% HDI "
              f"[{row['hdi_low']:+.4f}, {row['hdi_high']:+.4f}]   "
              f"P(<0) = {row['p_negative']:.3f}")
        print(f"  SESOI = {sesoi:.4f}  |  exceeds SESOI: {bool(row['exceeds_sesoi'])}  "
              f"|  inside ROPE: {bool(row['inside_rope'])}")
    if agreement:
        print(f"\nartifact cross-check: {agreement['route_a']} = "
              f"{agreement['estimate_a']:+.4f}, {agreement['route_b']} = "
              f"{agreement['estimate_b']:+.4f}, discrepancy = "
              f"{agreement['discrepancy']:+.4f}")

    if robustness:
        _echo("Robustness model: item random effect via pair->item multi-membership")
        idata_item = mixed.fit(cfg, design, with_item=True, progressbar=True)
        c_item = mixed.contrast_table(cfg, idata_item, sesoi)
        print(c_item.to_string(index=False))
        c_item.to_parquet(out_dir / f"contrasts_item_{cfg.hash()}.parquet", index=False)
        idata_item.to_netcdf(str(out_dir / f"posterior_item_{cfg.hash()}.nc"))
        results["contrasts_item_re"] = c_item.to_dict("records")

    # ---- figures ---------------------------------------------------------
    if make_plots:
        plots.use_paper_defaults()
        fig_dir = out_dir / "figures"
        for dark in (False, True):
            suffix = "dark" if dark else "light"
            plots.save(
                plots.interaction_plot(cfg, design, idata, dark=dark),
                fig_dir / f"interaction_{suffix}_{cfg.hash()}.png",
            )
            plots.save(
                plots.forest_plot(cfg, contrasts, sesoi, term="slope", dark=dark),
                fig_dir / f"forest_slope_{suffix}_{cfg.hash()}.png",
            )
            plots.save(
                plots.forest_plot(cfg, contrasts, sesoi, term="intercept", dark=dark),
                fig_dir / f"forest_intercept_{suffix}_{cfg.hash()}.png",
            )
        print(f"\nfigures: {fig_dir}")

    r = runner_factory._runner
    if r is not None:
        print(f"\nforward passes: {r.n_forward}  (prompt cache hits: {r.n_cache_hits})")

    _write_results(out_dir, cfg, results)
    return 0


def _write_results(out_dir: Path, cfg: RunConfig, results: dict) -> None:
    path = out_dir / f"results_{cfg.hash()}.json"
    path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nresults: {path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run Stage 0 end-to-end for one model.")
    ap.add_argument("--config", required=True, help="path to a run YAML")
    ap.add_argument(
        "--robustness",
        action="store_true",
        help="also fit the preregistered item multi-membership robustness model",
    )
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument(
        "--artifacts", default=None, help="override the artifacts root directory"
    )
    ap.add_argument(
        "--stop-after",
        choices=["pass_a", "pass_b"],
        default=None,
        help="stop early; 'pass_a' runs ratings and the gate only",
    )
    args = ap.parse_args(argv)

    overrides = {"artifacts_dir": args.artifacts} if args.artifacts else None
    cfg = load_config(args.config, overrides)
    return run(
        cfg,
        robustness=args.robustness,
        make_plots=not args.no_plots,
        stop_after=args.stop_after,
    )


if __name__ == "__main__":
    sys.exit(main())
