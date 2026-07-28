# Consistency restoration vs truth tracking — Stage 0

Does an instruction-tuned language model, after making a choice, shift its stated preference
*toward what it picked and away from what it rejected* — and does that shift depend on how
hard the choice was?

In humans this is **spreading of alternatives**, the classic signature of dissonance
reduction. The project's larger question is whether **consistency restoration** and
**truth tracking** are separate, causally dissociable mechanisms in LMs. Stage 0 is the
**kill gate**: it asks only whether the behavioural effect exists at all. If it does not, no
later stage is built.

---

## Result

**Stage 0 is complete. All three surviving models returned `inconclusive`. Stage 1 is not
entered.**

| model | λ (chose − yoked) | 95% HDI | decision |
|---|---|---|---|
| gemma-2-2b | −0.0015 | [−1.08, +1.09] | inconclusive |
| qwen2.5-1.5b | −0.1367 | [−1.00, +0.65] | inconclusive |
| llama-3.2-3b | **+0.4361** | [+0.21, +0.70] | inconclusive |

Two further models were excluded at the reliability gate. llama's effect is credible but in
the **wrong direction**, and it does not survive a preregistered robustness restriction
(A4.4).

**The important part is why.** The study could not have detected an effect at the smallest
size it declared interesting, and the cause is diagnosed rather than guessed:

- Realised precision was **3.4–5.4× worse** than the blind power analysis predicted (A4.2).
- The design selects near-equal item pairs on the instrument scale `θ`, and gets them. **The
  outcome measure then answers those same pairs at 97.7% confidence** (A4.7). The power
  calculation assumed difficult pairs sit near p = 0.5 "by construction"; they do not. That
  alone costs **11× the information per observation**.
- Instrument and outcome are elicited by **different questions** and agree only 66–75% even
  on easy pairs (A3.12).

So **H1 is untested, not disconfirmed** — and the obstacle is a design property, not a
sample size. More data does not fix it.

---

## How to read this repository

| file | what it is |
|---|---|
| [`preregistration.md`](preregistration.md) | **The authority.** §1–13 were frozen before any experiment code existed. Four appended amendments record everything found since. Read §1–2, then the amendment headers. |
| [`PREREGISTRATION_LEDGER.md`](PREREGISTRATION_LEDGER.md) | Every preregistered element and its fate — kept, amended, superseded, withdrawn, violated. Start here for the honest count. |
| [`RETRACTIONS.md`](RETRACTIONS.md) | Claims we made and then refuted, with the cause of each. |
| [`STATUS.md`](STATUS.md) | Generated from provenance-stamped artifacts by `scripts/status.py`. Never hand-written. |
| [`preregistration_stage1.md`](preregistration_stage1.md) | Drafted blind, never triggered — the gate was not met. |

Artifacts and the full result set live on the **`stage0-passc`** branch:

```bash
git fetch origin && git checkout origin/stage0-passc -- artifacts results STATUS.md
```

---

## The design, as it actually is

The preregistration was amended four times and §1–13 no longer describes the running
system. What follows is current; the ledger records every difference.

**Pass A — the instrument.** Each item is compared against fixed anchors, both option
orders, five paraphrase templates. Bradley-Terry with a per-template position term `β` gives
each item a scale value `θ`. The original absolute 1–9 rating instrument was **retired**
(A1.1) after every model tested ignored reversed scale anchors — but it is still run at full
scale and reported as a validation record (A1.5), so the switch rests on evidence rather
than on a pilot.

**Pass B — pairs.** Within domain, 100 difficult (near-equal `θ`) and 100 easy pairs per
model, matched on mean rating so difficulty is not confounded with extremity.

**Pass C — the experiment.** For each pair the model states a preference, is told it chose /
was assigned / a third party chose, then states its preference again. **Eight conditions**
(A2.9.3), including a 2×2 separating transcript structure from attribution wording.
`yoked`/`random` and `3p-yoked`/`3p-random` use byte-identical wording and differ only in
which item is designated; a test enforces it, and that is what licenses reading
`yoked − random` as a pure selection-artifact estimate.

**The test.** `λ_chose − λ_yoked`, predicted negative — the *interaction* of agency with
difficulty. A main effect alone is not evidence: it is exactly what context-window
sensitivity predicts, and is how the published rebuttal eliminated the prior version of this
claim (§1.1). **There is a large main effect in these data, and the design declines to
interpret it.** That refusal is the design working.

---

## What the design guarantees

| Constraint | How it is enforced |
|---|---|
| Every DV from **one forward pass at one token position** | no `.generate()` under `src/`, checked per line, with a planted-violation control |
| **bf16 only**, no quantisation | the loader raises if the dtype is not bf16; there is no quantised path |
| All randomness seeded; counterbalancing **by construction** | complete crossings; the one seeded draw is exactly balanced within stratum |
| No pooling across devices | `provenance.assert_poolable` raises on device or dtype mismatch |
| Reported numbers originate on the run machine | `assert_reportable` requires CUDA, bf16, a pinned revision, a clean tree, and non-smoke |
| An artifact cannot be silently stale | every expensive artifact is keyed on the **source digest** of the code that writes it, not only on the config hash |

**Hardware split.** Dev (M1, MPS/CPU) for scaffolding and analysis; run machine
(RTX 2000 Ada, CUDA, bf16) for every reported number. `assert_reportable` makes the
separation mechanical rather than a convention.

---

## Layout

```
preregistration.md              frozen at 57ef60d + four amendments
configs/                        base.yaml + one YAML per model
src/config.py                   typed config, deterministic stage hashes
src/provenance.py               capture + pooling and reportability guards
src/stimuli/                    400 items (4 domains), 5 templates, invariant assertions
src/readout/                    digit map, expected-value readout, choice readout
src/models/runner.py            bf16 loading, one-position batched forward
src/experiments/                pass_a, pass_a_pairwise, pass_b, pass_c, run
src/analysis/                   bradley_terry, spread_model, power, plots
src/archive/                    retired modules, kept so their numbers stay traceable
scripts/                        status, sensitivity, pilot and validity checks
tests/                          281 tests; see below
```

---

## Reproducing

```bash
uv sync --extra dev
pytest -m "not slow"        # 265 tests
pytest -m slow              # 16 more; fits models, needs cores=1 on Windows

python -m src.experiments.run --config configs/stage0_gemma-2-2b.yaml
jupyter lab notebooks/stage0_analysis.ipynb
```

One command runs a model end to end. Each stage is skipped when its artifact exists under
the current keys.

---

## Out of scope for Stage 0

No steering hooks, no probes, no CAA, no knowledge-conflict paradigm, no truth-tracking
stimuli, no open-ended generation anywhere. Stage 0 is a kill gate; the gate was not passed,
so none of it was built.

---

## A note on the process record

This repository keeps its mistakes on purpose. `RETRACTIONS.md` lists claims withdrawn, the
ledger lists preregistered elements abandoned, and the amendments record occasions where a
change would have helped the hypothesis and was **declined** (A3.6), or was made and flagged
as an **override** rather than dressed as a pass (A4.6).

Some of it is uncomfortable: a power analysis falsified by its own data (A4.2), a
preregistered constant that went two amendments without being implemented (A3.9), a choice
prompt that showed the model neither option, and a decision rule that is unsatisfiable as
written (A3.1). Every one was found before it could change what Stage 0 reports.
