# Consistency restoration vs truth tracking — Stage 0

Tests whether consistency-restoration and truth-tracking are separate, causally
dissociable mechanisms in instruction-tuned LMs. Target: ARR October 2026.

**Stage 0 is a kill gate.** It asks only whether the behavioural effect exists at all:
does an instruction-tuned LM show spreading-of-alternatives after a free choice, and does
that spreading depend on choice difficulty in a way that requires the model's own
authorship? If the interaction is absent, no later stage is built.

Read [`preregistration.md`](preregistration.md) first. It was frozen before any experiment
code was written and before any model was run, and it is the authority on every design
decision below.

## Quick start

```bash
uv venv --python 3.11
uv pip install -e ".[dev]"

pytest -m "not slow"                       # fast suite
pytest -m slow                             # parameter recovery (fits models)

python -m src.experiments.run --config configs/stage0_qwen2.5-3b.yaml
jupyter lab notebooks/stage0_analysis.ipynb
```

One command runs Stage 0 end-to-end for one model: Pass A → validity gate → σ_between /
SESOI → Pass B → power simulation → Pass C → mixed model, contrasts, figures. Every stage
is skipped when its artifact already exists under the current config hash.

## What the design guarantees

| Constraint | How it is enforced |
|---|---|
| Every DV from **one forward pass at one token position** | no `.generate()` anywhere under `src/`, asserted by a test |
| **bf16 only**, no 4/8-bit | loader raises if the dtype is not bf16; there is no quantized path |
| Ratings are an **expected value over digit logits**, never argmax | `readout/expected_value.py`; tested against a bimodal case where argmax is wrong |
| All randomness seeded, counterbalancing **by construction** | Pass A and Pass C are complete crossings; the only seeded draw is exactly balanced within stratum |
| No pooling across devices | `provenance.assert_poolable` raises on any device or dtype mismatch |

### The 1–9 scale

The spec asked for 1–10. Measured on the actual tokenizers, `"10"` is two tokens on
Qwen2.5 and Gemma-2 **and its first token is byte-identical to `"1"`**, so a 1–10 scale
cannot be read at one token position on three of the five models — the one-position
constraint and the 1–10 scale were mutually unsatisfiable. At 1–9 every rating is a single
collision-free token on every tokenizer. Descending scores reverse as `10 − x` (not
`11 − x`, which belongs to a 1–10 scale and would bias every collapsed mean by +0.5).

### The five conditions

Receipt is matched across all five — every condition states that the model receives the
designated item, otherwise `3p-yoked − yoked` would confound endorsement with ownership.

| condition | designated item | authorship | endorsement |
|---|---|---|---|
| `chose` | model's own pick | self | — |
| `yoked` | model's own pick | none | none |
| `3p-yoked` | model's own pick | other | other person |
| `3p-random` | random | other | other person |
| `random` | random | none | none |

`yoked`/`random` and `3p-yoked`/`3p-random` use **byte-identical wording** and differ only
in which item is designated; a test enforces it. That is what licenses reading
`yoked − random` as a pure selection-artifact estimate.

**Primary test:** `(chose − yoked) × |diff|`, predicted negative, `|diff|` continuous. A main
effect of agency is reported but is *not* the test — on its own it is fully consistent with
context-window sensitivity, which is how the prior version of this claim was eliminated.

## Layout

```
preregistration.md            frozen before any experiment code
configs/                      base.yaml + one YAML per model; config hash in every filename
src/config.py                 typed config, deterministic hash
src/provenance.py             device/dtype/revision/seed/git capture + pooling guard
src/stimuli/                  400 items (4 domains), 5 templates, invariant assertions
src/readout/                  digit map, expected-value readout, choice readout
src/models/runner.py          bf16 device-agnostic loading, one-position batched forward
src/experiments/              pass_a, pass_b, pass_c, run (the one command)
src/analysis/                 reliability, mixed model, power, plots
artifacts/<stage>/<model>/<config_hash>/
notebooks/stage0_analysis.ipynb
```

## Hardware split

- **Dev (M1, MPS/CPU):** scaffolding, analysis, plots, smoke tests at ≤0.5B.
- **Run (RTX 2000 Ada, CUDA, bf16):** all reported numbers.

`provenance.assert_reportable` rejects any artifact that is not CUDA + bf16 + a pinned
commit SHA + a clean tree, so dev-machine output cannot reach the paper. `smoke: true` is
part of the config hash, so smoke artifacts land in their own directory.

`configs/wiring_check_*.yaml` disables the gate thresholds purely to exercise Pass B/C code
paths. It is an engineering artifact, not a result, and it carries `smoke: true`.

## Status

Scaffolding complete; all reported numbers still to be produced on the run machine.

**Open blocker.** In M1 smoke runs, all three locally cached models — Qwen2.5-0.5B-Instruct,
Qwen2.5-3B-Instruct, Gemma-2-2b-it — fail the preregistered polarity-validity gate
(median ρ = −0.95, −0.49, −0.95). They ignore reversed anchor definitions and answer on a
fixed higher-is-better mapping in both polarities. Ascending-only ratings retain usable
signal (`ICC(C,3)` on the selection score 0.71–0.85, `σ_between` 0.43–0.58), so the
instrument works — it is the polarity crossing, which the preregistration uses as its
validity device, that fails. Resolving this requires a decision about the preregistered
exclusion criterion, not a code change.

## Out of scope for Stage 0

No steering hooks, no probes, no CAA, no knowledge-conflict paradigm, no truth-tracking
stimuli, no free-text valence exposure (replaced by the calibrated `3p-*` conditions), no
open-ended generation anywhere. Stage 0 is a kill gate; if the interaction is absent, that
code is wasted.
