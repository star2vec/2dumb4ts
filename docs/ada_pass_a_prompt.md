# Startup prompt — RTX 2000 Ada, full Pass A ladder

Paste everything below the line into Claude Code on the run machine, from an empty
directory. Replace `<REPO_URL>` with the clone URL.

---

I'm running the Pass A stage of an interpretability experiment on this machine
(RTX 2000 Ada, 16GB, CUDA). This box produces all reported numbers for the paper;
my Mac is dev-only and its output is explicitly non-reportable.

**Goal:** run Pass A — item ratings — for all five models in the ladder, stopping
before Pass B/C. This run exists to answer one question: *does the rating
instrument discriminate between items at full sample, or does it squash everything
into "pretty good"?* On M1 smoke runs at 12 items/domain, three models compressed
all ratings into roughly 7.4–9.0 and all three failed the preregistered
scale-polarity gate. I need to know whether that survives at 400 items, and
whether it also holds for the two models I could not test locally
(Qwen2.5-1.5B-Instruct and Llama-3.2-3B-Instruct).

## Setup

```bash
git clone <REPO_URL> && cd 2dumb4ts
uv venv --python 3.11
uv sync --extra dev || uv pip install -e ".[dev]"
```

`pyproject.toml` already routes torch to the CUDA 12.6 index on Linux via
`[tool.uv.sources]`, so no manual torch install should be needed.

Llama-3.2-3B-Instruct is a gated repo. If you aren't already authenticated:

```bash
huggingface-cli login
```

## Run it

```bash
python scripts/preflight.py            # seconds; tokenizers only, no weights
python scripts/preflight.py --download # ~22 GB of weights
./scripts/run_pass_a_ladder.sh
```

`preflight.py` must pass before the ladder starts — it verifies CUDA and bf16,
that all five pinned revision SHAs resolve and match, that every tokenizer
supports the 1–9 one-position digit readout with no cross-digit collisions, disk
space, and git state. The ladder script refuses to run if preflight fails.

Expect roughly 20–45 minutes of compute for all five models (4,000 forward passes
each, ~76 tokens per prompt, ~300k prefill tokens per model). Download time
usually dominates.

## Rules — these are not negotiable

- **Do not loosen or edit any gate threshold.** They are preregistered in
  `configs/base.yaml`. Models being excluded is an expected, informative outcome,
  not a problem to work around. Exit code 2 means "excluded by a preregistered
  criterion" and the ladder script correctly continues past it.
- **Do not run Pass B or Pass C.** `--stop-after pass_a` is deliberate. Stage 0 is
  a kill gate and the later stages are not to be run until the instrument question
  is settled.
- **Do not commit anything, and keep the working tree clean.** `assert_reportable`
  rejects artifacts from a dirty tree, which would invalidate the whole run.
- `configs/wiring_check_*.yaml` and `configs/smoke_*.yaml` are engineering
  artifacts, not experiments. Don't run them here.
- If something fails, report the failure. Don't route around it.

## What I need back

Run `python scripts/summarize_pass_a.py` at the end (the ladder script already
does) and report:

1. The full ladder table.
2. **Rating range and IQR per model.** This is the headline. An ascending-scale
   IQR under ~1.0 point means the instrument is compressing and the difficulty
   manipulation has no room to work.
3. **`median_rho` per model** — scale-polarity validity. Negative means the model
   ignores reversed anchors and answers on a fixed higher-is-better mapping.
   I want to know whether this is universal or whether any model tracks polarity.
4. `icc_c1_asc` and `icc_c3_sel` (reliability) and `sigma_asc` vs `sigma_coll`.
5. Which models cleared the gate, and every exclusion reason printed verbatim.
6. `digit_mass_p05` — if this drops well below ~0.9 for any model, the readout
   position may be wrong for that tokenizer and I need to know immediately.

Then tell me how to copy the artifacts back:

```bash
rsync -av artifacts/ <me>@<mac>:~/Developer/2dumb4ts/artifacts/
```

Background on the design is in `preregistration.md` (frozen before any code was
written) and `README.md`. Read `preregistration.md` §4 and §5 before interpreting
anything.
