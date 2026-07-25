# Pass C assumption audit — what the pairwise switch breaks

Requested before any Pass C code is written. Audited against the implementation at
`97cc217`, not from memory. Severity tiers: **T1** breaks the science, **T2** is
mechanical but must be fixed before Pass C runs, **T3** is dead or mislabelled code.

---

## T1 — The one that matters: the DV is still measured with the retired instrument

`src/experiments/pass_c.py` computes

```
spread = (designated_post − designated_pre) − (other_post − other_pre)
```

and all four terms come from `read_expected_value` on the **absolute 1–9 scale**.
D1 retired that instrument for item scoring because it compressed variance. Nothing
about Pass C makes it work better there — it makes it work **worse**, because the DV
is a difference of differences of four compressed measurements, so it is four times
more exposed to readout noise than a single rating.

The arithmetic, using Qwen2.5-3B's own pilot numbers (σ_between = 0.581,
ICC(C,1) = 0.206):

| quantity | value |
|---|---|
| implied per-rating measurement SD | `0.581 × √((1−0.206)/0.206)` = **1.14** |
| implied spread noise SD (4 ratings) | `2 × 1.14` = **2.28** |
| SESOI on the interaction | `0.15 × 0.581` = **0.087** |
| noise-to-SESOI ratio | **≈ 26 ×** |

Detecting 0.087 against noise of 2.28 at 80% power needs roughly 5,400 effective
observations; the primary contrast has ~4,000 trials. That is the same conclusion
the pilot power simulation reached independently (power 0.07 at SESOI), and it is
not a sample-size problem — it is an instrument problem, in the one place the
instrument matters most.

**This decision is blocked on Task 4, not merely pending.** If the operating-window
diagnostic finds no band where choices are content-driven *and* pairs are
near-equal, Pass C dies and the DV question is moot. Do not settle it first.

### Four options, with what each costs and what each threatens

**(a) Keep absolute ratings for the DV.** 4 readouts/trial, ~26k forward passes per
model. Cheapest, and contradicts D1 in the one place D1 matters most. The table
above is the argument against it.

**(b) Full anchor-based θ for each of the four ratings.** Faithful to D1. Each
rating becomes 10 anchors × 2 orders = 20 readouts, so 80 per trial plus the choice:
**~482k forward passes per model** (pre 80k, post 400k), hours per model and several
nights for the ladder. Preserves everything; expensive.

**(c) Direct pairwise DV.** Measure the chosen-vs-rejected comparison *itself*, pre
and post, using the label logit as a continuous readout:

```
spread = logit p_post(designated ≻ other) − logit p_pre(designated ≻ other)
```

order-averaged. Cost is 2 timepoints × 2 orders = **4 readouts/trial — the same
budget as today** — and it measures the quantity the DV is actually about
(divergence between the two options) rather than reconstructing it by subtracting
two absolute scores.

**Two serious threats, and the first is close to disqualifying:**

1. **Heteroscedastic sensitivity that manufactures H1.** The logit scale has
   maximal sensitivity at p ≈ 0.5 and compresses at the extremes. Difficult pairs
   sit near p = 0.5 by construction; easy pairs sit near the extremes. So *any*
   perturbation — a real effect, or pure noise — produces larger logit movement in
   difficult pairs than easy ones. That is exactly the predicted direction of the
   primary interaction. The DV would generate H1 with no mechanism behind it.
   Mitigations: analyse on the probability scale (uniform sensitivity, but bounded,
   so easy pairs hit a ceiling); model the variance structure explicitly; or
   require that the interaction survive a scale transformation — an effect present
   on logit and absent on probability is a scale artifact, not a finding. None of
   these is free, and this is the same class of error as the extremity/ceiling
   confound that design-level matching was introduced to remove.
2. **Loses the bolstering/derogation decomposition.** The four-rating design can
   say whether the chosen item rose or the rejected item fell; a single directional
   comparison cannot. That distinction is substantive in the dissonance literature.

**(d) Reduced-anchor θ.** Each rating scored against a 3-anchor subset × 2 orders =
6 readouts, so 24 per trial: **~150k forward passes per model**. Keeps D1's
instrument, keeps the decomposition, keeps the DV on a scale whose sensitivity does
not co-vary with difficulty, at ~6× rather than ~20× the current cost. Precision per
rating is lower than 20 comparisons, but the DV averages over thousands of trials
and per-rating precision is not the binding quantity.

**Recommendation: (d), decided after Task 4.** It is the only option that satisfies
D1 without either the logit-scale artifact of (c) or the multi-night cost of (b).
But it is a preregistration-level choice about the dependent variable and it is
yours, not mine.

---

## T2 — Mechanical breakages, all of which would fire on the first Pass C run

**T2.1 `run.py` still gates on the retired polarity criterion.** `run.py:118` calls
`evaluate_gates`, which runs `validity_table` + `collapse_polarity` and can HALT on
Spearman ρ < 0.6. A1.2 retired that gate. As it stands the orchestrator would
exclude **every** model on a criterion that no longer exists. Must be rewired to the
order-invariance gate in `pass_a_pairwise.evaluate_order_invariance_gate`.

**T2.2 Pass B is fed rating-scale scores.** `run.py:155` calls
`item_scores(cfg, pass_a_frame)`, which returns polarity-collapsed absolute ratings
as `score_selection` / `score_analysis`. These must come from the Bradley-Terry fit,
fitted separately on the two disjoint template splits so the
selection-vs-analysis independence of §4.4 is preserved. `pass_b` itself needs no
change — it only consumes those two columns — which is the one piece of good news
in this section.

**T2.3 Thresholds are in rating points and do not transfer.**
`match_tolerance = 0.15` and `sigma_between_min = 0.5` (`config.py:146,161`) are
absolute-scale quantities. θ is a logit-scale latent whose location is fixed only by
the zero-sum anchor constraint and whose scale is set by the estimated `sigma_item`.
0.15 rating points against σ_between = 0.58 was 0.26 SD; 0.15 on θ against an
unknown σ_item is not the same quantity. Both must be re-expressed as fractions of
`sigma_item`, or as quantiles of the realized θ distribution. A1.8 explicitly
deferred this and it is still open.

**T2.4 SESOI cannot be re-expressed yet.** `sesoi_primary = 0.15 × σ_between`
applies to the interaction coefficient *on the spread DV*, whose units depend on
which T1 option is chosen. It is therefore blocked behind T1.
`sesoi_raw_secondary = 0.25` "raw rating points" becomes meaningless outright and
needs either a θ-scale replacement or deletion.

**T2.5 The spread bound check is scale-specific.** `pass_c.py:271` asserts
`|spread| ≤ 2 × (scale_max − scale_min)` = 16. Under a θ or logit DV there is no
such bound: the check silently never fires (option b/d) or fires spuriously
(option c). Replace with a DV-appropriate sanity check, and do not simply delete it
— it is the assertion that would catch a sign or join error in the DV construction.

**T2.6 Priors are calibrated to the ±16 spread scale.** `mixed.py:115` documents
`Normal(0,1)` / `HalfNormal(2)` as "weakly informative on the spread scale (bounded
±16, expected ~±2)". On a logit-difference DV the plausible range is different; on a
θ-difference DV it scales with `sigma_item`. Priors are preregistered in §7.1, so
this is an amendment item, not a silent retune.

**T2.7 The power simulation's noise model is invalid in three ways.**
`power.py:134-135` derives `sd_err = σ_between × √((1−ICC)/ICC)` and
`sd_spread = 2 × sd_err`. This assumes (i) rating-point units, (ii) ICC as the
reliability measure — but A1.1 replaced ICC with the posterior SD of θ, and (iii)
that the DV is a difference of four independent ratings, which is false under option
(c). The derivation must be rebuilt from the posterior SD of θ.

**T2.8 D3 is not yet applied to Pass C, and t4's label rendering is degraded there.**
`base.yaml` still sets `option_labels: [A, B]`, and Pass C's choice elicitation
(`choice_messages`, `post_messages`) uses it. This is the breakage that most
directly touches the primary contrast, because the choice is what defines yoking.

Separately, `stimuli/build.py:pair_block` renders options with each template's
**native** format, so the standardisation now in `readout/pairwise.py` (`lead_in`)
has not been applied there. Measured directly on a Pass C-style choice prompt under
digit labels:

| template | readout mass | rendering |
|---|---|---|
| t0 | 0.9993 | `1. item` |
| t1 | 0.9951 | `1) item` |
| t2 | 1.0000 | `1 - item` |
| t3 | 1.0000 | `1: item` |
| **t4** | **0.8365** | `[1] item` |

*Correcting an overstatement in the first draft of this audit:* I claimed t4 would
collapse to a median mass of 0.38 in Pass C, transferring the figure measured in the
anchor-comparison context. That transfer was not justified. In the Pass C choice
context t4 measures **0.8365 — degraded relative to the other four templates, but
above the 0.5 floor**. It would not invalidate most of the `chose` condition. The
mechanism is confirmed and the fix is still worth making for consistency and margin,
but this is a T2 tidy-up, not a threat to the condition.

**T2.9 Sign convention for a directional DV.** Under option (c) the comparison is
directional, so "designated ≻ other" must be fixed consistently and the
`designated`/`other` assignment must drive the sign rather than a re-measurement.

---

## T3 — What survives, and what becomes archive-only

**Survives untouched.** The spread DV *structure* (difference of differences); the
2 × 5 receipt-matched conditions with their verbatim-identical antecedents, shared
receipt and balanced mention counts; the disjoint template split for
selection-vs-analysis; design-level matching on mean pair score; continuous
difficulty with the interaction as the sole primary test; cell-means coding with
zero-sum constrained random effects; the item multi-membership robustness model; the
cross-device pooling guard; stage-scoped cache keys; and the whole counterbalancing
structure.

**Pre-measurement reuse survives, which is worth stating explicitly.** Pre is
measured once per (pair, template, option order) and reused across all five
conditions. Under options (b) and (d) this still holds — θ for both items is
measured before the manipulation. Under option (c) the physical measurement
(item1 vs item2) is also shared; only its *sign* changes with designation. So the
5× saving is preserved in every option.

**Becomes archive-only, and must not sit on the live path.**
`reverse_polarity`, `collapse_polarity`, `validity_table` and the polarity branch of
`evaluate_gates` are still needed for the instrument-validation record (A1.5) but
must not gate anything. `scripts/summarize_pass_a.py` calls `evaluate_gates`, which
is correct for the archive and would be wrong if read as a live gate — it needs a
banner saying so.

**Tests asserting retired behaviour** — `test_reverse_polarity_maps_endpoints_and_midpoint`,
`test_polarity_collapse_*`, `test_gate_excludes_a_polarity_blind_model`,
`test_validity_is_*` — should be **relabelled as instrument-validation tests, not
deleted**. The polarity arithmetic still has to be right for the archive; it just no
longer decides anything.

**`digit_map` becomes unused on the live path** under options (b)–(d): only
`label_map` is needed for pairwise readout. It remains required for the
instrument-validation record.

---

## Recommended order

1. **Task 4 first.** The operating window decides whether Pass C is viable at all.
   Everything in T1 is downstream of it.
2. **T2.1, T2.2, T2.8 next** — they are unambiguous and independent of the DV choice.
   T2.8 in particular should be fixed before *any* further choice readout runs.
3. **T1 decision**, then T2.3–T2.7, which are all downstream of it, as Amendment 2.
4. Only then write Pass C.
