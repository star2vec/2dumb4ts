# Preregistration — is preference magnitude represented but not expressed?

**Standalone. Not Stage 1, not Stage 0-bis.** Written 2026-07-29, before any activation has
been collected or read.

---

## 0. Evidentiary status — what has and has not been seen

**Blind:** no activations exist. Nothing in this document has been informed by any hidden
state, and no probe has been fitted.

**Not blind, and stated so it cannot be claimed later:**

- Stage 0 is complete and its result is known — three models, all `inconclusive`.
- `|diff|`, the output readouts, and every number in A4.2–A4.8 have been seen.
- **This hypothesis was generated post-hoc from A4.7.** It was not anticipated by the Stage 0
  preregistration and does not appear in it. It exists because a check written for a
  different purpose returned an unexpected result.

That origin does not invalidate the study — post-hoc hypotheses are how most research
starts — but it does mean this preregistration is the *only* protection against the usual
failure, and it must therefore be specific enough to fail.

**Not blocked by A3.7.** That gate governs entry to Stage 1's dissonance hypotheses, which
are not entered and are not asked here. This asks a different question about the instrument,
and its answer does not bear on H1 in either direction.

---

## 1. The question

Stage 0's dependent variable cannot express difficulty. On item pairs the Bradley-Terry
instrument places at `|Δθ| < 0.16` — no measurable preference — the model answers *"which do
you prefer?"* at **p = 0.977**. Stated confidence correlates with measured preference at
**ρ = 0.093** (A4.7), and position bias accounts for only **5%** of the readout's variance
(A4.8), so this is not an artifact of the elicitation's geometry.

**Two explanations remain, and they are distinguishable.**

- **(E) Elicitation.** The model represents how much it prefers one item, and this readout
  does not expose it. The information is present in the forward pass and destroyed at the
  output.
- **(R) Representation.** The model encodes an *ordering* over items and no *magnitude*. There
  is nothing for the readout to expose.

### Primary claim, framed to survive "probes decode everything"

The headline is **actionable, not representational**:

> **If `|diff|` is linearly decodable from activations at the readout position while the
> output does not express it, the information is present in the forward pass and is
> destroyed at the readout — so the readout is what to change.**

That claim is about *where the information is lost*, not about how preference is encoded, and
it is robust to the standard objection that linear probes recover weak signal from almost
any high-dimensional representation. The stronger representational claim — order encoded,
magnitude not — requires the sign/magnitude contrast in §5 and is licensed only by it.

---

## 2. What is collected

A **replay** pass: the identical Pass C **pre** prompts, re-run to capture hidden states.
No new stimuli, no new conditions, no manipulation. The prompt digest is asserted to match
the recorded Pass C prompts, or the run aborts.

| | |
|---|---|
| rows | Pass C **pre** only — 200 pairs × 5 templates × 2 orders = **2,000 per model** |
| position | the single readout position, the same one the DV is read from |
| layers | **all of them** |
| dtype | bf16, as collected |
| models | **all three**: gemma-2-2b, qwen2.5-1.5b, llama-3.2-3b |

**All layers, not a subset.** A3.5's per-pass figures already are all layers
(gemma 2304×27×2 = 122 KB, qwen 1536×29×2 = 87 KB, llama 3072×29×2 = 174 KB — verified).
At 2,000 rows that is **237 + 170 + 340 = 747 MB** for the three models. Subsetting would buy
nothing and would leave a negative result open to "you probed where you chose to look."

**Pre rows only.** They precede every manipulation, so nothing collected here can bear on
H1. Post rows would ask whether the manipulation changes the representation — a larger
question that touches Stage 1's territory, and it is not asked here.

**Folded with the `p_item1` re-collection.** A4.7's saturation is measured on gemma alone;
llama's precision matched prediction (1.05×), implying it is far less saturated, but that is
inference because `p_item1` exists only for gemma. Both need the same forward pass. **One
run, two results.**

**llama is the scientifically decisive case.** If gemma's output is saturated and llama's is
not, and `|diff|` decodes in both, the claim is that the information is present and one
readout destroys it. A single-model result cannot separate that from "gemma is unusual."

---

## 3. The probe

**Target:** `|diff|` — `|θᵢ − θⱼ|` on the analysis template split, the same quantity the
Stage 0 primary uses as its regressor.

**Class:** ridge regression, one probe per layer, on the raw bf16 activation vector cast to
float32. No feature selection, no dimensionality reduction — both would introduce a second
place for the item-identity confound (§4) to enter.

**Regularisation:** α selected by **inner** cross-validation, on inner folds that respect the
same item split as the outer folds. Selecting α on folds that share items with the training
set leaks the confound into the hyperparameter and is the most likely way to get a
spuriously good result without noticing.

**Metric:** Spearman ρ between predicted and true `|diff|` on held-out data, reported with a
bootstrap interval over **pairs** (§4).

---

## 4. The confound that decides this study, and the split that controls it

**`|diff|` is a property of the item pair, and the activations have seen both items.** A probe
can recover `|diff|` by recognising *which pair this is* rather than by reading any
represented preference. That probe would decode beautifully and mean nothing.

**Primary design constraint: hold out ITEMS, not rows.**

- Partition the **item pool**, not the trials.
- A pair goes to test if **either** of its items is in the held-out set.
- Training therefore never sees an item that appears at test, in any pair, under any
  template or order.
- 5 outer folds over items. With ~25% of items held out, roughly 44% of pairs move to test,
  leaving **~112 training pairs**.

A probe that decodes across an item split has learned something that generalises over items.
A probe that decodes only within-item has learned identity. **Both are reported**; only the
first licenses any claim.

### Clustering: the effective sample size is 200, not 2,000

Each pair contributes **10 rows** — 5 templates × 2 orders — and all ten share the *same two
items*. `|diff|` is **identical** across all ten by construction. Row-level intervals would
therefore be badly overstated, and every resampling unit must be the **pair**:

- bootstrap over pairs, never rows;
- cross-validation folds split on **items** (which implies pairs);
- report **effective n = 200 per model**, alongside the row count, wherever an interval is
  given.

**On the justification.** This is *not* the template non-independence of R4 — that claim was
**retracted** (`RETRACTIONS.md` R4: the ICC of 0.529 conflated genuine between-cell variation
in `p` with excess dependence; the proper over-dispersion test returned **1.070, 95% CI
[0.907, 1.258]**, consistent with independence, and no design effect exists). The retracted
√3.12 ≈ 1.77 inflation must not be carried into this document.

The dependence here is **structural and exact**: ten rows share two items and one target
value. It needs no ICC estimate and does not depend on R4 being right or wrong.

**~112 training pairs against 2,304 features is underdetermined**, which is why ridge is
specified rather than ordinary least squares, and why α is selected rather than fixed.

---

## 5. The positive control — pre-specified because a null is likely

A null probe is uninterpretable on its own: it cannot distinguish "the model does not
represent magnitude" from "the probe, the layer choice, or the 112 training pairs were
inadequate." `preregistration_stage1.md` §5 already requires that a null count only if a
positive control succeeds on the same activations. That rule is carried over here, and the
control is the one already present in the data.

**The output expresses ORDER but not MAGNITUDE.** So run two probes on **identical
activations, identical probe class, identical item-held-out folds**:

| probe | target | expectation |
|---|---|---|
| **control** | `sign(θᵢ − θⱼ)` — which item is preferred | expected to decode |
| **question** | `\|θᵢ − θⱼ\|` — by how much | the study |

**The comparison is pre-specified, not an absolute threshold.** "Magnitude decodes about as
well as sign" interprets itself. "`|diff|` decodes at ρ = 0.31" does not, because linear
probes on 2,304 dimensions recover weak signal from nearly anything.

---

## 6. Pre-specified outcomes

Read at the best layer by the control probe, with the item-held-out split, bootstrapped over
pairs.

| result | reading |
|---|---|
| sign decodes **and** magnitude decodes comparably | **(E) Elicitation.** The information is in the forward pass and destroyed at the readout. The primary claim holds: change the readout. |
| sign decodes, magnitude **does not** | **(R) Representation.** Order is encoded, magnitude is not. The sharper claim, and the control licenses it. |
| **neither** decodes | Uninterpretable. The probe or the collection is inadequate; report as a failed measurement, not as evidence for (R). |
| magnitude decodes only **within** item split | Item identity. Reported as a negative result for both (E) and (R). |

**No threshold is set on the magnitude probe's ρ in isolation**, deliberately. Every reading
above is a *comparison* against the control on the same activations.

---

## 7. What is not claimed

- Nothing about H1. Pre rows carry no manipulation and no condition.
- Nothing causal. A probe shows information is *present*, never that the model *uses* it.
- Nothing about other elicitation formats. If (E) holds, "change the readout" is a direction,
  not a validated remedy — testing a replacement is a further study.

---

## 8. Order of work

1. This document is sent and fixed **before any activation is collected**.
2. Replay collection, all three models, pre rows, all layers, `p_item1` persisted in the same
   pass.
3. Prompt-digest assertion against the recorded Pass C prompts.
4. Control probe, then magnitude probe, on identical folds.
5. Report both, with effective n, whichever way they fall.
