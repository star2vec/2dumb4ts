# Stage 1 Preregistration — DRAFT

**Status: DRAFT, not frozen.** Written 2026-07-27, while blind to Pass C. The Stage 0 run
producing the first H1 estimate is in flight and no λ has been observed. This document is kept
separate from `preregistration.md` because that file's §1–13 and Amendments 1–2 are frozen and must
not be edited to accommodate a later stage.

Stage 1 freezes as its own commit, before any Stage 1 code, exactly as Stage 0 did.

---

## 1. What Stage 1 claims

Stage 0 asks whether the behaviour exists: does a free choice produce spreading of alternatives, and
does that spreading depend on choice difficulty. Stage 1 asks whether there is an **internal state
that carries it**, and whether that state is **causally responsible** rather than merely correlated.

**H2 (read).** There exists a linear direction in the residual stream whose projection tracks the
difficulty-conditional consistency pressure — the same interaction Stage 0 estimates, not the main
effect.

**H3 (write).** Adding that direction to the residual stream shifts the comparison DV **toward the
designated item**, more strongly on difficult pairs than easy ones, and a norm-matched random
direction does not.

### 1.1 What Stage 1 does NOT claim

- It does not claim the direction is *the* mechanism, only that it is sufficient to move the
  behaviour and necessary in the ablation sense tested in §6.3.
- It does not compare consistency-restoration against truth-tracking. That dissociation is Stage 2,
  a separate programme, and no part of this document licenses a claim about it.
- Linear decodability is an assumption about the *readout*, not about the model. §7.3 pre-commits
  what a failed probe does and does not mean, because the temptation to reinterpret a null probe as
  evidence about mechanism is the single most likely way this stage goes wrong.

---

## 2. Dependence on Stage 0, pre-specified as branches

Stage 1's probe target is **not determined yet**, and cannot be, because it depends on which factor
of Stage 0's 2×2 carries the effect. Choosing it after seeing Stage 0 would be legitimate only if
the rule is fixed in advance. It is fixed here.

Recall the 2×2 (Amendment 2 A2.9.3):

| condition | transcript structure | attribution wording |
|---|---|---|
| `chose` | assistant turn present | "you chose" |
| `structure-control` | assistant turn present | "assigned" |
| `self-recounted` | assistant turn absent | "you chose" |
| `yoked` | assistant turn absent | "assigned" |

**Branch A — the 2×2 identifies turn-presence** (`chose − self-recounted` and
`structure-control − yoked` agree, per `spread_model.structure_factor_agreement`). Probe contrast is
**`chose` vs `self-recounted`**: wording held byte-identical, structure varied. This is the strongest
available contrast because the two prompts differ only in whether an assistant turn is present.

**Branch B — the 2×2 identifies wording.** Probe contrast is **`chose` vs `structure-control`**:
structure held constant, wording varied. Weaker, because the surface strings differ and a probe can
decode wording from token identity alone; §4.3's positional control becomes load-bearing rather than
confirmatory.

**Branch C — the two turn-presence estimates disagree** (structure interacts with wording). Then no
single edge is interpretable and there is no identified target. Stage 1 **does not proceed to
steering**. It reports the probe on `chose` vs `yoked` as exploratory, labelled as such, and the
submission is Stage 0 plus a negative methodological result about probe identifiability. This branch
is written down because it is the one where a motivated analyst would quietly pick whichever edge
looked better.

**Branch D — Stage 0's primary is inconclusive or null.** Stage 1 runs H2 only, as a measurement
question ("is there a decodable state even absent a behavioural interaction?"), and H3 is withdrawn.
Steering toward a behavioural effect that was not established is uninterpretable.

---

## 3. Activation collection

### 3.1 A discrepancy, recorded rather than patched

A2.9.6 states that last-token hidden states "are written during Pass C so Stage 1 does not require
re-running it." **This is not implemented.** No module in `src/` collects hidden states; `pass_c.py`
writes parquet only. The Pass C run in flight will therefore not produce the activation cache, and
Stage 1 **does** require re-running Pass C's forward passes.

That is recorded as a discrepancy rather than fixed mid-run, for two reasons. The run machine is
disk-constrained (no second drive; 3B weights required `HF_HUB_DISABLE_XET=1`), so writing ~7 GB
during the run risks failing it. And the dominant Pass C cost is the instrument fit, which is now
cached — re-running the forward passes later is roughly 30–60 minutes per model, which is the cheaper
of the two mistakes.

**Cost of the discrepancy**, for the three models that passed A2.2, at 18,000 observations each
(2,000 shared pre + 16,000 post):

| model | KB/pass | total |
|---|---|---|
| qwen2.5-1.5b | 87 | 1.57 GB |
| gemma-2-2b | 122 | 2.20 GB |
| llama-3.2-3b | 174 | 3.13 GB |
| **total** | | **6.90 GB** |

### 3.2 What is collected

A separate `collect_activations` pass replays the **identical** Pass C prompts — deterministic given
the seed and the stage hash, so correspondence is exact and verified by asserting the prompt digest
matches Pass C's — and writes, per trial:

- residual-stream hidden state at **one token position**: the DV readout position, i.e. the position
  whose logits produce the comparison outcome
- **all layers**, embeddings through final, bf16
- the trial's `pair_id`, `template`, `option_order`, `condition`, `timepoint`, `d`, `s`,
  `diff_analysis`, and the observed `item1_wins`

This satisfies the standing constraint that every DV be readable from one forward pass at one token
position: collecting hidden states is a side channel of the same forward pass, and no generation
occurs. bf16 only, per the standing constraint; float32 storage would double the figures above and
buys nothing, since the forward pass is bf16 regardless.

Provenance is stamped as everywhere else, and `assert_poolable` forbids mixing devices. Activations
collected on a different device from the Pass C outcomes they are paired with are **not poolable**,
so the collection pass must run on the same machine as Pass C.

---

## 4. The probe

### 4.1 Target

The probe predicts the **interaction**, not the condition. Training on the condition label directly
would give a probe that decodes "is this the chose condition", which is decodable from the prompt and
says nothing about consistency pressure — it is the internal analogue of the main effect that the
PNAS rebuttal already defeated at the behavioural level.

Concretely, the labelled contrast is formed **within difficulty stratum** and the probe direction is
the difference of the two strata's difference-of-means:

    v = (mean[chose, difficult] − mean[ref, difficult]) − (mean[chose, easy] − mean[ref, easy])

where `ref` is the branch-specified reference condition from §2. Any component of the representation
that distinguishes the conditions equally at both difficulty levels cancels out of `v` by
construction. That is the point: the surface wording difference is difficulty-independent, so it
subtracts away.

### 4.2 Family

**Primary: difference-of-means.** Chosen over logistic regression because it is the estimator whose
direction is meaningful to *add* — steering along a logistic-regression weight vector conflates the
signal direction with the covariance structure it inverts, and the resulting vector is not the one
the probe's accuracy was measured on. Difference-of-means keeps §5's steering vector and §4's probe
direction the same object.

**Secondary, reported: logistic regression** with L2, penalty chosen on the inner split only. If the
two disagree on which layers carry signal, that is reported; it is not resolved by preferring
whichever supports H2.

### 4.3 Controls, all pre-specified

1. **Positional control.** A probe trained on activations at the *same* position in prompts where
   the manipulation text is present but the DV is not requested must not achieve the primary probe's
   accuracy. If it does, the probe is reading the prompt, not a state.
2. **Norm-matched random direction.** Projection onto a random direction of equal norm, same
   evaluation. This is the null for every accuracy figure.
3. **Difficulty-only probe.** A probe trained to decode difficulty stratum alone, ignoring
   condition. Difficulty is a property of the *pair*, present at `pre`, and so must be decodable;
   if it is not, the activations are not carrying what we think they are and the pipeline is
   suspect. This is a positive control on the collection, not a hypothesis test.
4. **Pre-manipulation floor.** The probe applied to `pre` activations, where no manipulation has
   occurred, must be at chance. A probe above chance at `pre` is decoding pair identity.

### 4.4 Generalisation, two separate tests

Reported separately and never pooled, because they fail for different reasons:

- **Leave-items-out.** Train on 80% of pairs, test on the held-out 20%, 5 folds. Failure means the
  probe memorised items.
- **Leave-templates-out.** Train on the Pass A selection templates (t0–t2), test on t3–t4. Failure
  means the probe is paraphrase-specific and the direction is not the state.

Layer selection is made on the **training folds only**, and the selected layer is reported per model.
Sweeping layers on the test split and reporting the best is the standard way this literature
overstates itself; the sweep is pre-registered as a sweep, and its full curve is reported, not its
maximum.

---

## 5. Steering

Contrastive activation addition: `h ← h + α·v̂`, `v̂` the unit-normalised §4.1 direction, applied at
every token position from the start of the manipulation text onward, at the layer selected in §4.4.
The DV is read exactly as in Pass C — one forward pass, one position, no generation.

### 5.1 Dose

`α ∈ {−2, −1, −0.5, 0, +0.5, +1, +2}` in units of the **within-layer residual-stream norm** at the
readout position, so doses are comparable across models and layers. Zero is included as the
within-run control; the α = 0 arm must reproduce the Pass C outcome, which is a check on the hook.

### 5.2 Validity criteria — all five required

A steering result is interpretable **only if all five hold**. Failing any one means the intervention
is reported as invalid and H3 is not evaluated with it.

1. **Direction-specific.** Norm-matched random directions at the same α produce no effect on the DV.
2. **Dose-monotone.** The DV shift is monotone in α across the swept range, and signed:
   negative α moves the DV *away* from the designated item.
3. **Designation-relative, not slot-relative.** This is the load-bearing one. A vector that merely
   pushes outcomes toward slot 1 would look like an effect while being pure position bias. Because
   the Stage 0 likelihood already separates `β·s` (slot) from `d·(γ + λ·diff_z)` (designation),
   steering is added as a factor to **that same model**, and the criterion is that steering loads on
   the designation term and not on β. The confound is controlled by construction rather than argued
   about.
4. **Readout intact.** The digit/label mass floor (A1.6) must still hold under steering. A vector
   that degrades the readout produces outcome shifts that are artefacts of a broken distribution.
5. **Capability preserved.** Comparison accuracy on the **easy** pairs at `pre` — where the model's
   preference is near-deterministic and no manipulation applies — must not degrade beyond a
   pre-specified 5 percentage points. This substitutes for a fluency check, which is unavailable
   because generation is prohibited in the measurement path.

### 5.3 The primary Stage 1 test

Fit `spread_model` with steering dose as an added factor and test:

    lambda_steered − lambda_unsteered  at  alpha = +1,  predicted NEGATIVE

i.e. steering along `v̂` strengthens the difficulty-conditional shift toward the designated item.
The SESOI and decision rule are inherited from Stage 0 §9, with the Amendment 3 correction: the
**minimum detectable effect at 80% power** is reported, and "80% power at the SESOI" is not used,
because it is unsatisfiable (A3.1). Power is computed on the realised Stage 1 design by
`power.analyze` before the steering runs, blind.

---

## 6. Analysis

1. **Probe accuracy** per layer, per generalisation split, against the norm-matched random null.
   Reported as full curves.
2. **Primary steering contrast** as §5.3.
3. **Ablation.** Projecting `v̂` out of the residual stream should *attenuate* the Stage 0
   interaction. Reported as a third result, not as a hypothesis test: projection removes whatever
   else shares that subspace, so attenuation is consistent with but not diagnostic of necessity.
   Stated this way in advance so it is not later upgraded into a necessity claim.

Convergence, `r_hat ≤ 1.01`, `ess_bulk ≥ 400`, and the device-pooling guard apply unchanged.

---

## 7. Decision rules

### 7.1 H2

**Supported** if the probe exceeds the norm-matched null on **both** generalisation splits, controls
§4.3(1) and §4.3(4) hold, and the positive control §4.3(3) passes.

### 7.2 H3

**Supported** if all five §5.2 criteria hold and the §5.3 contrast's HDI excludes zero in the
predicted direction with median magnitude beyond the SESOI.

### 7.3 The failed-probe rule — pre-committed

If the probe does not exceed its null, **that is not evidence against H1 or against the existence of
a consistency-restoration mechanism.** Linear decodability at one position in one layer is a strong
and possibly wrong assumption about how such a state would be represented. The pre-committed report
is:

> "We found no linear direction at the DV readout position that tracks the difficulty-conditional
> effect, under the probe families and generalisation tests specified in §4. This bounds what a
> linear probe at this position can recover; it does not bound the model."

Specifically **prohibited**, and listed so the prohibition is checkable: reframing a null probe as
evidence for a distributed or non-linear representation (unfalsifiable as stated), sweeping
additional positions or layers after seeing the null and reporting the best, or moving to a
non-linear probe and reporting only that it worked. Any post-null exploration is reported as
exploratory in a separate section with its own multiplicity stated.

### 7.4 Failed steering with a successful probe

A decodable direction that does not steer is a **real and reportable result**: the state exists and
is not causally sufficient at this site and dose. It is written up as such. It is not reported as a
power problem unless the §5 power calculation, computed blind, actually says so.

---

## 8. What a fully positive Stage 1 would NOT establish

- Not that the direction is unique, minimal, or the model's own causal variable — only that it is
  sufficient to move the behaviour and shares a subspace with whatever is.
- Not anything about truth-tracking. Stage 2 is separate.
- Not that the mechanism is dissonance rather than a simpler consistency heuristic. Stage 0's
  `chose-provisional` arm speaks to reversibility; Stage 1 adds no leverage there.
- Not generalisation beyond instruction-tuned models at this scale, four domains, and this
  paradigm.

---

## 9. Costs

| item | cost |
|---|---|
| activation collection, 3 models | ~6.9 GB, ~30–60 min forward passes per model |
| probe fits | cheap; CPU, no sampling |
| steering runs | 7 doses × 18,000 observations × 3 models ≈ 378,000 forward passes |
| Stage 1 spread fits | one per dose-set per model |

The steering sweep is the real cost and it is a forward-pass cost, so it belongs on the run machine.
If it must be cut, the pre-specified reduction is **doses first** (to `{−1, 0, +1}`, retaining sign
and the zero control but losing monotonicity evidence, which then downgrades §5.2(2) from satisfied
to untested), **never conditions and never models**.

---

## 10. Open items before this can freeze

1. §2's branch cannot be resolved until Stage 0 reports. The branch *rule* is frozen here; the
   selection is mechanical once λ and the 2×2 agreement land.
2. The A2.9.6 discrepancy in §3.1 needs a decision: implement collection inside Pass C for any
   future re-run, or keep the separate pass. Recommendation is the separate pass — it decouples
   Stage 1's disk needs from Stage 0's runtime and lets Stage 0's artifacts stand as they are.
3. §5.2(5)'s 5-point capability threshold is a judgement call and is the one number here I would
   want challenged before freezing.
4. Power for §5.3 must be computed on the realised design once Pass B's pairs are final, and before
   any steering run.
