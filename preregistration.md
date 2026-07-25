# Preregistration — Stage 0: Free-Choice Spreading of Alternatives in Instruction-Tuned LMs

| | |
|---|---|
| **Version** | 1.0 |
| **Date frozen** | 2026-07-25 |
| **Target venue** | ARR October 2026 cycle |
| **Status** | Frozen before any experiment code was written and before any model was run. |
| **Scope** | Stage 0 only. This document preregisters a behavioral existence test and one forward-looking equivalence rule (§10). It preregisters no mechanistic claim. |

---

## 1. Claim under test (project-level) and what Stage 0 actually tests

**Project-level claim.** Consistency-restoration and truth-tracking are separate, causally
dissociable mechanisms in instruction-tuned language models.

**Stage 0 tests only whether the behavioral effect exists at all.** It measures whether an
instruction-tuned LM exhibits spreading-of-alternatives — post-decision divergence between a
designated and a non-designated option — and whether that spreading depends on choice
difficulty in a way that requires the model's own authorship of the choice.

Stage 0 makes **no** claim about mechanism, about truth-tracking, or about dissociation. It is a
**kill gate**: if the target interaction is absent, the project does not proceed to representation-
level work.

### 1.1 Why the interaction, and not the main effect

A main effect of agency on post-rating divergence is fully consistent with a null hypothesis of
mere context-window sensitivity: a context that names one option in a positive frame will raise
that option's subsequent rating regardless of whether any decision-like process occurred. This
is the argument that the published PNAS rebuttal used to eliminate the prior version of this
claim, and it is correct. A main effect of agency therefore **proves nothing** and is not the
primary test here.

Consistency-restoration, by contrast, predicts a specific *dependency structure*: the pressure to
restore consistency should scale with how much conflict the decision created, i.e. with how close
the two options were before the choice. That is an interaction, and context-window sensitivity
does not predict it.

Accordingly this design (a) makes the difficulty × agency interaction the sole primary test, and
(b) instantiates the rebuttal's own proposed mechanism as **measured experimental conditions**
(`3p-yoked`, `3p-random`) rather than as something argued away in discussion.

---

## 2. Hypotheses

`|diff|` denotes the absolute pre-choice rating difference within a pair, measured independently
of the difficulty selection variable (§4.4). Small `|diff|` = difficult choice.

| ID | Contrast | Directional prediction | Role |
|---|---|---|---|
| **H1** | `(chose − yoked) × |diff|` | **Negative** — the agency effect on spread is larger for difficult (small-`\|diff\|`) pairs | **PRIMARY** |
| H2 | `chose − 3p-yoked` | Positive | Authorship + self-relevance, information held constant |
| H3 | `3p-yoked − yoked` | Positive under the rebuttal's account; ≈0 under a pure-authorship account | Information effect, designation held constant |
| H4 | `yoked − random` | Negative or ≈0 | Selection artifact (choice/pre-rating noise correlation) |
| H5 | `3p-random − random` | Positive under the rebuttal's account | Pure context effect, decorrelated from prior preference |

**Discriminating structure.** The context-sensitivity account predicts that the effect lives in H3
and H5 and that **H1 is null**. The consistency-restoration account predicts H1 is non-null and
survives regardless of H3 and H5. H1 is therefore the only test that can decide the gate, and H3
and H5 are reported whatever H1 shows.

**Artifact cross-check (not a hypothesis).** The selection artifact is estimated twice:
`yoked − random` and `3p-yoked − 3p-random`. These estimate the same quantity by different routes
and should agree. The discrepancy is reported; a large discrepancy indicates the designation
manipulation is not behaving as modeled and is grounds for reporting Stage 0 as uninterpretable
rather than as a pass or a fail.

**Why H1 is not contaminated by the artifact.** `chose` and `yoked` designate the *same* item —
the model's own pick, elicited within the same option order — so the selection artifact enters
both arms equally and cancels in the difference. No regression-to-the-mean covariate is used.

---

## 3. Measurement

### 3.1 Hard constraints

1. **Every dependent variable is read from ONE forward pass at ONE token position.** There is no
   open-ended generation anywhere in the measurement path, at any point, for any variable. This is
   enforced by a test that fails if `.generate(` appears anywhere under `src/`.
2. **bfloat16 only.** No 4-bit or 8-bit quantization at any stage. The project will make
   equivalence claims in later stages, and quantization noise inflates exactly the variance those
   claims are computed against.
3. **Ratings are expected values over the logit distribution across digit tokens, never argmax.**
4. **All randomness is seeded. Item order, option order, and template assignment are
   counterbalanced by construction, not sampled per run.**

### 3.2 Rating scale: 1–9, not 1–10

The specified 1–10 scale is **not readable at one token position** on the majority of the model
set, and constraints (1) and (3) above are therefore in direct conflict at 1–10. Measured on the
actual tokenizers:

| tokenizer | `"10"` | `"1"` | `"9"` |
|---|---|---|---|
| Qwen2.5 | 2 tokens `[16, 15]` | 1 token `[16]` | 1 token `[24]` |
| Gemma-2 | 2 tokens `[235274, 235276]` | 1 token `[235274]` | 1 token `[235315]` |
| Llama-3.2 | 1 token `[605]` | 1 token `[16]` | 1 token `[24]` |

On Qwen2.5 and Gemma-2 the first token of `"10"` is byte-identical to the token for `"1"`, so the
probability mass at a single position cannot be attributed between ratings 1 and 10. Resolving
this by reading a second position would violate constraint (1) and would make the readout
mechanism non-uniform across the model set (one position for Llama, two for Qwen and Gemma).

**Resolution: a 1–9 integer scale.** All nine digits are single tokens with no cross-digit
collisions on every tokenizer in the set, so the readout mechanism is byte-for-byte identical
across models. The cost is one scale point.

### 3.3 Rating readout

For a prompt whose final token precedes the rating position:

```
rating = Σ_{d=1..9} d · p(d),     p = softmax(logits restricted to the digit token ids)
```

- The digit token map is constructed **per tokenizer at load time**: every single-token surface
  form decoding to each digit (bare and leading-space variants) is enumerated and its probability
  summed into that digit's bin.
- The run **aborts** if the map does not cover all of 1–9, or if any token id maps to more than one
  digit. Silent partial coverage is treated as a fatal error, not a degraded measurement.
- Renormalization is over the nine digit bins only. The total probability mass falling on those
  bins is recorded per trial as `digit_mass` and reported as a diagnostic; it is not used to filter
  trials.

### 3.4 Polarity reversal

Descending-polarity presentations are reversed as **`10 − x`**, the correct constant for a 1–9
scale. (`11 − x` belongs to a 1–10 scale and would add a constant +1 to every descending score,
biasing every polarity-collapsed mean by +0.5.)

### 3.5 Choice readout

The choice is a discrete commitment rather than a graded measurement, so it is read as the
**argmax over the option-label token ids at one position** from one forward pass. In the `chose`
condition only, the resulting label is written into the context as the assistant turn. The full
label distribution is recorded.

---

## 4. Pass A — item rating

### 4.1 Stimuli

400 items: 100 each in four neutral domains — destinations, electronics, foods, activities.
Items are real, familiar, non-controversial entities, framed exclusively in terms of personal
appeal.

**No factual claim, verifiable property, or correctness-bearing statement appears anywhere in the
stimuli.** This is a design requirement, not a preference: Stage 0 must not contain any material
that could engage truth-tracking, because the project's later claims depend on the two
paradigms being non-overlapping.

### 4.2 Design

5 templates × 2 polarities {ascending, descending} = 10 single-item conditions per item.
**4,000 forward passes per model.** One item per forward pass, no shared context between items,
so no item's rating can be affected by any other item's presence or position.

The 5 templates are **the same 5 templates used in Pass C**. Template is therefore a consistently
modeled factor across the whole of Stage 0 rather than two separate unmodeled batteries.

### 4.3 Polarity collapse

```
s[item, t] = mean( asc[item, t],  10 − desc[item, t] )
```
yielding 5 template scores per item.

### 4.4 Split of Pass A into selection and analysis measurements

| Quantity | Source | Purpose |
|---|---|---|
| selection score | `mean(T1, T2, T3)` | selecting difficult and easy pairs (Pass B) |
| analysis `\|diff\|` | `mean(T4, T5)` | the continuous `\|diff\|` regressor in the primary model |
| `σ_between` | between-item SD of `mean(T4, T5)` | fixes the SESOI (§9) before Pass C runs |

Selecting on a noisy `|diff|` and then *analyzing* that same noisy `|diff|` would make the
regressor regression-contaminated. The independent template split provides an uncontaminated
measurement of the same underlying quantity. The split favours selection (3 templates vs 2)
because the DV's pre-rating is measured within-trial in Pass C, so the Pass A baseline feeds only
robustness analyses while selection precision directly determines the strength of the difficulty
manipulation.

---

## 5. Exclusion criteria

Evaluated per model, from Pass A only, **before** Pass B or Pass C is run for that model.

### 5.1 Polarity validity — the sole categorical exclusion

Per template, Spearman ρ between ascending scores and reversed descending scores across the 400
items. **Require ρ ≥ 0.6.**

- A model is **excluded** if the **median ρ across its 5 templates < 0.6**. A model that fails is
  not read as evidence about the hypotheses in either direction; it is reported as not having
  demonstrated that it reads the response scale at all.
- Any **individual** template with ρ < 0.6 is dropped from that model's template battery and
  reported. If fewer than 3 templates survive, the model is excluded.
- All five per-template ρ values are reported for every model, passing or not.

This threshold doubles as a substantive result: it locates the bottom of the usable model range
empirically instead of by assumption, and both the validity and reliability numbers are reported
as our own prompt-sensitivity diagnostics rather than left to be extracted by a reviewer.

### 5.2 Dynamic range

A model is **excluded** if `σ_between < 0.5` rating points. With no between-item variance there is
no `|diff|` range, the difficulty manipulation has nothing to act on, and the primary interaction
is undefined rather than null.

### 5.3 Reliability — a tripwire, not a scientific halt

`ICC(C,1)` across the 5 templates. The **consistency** form is used because the templates are a
fixed battery and additive per-template offsets cancel in the within-pair differences that
constitute the DV; absolute agreement across templates is not required for the design to work.

`ICC(2,3)` — the reliability of the actual 3-template selection score — is computed
**empirically** from the template scores, **not** projected via Spearman–Brown, because template
errors are correlated and the projection would overstate it.

**Reliability does not exclude a model.** `ICC(C,1) < 0.4` is a tripwire that triggers
investigation for an implementation fault before proceeding. Otherwise reliability enters only the
power simulation (§8). Where reliability is genuinely low but validity passes, the binding
criterion is power on the primary contrast at the SESOI, not the ICC itself.

### 5.4 Trial-level exclusions

**None.** All trials are retained, including trials whose choice flips across option order —
yoking is defined within option order (§6.4), so a flip does not desynchronise `chose` from
`yoked`. Choice flip rate is reported as a position-bias diagnostic.

---

## 6. Pass B and Pass C — design

### 6.1 Pass B: pair construction, per model, from that model's own Pass A ratings

Pairs are constructed **separately for every model, from that model's own ratings**. No shared,
human-intuited, or cross-model pair list is used anywhere.

- Within-domain pairs only.
- Difficulty selected on the selection score `mean(T1–T3)`: **difficult** = bottom decile of
  `|diff|`, **easy** = top quartile.
- **100 difficult + 100 easy** pairs per model.
- Each item used in **at most 2** pairs. **No item appears in both difficulty levels.**
- Difficulty × domain balanced: 25 difficult + 25 easy per domain.
- **Difficult and easy pairs are matched on mean pair rating.** Difficulty is otherwise confounded
  with extremity, and ceiling/floor compression can manufacture the difficulty interaction with no
  mechanism behind it. This is handled by design-level matching rather than by a covariate,
  because matching removes the confound instead of modeling it, and the 400-item pool is large
  enough to make matching feasible.
- Realized `|diff|` distributions per level, and matching diagnostics, are reported.

### 6.2 Pass C: five conditions × difficulty

**Receipt is matched across all five conditions** — every condition states that the model receives
the designated item. Without this, `3p-yoked − yoked` would confound endorsement with ownership.

| condition | designated item | authorship | endorsement |
|---|---|---|---|
| `chose` | model's own pick | self | — |
| `yoked` | model's own pick | none | none |
| `3p-yoked` | model's own pick | other | other person |
| `3p-random` | random | other | other person |
| `random` | random | none | none |

Third-party conditions state that another person chose the designated item over its partner *for
the model*, and that the model receives it. This replaces free-text valenced exposure as the
rebuttal control: free text is uncalibrated in valence magnitude and would convey more valence
than a choice does, making a null uninformative and a positive result unattributable.

The DV labels the designated/endorsed item as *chosen* throughout, in all five conditions.

### 6.3 Counterbalancing and context structure

- 5 paraphrased templates per condition. **Template is a modeled factor, never silently averaged.**
- Option order counterbalanced.
- **`pre_order` is not a factor and is not recorded.** With separate contexts per rating (below)
  there is no causal channel by which "which item was rated first" could affect anything, so it is
  removed from the design rather than logged as an inert column.

Contexts share a cached prefix `P = [pair presented, A = X, B = Y]`:

```
pre_X   = fwd( P + "rate X" )
pre_Y   = fwd( P + "rate Y" )
post_X  = fwd( P + manipulation + "rate X" )
post_Y  = fwd( P + manipulation + "rate Y" )
```

**The model never sees its own prior rating.** Writing a pre-rating back into the context would
give the post-rating a token to copy, which is a first-order threat to a difference-of-differences
DV and is asymmetric across conditions. This also matches the human paradigm, in which
participants do not review their earlier ratings before re-rating.

**Pre-ratings are measured once per (pair, template, option order) and reused across all five
conditions.** This is exact, not an approximation: the pre context is identical across conditions
because it precedes the manipulation. It also induces cross-condition correlation within pair,
which is why the model carries a per-pair random intercept (§7).

### 6.4 Choice elicitation and yoking

The choice is elicited per **(pair, template, option order)**, and yoking is defined **within**
option order. Counterbalancing therefore stays intact and `chose`/`yoked` are always
designation-matched. Choice flip rate across option orders is reported.

### 6.5 Dependent variable

```
spread = (chosen_post − chosen_pre) − (rejected_post − rejected_pre)
```

On a 1–9 scale each constituent difference is bounded by ±8, so `spread` is bounded by **±16**.
Realistic magnitudes are expected around ±2. `spread` is interpreted as **points of divergence
between the two options**, not as a rating.

### 6.6 Forward-pass budget

| | passes/model |
|---|---|
| Pass A | 4,000 |
| Pass C pre (reused across conditions) | 4,000 |
| Pass C choice | 2,000 |
| Pass C post | 20,000 |
| **total** | **≈ 30,000** |

---

## 7. Statistical model

Bayesian mixed-effects regression, specified directly in PyMC (with ArviZ for
posterior summaries). The model is written out rather than expressed through a
formula interface so that the cell-means parameterization below, and the names of
every coefficient it produces, are exact.

```
spread ~ condition * diff_z + option_order + (1|pair) + (1|template)
```

- `condition`: 5-level factor under **cell-means coding** — each condition receives its own
  intercept and its own `diff_z` slope. Every contrast in §2 is then a plain difference of
  posterior draws, exact and independent of any contrast-coding scheme. (Cell means supersede the
  reference-level formulation: with `yoked` as a reference the primary contrast is one coefficient,
  but the secondary contrasts are not, and would depend on the coding.)
- `diff_z`: `|diff|` from `mean(T4, T5)`, **z-scored within model** over that model's Pass C pair
  pool. The interaction coefficient is then in spread points per SD of `|diff|`, directly
  comparable to the SESOI.
- `option_order`: fixed nuisance term; balanced by construction, included to absorb variance.
- Domain is not entered: it is absorbed by `(1|pair)`, since pairs are within-domain.
- **Primary analysis is continuous in `|diff|`.** Extreme-groups selection maximises range while
  continuous analysis avoids the ~36% variance loss from dichotomising. The binary
  difficult/easy version of the same model is reported as a secondary analysis.

### 7.1 Priors

Weakly informative, on the spread scale (bounded ±16, expected ~±2):

| parameter | prior |
|---|---|
| intercept | `Normal(0, 1)` |
| condition effects | `Normal(0, 1)` |
| `diff_z` slope | `Normal(0, 1)` |
| interaction terms | `Normal(0, 1)` |
| `option_order` | `Normal(0, 1)` |
| random-intercept SDs (pair, template) | `HalfNormal(1)` |
| random intercepts themselves | `ZeroSumNormal(σ)` — see below |
| residual σ | `HalfNormal(2)` |

Random intercepts are **zero-sum constrained**. Under cell-means coding there is no global
intercept, so unconstrained random intercepts leave the overall level unidentified: adding a
constant to every `b_cond` and subtracting it from every `u_pair` leaves the likelihood
unchanged. Every reported quantity is a contrast and so is identified regardless, but the
ridge degrades sampling and makes `b_cond` uninterpretable as a cell mean. The constraint
removes it at no cost to anything the design requires.

Sampling: 4 chains, 2000 warmup + 2000 draws. Convergence required at `R̂ < 1.01` and ESS > 400
for every reported parameter; failure is reported rather than silently re-tuned.

### 7.2 Item random effect — preregistered robustness model

The DV is per-pair while each pair contains two items, so a single `item` grouping factor is not
well defined alongside `pair`: it requires a multi-membership structure. The preregistered
**primary** model is the one specified above. A preregistered **robustness** model adds an item
random effect as the sum of the two member items' intercepts, `u_item[item1] + u_item[item2]`,
which is the multi-membership form written out explicitly. Item reuse is capped at 2 pairs, so the
two models are expected to agree closely; the discrepancy is reported either way.

---

## 8. Power

Power is **simulated after Pass A and before Pass C**, per model, from that model's own measured
quantities: `σ_between`, the measurement noise implied by `ICC(C,1)`, and the realized `|diff|`
distribution from Pass B. Simulation reports the smallest true interaction whose 95% HDI excludes
zero in ≥80% of simulated datasets.

**If power at the SESOI is short, items are scaled first and pairs second.** Adding pairs beyond
the item pool's capacity forces reuse and pushes the selection thresholds toward the middle of the
`|diff|` distribution, diluting the difficulty manipulation itself. The power simulation and its
result are reported for every model whether or not scaling was triggered.

---

## 9. SESOI and the gate decision rule

### 9.1 SESOI

**Primary: SESOI = 0.15 × σ_between**, where `σ_between` is the between-item rating SD measured in
Pass A, polarity-collapsed, `mean(T4, T5)`. This resolves to a fixed number **before Pass C
begins** and is reported per model with its raw-point equivalent.

**Secondary: a fixed 0.25 raw rating points**, reported as a scale-free anchor for readers who
want a bound not indexed to this study's variance.

A per-model standardization is used rather than a single raw constant because the model ladder
spans 0.5B–3B and rating variance is not expected to be constant across it; a fixed raw bound
would silently be strict for low-variance models and lax for high-variance ones.

### 9.2 Gate decision rule

Per model, on the H1 interaction coefficient:

| outcome | criterion | consequence |
|---|---|---|
| **pass** | 95% HDI excludes 0 in the predicted (negative) direction **and** posterior median magnitude > SESOI | model supports proceeding |
| **fail** | 95% HDI lies entirely inside the ROPE `[−SESOI, +SESOI]` | equivalence to null for that model |
| **inconclusive** | HDI overlaps both 0 and the SESOI boundary | reported as inconclusive; resolved by scaling items, not by reanalysis |

**Project-level gate:** Stage 1 is entered only if **at least two non-excluded models pass**. The
ladder pattern — whether the interaction emerges with scale — is reported as a primary descriptive
result regardless of the gate outcome. A gate failure is reported as a negative result, and the
later-stage code named in §11 is not written.

---

## 10. Equivalence bounds for later stages

Preregistered now, applied in Stage 2. **`f = 0.25`.**

Stage 2 equivalence is expressed as the **posterior distribution of the ratio**

```
truth-vector effect / consistency-vector effect
```

with the equivalence claim being that the ratio's HDI falls within **±f**. Taking the ratio's
posterior propagates uncertainty in **both** effects, rather than dividing a point estimate by a
point estimate and understating the uncertainty of the denominator.

**Equivalence is secondary throughout the project.** The primary Stage 2 claim is the
*differential* effect — an interaction — so the strictness of `f` is not load-bearing for the
central result. `f` is fixed in advance so that the secondary equivalence claim cannot be tuned
after the fact.

---

## 11. Out of scope for Stage 0

Not built, not run, not analyzed, and not reported in Stage 0: steering hooks; linear or nonlinear
probes; contrastive activation addition; the knowledge-conflict paradigm; any truth-tracking
stimuli; any free-text valence exposure condition (replaced by the calibrated `3p-*` conditions,
§6.2); any open-ended generation anywhere.

Stage 0 is a kill gate. If H1 is null, the above is wasted engineering and is not written.

---

## 12. Reproducibility and provenance

- One typed configuration object; one YAML per run; the **config hash appears in every output
  filename**.
- Every artifact records: device, dtype, torch version, transformers version, model revision
  (HuggingFace commit SHA), seed, config hash, git SHA, platform.
- **The analysis layer raises an error if asked to pool artifacts across devices or dtypes.**
  Numbers produced on the development machine (Apple M1, MPS/CPU) cannot be silently combined
  with numbers produced on the run machine.
- **All reported numbers originate on the run machine**: RTX 2000 Ada, CUDA, bfloat16. The M1 is
  used for scaffolding, analysis, plots, and smoke tests at ≤0.5B only.
- Every stage writes versioned artifacts (parquet for tabular, safetensors for tensors) and is
  re-runnable from cached upstream output.

### 12.1 Model set

Qwen2.5-Instruct 0.5B / 1.5B / 3B, Llama-3.2-3B-Instruct, Gemma-2-2b-it. All revision-pinned; the
resolved commit SHA is recorded per run.

---

## 13. Deviations from the original specification

Recorded here so that every departure is visible rather than buried in code.

| # | Specification | Change | Reason |
|---|---|---|---|
| 1 | Ratings 1–10 | **1–9** | 1–10 is unreadable at one token position on Qwen2.5 and Gemma-2 (§3.2); the original constraints were mutually unsatisfiable. |
| 2 | Reversal `11 − x` | **`10 − x`** | Correct constant for a 1–9 scale; `11 − x` would bias every polarity-collapsed mean by +0.5. |
| 3 | `spread` range stated as ±18 | **±16** | ±18 is the 1–10 bound; ±8 per difference on a 1–9 scale. |
| 4 | ~200 items | **400** | Required to make difficulty/easy matching on mean pair rating feasible at 100 pairs per level. |
| 5 | Pass A run twice with different order seeds | **5 templates × 2 polarities, one item per forward pass** | With one item per pass and expected-value readout, two runs differing only in item order are bit-identical and test–retest ρ is trivially 1.0. The original reliability check was vacuous. |
| 6 | 2×2 difficulty × agency | **2 × 5 conditions** | `chose`/`assigned` cannot separate authorship from information; `3p-yoked`, `3p-random`, `random` are required to instantiate the rebuttal's mechanism and to estimate the selection artifact. |
| 7 | Valence control via valenced text | **`3p-yoked` / `3p-random`** | Free-text valence is uncalibrated relative to the valence a choice conveys, making both null and positive outcomes unattributable. |
| 8 | Mean pair rating as a covariate | **design-level matching** | Matching removes the extremity/ceiling confound; a covariate only models it. |
| 9 | Difficulty as a binary factor | **continuous `\|diff\|`, binary secondary** | Extreme-groups selection with continuous analysis avoids the ~36% variance loss from dichotomising. |
| 10 | Test–retest ρ < 0.7 halts | **polarity validity ρ < 0.6 halts; ICC is a tripwire at 0.4** | Separates two distinct properties — whether the model reads the scale (categorical) and how noisy it is (a power question, §5.3). |

Items 1–3 are arithmetic or tokenizer facts. Items 4–10 are design decisions taken before any data
existed.
