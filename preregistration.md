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

> **The `pass` cell is unchanged, and A3.6 records a considered refusal to change it.** Dropping
> the `median > SESOI` conjunct was computed blind and declined; see A3.6 for the arithmetic and
> the reasoning. The `inconclusive` row's "resolved by scaling items" is retained as written but
> should be read against **A3.2**, which prices that remedy and finds it close to worthless for
> the primary contrast.
>
> **The project-level gate below is revised by A3.7**, also blind. The per-model outcome rule and
> the project gate are separable: the first defines a claim about the world, the second allocates
> resources. Only the second is revised.

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

---

# Amendment 1 — 2026-07-26

**Appended, not edited.** §1–13 above are frozen at commit `57ef60d` and are unchanged. This section
records three instrument-level amendments, the pilot evidence that motivated them, and one newly
preregistered gate.

## A1.0 Evidentiary status — read before anything below

**No confirmatory data has been observed.** Every number cited in this amendment comes from
development-machine pilot runs (Apple M1, MPS) and from a standalone diagnostic probe. Not one
figure below comes from the run machine, and no analysis bearing on H1–H5 has been performed on any
model at any scale.

**The pilot set is not the confirmatory set.** Pilots used **12 items per domain (48 items)** against
the confirmatory pool of **400**, **8 pairs** against 200, and **3 of the 5 models**
(Qwen2.5-0.5B-Instruct, Qwen2.5-3B-Instruct, Gemma-2-2b-it) — the three that happened to be cached
locally. Qwen2.5-1.5B-Instruct and Llama-3.2-3B-Instruct were never run. Pilot items, pilot pair
counts, pilot models and pilot device are all disjoint from the confirmatory design, and pilot
artifacts are rejected by `provenance.assert_reportable`.

These amendments therefore change the **instrument**, on evidence about the instrument. They do not
change any hypothesis, contrast, or decision rule in §2, §7 or §9.

## A1.1 (D1) Rating instrument: absolute Likert → anchor-based pairwise

**What changed.** Item scores are no longer read as an absolute 1–9 Likert judgement of a single
item. Each pool item is instead compared against each of **10 fixed anchor items in both orders**
(20 binary outcomes per item), and a latent appeal score `θ` is estimated by hierarchical
Bradley-Terry with partial pooling across items, fitted per model. Anchors lie **outside** the
400-item pool, span the full appeal range including genuinely poor options, and are identical across
all models. Order is averaged **structurally** — both orders are always run — rather than corrected
for at analysis time. Precision is reported as the **posterior SD of `θ`**, not as an ICC.

**Why.** The absolute scale compressed almost all variance out of the measurement:

| model | ascending rating range | mean | σ_between (ascending) |
|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 2.34 – 7.96 | 5.30 | 0.425 |
| Gemma-2-2b-it | 2.44 – 8.52 | 6.78 | 0.483 |
| Qwen2.5-3B-Instruct | 3.18 – 9.00 | 7.58 | 0.581 |

The consequence propagated directly into the difficulty manipulation. In a pilot Pass B, difficult
and easy pools separated cleanly on the **selection** score (mean |diff| 0.001 vs 0.575) but were
**indistinguishable on the independent analysis measurement** (0.295 vs 0.302). Difficulty was being
selected on measurement error — which is precisely what the disjoint template split of §4.4 exists
to detect, and it detected it.

Critically, the latent ordering is intact; only the absolute readout destroys it. In the choice
probe, Qwen2.5-3B assigns p = 0.958 to "Vienna" over "a motorway service station" when asked to
choose, while rating both in the 7–9 band when asked to rate. A comparative instrument recovers what
an absolute one flattens.

## A1.2 (D2) Polarity-validity gate retired → order-invariance gate

**What changed.** The scale-polarity gate of §5.1 (Spearman ρ ≥ 0.6 between ascending and reversed
descending scores) is **retired as an exclusion criterion**. It is replaced by an **order-invariance
gate** defined in A1.4. Polarity remains measured and reported for the instrument-validation record
(A1.5) but no longer excludes any model.

**Why.** Two reasons, and the second is the more serious.

First, it excluded the entire ladder. Median ρ was **−0.951** (Qwen2.5-0.5B), **−0.490**
(Qwen2.5-3B) and **−0.953** (Gemma-2-2b) — three models across two families, all strongly negative,
meaning each answers on a fixed higher-is-better mapping and ignores a reversed anchor definition. A
categorical criterion that rejects every candidate measures the elicitation, not the models.

Second, it **double-counted**. When a model is polarity-blind, ascending ≈ descending, so the
polarity-collapsed score `(asc + C − desc)/2` is mechanically driven toward the constant `C/2` for
every item. The observed collapse of `σ_between` (0.425 → 0.042, 0.483 → 0.118, 0.581 → 0.299) is
therefore an arithmetic consequence of the polarity failure, not independent evidence of a second
defect. §5.2's dynamic-range criterion was firing on an artifact of §5.1's failure. Under a pairwise
instrument neither quantity is defined in its original form, so both are superseded.

Polarity-insensitivity is retained as a **reported property of these models**, not as grounds for
exclusion: it is a substantive finding about instruction-tuned LMs and a reviewer will ask about it.

## A1.3 (D3) Option elicitation: letter labels → digit labels

**What changed.** Binary choices are elicited with `1`/`2` rather than `A`/`B`, in the pairwise
readout and everywhere else a choice is read. Item-name first-token readout is retained as a
**secondary arm**, reported alongside.

**Why.** Letter labels were the worst-performing scheme tested on every model. On pairs with a wide,
obvious appeal gap:

| model | letters | digits | item-name |
|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 0.513 | 0.550 | 0.667 |
| Gemma-2-2b-it | 0.825 | **0.858** | 0.667 |
| Qwen2.5-3B-Instruct | 0.783 | **0.829** | 0.688 |

0.500 is chance. Qwen2.5-0.5B under letters is at chance *on obvious pairs*, and its order-invariance
there is **0.025** — near-pure position responding, in which the same slot is chosen regardless of
content so the winner flips on essentially every reversal. On close pairs the switch matters most:
Qwen2.5-3B's rate of choosing the first-displayed option falls from **0.854 under letters to 0.421
under digits**.

A methodological note belongs here because it nearly produced a false result. The first version of
the content-addressed arm retained lettered options in the prompt, so the model answered with the
letter: **98.7% of the probability mass sat on the label tokens and 0.8% on the item-name tokens**,
while the readout was reading item-name tokens. It returned clean-looking accuracy figures that were
meaningless. This motivates A1.6.

**Label rendering is standardised across templates; template prose is not.** In the pairwise readout
each template contributes its own lead-in sentence and its own question wording, but options are
always rendered as `N. item`. Template t4 natively renders options as `[1] item`; the model copies
that format and answers `"[1]"`, so at the single readout position it emits `[` and the digit tokens
lose their mass — median readout mass 0.38 with 568 of 800 comparisons below the floor, against
~1.00 under a standardised rendering. Label syntax is part of the readout mechanism, not part of the
paraphrase manipulation: varying it tests the parser rather than the model's robustness to wording.
Paraphrase variation is preserved where it is meaningful, in the prose.

## A1.4 (NEW GATE) Order invariance — preregistered before any pairwise data

**Statistic.** Per model, the proportion of (pool item × anchor) comparisons in which the **same item
wins under both presentation orders**. Computed on the **digits** arm. Reference points:

| responding | order invariance |
|---|---|
| perfectly content-driven | 1.00 |
| purely random | 0.50 |
| purely positional (always the same slot) | 0.00 |

Note the statistic separates the two failure modes, which a simple accuracy measure does not: a
position-only responder scores 0, not 0.5, because the winner flips on every reversal.

**Threshold.** A model is **excluded** if median order invariance across templates is **< 0.60**, or
if fewer than **3 of 5** templates individually reach 0.60. Both failure regions are reported
distinctly:

- invariance **< 0.50** → *position-dominated*: responding is driven by slot, not content.
- **0.50 ≤ invariance < 0.60** → *random-dominated*: responding carries no recoverable signal.

**Justification for 0.60.** This is a floor against degenerate responding, not a reliability bar, and
it is set that way deliberately. Because both orders are always run and each item accumulates 20
binary outcomes, Bradley-Terry tolerates substantial per-comparison noise; the quantity that governs
whether `θ` is usable is its **posterior SD**, which is reported separately and is not a gate. The
gate's job is only to establish that comparisons carry content-driven signal at all.

Pilot calibration on wide-gap comparisons (the analogue of anchor comparisons) gave invariance of
**0.708** (Qwen2.5-3B, digits), **0.733** (Gemma-2-2b, digits) and **0.750** (Qwen2.5-0.5B,
item-name) — so 0.60 passes all pilot-tested configurations with margin while decisively excluding
the position-dominated case (0.025). A threshold that no candidate can clear is not a gate, and one
that everything clears regardless of behaviour is not either; 0.60 sits between the observed failure
mode and the observed working mode.

Sampling error is negligible (~4,000 comparisons per template; SE on a proportion ≈ 0.008), so the
threshold is a substantive judgement, not a statistical one.

> **Correction, entered 2026-07-26 after the first pairwise pilot run.** The paragraph originally
> here argued that anchor comparisons are "where signal should be strongest, since anchors span the
> appeal range by construction and most pool-vs-anchor gaps are wide." **That reasoning was wrong,
> and self-contradictory.** An anchor set that spans the appeal range necessarily produces many
> *narrow* gaps — that is what spanning means. The high-tier anchors sit close to a typical pool
> item by design, so the gate is a considerably more demanding test than the pilot calibration
> figures (0.708–0.750, measured on deliberately wide-gap pairs) implied.
>
> The threshold of 0.60 is **not** changed: it was chosen as a floor against degenerate responding,
> and that rationale is unaffected. What is withdrawn is the claim that pilot wide-gap invariance
> transfers to anchor comparisons as a margin estimate. It does not, and the first pilot run showed
> the difference directly — Qwen2.5-3B scored 0.744 against low-tier anchors and 0.290 against
> high-tier ones, a range the original paragraph would not have anticipated.

## A1.5 Instrument-validation record

The preregistered absolute-rating Pass A is run **once at full scale** (400 items, all five models,
run machine) and archived under `artifacts/instrument_validation/`. It is expected to fail the
retired polarity gate. It is **never read by any downstream stage** and contributes to no
hypothesis test. Its sole purpose is to let the paper report "the preregistered instrument was run at
full scale and these are the numbers" rather than resting the amendment on pilot runs.

## A1.6 (NEW INVARIANT) Readout mass floor

Every readout in the codebase — rating, choice, and pairwise — now checks the total probability mass
falling on its candidate tokens. Below the floor the trial is marked **invalid and logged**; it is
never silently scored. Invalid-trial counts are reported per model and per elicitation arm.

This is a direct response to the failure in A1.3, where a readout with 0.8% of its mass on the
candidate tokens produced plausible numbers. A readout that is not reading the intended distribution
must fail loudly, not return a value. Applied retrospectively to the instrument-validation record via
its recorded per-trial `digit_mass`.

## A1.7 (NEW DIAGNOSTIC) Operating window — evaluated before Pass C

The paradigm needs pairs that are **near-equal** (so the choice is difficult) and on which the model
still responds to **content** rather than position. The pilot probe found content-driven choice only
at *wide* gaps. Whether those two requirements overlap at all is therefore an open empirical
question, and it is answered before any Pass C compute is spent.

Choice consistency under order reversal is plotted against `|θ_i − θ_j|` in bins. A usable operating
window requires a band where consistency is **above chance** while the gap is **small**. **If no such
band exists, Pass C as designed cannot work**, and the paradigm is reconsidered rather than run.

## A1.8 Unchanged by this amendment

The spread DV structure; the 2 × 5 receipt-matched conditions; continuous difficulty selected on one
template set and analysed on a disjoint one; the difficulty × agency interaction as the sole primary
test; the third-party valence control arms; the cross-device pooling guard; zero-sum constrained
random effects; and the exclusion of reporting thresholds from forward-pass cache keys.

Where §4–§6 specify quantities in rating points (`match_tolerance`, `sigma_between_min`, the ±16
spread bound), those units are defined against the absolute scale and do not transfer to `θ`. Their
re-expression is deferred to a further amendment and is **not** settled here.

---

# Amendment 2 — 2026-07-26

**Appended, not edited.** §1–13 remain frozen at `57ef60d`; Amendment 1 is unchanged. This section
adds an order term to the Bradley-Terry likelihood, retires the A1.4 gate, and withdraws two
conclusions previously reported — one of which was the basis of a strategic plan.

Prompted by external review of the repository. Each claim was verified against code and data before
being acted on; two of the review's premises were also refuted and are recorded as such.

## A2.0 Evidentiary status

**No confirmatory data bearing on H1–H5 has been observed.** No Pass C data of any kind exists. The
Bradley-Terry refits below reuse comparisons already collected and add **zero forward passes**; no
new stimuli, models, or conditions were run to produce them. A1.0's statement stands unqualified.

## A2.1 Order term added: `β` per template

**What changed.** The Bradley-Terry likelihood becomes

```
P(item i beats anchor a) = sigmoid(θ_i − α_a + s · β_t)     s = +1 if the item is in slot 1, −1 if slot 2
```

with `β_t` a free parameter **per template**. Order remains fully crossed — both orders are run for
every cell, as before. What changes is that the antisymmetric half of that design is now *used*
rather than averaged away.

**Why the previous justification was wrong.** The code justified omitting an order term on the
grounds that "a position-bias parameter would be estimated from the very comparisons it is meant to
purge." That is backwards. Because both orders are run for every cell, per cell we observe
`logit p₀ = x + β` and `logit p₁ = x − β` where `x = θ_i − α_a`. The symmetric contrast
`(logit p₀ + logit p₁)/2 = x` and the antisymmetric contrast `(logit p₀ − logit p₁)/2 = β` are
**orthogonal contrasts of the same two observations**. Balance is exactly what makes `β` cleanly
identifiable, not what makes it unnecessary. Simulation: true `β = 1.0`, recovered **1.016**.

**Why it matters — the omission compresses θ, worst where the design needs it most.** Fitting one
Bernoulli probability to two cells whose true probabilities are `sigmoid(x+β)` and `sigmoid(x−β)`
puts the MLE at their average, which by Jensen lies closer to 0.5 than `sigmoid(x)`. The induced bias
is 0 at `x = 0`, saturates at `−ln cosh β`, and — critically — the **local slope `dx̂/dx` is minimal
at `x = 0`** (0.81 at β = 0.94; 0.60 at β = 1.5), rising toward 1 in the tails. So the latent scale is
compressed **maximally at small gaps**, which is the only regime the paradigm operates in.

Simulation (60 items, 10 anchors, true `β = 1.0`, true `σ_item = 1.086`):

| model | σ_item | corr. with true θ | slope on true θ |
|---|---|---|---|
| no order term | 0.928 | 0.937 | **0.746** |
| with order term | 1.118 | 0.936 | 0.907 |

Rankings survive (correlation essentially unchanged); the **metric** does not. Since Pass B selects
on gap *magnitude* and the operating window stratifies on it, the metric is what the design depends
on.

**On real data (Gemma-2-2b, 4,170 valid comparisons):** `β = +1.367 (sd 0.059)` for a single global
term; `σ_item` rises 1.174 → 1.438 → **1.630** under a per-template `β`.

**β is heterogeneous and signed.** Per template: `+1.645, +1.360, +1.211, +4.302, +0.551`
(t0…t4), max–min spread **3.750 [3.254, 4.272]**. Excluding t3 — whose ~48% invalid-readout rate in
that run makes its survivors a selected subset — the clean templates still span 3×. And the *sign* is
model-specific: Gemma's slot-1 win rate is 0.654 (`β > 0`) while Qwen2.5-3B's is 0.28 (`β < 0`). This
is not a universal primacy or recency effect.

A global `β` leaves **residual compression**: regressing consistency excess on gap over 2,002 cells
gives a slope of **+0.0240, 95% CI [+0.0140, +0.0338]**, excluding zero. Per-template `β` is
therefore preregistered, not a global one.

**Fit-quality criterion.** That excess-on-gap slope is retained as a preregistered diagnostic,
compared against its **posterior-predictive null** rather than against zero — see A2.7. It is biased
away from zero even under correct specification, because the prediction plugs a posterior mean into a
nonlinear function.

**No cell-level random effect is added.** See A2.3.

## A2.2 The A1.4 order-invariance gate is retired

**What changed.** The order-invariance gate (median ≥ 0.60 across templates, ≥3 of 5 clearing) is
**retired as an exclusion criterion**. It is replaced by a **θ reliability criterion**:

> **Gate:** empirical split-half reliability of `θ` — the Spearman correlation between `θ` estimated
> independently on the two disjoint template sets of §4.4 — must be **≥ 0.70**.

Model-internal reliability `σ_item² / (σ_item² + E[posterior var])` is reported alongside, and the
gap between the two is a preregistered misspecification diagnostic — but the comparison must be
**length-matched**, which the original statement of this criterion got wrong.

Model reliability comes from a fit using every template; the empirical figure is a correlation
between two **shorter** fits, each on a subset. Each half carries a noisier `θ`, so its correlation
is lower even under perfect specification, and comparing the two directly guarantees an apparent gap
for every model. With `k = 1/r_model − 1` and halves holding fractions `f_a`, `f_b` of the data, the
prediction is `r_split = 1 / sqrt((1 + k/f_a)(1 + k/f_b))`. The observed split-half is compared
against **that**.

On the 3/2 template split this matters: gemma-2-2b's shortfall is −0.068 rather than −0.105, and
Qwen2.5-1.5B's is −0.175 rather than −0.216. The correction does not erase the difference between
them — which is the point, since it is the difference that carries information.

**Why the old gate does not measure the intended thing.** Under position bias alone with no content
signal at all, expected order invariance at `x = 0` is `2s(1−s)` with `s = sigmoid(β)` — **not 0.5**.
With `β` per template ranging 0.55–4.30 in a single model, that null ranges from **0.464 to 0.026**.
A flat 0.60 threshold applied against a null that moves with a signed nuisance parameter is
predominantly a test of `|β|`, and would exclude a model for having a large position bias rather than
for lacking content signal. Those are now separable, so the gate should not conflate them.

Redefining the gate on high-tier comparisons only (a reviewer suggestion) does not fix this: the null
still moves with `β`, and `β` itself varies by tier.

**Why reliability, and why empirical rather than model-internal.** Reliability is what Pass B
actually depends on — items must be rankable well enough to select near-equal pairs and to detect
movement. It is `β`-free once `β` is modelled. The **empirical** split-half version is preferred
because it cannot be inflated by a misspecified likelihood, and it measures precisely the
selection-versus-analysis independence §4.4 requires.

**Threshold justification.** Gemma-2-2b reaches 0.738 (Spearman) under the corrected model; the two
models that scored 0.000 and 0.275 on the retired gate have essentially no recoverable `θ` variance
and fall far below. 0.70 sits between them. It is deliberately **not** set above the only currently
passing model's value — doing so would repeat the polarity gate's error of adopting a criterion no
candidate can meet.

## A2.3 Withdrawals

**W1 — "The operating window is closed at near-equal pairs." WITHDRAWN.**

Reported in conversation as a paradigm-level finding, and the basis of a plan to fall back to a
methods paper. It rested on comparing order-reversal consistency against a null of 0.5. The correct
null under position bias is `2s(1−s)`; for Gemma `β = 1.367` gives **0.324**. Observed consistency
tracks a content-plus-position model to within ±0.05:

| gap | observed | predicted | excess |
|---|---|---|---|
| 0.30 | 0.288 | 0.334 | −0.046 |
| 0.90 | 0.354 | 0.408 | −0.053 |
| 1.68 | 0.536 | 0.569 | −0.033 |
| 2.56 | 0.777 | 0.757 | +0.020 |
| 4.48 | 0.964 | 0.955 | +0.009 |

The bins reported as "below chance" are what `β = 1.37` predicts **with content signal present**. The
estimator could not distinguish "no signal" from "signal plus unmodelled additive bias," and it was
the second. Additionally, the window tables produced by the pipeline stratified on gaps computed from
the *uncompressed* θ, so their x-axes were also wrong.

**Consequence: Stage 0's hypothesis is live.** The instrument may reach the regime the paradigm
requires. The plan built on W1 is void.

**W2 — "Template responses are non-independent (ICC 0.529, design effect 3.12, posterior SD
understated ~1.8×)." WITHDRAWN.**

This was our own finding, not the review's, and the review built on it. The ICC treated templates as
raters over cells, which conflates **genuine between-cell variation in `p`** — which the model
already captures through `θ − α` — with excess dependence. The correct test is over-dispersion of
within-cell success counts against the binomial expectation implied by the fitted probabilities:

> dispersion = **1.070, 95% CI [0.907, 1.258]**; independence predicts 1.0.

Consistent with independence. There is **no design effect**, so:

- no cell-level random effect is added to the likelihood;
- §4.4's disjoint-split independence is **intact** — the correlated-error concern does not arise;
- the separation ratio is not reduced. Under the corrected model it **rises**: 4.39 → **4.82**, with
  σ_item 1.630, median posterior SD 0.338, model reliability 0.957.

## A2.4 Open, not resolved

Empirical split-half test–retest is **0.770** (Spearman 0.738) against a model-implied reliability of
**0.957**. Neither the order term nor over-dispersion explains the gap; the corrected model's figure
(0.770) is marginally *lower* than the uncorrected one's (0.799).

Leading candidate: in that run t3 was broken (~48% invalid readouts) and t3 falls in the two-template
split, so one half is half-contaminated. The t3 re-collection discriminates this directly. Remaining
candidates: residual misspecification (the +0.024 excess slope), or a template × item interaction too
small to register in the dispersion test yet large enough to decorrelate halves.

This is recorded as an open discrepancy rather than resolved, and the reliability gate deliberately
uses the **lower** empirical figure.

## A2.6 The operating-window diagnostic is redefined

**What changed.** A1.7 asked whether order-reversal consistency exceeded 0.5 at small gaps. That is
replaced by a **discriminability** measure: per gap stratum, the fraction of comparisons whose
`θ_i − α_a` has a credible sign given posterior uncertainty. Consistency is still reported, against
the `β`-model prediction, but **only as a fit check**.

**Why.** The old form was wrong twice. Its null was 0.5 when the correct null under position bias is
`2s(1−s)` (A2.3 W1). And more fundamentally: once `β` is in the model, order-reversal consistency
carries no information about content signal beyond what `θ` already encodes, so testing it against
its own model's prediction is vacuous by construction.

The question that actually decides Pass C viability is a **precision** question, not a consistency
one: at the gaps Pass B selects as difficult, is the sign of the preference credibly determined? If
not, the difficulty regressor is attenuated by measurement error and Pass C requires a power
simulation against that attenuation — which is a quantitative obstacle, not the categorical
impossibility W1 claimed.

**Decision rule.** Discriminability at Pass B's difficult-decile gap is reported with its interval.
Below 0.5, Pass C is not run until a power simulation accounting for regressor attenuation is
produced. This is a **power gate, not a viability gate** — the distinction W1 got wrong.

## A2.7 Pre-committed rule for the residual misspecification

Written **before** the fits that will evaluate it, because a threshold chosen after
seeing residuals is not a threshold.

**Statistic.** The slope of excess consistency on gap, with a bootstrap interval, compared
against its **posterior-predictive null** — regenerating outcomes from the fitted model on
the same design, refitting, recomputing. The null is **not zero**: plug-in prediction of a
nonlinear quantity biases the statistic even under correct specification (Gemma's design
gives a null of +0.0044, sd 0.0025). Comparing against zero, as an earlier draft of A2.1
did, would report a correctly specified model as broken.

**Why the null is simulated rather than derived.** The observed statistic and the null both plug
posterior means into a nonlinear consistency function, so the Jensen bias is **common-mode and
cancels**. That is precisely what makes the comparison valid — and it is why the plug-in bias must
not be "corrected" in the statistic while the null still carries it. If that bias is ever removed,
every stored null must be regenerated in the same change.

**Current value.** Observed **+0.0202** [+0.0111, +0.0285]; null **+0.0038 (sd 0.0036)** over
**24** replicates, 95% range [−0.0017, +0.0103].

**Corrected: +2.9 sd, not +4.5.** The earlier figure divided by the null sd alone, treating the
observed statistic as fixed. The observed CI width of 0.0174 implies its own sd of 0.00444; combined
with the null's 0.00360 that is 0.00572, so `(0.0202 − 0.0038)/0.00572 = 2.87`. Consistent with the
intervals nearly touching (+0.0111 against +0.0103). Real and modest, not overwhelming. This is the
same error family as the retractions in RETRACTIONS.md — a variance component left out of a
comparison.

**A statistical trigger was the wrong instrument here.** At n = 2002 the 2-sd trigger fires at a
slope of ~0.0072, which corresponds to a consistency swing of about 0.03 across the whole gap range.
At this sample size detecting *some* misspecification is close to guaranteed, so the trigger measures
sample size as much as it measures model adequacy. It is replaced by a **decision-relevance**
threshold: the smallest residual that would change a reported quantity.

**Decision relevance of the observed residual.** A discrimination multiplier of g ≈ 1.6 is required
to reproduce a slope of +0.0202. Under a *uniform* scale distortion, Pass B's difficult-decile
membership is exactly **invariant** (overlap 1.000) because selection is quantile-based on `|Δθ|` and
a monotone rescaling preserves the ordering. Under a *non-uniform* distortion of comparable
magnitude, decile membership changes by **≤ 4%** (overlap 0.958 at b = 1.0 and b = 1.5). So the
residual is statistically detectable and decision-irrelevant at the magnitude observed. **That is the
basis for stopping**, and it is a stronger basis than a significance threshold.

**The two nulls reconciled.** An earlier control reported +0.0109 [+0.0019, +0.0199] and appeared to
disagree with the posterior-predictive null by 2.5×. It did not: that figure was a **single
simulated replicate** whose bootstrap interval describes resampling within that one dataset, not the
replicate-to-replicate distribution of the statistic. Rebuilt properly, the harnesses agree —
generic parameters with a complete design give +0.0039 (sd 0.0023), fitted parameters with a
complete design +0.0061 (sd 0.0039), and the real cells with real missingness +0.0038 (sd 0.0036).
The null is ~+0.004 regardless of design detail, so the verdict holds under either harness.

**Candidates already eliminated, by evidence rather than by preference:**

- *Position-capture mixture λ* — ruled out. `dC/dλ` is negative at every gap and largest
  at wide gaps, so λ cannot generate a sign-changing residual; grid optimum is λ = 0.000.
- *Cell-level random effect at (item, anchor)* — ruled out as the explanation. Simulating
  data **with** a cell RE and fitting **without** it moves the slope *down*
  (+0.0035 at σ_u = 0.5, −0.0044 at σ_u = 1.0), not up. It has the wrong sign.

**The rule.**

1. If the t3 re-collection brings the slope within **2 sd of its posterior-predictive
   null**, the residual is attributed to t3 contamination, no parameter is added, and the
   matter is closed.
2. ~~If a slope beyond 2 sd survives clean t3, the next candidate is prior-induced shrinkage of the
   θ and α scales.~~ **TESTED AND ELIMINATED, 2026-07-26.** The fitted α spread (2.649) exceeds the
   `ZeroSumNormal(2)` prior sd, which looked like active shrinkage — but that inference was wrong,
   and the direct test says so. Loosening the α prior on the real data moves nothing: at σ = 2, 5
   and 10 the α spread is 2.649 / 2.690 / 2.696, `σ_item` is 1.625 / 1.644 / 1.650, and the slope is
   +0.0202 / +0.0190 / +0.0187. A mechanism screen agrees: data generated with a true α spread of
   2.79 and fitted under the tight prior recovers 2.703 and yields a slope of +0.0064
   [−0.0032, +0.0167], inside the null. A prior can only bind if the likelihood is weak, and at this
   data volume it is not. **Spread exceeding the prior sd is not evidence that the prior binds.**
3. **λ is not revisited.** It was eliminated on a sign argument before either of the above, and
   nothing since has revived it. If it were ever fitted it would enter as a **declared robustness
   model compared by LOO**, never as primary.

3b. **Discrimination/scale mismatch — TESTED AND ELIMINATED.** Fitting a free discrimination
   multiplier on the same comparisons with `θ` and `α` held fixed gives **g = 1.040, 95%
   [0.981, 1.104]** — including 1.0. The residual is not a scale mismatch, which closes the last
   structural explanation.

3c. **A transfer-error explanation does not apply to this statistic.** It was proposed that the
   residual arises because `θ` and `β` are fitted on item-vs-anchor comparisons while the window
   predicts item-vs-item consistency — different tasks. Verified against the code: both
   `excess_consistency_slope` and the in-pipeline `operating_window` pivot on
   `["item_id", "anchor_id", "template"]`, i.e. they are computed on **item-vs-anchor comparisons
   using anchor-fitted parameters**. There is no transfer, so transfer error cannot explain the
   +0.0202. The concern *is* live for the separate stratified module in
   `src/experiments/operating_window.py`, which does use anchor-fitted `θ` to predict genuine
   item-vs-item comparisons; that is recorded as a caveat on that module and is testable whenever
   fresh item-vs-item data exists.

   **RESOLVED for gemma-2-2b, 2026-07-26.** With t3 and t4 repaired and all five templates at zero
   invalid readouts, the full-scale (400-item) refit gives an excess slope of **−0.0009 — flat**,
   against +0.0202 on the contaminated data. Rule 1 is satisfied: the residual is attributed to t3
   contamination, **no parameter is added, and the matter is closed for that model.** λ, a
   cell-level random effect and prior shrinkage all remain eliminated and are not revisited.

   **NOT resolved for the smaller models.** With the same clean templates, Qwen2.5-1.5B still shows
   **+0.0197** and Qwen2.5-0.5B **+0.034**. So t3 explained gemma's residual but not theirs, and the
   remaining effect is **model-dependent, concentrated in the weaker models**. The length-matched
   reliability shortfall agrees and separates them the same way: −0.068 for gemma against −0.175 for
   the 1.5B. Two independent diagnostics ordering the models identically is evidence the residual is
   real rather than an artifact of either statistic.

   Per the time-box this is now **documented as a limitation, not pursued**. It is recorded as a
   caveat on any `θ` reported for the 0.5B and 1.5B, and it does not propagate: nothing downstream
   consumes `θ`'s absolute scale.
4. If any step brings the slope within its null, remaining candidates are **dropped, not
   revisited**. No further parameters are added to chase a statistic that no longer fires.

**Nothing downstream currently consumes θ's absolute scale**, so waiting does not
propagate bias into a pending decision. Verified: Pass B difficulty selection is
quantile-based and therefore rank-only; the A2.6 discriminability measure is a ratio of
gap to gap SD and so scale-invariant; the A2.2 reliability gate is a Spearman
correlation. The one scale-dependent quantity, `pass_b.match_tolerance`, is expressed in
absolute rating points and its re-expression is explicitly deferred (A1.8, A2.5).

## A2.8 What a positive H1 would and would not establish

**Written before any Pass C data exists.** This is a limitation of the design, not a caveat
discovered after seeing a result, and it is the reason the project's unit of submission is Stage 0
**and** Stage 1 together rather than Stage 0 alone.

### The deflation

A positive H1 — a negative `(chose − yoked) × |diff|` interaction — rules out the crude
context-window account of §1.1, because a context that merely names one option favourably does not
predict a *dependency on choice difficulty*. That was the account the published rebuttal used, and
ruling it out is real.

It does **not** rule out at least two others:

**Self-perception (Bem).** The model infers its preference from its own observed behaviour rather
than resolving any conflict. Under this account, a choice between near-equal options is the more
*informative* observation — precisely because it was not forced by a large prior preference — so
self-perception predicts **larger spreading for difficult choices, the same sign as H1.**

**Bayesian conditioning on the self-generated choice token.** The choice token enters the context
and the model conditions on it. When the pre-choice preference is weak, the token carries more
information about which option is preferred, so the posterior update is larger. Again the **same
sign as H1**, and it requires no dissonance-like mechanism at all — only that the model treats its
own output as evidence.

### Why the existing conditions do not separate them

`3p-yoked` holds designation and information constant while removing authorship, so it separates
authorship from information. It does **not** separate dissonance from self-perception or from
conditioning, because another agent's choice is *weaker evidence about the model's own preferences*
than its own choice is. All three accounts therefore predict `chose > 3p-yoked`, and the contrast
cannot adjudicate between them.

### Why behaviour cannot settle it here

In the human literature this was settled with arousal-misattribution designs — manipulating an
incidental state the participant could attribute the discomfort to. There is no available analogue
for a language model, and inventing one would be a larger research program than this paper.

**Consequence: Stage 0 alone is deflatable.** A skeptic can grant every result in it and still hold
that the model is doing self-perception or Bayesian updating. **Stage 1 — representation-level
evidence that the manipulation acts on a specific, identifiable mechanism rather than on general
belief updating — is what makes the claim survive contact with that skeptic.** Stage 0 is reported
as a kill gate, which is what §1 always said it was.

### The one behavioural lever available

Commitment/irrevocability is a boundary condition for dissonance but **not** for self-perception: a
model observing its own provisional choice has observed it just as much as a final one, whereas
dissonance requires the decision to be difficult to undo. A `chose-provisional` arm therefore
carries genuine discriminating weight at low cost, and it is preregistered in A2.9.

It is not decisive on its own — a null there is also consistent with the model simply not
representing revocability — but it is the only behavioural handle this paradigm affords, and it is
cheap enough that omitting it would be indefensible.

## A2.9 Pass C dependent variable, conditions, and closing figures

**This section closes Amendment 2. Anything discovered after 2026-07-26 goes to Amendment 3, after
Pass C exists.**

### A2.9.1 The DV: option (c), modelled jointly

The spread DV is measured as the **designated-versus-other comparison itself**, pre and post, on the
pairwise instrument — not as a difference of four absolute ratings (audit item T1, resolved).

**The outcomes are modelled directly. No per-pair spread is ever computed.**

```
logit P(designated beats other) =
      δ_pair  +  β_t · s  +  post · ( γ_c  +  λ_c · diff_z )
      +  u_pair  +  u_template
```

`s = ±1` for slot, `post ∈ {0,1}`, `c` indexes condition, `diff_z` is the z-scored `|Δθ|` from the
disjoint analysis template set (§4.4). **PRIMARY TEST: `λ_chose − λ_yoked`, predicted negative.**

**Two-stage estimation is prohibited.** Estimating a per-pair spread and regressing it on gap
reintroduces exactly the artifact for which option (c) was originally rejected: with only two binary
outcomes per timepoint, a per-pair spread is a ratio of noisy quantities whose sampling variance
depends on where the pair sits on the logit curve — maximal at p ≈ 0.5, which is where difficult
pairs sit by construction. That manufactures the predicted interaction from noise. The interaction
must be estimated **inside the same joint likelihood**, where the binary nature of the observations
is respected and no intermediate quantity is formed.

**Priors on the new scale.** The DV is a logit shift, not rating points:

| parameter | prior |
|---|---|
| `γ_c` (post shift per condition) | `Normal(0, 1)` |
| `λ_c` (difficulty slope per condition) | `Normal(0, 1)` |
| `β_t` (position, per template) | `Normal(0, 1.5)` |
| `δ_pair` | `Normal(0, 2)` |
| random-intercept SDs | `HalfNormal(1)`, zero-sum constrained |

A shift of 1 logit moves a coin-flip to p = 0.73, so `Normal(0,1)` is weakly informative and not
restrictive.

### A2.9.2 θ-scale re-expression of the deferred quantities

Both were defined in absolute rating points and do not transfer (A1.8). Re-expressed as fractions of
the measured between-item scale, which is what made them meaningful in the first place:

- **`match_tolerance`**: was 0.15 rating points against `σ_between ≈ 0.58`, i.e. **0.26 σ**. Now
  `0.26 × σ_item`, computed per model from that model's own fit.
- **SESOI on the primary interaction**: was `0.15 × σ_between`. Now **`0.15 × σ_item`**, in logit
  units per SD of `|diff|`, reported with its raw-logit equivalent per model. This preserves the
  original "a fraction of the between-item scale" logic across the change of instrument.
- `sesoi_raw_secondary` (0.25 raw rating points) is **retired**; it has no meaning on a logit scale.

### A2.9.3 Conditions: a 2×2 on transcript structure × attribution

The `chose`/`yoked` contrast confounds three things: an extra user turn, an assistant turn with role
markers, and the antecedent wording. Diffed on rendered prompts, `chose` (144 tokens) against a
single-turn version with a byte-identical antecedent (118 tokens), the difference is exactly:

```
-Which one would you choose for yourself? Reply with 1 or 2 only.<|im_end|>
-<|im_start|>assistant
-1<|im_end|>
-<|im_start|>user
```

**A transformer retains no record of having produced a token.** Model-generated text fed back as
context is processed identically to experimenter-supplied text. So this contrast isolates
**role-attribution in the transcript**, not production. The term "authorship" is retired from the
paper wherever it implies otherwise; the manipulation is transcript-structural.

The four conditions form a 2×2 **on the two factors that matter**, with one caveat entered on
implementation and stated here rather than left to be noticed later:

| | antecedent "You chose X over Y" | antecedent "X, rather than Y, assigned" |
|---|---|---|
| **assistant turn present** | `chose` (turn content = its choice) | `structure-control` (turn content = neutral) |
| **assistant turn absent** | `self-recounted` | `yoked` |

`structure-control` carries an assistant turn whose content is choice-irrelevant. **Without it a
positive probe is uninterpretable**, because "an assistant turn is present" is trivially represented
and a probe separating `chose` from `self-recounted` would likely read that.

> **Correction entered 2026-07-26 on implementation.** Calling this "fully crossed" was imprecise.
> Verified on rendered prompts, the edges are:
>
> | edge | factor isolated | changed lines |
> |---|---|---|
> | `chose` − `self-recounted` | turn presence, at "you chose" wording | 4 |
> | `structure-control` − `yoked` | turn presence, at "assigned" wording | 4 |
> | `self-recounted` − `yoked` | wording only | 2 |
> | `chose` − `structure-control` | turn **content** *and* wording | 6 |
>
> Three edges isolate one factor; the fourth does not, because an assistant turn containing a
> *choice* requires a *choice question* to elicit it, and coherence then requires the "you chose"
> antecedent. The missing cell — turn present, neutral content, "you chose" wording — is semantically
> incoherent (told you chose, but you acknowledged rather than chose) and is not added.
>
> **What this identifies is still sufficient for the purpose.** Turn presence is estimated at *both*
> wording levels, so if the two estimates agree, a probe reading turn-presence is identified and can
> be subtracted — which is exactly what the structure control was added to permit. What is *not*
> separately identified is turn content independent of wording, and no claim will rest on it.
>
> Token counts, as a check on the structure factor being about structure: `chose` and
> `structure-control` are both **125 tokens** — matched exactly. `self-recounted` is 97 and `yoked`
> 103; the 6-token difference is the wording factor itself and is irreducible.

Plus `3p-yoked`, `3p-random`, `random` (unchanged), and **`chose-provisional`** — identical to
`chose` but with the finality clause replaced. A finality clause is added to **all** conditions so
receipt-matching is preserved and the reversibility contrast is one word-group. Commitment is a
boundary condition for dissonance but not for self-perception (A2.8), which is the only behavioural
leverage this paradigm affords.

**Eight conditions. 20,000 passes per model** (pre 2,000 + choice 2,000 + post 16,000), 100,000
across five.

### A2.9.4 Deliberation arm

A single-token forced readout confounds a null H1 with the readout itself: a null cannot distinguish
"no effect" from "an effect requiring deliberation the design forbids." **This is the only
configuration in which a null H1 is publishable**, so it is preregistered rather than optional.

Hard constraint 1 is unchanged and not weakened: **generation is permitted in context construction,
never in measurement.** The choice write-back already works this way. The arm generates a reasoning
span, appends it, and the DV is still read from one forward pass at one token position.

Bounded for cost: `chose` and `yoked` only, difficult pairs only, one template, 64 generated tokens.
100 pairs × 1 template × 2 orders × 2 conditions = 400 trials ≈ **25,600 generation steps + 800
readouts** per model. Reported as a secondary arm; a null in the main design is interpreted only in
light of it.

### A2.9.5 Inconclusive branch, costed

If the primary HDI overlaps both 0 and the SESOI, §8 resolves it by scaling items first. Cost, stated
now so it is not discovered later: 400 → 800 items **doubles pairwise Pass A** (40,000 → 80,000
comparisons per model) and doubles Pass B's candidate pool. Measured throughput is **8.1 passes/s**
for the 0.5B on the development machine, so this is an overnight run per model on the run machine,
not an afternoon — roughly one additional week including re-authoring 400 items to the same
no-factual-content standard.

### A2.9.6 Activation cache

Last-token hidden states across all layers are written during Pass C so Stage 1 does not require
re-running it. Sized against real configs (bf16, layers + 1):

| model | KB/pass | post only (16k) | all (20k) |
|---|---|---|---|
| qwen2.5-0.5b | 44 | 0.72 GB | 0.90 GB |
| qwen2.5-1.5b | 87 | 1.43 | 1.78 |
| qwen2.5-3b | 148 | 2.42 | 3.03 |
| gemma-2-2b | 122 | 1.99 | 2.49 |
| llama-3.2-3b | 174 | 2.85 | 3.56 |
| **total** | | **9.41 GB** | **11.76 GB** |

Tensors are written as safetensors per §12, with a **row-aligned sidecar parquet carrying the
provenance columns** — without it activation artifacts sit outside `assert_poolable` entirely. Cache
scope is decided by the run machine's storage, not the reverse; activations are write-once and can
be moved off the box after collection.

### A2.9.7 Stage 1 checkpoint criteria, both directions

Fixed before any activations exist.

**Publish Stage 0 alone if** a linear probe is at chance on all layers in every model that cleared
the reliability gate, **and** a positive-control probe on the same activations succeeds — the
designated item's identity, which must be represented. Without that control a null probe is
indistinguishable from a broken pipeline.

**A positive probe result counts only if it** (a) survives the `structure-control` condition, i.e.
separates `chose` from `structure-control` and not merely turn-present from turn-absent;
(b) generalises to held-out items; and (c) generalises across templates. All three, specified now.

## A2.5 Unchanged

Everything in A1.8 stands, minus the retired A1.4 gate. In particular: the spread DV structure, the
2 × 5 receipt-matched conditions, continuous difficulty selected on one template set and analysed on
a disjoint one, the interaction as the sole primary test, the mass-floor invariant, and the
device-pooling guard.

The Pass C dependent-variable choice (audit item T1) remains **unsettled** and is not decided here.
T1's rejection of absolute-Likert ratings for the DV stands — that instrument fails polarity validity
at full sample across all three models tested (ρ = −0.964, −0.916, −0.944) — but the choice among the
θ-based alternatives is deferred, and will be made without inspecting any H1-bearing quantity.

---

# Amendment 3 — 2026-07-27

**Status: OPEN.** Amendment 2 is frozen; this records what was found after it closed. Written
**while blind to Pass C** — the run producing the first H1 estimate was launched before this was
computed and no result from it had been seen. That sequencing is the point: A3.1 changes how a
result is *judged*, and it would be illegitimate to decide it after seeing the estimate.

## A3.1 §8's power criterion is unsatisfiable, and is withdrawn

§8 requires **"power on the primary contrast at the SESOI"** to reach 0.80. §9.2's `pass` cell
requires the 95% HDI to exclude 0 **and** the posterior median magnitude to **exceed** the SESOI.

Those two rules cannot both be met, at any sample size, by any design.

*Proof.* Let the true effect equal the SESOI exactly. As the standard error shrinks the posterior
median converges to the true value, i.e. to the threshold the `pass` cell requires it to exceed. The
median is asymptotically symmetric about it, so P(median exceeds the threshold) → **0.500** from
below. It is bounded above by 0.5 for every finite n and never reaches 0.80. Computed:

| SE | 1.96·SE | power at the SESOI | MDE at 80% |
|---|---|---|---|
| 0.200 | 0.392 | 0.212 | 0.564 |
| 0.101 | 0.199 | 0.495 | 0.323 |
| 0.050 | 0.098 | 0.498 | 0.278 |
| 0.020 | 0.039 | 0.499 | 0.253 |
| 0.001 | 0.002 | **0.500** | 0.237 |

This is a defect in the interaction of two preregistered rules, not a property of any model or of
the realized design. It was inherited from the rating-scale power module, which powered against an
HDI-only rule and never had to reconcile the two.

**Withdrawn:** "power at the SESOI ≥ 0.80" as a design criterion. It is not restated in a weaker
form and no threshold replaces it.

**Reported instead:** the **minimum detectable effect at 80% power** — the smallest |λ| the §9.2
rule can certify 80% of the time — together with its ratio to the SESOI. Its floor is the SESOI
itself, for the same reason.

The SESOI is **not** changed. 0.15 × σ_item stands exactly as Amendment 2 set it. Rescaling it in
response to a power calculation, while a run is in flight, is precisely the move this project
forbids. The arithmetic is raised; the threshold is untouched.

## A3.2 The A2.9.5 scaling branch is close to worthless for the primary contrast

§8 and A2.9.5 resolve an inconclusive primary by scaling items first, pairs second. Priced on
gemma's realized design:

| pairs | observations | SE | MDE | as ×SESOI |
|---|---|---|---|---|
| 200 (realized) | 18,000 | 0.1014 | 0.323 | 1.37× |
| 400 | 36,000 | 0.0714 | 0.297 | 1.26× |
| 800 | 72,000 | 0.0505 | 0.279 | 1.18× |

**Quadrupling the experiment moves the detectable effect by 14%.** Once 1.96·SE falls below the
SESOI — which it already has at 200 pairs — the SESOI, not precision, is the binding term, and
extra data buys almost nothing. A2.9.5 costed that branch at roughly one additional week per model;
it is now costed at one additional week for a 14% improvement, which is not worth spending.

**Consequence, stated before the data:** if the primary lands inconclusive, scaling is **not** the
remedy. The honest report is an inconclusive primary with the MDE stated. What would actually move
the answer is reducing σ_item (a tighter item pool shrinks the SESOI in absolute terms) or reducing
the position bias β, which costs up to 49% of the information on the worst template. Neither is a
Stage 0 action. Both are recorded here so that the inconclusive branch is not silently converted
into an open-ended scaling programme.

## A3.3 The equivalence branch is reachable, but only just

§9.2's `fail` cell needs the whole 95% HDI inside [−SESOI, +SESOI], i.e. 1.96·SE < SESOI. On
gemma's realized design 1.96·SE = 0.199 against a SESOI of 0.236 — reachable, with 16% of margin.
A model with a larger σ_item gets a proportionally larger SESOI and so more margin; a model with a
larger β loses information and may lose the branch entirely. `power.analyze` therefore reports
`equivalence_reachable` per model, and any model where it is false may report a null **only** as
inconclusive, never as equivalence.

**A3.1's argument applied to the `fail` cell, which A3.3 originally stopped one short of.**
Reachable is not the same as likely. Equivalence needs the whole HDI inside the ROPE, i.e. a median
within `SESOI − 1.96·SE = 0.236 − 0.199 = 0.038`. Under a true null the median has sampling SD ≈ SE,
so that window is only 0.37 SE wide either side. The three cells, computed blind on gemma's realised
design:

| truth | `pass` | `fail` (equivalence) | **`inconclusive`** |
|---|---|---|---|
| true null | 0.010 | 0.292 | **0.698** |
| true effect = SESOI | 0.495 | 0.022 | **0.483** |
| true effect = MDE (0.323) | 0.800 | 0.002 | **0.198** |

**Inconclusive is the modal outcome across the plausible range**, including when the effect is real
and exactly at the smallest size we declared interesting. This is stated before the data so that an
inconclusive primary is understood as the design's expected behaviour rather than as a failure or a
surprise, and so that §A3.2's finding — that scaling does not fix it — is read alongside it.

## A3.4 What the power module is, and what it is not

It computes the Fisher information of the exact Pass C design under the exact `spread_model`
likelihood: I = X′WX with W = diag(p(1−p)), SE = √(c′I⁻¹c). It is a design calculation, checked
against an actual sampler fit rather than trusted on its algebra. Pair and template effects are
absorbed as fixed rather than partially pooled, which makes every figure above **conservative**.

The one quantity unknown before Pass C is γ, the post shift. It enters only through p(1−p).

**Correction to this paragraph's first draft.** It originally read "across γ ∈ [0, 1.5] the SE moves
from 0.1629 to 0.1642, the assumption is not load-bearing." Those figures came from a version of the
design calculation that modelled designation as alternating across conditions. Six of the eight
conditions actually designate the model's own pick, and on easy pairs that pick is near-deterministic,
so `d` aligns with the pair gap. Correcting it changes the sensitivity materially:

| γ | SE | MDE | equivalence reachable |
|---|---|---|---|
| 0.0 | 0.0941 | 0.316 | yes |
| 0.4 (assumed) | 0.1014 | 0.323 | yes |
| 0.8 | 0.1101 | 0.330 | yes |
| 1.5 | 0.1299 | 0.365 | **no** |

So γ **is** load-bearing, for the equivalence branch specifically: a large post shift pushes
observations away from p = 0.5, costs information, and closes the `fail` cell entirely. The MDE is
comparatively stable (0.316 → 0.365). γ is measured directly by Pass C, so this resolves itself on
first contact with the data — but if γ lands near 1.5, A3.3's reachability claim does not hold and a
null is reportable only as inconclusive.

Two structural facts it makes visible, both properties of the design rather than choices:

1. **Difficult pairs sit at p ≈ 0.5, where W is maximal**, so the design concentrates information
   where the interaction is identified. This is the favourable face of the same geometry that makes
   the two-stage estimator biased — there, near-0.5 pairs have the largest sampling variance in a
   per-pair spread; here they carry the most information in the joint likelihood.
2. **Six of the eight conditions designate the model's own pick**, and on easy pairs that pick is
   near-deterministic, so `d` aligns with the pair gap and pushes those observations further from
   0.5. Easy pairs lose information twice. Modelling designation as alternating across conditions —
   the obvious synthetic shortcut — inflates the SE by about 60% by inventing within-pair contrast
   the real design does not have.

## A3.5 A2.9.6's activation cache is not implemented

A2.9.6 states that last-token hidden states "are written during Pass C so Stage 1 does not require
re-running it." **No module in `src/` collects hidden states.** `pass_c.py` writes parquet only. The
Pass C run in flight will not produce the cache, so Stage 1 does require re-running Pass C's forward
passes — roughly 30–60 minutes per model.

Recorded rather than fixed mid-run, deliberately. The run machine has no second drive and needed
`HF_HUB_DISABLE_XET=1` to fetch the 3B weights at all, so adding ~6.9 GB of writes to a live run
risks failing it. The dominant Pass C cost was the instrument fit, and that is now cached. Re-running
forward passes later is the cheaper of the two mistakes.

Sized for the three models that passed A2.2, at 18,000 observations each (2,000 shared pre + 16,000
post — A2.9.6's table predates the eight-condition design and assumed 20,000):

| model | KB/pass | total |
|---|---|---|
| qwen2.5-1.5b | 87 | 1.57 GB |
| gemma-2-2b | 122 | 2.20 GB |
| llama-3.2-3b | 174 | 3.13 GB |
| **total** | | **6.90 GB** |

The Stage 1 draft (`preregistration_stage1.md` §3) specifies a separate `collect_activations` pass
that replays the identical prompts and asserts a matching prompt digest, rather than amending Pass C.
That keeps Stage 0's artifacts as they are and decouples Stage 1's disk needs from Stage 0's runtime.

## A3.6 Considered and declined: removing `median > SESOI` from §9.2's `pass` cell

Considered removing the `median > SESOI` conjunct from §9.2's `pass` cell. Computed blind, on
gemma's realised design (SE 0.1014, SESOI 0.2359): power at the SESOI would rise from **0.495 to
0.641**, still short of 0.80, and the minimum detectable effect from **1.367× to 1.207× SESOI**.

**Declined.** Three reasons, in increasing order of weight.

1. *It is not the defect A3.1 identified.* That defect was in §8's power **target**, which is
   withdrawn and not restated. The conjunct is the surviving half of the pair and is doing work.

2. *It reduces to a rescale of the SESOI, obtained by deletion rather than by editing the number.*
   With the conjunct gone, `pass` is `1.96·SE`, which on this design is **0.199 against a SESOI of
   0.236** — a 16.0% reduction in the effective bar. A3.1 states that rescaling the SESOI in
   response to a power calculation while a run is in flight is precisely what this project forbids.
   Reaching the same place by deleting a clause does not make it a different act. If anything it is
   worse than an honest rescale: a rescale would move the ROPE too, whereas deletion loosens `pass`
   while leaving `fail` at ±0.236 — easier to declare an effect, no easier to declare equivalence.

3. *It makes the threshold float with sample size.* Once the bar is `1.96·SE` it is whatever
   precision happens to be, and shrinks toward zero as data accumulate. A SESOI exists so that the
   bar does **not** do that. This is structural and does not depend on any figure above.

**The cost is also larger than the gain.** Under a true null the per-model `pass` rate rises from
**0.0097 to 0.0247** — a 2.55× increase in false positives, and the amended rule sits at exactly the
nominal one-sided 2.5%. The current rule is conservative beyond nominal, and that conservatism is
what the deletion spends. Fifteen points of power, on a design that remains underpowered either way,
bought with a 2.55× false-positive increase and a floating threshold, taken immediately before
reading results. Blindness is a partial defence only; the *direction* of the change is predictable
without seeing anything, which is exactly what makes it attackable.

Recorded rather than silently not-done, because a refusal under live temptation is evidence about
the process and an undocumented non-decision is not. **`_decide_pass` in `src/analysis/power.py`
retains the `max(z·post_sd, sesoi)` term, and the docstring points here.**

## A3.7 The project-level gate is revised; the per-model `pass` rule is not

§9.2 holds two separable objects, and A3.6 turns on the distinction:

- **the per-model outcome rule** — what counts as a `pass`. It defines a claim about the world.
  **Frozen.**
- **the project gate** — "Stage 1 is entered only if at least two non-excluded models pass." It
  allocates compute and asserts nothing. **Revised here, blind.**

**The defect it inherits.** The project gate is a conjunction over per-model `pass`, and per-model
`pass` is capped near 0.5 at the SESOI by A3.1's argument. So the gate inherits the cap: with three
non-excluded models and a true effect exactly at the SESOI, **P(≥2 pass) = 0.493**. A real effect at
the smallest size we declared interesting fails to open Stage 1 more than half the time. The gate is
not merely strict, it is anti-correlated with what it is for.

The conjunction itself is well-behaved and is not the problem — it is roughly neutral at the SESOI
and *helps* above it, and it drives false entry under a true null to 0.0003. All of the damage comes
from the per-model cap, which is frozen.

**Priced and rejected: pooling.** Fitting the interaction hierarchically across models and gating on
the pooled contrast gives **0.497** at the SESOI — no better than the current gate. The SESOI-floored
bar caps at 0.5 at the SESOI *regardless of precision*, so pooling three models' data does not escape
A3.1's cap; it only improves the MDE (1.247× → 1.211× SESOI). Not worth a new model, three models is
a poor basis for a model-level variance, and λ would first have to be re-expressed per σ_item to be
commensurable across the ladder. Recorded so it is not re-proposed.

**Adopted — a corroboration gate.** Stage 1 is entered if:

> **at least one** non-excluded model **passes** §9.2 in full (both conjuncts, unamended),
> **and at least two** non-excluded models are **directional**, defined as
> `P(λ_interaction < 0 | data) ≥ 0.95`.

A passing model necessarily satisfies the directional condition, so it counts toward both; the gate
therefore asks for one model that clears the full bar plus one further model that agrees in
direction with high posterior mass. It does not ask lightning to strike twice.

`τ = 0.95` was chosen by a rule stated before the number was picked: **the smallest conventional
threshold whose project-level MDE is not below the SESOI.** A gate that could be cleared 80% of the
time by an effect smaller than the smallest effect we declared interesting would be a backdoor
loosening of the `pass` rule, which is the thing A3.6 refused. Computed over 400,000 simulated
three-model draws:

| τ | false entry (true null) | entry at the SESOI | project MDE / SESOI |
|---|---|---|---|
| — | 0.0003 | 0.493 | 1.247 | ← current: ≥2 of 3 pass |
| 0.90 | 0.0051 | 0.839 | **0.959** ← below the SESOI |
| 0.94 | 0.0031 | 0.799 | 1.001 ← on the boundary |
| **0.95** | **0.0025** | **0.779** | **1.019** |
| 0.975 | 0.0011 | 0.680 | 1.104 |

Entry at the SESOI rises **0.493 → 0.779**; false entry under a true null rises 0.0003 → 0.0025,
i.e. one Stage 1 wrongly begun in four hundred. That is a resource risk, not an inferential one: no
claim is made by entering Stage 1, and Stage 1 has its own preregistered gates.

**What is not changed.** The per-model `pass`, `fail` and `inconclusive` cells; the SESOI; the ROPE.
A model that is inconclusive is still reported as inconclusive — the corroboration gate lets an
inconclusive-but-directional model support *entry* without letting it support a *claim*. The ladder
pattern is still reported as a primary descriptive result regardless of the gate outcome, and a
failure to enter is still reported as a negative result.

**If fewer than three models survive the reliability gate**, the rule is unchanged in form: one full
pass plus two directional, counted over the surviving set. With two survivors it requires both. With
one it cannot be met, and Stage 1 is not entered on a single model.

## A3.8 §7.2's item random effect does not transfer to the logit DV

§7.2 preregisters a **robustness model** adding an item random effect as
`u_item[item1] + u_item[item2]`, the multi-membership form. `spread_model` does not implement
it: there is no `u_item` and no `with_item` switch. Found while migrating the analysis
notebook, which still called the retired `mixed.fit(cfg, design, with_item=True)`.

This is not merely unimplemented. **The specified form is wrong for the current DV, and the
corrected form is close to unidentified.**

*The sum form does not transfer.* §7.2 was written when the DV was a per-pair spread in
rating points, where both items contribute additively to one number. A2.9.1 replaced that
with a comparison on a fixed pair axis — `logit P(item1 beats item2)` — where item quality
enters as a **difference**, `u_item[item1] − u_item[item2]`, not a sum. Carrying the sum over
would add a term with no interpretation on the new scale.

*The difference form is barely identified.* The model already carries a free partially pooled
`u_pair` per pair. For pair `p = (i1, i2)` the baseline would become
`u_pair[p] + (u_item[i1] − u_item[i2])`, and those are separable only through items appearing
in **more than one pair**. §5's cap is **two uses per item**, so each `u_item` is informed by
at most two pairs whose own `u_pair` is free. The posterior would be dominated by the prior,
and the "discrepancy is reported either way" clause of §7.2 would then be reporting prior
sensitivity rather than robustness — the opposite of its purpose.

**Withdrawn as specified**, and *not* silently replaced. §7.2's concern is about the **DV**
side — whether item-specific susceptibility to the manipulation is left unmodelled — so the
answer has to be on that side too. It is: **`u_pair` is free per pair and absorbs
item-pair-specific susceptibility entirely.** Every Pass C observation belongs to a pair whose
baseline is estimated, so there is no unmodelled pair-level susceptibility for `u_item` to
capture. What the design cannot do is *separate* `u_item` from `u_pair`, and that is a
statement about identification, not about coverage: the variance §7.2 wanted modelled is
already absorbed, just not attributed to items.

(An earlier draft justified this by pointing at `θ` and `|diff|`. That answers the
**regressor** side — whether the item scale is well measured — which was not what §7.2 asked.)

## A3.9 `match_tolerance` was amended and never implemented

**A2.9.2 re-expressed the matching tolerance as `0.26 × σ_item`, per model. The code was never
changed.** `config.py` kept `match_tolerance: float = 0.15  # rating points` and
`pass_b.py` applied it directly as a hard filter — `if gap > tol: continue` — where `gap` is
`|mean_selection(difficult) − mean_selection(easy)|` on the **θ scale, in logits**. A constant
declared in units of a retired instrument was filtering a quantity measured in a different one.

**This is a deviation from a frozen preregistration**, found before any Pass C data for this
design existed, and it is stated as a deviation rather than absorbed.

**The damage is not strictness — it is scale.** For gemma, `σ_item = 1.573`, so A2.9.2
specifies **0.409** where the code used **0.15**: 2.7× too strict. But a *fixed* logit constant
is a *different* effective strictness for every model, varying inversely with `σ_item`. Each
model's matched sets were therefore built under a different matching rule, while the ladder
comparison in §9.2 assumes they were not. Within any single model this is invisible: the run
either fills its complement or raises, and gives no sign that the rule differed elsewhere.

**Audit item T2.3 had already caught it**, and named two constants: `match_tolerance` and
`sigma_between_min`. It was marked closed with nothing verifying it. Document and code then
diverged unwatched across two amendments. That is the failure this amendment is really about.

### The audit T2.3 should have had

Every numeric constant in `config.py` and `base.yaml`, checked against what the frozen
amendments specify:

| constant | status |
|---|---|
| `match_tolerance` | **live deviation** — fixed here |
| `sesoi_sigma_fraction` = 0.15 | correct per A2.9.2 (0.15 × 1.573 = 0.2359, the figure A3.3 quotes) |
| `chains/tune/draws`, `hdi_prob`, `rhat_max`, `ess_min` | correct per §7.1 |
| `ppc_null_replicates`, `power_target` | correct |
| `sigma_between_min` = 0.5 | **deferred by A1.8 and never resolved** — A2.9.2 resolved `match_tolerance` and the SESOI but not this. Survivable only because it now lives solely in the absolute-Pass-A path, which A1.5 made an instrument-validation record that gates nothing. |
| `validity_rho_min`, `validity_min_surviving_templates`, `icc_tripwire` | same — retired or inert, confined to that record |
| `sesoi_raw_secondary` | retired by A2.9.2; still a field, reaches no live analysis code |
| `power_n_sims` | dead — `power.py` is closed-form since A3.1; nothing reads it |
| the ±16 spread bound | deferred by A1.8; moot, the Gaussian DV it bounded is retired |

So exactly one constant was actively wrong. The point is that **the process had no way to know
that**, which is why the finding is the missing guarantee rather than the single value.

### The fix, and what it costs

`match_tolerance` is now `match_tolerance_sigma_fraction` (0.26) times the model's own
`σ_item`, computed at runtime. It is **not** retuned: any problem with 0.26 × σ_item is an
Amendment 4 item found later, not a licence to pick a new number now.

`pass_a` hashes are **unchanged** — the 40k anchor comparisons per model survive. `pass_b` and
`pass_c` hashes move, so Pass B and Pass C must be rebuilt. A3.2 and A3.3's figures are
recomputed from the rebuilt Pass B on the run machine, which resolves their
development-machine provenance as a side effect.

**One collision had to be closed first.** The config hash keys on the *fraction*, which is
identical across models, while the value actually applied depends on `σ_item` — and `σ_item`
comes from the instrument fit, keyed on the estimator's **source digest**, which no config
field can see. Change the estimator and the tolerance moves while the `pass_b` hash stands
still: one hash, two different pair sets. That is the stimulus-digest failure again. The
realized tolerance and `σ_item` are therefore written into the pairs artifact itself, and a
cached Pass B whose recorded tolerance disagrees with the current one is **refused**, as is any
artifact predating this amendment (those were built with 0.15 and cannot be verified).

### A3.10 The ledger

Four preregistered elements have now been withdrawn outright, seven superseded, one inverted
and one left open. **`PREREGISTRATION_LEDGER.md`** is the first-hand count: every element of
§1–§13, its fate, a one-line reason, and the amendment that decided it. It is built now, before
write-up, because a reviewer tallying withdrawals across five amendments will arrive at a number
and not at the reasons.

Two facts decide how the count reads, and both are checkable: **every withdrawal is on the
instrument or inference side — none touches a hypothesis** — and **every one was made with no
H1-bearing data in hand**. Nine of the twelve non-kept rows trace to a single decision, A1.1's
replacement of the absolute rating instrument, propagating through the scale, the polarity
machinery, the DV, the model, the priors and every constant expressed in rating points.

The ledger corrects one of our own counts. Withdrawals were being tallied as three (§5.1, §8,
§7.2); **§6.5 belongs on the list and is the largest of them** — the dependent variable named in
the original specification is not computed anywhere, and the module that replaced it exposes no
function returning one. Calling that an amendment understates it.

### A3.11 A2.4's discrepancy is measured against the wrong comparator

A2.4 records the reliability discrepancy as **"empirical split-half 0.770 against a
model-implied reliability of 0.957"**. Those two figures are not comparable, and
`bradley_terry.predicted_split_half` exists to say so — its docstring states that comparing
them directly "guarantees an apparent gap and would report every model as misspecified."

Model reliability is fitted on **all five** templates. The empirical figure is the
correlation between **two shorter fits**, on 3 and 2 templates, each carrying more error
variance. With `r_full = σ²/(σ²+v)` and `k = 1/r_full − 1`, a half holding fraction `f` of
the data has error variance `v/f`, so the length-matched prediction is

```
r_split = 1 / sqrt((1 + k/f_a) · (1 + k/f_b))
```

At `r_full = 0.957` on the 3/2 split this gives **0.9145**, not 0.957.

| | comparator | shortfall |
|---|---|---|
| as recorded in A2.4 | 0.957 | **−0.187** |
| length-matched (correct) | 0.9145 | **−0.145** |

**A2.4 remains open.** The correction removes about 23% of the recorded gap and does not
close it: 0.770 is still well below 0.9145. What changes is the magnitude, and it changes
in the direction that had the instrument looking worse than it is. A2.4's leading candidate
— t3's ~48% invalid readouts contaminating one half — is unaffected and still discriminated
by the t3 re-collection.

**Why it survived.** `pass_a_pairwise.main()` computed and printed the length-matched
prediction, with the line "compare against THIS, not against the full-data figure".
`run.py` — the path the reported numbers actually come from — did not. Two report paths for
one quantity, and the amendment was written from the one that omitted the correction. Both
now print it, and a test asserts both do.

This is the R3/R4 class a third time: a statistic computed correctly and compared against
the wrong null. It is recorded here rather than edited into A2.4, which is frozen.

**Also fixed, same sweep.** `excess_slope_ppc_null` returned `NaN` for the whole null if any
single replicate fell under the 20-cell floor, and `run.py` guards on `null_sd > 0`, which
is `False` for `NaN` — so the misspecification diagnostic reported nothing at all, silently.
Failed replicates are now dropped and counted, and losing more than half is an error.

### A3.12 The difficulty manipulation and the DV are elicited by different questions

Found by rendering every prompt the pipeline sends and reading them — the same sweep that
found the choice-prompt defect. **Pass A elicits every anchor comparison with the CHOICE
question; the Pass C DV asks about PREFERENCE.** Per template:

| template | Pass A (θ, and the Pass C choice) | Pass C DV, pre and post |
|---|---|---|
| t0 | "Which one would you **choose** for yourself?" | "Which of the two do you **prefer**?" |
| t1 | "**Pick** the one you would rather have." | "Which one do you **prefer**?" |
| t3 | "**Select** the one you want." | "State which you **prefer**." |

So `θ` — and therefore `|diff|`, the difficulty manipulation and the primary regressor —
lives on the **choice** framing, while the outcome that moves is measured on the
**preference** framing. This is recorded nowhere in §1–§13 or in Amendments 1–2.

**It is not obviously wrong, and one half of it is right.** The Pass C choice elicitation
uses the same question as Pass A, so `θ` predicts the model's own pick on the same task it
was measured on — which is what `OWN_PICK_CONDITIONS` designation needs, and what makes
`chose` coherent. The DV *has* to differ from the choice question, or the post measurement
would re-elicit the choice rather than measure preference change.

**The exposure is on the regressor.** "Difficult" means *θ-choose values are close*. If the
two framings order items differently, `|diff|` is a noisy proxy for the difficulty of the
question the DV actually asks, and `λ` is **attenuated toward zero**. That direction matters:
it cannot manufacture H1, only hide it. On a design A3.3 already shows is underpowered,
though, attenuation is not a comfortable bias to carry unmeasured.

**Pre-specified check, blind, before Pass C.** `bradley_terry.framing_transfer` computes the
rate at which the **pre-manipulation** preference agrees with the sign of `θ`, overall and
by difficulty stratum. It runs on the PRE rows only — the shared baseline, measured before
any manipulation exists — so it **carries no information about H1**: at pre there is no
condition, and it cannot distinguish `chose` from `yoked`. That is what makes it safe to
compute before the primary analysis, and it is why it is specified here rather than after.

Reported per model, interpreted as follows, and **committed to now**:

- **Difficult stratum near 0.5 is expected and is not a finding** — those are the pairs
  where `θ` says the items are close.
- **The easy stratum is the diagnostic.** Agreement there should be well above chance. If it
  is not, the two framings order items differently and `|diff|` is measuring the difficulty
  of a different question.
- This is **reported, not a gate.** No model is excluded on it. A low easy-stratum figure is
  reported as a limitation bounding the interpretation of `λ`, because inventing an
  exclusion rule after seeing the instrument behave is the move A3.6 declined.

`pass_b` now persists **signed** `theta_item1` / `theta_item2`. It previously emitted only
`diff_analysis = |θ₁ − θ₂|`, which cannot say *which* item the instrument prefers, so the
check was not computable from the artifact at all.

### A3.13 The match-gap exclusion is vacuous as posed, and the non-vacuous version is elsewhere

Review item 9 asks for pairs exceeding their own matching tolerance to be excluded, with N
reported and the primary reported both ways. **As posed it can never exclude anything.**

`_match_domain` applies the tolerance as a **hard filter at construction** —
`if gap > tol: continue` — and raises rather than accept fewer matched sets. So every
committed set satisfies `gap ≤ tol` on the selection score by construction, and N is
identically 0 at any tolerance. The run machine's realized maxima confirm it empirically:
**0.406 against a tolerance of 0.409** (gemma) and **0.418 against 0.418** (qwen-1.5b) —
pressed flush against the bound, which is the signature of a hard filter rather than of
comfortable matching.

Reporting "0 pairs excluded" would therefore be true and misleading. It is reported as
`selection_is_bounded_by_construction`, so the zero reads as *enforced upstream* rather
than *checked and found clean*.

**What is not bounded, and is the real exposure.** Matching is enforced on `mean_selection`,
from templates T1–T3. The analysis measurement `mean_analysis`, from the **disjoint** T4–T5
(§4.4), is a separate estimate of the same quantity and **nothing constrains it**. Residual
imbalance on the analysis scale can and does exceed the tolerance — the run machine reports
analysis-gap maxima of **1.52** and **1.98**, i.e. 3.7× and 4.7× their tolerances.

That is exactly the leakage the concern points at: difficulty confounded with extremity
*through the measurement the primary model actually uses*, which is the confound
design-level matching existed to remove (§13 item 8). §4.4's disjoint split makes it
unavoidable — the price of an uncontaminated regressor is that matching cannot bind it.

**Pre-specified now, blind.** `pass_b.match_gap_exclusions` flags matched sets whose
**analysis-scale** gap exceeds the **same preregistered `0.26 × σ_item`**. No new number is
introduced; the threshold is A2.9.2's, applied to the quantity matching does not bound.

- Reported per model: N and fraction of matched sets over tolerance, and the analysis-gap
  mean and max.
- The primary is reported **both ways** — all matched sets, and restricted to those within
  tolerance on the analysis scale — as review item 9 asked.
- **Neither is the primary.** The full-sample estimate remains the preregistered primary;
  the restricted one is a sensitivity analysis. Promoting whichever is larger after seeing
  both is the move A3.6 declined.

### A3.14 The readout-mass floor guards the instrument and not the outcome

Three statements are in tension, and the code implements a fourth thing:

| source | says |
|---|---|
| §5.4 (frozen) | "Trial-level exclusions. **None.** All trials are retained." |
| A1.6 | below the floor a trial is "marked **invalid and logged**; never silently scored" |
| `readout/validity.py:14` | "The floor is **not a filter applied at analysis time**. It travels with the trial." |
| `bradley_terry.py:121` | `block = block[block["readout_valid"]]` — **it is a filter, at analysis time** |

`fit_bradley_terry` drops invalid readouts before fitting, as do `order_invariance`,
`excess_consistency_slope` and `excess_slope_ppc_null`. So there **is** a trial-level
exclusion, it operates on the instrument, and three separate places say there isn't one.

**The asymmetry is the substantive part.** `spread_model.prepare` never mentions
`readout_valid` — the string does not appear in the module. Pass C trials below the floor
therefore enter the **DV** unfiltered. `MASS_FLOOR = 0.5`, so such a trial is one where
**most of the probability mass went somewhere other than the option labels**, and its
`item1_wins` is an argmax over a minority of the distribution. The mass floor was added by
A1.6 in direct response to a readout with 0.8% of its mass on the candidate tokens
producing plausible numbers — and that exact failure is filtered out of `θ` and left in the
outcome.

This is not idle: A2.4 records a Pass A template running at **~48% invalid readouts**. High
invalid rates occur in this project.

**Direction of the risk.** Invalid trials are near-arbitrary win/loss draws, so they add
noise to the DV. That **attenuates `λ` toward zero** — it cannot manufacture H1. But it
inflates exactly the variance the equivalence claims are computed against, on a design A3.3
already shows is underpowered, which is the reason §3.1 forbids quantization.

**Pre-specified now, blind.** No filter is introduced mid-run and §5.4's rule is not
rewritten. Instead:

- **Reported per model and per condition**: the count and fraction of Pass C trials below
  the floor. `run.py` already prints `validity.summarize(trials, by=["condition"])`; that
  figure becomes a reported result rather than console output.
- **If any condition exceeds 5% invalid**, the primary is additionally reported on the
  valid-only subset as a **sensitivity analysis**. The full sample stays primary. Choosing
  between them after seeing both is the move A3.6 declined.
- **A condition above 20% invalid is an instrument failure for that condition**, reported
  as such, and its contrasts are not interpreted. That is a statement about the readout,
  not about H1.

The 5% and 20% figures are set here, before the data, precisely because they cannot be set
afterwards.

### A3.15 A2.9.3's token-match check is wrong

A2.9.3 reports, as evidence that the structure factor is about structure rather than
length: *"`chose` and `structure-control` are both **125 tokens** — matched exactly."*

Measured on the actual templates, Qwen2.5 tokenizer, identical item strings in both
conditions:

| template | `chose` | `structure-control` | difference |
|---|---|---|---|
| t0 | 121 | 123 | **−2** |
| t1 | 121 | 118 | **+3** |
| t2 | 120 | 119 | **+1** |
| t3 | 125 | 122 | **+3** |
| t4 | 127 | 127 | 0 |

**Only t4 is matched.** The counts are not a single 125 either — they range 120–127 with
real items and 102–109 with placeholders, because token count depends on the item strings,
which vary across 400 items. The quoted 125 appears to be `chose` on t3 alone, reported as
though it characterised both conditions across the battery.

**Why it is not fatal.** The gap is ≤3 tokens on ~122, under 2.5%, and it is the
confirm-versus-choice wording — which **is** the factor being manipulated, not an
extraneous confound. A perfectly length-matched pair would require the two questions to
tokenize identically, which no paraphrase battery can guarantee.

**Why it still matters.** `template` is a modelled factor, so *between*-template length
variation is absorbed by `u_template`. The `chose` − `structure-control` difference is
**within** template, so `u_template` does not absorb it and it lands in the condition
contrast. More importantly, A2.9.3 performed this check *because* length confounding
matters, and then recorded a number that does not hold.

**Corrected claim**, replacing "matched exactly": the two conditions differ by **−2 to +3
tokens depending on template, and are exactly matched on one of five**. The per-template
counts are asserted by `tests/test_design.py` so the figure cannot drift again, and the
difference is reported alongside the structure contrast rather than asserted away.

### The standing rule this creates

**An amendment that changes a number gets a test that reads the rule and asserts the config
implements it.** Amendments without enforcement tests are aspirational.
`tests/test_amendments_are_implemented.py` holds them, and each carries a **negative control**
— a demonstration that it fails when the defect is present. The first version of that file
passed two checks vacuously by matching compiled bytecode, which is the same failure class one
level up, and is why the controls are mandatory rather than encouraged.

### Disclosure

One Pass C artifact has been opened:
`artifacts/analysis/qwen2.5-3b-instruct/24e7b03199a1/results_24e7b03199a1.json`, read on
2026-07-27 while enumerating artifacts during this audit. It is `smoke: True` — reduced
stimuli — on `device: mps`, `git_sha: no-commit`, `git_dirty: True`, dated 2026-07-26, under
the pre-Amendment-2 absolute-rating instrument with the Gaussian spread DV and five conditions,
for a model the current reliability gate **excludes**. `assert_reportable` rejects it on four
independent grounds.

**Only the `outcome` string was read** (`primary-inconclusive`). No effect size, HDI,
direction, or contrast value was viewed. This is disclosed because A2.0 states "No Pass C data
of any kind exists", and a smoke-mode Pass C artifact is a kind of Pass C data — the claim was
true of the design but not literally true as written.

Any replacement robustness model would have to be specified against the logit DV and its
identification demonstrated before it is fit. That is Stage 1 work at the earliest, and doing
it now, mid-run, is the move A3.6 declined. The notebook reports this section as withdrawn
with a pointer here rather than omitting it.

---

# Amendment 4 — 2026-07-28

**Status: OPEN. This amendment is POST-HOC and every entry in it is to be read that way.**

Amendment 3 was written blind. Stage 0 is now complete and its outcome is known: two models
excluded at the reliability gate, three through to Pass C, **all three primaries
`inconclusive`** — gemma and qwen-1.5B near zero, llama a credible effect in the **wrong
(positive)** direction. Nothing recorded from here can claim the protection A1.0, A2.0 and
A3's header claimed, and no entry here may change a decision rule. That boundary is dated
rather than blurred.

**The result is what the design predicted.** A3.3, computed blind, put `inconclusive` at
~48% even when the effect is real and exactly at the SESOI, and ~70% under a true null. All
three inconclusive is the modal outcome, not a surprise, and it will not be written up as
one.

## A4.1 llama's H1 fit has 362 divergences: a `sd_template` funnel

Reported by the run machine. **4.5% of draws (362/8000) diverged**, across 3 of 4 chains.
Energy is healthy (BFMI 0.90–1.00), so this is local geometry rather than a mixing failure.

**Mechanism: Neal's funnel on the template random-effect scale.** `spread_model.py:178` uses
a **centered** parameterization, `u_template = ZeroSumNormal(sigma=sd_template)`, which
funnels when the group SD approaches zero. Divergent draws sit at `sd_template` median
**0.003** against **0.031** for non-divergent — ten times deeper into the neck. `sd_pair`
shows no such split (1.017 vs 1.014), so it is specific to the template scale.

**Divergence count tracks the group SD exactly across models**, which is what makes the
mechanism rather than a coincidence:

| model | `sd_template` mean | divergences |
|---|---|---|
| gemma | 0.303 | 7 (0.1%) |
| qwen-1.5B | 0.132 | 0.9% (70) |
| llama | 0.043 | **4.5% (362)** |

llama's templates produce nearly identical spread, so its `sd_template` posterior piles
against zero — the sharpest part of the funnel. Same model, harder data.

**What it does and does not touch.** Everything feeding the primary is clean:
`lambda[chose]` R̂ 1.0 / ESS 3831, `lambda[yoked]` R̂ 1.0 / ESS 3281; all `lambda`, `gamma`
and `beta` at max R̂ 1.0 and min ESS_bulk 3000. The single degraded parameter is
`sd_template` itself, at **ESS_bulk 646, ESS_tail 357**.

**That is a §7.1 violation and it is recorded as one.** §7.1 requires ESS > 400 "for every
reported parameter; failure is reported rather than silently re-tuned." ESS_tail 357 is
below it. `sd_template` is a nuisance variance and no claim rests on it, so the consequence
is narrow — **"how much templates vary in spread" cannot be reported for llama** — but the
threshold is stated per parameter and this parameter misses it.

### The reparameterization, and the commitment made BEFORE running it

Non-centering the template effect is a **pure reparameterization**: identical model,
identical posterior in expectation, different sampler geometry. It is the standard remedy
and it is correct. It is also being contemplated *after* seeing an inconvenient result on
the model that produced it, which is exactly the position A3.6 refused to exploit.

So the prediction and the decision rule are fixed **here, before the refit runs**:

1. **Prediction.** `lambda_chose − lambda_yoked` for llama is **+0.436, 95% HDI
   [+0.213, +0.699]** under the centered fit. The non-centered fit is predicted to
   reproduce it to within Monte Carlo error. Divergences are predicted to fall to ~0 and
   `sd_template` ESS to recover above 400.
2. **The centered fit remains PRIMARY.** The non-centered fit is a **robustness check**.
   This holds whichever direction the numbers move, and is committed now precisely so it
   cannot be chosen later.
3. **If the estimate moves materially, that is a finding, not a correction.** It would mean
   the divergences did bias the primary, and the conclusion would be that llama's H1
   estimate is not trustworthy under *either* parameterization — not that the second one is
   the right answer.
4. **Both fits are reported for all three models**, not just llama. Refitting only the model
   with an unwelcome result would be indefensible regardless of the reparameterization's
   validity.
5. The change is confined to `u_template`. `u_pair` is **not** non-centered: its diagnostics
   show no funnel, and changing an unimplicated part of the model after seeing results has
   no justification.

`target_accept` is **not** raised. It is a weaker remedy that does not clear a centered
funnel, and moving a sampler knob after seeing the answer is the move §7.1's "rather than
silently re-tuned" was written against.

## A4.2 The blind power analysis over-estimated precision by up to 5.4×

Measured on the completed run. This is a **post-hoc measurement of the design**, not a
change to any rule, and it is the most consequential thing Stage 0 produced.

| model | SESOI | SE predicted (A3.1, blind) | SE realized | ratio | MDE predicted | **MDE realized** |
|---|---|---|---|---|---|---|
| llama-3.2-3b | 0.1665 | 0.1178 | 0.1241 | **1.05×** | 1.99× SESOI | **2.09× SESOI** |
| qwen2.5-1.5b | 0.2418 | 0.1223 | 0.4202 | **3.44×** | 1.43× SESOI | **4.87× SESOI** |
| gemma-2-2b | 0.2355 | 0.1027 | 0.5525 | **5.38×** | 1.37× SESOI | **6.57× SESOI** |

`SE realized` is the posterior HDI width of `lambda_chose − lambda_yoked` divided by
2 × 1.96. All three ran the full 200 pairs, so this is not a sample-size shortfall.

**The stated direction of the error is falsified.** `power.py:52` claims:

> *CONSERVATISM. Pair and template effects are absorbed as fixed effects, whereas the
> fitted model partially pools them.*

— i.e. the design SE should come out **wide, never narrow**. `tests/test_power.py`
asserted the same, and A3.4 repeated it. Reality is narrow by up to 5.4×. The one
directional guarantee the power module made about its own error is wrong, and it is wrong
in the **anti-conservative** direction.

**Candidate mechanism, offered as a hypothesis and not as an established result.** The
error tracks the fitted between-template SD monotonically across all three models:

| model | `sd_template` | SE ratio |
|---|---|---|
| gemma | 0.303 | 5.38× |
| qwen-1.5B | 0.132 | 3.44× |
| llama | 0.043 | 1.05× |

Absorbing template as a fixed effect removes its variance from the Fisher information; the
fitted model estimates it, and its uncertainty propagates. Where between-template variance
is near zero (llama) the two agree almost exactly, which is what a correctly-specified
information calculation should do. **n = 3. This ordering is suggestive, not demonstrated**,
and it must not be reported as a mechanism without a simulation that varies `sd_template`
directly.

**Consequences, all of which are reporting consequences.**

1. **A3.2's and A3.3's figures are superseded as descriptions of the realized design.** They
   were computed blind on gemma's staged design and were honest at the time; they are
   retained as what was predicted, alongside what occurred.
2. **The study is substantially more underpowered than A3.3 said**, and A3.3 already said it
   was underpowered. The realized MDE is 2.1–6.6× the SESOI. An effect at the SESOI was
   never detectable on this design, and now that is measured rather than modelled.
3. **The three `inconclusive` outcomes are the expected result of the realized precision.**
   With gemma's HDI spanning [−1.08, +1.09] against a SESOI of 0.236, `inconclusive` was
   close to the only reachable cell — the ROPE is 22× narrower than the interval.
4. **A3.1's withdrawal of §8 is reinforced, not undermined.** §8 asked for 80% power at the
   SESOI; the realized design cannot reach 80% power at 6× the SESOI for gemma.

**Why the power module's own validation missed it.** `tests/test_power.py` checks the closed
form against an actual sampler fit and passed, at `rel=0.5`. It used 60 pairs and 3
templates of synthetic data generated from the model — a design whose between-template
variance is whatever `_synth` happened to produce. The test could not express the failure
because its fixture did not vary the quantity the error depends on. That is the same
fixture-too-small-to-express-the-failure pattern recorded three times in the test suite, and
this is its most expensive instance: it validated a power analysis that was wrong by 5×.

**Not changed.** No threshold, no SESOI, no decision rule. The gate outcome stands as
computed.
