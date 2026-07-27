# Preregistration ledger

Every element of `preregistration.md` §1–§13, its current fate, and the amendment that
decided it. Built 2026-07-27, before any confirmatory Pass C data existed.

**Why this exists.** Four preregistered elements have now been withdrawn outright and several
more superseded. Each is defensible on its own; a reviewer counting them is not going to read
five amendments to discover that. This table is the count, made first-hand, with the reasons
attached — and it carries the two facts that decide how the count should be read:

1. **Every withdrawal is on the instrument or inference side. None touches a hypothesis.**
   H1 — the difficulty × agency interaction — is stated in §2 exactly as it was at `57ef60d`,
   and §1.1's argument for why the interaction and not the main effect is untouched.
2. **Every withdrawal was made with zero H1-bearing data in hand.** A1.0, A2.0 and A3's header
   each record the evidentiary state at the time; the one Pass C artifact ever opened is
   disclosed in A3.9 and is smoke-mode output from a retired instrument on an excluded model.

The SESOI has never been changed. It was fixed at `0.15 × σ` before Pass C and re-expressed
onto the θ scale without altering the fraction (A2.9.2), and A3.6 records a considered refusal
to loosen the criterion it feeds while a run was in flight.

Legend: **KEPT** · **AMENDED** (same purpose, changed form) · **SUPERSEDED** (purpose overtaken
by a later design) · **WITHDRAWN** (removed, nothing put in its place) · **OPEN** (integrity
question unresolved).

---

## §1–§3 Claim, hypotheses, measurement

| Element | Fate | Reason | Amendment |
|---|---|---|---|
| §1 Claim under test | **KEPT** | unchanged since `57ef60d` | — |
| §1.1 Interaction, not main effect | **KEPT** | the rebuttal argument that makes the design worth running | — |
| §2 H1 (difficulty × agency) | **KEPT** | the Stage 0 test, stated as originally written | — |
| §3.1 One forward pass, one position | **KEPT** | — | — |
| §3.1 bf16 only, no quantisation | **KEPT** | — | — |
| §3.1 Expected value, never argmax | **KEPT** | retained for the digit readout | — |
| §3.1 Seeded, counterbalanced by construction | **KEPT** | — | — |
| §3.2 1–9 rating scale | **SUPERSEDED** | absolute Likert retired as the item instrument; anchor-based pairwise replaces it | A1.1 |
| §3.3 EV over digit tokens | **AMENDED** | mechanism retained, now reads anchor comparisons rather than item ratings | A1.1 |
| §3.4 Polarity reversal `10 − x` | **SUPERSEDED** | polarity only exists on the absolute scale; survives solely in the inert A1.5 record | A1.1, A1.5 |
| §3.5 Choice readout, argmax over labels | **AMENDED** | letter labels → digit labels | A1.3 |

## §4 Pass A

| Element | Fate | Reason | Amendment |
|---|---|---|---|
| §4.1 400 items, 4 domains | **KEPT** | — | — |
| §4.2 5 templates × 2 polarities, absolute | **SUPERSEDED** | replaced by anchor comparisons | A1.1 |
| §4.3 Polarity collapse | **SUPERSEDED** | no polarity on the θ scale | A1.1 |
| §4.4 Disjoint selection/analysis split | **KEPT — OPEN** | the split itself is intact and A2.3 W2 confirms its independence assumption; but t3 was broken (~48% invalid readouts) in the reported run and falls in the two-template analysis half, so one side is half-contaminated. The t3 re-collection discriminates it. | A2.3 W2, A2.4 |

## §5 Exclusion criteria

| Element | Fate | Reason | Amendment |
|---|---|---|---|
| §5.1 Polarity validity ρ ≥ 0.6 — "the sole categorical exclusion" | **WITHDRAWN** | retired and replaced by an order-invariance gate — **which was itself retired**, so the original criterion has no successor. Recorded in the A1.5 instrument-validation record, which gates nothing. | A1.2, then A2.2 |
| §5.2 Dynamic range, `σ_between < 0.5` | **SUPERSEDED** | stated in rating points; A1.8 deferred re-expressing it and no amendment ever did. Survives only inside the inert A1.5 record. | A1.8, A3.9 |
| §5.3 Reliability "does not exclude a model" | **INVERTED** | reliability became **the sole exclusion criterion** — the exact opposite of what §5.3 preregistered. The empirical split-half of θ is used, not the model-internal figure. | A2.2 |
| §5.4 No trial-level exclusions | **KEPT** | — | — |

## §6 Pass B and Pass C design

| Element | Fate | Reason | Amendment |
|---|---|---|---|
| §6.1 Pair construction, matched on mean rating | **KEPT** | matching survives; its tolerance was re-expressed as `0.26 × σ_item` and only implemented in A3.9 | A2.9.2, A3.9 |
| §6.2 Five conditions × difficulty | **AMENDED** | eight conditions: the four-cell 2×2 on transcript structure × attribution, plus `chose-provisional` for reversibility | A2.9.3, A2.8 |
| §6.3 Counterbalancing, shared prefix, KV cache | **KEPT** | — | — |
| §6.4 Choice elicitation, yoking within order | **KEPT** | — | — |
| §6.5 DV = per-pair `spread`, bounded ±16 | **WITHDRAWN** | outcomes are modelled directly; **no per-pair spread is ever formed**. A two-stage estimate has sampling variance maximal at p = 0.5, which is where difficult pairs sit by construction, so it manufactures the predicted interaction out of noise. | A2.9.1 |
| §6.6 Forward-pass budget (~30k/model) | **SUPERSEDED** | recomputed for eight conditions | A2.9.3, A3.5 |

## §7 Statistical model

| Element | Fate | Reason | Amendment |
|---|---|---|---|
| §7 Gaussian cell-means model on `spread` | **SUPERSEDED** | Bernoulli logit on the comparison, with `λ_c` as the interaction inside the likelihood | A2.9.1 |
| §7.1 Sampling: 4 chains, 2000+2000, R̂ < 1.01, ESS > 400 | **KEPT** | convergence failure is reported, never silently re-tuned | — |
| §7.1 Priors on the spread scale | **AMENDED** | re-expressed on the logit scale | A2.9.1 |
| §7.2 Item random effect (robustness model) | **WITHDRAWN** | `u_item[item1] + u_item[item2]` was written for an additive per-pair DV; on the fixed pair axis item identity enters as a **difference**, separable from the free per-pair `u_pair` only through items in several pairs — and reuse is capped at 2. Would report prior sensitivity, not robustness. `u_pair` already absorbs the variance §7.2 wanted modelled. | A3.8 |

## §8–§9 Power, SESOI, gates

| Element | Fate | Reason | Amendment |
|---|---|---|---|
| §8 "80% power at the SESOI" | **WITHDRAWN** | unsatisfiable at any n by any design: with the true effect at the SESOI, `pass` requires the median to exceed a threshold it is centred on, so power → 0.500 from below. Nothing replaces it. | A3.1 |
| §8 Power reported before Pass C | **KEPT — strengthened** | now the **minimum detectable effect**, computed between Pass B and Pass C by code that has seen no outcome | A3.1 |
| §8/A2.9.5 Inconclusive → scale items | **KEPT — priced** | retained, but 4× the data moves the MDE 1.37× → 1.18× SESOI; recorded as close to worthless for the primary contrast | A3.2 |
| §9.1 SESOI = `0.15 × σ_between` | **AMENDED** | now `0.15 × σ_item`; the **fraction is unchanged** | A2.9.2 |
| §9.1 Secondary anchor, 0.25 raw points | **WITHDRAWN** | no meaning on a logit scale | A2.9.2 |
| §9.2 `pass` / `fail` / `inconclusive` cells | **KEPT** | removing the `median > SESOI` conjunct was computed and **declined**: it sets the bar to `1.96·SE`, which floats with n, and raises the false-positive rate 2.55× | A3.6 |
| §9.2 Project gate: "≥ 2 models pass" | **AMENDED** | inherited A3.1's cap (49.3% entry on a real effect at the SESOI). Now one full pass **plus** two models directional at `P(λ<0) ≥ 0.95`. | A3.7 |

## §10–§13 Scope, provenance, prior deviations

| Element | Fate | Reason | Amendment |
|---|---|---|---|
| §10 Stage 2 equivalence, ratio posterior, f = 0.25 | **KEPT** | deferred, unmodified | — |
| §11 Out of scope for Stage 0 | **KEPT** | no steering, probes, CAA or knowledge-conflict built | — |
| §12 Provenance, cross-device pooling guard | **KEPT — strengthened** | `assert_reportable`; stage hashes made checkout- and platform-independent | A3.9 |
| §12.1 Five-model ladder | **REDUCED** | three models survive the reliability gate; the ladder is reported regardless of gate outcome | A2.2 |
| §13 Deviations table (10 entries) | **SUPERSEDED** | this ledger extends it; §13's entries 1–3 are arithmetic facts and 4–10 predate all data | — |

---

## The count

**Withdrawn outright: four** — §5.1 (and its replacement), §6.5, §7.2, §8.
**Superseded: seven.** **Inverted: one** (§5.3). **Open: one** (§4.4).

A note on the arithmetic, since it will be checked. An earlier count put withdrawals at three
(§5.1, §8, §7.2). **§6.5 belongs on the list and is the largest of them**: the dependent
variable named in the original specification is not computed anywhere, and the module that
replaced it deliberately exposes no function returning one. Calling that an amendment
understates it.

**What the pattern is.** Nine of the twelve non-KEPT rows trace to a single decision — A1.1's
replacement of the absolute rating instrument with anchor-based pairwise comparisons — and its
consequences: the scale, the polarity machinery, the DV, the model, the priors, and every
constant expressed in rating points. That is one instrument change propagating, not twelve
independent retreats. The three that do not trace to it are §5.3's inversion, §8's
unsatisfiability, and §7.2's identification failure, and each was found by checking a
preregistered claim against arithmetic or code rather than against a result.

**What it is not.** No hypothesis has been weakened, narrowed, or withdrawn. No threshold has
been loosened — the one opportunity to do so was declined and recorded (A3.6). No withdrawal
followed sight of an H1-bearing estimate.

Related: `RETRACTIONS.md` records claims we made and then refuted, which is a different ledger
from this one — that file is about our own errors, this one about the specification's fate.
