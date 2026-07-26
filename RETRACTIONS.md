# Retractions

Every claim reported to the project owner that was later refuted, with what refuted
it. Kept because four have now occurred, and because the pattern they share is
reporting a number faster than verifying it.

None of these reached a paper, a preprint, or the run machine's confirmatory record.
All were caught internally. That is the system working, but the frequency is the
problem, not the catching.

**The structural fix is `scripts/status.py`.** Reported numbers are generated from
artifacts, not typed. Three of the four below involved a number that was transcribed
or transferred by hand.

---

## R5 — "lambda <= 0.04 from the high-gap consistency ceiling"

- **Reported:** in conversation, as an argument that a position-capture mixture was
  small enough not to need fitting.
- **Refuted by:** the bound was derived against a null of 0.5. That is the same wrong
  null that produced R3, so the bound is void rather than merely imprecise.
- **Does not need redoing.** Independently of the bound, lambda is ruled out on a sign
  argument: dC/dlambda is negative at every gap, and its magnitude is *largest* at wide
  gaps (-0.95 at gap 4.5 versus -0.25 at gap 0.3), where the residual is already
  positive. A grid search over (lambda, beta) against the observed curve puts the
  optimum at **lambda = 0.000 exactly**. A mixture weight cannot produce a residual
  that changes sign with gap.
- **Consequences:** none downstream. Recorded because the log's value is being complete,
  and because this is the third instance of the same error class.

## R4 — "Template responses are non-independent (ICC 0.529, design effect 3.12,
posterior SD understated ~1.8x)"

- **Reported:** in conversation, and committed in `1d600ae` with
  `src/analysis/template_dependence.py`.
- **Refuted by:** a proper over-dispersion test. Within-cell success counts against
  the binomial expectation implied by the fitted probabilities give dispersion
  **1.070, 95% CI [0.907, 1.258]** — consistent with independence.
- **Cause:** the ICC treated templates as raters over cells, which conflates genuine
  between-cell variation in `p` — already captured by `θ − α` — with excess
  dependence. Cells differ in `p` by design, so that ICC is large under perfect
  independence.
- **Consequences:** no design effect exists. No cell-level random effect is added.
  §4.4's disjoint-split independence is intact. The separation ratio is not reduced
  by 4.39 → 2.5; under the corrected model it *rises* to 4.77. Methods-paper
  candidate #4 is withdrawn.
- **This one was ours, not the reviewer's**, and the review built a priority on it.
- Recorded in preregistration.md A2.3 W2.

## R3 — "The operating window is closed at near-equal pairs"

- **Reported:** in conversation as a paradigm-level finding, and used as the basis of
  a plan to abandon Stage 0 for a methods paper. Committed in `c86968a`, `48b1fe1`.
- **Refuted by:** fitting the order term the model was missing. `β = +1.367` on
  Gemma, which makes the correct null for order-reversal consistency at `x = 0`
  equal to `2s(1−s) = 0.324`, **not 0.5**. Observed consistency tracks a
  content-plus-position model to within ±0.05.
- **Cause:** the estimator could not distinguish "no content signal" from "signal
  plus unmodelled additive bias." It was the second. Compounding it, the pipeline
  stratified on gaps computed from the uncompressed θ, so the x-axis was wrong too.
- **Consequences:** Stage 0's hypothesis is live again. The strategic plan built on
  this claim is void. The diagnostic is redefined as discriminability (A2.6).
- Recorded in preregistration.md A2.3 W1.

## R2 — "t4's bracket rendering collapses Pass C readout mass to a median 0.38"

- **Reported:** in the first draft of `docs/pass_c_assumption_audit.md`, claiming it
  would invalidate most of the `chose` condition.
- **Refuted by:** measuring it directly on a Pass C-style choice prompt: **0.8365**,
  degraded relative to 0.995–1.000 for the other templates but above the 0.5 floor.
- **Cause:** the 0.38 figure was measured in the anchor-comparison context and
  transferred to Pass C without checking that it applied.
- **Consequences:** a T2 tidy-up rather than a threat to the condition. The mechanism
  is real; the magnitude was not.
- Corrected in place in the audit, with the correction stated rather than the number
  quietly changed.

## R1 — the first stratified operating-window run

- **Reported:** consistency of 0.07–0.32 at every gap, verdict "window closed,
  instrument failing."
- **Refuted by:** internal incoherence — the anchor run had shown 0.98 consistency at
  the widest gap. Investigation found **39% of readouts below the mass floor**.
- **Cause:** D3 switched option labels from letters to digits, but templates t2 and
  t3 still read "the single **letter** {label_a} or {label_b}", rendering as "the
  single letter 1 or 2". Models answered in prose.
- **Consequences:** that run's numbers discarded. It also exposed a worse bug —
  stimulus file *contents* were not in any config hash, so fixing the templates
  invalidated no cache and artifacts would have been silently reused against
  different prompts. Fixed in `806579a`.

---

## Pattern

R1 and R2 were transcription or transfer errors. R3 and R4 were estimator errors —
a statistic computed correctly but measuring something other than the intended
quantity, in both cases because a nuisance component was left unmodelled (`β`) or
misattributed (between-cell variance read as dependence).

The transcription class is addressed by generated status reporting. The estimator
class is addressed by preregistered fit-quality diagnostics: the excess-consistency
slope (A2.1) and the model-versus-empirical reliability gap (A2.2) both exist because
of R3 and R4, and both are currently flagging that the order model remains
misspecified.
