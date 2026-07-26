# Related work: what is pre-empted, what survives

Written 2026-07-26 to test three claims about whether our instrument findings are novel.
**One is confirmed, two are not supported by what I could retrieve.** Every statement below
was checked against the paper's own text, not against an abstract or a search summary.

Verification limits, stated up front: arXiv HTML full text was fetched for 2505.15240,
2412.05579 and 2607.02104. For 2412.05579 the retrieval reached §7.1.1 and the citation
list but I cannot guarantee it covered every section, so the negative finding there is
"not found" rather than "absent". 2606.13221 was checked at abstract level plus targeted
query.

---

## Claim 1 — "the order/home-advantage parameter is already in the comparative LLM-as-judge
literature." CONFIRMED. Our β is not novel.

**Fathullah & Gales, "Generalised Probabilistic Modelling and Improved Uncertainty
Estimation in Comparative LLM-as-a-judge", arXiv:2505.15240 (21 May 2025).**

§3.3 introduces exactly our parameter. It states the problem as "the LLM-based judge can
assign conflicting probabilities when comparing i to j as opposed to j to i", and models it
by modifying the comparison function to

    f(s_i − s_j; Δ) = f(s_i − s_j − Δ)

a "home advantage" Δ estimated by maximum likelihood. It cites **Caron & Doucet (2012),
"Efficient Bayesian inference for generalized Bradley–Terry models"** for the home-advantage
construction, and Zheng et al. (2023) and Liusie et al. (2024a) for positional bias in LLM
judges.

It also frames the two remedies as alternatives: **permutation debiasing** (two calls, one
per ordering, average the result) versus **home-advantage modelling** (estimate Δ).

**Verdict.** "Add an order term to Bradley-Terry for LLM pairwise judging" is prior work,
in the LLM-judge literature specifically and not merely in classical BT. Our A2.1 is a
*correction to our own model*, not a contribution. Any framing that presents it as novel
would be wrong, and a reviewer who knows this literature would catch it immediately.

Corroborating, independently: **Xu, Zeng, Paisley & Zhao, "Ask the Right Comparison:
Bias-Aware Bayesian Active Top-k Ranking with LLM Judges", arXiv:2607.02104 (2 July 2026)**
puts a position term κ directly in the BT likelihood:
`Pr(y=1|a,b) = σ(θ_a − θ_b + Σ c_m(x_a,m − x_b,m) + κ)`. Two independent groups, one within
the last month. This is settled.

---

## Claim 2 — "gap-dependence of swap consistency is stated in the LLM-judge survey."
NOT SUPPORTED by what I could retrieve.

**Li, Dong, Chen, Su, Zhou, Ai, Ye & Liu, "LLMs-as-Judges: A Comprehensive Survey on
LLM-based Evaluation Methods", arXiv:2412.05579 (7 Dec 2024, rev. 10 Dec 2024).**

The survey does catalogue position bias, under §7.1.1 "Presentation-Related Bias", with a
long citation list reaching back to Blunch (1984) and Raghubir & Valenzuela (2006). What I
could **not** find is any statement that bias magnitude or swap consistency **depends on the
quality gap between the two candidates**. The retrieved text catalogues the phenomenon
without theorising about when it intensifies.

Checked the two modelling papers as well: 2505.15240's Δ and 2607.02104's κ are both
**constant across comparisons**. 2607.02104 explicitly "treats bias coefficients as fixed
parameters across all comparisons, without conditioning them on the quality gap between
items being compared."

**Verdict.** I think you are wrong on this one, and it matters. Nobody I can find models the
gap-dependence, and the two papers that model position bias at all assume it is constant. Our
tier decomposition (invariance 0.290 at high-tier anchors rising to 0.744 at low-tier) and
the β-by-tier spread of 1.667 [1.300, 2.352] are measurements of the thing those models
assume away. That said — see the caveat below on whether it is a *finding* or a *tautology*.

---

## Claim 3 — "practice papers still assert swap-and-average is sufficient." NOT SUPPORTED
as characterised.

**arXiv:2606.13221 is Kargi & Salinas, "From Uncertain Judgments to Calibrated Rankings:
Conformal Elo Estimation for LLM Evaluation" (11 June 2026, rev. 12 June).** It uses pairwise
comparisons and propagates calibrated win probabilities into a Bradley-Terry procedure. It
names position bias among the problems it addresses — and it does **not** claim
swap-and-average is adequate. It proposes conformal uncertainty quantification instead.

So the specific target is wrong. The broader point is *partly* available: 2505.15240 presents
permutation debiasing as a live alternative to home-advantage modelling, and — checked
directly — it does **not** critique permutation debiasing as insufficient, does not note that
averaging probabilities across orders compresses the latent scale, and provides **no
empirical head-to-head** between the two remedies. It reports permutation debiasing
favourably, as reducing the comparisons needed from ~60–80 to ~20–25.

**Verdict.** There is a real gap, but it is narrower and differently shaped than claimed. The
unoccupied ground is not "people think swapping is enough" — it is that **nobody has shown
what swapping-and-averaging costs you**. Our Jensen result is the missing piece: averaging
*probabilities* across orders is not the same as averaging *logits*, the MLE lands on the
average of the two probabilities, and the induced compression has local slope minimal at
x = 0, saturating at −ln cosh β. Measured: slope on true θ 0.746 without the term against
0.907 with it.

---

## What actually survives, ranked

1. **Permutation debiasing is not free, and its cost is worst where it matters most.** The
   Jensen argument plus the measured compression. Not found in any of the four papers
   checked. Directly relevant to anyone who randomises presentation order and treats the
   design as balanced — which includes Arena-style pipelines and 2505.15240's own
   recommended path. This is the strongest surviving item.
2. **Position bias is signed and model-specific.** Gemma-2-2b favours slot 1 (β = +1.367);
   Qwen2.5-3B favours slot 2. Both papers that model it use a single constant, and a constant
   cannot be both. Modest but concrete.
3. **β varies ~3× across prompt paraphrases within one model** (+0.550 to +1.645 on clean
   templates; +4.312 on a template that was separately broken). Both modelling papers assume
   a constant. This is the sharpest challenge to existing practice.
4. **Universal scale-polarity blindness.** ρ = −0.944 / −0.964 / −0.916 at full sample across
   two model families. Unrelated to the BT literature; belongs to the instrument-validation
   section of the Stage 0 paper.
5. **Gap-dependence of β.** Live per Claim 2, but see the caveat.

## Caveat on item 5, which I raise against our own interest

Order-reversal consistency at gap zero is `2s(1−s)` with `s = sigmoid(β)` **by construction**.
So "consistency degrades as items get closer" is partly definitional, and stating it as an
empirical finding risks being tautological — the objection previously raised against our
methods finding #3, which I have not fully retired. What is *not* tautological is β itself
varying with the gap, since both published models hold it constant. Before claiming item 5,
we need to show that β_tier heterogeneity survives with θ and α properly modelled, and my
current tier estimate is confounded (tier correlates with |x|, so β_tier can absorb
tier-specific misfit). **Unresolved, and it should not go in a paper until it is.**

## Consequence for scope

This supports the decision to make Stage 0 the paper. Item 1 is a genuine contribution but a
short one, and demonstrating it moves real rankings requires an Arena reanalysis that is
explicitly out of scope. Items 1–4 are a strong instrument-validation section for a paper
whose main claim is elsewhere. They are not a main-track paper on their own.

## Future work, logged and not being done

- Arena-style reanalysis showing permutation-debiased BT rankings shift under home-advantage
  modelling.
- Resolving whether β_tier heterogeneity survives proper conditioning (item 5's caveat).
- Bolstering/derogation decomposition of the spread DV.

## Sources

- [arXiv:2505.15240](https://arxiv.org/abs/2505.15240) — Fathullah & Gales, home advantage Δ
- [arXiv:2412.05579](https://arxiv.org/abs/2412.05579) — Li et al., LLMs-as-Judges survey
- [arXiv:2606.13221](https://arxiv.org/abs/2606.13221) — Kargi & Salinas, Conformal Elo
- [arXiv:2607.02104](https://arxiv.org/abs/2607.02104) — Xu et al., bias-aware active ranking
- Caron & Doucet (2012), generalized Bradley–Terry — cited by 2505.15240 for home advantage
