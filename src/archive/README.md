# Archive — retired, not deleted

Code that no longer runs and must not be quoted. Kept because each module produced numbers
that appear in the project's written record, and a reader tracing those numbers should find
the code that made them rather than a gap.

**Nothing here is imported by `src/`, invoked by any script, or covered by any test.**
`tests/test_archive_is_inert.py` enforces the first of those.

| module | why it is here |
|---|---|
| `template_dependence.py` | Computes the template ICC behind **R4** — "template responses are non-independent (ICC 0.529, design effect 3.12)". `RETRACTIONS.md` R4 withdraws that: the ICC treated templates as raters over cells, conflating genuine between-cell variation in `p` (already captured by `θ − α`) with excess dependence. The proper over-dispersion test gives **1.070, 95% CI [0.907, 1.258]** — consistent with independence. No design effect exists. |
| `operating_window.py` | Implements the **A1.7** operating-window diagnostic, which asked whether order-reversal consistency exceeded 0.5. **A2.6 redefined it.** The old form was wrong twice: the null under position bias is `2s(1−s)`, not 0.5 (A2.3 W1); and once `β` is modelled, consistency carries no information about content signal beyond what `θ` already encodes, so testing it is vacuous by construction. The live replacement is `pass_a_pairwise.operating_window` — a *function*, unrelated to this module despite the shared name. |
| `probe_choice.py` | A side experiment on choice-readout schemes, off the Stage 0 run path. Its artifacts under `artifacts/probe_choice/` are retained. |

## Known defects, deliberately unfixed

mypy reports `float | None` reaching arithmetic in `template_dependence.py:123,129` and a
`None` iterated in `probe_choice.py:270`. They are left as they were when the code was
retired. Fixing dead code would misrepresent it as maintained, and the numbers it produced
were computed by *this* version.

## If you are tempted to use something here

Don't. Take the replacement:

- template dependence → the over-dispersion test in `bradley_terry.excess_slope_ppc_null`
- operating window → `pass_a_pairwise.operating_window` (A2.6's discriminability form)
- choice readout → `readout/choice.py`
