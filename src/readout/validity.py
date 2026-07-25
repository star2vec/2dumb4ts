"""The readout mass floor -- a global invariant on every readout in the codebase.

preregistration.md A1.6.

A readout restricted to a candidate token set is only meaningful if the model
actually puts its probability there. The first version of the content-addressed
choice probe read item-name tokens from a prompt that still showed lettered
options: the model answered with the letter, 98.7% of the mass sat on the labels,
0.8% on the tokens being read, and the probe returned clean-looking accuracy
figures that meant nothing.

So: below the floor, a trial is INVALID and logged. Never silently scored.

The floor is not a filter applied at analysis time. It travels with the trial, as
a column, so that a downstream reader cannot fail to notice it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Total probability on candidate tokens below which a readout is not a readout.
#: Well clear of a working readout (observed 0.92-1.00) and of the pathological
#: case that motivated it (0.003-0.010).
MASS_FLOOR = 0.5

VALID_COL = "readout_valid"
MASS_COL = "readout_mass"


class ReadoutInvalidError(RuntimeError):
    """Raised when an ENTIRE readout batch is below the floor.

    A handful of invalid trials is data. Every trial invalid means the prompt or
    the candidate token set is wrong, and continuing would produce a table of
    numbers that look like measurements.
    """


@dataclass(frozen=True)
class MassReport:
    n: int
    n_invalid: int
    min_mass: float
    median_mass: float
    floor: float

    @property
    def fraction_invalid(self) -> float:
        return self.n_invalid / self.n if self.n else 0.0

    def __str__(self) -> str:
        return (
            f"readout mass: median {self.median_mass:.4f}, min {self.min_mass:.4f}, "
            f"{self.n_invalid}/{self.n} below floor {self.floor}"
            f" ({self.fraction_invalid:.1%})"
        )


def check_mass(
    mass: np.ndarray,
    *,
    floor: float = MASS_FLOOR,
    context: str = "",
    abort_if_all_invalid: bool = True,
) -> tuple[np.ndarray, MassReport]:
    """Return (valid_flags, report) and shout about anything below the floor.

    Args:
        mass: total candidate-token probability per trial.
        context: what is being read, for the log line.
        abort_if_all_invalid: raise when nothing clears the floor. A wholesale
            failure is a bug in the prompt or the token map, not a data point.
    """
    mass = np.asarray(mass, dtype=float)
    valid = mass >= floor
    report = MassReport(
        n=len(mass),
        n_invalid=int((~valid).sum()),
        min_mass=float(mass.min()) if len(mass) else float("nan"),
        median_mass=float(np.median(mass)) if len(mass) else float("nan"),
        floor=floor,
    )

    if report.n_invalid:
        where = f" [{context}]" if context else ""
        print(f"  READOUT MASS{where}: {report}", flush=True)

    if abort_if_all_invalid and len(mass) and not valid.any():
        raise ReadoutInvalidError(
            f"every readout is below the mass floor {floor}{' in ' + context if context else ''} "
            f"(max observed {mass.max():.4f}). The model is not answering in the "
            "expected token space -- check the prompt and the candidate token map "
            "before trusting anything downstream."
        )
    return valid, report


def attach(frame: pd.DataFrame, mass: np.ndarray, *, floor: float = MASS_FLOOR,
           context: str = "") -> pd.DataFrame:
    """Add the mass and validity columns to a trial frame, in place-ish."""
    valid, _ = check_mass(mass, floor=floor, context=context)
    out = frame.copy()
    out[MASS_COL] = np.asarray(mass, dtype=float)
    out[VALID_COL] = valid
    return out


def apply_retrospectively(
    frame: pd.DataFrame, mass_column: str, *, floor: float = MASS_FLOOR,
    context: str = "",
) -> pd.DataFrame:
    """Apply the floor to an artifact written before the invariant existed.

    The preregistered absolute-rating Pass A records per-trial `digit_mass` but
    predates A1.6, so the floor can be applied without re-running it. The result
    lives in the analysis layer rather than in the file -- stated plainly here
    rather than left for a reader to infer.
    """
    if mass_column not in frame.columns:
        raise KeyError(
            f"{mass_column!r} not in frame; cannot apply the mass floor "
            "retrospectively to an artifact that did not record it"
        )
    return attach(frame.drop(columns=[MASS_COL, VALID_COL], errors="ignore"),
                  frame[mass_column].to_numpy(), floor=floor, context=context)


def summarize(frame: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Invalid-trial counts, reported per model and per elicitation arm (A1.6)."""
    if VALID_COL not in frame.columns:
        raise KeyError(f"{VALID_COL!r} missing; run attach() or apply_retrospectively()")
    group = by or []
    if not group:
        return pd.DataFrame([{
            "n": len(frame),
            "n_invalid": int((~frame[VALID_COL]).sum()),
            "frac_invalid": float((~frame[VALID_COL]).mean()),
            "mass_median": float(frame[MASS_COL].median()),
            "mass_min": float(frame[MASS_COL].min()),
        }])
    rows = []
    for key, block in frame.groupby(group, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        rows.append(
            {**dict(zip(group, key)),
             "n": len(block),
             "n_invalid": int((~block[VALID_COL]).sum()),
             "frac_invalid": float((~block[VALID_COL]).mean()),
             "mass_median": float(block[MASS_COL].median()),
             "mass_min": float(block[MASS_COL].min())}
        )
    return pd.DataFrame(rows)
