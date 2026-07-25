"""Expected-value rating readout: one forward pass, one token position.

    rating = sum_{d=1..9} d * p(d),   p = softmax over the digit token ids only

This is an expected value over the logit distribution, never an argmax. Argmax
would discard exactly the graded information the difference-of-differences DV
depends on: a shift from p(6)=0.51 to p(6)=0.95 is a real change in the model's
rating and argmax reports it as zero.

Softmax is computed in float32 even though the model runs in bf16. bf16 has ~3
decimal digits of mantissa; taking the softmax in it would inject noise of the
same order as the effects this project measures, and later stages make
equivalence claims against that variance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.readout.digits import DigitMap


@dataclass(frozen=True)
class Readout:
    """One batch of readouts at one position."""

    #: expected value over the digit distribution, shape [n]
    value: np.ndarray
    #: total probability (over the FULL vocabulary) landing on any digit token
    mass: np.ndarray
    #: renormalized per-digit probabilities, shape [n, n_digits]
    probs: np.ndarray
    #: the modal digit, recorded as a diagnostic only -- never the DV
    argmax: np.ndarray

    def __len__(self) -> int:
        return len(self.value)


def read_expected_value(logits: torch.Tensor, dmap: DigitMap) -> Readout:
    """Read expected values from last-position logits.

    Args:
        logits: [n, vocab] logits at the single readout position.
        dmap: digit token map for this tokenizer.

    `mass` is the diagnostic that catches a prompt whose readout position is not
    actually the digit position -- for example a tokenizer that emits a separate
    whitespace token first. It is reported, not used to filter trials.
    """
    if logits.ndim != 2:
        raise ValueError(f"expected [n, vocab] logits, got shape {tuple(logits.shape)}")

    logits32 = logits.detach().to(torch.float32)
    log_probs = torch.log_softmax(logits32, dim=-1)

    ids = torch.as_tensor(dmap.flat_ids, dtype=torch.long, device=logits.device)
    sel = log_probs.index_select(1, ids).exp()  # [n, n_ids], full-vocab probabilities

    n_digits = len(dmap.digits)
    index = torch.as_tensor(dmap.digit_index, dtype=torch.long, device=logits.device)
    binned = torch.zeros(sel.shape[0], n_digits, dtype=sel.dtype, device=sel.device)
    binned.index_add_(1, index, sel)  # sum surface-form variants into one bin per digit

    mass = binned.sum(dim=-1)
    if torch.any(mass <= 0):
        raise ValueError("zero probability mass on every digit token; readout is invalid")
    probs = binned / mass.unsqueeze(-1)

    values = torch.as_tensor(dmap.digits, dtype=probs.dtype, device=probs.device)
    value = (probs * values).sum(dim=-1)
    argmax = values[probs.argmax(dim=-1)]

    return Readout(
        value=value.cpu().numpy(),
        mass=mass.cpu().numpy(),
        probs=probs.cpu().numpy(),
        argmax=argmax.cpu().numpy(),
    )


def reverse_polarity(scores: np.ndarray, reversal_constant: int) -> np.ndarray:
    """Map descending-scale scores onto the ascending scale.

    `reversal_constant` is scale_min + scale_max, i.e. 10 for a 1-9 scale. Using
    11 (the 1-10 constant) would add +1 to every descending score and bias every
    polarity-collapsed mean by +0.5.
    """
    return reversal_constant - np.asarray(scores, dtype=float)
