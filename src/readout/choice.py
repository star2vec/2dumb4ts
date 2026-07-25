"""Choice readout: argmax over option-label tokens at one position.

The choice is a discrete commitment rather than a graded measurement, so argmax
is the right reduction here -- unlike ratings, where argmax would discard the
graded information the DV needs. The full label distribution is recorded anyway,
because label-probability margin is the natural covariate for a position-bias
diagnostic.

The emitted label is written back into the `chose` condition's context as an
assistant turn. That write-back is what supplies authorship, and it is the only
place in Stage 0 where a model output re-enters a prompt.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.readout.digits import DigitMap


@dataclass(frozen=True)
class ChoiceReadout:
    #: index into the option labels, shape [n]
    index: np.ndarray
    #: renormalized probability per label, shape [n, n_labels]
    probs: np.ndarray
    #: total full-vocabulary probability landing on any label token
    mass: np.ndarray
    #: p(chosen) - p(other); near zero means the choice was near-arbitrary
    margin: np.ndarray

    def __len__(self) -> int:
        return len(self.index)


def read_choice(logits: torch.Tensor, lmap: DigitMap) -> ChoiceReadout:
    if logits.ndim != 2:
        raise ValueError(f"expected [n, vocab] logits, got shape {tuple(logits.shape)}")

    log_probs = torch.log_softmax(logits.detach().to(torch.float32), dim=-1)
    ids = torch.as_tensor(lmap.flat_ids, dtype=torch.long, device=logits.device)
    sel = log_probs.index_select(1, ids).exp()

    n_labels = len(lmap.digits)
    index = torch.as_tensor(lmap.digit_index, dtype=torch.long, device=logits.device)
    binned = torch.zeros(sel.shape[0], n_labels, dtype=sel.dtype, device=sel.device)
    binned.index_add_(1, index, sel)

    mass = binned.sum(dim=-1)
    if torch.any(mass <= 0):
        raise ValueError("zero probability mass on every option label token")
    probs = binned / mass.unsqueeze(-1)

    top = probs.argmax(dim=-1)
    sorted_probs, _ = probs.sort(dim=-1, descending=True)
    margin = sorted_probs[:, 0] - sorted_probs[:, 1]

    return ChoiceReadout(
        index=top.cpu().numpy(),
        probs=probs.cpu().numpy(),
        mass=mass.cpu().numpy(),
        margin=margin.cpu().numpy(),
    )
