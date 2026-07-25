"""Per-tokenizer digit token map for the 1-9 rating scale.

Why 1-9 and not 1-10: on Qwen2.5 and Gemma-2 the string "10" is two tokens and
its FIRST token is byte-identical to the token for "1", so the probability mass
at a single position cannot be attributed between ratings 1 and 10. Reading a
second position would violate the one-position constraint and would make the
readout mechanism differ across model families. At 1-9 every rating is a single
token with no cross-digit collision on every tokenizer in the set.

Residual measurement caveat, recorded for the paper: probability the model
intended for an out-of-range value (>9) would land on that value's first digit.
The scale tops out at 9, so any such mass is an instruction violation rather than
an in-range confusion, and gross misbehaviour of this kind shows up as depressed
`digit_mass`, which is recorded per trial.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from transformers import PreTrainedTokenizerBase


class DigitMapError(RuntimeError):
    """Raised when a tokenizer cannot support the readout. Never downgraded."""


@dataclass(frozen=True)
class DigitMap:
    """Which token ids count as which digit, for one tokenizer."""

    digits: tuple[int, ...]
    ids_by_digit: dict[int, tuple[int, ...]]
    flat_ids: tuple[int, ...]
    #: digit_index[k] is the position of flat_ids[k] within `digits`
    digit_index: tuple[int, ...]
    surfaces_by_digit: dict[int, tuple[str, ...]]

    @property
    def n_ids(self) -> int:
        return len(self.flat_ids)


def build_digit_map(
    tokenizer: PreTrainedTokenizerBase, digits: tuple[int, ...]
) -> DigitMap:
    """Enumerate every single-token surface form for each digit.

    Two independent sources, unioned:
      1. direct encode of the bare and leading-space forms, kept only when the
         result is a single token;
      2. a full vocabulary scan for tokens whose decoded string is the digit
         possibly preceded by whitespace.

    Aborts on incomplete coverage or on any token id claimed by two digits.
    Silent partial coverage would be a degraded measurement masquerading as a
    clean one, so it is a fatal error rather than a warning.
    """
    ids_by_digit: dict[int, set[int]] = {d: set() for d in digits}
    surfaces: dict[int, set[str]] = {d: set() for d in digits}

    # (1) explicit candidates
    for d in digits:
        for surface in (str(d), f" {str(d)}"):
            enc = tokenizer.encode(surface, add_special_tokens=False)
            if len(enc) == 1:
                ids_by_digit[d].add(enc[0])
                surfaces[d].add(surface)

    # (2) vocabulary scan, to catch family-specific forms such as SentencePiece
    #     "▁1" that encode() would decompose.
    patterns = {d: re.compile(rf"^\s*{d}$") for d in digits}
    for token, tid in tokenizer.get_vocab().items():
        try:
            text = tokenizer.convert_tokens_to_string([token])
        except Exception:  # noqa: BLE001 - malformed byte tokens are expected
            continue
        if not text or len(text) > 4:
            continue
        for d, pat in patterns.items():
            if pat.match(text):
                ids_by_digit[d].add(tid)
                surfaces[d].add(text)
                break

    empty = [d for d in digits if not ids_by_digit[d]]
    if empty:
        raise DigitMapError(
            f"tokenizer has no single-token surface form for digit(s) {empty}; "
            "the one-position expected-value readout is not available on this model"
        )

    owner: dict[int, int] = {}
    collisions: list[str] = []
    for d in digits:
        for tid in ids_by_digit[d]:
            if tid in owner and owner[tid] != d:
                collisions.append(f"token id {tid} claimed by digits {owner[tid]} and {d}")
            owner[tid] = d
    if collisions:
        raise DigitMapError(
            "digit token collision, readout would be ambiguous:\n  - "
            + "\n  - ".join(collisions)
        )

    flat: list[int] = []
    index: list[int] = []
    for pos, d in enumerate(digits):
        for tid in sorted(ids_by_digit[d]):
            flat.append(tid)
            index.append(pos)

    return DigitMap(
        digits=digits,
        ids_by_digit={d: tuple(sorted(ids_by_digit[d])) for d in digits},
        flat_ids=tuple(flat),
        digit_index=tuple(index),
        surfaces_by_digit={d: tuple(sorted(surfaces[d])) for d in digits},
    )


def build_label_map(
    tokenizer: PreTrainedTokenizerBase, labels: tuple[str, ...]
) -> DigitMap:
    """The same construction for option labels ("A"/"B") in the choice readout.

    Reuses DigitMap: `digits` holds label positions 0..n-1 rather than rating
    values, so the choice readout shares the restricted-softmax code path.
    """
    ids_by_pos: dict[int, set[int]] = {i: set() for i in range(len(labels))}
    surfaces: dict[int, set[str]] = {i: set() for i in range(len(labels))}

    for i, label in enumerate(labels):
        for surface in (label, f" {label}"):
            enc = tokenizer.encode(surface, add_special_tokens=False)
            if len(enc) == 1:
                ids_by_pos[i].add(enc[0])
                surfaces[i].add(surface)

    empty = [labels[i] for i, v in ids_by_pos.items() if not v]
    if empty:
        raise DigitMapError(f"no single-token surface form for option label(s) {empty}")

    owner: dict[int, int] = {}
    for i, tids in ids_by_pos.items():
        for tid in tids:
            if tid in owner and owner[tid] != i:
                raise DigitMapError(
                    f"option label token collision: id {tid} claimed by "
                    f"{labels[owner[tid]]!r} and {labels[i]!r}"
                )
            owner[tid] = i

    flat: list[int] = []
    index: list[int] = []
    for i in range(len(labels)):
        for tid in sorted(ids_by_pos[i]):
            flat.append(tid)
            index.append(i)

    return DigitMap(
        digits=tuple(range(len(labels))),
        ids_by_digit={i: tuple(sorted(ids_by_pos[i])) for i in ids_by_pos},
        flat_ids=tuple(flat),
        digit_index=tuple(index),
        surfaces_by_digit={i: tuple(sorted(surfaces[i])) for i in surfaces},
    )


def describe(dmap: DigitMap) -> str:
    lines = []
    for d in dmap.digits:
        ids = dmap.ids_by_digit[d]
        surf = [repr(s) for s in dmap.surfaces_by_digit[d]]
        lines.append(f"  {d}: ids={list(ids)} surfaces={surf}")
    return "\n".join(lines)
