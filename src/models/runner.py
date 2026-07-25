"""Model loading and the single-position batched forward pass.

Device-agnostic: the same code runs on CUDA (run machine), MPS (dev machine) and
CPU. Which one was used is recorded in every artifact's provenance block, and the
analysis layer refuses to pool across devices.

bf16 only. No 4-bit or 8-bit path exists here, deliberately: later stages make
equivalence claims, and quantization noise inflates exactly the variance those
claims are computed against.

On prefix reuse: identical prompts are memoized, which is where the real saving
is -- pre-ratings are computed once per (pair, template, option order) and joined
to all five conditions, as preregistration.md 6.3 specifies. KV-block reuse
across *different* prompts sharing a prefix is deliberately NOT implemented:
combined with left padding it is easy to get subtly wrong, the total budget is
~30k short forward passes per model, and a silent cache bug would corrupt the
numbers rather than slow them down.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from src.config import RunConfig
from src.readout.choice import ChoiceReadout, read_choice
from src.readout.digits import DigitMap, build_digit_map, build_label_map, describe
from src.readout.expected_value import Readout, read_expected_value

DTYPES = {"bfloat16": torch.bfloat16}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # TF32 would silently change fp32 reductions between machines.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


@dataclass
class Runner:
    """Holds the model, tokenizer and readout maps for one run."""

    model: torch.nn.Module
    tokenizer: PreTrainedTokenizerBase
    digit_map: DigitMap
    label_map: DigitMap
    device: str
    batch_size: int
    _cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = field(
        default_factory=dict, repr=False
    )
    n_forward: int = 0
    n_cache_hits: int = 0

    @property
    def _supports_logits_to_keep(self) -> bool:
        import inspect

        try:
            return "logits_to_keep" in inspect.signature(self.model.forward).parameters
        except (TypeError, ValueError):
            return False

    # -- prompt rendering ---------------------------------------------------

    def render(self, messages: list[dict[str, str]]) -> str:
        """Apply the chat template and open the assistant turn.

        The token immediately after this prompt is the single readout position.
        """
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # -- forward ------------------------------------------------------------

    @torch.no_grad()
    def last_logits(self, prompts: list[str]) -> torch.Tensor:
        """Logits at the final position for each prompt. [n, vocab]

        Left padding puts the true final token at index -1 for every sequence.
        `position_ids` is derived from the attention mask rather than left to
        default to arange, which with left padding would offset every position.
        """
        out = []
        for start in range(0, len(prompts), self.batch_size):
            batch = prompts[start : start + self.batch_size]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                # apply_chat_template already emits BOS; adding specials again
                # would double it.
                add_special_tokens=False,
            ).to(self.device)

            mask = enc["attention_mask"]
            position_ids = mask.long().cumsum(-1) - 1
            position_ids = position_ids.masked_fill(mask == 0, 1)

            # Only the final position is ever read, so ask the model not to
            # materialise logits for the rest. On a 32x86 batch with a 152k vocab
            # this is the difference between an 836 MB logits tensor and a 10 MB
            # one -- decisive on an 8 GB card holding 6.4 GB of bf16 weights.
            kwargs = dict(
                input_ids=enc["input_ids"],
                attention_mask=mask,
                position_ids=position_ids,
            )
            if self._supports_logits_to_keep:
                kwargs["logits_to_keep"] = 1
            logits = self.model(**kwargs).logits[:, -1, :]
            out.append(logits.float().cpu())
            self.n_forward += len(batch)
        return torch.cat(out, dim=0)

    # -- readouts -----------------------------------------------------------

    def rate(self, message_lists: list[list[dict[str, str]]]) -> Readout:
        """Expected-value ratings. One forward pass, one position, per prompt."""
        prompts = [self.render(m) for m in message_lists]
        return self._cached_rate(prompts)

    def _cached_rate(self, prompts: list[str]) -> Readout:
        todo = [p for p in dict.fromkeys(prompts) if p not in self._cache]
        if todo:
            logits = self.last_logits(todo)
            r = read_expected_value(logits, self.digit_map)
            for i, p in enumerate(todo):
                self._cache[p] = (r.value[i], r.mass[i], r.probs[i], r.argmax[i])
        self.n_cache_hits += len(prompts) - len(todo)

        rows = [self._cache[p] for p in prompts]
        return Readout(
            value=np.array([x[0] for x in rows], dtype=float),
            mass=np.array([x[1] for x in rows], dtype=float),
            probs=np.stack([x[2] for x in rows]),
            argmax=np.array([x[3] for x in rows], dtype=float),
        )

    def choose(self, message_lists: list[list[dict[str, str]]]) -> ChoiceReadout:
        """Argmax over option-label tokens. A discrete commitment, not a rating."""
        prompts = [self.render(m) for m in message_lists]
        logits = self.last_logits(prompts)
        return read_choice(logits, self.label_map)


def load_runner(cfg: RunConfig, device: str | None = None) -> Runner:
    from src.provenance import resolve_device

    seed_everything(cfg.seed)
    device = device or resolve_device(cfg.model.device)
    dtype = DTYPES[cfg.model.dtype]

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.hf_id, revision=cfg.model.revision
    )
    # We read the LAST position, so padding must be on the left.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.hf_id,
        revision=cfg.model.revision,
        dtype=dtype,
        attn_implementation=cfg.model.attn_implementation,
    )
    model.to(device)
    model.eval()

    if model.dtype != dtype:
        raise RuntimeError(
            f"model loaded as {model.dtype}, expected {dtype}. bf16 is a hard "
            "constraint: quantization noise would inflate the variance that later "
            "equivalence claims are computed against."
        )

    digit_map = build_digit_map(tokenizer, cfg.readout.digits)
    label_map = build_label_map(tokenizer, cfg.readout.option_labels)

    return Runner(
        model=model,
        tokenizer=tokenizer,
        digit_map=digit_map,
        label_map=label_map,
        device=device,
        batch_size=cfg.batch_size,
    )


def describe_maps(runner: Runner) -> str:
    return (
        f"digit map ({runner.digit_map.n_ids} ids):\n{describe(runner.digit_map)}\n"
        f"label map ({runner.label_map.n_ids} ids):\n{describe(runner.label_map)}"
    )
