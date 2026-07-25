"""Preflight for the run machine. Fails BEFORE anything expensive starts.

Checks, in order of how annoying they are to discover late:

  1. CUDA present, bf16 supported, enough VRAM for the largest model
  2. every pinned revision resolves (Llama-3.2 is gated -- needs an accepted
     licence and a token)
  3. every tokenizer supports the 1-9 one-position digit readout, with no
     cross-digit collisions, and the option-label readout works
  4. disk space for the weights
  5. git state, since assert_reportable() rejects a dirty tree or no commit

Tokenizers only by default -- no weights are downloaded, so this takes seconds.
Pass --download to pre-fetch weights afterwards.

    python scripts/preflight.py
    python scripts/preflight.py --download
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.config import load_config  # noqa: E402
from src.provenance import git_state, resolve_device, resolve_revision  # noqa: E402

CONFIGS = [
    "configs/stage0_qwen2.5-0.5b.yaml",
    "configs/stage0_qwen2.5-1.5b.yaml",
    "configs/stage0_qwen2.5-3b.yaml",
    "configs/stage0_gemma-2-2b.yaml",
    "configs/stage0_llama-3.2-3b.yaml",
]

# Approximate bf16 weight sizes, for the disk and VRAM checks.
APPROX_GB = {
    "qwen2.5-0.5b-instruct": 1.0,
    "qwen2.5-1.5b-instruct": 3.1,
    "qwen2.5-3b-instruct": 6.2,
    "gemma-2-2b-it": 5.2,
    "llama-3.2-3b-instruct": 6.4,
}

OK, WARN, FAIL = "  ok  ", " warn ", " FAIL "


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="pre-fetch model weights")
    args = ap.parse_args()

    problems: list[str] = []
    warnings: list[str] = []

    def check(status: str, msg: str) -> None:
        print(f"[{status}] {msg}")
        if status == FAIL:
            problems.append(msg)
        elif status == WARN:
            warnings.append(msg)

    # ---- 1. device ------------------------------------------------------
    print("\n== device ==")
    import torch

    device = resolve_device("auto")
    check(OK if device == "cuda" else FAIL,
          f"device resolves to {device!r}" +
          ("" if device == "cuda" else " -- reported numbers require CUDA"))

    if device == "cuda":
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        cap = torch.cuda.get_device_capability(0)
        check(OK, f"{name}, {total:.1f} GB, compute capability {cap[0]}.{cap[1]}")
        check(OK if torch.cuda.is_bf16_supported() else FAIL,
              "bf16 supported (hard requirement -- there is no quantized path)")
        need = max(APPROX_GB.values()) * 1.35  # weights + activations + fragmentation
        check(OK if total >= need else WARN,
              f"VRAM headroom: largest model needs ~{need:.1f} GB of {total:.1f} GB")
    check(OK, f"torch {torch.__version__}")

    # ---- 2 & 3. models, revisions, readout -------------------------------
    print("\n== models, revisions, readout ==")
    from src.readout.digits import DigitMapError, build_digit_map, build_label_map
    from transformers import AutoTokenizer

    total_gb = 0.0
    for path in CONFIGS:
        cfg = load_config(REPO / path)
        label = cfg.model.name
        total_gb += APPROX_GB.get(label, 6.0)

        sha, pinned = resolve_revision(cfg.model.hf_id, cfg.model.revision)
        if not pinned:
            check(FAIL, f"{label}: revision {cfg.model.revision!r} did not resolve "
                        "(gated repo needs `huggingface-cli login` and an accepted licence)")
            continue
        short = sha[:12]
        matches = sha == cfg.model.revision
        check(OK if matches else FAIL,
              f"{label}: revision {short} " +
              ("matches the pin" if matches else f"DIFFERS from pinned {cfg.model.revision[:12]}"))

        try:
            tok = AutoTokenizer.from_pretrained(cfg.model.hf_id, revision=cfg.model.revision)
        except Exception as exc:  # noqa: BLE001
            check(FAIL, f"{label}: tokenizer failed to load -- {type(exc).__name__}: {exc}")
            continue

        try:
            dmap = build_digit_map(tok, cfg.readout.digits)
            lmap = build_label_map(tok, cfg.readout.option_labels)
        except DigitMapError as exc:
            check(FAIL, f"{label}: readout unavailable -- {exc}")
            continue

        multi = [d for d in cfg.readout.digits if len(dmap.ids_by_digit[d]) > 1]
        check(OK, f"{label}: digit map 1-9 complete, no collisions"
                  + (f"; {len(multi)} digit(s) have >1 surface form" if multi else ""))
        check(OK, f"{label}: option labels {cfg.readout.option_labels} single-token "
                  f"({lmap.n_ids} ids)")

        ten = tok.encode("10", add_special_tokens=False)
        if len(ten) == 1:
            check(OK, f"{label}: note -- '10' IS a single token here, but the scale is 1-9 "
                      "so the readout stays uniform across the ladder")

    # ---- 4. disk --------------------------------------------------------
    print("\n== disk ==")
    from huggingface_hub import constants

    cache = Path(constants.HF_HUB_CACHE)
    cache.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(cache).free / 1e9
    check(OK if free > total_gb * 1.2 else WARN,
          f"{free:.1f} GB free at {cache}; weights need ~{total_gb:.1f} GB")

    # ---- 5. git ---------------------------------------------------------
    print("\n== git ==")
    sha, dirty = git_state()
    check(FAIL if sha == "no-commit" else OK, f"HEAD = {sha[:12]}")
    check(WARN if dirty else OK,
          "working tree is dirty -- assert_reportable() will reject these artifacts"
          if dirty else "working tree clean")

    # ---- optional download ----------------------------------------------
    if args.download and not problems:
        print("\n== downloading weights ==")
        from huggingface_hub import snapshot_download

        for path in CONFIGS:
            cfg = load_config(REPO / path)
            print(f"  {cfg.model.name} ...", flush=True)
            snapshot_download(cfg.model.hf_id, revision=cfg.model.revision)
        print("  done")

    # ---- verdict --------------------------------------------------------
    print("\n" + "=" * 68)
    if problems:
        print(f"PREFLIGHT FAILED -- {len(problems)} blocking problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("PREFLIGHT PASSED" + (f" with {len(warnings)} warning(s)" if warnings else ""))
    for w in warnings:
        print(f"  - {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
