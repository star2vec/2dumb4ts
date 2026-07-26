"""Provenance capture and the cross-device pooling guard.

Every artifact carries the full record of how it was produced. Two guards use it:

  assert_poolable(frames)  -- raises if artifacts differ in device or dtype.
                              M1 numbers can never be silently averaged with
                              Ada numbers.
  assert_reportable(prov)  -- raises unless the artifact came off the run
                              machine at bf16 from a pinned model revision in a
                              clean tree. An unpinned or dev-machine run cannot
                              reach the paper by accident.
"""

from __future__ import annotations

import platform
import re
import subprocess
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import torch
from pydantic import BaseModel, ConfigDict

from src.config import RunConfig

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
PROV_PREFIX = "prov_"

# Guarding on these is the whole point; a mismatch is an error, not a warning.
POOL_CRITICAL_FIELDS = ("device", "dtype", "attn_implementation")
# A mismatch here changes kernels and numerics but not the experiment; warn.
POOL_ADVISORY_FIELDS = ("torch_version", "transformers_version", "model_revision")


class ProvenanceError(RuntimeError):
    """Raised on an attempt to pool or report artifacts that must not be."""


class Provenance(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    device: str
    device_name: str
    dtype: str
    attn_implementation: str
    torch_version: str
    transformers_version: str
    #: Analysis stack. The PyTensor backend changes how the sampler is compiled,
    #: which changes the draws (not the posterior), so it must not be an untracked
    #: variable in a project that records everything else.
    pymc_version: str
    pytensor_version: str
    pytensor_mode: str
    model_name: str
    model_hf_id: str
    model_revision: str
    model_revision_pinned: bool
    seed: int
    config_hash: str
    git_sha: str
    git_dirty: bool
    platform: str
    python_version: str
    created_utc: str

    def as_columns(self) -> dict:
        return {f"{PROV_PREFIX}{k}": v for k, v in self.model_dump().items()}


# ---------------------------------------------------------------------------
# device


def resolve_device(requested: str = "auto") -> str:
    """Pick a device. Device-agnostic code, explicit record of what was used."""
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_name(device: str) -> str:
    if device == "cuda":
        return torch.cuda.get_device_name(0)
    if device == "mps":
        return f"apple-{platform.machine()}"
    return platform.processor() or platform.machine()


# ---------------------------------------------------------------------------
# model revision


def resolve_revision(hf_id: str, revision: str) -> tuple[str, bool]:
    """Resolve a branch name to a commit SHA. Returns (sha_or_original, pinned).

    Tries the local hub cache first so this works offline on the run machine.
    """
    if _SHA_RE.match(revision):
        return revision, True

    try:
        from huggingface_hub import constants

        cache = Path(constants.HF_HUB_CACHE)
        ref = cache / f"models--{hf_id.replace('/', '--')}" / "refs" / revision
        if ref.is_file():
            sha = ref.read_text().strip()
            if _SHA_RE.match(sha):
                return sha, True
    except Exception:  # noqa: BLE001 - cache layout is best-effort
        pass

    try:
        from huggingface_hub import HfApi

        sha = HfApi().model_info(hf_id, revision=revision).sha
        if sha and _SHA_RE.match(sha):
            return sha, True
    except Exception:  # noqa: BLE001 - offline is a normal state here
        pass

    return revision, False


# ---------------------------------------------------------------------------
# git


def git_state() -> tuple[str, bool]:
    def run(args: Sequence[str]) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=Path(__file__).resolve().parent.parent,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    sha = run(["rev-parse", "HEAD"]) or "no-commit"
    status = run(["status", "--porcelain"])
    dirty = bool(status) if status is not None else True
    return sha, dirty


# ---------------------------------------------------------------------------
# capture


def _analysis_stack() -> dict:
    """PyMC/PyTensor versions and the active compilation backend.

    `pytensor_mode` reflects PYTENSOR_FLAGS, which is set at the environment level
    rather than in code -- the exact kind of variable that goes unrecorded and then
    cannot be reconstructed when someone asks how a posterior was produced.
    """
    import os

    out = {"pymc_version": "absent", "pytensor_version": "absent"}
    try:
        import pymc

        out["pymc_version"] = pymc.__version__
    except Exception:  # noqa: BLE001
        pass
    try:
        import pytensor

        out["pytensor_version"] = pytensor.__version__
        mode = pytensor.config.mode
    except Exception:  # noqa: BLE001
        mode = "unknown"
    flags = os.environ.get("PYTENSOR_FLAGS", "")
    out["pytensor_mode"] = f"{mode}|PYTENSOR_FLAGS={flags}" if flags else str(mode)
    return out


def capture(cfg: RunConfig, device: str | None = None) -> Provenance:
    import transformers

    device = device or resolve_device(cfg.model.device)
    revision, pinned = resolve_revision(cfg.model.hf_id, cfg.model.revision)
    sha, dirty = git_state()

    return Provenance(
        device=device,
        device_name=device_name(device),
        dtype=cfg.model.dtype,
        attn_implementation=cfg.model.attn_implementation,
        torch_version=torch.__version__,
        transformers_version=transformers.__version__,
        **_analysis_stack(),
        model_name=cfg.model.name,
        model_hf_id=cfg.model.hf_id,
        model_revision=revision,
        model_revision_pinned=pinned,
        seed=cfg.seed,
        config_hash=cfg.hash(),
        git_sha=sha,
        git_dirty=dirty,
        platform=platform.platform(),
        python_version=platform.python_version(),
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


# ---------------------------------------------------------------------------
# guards


def assert_reportable(prov: Provenance) -> None:
    """Reported numbers must originate on the run machine, bf16, pinned, clean."""
    problems = []
    if prov.device != "cuda":
        problems.append(f"device is {prov.device!r}, reported numbers require 'cuda'")
    if prov.dtype != "bfloat16":
        problems.append(f"dtype is {prov.dtype!r}, reported numbers require 'bfloat16'")
    if not prov.model_revision_pinned:
        problems.append(
            f"model revision {prov.model_revision!r} is not a pinned commit SHA"
        )
    if prov.git_dirty:
        problems.append("working tree is dirty")
    if prov.git_sha == "no-commit":
        problems.append("no git commit to attribute the run to")
    if problems:
        raise ProvenanceError(
            "artifact is not reportable:\n  - " + "\n  - ".join(problems)
        )


def provenance_of(frame: pd.DataFrame) -> dict:
    """Extract the (necessarily constant) provenance block from an artifact."""
    cols = [c for c in frame.columns if c.startswith(PROV_PREFIX)]
    if not cols:
        raise ProvenanceError("frame carries no provenance columns")
    out = {}
    for c in cols:
        vals = frame[c].dropna().unique()
        if len(vals) > 1:
            raise ProvenanceError(
                f"provenance column {c!r} is not constant within one artifact: {vals[:5]}"
            )
        out[c[len(PROV_PREFIX) :]] = vals[0] if len(vals) else None
    return out


def assert_poolable(frames: Iterable[pd.DataFrame], *, context: str = "") -> None:
    """Raise if these artifacts must not be combined.

    Device and dtype mismatches are hard errors: they are exactly the mixing this
    project cannot tolerate, since equivalence claims are computed against
    variance that quantization and backend differences inflate.
    """
    provs = [provenance_of(f) for f in frames]
    if len(provs) < 2:
        return

    where = f" ({context})" if context else ""
    for field in POOL_CRITICAL_FIELDS:
        seen = {p.get(field) for p in provs}
        if len(seen) > 1:
            raise ProvenanceError(
                f"refusing to pool artifacts across {field}{where}: {sorted(map(str, seen))}. "
                "Artifacts produced on different devices or dtypes are not comparable; "
                "re-run the odd one out on the run machine."
            )

    for field in POOL_ADVISORY_FIELDS:
        seen = {p.get(field) for p in provs}
        if len(seen) > 1:
            warnings.warn(
                f"pooling artifacts that differ in {field}{where}: {sorted(map(str, seen))}",
                stacklevel=2,
            )


# ---------------------------------------------------------------------------
# artifact io


def write_parquet(frame: pd.DataFrame, path: Path, prov: Provenance) -> pd.DataFrame:
    """Write a tabular artifact with its provenance block attached as columns.

    Returns the STAMPED frame, not the input. Returning the unstamped frame would
    mean a freshly computed stage behaves differently from a cached one -- the
    cached path reads provenance back off disk, so the pooling guard would pass on
    a reload and fail on a fresh run.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame.copy()
    for k, v in prov.as_columns().items():
        out[k] = v
    out.to_parquet(path, index=False)
    return out


def read_parquet(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    provenance_of(frame)  # fails loudly on an artifact with no provenance
    return frame


def payload_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if not c.startswith(PROV_PREFIX)]
