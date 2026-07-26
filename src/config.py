"""Typed run configuration.

One YAML per run. The config hash goes into every output filename, so a changed
parameter can never silently overwrite or be confused with results produced
under a different parameter.

The hash covers everything that can change the numbers, including `batch_size`
(bf16 reductions are batch-order dependent) and `smoke`. It excludes only
`artifacts_dir`, which is a filesystem location rather than a scientific
parameter.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

REPO_ROOT = Path(__file__).resolve().parent.parent
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

CONDITIONS: tuple[str, ...] = ("chose", "yoked", "3p-yoked", "3p-random", "random")
POLARITIES: tuple[str, ...] = ("ascending", "descending")

# Which config fields can change each stage's output. A stage's cache is keyed on
# exactly these, so tuning a gate threshold or a prior re-runs the analysis without
# invalidating tens of thousands of cached forward passes. `analysis` is absent,
# which means the full config -- everything reaches it.
_PASS_A_FIELDS = (
    "model",
    "readout",
    "stimuli",
    "seed",
    "batch_size",
    "smoke",
    "smoke_items_per_domain",
)
_PASS_B_FIELDS = _PASS_A_FIELDS + ("pass_a", "pass_b", "smoke_pairs_per_level")
_PASS_C_FIELDS = _PASS_B_FIELDS + ("pass_c",)

STAGE_HASH_FIELDS: dict[str, tuple[str, ...]] = {
    "pass_a": _PASS_A_FIELDS,
    "pass_b": _PASS_B_FIELDS,
    "pass_c": _PASS_C_FIELDS,
}


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ModelConfig(Frozen):
    """Which model, pinned how.

    `revision` should be a 40-hex commit SHA for any run whose numbers will be
    reported. "main" is permitted for scaffolding but `provenance.assert_reportable`
    rejects it, so an unpinned run cannot reach the paper by accident.
    """

    name: str
    hf_id: str
    revision: str = "main"
    dtype: Literal["bfloat16"] = "bfloat16"
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    # Changes attention numerics, so it is part of the config hash. Gemma-2 uses
    # logit soft-capping, which the sdpa path does not implement; "eager" matches
    # the reference implementation for that family.
    attn_implementation: Literal["eager", "sdpa"] = "sdpa"

    @property
    def revision_is_pinned(self) -> bool:
        return bool(_SHA_RE.match(self.revision))


class ReadoutConfig(Frozen):
    """Rating and choice readout.

    scale_max is 9, not 10: on Qwen2.5 and Gemma-2 the first token of "10" is
    byte-identical to the token for "1", so a 1-10 scale is unreadable at one
    token position. See preregistration.md section 3.2.
    """

    scale_min: Literal[1] = 1
    scale_max: Literal[9] = 9
    option_labels: tuple[str, str] = ("A", "B")

    @property
    def digits(self) -> tuple[int, ...]:
        return tuple(range(self.scale_min, self.scale_max + 1))

    @property
    def reversal_constant(self) -> int:
        """Descending scores are reversed as `reversal_constant - x`.

        For a 1-9 scale this is 10. Hardcoding 11 (the 1-10 constant) would add
        +1 to every descending score and bias every polarity-collapsed mean by
        +0.5.
        """
        return self.scale_min + self.scale_max


class StimuliConfig(Frozen):
    items_path: Path = Path("src/stimuli/items.yaml")
    templates_path: Path = Path("src/stimuli/templates.yaml")
    domains: tuple[str, ...] = ("destinations", "electronics", "foods", "activities")
    items_per_domain: int = 100
    n_templates: int = 5


class PassAConfig(Frozen):
    """Template split: selection vs analysis measurement.

    Selecting difficulty on a noisy |diff| and then analysing that same noisy
    |diff| makes the regressor regression-contaminated. Disjoint template sets
    give an uncontaminated measurement of the same quantity. 3/2 favours
    selection because the DV's pre-rating is measured within-trial in Pass C.
    """

    selection_templates: tuple[int, ...] = (0, 1, 2)
    analysis_templates: tuple[int, ...] = (3, 4)

    @model_validator(mode="after")
    def _disjoint(self) -> "PassAConfig":
        overlap = set(self.selection_templates) & set(self.analysis_templates)
        if overlap:
            raise ValueError(
                f"selection and analysis template sets must be disjoint, shared: {sorted(overlap)}"
            )
        if not self.selection_templates or not self.analysis_templates:
            raise ValueError("both template sets must be non-empty")
        return self


class GateConfig(Frozen):
    """Exclusion criteria. See preregistration.md section 5."""

    # Polarity validity: the sole categorical exclusion.
    validity_rho_min: float = 0.6
    validity_min_surviving_templates: int = 3
    # Dynamic range: no between-item variance means |diff| has no range.
    sigma_between_min: float = 0.5
    # Reliability: a tripwire for implementation faults, NOT a scientific halt.
    icc_tripwire: float = 0.4


class PassBConfig(Frozen):
    n_difficult: int = 100
    n_easy: int = 100
    difficult_quantile: float = 0.10  # bottom decile of |diff|
    easy_quantile: float = 0.75  # top quartile of |diff|
    max_uses_per_item: int = 2
    within_domain: bool = True
    # Difficult and easy pools are matched on mean pair rating; matching removes
    # the extremity/ceiling confound rather than modelling it.
    match_on_mean_rating: bool = True
    match_tolerance: float = 0.15  # rating points, per matched stratum

    @model_validator(mode="after")
    def _balanced(self) -> "PassBConfig":
        if not 0.0 < self.difficult_quantile < self.easy_quantile < 1.0:
            raise ValueError("require 0 < difficult_quantile < easy_quantile < 1")
        return self


class PassCConfig(Frozen):
    conditions: tuple[str, ...] = CONDITIONS
    n_option_orders: Literal[2] = 2

    @field_validator("conditions")
    @classmethod
    def _known(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        unknown = set(v) - set(CONDITIONS)
        if unknown:
            raise ValueError(f"unknown conditions: {sorted(unknown)}")
        for required in ("chose", "yoked"):
            if required not in v:
                raise ValueError(f"'{required}' is required for the primary contrast")
        return v


class AnalysisConfig(Frozen):
    """Model, SESOI, and the equivalence rule for later stages."""

    reference_condition: str = "yoked"
    # SESOI on the primary interaction, primary form: fraction of sigma_between.
    sesoi_sigma_fraction: float = 0.15
    # Secondary scale-free anchor, in raw rating points.
    sesoi_raw_secondary: float = 0.25
    # Stage 2 equivalence bound, as a fraction of the comparison vector's effect.
    equivalence_f: float = 0.25
    chains: int = 4
    tune: int = 2000
    draws: int = 2000
    hdi_prob: float = 0.95
    rhat_max: float = 1.01
    ess_min: int = 400
    power_target: float = 0.80
    power_n_sims: int = 500

    @field_validator("reference_condition")
    @classmethod
    def _ref_known(cls, v: str) -> str:
        if v not in CONDITIONS:
            raise ValueError(f"reference_condition must be one of {CONDITIONS}")
        return v


class RunConfig(Frozen):
    """A complete Stage 0 run."""

    model: ModelConfig
    stimuli: StimuliConfig = StimuliConfig()
    readout: ReadoutConfig = ReadoutConfig()
    pass_a: PassAConfig = PassAConfig()
    pass_b: PassBConfig = PassBConfig()
    pass_c: PassCConfig = PassCConfig()
    gates: GateConfig = GateConfig()
    analysis: AnalysisConfig = AnalysisConfig()

    seed: int = 20261025
    batch_size: int = 32
    artifacts_dir: Path = Field(default=Path("artifacts"), exclude=True)

    # Smoke mode subsets the stimuli for M1 wiring checks. It is part of the
    # hash, so smoke artifacts can never be mistaken for run artifacts.
    smoke: bool = False
    smoke_items_per_domain: int = 10
    smoke_pairs_per_level: int = 5

    @property
    def items_per_domain(self) -> int:
        return self.smoke_items_per_domain if self.smoke else self.stimuli.items_per_domain

    @property
    def n_difficult(self) -> int:
        return self.smoke_pairs_per_level if self.smoke else self.pass_b.n_difficult

    @property
    def n_easy(self) -> int:
        return self.smoke_pairs_per_level if self.smoke else self.pass_b.n_easy

    def stimuli_digest(self) -> str:
        """Digest of the actual stimulus FILE CONTENTS, not just their paths.

        Without this, editing items.yaml, templates.yaml or anchors.yaml leaves
        every hash unchanged, so cached artifacts are silently reused against
        different prompts. That is the worst class of reproducibility bug: it
        produces no error and no warning, just numbers attributed to stimuli that
        did not generate them. Discovered after a template wording fix changed no
        hash at all.
        """
        parts = []
        for rel in (
            self.stimuli.items_path,
            self.stimuli.templates_path,
            Path("src/stimuli/anchors.yaml"),
        ):
            p = self.resolve(rel)
            parts.append(p.read_bytes() if p.exists() else b"")
        return hashlib.sha256(b"\x00".join(parts)).hexdigest()[:12]

    def hash(self, stage: str | None = None) -> str:
        """Deterministic 12-hex digest.

        With no `stage`, covers every number-affecting parameter -- this is the run
        identity, recorded in the provenance of every artifact.

        With a `stage`, covers only the fields that can change THAT stage's output.
        Forward passes are expensive and gate thresholds, priors and SESOI
        fractions do not affect them, so adjusting a reported threshold must not
        invalidate 30k cached forward passes. Anything that could change the
        numbers a stage produces is still inside its scope.
        """
        payload = self.model_dump(mode="json")
        fields = STAGE_HASH_FIELDS.get(stage) if stage else None
        if fields:
            payload = {k: payload[k] for k in fields}
        # Stimulus content reaches every stage that runs a forward pass.
        payload["_stimuli_digest"] = self.stimuli_digest()
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]

    def artifact_dir(self, stage: str) -> Path:
        """artifacts/<stage>/<model>/<stage_hash>/"""
        return self.artifacts_dir / stage / self.model.name / self.hash(stage)

    def resolve(self, p: Path) -> Path:
        return p if p.is_absolute() else REPO_ROOT / p


def load_config(path: str | Path, overrides: dict | None = None) -> RunConfig:
    """Load a run YAML, merging `base.yaml` underneath it if `extends:` is set."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}

    base_name = raw.pop("extends", None)
    if base_name is not None:
        base_raw = yaml.safe_load((path.parent / base_name).read_text()) or {}
        base_raw.pop("extends", None)
        raw = _deep_merge(base_raw, raw)

    if overrides:
        raw = _deep_merge(raw, overrides)
    return RunConfig(**raw)


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out
