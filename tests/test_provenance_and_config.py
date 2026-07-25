"""Provenance, the cross-device pooling guard, and config hashing."""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import load_config
from src.provenance import (
    PROV_PREFIX,
    Provenance,
    ProvenanceError,
    assert_poolable,
    assert_reportable,
    provenance_of,
    resolve_revision,
)

CONFIGS = [
    "configs/stage0_qwen2.5-0.5b.yaml",
    "configs/stage0_qwen2.5-1.5b.yaml",
    "configs/stage0_qwen2.5-3b.yaml",
    "configs/stage0_llama-3.2-3b.yaml",
    "configs/stage0_gemma-2-2b.yaml",
]


def _prov(**overrides) -> Provenance:
    base = dict(
        device="cuda",
        device_name="NVIDIA RTX 2000 Ada Generation",
        dtype="bfloat16",
        attn_implementation="sdpa",
        torch_version="2.8.0",
        transformers_version="4.57.0",
        model_name="qwen2.5-3b-instruct",
        model_hf_id="Qwen/Qwen2.5-3B-Instruct",
        model_revision="aa8e72537993ba99e69dfaafa59ed015b17504d1",
        model_revision_pinned=True,
        seed=20261025,
        config_hash="abc123abc123",
        git_sha="f" * 40,
        git_dirty=False,
        platform="Linux-6.8",
        python_version="3.11.9",
        created_utc="2026-07-25T12:00:00+00:00",
    )
    base.update(overrides)
    return Provenance(**base)


def _frame(prov: Provenance, n: int = 4) -> pd.DataFrame:
    frame = pd.DataFrame({"spread": [0.1] * n})
    for k, v in prov.as_columns().items():
        frame[k] = v
    return frame


# ---------------------------------------------------------------------------
# the pooling guard -- the requirement that motivated this module


def test_pooling_across_devices_raises():
    """The analysis layer must ERROR if asked to pool artifacts across devices.
    M1 numbers can never be silently averaged with run-machine numbers."""
    with pytest.raises(ProvenanceError, match="refusing to pool artifacts across device"):
        assert_poolable([_frame(_prov()), _frame(_prov(device="mps", device_name="apple-arm64"))])


def test_pooling_across_dtypes_raises():
    with pytest.raises(ProvenanceError, match="refusing to pool artifacts across dtype"):
        assert_poolable([_frame(_prov()), _frame(_prov(dtype="float16"))])


def test_pooling_across_attn_implementation_raises():
    """Attention implementation changes numerics, so it is a hard guard too."""
    with pytest.raises(ProvenanceError, match="attn_implementation"):
        assert_poolable([_frame(_prov()), _frame(_prov(attn_implementation="eager"))])


def test_pooling_identical_provenance_is_allowed():
    assert_poolable([_frame(_prov()), _frame(_prov())])


def test_pooling_warns_but_allows_a_library_version_difference():
    with pytest.warns(UserWarning, match="torch_version"):
        assert_poolable([_frame(_prov()), _frame(_prov(torch_version="2.7.1"))])


def test_frame_without_provenance_raises():
    with pytest.raises(ProvenanceError, match="no provenance columns"):
        provenance_of(pd.DataFrame({"spread": [1.0]}))


def test_non_constant_provenance_within_one_artifact_raises():
    frame = _frame(_prov())
    frame.loc[0, f"{PROV_PREFIX}device"] = "mps"
    with pytest.raises(ProvenanceError, match="not constant within one artifact"):
        provenance_of(frame)


# ---------------------------------------------------------------------------
# reportability


def test_reportable_requires_cuda_bf16_pinned_and_clean():
    assert_reportable(_prov())  # does not raise


@pytest.mark.parametrize(
    "override,expected",
    [
        ({"device": "mps"}, "require 'cuda'"),
        ({"dtype": "float16"}, "require 'bfloat16'"),
        ({"model_revision": "main", "model_revision_pinned": False}, "not a pinned commit"),
        ({"git_dirty": True}, "working tree is dirty"),
        ({"git_sha": "no-commit"}, "no git commit"),
    ],
)
def test_unreportable_artifacts_are_rejected(override, expected):
    with pytest.raises(ProvenanceError, match=expected):
        assert_reportable(_prov(**override))


def test_smoke_artifacts_are_never_reportable():
    """Dev-machine smoke output must not be able to reach the paper."""
    with pytest.raises(ProvenanceError):
        assert_reportable(_prov(device="mps", git_dirty=True))


# ---------------------------------------------------------------------------
# config


@pytest.mark.parametrize("path", CONFIGS)
def test_every_model_config_loads_and_is_pinned(path):
    cfg = load_config(path)
    assert cfg.model.dtype == "bfloat16"
    assert cfg.model.revision_is_pinned, f"{cfg.model.name} revision is not a commit SHA"
    assert len(cfg.hash()) == 12


def test_config_hash_is_stable_and_sensitive():
    cfg = load_config(CONFIGS[0])
    assert cfg.hash() == load_config(CONFIGS[0]).hash()

    # batch_size changes bf16 reduction order, so it must change the hash.
    assert cfg.model_copy(update={"batch_size": cfg.batch_size + 1}).hash() != cfg.hash()
    assert cfg.model_copy(update={"seed": cfg.seed + 1}).hash() != cfg.hash()
    assert cfg.model_copy(update={"smoke": True}).hash() != cfg.hash()


def test_artifacts_dir_is_excluded_from_the_hash():
    """A filesystem location is not a scientific parameter."""
    from pathlib import Path

    cfg = load_config(CONFIGS[0])
    assert cfg.model_copy(update={"artifacts_dir": Path("/tmp/elsewhere")}).hash() == cfg.hash()


def test_a_hash_appears_in_every_artifact_path():
    cfg = load_config(CONFIGS[0])
    for stage in ("pass_a", "pass_b", "pass_c", "analysis"):
        assert cfg.hash(stage) in str(cfg.artifact_dir(stage))
    # Analysis is keyed on the full config: everything reaches it.
    assert cfg.hash("analysis") == cfg.hash()


def test_stage_hashes_ignore_parameters_that_cannot_change_that_stage():
    """Forward passes are expensive. Tuning a gate threshold or a prior must not
    invalidate cached forward passes, but MUST change the run identity."""
    cfg = load_config(CONFIGS[0])

    gates = cfg.model_copy(
        update={"gates": cfg.gates.model_copy(update={"validity_rho_min": 0.5})}
    )
    priors = cfg.model_copy(
        update={"analysis": cfg.analysis.model_copy(update={"draws": 1234})}
    )
    for changed in (gates, priors):
        for stage in ("pass_a", "pass_b", "pass_c"):
            assert changed.hash(stage) == cfg.hash(stage)
        assert changed.hash() != cfg.hash()


def test_stage_hashes_do_track_parameters_that_change_that_stage():
    cfg = load_config(CONFIGS[0])

    # batch_size changes bf16 reduction order, so it reaches the forward passes.
    assert cfg.model_copy(update={"batch_size": 64}).hash("pass_a") != cfg.hash("pass_a")

    # A pair-construction change must not invalidate Pass A, but must invalidate B and C.
    pb = cfg.model_copy(
        update={"pass_b": cfg.pass_b.model_copy(update={"n_difficult": 40, "n_easy": 40})}
    )
    assert pb.hash("pass_a") == cfg.hash("pass_a")
    assert pb.hash("pass_b") != cfg.hash("pass_b")
    assert pb.hash("pass_c") != cfg.hash("pass_c")

    # A Pass C condition change leaves Pass A and B intact.
    pc = cfg.model_copy(
        update={
            "pass_c": cfg.pass_c.model_copy(
                update={"conditions": ("chose", "yoked", "random")}
            )
        }
    )
    assert pc.hash("pass_b") == cfg.hash("pass_b")
    assert pc.hash("pass_c") != cfg.hash("pass_c")


def test_every_artifact_records_the_full_config_hash():
    """The stage hash keys the cache; the FULL hash is in provenance, so any
    artifact can still be traced back to the complete configuration."""
    prov = _prov(config_hash=load_config(CONFIGS[0]).hash())
    assert provenance_of(_frame(prov))["config_hash"] == load_config(CONFIGS[0]).hash()


def test_all_five_models_have_distinct_hashes():
    hashes = {load_config(p).model.name: load_config(p).hash() for p in CONFIGS}
    assert len(set(hashes.values())) == len(hashes)


def test_smoke_configs_differ_from_run_configs():
    run = load_config("configs/stage0_qwen2.5-0.5b.yaml")
    smoke = load_config("configs/smoke_qwen2.5-0.5b.yaml")
    assert smoke.hash() != run.hash()
    assert smoke.smoke and not run.smoke


def test_selection_and_analysis_template_sets_must_be_disjoint():
    from src.config import PassAConfig

    with pytest.raises(ValueError, match="disjoint"):
        PassAConfig(selection_templates=(0, 1, 2), analysis_templates=(2, 3))


def test_primary_contrast_conditions_are_required():
    from src.config import PassCConfig

    with pytest.raises(ValueError, match="required for the primary contrast"):
        PassCConfig(conditions=("yoked", "random"))


def test_resolve_revision_passes_through_a_sha():
    sha = "a" * 40
    assert resolve_revision("some/model", sha) == (sha, True)
