"""Provenance, the cross-device pooling guard, and config hashing."""

from __future__ import annotations

import pandas as pd
import pytest
from pathlib import Path

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
        pymc_version="5.28.5",
        pytensor_version="2.38.3",
        pytensor_mode="Mode",
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


def test_pass_c_wording_edits_do_not_invalidate_pass_a(tmp_path):
    """A Pass C template edit must not orphan Pass A's forward passes.

    The first stimulus digest hashed templates.yaml whole, so adding the Pass C fields
    Amendment 2 requires moved the pass_a hash and orphaned 40,000 cached comparisons per
    model against Pass A prompts that were byte-for-byte unchanged. Caught by the run
    machine before it burned the passes.

    Hashes are captured BEFORE the edit: `hash()` re-reads the stimulus files on every
    call, so comparing a live config against itself after an edit compares the new value
    to the new value and passes vacuously.
    """
    import yaml

    from src.config import load_config

    cfg = load_config("configs/stage0_qwen2.5-0.5b.yaml")
    before = {s: cfg.hash(s) for s in ("pass_a", "pass_b", "pass_c")}
    tp = cfg.resolve(cfg.stimuli.templates_path)
    original = tp.read_bytes()

    def rehash(mutate):
        loaded = yaml.safe_load(tp.read_text())
        rows = loaded["templates"] if isinstance(loaded, dict) else loaded
        mutate(rows)
        tp.write_text(yaml.safe_dump(loaded if isinstance(loaded, dict) else rows))
        c = load_config("configs/stage0_qwen2.5-0.5b.yaml")
        return {s: c.hash(s) for s in ("pass_a", "pass_b", "pass_c")}

    try:
        after = rehash(lambda r: r[0].update(receipt="DIFFERENT", provisional="ALSO DIFFERENT"))
        assert after["pass_a"] == before["pass_a"], "Pass C edit moved the pass_a hash"
        assert after["pass_b"] == before["pass_b"], "Pass C edit moved the pass_b hash"
        assert after["pass_c"] != before["pass_c"], "Pass C edit must move the pass_c hash"

        # ...and the R1 guard still fires for Pass A wording, which is the whole point of
        # having a stimulus digest at all.
        tp.write_bytes(original)
        after = rehash(lambda r: r[0]["rating"].update(ascending="DIFFERENT PROMPT"))
        assert after["pass_a"] != before["pass_a"], "R1 guard broken: Pass A edit ignored"
        assert after["pass_c"] != before["pass_c"], "a Pass A edit must reach Pass C too"
    finally:
        tp.write_bytes(original)


def test_stimulus_digest_ignores_formatting_but_not_content():
    """Comments and whitespace never reach the model, so they must not invalidate a cache.

    The scoped digest canonicalises the projected fields as JSON instead of hashing raw
    bytes, which is what makes this true for Pass A. Pass C still hashes whole bytes.
    """
    from src.config import load_config

    cfg = load_config("configs/stage0_qwen2.5-0.5b.yaml")
    tp = cfg.resolve(cfg.stimuli.templates_path)
    original = tp.read_bytes()
    before = cfg.hash("pass_a")
    try:
        tp.write_bytes(b"# a new comment that no model will ever see\n" + original)
        assert load_config("configs/stage0_qwen2.5-0.5b.yaml").hash("pass_a") == before
    finally:
        tp.write_bytes(original)


def test_stimulus_digest_is_independent_of_line_endings():
    """A CRLF checkout must compute the same digest as an LF one.

    The digest hashed items.yaml and anchors.yaml as raw bytes, so it depended on the
    CHECKOUT rather than the content: the blobs are LF, and a Windows clone with
    core.autocrlf=true (the default) gets CRLF and a different digest for identical
    stimuli. No cache could ever transfer between the development machine and the run
    machine. Silent, permanent, and strictly worse than the over-invalidation the scoping
    was meant to fix -- so it is pinned here in both directions.
    """
    from src.config import load_config

    cfg = load_config("configs/stage0_qwen2.5-0.5b.yaml")
    files = [cfg.resolve(cfg.stimuli.items_path),
             cfg.resolve(cfg.stimuli.templates_path),
             cfg.resolve(Path("src/stimuli/anchors.yaml"))]
    before = {s: cfg.hash(s) for s in ("pass_a", "pass_b", "pass_c")}
    originals = {f: f.read_bytes() for f in files}
    try:
        for f, raw in originals.items():
            f.write_bytes(raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
        assert any(b"\r\n" in f.read_bytes() for f in files), "CRLF conversion did nothing"
        after = {s: load_config("configs/stage0_qwen2.5-0.5b.yaml").hash(s)
                 for s in ("pass_a", "pass_b", "pass_c")}
        assert after == before, f"line endings changed the digest: {before} -> {after}"
    finally:
        for f, raw in originals.items():
            f.write_bytes(raw)


@pytest.mark.parametrize("config", CONFIGS)
def test_config_hash_is_independent_of_the_os_path_separator(config):
    """The stage hash must not encode which operating system computed it.

    `model_dump(mode="json")` renders a Path with `str()`. On Windows the field holds a
    `WindowsPath`, so `items_path` serialised as `src\\stimuli\\items.yaml` and went into
    the hash payload with backslashes. The same commit therefore produced `417d714030e5`
    on macOS and `d870ea4af66f` on the run machine, for byte-identical stimuli and
    byte-identical config -- the run machine could never reproduce a hash quoted here, and
    the natural reading of that was a corrupted checkout rather than a defect in the hash.

    This is the third checkout- or platform-dependent input found in the digest, after the
    unscoped templates.yaml and the raw-bytes stimulus read, so the assertion is the blunt
    one: no separator-bearing string may appear anywhere in the payload.
    """
    import json

    from src.config import STAGE_HASH_FIELDS

    cfg = load_config(config)
    dumped = cfg.model_dump(mode="json")
    for stage, fields in STAGE_HASH_FIELDS.items():
        blob = json.dumps({k: dumped[k] for k in fields}, sort_keys=True)
        assert "\\" not in blob, f"{stage} payload carries an OS path separator: {blob}"


def test_paths_serialise_posix_style_even_when_the_path_is_windows_flavoured():
    """Pins the mechanism, not just the symptom: the serialiser is `as_posix`.

    The payload assertion above passes vacuously on a POSIX box -- there are no
    backslashes to find -- so the actual conversion is exercised directly here, since CI
    and both working machines are POSIX and would never construct a `WindowsPath`.
    """
    from pathlib import PureWindowsPath

    from src.config import StimuliConfig

    windows = PureWindowsPath("src") / "stimuli" / "items.yaml"
    assert "\\" in str(windows), "precondition: PureWindowsPath must use backslashes"
    assert StimuliConfig()._as_posix(windows) == "src/stimuli/items.yaml"
