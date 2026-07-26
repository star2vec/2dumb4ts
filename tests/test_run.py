"""The orchestrator's caching contract.

The instrument fit is 27 Bayesian fits -- hours at production sampling on any machine,
because it is CPU-bound MCMC rather than GPU work. Caching it is what makes a Pass C run
affordable. But caching a FIT is riskier than caching forward passes: forward passes are
deterministic given model and prompt, whereas a fit depends on the model SPECIFICATION,
which lives in code the config hash knows nothing about.

So the one behaviour worth a test is not that the cache hits. It is that the cache MISSES
when the likelihood changes.
"""

from __future__ import annotations

from pathlib import Path

from src.config import load_config
from src.experiments import run as R

CONFIG = "configs/stage0_qwen2.5-0.5b.yaml"


def _key(cfg, sources) -> str:
    """Recompute the cache key under a patched source list."""
    import hashlib

    root = Path(R.__file__).resolve().parents[2]
    d = hashlib.sha256()
    for rel in sources:
        d.update((root / rel).read_bytes())
    return f"{cfg.hash('pass_a')}-{d.hexdigest()[:8]}"


def test_instrument_cache_key_tracks_the_fitting_code(tmp_path):
    """Editing the likelihood must invalidate the cached fit.

    Keyed on the config hash alone, a changed Bradley-Terry model would silently serve
    numbers produced by the previous version of that model -- the worst kind of stale,
    because the run would report a gate decision the current code never made.
    """
    cfg = load_config(CONFIG)
    real = _key(cfg, R._INSTRUMENT_SOURCES)

    edited = tmp_path / "bradley_terry.py"
    original = (Path(R.__file__).resolve().parents[2] / R._INSTRUMENT_SOURCES[0]).read_bytes()
    edited.write_bytes(original + b"\n# a change to the likelihood\n")

    import hashlib

    d = hashlib.sha256()
    d.update(edited.read_bytes())
    for rel in R._INSTRUMENT_SOURCES[1:]:
        d.update((Path(R.__file__).resolve().parents[2] / rel).read_bytes())
    assert f"{cfg.hash('pass_a')}-{d.hexdigest()[:8]}" != real


def test_instrument_cache_key_tracks_pass_a_parameters():
    """A different instrument configuration must not reuse another one's fit."""
    cfg = load_config(CONFIG)
    other = cfg.model_copy(update={"seed": cfg.seed + 1})
    assert _key(cfg, R._INSTRUMENT_SOURCES) != _key(other, R._INSTRUMENT_SOURCES)


def test_ppc_replicate_count_is_not_in_the_pass_a_hash():
    """Sampling effort must not change the stage hash.

    It is a cost knob, not a parameter of the instrument. If it entered the pass_a hash,
    lowering it for a wiring check would orphan 40,000 cached forward passes -- the same
    failure as the gate thresholds, which we already paid for once.
    """
    cfg = load_config(CONFIG)
    cheap = cfg.model_copy(
        update={"analysis": cfg.analysis.model_copy(update={"ppc_null_replicates": 3})}
    )
    assert cheap.hash("pass_a") == cfg.hash("pass_a")
    assert cheap.hash("pass_b") == cfg.hash("pass_b")
    assert cheap.hash("pass_c") == cfg.hash("pass_c")
