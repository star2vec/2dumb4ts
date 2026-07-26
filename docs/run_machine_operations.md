# Run machine operations

RTX 2000 Ada Generation Laptop GPU, 8 GB VRAM, Windows. Single 1 TB NVMe (SK hynix
PC801), one usable Windows volume, **~6.7 GB free**. There is no second drive, so
`HF_HOME` and `artifacts_dir` cannot be relocated.

## Disabling hf-xet is what made the 3B models downloadable

```
$env:HF_HUB_DISABLE_XET = "1"
```

The earlier download failures were **not** raw model size. `hf-xet` reconstructs a
download from chunks and needs scratch space *beyond* the final file size, so a 6.4 GB
model needed materially more than 6.4 GB free and failed at 6.7 GB. With xet disabled
the weights are written as a single blob at 1× size and a 6.4 GB model fits, one at a
time, freeing between models.

This is why the full five-model ladder completed on a box that looked too small for it.
Recorded because the diagnosis is not obvious from the error and would otherwise be
rediscovered.

## What is finished and needs no weights

Pass A (absolute and pairwise) is complete for all five models and the artifacts are
provenance-stamped. **Re-fitting requires no weights** — comparisons are cached, so any
model change or re-analysis is a CPU job. Copy `artifacts/` to the dev machine and all
re-fitting can happen there.

## The binding constraint is now activations, not weights

Weights and activations are resident *simultaneously* during Pass C, because activations
are written while the model is loaded. Per-model, caching post-manipulation passes only
(16,000 passes, bf16, all layers + 1):

| model | weights | activations | both resident |
|---|---|---|---|
| qwen2.5-0.5b | ~1.0 GB | 0.72 GB | 1.7 GB |
| qwen2.5-1.5b | ~3.1 | 1.43 | 4.5 |
| gemma-2-2b | ~5.2 | 1.99 | 7.2 |
| qwen2.5-3b | ~6.2 | 2.42 | 8.6 |
| llama-3.2-3b | ~6.4 | 2.85 | 9.3 |

Three of five exceed 6.7 GB. Sharded writing with incremental off-box transfer reduces
the peak to `weights + one shard`, but **Llama-3.2-3B's weights alone are 6.4 GB against
6.7 GB free**, which leaves no working margin regardless of shard size.

### Recommended fix: relocate, do not delete

The Pythia models occupy ~17.4 GB and are to be kept. **Move them to external storage or
an archive rather than deleting them** — that frees 17.4 GB, taking free space to ~24 GB,
at which point weights, activations and shard buffers all fit comfortably and the
juggling stops.

Failing that, ranked:

1. External drive for `artifacts_dir` only. Weights stay on C:, activations stream off.
2. Reduce cache scope to the 2×2 conditions Stage 1 actually probes
   (`chose`, `structure-control`, `self-recounted`, `yoked`) — 8,000 post passes rather
   than 16,000, roughly halving activation volume.
3. Subsample layers. Avoid unless forced: layer sweeps are how a representation gets
   located, and this forecloses that.

Activation quantization is **not** an option — bf16 is a hard constraint (§3.1) and
though probing might tolerate fp8, adding noise to the Stage 1 measurement to save disk
is the wrong trade.

## Standing rules

- `HF_HUB_DISABLE_XET=1` for every download.
- One model resident at a time; free the previous before the next.
- Never `git pull` while a ladder is running — the driver spawns a fresh subprocess per
  model, so a mid-run checkout change builds the ladder from two different commits.
- Windows Developer Mode on, so `huggingface_hub` symlinks rather than copying.
- `uv run` throughout; no venv activation needed.
- Keep the working tree clean. `assert_reportable` rejects artifacts from a dirty tree.
