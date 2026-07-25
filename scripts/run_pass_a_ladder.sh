#!/usr/bin/env bash
# Full Pass A for the whole model ladder, on the run machine.
#
#   400 items x 5 templates x 2 polarities = 4,000 forward passes per model.
#
# Stops after Pass A for every model: this run exists to answer whether the
# rating instrument discriminates at full sample, not to run the experiment.
#
# Exit code 2 from a model means it was excluded by a preregistered criterion.
# That is an expected outcome, not a failure, so the loop continues.
#
#   ./scripts/run_pass_a_ladder.sh

set -uo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-.venv/bin/python}"
LOGS="artifacts/logs"
mkdir -p "$LOGS"

CONFIGS=(
  configs/stage0_qwen2.5-0.5b.yaml
  configs/stage0_qwen2.5-1.5b.yaml
  configs/stage0_gemma-2-2b.yaml
  configs/stage0_qwen2.5-3b.yaml
  configs/stage0_llama-3.2-3b.yaml
)

echo "== preflight =="
"$PY" scripts/preflight.py || { echo "preflight failed; not starting the ladder"; exit 1; }

declare -a EXCLUDED=() PASSED=() ERRORED=()

for cfg in "${CONFIGS[@]}"; do
  name="$(basename "$cfg" .yaml)"
  echo
  echo "=============================================================="
  echo "  $name"
  echo "=============================================================="
  "$PY" -m src.experiments.run --config "$cfg" --stop-after pass_a --no-plots \
    2>&1 | tee "$LOGS/${name}_pass_a.log"
  code=${PIPESTATUS[0]}
  case $code in
    0) PASSED+=("$name") ;;
    2) EXCLUDED+=("$name") ;;
    *) ERRORED+=("$name (exit $code)") ;;
  esac
done

echo
echo "=============================================================="
echo "  ladder complete"
echo "=============================================================="
echo "gate passed  : ${PASSED[*]:-none}"
echo "excluded     : ${EXCLUDED[*]:-none}"
echo "errored      : ${ERRORED[*]:-none}"
echo
"$PY" scripts/summarize_pass_a.py

[ ${#ERRORED[@]} -eq 0 ] || exit 1
