"""Full Pass A for the whole model ladder. Cross-platform (Windows / Linux / macOS).

    400 items x 5 templates x 2 polarities = 4,000 forward passes per model.

Stops after Pass A for every model: this run exists to answer whether the rating
instrument discriminates at full sample, not to run the experiment.

Exit code 2 from a model means it was excluded by a preregistered criterion. That
is an expected outcome, not a failure, so the loop continues.

Uses sys.executable throughout, so it works from an activated venv, from
`uv run`, or from a bare interpreter path -- no .venv/bin vs .venv\\Scripts split.

    uv run python scripts/run_pass_a_ladder.py
    uv run python scripts/run_pass_a_ladder.py --skip-preflight
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

CONFIGS = [
    "configs/stage0_qwen2.5-0.5b.yaml",
    "configs/stage0_qwen2.5-1.5b.yaml",
    "configs/stage0_gemma-2-2b.yaml",
    "configs/stage0_qwen2.5-3b.yaml",
    "configs/stage0_llama-3.2-3b.yaml",
]

EXCLUDED_BY_CRITERION = 2


def stream(cmd: list[str], log: Path | None = None) -> int:
    """Run a command, echoing output live and optionally teeing it to a file."""
    print(f"$ {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, encoding="utf-8", errors="replace",
    )
    handle = log.open("w", encoding="utf-8") if log else None
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if handle:
                handle.write(line)
    finally:
        if handle:
            handle.close()
    return proc.wait()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-preflight", action="store_true")
    args = ap.parse_args()

    py = sys.executable
    logs = REPO / "artifacts" / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    if not args.skip_preflight:
        print("== preflight ==")
        if stream([py, "scripts/preflight.py"]) != 0:
            print("\npreflight failed; not starting the ladder")
            return 1

    passed: list[str] = []
    excluded: list[str] = []
    errored: list[str] = []

    for cfg in CONFIGS:
        name = Path(cfg).stem
        print(f"\n{'=' * 62}\n  {name}\n{'=' * 62}", flush=True)
        code = stream(
            [py, "-m", "src.experiments.run", "--config", cfg,
             "--stop-after", "pass_a", "--no-plots"],
            log=logs / f"{name}_pass_a.log",
        )
        if code == 0:
            passed.append(name)
        elif code == EXCLUDED_BY_CRITERION:
            excluded.append(name)
        else:
            errored.append(f"{name} (exit {code})")

    print(f"\n{'=' * 62}\n  ladder complete\n{'=' * 62}")
    print(f"gate passed  : {', '.join(passed) or 'none'}")
    print(f"excluded     : {', '.join(excluded) or 'none'}")
    print(f"errored      : {', '.join(errored) or 'none'}")
    print()
    stream([py, "scripts/summarize_pass_a.py"])

    return 1 if errored else 0


if __name__ == "__main__":
    sys.exit(main())
