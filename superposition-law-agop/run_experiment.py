#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = ROOT / "experiments"
RESULTS_DIR = ROOT / "results"
DATA_DIR = ROOT / "data"

EXPERIMENTS = {
    "cnn": {
        "script": EXPERIMENTS_DIR / "cnn_shape_sweep_cifar10_agop.py",
        "default_out_dir": RESULTS_DIR / "cnn_shape_sweep",
        "default_args": ["--data_dir", str(DATA_DIR / "cifar10")],
    },
    "mlp": {
        "script": EXPERIMENTS_DIR / "mlp_shape_sweep_supervised_pde_agop.py",
        "default_out_dir": RESULTS_DIR / "mlp_pde_shape_sweep",
        "default_args": [],
    },
    "rnn": {
        "script": EXPERIMENTS_DIR / "rnn_shape_sweep_mackeyglass_superposition_agop.py",
        "default_out_dir": RESULTS_DIR / "rnn_mackeyglass_shape_sweep",
        "default_args": [],
    },
    "transformer": {
        "script": EXPERIMENTS_DIR / "transformer_shape_agop.py",
        "default_out_dir": RESULTS_DIR / "transformer_shape_sweep",
        "default_args": [],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified launcher for superposition-law AGOP experiments."
    )
    parser.add_argument(
        "experiment",
        choices=sorted(EXPERIMENTS.keys()) + ["all", "list"],
        help="Experiment to run, or `all` to run every experiment sequentially.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device passed through to experiment scripts, e.g. `cuda` or `cpu`.",
    )
    parser.add_argument(
        "--out-root",
        default=str(RESULTS_DIR),
        help="Root directory for experiment outputs. Each experiment gets its own subdirectory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command(s) without executing them.",
    )
    args, script_args = parser.parse_known_args()
    args.script_args = script_args
    return args


def build_command(name: str, args: argparse.Namespace) -> list[str]:
    spec = EXPERIMENTS[name]
    out_dir = Path(args.out_root).resolve() / spec["default_out_dir"].name
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, str(spec["script"]), "--out_dir", str(out_dir)]
    cmd.extend(spec["default_args"])
    if args.device:
        cmd.extend(["--device", args.device])

    forwarded = list(args.script_args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    cmd.extend(forwarded)
    return cmd


def run_one(name: str, args: argparse.Namespace) -> int:
    cmd = build_command(name, args)
    print(f"\n[{name}] {' '.join(cmd)}")
    if args.dry_run:
        return 0
    return subprocess.run(cmd, cwd=str(ROOT), check=False).returncode


def main() -> int:
    args = parse_args()

    if args.experiment == "list":
        print("Available experiments:")
        for name in sorted(EXPERIMENTS.keys()):
            print(f"  - {name}: {EXPERIMENTS[name]['script'].name}")
        return 0

    names = sorted(EXPERIMENTS.keys()) if args.experiment == "all" else [args.experiment]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for name in names:
        code = run_one(name, args)
        if code != 0:
            print(f"\nExperiment `{name}` failed with exit code {code}.")
            return code

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
