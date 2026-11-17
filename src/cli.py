from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .data import prepare_data
from .experiments import analyze, ensure_data, mia_check, read_config, run_baseline, run_experiments, solve


def main():
    parser = argparse.ArgumentParser(description="Run tabular GAN privacy and Stackelberg experiments")
    parser.add_argument("command", choices=["prepare-data", "baseline", "mia-check", "run-experiments",
                                            "solve-game", "analyze", "smoke-test"])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    args = parser.parse_args()
    cfg = read_config(args.config)
    if args.command == "prepare-data":
        raw = cfg["dataset"]["raw_dir"]
        out = Path(cfg["dataset"]["processed_dir"]) / "adult_clean.csv"
        prepared = prepare_data(raw, out, cfg["dataset"]["features"], cfg["dataset"]["target"])
        print(f"Prepared dataset: {prepared}")
    elif args.command == "baseline":
        print(run_baseline(args.config, args.mode).to_string(index=False))
    elif args.command == "mia-check":
        mia_check(args.config)
    elif args.command == "run-experiments":
        data = run_experiments(args.config, args.mode)
        print(data.to_string(index=False))
    elif args.command == "solve-game":
        print(solve(args.config).to_string(index=False))
    elif args.command == "analyze":
        analyze(args.config)
        print(f"Analysis written under {cfg['paths']['figures']} and {cfg['paths']['report']}")
    elif args.command == "smoke-test":
        data = run_experiments(args.config, "smoke")
        game = solve(args.config)
        analyze(args.config)
        preview = pd.read_csv(cfg["paths"].get("synthetic_preview", "results/processed/synthetic_preview.csv"))
        expected = cfg["dataset"]["features"] + [cfg["dataset"]["target"]]
        if list(preview.columns) != expected:
            raise RuntimeError(f"Synthetic schema mismatch: {list(preview.columns)} != {expected}")
        if data.empty or game.empty:
            raise RuntimeError("Smoke experiment or game produced no rows")
        print("Smoke pipeline completed: data, attacks, game, comparisons, figures and report were generated.")


if __name__ == "__main__":
    main()
