#!/usr/bin/env python3
"""Export ns-3 DeePC Hankel artifacts to a MATLAB .mat file.

The MATLAB/YALMIP file controller loads this .mat file once, then consumes
request_XXXXXX.json files from the ns-3 bridge.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import savemat


DEFAULT_DATASET_DIR = Path("data/output/deepc_open_loop_250veh")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export DeePC data for MATLAB/YALMIP.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_DATASET_DIR / "matlab_deepc_data.mat",
        help="Output .mat file.",
    )
    return parser.parse_args()


def matlab_cellstr(values: list[str]) -> np.ndarray:
    return np.asarray(values, dtype=object).reshape(1, -1)


def main() -> None:
    args = parse_args()
    summary_path = args.dataset_dir / "dataset_summary.json"
    scaler_path = args.dataset_dir / "scaler.json"
    hankel_path = args.dataset_dir / "hankel_train_normalized.npz"

    with summary_path.open() as f:
        summary = json.load(f)
    with scaler_path.open() as f:
        scaler = json.load(f)
    with np.load(hankel_path) as loaded:
        hankel = {key: loaded[key] for key in loaded.files}

    input_cols = summary["input_cols"]
    output_cols = summary["output_cols"]
    context_cols = summary["context_cols"]
    all_cols = input_cols + output_cols + context_cols

    mean = np.asarray([scaler["mean"][col] for col in all_cols], dtype=float)
    std = np.asarray([scaler["std"][col] for col in all_cols], dtype=float)
    input_mean = np.asarray([scaler["mean"][col] for col in input_cols], dtype=float)
    input_std = np.asarray([scaler["std"][col] for col in input_cols], dtype=float)
    output_mean = np.asarray([scaler["mean"][col] for col in output_cols], dtype=float)
    output_std = np.asarray([scaler["std"][col] for col in output_cols], dtype=float)
    context_mean = np.asarray([scaler["mean"][col] for col in context_cols], dtype=float)
    context_std = np.asarray([scaler["std"][col] for col in context_cols], dtype=float)

    payload = {
        "Up": hankel["u_p"],
        "Uf": hankel["u_f"],
        "Yp": hankel["y_p"],
        "Yf": hankel["y_f"],
        "Dp": hankel["d_p"],
        "Df": hankel["d_f"],
        "window_start_time_s": hankel["window_start_time_s"],
        "input_cols": matlab_cellstr(input_cols),
        "output_cols": matlab_cellstr(output_cols),
        "context_cols": matlab_cellstr(context_cols),
        "all_cols": matlab_cellstr(all_cols),
        "mean": mean.reshape(1, -1),
        "std": std.reshape(1, -1),
        "input_mean": input_mean.reshape(-1, 1),
        "input_std": input_std.reshape(-1, 1),
        "output_mean": output_mean.reshape(-1, 1),
        "output_std": output_std.reshape(-1, 1),
        "context_mean": context_mean.reshape(-1, 1),
        "context_std": context_std.reshape(-1, 1),
        "past_horizon_samples": int(summary["past_horizon_samples"]),
        "future_horizon_samples": int(summary["future_horizon_samples"]),
        "sample_time_s": float(summary.get("sample_time_s") or 0.1),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    savemat(args.out, payload, do_compression=True)
    print(f"Saved MATLAB DeePC data to {args.out}")


if __name__ == "__main__":
    main()
