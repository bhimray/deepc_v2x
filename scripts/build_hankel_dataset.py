#!/usr/bin/env python3
"""Build DeePC-ready datasets and Hankel matrices from ns-3 KPI logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_KPI = Path("data/output/kpi_timeseries_10min_250veh_run01.csv")
DEFAULT_OUT_DIR = Path("data/output/deepc_open_loop_250veh")

INPUT_COLS = ["tx_power_dbm", "beacon_interval_s"]
OUTPUT_COLS = ["prr_150m", "pir_s", "cbr"]
CONTEXT_COLS = [
    "active_vehicle_count_core",
    "density_veh_per_km_core",
    "mean_neighbors_150m",
    "mean_neighbors_300m",
    "sensing_exclusion_ratio",
]


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def metadata_path(kpi_csv: Path) -> Path:
    return Path(f"{kpi_csv.with_suffix('')}_metadata.json")


def load_metadata(kpi_csv: Path, override: Path | None) -> dict:
    path = override or metadata_path(kpi_csv)
    if path.exists():
        with path.open() as f:
            return json.load(f)
    return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create normalized DeePC datasets and Hankel matrices from ns-3 KPI CSVs."
    )
    parser.add_argument("--kpi-csv", type=Path, default=DEFAULT_KPI, help="Input KPI CSV.")
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=None,
        help="Metadata JSON. Defaults to a sidecar next to --kpi-csv.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory.")
    parser.add_argument(
        "--input-cols",
        default=",".join(INPUT_COLS),
        help="Comma-separated DeePC input columns.",
    )
    parser.add_argument(
        "--output-cols",
        default=",".join(OUTPUT_COLS),
        help="Comma-separated DeePC output columns.",
    )
    parser.add_argument(
        "--context-cols",
        default=",".join(CONTEXT_COLS),
        help="Comma-separated measured context/disturbance columns.",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=None,
        help="Warm-up cutoff in seconds. Defaults to metadata evaluation_start_s.",
    )
    parser.add_argument(
        "--eval-end",
        type=float,
        default=None,
        help="Evaluation end time in seconds. Defaults to metadata evaluation_end_s.",
    )
    parser.add_argument("--train-frac", type=float, default=0.60, help="Time-ordered train share.")
    parser.add_argument("--val-frac", type=float, default=0.20, help="Time-ordered validation share.")
    parser.add_argument("--past-horizon", type=int, default=20, help="Past horizon Tini in samples.")
    parser.add_argument("--future-horizon", type=int, default=10, help="Future horizon N in samples.")
    return parser.parse_args()


def require_columns(df: pd.DataFrame, path: Path, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def split_time_ordered(df: pd.DataFrame, train_frac: float, val_frac: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not (0.0 < train_frac < 1.0):
        raise ValueError("--train-frac must be in (0, 1)")
    if not (0.0 <= val_frac < 1.0):
        raise ValueError("--val-frac must be in [0, 1)")
    if train_frac + val_frac >= 1.0:
        raise ValueError("--train-frac + --val-frac must be less than 1")

    n = len(df)
    train_end = int(np.floor(n * train_frac))
    val_end = int(np.floor(n * (train_frac + val_frac)))
    if train_end <= 0 or val_end <= train_end or val_end >= n:
        raise ValueError("Dataset split produced an empty train, validation, or test set")

    return (
        df.iloc[:train_end].copy(),
        df.iloc[train_end:val_end].copy(),
        df.iloc[val_end:].copy(),
    )


def fit_scaler(train: pd.DataFrame, columns: list[str]) -> dict:
    mean = train[columns].mean()
    std = train[columns].std(ddof=0)
    std = std.mask(std.abs() < 1e-12, 1.0)
    return {
        "columns": columns,
        "mean": {col: float(mean[col]) for col in columns},
        "std": {col: float(std[col]) for col in columns},
    }


def normalize(df: pd.DataFrame, scaler: dict) -> pd.DataFrame:
    out = df.copy()
    for col in scaler["columns"]:
        out[col] = (out[col] - scaler["mean"][col]) / scaler["std"][col]
    return out


def window_matrix(values: np.ndarray, start: int, length: int) -> np.ndarray:
    return values[start : start + length].reshape(-1)


def build_hankel_blocks(
    df: pd.DataFrame,
    input_cols: list[str],
    output_cols: list[str],
    context_cols: list[str],
    past_horizon: int,
    future_horizon: int,
) -> dict[str, np.ndarray]:
    total = past_horizon + future_horizon
    n_windows = len(df) - total + 1
    if n_windows <= 0:
        raise ValueError(
            f"Need more than {total} rows to build Hankel windows; got {len(df)} rows"
        )

    u = df[input_cols].to_numpy(dtype=float)
    y = df[output_cols].to_numpy(dtype=float)
    d = df[context_cols].to_numpy(dtype=float) if context_cols else np.empty((len(df), 0))
    times = df["time_s"].to_numpy(dtype=float)

    up, uf, yp, yf, dp, dfut, window_start_time = [], [], [], [], [], [], []
    for i in range(n_windows):
        up.append(window_matrix(u, i, past_horizon))
        uf.append(window_matrix(u, i + past_horizon, future_horizon))
        yp.append(window_matrix(y, i, past_horizon))
        yf.append(window_matrix(y, i + past_horizon, future_horizon))
        if context_cols:
            dp.append(window_matrix(d, i, past_horizon))
            dfut.append(window_matrix(d, i + past_horizon, future_horizon))
        window_start_time.append(times[i])

    blocks = {
        "u_p": np.asarray(up).T,
        "u_f": np.asarray(uf).T,
        "y_p": np.asarray(yp).T,
        "y_f": np.asarray(yf).T,
        "window_start_time_s": np.asarray(window_start_time),
    }
    if context_cols:
        blocks["d_p"] = np.asarray(dp).T
        blocks["d_f"] = np.asarray(dfut).T
    return blocks


def finite_checks(df: pd.DataFrame, columns: list[str]) -> dict:
    checks = {
        "rows": int(len(df)),
        "finite": bool(np.isfinite(df[columns].to_numpy(dtype=float)).all()),
        "nan_counts": {col: int(df[col].isna().sum()) for col in columns},
    }
    if "cbr" in df:
        checks["cbr_range"] = [float(df["cbr"].min()), float(df["cbr"].max())]
    if "prr_150m" in df:
        checks["prr_150m_range"] = [float(df["prr_150m"].min()), float(df["prr_150m"].max())]
    if "beacon_interval_s" in df:
        checks["beacon_interval_s_range"] = [
            float(df["beacon_interval_s"].min()),
            float(df["beacon_interval_s"].max()),
        ]
    return checks


def main() -> None:
    args = parse_args()
    input_cols = parse_csv_list(args.input_cols)
    output_cols = parse_csv_list(args.output_cols)
    context_cols = parse_csv_list(args.context_cols)
    all_model_cols = input_cols + output_cols + context_cols

    metadata = load_metadata(args.kpi_csv, args.metadata_json)
    warmup = args.warmup
    if warmup is None:
        warmup = float(metadata.get("evaluation_start_s", metadata.get("warmup_s", 0.0)))
    eval_end = args.eval_end
    if eval_end is None and "evaluation_end_s" in metadata:
        eval_end = float(metadata["evaluation_end_s"])

    raw = pd.read_csv(args.kpi_csv)
    require_columns(raw, args.kpi_csv, ["time_s", *all_model_cols])

    clean = raw.copy()
    clean = clean[clean["time_s"] >= warmup]
    if eval_end is not None:
        clean = clean[clean["time_s"] <= eval_end]
    clean = clean[["time_s", *all_model_cols]].replace([np.inf, -np.inf], np.nan).dropna()
    clean = clean.sort_values("time_s").reset_index(drop=True)

    if clean.empty:
        raise ValueError("No rows remain after filtering and dropping NaNs")
    if "cbr" in clean and ((clean["cbr"] < 0.0) | (clean["cbr"] > 1.0)).any():
        raise ValueError("CBR values must be in [0, 1]")
    if "prr_150m" in clean and ((clean["prr_150m"] < 0.0) | (clean["prr_150m"] > 1.0)).any():
        raise ValueError("PRR values must be in [0, 1]")
    if "beacon_interval_s" in clean and (clean["beacon_interval_s"] <= 0.0).any():
        raise ValueError("Beacon interval must be positive")

    train, val, test = split_time_ordered(clean, args.train_frac, args.val_frac)
    scaler = fit_scaler(train, all_model_cols)
    train_norm = normalize(train, scaler)
    val_norm = normalize(val, scaler)
    test_norm = normalize(test, scaler)
    clean_norm = normalize(clean, scaler)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    clean.to_csv(args.out_dir / "deepc_clean.csv", index=False)
    train.to_csv(args.out_dir / "deepc_train.csv", index=False)
    val.to_csv(args.out_dir / "deepc_val.csv", index=False)
    test.to_csv(args.out_dir / "deepc_test.csv", index=False)
    clean_norm.to_csv(args.out_dir / "deepc_clean_normalized.csv", index=False)
    train_norm.to_csv(args.out_dir / "deepc_train_normalized.csv", index=False)
    val_norm.to_csv(args.out_dir / "deepc_val_normalized.csv", index=False)
    test_norm.to_csv(args.out_dir / "deepc_test_normalized.csv", index=False)

    with (args.out_dir / "scaler.json").open("w") as f:
        json.dump(scaler, f, indent=2)

    hankel = build_hankel_blocks(
        train_norm,
        input_cols,
        output_cols,
        context_cols,
        args.past_horizon,
        args.future_horizon,
    )
    np.savez_compressed(args.out_dir / "hankel_train_normalized.npz", **hankel)

    summary = {
        "source_kpi_csv": str(args.kpi_csv),
        "source_metadata_json": str(args.metadata_json or metadata_path(args.kpi_csv)),
        "output_dir": str(args.out_dir),
        "input_cols": input_cols,
        "output_cols": output_cols,
        "context_cols": context_cols,
        "warmup_s": warmup,
        "eval_end_s": eval_end,
        "sample_time_s": metadata.get("sample_time_s"),
        "past_horizon_samples": args.past_horizon,
        "future_horizon_samples": args.future_horizon,
        "past_horizon_s": args.past_horizon * float(metadata.get("sample_time_s", 0.1)),
        "future_horizon_s": args.future_horizon * float(metadata.get("sample_time_s", 0.1)),
        "split_rows": {
            "clean": int(len(clean)),
            "train": int(len(train)),
            "validation": int(len(val)),
            "test": int(len(test)),
        },
        "train_time_range_s": [float(train["time_s"].min()), float(train["time_s"].max())],
        "validation_time_range_s": [float(val["time_s"].min()), float(val["time_s"].max())],
        "test_time_range_s": [float(test["time_s"].min()), float(test["time_s"].max())],
        "hankel_shapes": {key: list(value.shape) for key, value in hankel.items()},
        "data_checks": finite_checks(clean, all_model_cols),
        "metadata": metadata,
    }
    with (args.out_dir / "dataset_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved DeePC dataset artifacts to: {args.out_dir}")
    print(f"Rows clean/train/val/test: {len(clean)}/{len(train)}/{len(val)}/{len(test)}")
    print(
        "Hankel train shapes: "
        + ", ".join(f"{key}={value.shape}" for key, value in hankel.items())
    )


if __name__ == "__main__":
    main()
