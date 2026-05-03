#!/usr/bin/env python3
"""Open-loop DeePC-style prediction on held-out ns-3 KPI data.

This script does not run ns-3. It uses already collected open-loop PRBS data to
test whether past input/output/context plus known future inputs/context can
predict future PRR, PIR, and CBR.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_DATASET_DIR = Path("data/output/deepc_open_loop_250veh")
DEFAULT_OUT_DIR = DEFAULT_DATASET_DIR / "open_loop_predictions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate linear and kernelized open-loop DeePC predictors."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Directory created by scripts/build_hankel_dataset.py.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory.")
    parser.add_argument("--past-horizon", type=int, default=None, help="Override Tini samples.")
    parser.add_argument("--future-horizon", type=int, default=None, help="Override N samples.")
    parser.add_argument("--ridge", type=float, default=1e-4, help="Linear ridge regularization.")
    parser.add_argument(
        "--kernel-ridge",
        type=float,
        default=1e-3,
        help="Kernel ridge regularization for robust/kernelized predictor.",
    )
    parser.add_argument(
        "--kernel-gamma",
        type=float,
        default=None,
        help="RBF gamma. Defaults to median-distance heuristic.",
    )
    parser.add_argument(
        "--tune-kernel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Tune RBF gamma/ridge on the validation split before testing.",
    )
    parser.add_argument(
        "--final-train-with-val",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Refit final predictors on train+validation after tuning.",
    )
    parser.add_argument(
        "--max-train-windows",
        type=int,
        default=1500,
        help="Maximum train windows for kernel predictor.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def window_matrix(values: np.ndarray, start: int, length: int) -> np.ndarray:
    return values[start : start + length].reshape(-1)


def make_supervised_windows(
    df: pd.DataFrame,
    input_cols: list[str],
    output_cols: list[str],
    context_cols: list[str],
    past_horizon: int,
    future_horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = past_horizon + future_horizon
    n_windows = len(df) - total + 1
    if n_windows <= 0:
        raise ValueError(f"Need more than {total} rows; got {len(df)}")

    u = df[input_cols].to_numpy(dtype=float)
    y = df[output_cols].to_numpy(dtype=float)
    d = df[context_cols].to_numpy(dtype=float) if context_cols else np.empty((len(df), 0))
    times = df["time_s"].to_numpy(dtype=float)

    features = []
    targets = []
    target_times = []
    for i in range(n_windows):
        parts = [
            window_matrix(u, i, past_horizon),
            window_matrix(y, i, past_horizon),
            window_matrix(u, i + past_horizon, future_horizon),
        ]
        if context_cols:
            parts.insert(2, window_matrix(d, i, past_horizon))
            parts.append(window_matrix(d, i + past_horizon, future_horizon))
        features.append(np.concatenate(parts))
        targets.append(window_matrix(y, i + past_horizon, future_horizon))
        target_times.append(times[i + past_horizon])

    return np.asarray(features), np.asarray(targets), np.asarray(target_times)


def select_train_windows(x: np.ndarray, y: np.ndarray, max_windows: int) -> tuple[np.ndarray, np.ndarray]:
    if max_windows <= 0 or len(x) <= max_windows:
        return x, y
    idx = np.linspace(0, len(x) - 1, max_windows).round().astype(int)
    return x[idx], y[idx]


def fit_linear_ridge(x_train: np.ndarray, y_train: np.ndarray, ridge: float) -> np.ndarray:
    x_aug = np.column_stack([x_train, np.ones(len(x_train))])
    lhs = x_aug.T @ x_aug
    reg = ridge * np.eye(lhs.shape[0])
    reg[-1, -1] = 0.0
    rhs = x_aug.T @ y_train
    return np.linalg.solve(lhs + reg, rhs)


def predict_linear_ridge(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    x_aug = np.column_stack([x, np.ones(len(x))])
    return x_aug @ beta


def squared_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a2 = np.sum(a * a, axis=1)[:, None]
    b2 = np.sum(b * b, axis=1)[None, :]
    return np.maximum(a2 + b2 - 2.0 * (a @ b.T), 0.0)


def median_gamma(x: np.ndarray, max_points: int = 800) -> float:
    if len(x) > max_points:
        idx = np.linspace(0, len(x) - 1, max_points).round().astype(int)
        sample = x[idx]
    else:
        sample = x
    d2 = squared_distances(sample, sample)
    upper = d2[np.triu_indices_from(d2, k=1)]
    upper = upper[upper > 1e-12]
    if len(upper) == 0:
        return 1.0
    return 1.0 / float(np.median(upper))


def rbf_kernel(a: np.ndarray, b: np.ndarray, gamma: float) -> np.ndarray:
    return np.exp(-gamma * squared_distances(a, b))


def fit_kernel_ridge(
    x_train: np.ndarray, y_train: np.ndarray, ridge: float, gamma: float
) -> np.ndarray:
    k_train = rbf_kernel(x_train, x_train, gamma)
    return np.linalg.solve(k_train + ridge * np.eye(len(k_train)), y_train)


def predict_kernel_ridge(
    x_query: np.ndarray, x_train: np.ndarray, alpha: np.ndarray, gamma: float
) -> np.ndarray:
    return rbf_kernel(x_query, x_train, gamma) @ alpha


def normalized_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_pred - y_true) ** 2))


def tune_kernel_hyperparameters(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    base_gamma: float,
    max_train_windows: int,
) -> dict:
    x_kernel, y_kernel = select_train_windows(x_train, y_train, max_train_windows)
    gamma_factors = [0.05, 0.1, 0.3, 1.0, 3.0, 10.0]
    ridge_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

    best = {
        "score": float("inf"),
        "gamma": base_gamma,
        "kernel_ridge": 1e-3,
        "gamma_factor": 1.0,
    }
    for factor in gamma_factors:
        gamma = base_gamma * factor
        k_train = rbf_kernel(x_kernel, x_kernel, gamma)
        k_val = rbf_kernel(x_val, x_kernel, gamma)
        eye = np.eye(len(k_train))
        for ridge in ridge_values:
            alpha = np.linalg.solve(k_train + ridge * eye, y_kernel)
            pred = k_val @ alpha
            score = normalized_mse(y_val, pred)
            if score < best["score"]:
                best = {
                    "score": score,
                    "gamma": gamma,
                    "kernel_ridge": ridge,
                    "gamma_factor": factor,
                }
    return best


def denormalize_outputs(y_norm: np.ndarray, output_cols: list[str], scaler: dict) -> np.ndarray:
    y = y_norm.copy()
    n_outputs = len(output_cols)
    future_horizon = y.shape[1] // n_outputs
    y = y.reshape(len(y), future_horizon, n_outputs)
    for j, col in enumerate(output_cols):
        y[:, :, j] = y[:, :, j] * scaler["std"][col] + scaler["mean"][col]
    return y


def clip_physical(y: np.ndarray, output_cols: list[str]) -> np.ndarray:
    out = y.copy()
    for j, col in enumerate(output_cols):
        if col in {"prr_150m", "cbr", "sensing_exclusion_ratio"}:
            out[:, :, j] = np.clip(out[:, :, j], 0.0, 1.0)
        elif col == "pir_s":
            out[:, :, j] = np.maximum(out[:, :, j], 0.0)
    return out


def metrics_by_output(
    y_true: np.ndarray, y_pred: np.ndarray, output_cols: list[str], model_name: str
) -> dict:
    result = {"model": model_name, "horizons": {}, "overall": {}}
    errors = y_pred - y_true
    for j, col in enumerate(output_cols):
        e = errors[:, :, j].reshape(-1)
        truth = y_true[:, :, j].reshape(-1)
        rmse = float(np.sqrt(np.mean(e * e)))
        mae = float(np.mean(np.abs(e)))
        denom = float(np.sum((truth - np.mean(truth)) ** 2))
        r2 = float(1.0 - np.sum(e * e) / denom) if denom > 1e-12 else float("nan")
        result["overall"][col] = {"rmse": rmse, "mae": mae, "r2": r2}

    for h in range(y_true.shape[1]):
        horizon_key = str(h + 1)
        result["horizons"][horizon_key] = {}
        for j, col in enumerate(output_cols):
            e = errors[:, h, j]
            truth = y_true[:, h, j]
            denom = float(np.sum((truth - np.mean(truth)) ** 2))
            result["horizons"][horizon_key][col] = {
                "rmse": float(np.sqrt(np.mean(e * e))),
                "mae": float(np.mean(np.abs(e))),
                "r2": float(1.0 - np.sum(e * e) / denom) if denom > 1e-12 else float("nan"),
            }
    return result


def write_first_step_predictions(
    out_dir: Path,
    times: np.ndarray,
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    output_cols: list[str],
) -> None:
    rows = {"time_s": times}
    for j, col in enumerate(output_cols):
        rows[f"measured_{col}"] = y_true[:, 0, j]
    for name, pred in predictions.items():
        for j, col in enumerate(output_cols):
            rows[f"{name}_{col}"] = pred[:, 0, j]
    pd.DataFrame(rows).to_csv(out_dir / "first_step_predictions.csv", index=False)


def plot_predictions(
    out_dir: Path,
    times: np.ndarray,
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    output_cols: list[str],
) -> None:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    max_points = min(len(times), 1200)
    sl = slice(0, max_points)
    for j, col in enumerate(output_cols):
        plt.figure(figsize=(10, 4))
        plt.plot(times[sl], y_true[sl, 0, j], label=f"measured {col}", linewidth=1.8)
        for name, pred in predictions.items():
            plt.plot(times[sl], pred[sl, 0, j], label=f"{name} {col}", alpha=0.8)
        plt.xlabel("Time (s)")
        plt.ylabel(col)
        plt.title(f"One-step open-loop prediction: {col}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / f"{col}_timeseries.png", dpi=180)
        plt.close()

        plt.figure(figsize=(5, 5))
        for name, pred in predictions.items():
            plt.scatter(y_true[:, 0, j], pred[:, 0, j], s=8, alpha=0.35, label=name)
        lo = float(min(y_true[:, 0, j].min(), *(pred[:, 0, j].min() for pred in predictions.values())))
        hi = float(max(y_true[:, 0, j].max(), *(pred[:, 0, j].max() for pred in predictions.values())))
        plt.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        plt.xlabel(f"Measured {col}")
        plt.ylabel(f"Predicted {col}")
        plt.title(f"Measured vs predicted: {col}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / f"{col}_scatter.png", dpi=180)
        plt.close()


def main() -> None:
    args = parse_args()
    summary = load_json(args.dataset_dir / "dataset_summary.json")
    scaler = load_json(args.dataset_dir / "scaler.json")

    input_cols = summary["input_cols"]
    output_cols = summary["output_cols"]
    context_cols = summary["context_cols"]
    past_horizon = args.past_horizon or int(summary["past_horizon_samples"])
    future_horizon = args.future_horizon or int(summary["future_horizon_samples"])

    train = pd.read_csv(args.dataset_dir / "deepc_train_normalized.csv")
    val = pd.read_csv(args.dataset_dir / "deepc_val_normalized.csv")
    test = pd.read_csv(args.dataset_dir / "deepc_test_normalized.csv")

    x_train, y_train, _ = make_supervised_windows(
        train, input_cols, output_cols, context_cols, past_horizon, future_horizon
    )
    x_val, y_val, _ = make_supervised_windows(
        val, input_cols, output_cols, context_cols, past_horizon, future_horizon
    )
    x_test, y_test, target_times = make_supervised_windows(
        test, input_cols, output_cols, context_cols, past_horizon, future_horizon
    )

    base_gamma = args.kernel_gamma if args.kernel_gamma is not None else median_gamma(x_train)
    kernel_choice = {
        "score": None,
        "gamma": base_gamma,
        "kernel_ridge": args.kernel_ridge,
        "gamma_factor": None,
    }
    if args.tune_kernel:
        kernel_choice = tune_kernel_hyperparameters(
            x_train, y_train, x_val, y_val, base_gamma, args.max_train_windows
        )

    if args.final_train_with_val:
        train_final = pd.concat([train, val], ignore_index=True)
        x_linear_train, y_linear_train, _ = make_supervised_windows(
            train_final, input_cols, output_cols, context_cols, past_horizon, future_horizon
        )
    else:
        x_linear_train, y_linear_train = x_train, y_train

    x_kernel, y_kernel = select_train_windows(
        x_linear_train, y_linear_train, args.max_train_windows
    )

    beta = fit_linear_ridge(x_linear_train, y_linear_train, args.ridge)
    y_linear_norm = predict_linear_ridge(x_test, beta)

    gamma = float(kernel_choice["gamma"])
    kernel_ridge = float(kernel_choice["kernel_ridge"])
    alpha = fit_kernel_ridge(x_kernel, y_kernel, kernel_ridge, gamma)
    y_kernel_norm = predict_kernel_ridge(x_test, x_kernel, alpha, gamma)

    y_true = denormalize_outputs(y_test, output_cols, scaler)
    predictions = {
        "linear_deepc": clip_physical(denormalize_outputs(y_linear_norm, output_cols, scaler), output_cols),
        "kernel_rokdeepc": clip_physical(
            denormalize_outputs(y_kernel_norm, output_cols, scaler), output_cols
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_first_step_predictions(args.out_dir, target_times, y_true, predictions, output_cols)
    plot_predictions(args.out_dir, target_times, y_true, predictions, output_cols)

    metrics = {
        "dataset_dir": str(args.dataset_dir),
        "out_dir": str(args.out_dir),
        "input_cols": input_cols,
        "output_cols": output_cols,
        "context_cols": context_cols,
        "past_horizon_samples": past_horizon,
        "future_horizon_samples": future_horizon,
        "train_windows": int(len(x_train)),
        "validation_windows": int(len(x_val)),
        "final_linear_train_windows": int(len(x_linear_train)),
        "kernel_train_windows": int(len(x_kernel)),
        "test_windows": int(len(x_test)),
        "linear_ridge": args.ridge,
        "kernel_ridge": kernel_ridge,
        "kernel_tuning_enabled": args.tune_kernel,
        "kernel_validation_choice": kernel_choice,
        "kernel_gamma": gamma,
        "models": [
            metrics_by_output(y_true, predictions["linear_deepc"], output_cols, "linear_deepc"),
            metrics_by_output(y_true, predictions["kernel_rokdeepc"], output_cols, "kernel_rokdeepc"),
        ],
    }
    with (args.out_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved open-loop prediction artifacts to: {args.out_dir}")
    for model in metrics["models"]:
        print(f"\n{model['model']}")
        for col, values in model["overall"].items():
            print(
                f"  {col}: rmse={values['rmse']:.6f}, "
                f"mae={values['mae']:.6f}, r2={values['r2']:.6f}"
            )


if __name__ == "__main__":
    main()
