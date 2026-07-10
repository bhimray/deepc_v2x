#!/usr/bin/env python3
"""Closed-loop classical DeePC rollout using Hankel matrices.

This is the direct DeePC formulation: the optimizer solves for the trajectory
coefficient vector g against the saved Hankel matrices:

    U_p g ~= u_ini,  Y_p g ~= y_ini,  D_p g ~= d_ini,  D_f g ~= d_future
    u_future = U_f g,  y_future = Y_f g

At each rollout step, only the first optimized input is applied. The predicted
first output is appended to the history and the optimization repeats.

This script does not run ns-3. It is an offline closed-loop rollout on the
Hankel DeePC surrogate. The saved control schedule can be replayed in ns-3 next.
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
from scipy.optimize import LinearConstraint, minimize


DEFAULT_DATASET_DIR = Path("data/output/deepc_open_loop_250veh")
DEFAULT_OUT_DIR = DEFAULT_DATASET_DIR / "closed_loop_classical_deepc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a classical Hankel DeePC closed-loop surrogate rollout."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--past-horizon", type=int, default=None, help="Tini samples.")
    parser.add_argument("--control-horizon", type=int, default=None, help="Future horizon N.")
    parser.add_argument("--rollout-steps", type=int, default=400)
    parser.add_argument(
        "--max-hankel-cols",
        type=int,
        default=300,
        help="Nearest Hankel columns used per solve. Use <=0 for all columns.",
    )
    parser.add_argument("--target-prr", type=float, default=0.95)
    parser.add_argument("--target-pir", type=float, default=0.10)
    parser.add_argument("--target-cbr", type=float, default=0.35)
    parser.add_argument("--target-power", type=float, default=16.0)
    parser.add_argument("--target-tb", type=float, default=0.10)
    parser.add_argument("--weight-prr", type=float, default=40.0)
    parser.add_argument("--weight-pir", type=float, default=3.0)
    parser.add_argument("--weight-cbr", type=float, default=8.0)
    parser.add_argument("--weight-power", type=float, default=0.002)
    parser.add_argument("--weight-tb", type=float, default=0.5)
    parser.add_argument("--weight-du-power", type=float, default=0.05)
    parser.add_argument("--weight-du-tb", type=float, default=15.0)
    parser.add_argument("--lambda-ini-u", type=float, default=2.0e3)
    parser.add_argument("--lambda-ini-y", type=float, default=2.0e3)
    parser.add_argument("--lambda-ini-d", type=float, default=4.0e2)
    parser.add_argument("--lambda-future-d", type=float, default=4.0e2)
    parser.add_argument("--lambda-g", type=float, default=1.0e-3)
    parser.add_argument("--min-power", type=float, default=10.0)
    parser.add_argument("--max-power", type=float, default=23.0)
    parser.add_argument("--min-tb", type=float, default=0.05)
    parser.add_argument("--max-tb", type=float, default=0.50)
    parser.add_argument(
        "--sum-to-one",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use affine DeePC constraint sum(g)=1.",
    )
    parser.add_argument("--max-iter", type=int, default=250)
    parser.add_argument("--ftol", type=float, default=1.0e-7)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def table_to_array(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return df[cols].to_numpy(dtype=float)


def row_flat(values: np.ndarray) -> np.ndarray:
    return values.reshape(-1)


def scaler_arrays(cols: list[str], scaler: dict) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray([scaler["mean"][col] for col in cols], dtype=float)
    std = np.asarray([scaler["std"][col] for col in cols], dtype=float)
    return mean, std


def normalize_row(row: np.ndarray, cols: list[str], scaler: dict) -> np.ndarray:
    mean, std = scaler_arrays(cols, scaler)
    return (row - mean) / std


def denormalize_rows(values: np.ndarray, cols: list[str], scaler: dict) -> np.ndarray:
    mean, std = scaler_arrays(cols, scaler)
    return values * std + mean


def repeated_normalized_target(
    targets: list[float], cols: list[str], scaler: dict, horizon: int
) -> np.ndarray:
    target = normalize_row(np.asarray(targets, dtype=float), cols, scaler)
    return np.tile(target, horizon)


def initial_histories(
    test_norm: pd.DataFrame,
    input_cols: list[str],
    output_cols: list[str],
    context_cols: list[str],
    past_horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        row_flat(table_to_array(test_norm.iloc[:past_horizon], input_cols)),
        row_flat(table_to_array(test_norm.iloc[:past_horizon], output_cols)),
        row_flat(table_to_array(test_norm.iloc[:past_horizon], context_cols)),
    )


def update_history(history: np.ndarray, block_size: int, new_block: np.ndarray) -> np.ndarray:
    return np.concatenate([history[block_size:], new_block])


def block_difference_matrix(horizon: int, block_size: int) -> np.ndarray:
    rows = []
    for h in range(1, horizon):
        row = np.zeros((block_size, horizon * block_size))
        row[:, h * block_size : (h + 1) * block_size] = np.eye(block_size)
        row[:, (h - 1) * block_size : h * block_size] = -np.eye(block_size)
        rows.append(row)
    if not rows:
        return np.zeros((0, horizon * block_size))
    return np.vstack(rows)


def select_hankel_columns(
    hankel: dict[str, np.ndarray],
    u_ini: np.ndarray,
    y_ini: np.ndarray,
    d_ini: np.ndarray,
    d_future: np.ndarray,
    max_cols: int,
) -> np.ndarray:
    n_cols = hankel["u_p"].shape[1]
    if max_cols <= 0 or max_cols >= n_cols:
        return np.arange(n_cols)

    blocks = [
        hankel["u_p"] - u_ini[:, None],
        hankel["y_p"] - y_ini[:, None],
        0.5 * (hankel["d_p"] - d_ini[:, None]),
        0.5 * (hankel["d_f"] - d_future[:, None]),
    ]
    score = np.sum(np.vstack(blocks) ** 2, axis=0)
    idx = np.argpartition(score, max_cols)[:max_cols]
    return idx[np.argsort(score[idx])]


def solve_deepc_step(
    hankel: dict[str, np.ndarray],
    selected_cols: np.ndarray,
    u_ini: np.ndarray,
    y_ini: np.ndarray,
    d_ini: np.ndarray,
    d_future: np.ndarray,
    prev_u_norm: np.ndarray,
    args: argparse.Namespace,
    scaler: dict,
    input_cols: list[str],
    output_cols: list[str],
) -> dict:
    up = hankel["u_p"][:, selected_cols]
    uf = hankel["u_f"][:, selected_cols]
    yp = hankel["y_p"][:, selected_cols]
    yf = hankel["y_f"][:, selected_cols]
    dp = hankel["d_p"][:, selected_cols]
    dfut = hankel["d_f"][:, selected_cols]

    n_g = len(selected_cols)
    nu = len(input_cols)
    horizon = args.control_horizon

    y_ref = repeated_normalized_target(
        [args.target_prr, args.target_pir, args.target_cbr], output_cols, scaler, horizon
    )
    u_ref = repeated_normalized_target(
        [args.target_power, args.target_tb], input_cols, scaler, horizon
    )
    q = np.tile([args.weight_prr, args.weight_pir, args.weight_cbr], horizon)
    r = np.tile([args.weight_power, args.weight_tb], horizon)

    first_u = np.zeros((nu, horizon * nu))
    first_u[:, :nu] = np.eye(nu)
    first_du_a = first_u @ uf
    diff = block_difference_matrix(horizon, nu)
    future_du_a = diff @ uf

    residuals = [
        (np.sqrt(q)[:, None] * yf, np.sqrt(q) * y_ref),
        (np.sqrt(r)[:, None] * uf, np.sqrt(r) * u_ref),
        (np.sqrt(args.lambda_ini_u) * up, np.sqrt(args.lambda_ini_u) * u_ini),
        (np.sqrt(args.lambda_ini_y) * yp, np.sqrt(args.lambda_ini_y) * y_ini),
        (np.sqrt(args.lambda_ini_d) * dp, np.sqrt(args.lambda_ini_d) * d_ini),
        (np.sqrt(args.lambda_future_d) * dfut, np.sqrt(args.lambda_future_d) * d_future),
        (np.sqrt(args.lambda_g) * np.eye(n_g), np.zeros(n_g)),
        (
            np.sqrt([args.weight_du_power, args.weight_du_tb])[:, None] * first_du_a,
            np.sqrt([args.weight_du_power, args.weight_du_tb]) * prev_u_norm,
        ),
    ]
    if len(future_du_a):
        smooth_weights = np.tile([args.weight_du_power, args.weight_du_tb], horizon - 1)
        residuals.append(
            (np.sqrt(smooth_weights)[:, None] * future_du_a, np.zeros(len(smooth_weights)))
        )

    a = np.vstack([item[0] for item in residuals])
    b = np.concatenate([item[1] for item in residuals])
    hessian = a.T @ a
    linear = a.T @ b

    def objective(g: np.ndarray) -> float:
        return 0.5 * float(g @ hessian @ g - 2.0 * linear @ g + b @ b)

    def gradient(g: np.ndarray) -> np.ndarray:
        return hessian @ g - linear

    input_lb = normalize_row(np.asarray([args.min_power, args.min_tb]), input_cols, scaler)
    input_ub = normalize_row(np.asarray([args.max_power, args.max_tb]), input_cols, scaler)
    constraints = [LinearConstraint(uf, np.tile(input_lb, horizon), np.tile(input_ub, horizon))]
    if args.sum_to_one:
        constraints.append(LinearConstraint(np.ones((1, n_g)), np.ones(1), np.ones(1)))

    g0 = np.full(n_g, 1.0 / n_g)
    result = minimize(
        objective,
        g0,
        jac=gradient,
        method="SLSQP",
        constraints=constraints,
        options={"maxiter": args.max_iter, "ftol": args.ftol, "disp": False},
    )
    g = result.x if result.x is not None else g0
    return {
        "u_future_norm": uf @ g,
        "y_future_norm": yf @ g,
        "status": int(result.status),
        "success": bool(result.success),
        "message": str(result.message),
        "objective": float(result.fun) if result.fun is not None else float("nan"),
        "selected_cols": int(n_g),
        "sum_g": float(np.sum(g)),
        "g_norm": float(np.linalg.norm(g)),
    }


def compute_metrics(result: pd.DataFrame) -> dict:
    return {
        "rollout_rows": int(len(result)),
        "mean_deepc_prr_awareness": float(result["deepc_prr_awareness"].mean()),
        "mean_deepc_pir_s": float(result["deepc_pir_s"].mean()),
        "mean_deepc_cbr": float(result["deepc_cbr"].mean()),
        "mean_measured_open_loop_prr_awareness": float(
            result["measured_open_loop_prr_awareness"].mean()
        ),
        "mean_measured_open_loop_pir_s": float(result["measured_open_loop_pir_s"].mean()),
        "mean_measured_open_loop_cbr": float(result["measured_open_loop_cbr"].mean()),
        "deepc_cbr_gt_0p6_rate": float((result["deepc_cbr"] > 0.6).mean()),
        "measured_open_loop_cbr_gt_0p6_rate": float(
            (result["measured_open_loop_cbr"] > 0.6).mean()
        ),
        "mean_tx_power_dbm": float(result["deepc_tx_power_dbm"].mean()),
        "mean_beacon_interval_s": float(result["deepc_beacon_interval_s"].mean()),
        "optimizer_success_rate": float(result["optimizer_success"].mean()),
    }


def json_safe_args(args: argparse.Namespace) -> dict:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def plot_results(result: pd.DataFrame, out_dir: Path) -> None:
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)

    axes[0].plot(result["time_s"], result["deepc_prr_awareness"], label="classical DeePC surrogate")
    axes[0].plot(
        result["time_s"],
        result["measured_open_loop_prr_awareness"],
        "--",
        label="open-loop",
    )
    axes[0].set_ylabel("PRR")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(result["time_s"], result["deepc_cbr"], label="classical DeePC surrogate")
    axes[1].plot(result["time_s"], result["measured_open_loop_cbr"], "--", label="open-loop")
    axes[1].axhline(0.6, color="k", linestyle=":", linewidth=1)
    axes[1].set_ylabel("CBR")
    axes[1].grid(alpha=0.3)

    axes[2].plot(result["time_s"], result["deepc_pir_s"], label="classical DeePC surrogate")
    axes[2].plot(result["time_s"], result["measured_open_loop_pir_s"], "--", label="open-loop")
    axes[2].set_ylabel("PIR (s)")
    axes[2].grid(alpha=0.3)

    axes[3].plot(result["time_s"], result["deepc_tx_power_dbm"], label="Tx power")
    axes[3].set_ylabel("P (dBm)")
    axes[3].grid(alpha=0.3)
    ax2 = axes[3].twinx()
    ax2.plot(result["time_s"], result["deepc_beacon_interval_s"], color="tab:orange", label="Tb")
    ax2.set_ylabel("Tb (s)")
    axes[3].set_xlabel("Time (s)")

    fig.tight_layout()
    fig.savefig(plots / "closed_loop_timeseries.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = load_json(args.dataset_dir / "dataset_summary.json")
    scaler = load_json(args.dataset_dir / "scaler.json")
    input_cols = summary["input_cols"]
    output_cols = summary["output_cols"]
    context_cols = summary["context_cols"]
    args.past_horizon = args.past_horizon or int(summary["past_horizon_samples"])
    args.control_horizon = args.control_horizon or int(summary["future_horizon_samples"])

    with np.load(args.dataset_dir / "hankel_train_normalized.npz") as loaded:
        hankel = {key: loaded[key] for key in loaded.files}
    required = {"u_p", "u_f", "y_p", "y_f", "d_p", "d_f"}
    missing = required.difference(hankel)
    if missing:
        raise ValueError(f"Missing Hankel blocks: {sorted(missing)}")

    nu = len(input_cols)
    ny = len(output_cols)
    nd = len(context_cols)
    expected = {
        "u_p": args.past_horizon * nu,
        "u_f": args.control_horizon * nu,
        "y_p": args.past_horizon * ny,
        "y_f": args.control_horizon * ny,
        "d_p": args.past_horizon * nd,
        "d_f": args.control_horizon * nd,
    }
    for key, rows in expected.items():
        if hankel[key].shape[0] != rows:
            raise ValueError(f"{key} has {hankel[key].shape[0]} rows, expected {rows}")

    test_norm = pd.read_csv(args.dataset_dir / "deepc_test_normalized.csv")
    test_phys = pd.read_csv(args.dataset_dir / "deepc_test.csv")
    u_ini, y_ini, d_ini = initial_histories(
        test_norm, input_cols, output_cols, context_cols, args.past_horizon
    )
    prev_u_norm = u_ini[-nu:]
    max_steps = min(args.rollout_steps, len(test_norm) - args.past_horizon - args.control_horizon)
    if max_steps <= 0:
        raise ValueError("Test dataset is too short for requested rollout")

    rows = []
    optimizer_log = []
    for k in range(max_steps):
        stage0 = args.past_horizon + k
        d_future = row_flat(
            table_to_array(test_norm.iloc[stage0 : stage0 + args.control_horizon], context_cols)
        )
        selected_cols = select_hankel_columns(
            hankel, u_ini, y_ini, d_ini, d_future, args.max_hankel_cols
        )
        solved = solve_deepc_step(
            hankel,
            selected_cols,
            u_ini,
            y_ini,
            d_ini,
            d_future,
            prev_u_norm,
            args,
            scaler,
            input_cols,
            output_cols,
        )
        u_apply_norm = solved["u_future_norm"][:nu]
        y_pred_norm = solved["y_future_norm"][:ny]
        u_apply = denormalize_rows(u_apply_norm.reshape(1, -1), input_cols, scaler)[0]
        y_pred = denormalize_rows(y_pred_norm.reshape(1, -1), output_cols, scaler)[0]
        measured = table_to_array(test_phys.iloc[[stage0]], output_cols)[0]
        context = table_to_array(test_phys.iloc[[stage0]], context_cols)[0]

        rows.append(
            [
                float(test_phys["time_s"].iloc[stage0]),
                float(u_apply[0]),
                float(u_apply[1]),
                float(y_pred[0]),
                float(y_pred[1]),
                float(y_pred[2]),
                float(measured[0]),
                float(measured[1]),
                float(measured[2]),
                *[float(v) for v in context],
                int(solved["status"]),
                int(solved["success"]),
                float(solved["objective"]),
                int(solved["selected_cols"]),
                float(solved["sum_g"]),
                float(solved["g_norm"]),
            ]
        )
        optimizer_log.append(
            {
                "step": int(k),
                "time_s": float(test_phys["time_s"].iloc[stage0]),
                "status": int(solved["status"]),
                "success": bool(solved["success"]),
                "message": solved["message"],
                "objective": float(solved["objective"]),
                "selected_cols": int(solved["selected_cols"]),
                "sum_g": float(solved["sum_g"]),
                "g_norm": float(solved["g_norm"]),
            }
        )

        d_now_norm = table_to_array(test_norm.iloc[[stage0]], context_cols)[0]
        u_ini = update_history(u_ini, nu, u_apply_norm)
        y_ini = update_history(y_ini, ny, y_pred_norm)
        d_ini = update_history(d_ini, nd, d_now_norm)
        prev_u_norm = u_apply_norm

    columns = [
        "time_s",
        "deepc_tx_power_dbm",
        "deepc_beacon_interval_s",
        "deepc_prr_awareness",
        "deepc_pir_s",
        "deepc_cbr",
        "measured_open_loop_prr_awareness",
        "measured_open_loop_pir_s",
        "measured_open_loop_cbr",
        *context_cols,
        "optimizer_status",
        "optimizer_success",
        "optimizer_objective",
        "selected_hankel_cols",
        "sum_g",
        "g_norm",
    ]
    result = pd.DataFrame(rows, columns=columns)
    result.to_csv(args.out_dir / "closed_loop_rollout.csv", index=False)
    result[["time_s", "deepc_tx_power_dbm", "deepc_beacon_interval_s"]].to_csv(
        args.out_dir / "closed_loop_control_schedule.csv", index=False
    )
    with (args.out_dir / "optimizer_log.json").open("w") as f:
        json.dump(optimizer_log, f, indent=2)

    metrics = {
        "args": json_safe_args(args),
        "input_cols": input_cols,
        "output_cols": output_cols,
        "context_cols": context_cols,
        "hankel_shapes": {key: list(value.shape) for key, value in hankel.items()},
        "formulation": {
            "decision_variable": "g",
            "past_equations": "soft penalties on U_p g, Y_p g, D_p g",
            "future_context": "soft penalty on D_f g matching held-out context",
            "future_trajectory": "u_f = U_f g, y_f = Y_f g",
            "input_bounds": "hard linear constraints on U_f g",
            "affine_constraint": "sum(g)=1" if args.sum_to_one else "disabled",
        },
        "metrics": compute_metrics(result),
    }
    with (args.out_dir / "closed_loop_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    plot_results(result, args.out_dir)

    print(f"Saved classical DeePC artifacts to: {args.out_dir}")
    print(json.dumps(metrics["metrics"], indent=2))


if __name__ == "__main__":
    main()
