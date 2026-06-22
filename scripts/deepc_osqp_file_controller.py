#!/usr/bin/env python3
"""Serve Tx-power-only DeePC controls through the ns-3 JSON file bridge."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import cvxpy as cp
import numpy as np


INPUT_COLS = ["tx_power_dbm"]
OUTPUT_COLS = ["prr_awareness", "pir_s", "cbr"]
CONTEXT_COLS = ["active_vehicle_count_core"]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--poll-interval", type=float, default=0.02)
    parser.add_argument("--idle-timeout", type=float, default=1200.0)
    parser.add_argument("--max-steps", type=int, default=0, help="Zero serves indefinitely.")
    parser.add_argument("--prr-reference", type=float, default=0.90)
    parser.add_argument("--pir-reference", type=float, default=0.10)
    parser.add_argument("--cbr-reference", type=float, default=0.35)
    parser.add_argument("--lambda-y", type=float, default=0.001)
    parser.add_argument("--lambda-g", type=float, default= 10.0)
    parser.add_argument("--min-power", type=float, default=10.0)
    parser.add_argument("--max-power", type=float, default=23.0)
    parser.add_argument("--residual-tolerance", type=float, default=1e-5)
    parser.add_argument("--q-prr", type=float, default=40.0)
    parser.add_argument("--q-pir", type=float, default=3.0)
    parser.add_argument("--q-cbr", type=float, default=8.0)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    os.replace(temp, path)


class TxPowerDeepc:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        with (args.dataset_dir / "dataset_summary.json").open() as handle:
            self.summary = json.load(handle)
        with (args.dataset_dir / "scaler.json").open() as handle:
            self.scaler = json.load(handle)
        hankel = np.load(args.dataset_dir / "hankel_train_normalized.npz")

        if self.summary["input_cols"] != INPUT_COLS:
            raise ValueError(f"Expected input columns {INPUT_COLS}, got {self.summary['input_cols']}")
        if self.summary["output_cols"] != OUTPUT_COLS:
            raise ValueError(
                f"Expected output columns {OUTPUT_COLS}, got {self.summary['output_cols']}"
            )
        self.context_cols = list(self.summary.get("context_cols") or [])
        if self.context_cols not in ([], CONTEXT_COLS):
            raise ValueError(
                f"Expected no context columns or {CONTEXT_COLS}, got {self.context_cols}"
            )
        self.use_context = self.context_cols == CONTEXT_COLS
        if self.summary["past_horizon_samples"] != 20:
            raise ValueError("Expected Tini=20")
        if self.summary["future_horizon_samples"] != 10:
            raise ValueError("Expected N=10")

        self.up = hankel["u_p"]
        self.uf = hankel["u_f"]
        self.yp = hankel["y_p"]
        self.yf = hankel["y_f"]
        self.dp = hankel["d_p"] if self.use_context else None
        self.df = hankel["d_f"] if self.use_context else None
        expected = {
            "Up": (20, 307),
            "Uf": (10, 307),
            "Yp": (60, 307),
            "Yf": (30, 307),
        }
        actual = {
            "Up": self.up.shape,
            "Uf": self.uf.shape,
            "Yp": self.yp.shape,
            "Yf": self.yf.shape,
        }
        if self.use_context:
            expected["Dp"] = (20, 307)
            expected["Df"] = (10, 307)
            actual["Dp"] = self.dp.shape
            actual["Df"] = self.df.shape
        if actual != expected:
            raise ValueError(f"Unexpected Hankel dimensions: expected {expected}, got {actual}")

        self.sample_time = float(self.summary["sample_time_s"])
        self.u_mean = self.scaler["mean"]["tx_power_dbm"]
        self.u_std = self.scaler["std"]["tx_power_dbm"]
        self.y_mean = np.array([self.scaler["mean"][name] for name in OUTPUT_COLS])
        self.y_std = np.array([self.scaler["std"][name] for name in OUTPUT_COLS])
        if self.use_context:
            self.d_mean = self.scaler["mean"]["active_vehicle_count_core"]
            self.d_std = self.scaler["std"]["active_vehicle_count_core"]
        else:
            self.d_mean = 0.0
            self.d_std = 1.0

        n_columns = self.up.shape[1]
        self.g = cp.Variable(n_columns)
        self.u_ini = cp.Parameter(self.up.shape[0])
        self.y_ini = cp.Parameter(self.yp.shape[0])
        self.d_ini = cp.Parameter(self.dp.shape[0]) if self.use_context else None
        self.d_future = cp.Parameter(self.df.shape[0]) if self.use_context else None
        self.y_ref = cp.Parameter(self.yf.shape[0])
        future_u = self.uf @ self.g
        future_y = self.yf @ self.g

        q = np.tile(np.array([args.q_prr, args.q_pir, args.q_cbr]), 10)
        r = np.full(10, 0.002)
        objective = (
            cp.sum(cp.multiply(r, cp.square(future_u)))
            + cp.sum(cp.multiply(q, cp.square(future_y - self.y_ref)))
            + args.lambda_g * cp.sum_squares(self.g)
        )
        min_power_norm = (args.min_power - self.u_mean) / self.u_std
        max_power_norm = (args.max_power - self.u_mean) / self.u_std
        constraints = [
            self.up @ self.g == self.u_ini,
            self.yp @ self.g == self.y_ini,
            future_u >= min_power_norm,
            future_u <= max_power_norm,
        ]
        if self.use_context:
            constraints.extend(
                [
                    self.dp @ self.g == self.d_ini,
                    self.df @ self.g == self.d_future,
                ]
            )
        self.problem = cp.Problem(cp.Minimize(objective), constraints)
        reference = np.array(
            [args.prr_reference, args.pir_reference, args.cbr_reference], dtype=float
        )
        self.y_ref.value = np.tile((reference - self.y_mean) / self.y_std, 10)

    def normalize_u(self, values: np.ndarray) -> np.ndarray:
        return (values - self.u_mean) / self.u_std

    def normalize_y(self, values: np.ndarray) -> np.ndarray:
        matrix = values.reshape(-1, 3)
        return ((matrix - self.y_mean) / self.y_std).reshape(-1)

    def denormalize_u(self, values: np.ndarray) -> np.ndarray:
        return values * self.u_std + self.u_mean

    def denormalize_y(self, values: np.ndarray) -> np.ndarray:
        matrix = values.reshape(-1, 3)
        return matrix * self.y_std + self.y_mean

    def normalize_d(self, values: np.ndarray) -> np.ndarray:
        return (values - self.d_mean) / self.d_std

    def denormalize_d(self, values: np.ndarray) -> np.ndarray:
        return values * self.d_std + self.d_mean

    def solve(self, request: dict) -> dict:
        previous_power = float(request["previous_u"][0])
        start = time.perf_counter()
        status = "controller_error"
        objective = None
        max_up_residual = None
        max_yp_residual = None
        max_dp_residual = None
        max_df_residual = None
        future_power: list[float] = []
        future_y = np.empty((0, 3))
        d_future_physical: list[float] = []
        error = None

        try:
            if request["past_horizon"] != 20 or request["future_horizon"] != 10:
                raise ValueError("Request horizons do not match Tini=20, N=10")
            if request["input_cols"] != INPUT_COLS or request["output_cols"] != OUTPUT_COLS:
                raise ValueError("Request signal columns do not match the controller dataset")
            u_ini = np.asarray(request["u_ini"], dtype=float)
            y_ini = np.asarray(request["y_ini"], dtype=float)
            if u_ini.shape != (20,) or y_ini.shape != (60,):
                raise ValueError(f"Bad history shapes: u_ini={u_ini.shape}, y_ini={y_ini.shape}")
            if self.use_context:
                if request.get("context_cols") != CONTEXT_COLS:
                    raise ValueError("Request context columns do not match the controller dataset")
                if request.get("density_forecast_mode", "hold_last") != "hold_last":
                    raise ValueError("Only density_forecast_mode=hold_last is supported")
                d_ini = np.asarray(request["d_ini"], dtype=float)
                if d_ini.shape != (20,):
                    raise ValueError(f"Bad density history shape: d_ini={d_ini.shape}")
                d_future_physical = np.full(10, d_ini[-1], dtype=float).tolist()
                self.d_ini.value = self.normalize_d(d_ini)
                self.d_future.value = self.normalize_d(np.asarray(d_future_physical, dtype=float))

            self.u_ini.value = self.normalize_u(u_ini)
            self.y_ini.value = self.normalize_y(y_ini)
            self.problem.solve(
                solver=cp.OSQP,
                warm_start=True,
                eps_abs=1e-6,
                eps_rel=1e-6,
                max_iter=100000,
                polishing=True,
                verbose=False,
            )
            status = str(self.problem.status)
            if status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or self.g.value is None:
                raise RuntimeError(f"OSQP returned {status}")

            g_value = np.asarray(self.g.value).reshape(-1)
            future_u_norm = self.uf @ g_value
            future_y_norm = self.yf @ g_value
            future_power_array = self.denormalize_u(future_u_norm)
            future_y = self.denormalize_y(future_y_norm)
            max_up_residual = float(np.max(np.abs(self.up @ g_value - self.u_ini.value)))
            max_yp_residual = float(np.max(np.abs(self.yp @ g_value - self.y_ini.value)))
            if self.use_context:
                max_dp_residual = float(np.max(np.abs(self.dp @ g_value - self.d_ini.value)))
                max_df_residual = float(np.max(np.abs(self.df @ g_value - self.d_future.value)))
            objective = float(self.problem.value)

            if not np.isfinite(future_power_array).all() or not np.isfinite(future_y).all():
                raise RuntimeError("OSQP returned non-finite predictions")
            if (
                future_power_array.min() < self.args.min_power - 1e-6
                or future_power_array.max() > self.args.max_power + 1e-6
            ):
                raise RuntimeError("Predicted power violates configured bounds")
            if (
                max_up_residual > self.args.residual_tolerance
                or max_yp_residual > self.args.residual_tolerance
                or (
                    self.use_context
                    and (
                        max_dp_residual > self.args.residual_tolerance
                        or max_df_residual > self.args.residual_tolerance
                    )
                )
            ):
                raise RuntimeError("DeePC equality residual exceeds tolerance")
            future_power = future_power_array.tolist()
        except Exception as exc:
            error = str(exc)

        solve_time = time.perf_counter() - start
        success = bool(future_power)
        response = {
            "step": int(request["step"]),
            "time_s": float(request["time_s"]),
            "success": success,
            "solver_status": status,
            "tx_power_dbm": float(future_power[0] if success else previous_power),
            "future_tx_power_dbm": future_power,
            "predicted_prr": float(future_y[0, 0]) if success else None,
            "predicted_pir_s": float(future_y[0, 1]) if success else None,
            "predicted_cbr": float(future_y[0, 2]) if success else None,
            "predicted_prr_horizon": future_y[:, 0].tolist() if success else [],
            "predicted_pir_s_horizon": future_y[:, 1].tolist() if success else [],
            "predicted_cbr_horizon": future_y[:, 2].tolist() if success else [],
            "objective": objective,
            "solve_time_s": solve_time,
            "max_up_residual": max_up_residual,
            "max_yp_residual": max_yp_residual,
            "max_dp_residual": max_dp_residual,
            "max_df_residual": max_df_residual,
            "density_forecast_mode": "hold_last" if self.use_context else "none",
            "density_forecast_active_vehicle_count_core": d_future_physical,
            "error": error,
        }
        return response


def open_logs(bridge_dir: Path) -> tuple[csv.DictWriter, object, csv.DictWriter, object]:
    diagnostic_file = (bridge_dir / "controller_diagnostics.csv").open("w", newline="")
    diagnostic_fields = [
        "step",
        "time_s",
        "success",
        "solver_status",
        "tx_power_dbm",
        "objective",
        "solve_time_s",
        "max_up_residual",
        "max_yp_residual",
        "max_dp_residual",
        "max_df_residual",
        "density_forecast_mode",
        "error",
    ]
    diagnostics = csv.DictWriter(diagnostic_file, fieldnames=diagnostic_fields)
    diagnostics.writeheader()

    prediction_file = (bridge_dir / "predictions.csv").open("w", newline="")
    prediction_fields = [
        "step",
        "request_time_s",
        "horizon_step",
        "predicted_time_s",
        "predicted_tx_power_dbm",
        "predicted_prr",
        "predicted_pir_s",
        "predicted_cbr",
        "forecast_active_vehicle_count_core",
    ]
    predictions = csv.DictWriter(prediction_file, fieldnames=prediction_fields)
    predictions.writeheader()
    return diagnostics, diagnostic_file, predictions, prediction_file


def main() -> None:
    args = parse_args()
    request_dir = args.bridge_dir / "requests"
    response_dir = args.bridge_dir / "responses"
    request_dir.mkdir(parents=True, exist_ok=True)
    response_dir.mkdir(parents=True, exist_ok=True)
    controller = TxPowerDeepc(args)
    diagnostics, diagnostic_file, predictions, prediction_file = open_logs(args.bridge_dir)
    print(
        "Loaded Tx-power DeePC: "
        f"Up={controller.up.shape}, Uf={controller.uf.shape}, "
        f"Yp={controller.yp.shape}, Yf={controller.yf.shape}"
        + (
            f", Dp={controller.dp.shape}, Df={controller.df.shape}"
            if controller.use_context
            else ""
        ),
        flush=True,
    )

    processed: set[Path] = set()
    completed = 0
    last_activity = time.monotonic()
    try:
        while args.max_steps <= 0 or completed < args.max_steps:
            pending = sorted(path for path in request_dir.glob("request_*.json") if path not in processed)
            if not pending:
                if args.idle_timeout > 0 and time.monotonic() - last_activity > args.idle_timeout:
                    print("Controller idle timeout reached.", flush=True)
                    break
                time.sleep(args.poll_interval)
                continue

            for request_path in pending:
                with request_path.open() as handle:
                    request = json.load(handle)
                response = controller.solve(request)
                response_path = response_dir / request_path.name.replace("request_", "response_")
                atomic_json(response_path, response)
                diagnostics.writerow({key: response.get(key) for key in diagnostics.fieldnames})
                diagnostic_file.flush()

                if response["success"]:
                    for index in range(10):
                        predictions.writerow(
                            {
                                "step": response["step"],
                                "request_time_s": response["time_s"],
                                "horizon_step": index + 1,
                                "predicted_time_s": response["time_s"]
                                + (index + 1) * controller.sample_time,
                                "predicted_tx_power_dbm": response["future_tx_power_dbm"][index],
                                "predicted_prr": response["predicted_prr_horizon"][index],
                                "predicted_pir_s": response["predicted_pir_s_horizon"][index],
                                "predicted_cbr": response["predicted_cbr_horizon"][index],
                                "forecast_active_vehicle_count_core": (
                                    response["density_forecast_active_vehicle_count_core"][index]
                                    if response["density_forecast_active_vehicle_count_core"]
                                    else None
                                ),
                            }
                        )
                    prediction_file.flush()
                processed.add(request_path)
                completed += 1
                last_activity = time.monotonic()
                print(
                    f"step={response['step']} status={response['solver_status']} "
                    f"success={int(response['success'])} power={response['tx_power_dbm']:.3f} "
                    f"solve={response['solve_time_s']:.4f}s",
                    flush=True,
                )
    finally:
        diagnostic_file.close()
        prediction_file.close()


if __name__ == "__main__":
    main()
