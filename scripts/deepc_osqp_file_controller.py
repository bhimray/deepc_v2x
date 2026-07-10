#!/usr/bin/env python3
"""Run the 10 Hz Tx-power DeePC controller through the ns-3 file bridge.

This is intentionally written for the final report experiment, not as a
general DeePC framework. The controller optimizes only tx_power_dbm. The beacon
interval stays fixed at 0.1 s and is passed as context together with the recent
active-vehicle count.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import cvxpy as cp
import numpy as np


TINI = 20
HORIZON = 10
TX_POWER_COL = "tx_power_dbm"
OUTPUT_COLS = ["prr_awareness", "pir_s", "cbr"]
CONTEXT_COLS = ["active_vehicle_count_core", "beacon_interval_s"]
FIXED_BEACON_INTERVAL_S = 0.1


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
    parser.add_argument("--q-prr", type=float, default=40.0)
    parser.add_argument("--q-pir", type=float, default=3.0)
    parser.add_argument("--q-cbr", type=float, default=8.0)
    parser.add_argument("--lambda-g", type=float, default=10.0)
    parser.add_argument("--lambda-y", type=float, default=0.001, help=argparse.SUPPRESS)

    parser.add_argument("--min-power", type=float, default=10.0)
    parser.add_argument("--max-power", type=float, default=23.0)
    parser.add_argument("--residual-tolerance", type=float, default=1e-5)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict) -> None:
    """Write a complete response file without exposing a half-written JSON."""
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    os.replace(temp, path)


class TxPowerDeepc:
    """DeePC QP for the final 10 Hz Tx-power experiment."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.summary = self.load_json(args.dataset_dir / "dataset_summary.json")
        self.scaler = self.load_json(args.dataset_dir / "scaler.json")
        hankel = np.load(args.dataset_dir / "hankel_train_normalized.npz")

        self.check_dataset_contract()

        # Hankel blocks are normalized. The measured past must match Up/Yp/Dp;
        # the optimized future comes from Uf/Yf while Df is the context forecast.
        self.up = hankel["u_p"]
        self.uf = hankel["u_f"]
        self.yp = hankel["y_p"]
        self.yf = hankel["y_f"]
        self.dp = hankel["d_p"]
        self.df = hankel["d_f"]
        self.check_hankel_shapes()

        self.sample_time = float(self.summary["sample_time_s"])
        self.u_mean = float(self.scaler["mean"][TX_POWER_COL])
        self.u_std = float(self.scaler["std"][TX_POWER_COL])
        self.y_mean = np.array([self.scaler["mean"][name] for name in OUTPUT_COLS])
        self.y_std = np.array([self.scaler["std"][name] for name in OUTPUT_COLS])
        self.d_mean = np.array([self.scaler["mean"][name] for name in CONTEXT_COLS])
        self.d_std = np.array([self.scaler["std"][name] for name in CONTEXT_COLS])

        self.build_problem()

    @staticmethod
    def load_json(path: Path) -> dict:
        with path.open() as handle:
            return json.load(handle)

    def check_dataset_contract(self) -> None:
        """Stop early if this script is pointed at the wrong dataset."""
        expected = {
            "input_cols": [TX_POWER_COL],
            "output_cols": OUTPUT_COLS,
            "context_cols": CONTEXT_COLS,
            "past_horizon_samples": TINI,
            "future_horizon_samples": HORIZON,
        }
        for key, value in expected.items():
            if self.summary.get(key) != value:
                raise ValueError(f"Expected dataset {key}={value}, got {self.summary.get(key)}")

    def check_hankel_shapes(self) -> None:
        expected = {
            "u_p": (TINI, 307),
            "u_f": (HORIZON, 307),
            "y_p": (TINI * len(OUTPUT_COLS), 307),
            "y_f": (HORIZON * len(OUTPUT_COLS), 307),
            "d_p": (TINI * len(CONTEXT_COLS), 307),
            "d_f": (HORIZON * len(CONTEXT_COLS), 307),
        }
        actual = {
            "u_p": self.up.shape,
            "u_f": self.uf.shape,
            "y_p": self.yp.shape,
            "y_f": self.yf.shape,
            "d_p": self.dp.shape,
            "d_f": self.df.shape,
        }
        if actual != expected:
            raise ValueError(f"Unexpected Hankel shapes: expected {expected}, got {actual}")

    def build_problem(self) -> None:
        """Build the CVXPY problem once; request data only updates parameters."""
        n_columns = self.up.shape[1]
        self.g = cp.Variable(n_columns)
        self.u_ini = cp.Parameter(TINI)
        self.y_ini = cp.Parameter(TINI * len(OUTPUT_COLS))
        self.d_ini = cp.Parameter(TINI * len(CONTEXT_COLS))
        self.d_future = cp.Parameter(HORIZON * len(CONTEXT_COLS))
        self.y_ref = cp.Parameter(HORIZON * len(OUTPUT_COLS))

        future_u = self.uf @ self.g
        future_y = self.yf @ self.g
        output_weights = np.tile(
            np.array([self.args.q_prr, self.args.q_pir, self.args.q_cbr]), HORIZON
        )
        power_weight = np.full(HORIZON, 0.002)
        min_power_norm = self.normalize_power(np.array([self.args.min_power]))[0]
        max_power_norm = self.normalize_power(np.array([self.args.max_power]))[0]

        objective = (
            cp.sum(cp.multiply(output_weights, cp.square(future_y - self.y_ref)))
            + cp.sum(cp.multiply(power_weight, cp.square(future_u)))
            + self.args.lambda_g * cp.sum_squares(self.g)
        )
        constraints = [
            self.up @ self.g == self.u_ini,
            self.yp @ self.g == self.y_ini,
            self.dp @ self.g == self.d_ini,
            self.df @ self.g == self.d_future,
            future_u >= min_power_norm,
            future_u <= max_power_norm,
        ]
        self.problem = cp.Problem(cp.Minimize(objective), constraints)

        reference = np.array(
            [self.args.prr_reference, self.args.pir_reference, self.args.cbr_reference]
        )
        self.y_ref.value = np.tile(self.normalize_outputs(reference.reshape(1, 3)), HORIZON)

    def normalize_power(self, values: np.ndarray) -> np.ndarray:
        return (values - self.u_mean) / self.u_std

    def denormalize_power(self, values: np.ndarray) -> np.ndarray:
        return values * self.u_std + self.u_mean

    def normalize_outputs(self, values: np.ndarray) -> np.ndarray:
        return ((values.reshape(-1, 3) - self.y_mean) / self.y_std).reshape(-1)

    def denormalize_outputs(self, values: np.ndarray) -> np.ndarray:
        return values.reshape(-1, 3) * self.y_std + self.y_mean

    def normalize_context(self, values: np.ndarray) -> np.ndarray:
        return ((values.reshape(-1, 2) - self.d_mean) / self.d_std).reshape(-1)

    def forecast_context(self, context_history: np.ndarray) -> np.ndarray:
        """Hold density constant and keep the beacon interval fixed at 0.1 s."""
        history = context_history.reshape(TINI, len(CONTEXT_COLS))
        last_active_count = history[-1, 0]
        future = np.column_stack(
            [
                np.full(HORIZON, last_active_count),
                np.full(HORIZON, FIXED_BEACON_INTERVAL_S),
            ]
        )
        return future.reshape(-1)

    def solve(self, request: dict) -> dict:
        """Solve one ns-3 request and return the response JSON payload."""
        start = time.perf_counter()
        previous_power = float(request["previous_u"][0])
        result = {
            "success": False,
            "solver_status": "controller_error",
            "objective": None,
            "future_power": [],
            "future_y": np.empty((0, 3)),
            "future_context": np.empty((0, 2)),
            "max_up_residual": None,
            "max_yp_residual": None,
            "max_dp_residual": None,
            "max_df_residual": None,
            "error": None,
        }

        try:
            u_ini, y_ini, d_ini = self.read_request_histories(request)
            d_future = self.forecast_context(d_ini)

            self.u_ini.value = self.normalize_power(u_ini)
            self.y_ini.value = self.normalize_outputs(y_ini)
            self.d_ini.value = self.normalize_context(d_ini)
            self.d_future.value = self.normalize_context(d_future)

            self.problem.solve(
                solver=cp.OSQP,
                warm_start=True,
                eps_abs=1e-6,
                eps_rel=1e-6,
                max_iter=100000,
                polishing=True,
                verbose=False,
            )
            result["solver_status"] = str(self.problem.status)
            if result["solver_status"] not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
                raise RuntimeError(f"OSQP returned {result['solver_status']}")
            if self.g.value is None:
                raise RuntimeError("OSQP did not return a DeePC coefficient vector")

            g_value = np.asarray(self.g.value).reshape(-1)
            future_power = self.denormalize_power(self.uf @ g_value)
            future_y = self.denormalize_outputs(self.yf @ g_value)
            future_context = d_future.reshape(HORIZON, len(CONTEXT_COLS))

            result.update(
                {
                    "success": True,
                    "objective": float(self.problem.value),
                    "future_power": future_power.tolist(),
                    "future_y": future_y,
                    "future_context": future_context,
                    "max_up_residual": self.max_abs(self.up @ g_value - self.u_ini.value),
                    "max_yp_residual": self.max_abs(self.yp @ g_value - self.y_ini.value),
                    "max_dp_residual": self.max_abs(self.dp @ g_value - self.d_ini.value),
                    "max_df_residual": self.max_abs(self.df @ g_value - self.d_future.value),
                }
            )
            self.check_solution(result)
        except Exception as exc:
            result["success"] = False
            result["future_power"] = []
            result["future_y"] = np.empty((0, 3))
            result["future_context"] = np.empty((0, 2))
            result["error"] = str(exc)

        return self.make_response(request, result, previous_power, time.perf_counter() - start)

    def read_request_histories(self, request: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if request["past_horizon"] != TINI or request["future_horizon"] != HORIZON:
            raise ValueError("Request horizons do not match Tini=20, N=10")
        if request["input_cols"] != [TX_POWER_COL] or request["output_cols"] != OUTPUT_COLS:
            raise ValueError("Request input/output columns do not match the controller")
        if request.get("context_cols") != CONTEXT_COLS:
            raise ValueError("Request context columns do not match the controller")

        u_ini = np.asarray(request["u_ini"], dtype=float)
        y_ini = np.asarray(request["y_ini"], dtype=float)
        d_ini = np.asarray(request["d_ini"], dtype=float)
        expected_shapes = {
            "u_ini": (TINI,),
            "y_ini": (TINI * len(OUTPUT_COLS),),
            "d_ini": (TINI * len(CONTEXT_COLS),),
        }
        actual_shapes = {"u_ini": u_ini.shape, "y_ini": y_ini.shape, "d_ini": d_ini.shape}
        if actual_shapes != expected_shapes:
            raise ValueError(f"Bad request shapes: expected {expected_shapes}, got {actual_shapes}")
        return u_ini, y_ini, d_ini

    @staticmethod
    def max_abs(values: np.ndarray) -> float:
        return float(np.max(np.abs(values)))

    def check_solution(self, result: dict) -> None:
        future_power = np.asarray(result["future_power"], dtype=float)
        future_y = np.asarray(result["future_y"], dtype=float)
        if not np.isfinite(future_power).all() or not np.isfinite(future_y).all():
            raise RuntimeError("OSQP returned non-finite values")
        if future_power.min() < self.args.min_power - 1e-6:
            raise RuntimeError("Predicted Tx power is below the configured minimum")
        if future_power.max() > self.args.max_power + 1e-6:
            raise RuntimeError("Predicted Tx power is above the configured maximum")

        residuals = [
            result["max_up_residual"],
            result["max_yp_residual"],
            result["max_dp_residual"],
            result["max_df_residual"],
        ]
        if max(residuals) > self.args.residual_tolerance:
            raise RuntimeError("DeePC equality residual exceeds tolerance")

    def make_response(
        self, request: dict, result: dict, previous_power: float, solve_time: float
    ) -> dict:
        success = bool(result["success"])
        future_y = result["future_y"]
        future_context = result["future_context"]
        future_power = result["future_power"]

        return {
            "step": int(request["step"]),
            "time_s": float(request["time_s"]),
            "success": success,
            "solver_status": result["solver_status"],
            "tx_power_dbm": float(future_power[0] if success else previous_power),
            "future_tx_power_dbm": future_power,
            "predicted_prr": float(future_y[0, 0]) if success else None,
            "predicted_pir_s": float(future_y[0, 1]) if success else None,
            "predicted_cbr": float(future_y[0, 2]) if success else None,
            "predicted_prr_horizon": future_y[:, 0].tolist() if success else [],
            "predicted_pir_s_horizon": future_y[:, 1].tolist() if success else [],
            "predicted_cbr_horizon": future_y[:, 2].tolist() if success else [],
            "objective": result["objective"],
            "solve_time_s": solve_time,
            "max_up_residual": result["max_up_residual"],
            "max_yp_residual": result["max_yp_residual"],
            "max_dp_residual": result["max_dp_residual"],
            "max_df_residual": result["max_df_residual"],
            "density_forecast_mode": "hold_last",
            "density_forecast_active_vehicle_count_core": (
                future_context[:, 0].tolist() if success else []
            ),
            "beacon_interval_forecast_s": future_context[:, 1].tolist() if success else [],
            "error": result["error"],
        }


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
        "forecast_beacon_interval_s",
    ]
    predictions = csv.DictWriter(prediction_file, fieldnames=prediction_fields)
    predictions.writeheader()
    return diagnostics, diagnostic_file, predictions, prediction_file


def log_prediction_horizon(
    predictions: csv.DictWriter, response: dict, sample_time: float
) -> None:
    for index in range(HORIZON):
        predictions.writerow(
            {
                "step": response["step"],
                "request_time_s": response["time_s"],
                "horizon_step": index + 1,
                "predicted_time_s": response["time_s"] + (index + 1) * sample_time,
                "predicted_tx_power_dbm": response["future_tx_power_dbm"][index],
                "predicted_prr": response["predicted_prr_horizon"][index],
                "predicted_pir_s": response["predicted_pir_s_horizon"][index],
                "predicted_cbr": response["predicted_cbr_horizon"][index],
                "forecast_active_vehicle_count_core": response[
                    "density_forecast_active_vehicle_count_core"
                ][index],
                "forecast_beacon_interval_s": response["beacon_interval_forecast_s"][index],
            }
        )


def main() -> None:
    args = parse_args()
    request_dir = args.bridge_dir / "requests"
    response_dir = args.bridge_dir / "responses"
    request_dir.mkdir(parents=True, exist_ok=True)
    response_dir.mkdir(parents=True, exist_ok=True)

    controller = TxPowerDeepc(args)
    diagnostics, diagnostic_file, predictions, prediction_file = open_logs(args.bridge_dir)
    print(
        "Loaded 10 Hz Tx-power DeePC: "
        f"Up={controller.up.shape}, Uf={controller.uf.shape}, "
        f"Yp={controller.yp.shape}, Yf={controller.yf.shape}, "
        f"Dp={controller.dp.shape}, Df={controller.df.shape}",
        flush=True,
    )

    processed: set[Path] = set()
    completed = 0
    last_activity = time.monotonic()
    try:
        while args.max_steps <= 0 or completed < args.max_steps:
            pending = sorted(
                path for path in request_dir.glob("request_*.json") if path not in processed
            )
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
                    log_prediction_horizon(predictions, response, controller.sample_time)
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
