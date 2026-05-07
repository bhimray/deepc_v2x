#!/usr/bin/env python3
"""Plot ns-3/MATLAB DeePC bridge run results.

Expected run folder contents:
  kpi_timeseries.csv
  applied_controls.csv
  controller_solve_times.csv
  bridge_requests/      optional
  bridge_responses/     optional
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a DeePC bridge run folder.")
    parser.add_argument("run_dir", type=Path, help="Run output directory.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Defaults to RUN_DIR/plots.")
    return parser.parse_args()


def load_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def metrics(kpi: pd.DataFrame, applied: pd.DataFrame | None, timing: pd.DataFrame | None) -> dict:
    result = {
        "rows": int(len(kpi)),
        "time_range_s": [float(kpi["time_s"].min()), float(kpi["time_s"].max())],
        "mean_prr_150m": float(kpi["prr_150m"].mean()),
        "mean_pir_s": float(kpi["pir_s"].mean()),
        "mean_cbr": float(kpi["cbr"].mean()),
        "cbr_gt_0p6_rate": float((kpi["cbr"] > 0.6).mean()),
        "mean_density_veh_per_km_core": float(kpi["density_veh_per_km_core"].mean()),
    }
    if applied is not None and len(applied):
        result.update(
            {
                "control_updates": int(len(applied)),
                "controller_success_rate": float(applied["response_success"].mean()),
                "mean_applied_tx_power_dbm": float(applied["tx_power_dbm"].mean()),
                "mean_applied_beacon_interval_s": float(applied["beacon_interval_s"].mean()),
            }
        )
    if timing is not None and len(timing):
        result.update(
            {
                "mean_bridge_wait_time_s": float(timing["wait_time_s"].mean()),
                "max_bridge_wait_time_s": float(timing["wait_time_s"].max()),
                "mean_controller_solve_time_s": float(timing["controller_solve_time_s"].mean()),
                "max_controller_solve_time_s": float(timing["controller_solve_time_s"].max()),
            }
        )
    return result


def plot_kpis(kpi: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(kpi["time_s"], kpi["prr_150m"], color="tab:green")
    axes[0].set_ylabel("PRR 150m")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].grid(alpha=0.3)

    axes[1].plot(kpi["time_s"], kpi["cbr"], color="tab:red")
    axes[1].axhline(0.6, color="k", linestyle=":", linewidth=1)
    axes[1].set_ylabel("CBR")
    axes[1].grid(alpha=0.3)

    axes[2].plot(kpi["time_s"], kpi["pir_s"], color="tab:blue")
    axes[2].set_ylabel("PIR (s)")
    axes[2].grid(alpha=0.3)

    axes[3].plot(kpi["time_s"], kpi["density_veh_per_km_core"], color="tab:purple")
    axes[3].set_ylabel("Density veh/km")
    axes[3].set_xlabel("Time (s)")
    axes[3].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "kpi_timeseries.png", dpi=180)
    plt.close(fig)


def plot_controls(kpi: pd.DataFrame, applied: pd.DataFrame | None, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(kpi["time_s"], kpi["tx_power_dbm"], label="KPI sampled input")
    if applied is not None and len(applied):
        axes[0].step(
            applied["time_s"],
            applied["tx_power_dbm"],
            where="post",
            linestyle="--",
            label="Controller response",
        )
    axes[0].set_ylabel("Tx power (dBm)")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(kpi["time_s"], kpi["beacon_interval_s"], label="KPI sampled input")
    if applied is not None and len(applied):
        axes[1].step(
            applied["time_s"],
            applied["beacon_interval_s"],
            where="post",
            linestyle="--",
            label="Controller response",
        )
    axes[1].set_ylabel("Beacon interval (s)")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "controls.png", dpi=180)
    plt.close(fig)


def plot_controller(timing: pd.DataFrame | None, out_dir: Path) -> None:
    if timing is None or not len(timing):
        return
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(timing["time_s"], timing["controller_solve_time_s"], marker=".")
    axes[0].set_ylabel("Solve time (s)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(timing["time_s"], timing["wait_time_s"], marker=".", color="tab:orange")
    axes[1].set_ylabel("ns-3 wait (s)")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "controller_timing.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    out_dir = args.out_dir or run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    kpi_path = run_dir / "kpi_timeseries.csv"
    if not kpi_path.exists():
        raise FileNotFoundError(f"Missing KPI file: {kpi_path}")
    kpi = pd.read_csv(kpi_path)
    applied = load_optional_csv(run_dir / "applied_controls.csv")
    timing = load_optional_csv(run_dir / "controller_solve_times.csv")

    plot_kpis(kpi, out_dir)
    plot_controls(kpi, applied, out_dir)
    plot_controller(timing, out_dir)

    summary = metrics(kpi, applied, timing)
    with (out_dir / "summary_metrics.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved plots to {out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
