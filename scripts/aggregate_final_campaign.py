#!/usr/bin/env python3
"""Aggregate run metrics for a modular final-results campaign."""

from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_CAMPAIGN_DIR = Path("data/output/final/ieee_250veh_deepc_campaign_01")
METHOD_ORDER = ["fixed_10hz", "fixed_5hz", "deepc_count_tb0p1_prr_only"]
EXPECTED_RUNS = [101, 102, 103, 104, 105]
METHOD_LABELS = {
    "fixed_10hz": "Fixed 10 Hz, 20 dBm",
    "fixed_5hz": "Fixed 5 Hz, 20 dBm",
    "deepc_count_tb0p1_prr_only": "DeePC Tx-power control",
    "threshold_dcc": "Threshold DCC",
}
MAIN_TABLE_METRICS = [
    "mean_prr_awareness",
    "mean_pir_s",
    "mean_cbr",
    "cbr_gt_0p6_rate",
    "controller_success_rate",
    "mean_controller_solve_time_s",
]
METRICS = [
    "mean_prr_awareness",
    "beacon_error_rate",
    "mean_pir_s",
    "mean_cbr",
    "cbr_gt_0p6_rate",
    "p95_delay_s",
    "mean_tx_power_dbm",
    "mean_beacon_interval_s",
    "mean_active_vehicle_count_core",
    "mean_density_veh_per_km_core",
    "mean_controller_solve_time_s",
    "controller_success_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate final campaign run metrics.")
    parser.add_argument("--campaign-dir", type=Path, default=DEFAULT_CAMPAIGN_DIR)
    parser.add_argument("--methods", nargs="*", default=METHOD_ORDER)
    parser.add_argument(
        "--expected-runs",
        default=",".join(str(run) for run in EXPECTED_RUNS),
        help="Comma-separated run ids required for the paper-ready matched set.",
    )
    return parser.parse_args()


def add_prr_alias(kpi: pd.DataFrame) -> pd.DataFrame:
    if "prr_awareness" not in kpi and "prr_150m" in kpi:
        kpi = kpi.copy()
        kpi["prr_awareness"] = kpi["prr_150m"]
    return kpi


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def eval_window(metadata: dict, kpi: pd.DataFrame) -> tuple[float, float | None]:
    start = float(metadata.get("evaluation_start_s", metadata.get("warmup_s", 0.0)))
    if "evaluation_end_s" in metadata:
        return start, float(metadata["evaluation_end_s"])
    cooldown = float(metadata.get("cooldown_s", 0.0))
    end = None
    if cooldown > 0.0 and not kpi.empty:
        end = float(kpi["time_s"].max()) - cooldown
    return start, end


def filter_eval(df: pd.DataFrame, start: float, end: float | None) -> pd.DataFrame:
    out = df[df["time_s"] >= start].copy()
    if end is not None:
        out = out[out["time_s"] <= end].copy()
    return out


def optional_csv(paths: list[Path]) -> pd.DataFrame | None:
    for path in paths:
        if path.exists():
            return pd.read_csv(path)
    return None


def run_number(run_dir: Path) -> int:
    name = run_dir.name
    if name.startswith("run_"):
        return int(name.split("_", 1)[1])
    return int(name)


def compute_run_metrics(method: str, run_dir: Path) -> dict | None:
    kpi_path = run_dir / "kpi_timeseries.csv"
    if not kpi_path.exists():
        return None

    metadata = load_json(run_dir / "kpi_timeseries_metadata.json")
    run_manifest = load_json(run_dir / "run_manifest.json")
    kpi = add_prr_alias(pd.read_csv(kpi_path))
    start, end = eval_window(metadata, kpi)
    kpi_eval = filter_eval(kpi, start, end)
    if kpi_eval.empty:
        print(f"Skipping incomplete run with no evaluation-window KPI rows: {kpi_path}")
        return None

    row = {
        "method": method,
        "run": run_number(run_dir),
        "seed": run_manifest.get("seed", metadata.get("seed")),
        "rows": int(len(kpi_eval)),
        "sample_first_time_s": float(kpi["time_s"].min()),
        "sample_last_time_s": float(kpi["time_s"].max()),
        "eval_start_s": start,
        "eval_end_s": end,
        "mean_prr_awareness": float(kpi_eval["prr_awareness"].mean()),
        "beacon_error_rate": float((1.0 - kpi_eval["prr_awareness"]).mean()),
        "mean_pir_s": float(kpi_eval["pir_s"].mean()),
        "mean_cbr": float(kpi_eval["cbr"].mean()),
        "cbr_gt_0p6_rate": float((kpi_eval["cbr"] > 0.6).mean()),
        "mean_tx_power_dbm": float(kpi_eval["tx_power_dbm"].mean()),
        "mean_beacon_interval_s": float(kpi_eval["beacon_interval_s"].mean()),
    }
    if "active_vehicle_count_core" in kpi_eval:
        row["mean_active_vehicle_count_core"] = float(kpi_eval["active_vehicle_count_core"].mean())
    if "density_veh_per_km_core" in kpi_eval:
        row["mean_density_veh_per_km_core"] = float(kpi_eval["density_veh_per_km_core"].mean())

    rx_path = run_dir / "kpi_timeseries_rx_packet_log.csv"
    if rx_path.exists():
        rx = pd.read_csv(rx_path)
        rx_eval = filter_eval(rx, start, end)
        if "delay_s" in rx_eval and not rx_eval.empty:
            row["p95_delay_s"] = float(rx_eval["delay_s"].quantile(0.95))

    applied = optional_csv([run_dir / "applied_controls.csv", run_dir / "bridge" / "applied_controls.csv"])
    if applied is not None and len(applied):
        row["controller_success_rate"] = float(applied["response_success"].mean())
        row["control_updates"] = int(len(applied))

    timing = optional_csv(
        [run_dir / "controller_solve_times.csv", run_dir / "bridge" / "controller_solve_times.csv"]
    )
    if timing is not None and len(timing):
        row["mean_controller_solve_time_s"] = float(timing["controller_solve_time_s"].mean())
        row["max_controller_solve_time_s"] = float(timing["controller_solve_time_s"].max())
        row["mean_bridge_wait_time_s"] = float(timing["wait_time_s"].mean())

    return row


def parse_expected_runs(raw: str) -> list[int]:
    runs = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            runs.append(int(item))
    return runs


def run_completion_status(run_dir: Path, expected_eval_end_s: float = 290.0) -> dict:
    kpi_path = run_dir / "kpi_timeseries.csv"
    if not kpi_path.exists():
        return {"status": "missing", "last_time_s": None, "rows": 0, "reason": "no KPI CSV"}

    metadata = load_json(run_dir / "kpi_timeseries_metadata.json")
    kpi = pd.read_csv(kpi_path, usecols=["time_s"])
    if kpi.empty:
        return {"status": "incomplete", "last_time_s": None, "rows": 0, "reason": "empty KPI CSV"}

    eval_end = max(float(metadata.get("evaluation_end_s", expected_eval_end_s)), expected_eval_end_s)
    last_time = float(kpi["time_s"].max())
    complete = last_time >= eval_end
    return {
        "status": "complete" if complete else "partial",
        "last_time_s": last_time,
        "rows": int(len(kpi)),
        "reason": "" if complete else f"KPI ends before evaluation_end_s={eval_end:g}",
    }


def write_main_table(agg: pd.DataFrame, tables_dir: Path) -> None:
    rows = []
    for method in METHOD_ORDER:
        method_rows = agg[agg["method"] == method]
        if method_rows.empty:
            continue
        row = {"method": method, "label": METHOD_LABELS.get(method, method)}
        for metric in MAIN_TABLE_METRICS:
            metric_rows = method_rows[method_rows["metric"] == metric]
            if metric_rows.empty:
                continue
            value = metric_rows.iloc[0]
            row[f"{metric}_mean"] = value["mean"]
            row[f"{metric}_ci95"] = value["ci95"]
            row[f"{metric}_n"] = int(value["n"])
        rows.append(row)

    if rows:
        pd.DataFrame(rows).to_csv(tables_dir / "paper_main_table.csv", index=False)


def write_readiness_report(
    campaign_dir: Path,
    methods: list[str],
    expected_runs: list[int],
    run_metrics: pd.DataFrame,
    agg: pd.DataFrame,
) -> None:
    analysis_dir = campaign_dir / "03_analysis"
    tables_dir = analysis_dir / "tables"
    report = analysis_dir / "paper_readiness_report.md"

    lines = [
        "# Paper Readiness Report",
        "",
        "This report is generated by `scripts/aggregate_final_campaign.py` for the DeePC NR-V2X conference-paper campaign.",
        "",
        "## Matched-Run Requirement",
        "",
        f"- Required run ids: {', '.join(str(run) for run in expected_runs)}",
        "- A run is paper-ready only when its KPI series reaches the evaluation end time from metadata, normally `290 s` for the 300 s simulation with 10 s cooldown.",
        "- DeePC should be compared primarily with `fixed_10hz`; `fixed_5hz` is a contextual lower-load baseline.",
        "",
        "## Run Status",
        "",
        "| Method | Expected run | Status | Last KPI time (s) | KPI rows | Note |",
        "|---|---:|---|---:|---:|---|",
    ]

    for method in methods:
        method_dir = campaign_dir / "02_runs" / method
        for run in expected_runs:
            status = run_completion_status(method_dir / f"run_{run}")
            last = "" if status["last_time_s"] is None else f"{status['last_time_s']:.1f}"
            lines.append(
                f"| {method} | {run} | {status['status']} | {last} | {status['rows']} | {status['reason']} |"
            )

        extra_runs = []
        if method_dir.exists():
            expected_names = {f"run_{run}" for run in expected_runs}
            for run_dir in sorted(method_dir.glob("run_*")):
                if run_dir.name not in expected_names and (run_dir / "kpi_timeseries.csv").exists():
                    extra_runs.append(run_dir.name)
        if extra_runs:
            lines.extend(["", f"Extra non-matched KPI runs for `{method}`: {', '.join(extra_runs)}.", ""])

    lines.extend(
        [
            "",
            "## Current Aggregate Evidence",
            "",
            "Use this table for orientation only until all methods have the same completed run ids.",
            "",
            "| Method | Metric | n | Mean | 95% CI |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for _, row in agg[agg["metric"].isin(MAIN_TABLE_METRICS)].iterrows():
        lines.append(
            f"| {row['method']} | {row['metric']} | {int(row['n'])} | {row['mean']:.6g} | {row['ci95']:.6g} |"
        )

    lines.extend(
        [
            "",
            "## Paper Guidance",
            "",
            "- Keep the abstract conservative until `paper_main_table.csv` contains completed matched runs for all main methods.",
            "- Exclude `threshold_dcc` from the main comparison until its placeholder controller is replaced and validated.",
            "- Report PRR as awareness-range PRR within 200 m, and CBR as PHY busy-time fraction from `NrSpectrumPhy::ChannelOccupied`.",
            f"- Generated tables: `{tables_dir / 'run_metrics.csv'}`, `{tables_dir / 'aggregate_metrics.csv'}`, `{tables_dir / 'paper_main_table.csv'}`.",
        ]
    )
    report.write_text("\n".join(lines) + "\n")


def add_density_bins(kpi: pd.DataFrame) -> pd.DataFrame:
    kpi = kpi.copy()
    active = kpi["active_vehicle_count_core"]
    if active.nunique() >= 3:
        kpi["density_bin"] = pd.qcut(
            active,
            q=3,
            labels=["low", "medium", "high"],
            duplicates="drop",
        )
    else:
        kpi["density_bin"] = "all"
    return kpi


def compute_density_bin_metrics(method: str, run_dir: Path) -> list[dict]:
    kpi_path = run_dir / "kpi_timeseries.csv"
    if not kpi_path.exists():
        return []
    metadata = load_json(run_dir / "kpi_timeseries_metadata.json")
    kpi = add_prr_alias(pd.read_csv(kpi_path))
    if "active_vehicle_count_core" not in kpi:
        return []
    start, end = eval_window(metadata, kpi)
    kpi_eval = filter_eval(kpi, start, end)
    if kpi_eval.empty:
        return []

    rows = []
    kpi_eval = add_density_bins(kpi_eval)
    for density_bin, group in kpi_eval.groupby("density_bin", observed=True):
        rows.append(
            {
                "method": method,
                "run": run_number(run_dir),
                "density_bin": str(density_bin),
                "rows": int(len(group)),
                "active_vehicle_count_min": float(group["active_vehicle_count_core"].min()),
                "active_vehicle_count_mean": float(group["active_vehicle_count_core"].mean()),
                "active_vehicle_count_max": float(group["active_vehicle_count_core"].max()),
                "mean_prr_awareness": float(group["prr_awareness"].mean()),
                "beacon_error_rate": float((1.0 - group["prr_awareness"]).mean()),
                "mean_pir_s": float(group["pir_s"].mean()),
                "mean_cbr": float(group["cbr"].mean()),
                "cbr_gt_0p6_rate": float((group["cbr"] > 0.6).mean()),
                "mean_tx_power_dbm": float(group["tx_power_dbm"].mean()),
                "mean_beacon_interval_s": float(group["beacon_interval_s"].mean()),
            }
        )
    return rows


def aggregate(run_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in run_metrics.groupby("method", sort=False):
        for metric in METRICS:
            if metric not in group:
                continue
            values = group[metric].dropna()
            if values.empty:
                continue
            n = len(values)
            std = float(values.std(ddof=1)) if n > 1 else 0.0
            ci95 = 1.96 * std / sqrt(n) if n > 1 else 0.0
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "n": n,
                    "mean": float(values.mean()),
                    "std": std,
                    "ci95": ci95,
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def plot_metric(agg: pd.DataFrame, metric: str, out_dir: Path, ylabel: str) -> None:
    subset = agg[agg["metric"] == metric].copy()
    if subset.empty:
        return
    subset["method_order"] = subset["method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
    subset = subset.sort_values("method_order")
    plt.figure(figsize=(8, 4.5))
    plt.bar(subset["method"], subset["mean"], yerr=subset["ci95"], capsize=4)
    plt.ylabel(ylabel)
    plt.xlabel("Method")
    plt.xticks(rotation=20, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{metric}_comparison.png", dpi=200)
    plt.close()


def write_plots(agg: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_metric(agg, "mean_prr_awareness", out_dir, "Mean PRR")
    plot_metric(agg, "beacon_error_rate", out_dir, "Beacon error rate")
    plot_metric(agg, "mean_pir_s", out_dir, "Mean PIR (s)")
    plot_metric(agg, "mean_cbr", out_dir, "Mean CBR")
    plot_metric(agg, "cbr_gt_0p6_rate", out_dir, "CBR > 0.6 rate")
    plot_metric(agg, "p95_delay_s", out_dir, "P95 delay (s)")


def write_density_bin_outputs(density_metrics: pd.DataFrame, tables_dir: Path) -> None:
    if density_metrics.empty:
        return
    density_metrics.to_csv(tables_dir.parent / "density_bin_metrics.csv", index=False)
    density_metrics.to_csv(tables_dir / "density_bin_metrics.csv", index=False)
    summary = (
        density_metrics.groupby(["method", "density_bin"], as_index=False)
        .agg(
            runs=("run", "nunique"),
            active_vehicle_count_mean=("active_vehicle_count_mean", "mean"),
            mean_prr_awareness=("mean_prr_awareness", "mean"),
            beacon_error_rate=("beacon_error_rate", "mean"),
            mean_pir_s=("mean_pir_s", "mean"),
            mean_cbr=("mean_cbr", "mean"),
            cbr_gt_0p6_rate=("cbr_gt_0p6_rate", "mean"),
        )
    )
    summary.to_csv(tables_dir.parent / "density_bin_summary.csv", index=False)
    summary.to_csv(tables_dir / "density_bin_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    expected_runs = parse_expected_runs(args.expected_runs)
    run_rows = []
    density_rows = []
    for method in args.methods:
        method_dir = args.campaign_dir / "02_runs" / method
        if not method_dir.exists():
            continue
        for run_dir in sorted(method_dir.glob("run_*")):
            row = compute_run_metrics(method, run_dir)
            if row is not None:
                run_rows.append(row)
            density_rows.extend(compute_density_bin_metrics(method, run_dir))

    if not run_rows:
        raise FileNotFoundError(f"No completed run KPI files found in {args.campaign_dir}")

    analysis_dir = args.campaign_dir / "03_analysis"
    tables_dir = analysis_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    run_metrics = pd.DataFrame(run_rows).sort_values(["method", "run"])
    agg = aggregate(run_metrics)
    density_metrics = pd.DataFrame(density_rows)

    run_metrics.to_csv(analysis_dir / "run_metrics.csv", index=False)
    agg.to_csv(analysis_dir / "aggregate_metrics.csv", index=False)
    run_metrics.to_csv(tables_dir / "run_metrics.csv", index=False)
    agg.to_csv(tables_dir / "aggregate_metrics.csv", index=False)
    write_main_table(agg, tables_dir)
    write_density_bin_outputs(density_metrics, tables_dir)
    write_plots(agg, analysis_dir / "plots")
    write_readiness_report(args.campaign_dir, args.methods, expected_runs, run_metrics, agg)

    print(f"Saved run metrics: {analysis_dir / 'run_metrics.csv'}")
    print(f"Saved aggregate metrics: {analysis_dir / 'aggregate_metrics.csv'}")
    print(f"Saved paper readiness report: {analysis_dir / 'paper_readiness_report.md'}")


if __name__ == "__main__":
    main()
