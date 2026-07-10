import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_DIR = Path("data/output")
WARMUP_S = 5.0
REQUIRED_COLUMNS = [
    "time_s",
    "beacon_interval_s",
    "active_vehicle_count_core",
    "prr_awareness",
    "pir_s",
    "cbr",
]


def add_prr_alias(kpi: pd.DataFrame) -> pd.DataFrame:
    if "prr_awareness" not in kpi and "prr_150m" in kpi:
        kpi = kpi.copy()
        kpi["prr_awareness"] = kpi["prr_150m"]
    return kpi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate KPI-only sanity plots from pruned NR-V2X KPI CSV files."
    )
    parser.add_argument(
        "kpi_csv",
        nargs="*",
        type=Path,
        help="One or more KPI CSV files. Defaults to campaign runs or data/output/kpi*.csv.",
    )
    parser.add_argument(
        "--campaign-dir",
        type=Path,
        default=None,
        help="Campaign root containing 02_runs/<method>/run_<id>/kpi_timeseries.csv.",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=None,
        help=f"Warm-up cutoff in seconds. Defaults to metadata, then {WARMUP_S}.",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=None,
        help="Final cool-down duration in seconds. Defaults to metadata, then 0.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Directory for generated PNG plots. Defaults to campaign 03_analysis/validation_plots.",
    )
    return parser.parse_args()


def default_kpi_csvs(campaign_dir: Path | None = None) -> list[Path]:
    if campaign_dir is not None:
        return sorted(campaign_dir.glob("02_runs/*/run_*/kpi_timeseries.csv"))
    return sorted(OUTPUT_DIR.glob("kpi*.csv"))


def run_label(kpi_csv: Path, campaign_dir: Path | None) -> str:
    if campaign_dir is not None:
        try:
            rel = kpi_csv.relative_to(campaign_dir / "02_runs")
            if len(rel.parts) >= 3:
                return f"{rel.parts[0]}/{rel.parts[1]}"
        except ValueError:
            pass
    return kpi_csv.stem


def run_method(label: str) -> str:
    return label.split("/", 1)[0]


def metadata_path(kpi_csv: Path) -> Path:
    return Path(f"{kpi_csv.with_suffix('')}_metadata.json")


def load_metadata(kpi_csv: Path) -> dict:
    path = metadata_path(kpi_csv)
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def eval_window(kpi_csv: Path, args: argparse.Namespace) -> tuple[float, float | None]:
    metadata = load_metadata(kpi_csv)
    warmup = args.warmup
    if warmup is None:
        warmup = float(metadata.get("evaluation_start_s", metadata.get("warmup_s", WARMUP_S)))

    if "evaluation_end_s" in metadata:
        return warmup, float(metadata["evaluation_end_s"])

    cooldown = args.cooldown
    if cooldown is None:
        cooldown = float(metadata.get("cooldown_s", 0.0))
    return warmup, None if cooldown <= 0.0 else -cooldown


def require_columns(df: pd.DataFrame, path: Path, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def save_current(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved {path}")


def trend_points(
    df: pd.DataFrame, x_col: str, y_col: str, max_exact_groups: int = 20, num_bins: int = 12
) -> pd.DataFrame:
    subset = df[[x_col, y_col]].dropna().sort_values(x_col)
    if subset.empty:
        return pd.DataFrame(columns=[x_col, y_col])

    if subset[x_col].nunique() <= max_exact_groups:
        return subset.groupby(x_col, as_index=False)[y_col].mean().sort_values(x_col)

    bins = min(num_bins, len(subset))
    if bins < 2:
        return pd.DataFrame(columns=[x_col, y_col])

    binned = subset.copy()
    binned["_trend_bin"] = pd.qcut(binned[x_col], q=bins, duplicates="drop")
    trend = binned.groupby("_trend_bin", as_index=False).agg({x_col: "mean", y_col: "mean"})
    return trend.sort_values(x_col)


def scatter_with_trend(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    plt.figure()
    plt.scatter(df[x_col], df[y_col], alpha=0.35, s=18, label="Samples")
    trend = trend_points(df, x_col, y_col)
    if len(trend) >= 2:
        plt.plot(trend[x_col], trend[y_col], color="crimson", linewidth=2.2, label="Trend")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    save_current(out_path)


def plot_time_series(
    df: pd.DataFrame,
    y_col: str,
    ylabel: str,
    title: str,
    out_path: Path,
    ylim: tuple[float, float] | None = None,
) -> None:
    plt.figure()
    for run, group in df.groupby("run"):
        plt.plot(group["time_s"], group[y_col], label=run)
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    if df["run"].nunique() <= 8:
        plt.legend()
    plt.grid(True)
    save_current(out_path)


def plot_time_series_with_active_count(
    df: pd.DataFrame,
    y_col: str,
    ylabel: str,
    title: str,
    out_path: Path,
    ylim: tuple[float, float] | None = None,
) -> None:
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    for run, group in df.groupby("run"):
        ax1.plot(group["time_s"], group[y_col], label=run)
        ax2.plot(
            group["time_s"],
            group["active_vehicle_count_core"],
            linestyle=":",
            alpha=0.45,
            label=f"{run} active vehicles",
        )
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel(ylabel)
    ax2.set_ylabel("Active core vehicles")
    ax1.set_title(title)
    if ylim is not None:
        ax1.set_ylim(*ylim)
    if df["run"].nunique() <= 4:
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize="small")
    ax1.grid(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def add_density_bins(kpi: pd.DataFrame) -> pd.DataFrame:
    kpi = kpi.copy()
    if kpi["active_vehicle_count_core"].nunique() >= 3:
        kpi["density_bin"] = pd.qcut(
            kpi["active_vehicle_count_core"],
            q=3,
            labels=["low", "medium", "high"],
            duplicates="drop",
        )
    else:
        kpi["density_bin"] = "all"
    return kpi


def write_density_bin_summary(kpi: pd.DataFrame, out_path: Path) -> None:
    summary = (
        kpi.groupby(["method", "density_bin"], observed=True)
        .agg(
            rows=("time_s", "size"),
            active_vehicle_count_min=("active_vehicle_count_core", "min"),
            active_vehicle_count_mean=("active_vehicle_count_core", "mean"),
            active_vehicle_count_max=("active_vehicle_count_core", "max"),
            mean_prr_awareness=("prr_awareness", "mean"),
            beacon_error_rate=("beacon_error_rate", "mean"),
            mean_pir_s=("pir_s", "mean"),
            mean_cbr=("cbr", "mean"),
            cbr_gt_0p6_rate=("cbr_gt_0p6", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(out_path, index=False)
    print(f"Saved {out_path}")


def plot_density_bin_metric(
    kpi: pd.DataFrame,
    y_col: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    summary = (
        kpi.groupby(["method", "density_bin"], observed=True)[y_col]
        .mean()
        .reset_index()
    )
    if summary.empty:
        return
    pivot = summary.pivot(index="density_bin", columns="method", values=y_col)
    pivot.plot(kind="bar", figsize=(8, 4.5))
    plt.xlabel("Observed density bin")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=0)
    plt.grid(True, axis="y", alpha=0.3)
    save_current(out_path)


def plot_input_timeseries(kpi: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for run, group in kpi.groupby("run"):
        axes[0].step(group["time_s"], group["tx_power_dbm"], where="post", label=run)
        axes[1].step(group["time_s"], group["beacon_interval_s"], where="post", label=run)
        axes[2].plot(group["time_s"], group["active_vehicle_count_core"], label=run)
    axes[0].set_ylabel("Tx power (dBm)")
    axes[1].set_ylabel("Beacon interval (s)")
    axes[2].set_ylabel("Active core vehicles")
    axes[2].set_xlabel("Time (s)")
    axes[0].set_title("PRBS inputs and measured traffic context")
    for axis in axes:
        axis.grid(True, alpha=0.3)
    if kpi["run"].nunique() <= 8:
        axes[0].legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def write_input_state_summary(kpi: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    aggregations = {
        "samples": ("time_s", "size"),
        "mean_active_vehicles": ("active_vehicle_count_core", "mean"),
        "mean_prr": ("prr_awareness", "mean"),
        "mean_pir_s": ("pir_s", "mean"),
        "mean_cbr": ("cbr", "mean"),
    }
    for column in ("tx_count", "eligible_count", "success_count"):
        if column in kpi:
            aggregations[f"mean_{column}"] = (column, "mean")
    summary = (
        kpi.groupby(["tx_power_dbm", "beacon_interval_s"], observed=True)
        .agg(**aggregations)
        .reset_index()
        .sort_values(["tx_power_dbm", "beacon_interval_s"])
    )
    summary.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    return summary


def plot_input_state_heatmaps(summary: pd.DataFrame, out_path: Path) -> None:
    metrics = [
        ("samples", "Sample count", ".0f"),
        ("mean_prr", "Mean PRR", ".3f"),
        ("mean_pir_s", "Mean PIR (s)", ".3f"),
        ("mean_cbr", "Mean CBR", ".3f"),
    ]
    powers = sorted(summary["tx_power_dbm"].unique())
    intervals = sorted(summary["beacon_interval_s"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, (column, title, value_format) in zip(axes.flat, metrics):
        pivot = (
            summary.pivot(
                index="tx_power_dbm",
                columns="beacon_interval_s",
                values=column,
            )
            .reindex(index=powers, columns=intervals)
        )
        values = pivot.to_numpy(dtype=float)
        image = axis.imshow(values, aspect="auto", origin="lower", cmap="viridis")
        axis.set_title(title)
        axis.set_xlabel("Beacon interval (s)")
        axis.set_ylabel("Tx power (dBm)")
        axis.set_xticks(range(len(intervals)), [f"{value:g}" for value in intervals])
        axis.set_yticks(range(len(powers)), [f"{value:g}" for value in powers])
        finite = values[np.isfinite(values)]
        midpoint = float(np.nanmean(finite)) if finite.size else 0.0
        for row in range(len(powers)):
            for col in range(len(intervals)):
                value = values[row, col]
                if np.isfinite(value):
                    color = "white" if value < midpoint else "black"
                    axis.text(
                        col,
                        row,
                        format(value, value_format),
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=8,
                    )
        fig.colorbar(image, ax=axis, shrink=0.82)
    fig.suptitle("PRBS input-state coverage and average response")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_metric_distributions(kpi: pd.DataFrame, out_path: Path) -> None:
    metrics = [
        ("prr_awareness", "PRR"),
        ("pir_s", "PIR (s)"),
        ("cbr", "CBR"),
        ("active_vehicle_count_core", "Active core vehicles"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, (column, label) in zip(axes.flat, metrics):
        axis.hist(kpi[column].dropna(), bins=25, color="#277da1", edgecolor="white")
        axis.set_xlabel(label)
        axis.set_ylabel("Samples")
        axis.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Training-data distributions")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_correlation_matrix(kpi: pd.DataFrame, out_path: Path) -> None:
    columns = [
        "tx_power_dbm",
        "beacon_interval_s",
        "active_vehicle_count_core",
        "prr_awareness",
        "pir_s",
        "cbr",
    ]
    correlation = kpi[columns].corr()
    fig, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(correlation, vmin=-1.0, vmax=1.0, cmap="RdBu_r")
    labels = [
        "Tx power",
        "Beacon interval",
        "Active vehicles",
        "PRR",
        "PIR",
        "CBR",
    ]
    axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    for row in range(len(labels)):
        for col in range(len(labels)):
            value = correlation.iloc[row, col]
            axis.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if abs(value) > 0.55 else "black",
            )
    axis.set_title("Pearson correlation matrix")
    fig.colorbar(image, ax=axis, label="Correlation")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_packet_accounting(kpi: pd.DataFrame, out_path: Path) -> None:
    columns = [
        column
        for column in ("tx_count", "eligible_count", "success_count")
        if column in kpi
    ]
    if not columns:
        return
    fig, axes = plt.subplots(len(columns), 1, figsize=(10, 7), sharex=True, squeeze=False)
    for axis, column in zip(axes.flat, columns):
        for run, group in kpi.groupby("run"):
            axis.plot(group["time_s"], group[column], label=run)
        axis.set_ylabel(column.replace("_", " ").title())
        axis.grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel("Time (s)")
    axes[0, 0].set_title("Packet accounting over the KPI window")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def load_kpi_frames(args: argparse.Namespace) -> list[pd.DataFrame]:
    kpi_paths = args.kpi_csv or default_kpi_csvs(args.campaign_dir)
    if not kpi_paths:
        source = args.campaign_dir if args.campaign_dir is not None else OUTPUT_DIR
        raise FileNotFoundError(f"No KPI CSV files found in {source}")

    frames = []
    for path in kpi_paths:
        df = add_prr_alias(pd.read_csv(path))
        require_columns(df, path, REQUIRED_COLUMNS)
        start_s, end_s = eval_window(path, args)
        df = df[df["time_s"] >= start_s].copy()
        if end_s is not None:
            if end_s < 0.0:
                end_s = float(df["time_s"].max()) + end_s if not df.empty else end_s
            df = df[df["time_s"] <= end_s].copy()
        if df.empty:
            continue
        label = run_label(path, args.campaign_dir)
        df["run"] = label
        df["method"] = run_method(label)
        frames.append(df)
    return frames


def main() -> None:
    args = parse_args()
    frames = load_kpi_frames(args)
    if not frames:
        raise ValueError("No KPI samples remain after warm-up filtering.")

    kpi = pd.concat(frames, ignore_index=True)
    plots_dir = args.plots_dir
    if plots_dir is None:
        plots_dir = (
            args.campaign_dir / "03_analysis" / "validation_plots"
            if args.campaign_dir is not None
            else OUTPUT_DIR / "plots"
        )
    plots_dir.mkdir(parents=True, exist_ok=True)

    if ((kpi["cbr"] < 0) | (kpi["cbr"] > 1)).any():
        raise ValueError("CBR sanity failed: values outside [0, 1].")
    if ((kpi["prr_awareness"] < 0) | (kpi["prr_awareness"] > 1)).any():
        raise ValueError("PRR sanity failed: values outside [0, 1].")
    if (kpi["active_vehicle_count_core"] < 0).any():
        raise ValueError("Active vehicle count sanity failed: negative values.")
    if (kpi["pir_s"] < 0).any():
        raise ValueError("PIR sanity failed: negative values.")

    kpi["pir_to_tb_ratio"] = kpi["pir_s"] / kpi["beacon_interval_s"].where(
        kpi["beacon_interval_s"] > 0
    )
    kpi["beacon_error_rate"] = 1.0 - kpi["prr_awareness"]
    kpi["cbr_gt_0p6"] = kpi["cbr"] > 0.6
    kpi = add_density_bins(kpi)

    scatter_with_trend(kpi, "beacon_interval_s", "cbr", "Beacon interval Tb (s)", "CBR", "CBR vs Tb", plots_dir / "cbr_vs_tb.png")
    scatter_with_trend(kpi, "beacon_interval_s", "prr_awareness", "Beacon interval Tb (s)", "PRR within awareness range", "PRR vs Tb", plots_dir / "prr_vs_tb.png")
    scatter_with_trend(kpi, "beacon_interval_s", "pir_s", "Beacon interval Tb (s)", "Mean PIR (s)", "Mean PIR vs Tb", plots_dir / "pir_vs_tb.png")
    scatter_with_trend(kpi, "beacon_interval_s", "pir_to_tb_ratio", "Beacon interval Tb (s)", "Mean PIR / Tb", "Normalized PIR vs Tb", plots_dir / "pir_ratio_vs_tb.png")
    if "tx_power_dbm" in kpi:
        scatter_with_trend(kpi, "tx_power_dbm", "cbr", "Tx power (dBm)", "CBR", "CBR vs Tx power", plots_dir / "cbr_vs_tx_power.png")
        scatter_with_trend(kpi, "tx_power_dbm", "prr_awareness", "Tx power (dBm)", "PRR within awareness range", "PRR vs Tx power", plots_dir / "prr_vs_tx_power.png")
        scatter_with_trend(kpi, "tx_power_dbm", "pir_s", "Tx power (dBm)", "Mean PIR (s)", "Mean PIR vs Tx power", plots_dir / "pir_vs_tx_power.png")
    scatter_with_trend(kpi, "cbr", "prr_awareness", "CBR", "PRR within awareness range", "PRR vs CBR", plots_dir / "prr_vs_cbr.png")
    scatter_with_trend(kpi, "cbr", "pir_s", "CBR", "Mean PIR (s)", "Mean PIR vs CBR", plots_dir / "pir_vs_cbr.png")
    scatter_with_trend(kpi, "prr_awareness", "pir_s", "PRR within awareness range", "Mean PIR (s)", "Mean PIR vs PRR", plots_dir / "pir_vs_prr.png")
    scatter_with_trend(kpi, "active_vehicle_count_core", "cbr", "Active core vehicles", "CBR", "CBR vs active vehicles", plots_dir / "cbr_vs_active_vehicles.png")
    scatter_with_trend(kpi, "active_vehicle_count_core", "prr_awareness", "Active core vehicles", "PRR within awareness range", "PRR vs active vehicles", plots_dir / "prr_vs_active_vehicles.png")
    scatter_with_trend(kpi, "active_vehicle_count_core", "pir_s", "Active core vehicles", "Mean PIR (s)", "PIR vs active vehicles", plots_dir / "pir_vs_active_vehicles.png")

    plot_time_series_with_active_count(kpi, "cbr", "CBR", "CBR and active vehicles over time", plots_dir / "cbr_timeseries_active_vehicles.png", ylim=(-0.02, 1.02))
    plot_time_series_with_active_count(kpi, "prr_awareness", "PRR within awareness range", "PRR and active vehicles over time", plots_dir / "prr_timeseries_active_vehicles.png", ylim=(-0.02, 1.02))
    plot_time_series_with_active_count(kpi, "pir_s", "Mean PIR (s)", "PIR and active vehicles over time", plots_dir / "pir_timeseries_active_vehicles.png")

    write_density_bin_summary(kpi, plots_dir / "density_bin_summary.csv")
    plot_density_bin_metric(kpi, "cbr", "Mean CBR", "CBR by observed density bin", plots_dir / "density_bin_cbr.png")
    plot_density_bin_metric(kpi, "prr_awareness", "Mean PRR", "PRR by observed density bin", plots_dir / "density_bin_prr.png")
    plot_density_bin_metric(kpi, "pir_s", "Mean PIR (s)", "PIR by observed density bin", plots_dir / "density_bin_pir.png")
    plot_density_bin_metric(kpi, "beacon_error_rate", "Beacon error rate", "Beacon error by observed density bin", plots_dir / "density_bin_beacon_error.png")

    if "tx_power_dbm" in kpi:
        plot_input_timeseries(kpi, plots_dir / "prbs_inputs_timeseries.png")
        input_summary = write_input_state_summary(kpi, plots_dir / "input_state_summary.csv")
        plot_input_state_heatmaps(input_summary, plots_dir / "input_state_heatmaps.png")
    plot_metric_distributions(kpi, plots_dir / "metric_distributions.png")
    plot_correlation_matrix(kpi, plots_dir / "correlation_matrix.png")
    plot_packet_accounting(kpi, plots_dir / "packet_accounting_timeseries.png")

    plt.figure()
    for run, group in kpi.groupby("run"):
        plt.plot(group["time_s"], group["pir_s"], label=f"{run} PIR")
        plt.plot(group["time_s"], group["beacon_interval_s"], linestyle="--", label=f"{run} Tb")
    plt.xlabel("Time (s)")
    plt.ylabel("Seconds")
    plt.title("Mean PIR and configured Tb over time")
    if kpi["run"].nunique() <= 4:
        plt.legend()
    plt.grid(True)
    save_current(plots_dir / "pir_vs_tb_timeseries.png")


if __name__ == "__main__":
    main()
