import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


OUTPUT_DIR = Path("data/output")
WARMUP_S = 5.0


def add_prr_alias(kpi: pd.DataFrame) -> pd.DataFrame:
    if "prr_awareness" not in kpi and "prr_150m" in kpi:
        kpi = kpi.copy()
        kpi["prr_awareness"] = kpi["prr_150m"]
    return kpi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate publication sanity plots for PHY CBR and sensing diagnostics."
    )
    parser.add_argument(
        "kpi_csv",
        nargs="*",
        type=Path,
        help="One or more KPI CSV files. Defaults to data/output/kpi*.csv.",
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
        default=OUTPUT_DIR / "plots",
        help="Directory for generated PNG plots.",
    )
    return parser.parse_args()


def default_kpi_csvs() -> list[Path]:
    return sorted(
        path
        for path in OUTPUT_DIR.glob("kpi*.csv")
        if not path.name.endswith(("_tx_packet_log.csv", "_rx_packet_log.csv"))
    )


def metadata_path(kpi_csv: Path) -> Path:
    return Path(f"{kpi_csv.with_suffix('')}_metadata.json")


def rx_log_path(kpi_csv: Path) -> Path:
    return Path(f"{kpi_csv.with_suffix('')}_rx_packet_log.csv")


def load_vehicle_count(kpi_csv: Path) -> int | None:
    path = metadata_path(kpi_csv)
    if not path.exists():
        return None
    with path.open() as f:
        metadata = json.load(f)
    value = metadata.get("vehicle_count")
    return int(value) if value is not None else None


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

    unique_x = subset[x_col].nunique()
    if unique_x <= max_exact_groups:
        trend = subset.groupby(x_col, as_index=False)[y_col].mean()
        return trend.sort_values(x_col)

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


def scatter_3d(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    xlabel: str,
    ylabel: str,
    zlabel: str,
    title: str,
    out_path: Path,
) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        df[x_col],
        df[y_col],
        df[z_col],
        c=df[z_col],
        cmap="viridis",
        alpha=0.45,
        s=18,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    fig.colorbar(scatter, ax=ax, pad=0.12, shrink=0.75, label=zlabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    args = parse_args()
    kpi_paths = args.kpi_csv or default_kpi_csvs()
    if not kpi_paths:
        raise FileNotFoundError(f"No KPI CSV files found in {OUTPUT_DIR}")

    frames = []
    rx_frames = []
    for path in kpi_paths:
        df = add_prr_alias(pd.read_csv(path))
        require_columns(
            df,
            path,
            [
                "time_s",
                "beacon_interval_s",
                "active_vehicle_count_core",
                "density_veh_per_km_core",
                "mean_neighbors_150m",
                "mean_neighbors_300m",
                "prr_awareness",
                "pir_s",
                "cbr",
                "sensing_exclusion_ratio",
            ],
        )
        start_s, end_s = eval_window(path, args)
        df = df[df["time_s"] >= start_s].copy()
        if end_s is not None:
            if end_s < 0.0:
                end_s = float(df["time_s"].max()) + end_s if not df.empty else end_s
            df = df[df["time_s"] <= end_s].copy()
        if df.empty:
            continue
        df["run"] = path.stem
        vehicle_count = load_vehicle_count(path)
        if "vehicle_count" not in df.columns:
            df["vehicle_count"] = vehicle_count
        frames.append(df)

        rx_path = rx_log_path(path)
        if rx_path.exists():
            rx = pd.read_csv(rx_path)
            require_columns(rx, rx_path, ["time_s", "pir_s"])
            rx = rx[rx["time_s"] >= start_s].copy()
            if end_s is not None:
                rx = rx[rx["time_s"] <= end_s].copy()
            if not rx.empty:
                rx["run"] = path.stem
                rx_frames.append(rx)

    if not frames:
        raise ValueError("No KPI samples remain after warm-up filtering.")

    kpi = pd.concat(frames, ignore_index=True)
    args.plots_dir.mkdir(parents=True, exist_ok=True)

    if ((kpi["cbr"] < 0) | (kpi["cbr"] > 1)).any():
        raise ValueError("PHY CBR sanity failed: values outside [0, 1].")
    if ((kpi["sensing_exclusion_ratio"] < 0) | (kpi["sensing_exclusion_ratio"] > 1)).any():
        raise ValueError("Sensing exclusion ratio sanity failed: values outside [0, 1].")
    if (kpi["active_vehicle_count_core"] < 0).any():
        raise ValueError("Active vehicle count sanity failed: negative values.")
    if (kpi["density_veh_per_km_core"] < 0).any():
        raise ValueError("Density sanity failed: negative values.")
    if (kpi["pir_s"] < 0).any():
        raise ValueError("PIR sanity failed: negative values.")

    kpi["pir_to_tb_ratio"] = kpi["pir_s"] / kpi["beacon_interval_s"].where(
        kpi["beacon_interval_s"] > 0
    )

    scatter_with_trend(
        kpi,
        "beacon_interval_s",
        "cbr",
        "Beacon interval Tb (s)",
        "PHY CBR",
        "PHY CBR vs Tb",
        args.plots_dir / "cbr_vs_tb.png",
    )

    scatter_with_trend(
        kpi,
        "beacon_interval_s",
        "sensing_exclusion_ratio",
        "Beacon interval Tb (s)",
        "Sensing exclusion ratio",
        "Sensing exclusion ratio vs Tb",
        args.plots_dir / "sensing_exclusion_vs_tb.png",
    )

    scatter_with_trend(
        kpi,
        "beacon_interval_s",
        "pir_s",
        "Beacon interval Tb (s)",
        "Mean PIR (s)",
        "Mean PIR vs Tb",
        args.plots_dir / "pir_vs_tb.png",
    )

    scatter_with_trend(
        kpi,
        "beacon_interval_s",
        "pir_to_tb_ratio",
        "Beacon interval Tb (s)",
        "Mean PIR / Tb",
        "Normalized PIR vs Tb",
        args.plots_dir / "pir_ratio_vs_tb.png",
    )

    density = kpi.dropna(subset=["vehicle_count"])
    if not density.empty:
        by_density = density.groupby("vehicle_count", as_index=False)["cbr"].mean()
        plt.figure()
        plt.plot(by_density["vehicle_count"], by_density["cbr"], marker="o")
        plt.xlabel("Vehicle count")
        plt.ylabel("Mean PHY CBR")
        plt.title("PHY CBR vs density")
        plt.grid(True)
        save_current(args.plots_dir / "cbr_vs_density.png")

        by_density = density.groupby("vehicle_count", as_index=False)[
            "sensing_exclusion_ratio"
        ].mean()
        plt.figure()
        plt.plot(
            by_density["vehicle_count"],
            by_density["sensing_exclusion_ratio"],
            marker="o",
        )
        plt.xlabel("Vehicle count")
        plt.ylabel("Mean sensing exclusion ratio")
        plt.title("Sensing exclusion ratio vs density")
        plt.grid(True)
        save_current(args.plots_dir / "sensing_exclusion_vs_density.png")
    else:
        print("Skipped CBR vs density: vehicle_count unavailable in KPI CSV/metadata.")

    scatter_with_trend(
        kpi,
        "density_veh_per_km_core",
        "cbr",
        "Measured density in core (veh/km)",
        "PHY CBR",
        "PHY CBR vs measured density",
        args.plots_dir / "cbr_vs_measured_density.png",
    )

    scatter_with_trend(
        kpi,
        "density_veh_per_km_core",
        "prr_awareness",
        "Measured density in core (veh/km)",
        "PRR within awareness range",
        "PRR vs measured density",
        args.plots_dir / "prr_vs_measured_density.png",
    )

    scatter_with_trend(
        kpi,
        "mean_neighbors_150m",
        "prr_awareness",
        "Mean neighbors within 150 m",
        "PRR within awareness range",
        "PRR vs local neighbor count",
        args.plots_dir / "prr_vs_mean_neighbors_150m.png",
    )

    scatter_with_trend(
        kpi,
        "cbr",
        "prr_awareness",
        "PHY CBR",
        "PRR within awareness range",
        "PRR vs PHY CBR",
        args.plots_dir / "prr_vs_cbr.png",
    )

    scatter_with_trend(
        kpi,
        "cbr",
        "pir_s",
        "PHY CBR",
        "Mean PIR (s)",
        "Mean PIR vs PHY CBR",
        args.plots_dir / "pir_vs_cbr.png",
    )

    scatter_with_trend(
        kpi,
        "prr_awareness",
        "pir_s",
        "PRR within awareness range",
        "Mean PIR (s)",
        "Mean PIR vs PRR",
        args.plots_dir / "pir_vs_prr.png",
    )

    scatter_with_trend(
        kpi,
        "sensing_exclusion_ratio",
        "prr_awareness",
        "Sensing exclusion ratio",
        "PRR within awareness range",
        "PRR vs sensing exclusion ratio",
        args.plots_dir / "prr_vs_sensing_exclusion.png",
    )

    scatter_3d(
        kpi,
        "density_veh_per_km_core",
        "cbr",
        "prr_awareness",
        "Measured density in core (veh/km)",
        "PHY CBR",
        "PRR within awareness range",
        "PRR vs PHY CBR vs measured density",
        args.plots_dir / "prr_cbr_density_3d.png",
    )

    plot_time_series(
        kpi,
        "cbr",
        "PHY CBR",
        "PHY CBR time-series sanity",
        args.plots_dir / "cbr_timeseries_sanity.png",
        ylim=(-0.02, 1.02),
    )

    plot_time_series(
        kpi,
        "sensing_exclusion_ratio",
        "Sensing exclusion ratio",
        "Sensing exclusion ratio time-series sanity",
        args.plots_dir / "sensing_exclusion_timeseries_sanity.png",
        ylim=(-0.02, 1.02),
    )

    plot_time_series(
        kpi,
        "pir_s",
        "Mean PIR (s)",
        "Mean PIR time-series sanity",
        args.plots_dir / "pir_timeseries_sanity.png",
    )

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
    save_current(args.plots_dir / "pir_vs_tb_timeseries.png")

    if rx_frames:
        rx_pir = pd.concat(rx_frames, ignore_index=True)
        rx_pir = rx_pir[rx_pir["pir_s"].notna() & (rx_pir["pir_s"] >= 0)].copy()
        if not rx_pir.empty:
            plt.figure()
            plt.hist(rx_pir["pir_s"], bins=80)
            plt.xlabel("Raw RX PIR (s)")
            plt.ylabel("Packet receptions")
            plt.title("Raw RX PIR distribution")
            plt.grid(True)
            save_current(args.plots_dir / "pir_raw_histogram.png")

            plt.figure()
            sorted_pir = rx_pir["pir_s"].sort_values().reset_index(drop=True)
            cdf = (sorted_pir.index + 1) / len(sorted_pir)
            plt.plot(sorted_pir, cdf)
            plt.xlabel("Raw RX PIR (s)")
            plt.ylabel("CDF")
            plt.title("Raw RX PIR CDF")
            plt.grid(True)
            save_current(args.plots_dir / "pir_raw_cdf.png")
    else:
        print("Skipped raw PIR distribution plots: RX packet logs unavailable.")


if __name__ == "__main__":
    main()
