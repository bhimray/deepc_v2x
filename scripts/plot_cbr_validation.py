import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
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
        df["run"] = run_label(path, args.campaign_dir)
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

    scatter_with_trend(kpi, "beacon_interval_s", "cbr", "Beacon interval Tb (s)", "CBR", "CBR vs Tb", plots_dir / "cbr_vs_tb.png")
    scatter_with_trend(kpi, "beacon_interval_s", "pir_s", "Beacon interval Tb (s)", "Mean PIR (s)", "Mean PIR vs Tb", plots_dir / "pir_vs_tb.png")
    scatter_with_trend(kpi, "beacon_interval_s", "pir_to_tb_ratio", "Beacon interval Tb (s)", "Mean PIR / Tb", "Normalized PIR vs Tb", plots_dir / "pir_ratio_vs_tb.png")
    scatter_with_trend(kpi, "cbr", "prr_awareness", "CBR", "PRR within awareness range", "PRR vs CBR", plots_dir / "prr_vs_cbr.png")
    scatter_with_trend(kpi, "cbr", "pir_s", "CBR", "Mean PIR (s)", "Mean PIR vs CBR", plots_dir / "pir_vs_cbr.png")
    scatter_with_trend(kpi, "prr_awareness", "pir_s", "PRR within awareness range", "Mean PIR (s)", "Mean PIR vs PRR", plots_dir / "pir_vs_prr.png")
    scatter_with_trend(kpi, "active_vehicle_count_core", "cbr", "Active core vehicles", "CBR", "CBR vs active vehicles", plots_dir / "cbr_vs_active_vehicles.png")
    scatter_with_trend(kpi, "active_vehicle_count_core", "prr_awareness", "Active core vehicles", "PRR within awareness range", "PRR vs active vehicles", plots_dir / "prr_vs_active_vehicles.png")

    plot_time_series(kpi, "cbr", "CBR", "CBR time-series sanity", plots_dir / "cbr_timeseries_sanity.png", ylim=(-0.02, 1.02))
    plot_time_series(kpi, "pir_s", "Mean PIR (s)", "Mean PIR time-series sanity", plots_dir / "pir_timeseries_sanity.png")

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
