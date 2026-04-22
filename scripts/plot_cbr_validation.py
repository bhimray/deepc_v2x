import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


OUTPUT_DIR = Path("data/output")
WARMUP_S = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate publication sanity plots for PHY/sensing-derived CBR."
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
        default=WARMUP_S,
        help=f"Warm-up cutoff in seconds. Default: {WARMUP_S}",
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


def load_vehicle_count(kpi_csv: Path) -> int | None:
    path = metadata_path(kpi_csv)
    if not path.exists():
        return None
    with path.open() as f:
        metadata = json.load(f)
    value = metadata.get("vehicle_count")
    return int(value) if value is not None else None


def require_columns(df: pd.DataFrame, path: Path, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def save_current(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved {path}")


def main() -> None:
    args = parse_args()
    kpi_paths = args.kpi_csv or default_kpi_csvs()
    if not kpi_paths:
        raise FileNotFoundError(f"No KPI CSV files found in {OUTPUT_DIR}")

    frames = []
    for path in kpi_paths:
        df = pd.read_csv(path)
        require_columns(
            df,
            path,
            ["time_s", "beacon_interval_s", "prr_150m", "pir_s", "cbr"],
        )
        df = df[df["time_s"] >= args.warmup].copy()
        if df.empty:
            continue
        df["run"] = path.stem
        vehicle_count = load_vehicle_count(path)
        if "vehicle_count" not in df.columns:
            df["vehicle_count"] = vehicle_count
        frames.append(df)

    if not frames:
        raise ValueError("No KPI samples remain after warm-up filtering.")

    kpi = pd.concat(frames, ignore_index=True)
    args.plots_dir.mkdir(parents=True, exist_ok=True)

    if ((kpi["cbr"] < 0) | (kpi["cbr"] > 1)).any():
        raise ValueError("CBR sanity failed: values outside [0, 1].")

    plt.figure()
    plt.scatter(kpi["beacon_interval_s"], kpi["cbr"], alpha=0.55)
    plt.xlabel("Beacon interval Tb (s)")
    plt.ylabel("CBR")
    plt.title("CBR vs Tb")
    plt.grid(True)
    save_current(args.plots_dir / "cbr_vs_tb.png")

    density = kpi.dropna(subset=["vehicle_count"])
    if not density.empty:
        by_density = density.groupby("vehicle_count", as_index=False)["cbr"].mean()
        plt.figure()
        plt.plot(by_density["vehicle_count"], by_density["cbr"], marker="o")
        plt.xlabel("Vehicle count")
        plt.ylabel("Mean CBR")
        plt.title("CBR vs density")
        plt.grid(True)
        save_current(args.plots_dir / "cbr_vs_density.png")
    else:
        print("Skipped CBR vs density: vehicle_count unavailable in KPI CSV/metadata.")

    plt.figure()
    plt.scatter(kpi["cbr"], kpi["prr_150m"], alpha=0.55)
    plt.xlabel("CBR")
    plt.ylabel("PRR within 150 m")
    plt.title("PRR vs CBR")
    plt.grid(True)
    save_current(args.plots_dir / "prr_vs_cbr.png")

    plt.figure()
    for run, group in kpi.groupby("run"):
        plt.plot(group["time_s"], group["cbr"], label=run)
    plt.xlabel("Time (s)")
    plt.ylabel("CBR")
    plt.title("CBR time-series sanity")
    plt.ylim(-0.02, 1.02)
    if kpi["run"].nunique() <= 8:
        plt.legend()
    plt.grid(True)
    save_current(args.plots_dir / "cbr_timeseries_sanity.png")


if __name__ == "__main__":
    main()
