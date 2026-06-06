import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_DIR = Path("data/output")

AWARENESS_RANGE_M = 150.0
KPI_WINDOW_S = 1.0
WARMUP_S = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate KPI CSV output against raw TX/RX packet logs."
    )
    parser.add_argument(
        "kpi_csv",
        nargs="?",
        type=Path,
        help="KPI CSV to validate. Defaults to the newest KPI CSV in data/output.",
    )
    parser.add_argument(
        "--campaign-dir",
        type=Path,
        default=None,
        help="Campaign root containing 02_runs/<method>/run_<id>/kpi_timeseries.csv.",
    )
    parser.add_argument("--method", default=None, help="Method folder under 02_runs.")
    parser.add_argument("--run", default=None, help="Run number, e.g., 101 or run_101.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Directory for validation plots.")
    parser.add_argument("--show", action="store_true", help="Show plots interactively.")
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
        "--awareness-range",
        type=float,
        default=None,
        help="PRR awareness range in meters. Defaults to metadata, then 150.",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=KPI_WINDOW_S,
        help=f"PRR validation window in seconds. Default: {KPI_WINDOW_S}",
    )
    parser.add_argument(
        "--require-packet-logs",
        action="store_true",
        help="Fail when TX/RX packet log sidecars are missing instead of running KPI-only checks.",
    )
    return parser.parse_args()


def normalize_run_name(run: str) -> str:
    return run if run.startswith("run_") else f"run_{int(run):03d}"


def resolve_kpi_csv(args: argparse.Namespace) -> Path:
    if args.kpi_csv is not None:
        return args.kpi_csv
    if args.campaign_dir is not None:
        if args.method is None or args.run is None:
            raise ValueError("--campaign-dir requires --method and --run when kpi_csv is omitted.")
        return (
            args.campaign_dir
            / "02_runs"
            / args.method
            / normalize_run_name(str(args.run))
            / "kpi_timeseries.csv"
        )
    return newest_kpi_csv()


def newest_kpi_csv() -> Path:
    candidates = [
        path
        for path in OUTPUT_DIR.glob("kpi*.csv")
        if not path.name.endswith(("_tx_packet_log.csv", "_rx_packet_log.csv"))
    ]
    if not candidates:
        raise FileNotFoundError(f"No KPI CSV files found in {OUTPUT_DIR}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def add_prr_alias(kpi: pd.DataFrame) -> pd.DataFrame:
    if "prr_awareness" not in kpi and "prr_150m" in kpi:
        kpi = kpi.copy()
        kpi["prr_awareness"] = kpi["prr_150m"]
    return kpi


def add_eligible_alias(tx: pd.DataFrame) -> pd.DataFrame:
    if "eligible_rx_count_awareness" not in tx and "eligible_rx_count_150m" in tx:
        tx = tx.copy()
        tx["eligible_rx_count_awareness"] = tx["eligible_rx_count_150m"]
    return tx


def log_paths(kpi_path: Path) -> tuple[Path, Path]:
    stem = kpi_path.with_suffix("")
    return (
        Path(f"{stem}_tx_packet_log.csv"),
        Path(f"{stem}_rx_packet_log.csv"),
    )


def metadata_path(kpi_path: Path) -> Path:
    return Path(f"{kpi_path.with_suffix('')}_metadata.json")


def load_metadata(kpi_path: Path) -> dict:
    path = metadata_path(kpi_path)
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def require_columns(df: pd.DataFrame, path: Path, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def print_optional_mean(df: pd.DataFrame, column: str, label: str) -> None:
    if column in df:
        print(f"{label}: {df[column].mean():.6f}")


def plot_kpi_timeseries(kpi_eval: pd.DataFrame, args: argparse.Namespace) -> None:
    for column, ylabel, filename in [
        ("prr_awareness", "PRR", "kpi_prr_timeseries.png"),
        ("pir_s", "PIR (s)", "kpi_pir_timeseries.png"),
        ("cbr", "CBR", "kpi_cbr_timeseries.png"),
        ("active_vehicle_count_core", "Active core vehicles", "kpi_active_vehicles_timeseries.png"),
    ]:
        if column not in kpi_eval:
            continue
        plt.figure()
        plt.plot(kpi_eval["time_s"], kpi_eval[column])
        plt.xlabel("Time (s)")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} over time")
        plt.grid()
        save_or_show(
            args.out_dir / filename if args.out_dir is not None else None,
            args.show,
        )


def kpi_only_validation(
    kpi_eval: pd.DataFrame,
    kpi_path: Path,
    warmup: float,
    eval_end: float | None,
    args: argparse.Namespace,
) -> None:
    print("\n===== KPI-ONLY VALIDATION =====")
    print(f"KPI CSV             : {kpi_path}")
    print("TX/RX packet logs   : not found; skipped raw-log PRR validation")
    print(f"Evaluation window   : {warmup:.1f} s to {eval_end if eval_end is not None else 'end'}")
    print(f"Rows                : {len(kpi_eval)}")
    print(f"Mean PRR KPI samples: {kpi_eval['prr_awareness'].mean():.6f}")
    print(f"Beacon error rate   : {(1.0 - kpi_eval['prr_awareness']).mean():.6f}")
    print(f"Mean PIR from KPI   : {kpi_eval['pir_s'].mean():.6f}")
    print(f"Mean Tb             : {kpi_eval['beacon_interval_s'].mean():.6f}")
    print_optional_mean(kpi_eval, "active_vehicle_count_core", "Mean active vehicles")
    print_optional_mean(kpi_eval, "density_veh_per_km_core", "Mean density veh/km ")
    print_optional_mean(kpi_eval, "mean_neighbors_150m", "Mean neighbors 150m ")
    print_optional_mean(kpi_eval, "mean_neighbors_300m", "Mean neighbors 300m ")
    print(f"Mean PHY CBR        : {kpi_eval['cbr'].mean():.6f}")
    print_optional_mean(kpi_eval, "sensing_exclusion_ratio", "Mean sensing exclusion ratio")
    plot_kpi_timeseries(kpi_eval, args)


def eligible_bin_columns(tx: pd.DataFrame) -> list[tuple[str, float, float]]:
    pattern = re.compile(r"^eligible_(\d+)_(\d+)m$")
    cols: list[tuple[str, float, float]] = []
    for col in tx.columns:
        match = pattern.match(col)
        if match:
            cols.append((col, float(match.group(1)), float(match.group(2))))
    return sorted(cols, key=lambda item: item[1])


def unique_rx_count(rx: pd.DataFrame) -> int:
    return rx[["tx_node_id", "rx_node_id", "seq"]].drop_duplicates().shape[0]


def filter_eval_window(df: pd.DataFrame, start_s: float, end_s: float | None) -> pd.DataFrame:
    out = df[df["time_s"] >= start_s].copy()
    if end_s is not None:
        out = out[out["time_s"] <= end_s].copy()
    return out


def save_or_show(path: Path | None, show: bool) -> None:
    plt.tight_layout()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=200)
        print(f"Saved {path}")
    if show or path is None:
        plt.show()
    plt.close()


def main() -> None:
    args = parse_args()
    kpi_path = resolve_kpi_csv(args)
    tx_path, rx_path = log_paths(kpi_path)
    metadata = load_metadata(kpi_path)

    if not kpi_path.exists():
        raise FileNotFoundError(kpi_path)

    kpi = add_prr_alias(pd.read_csv(kpi_path))
    require_columns(
        kpi,
        kpi_path,
        [
            "time_s",
            "prr_awareness",
            "pir_s",
            "beacon_interval_s",
            "active_vehicle_count_core",
            "cbr",
        ],
    )

    warmup = args.warmup
    if warmup is None:
        warmup = float(metadata.get("evaluation_start_s", metadata.get("warmup_s", WARMUP_S)))

    cooldown = args.cooldown
    if cooldown is None:
        cooldown = float(metadata.get("cooldown_s", 0.0))

    eval_end = metadata.get("evaluation_end_s")
    if eval_end is not None:
        eval_end = float(eval_end)
    elif cooldown > 0.0 and not kpi.empty:
        eval_end = float(kpi["time_s"].max()) - cooldown

    awareness_range = args.awareness_range
    if awareness_range is None:
        awareness_range = float(metadata.get("awareness_range_m", AWARENESS_RANGE_M))

    kpi_eval = filter_eval_window(kpi, warmup, eval_end)
    if kpi_eval.empty:
        last_time = kpi["time_s"].max() if not kpi.empty else float("nan")
        raise ValueError(
            f"No KPI samples remain in evaluation window starting at {warmup:.1f} s "
            f"for {kpi_path}. Last KPI sample time is {last_time:.6g} s."
        )

    missing_logs = [path for path in (tx_path, rx_path) if not path.exists()]
    if missing_logs:
        if args.require_packet_logs:
            raise FileNotFoundError(missing_logs[0])
        kpi_only_validation(kpi_eval, kpi_path, warmup, eval_end, args)
        return

    tx = add_eligible_alias(pd.read_csv(tx_path))
    rx = pd.read_csv(rx_path)
    require_columns(tx, tx_path, ["time_s", "eligible_rx_count_awareness"])
    require_columns(
        rx,
        rx_path,
        ["time_s", "tx_node_id", "rx_node_id", "seq", "distance_m", "pir_s"],
    )
    tx_eval = filter_eval_window(tx, warmup, eval_end)
    rx_eval = filter_eval_window(rx, warmup, eval_end)

    # =========================
    # 1. GLOBAL PRR VALIDATION
    # =========================
    denominator = tx_eval["eligible_rx_count_awareness"].sum()
    numerator = unique_rx_count(rx_eval[rx_eval["distance_m"] <= awareness_range])
    prr_raw = numerator / denominator if denominator else 0.0

    print("\n===== PRR VALIDATION =====")
    print(f"KPI CSV             : {kpi_path}")
    print(f"TX log              : {tx_path}")
    print(f"RX log              : {rx_path}")
    print(f"Evaluation window    : {warmup:.1f} s to {eval_end if eval_end is not None else 'end'}")
    print(f"Awareness range      : {awareness_range:.1f} m")
    if metadata:
        axis = metadata.get("core_axis", "x")
        print(
            f"Core {axis}-region        : "
            f"{metadata.get('core_min_m', metadata.get('core_x_min_m', 'n/a'))} to "
            f"{metadata.get('core_max_m', metadata.get('core_x_max_m', 'n/a'))} m"
        )
    print(f"PRR from raw logs    : {prr_raw:.6f}")
    print(f"Mean PRR KPI samples : {kpi_eval['prr_awareness'].mean():.6f}")

    # =========================
    # 2. TIME-WINDOW PRR VALIDATION
    # =========================
    time_bins = np.arange(
        kpi_eval["time_s"].min(),
        kpi_eval["time_s"].max() + args.window,
        args.window,
    )
    prr_time = []
    time_centers = []

    for t1 in time_bins:
        t0 = t1 - args.window
        tx_w = tx_eval[(tx_eval["time_s"] >= t0) & (tx_eval["time_s"] <= t1)]
        rx_w = rx_eval[(rx_eval["time_s"] >= t0) & (rx_eval["time_s"] <= t1)]

        denom = tx_w["eligible_rx_count_awareness"].sum()
        num = unique_rx_count(rx_w[rx_w["distance_m"] <= awareness_range])

        if denom > 0:
            prr_time.append(num / denom)
            time_centers.append(t1)

    plt.figure()
    plt.plot(time_centers, prr_time, label="PRR recomputed from raw logs", linewidth=2)
    plt.plot(kpi_eval["time_s"], kpi_eval["prr_awareness"], "--", label="PRR from KPI logger")
    plt.xlabel("Time (s)")
    plt.ylabel("PRR")
    plt.title("PRR validation")
    plt.legend()
    plt.grid()
    save_or_show(
        args.out_dir / "prr_validation.png" if args.out_dir is not None else None,
        args.show,
    )

    # =========================
    # 3. PIR AND CBR SANITY
    # =========================
    pir_valid = rx_eval["pir_s"].dropna()

    print("\n===== PIR / CBR SANITY =====")
    print(f"Mean PIR from RX log : {pir_valid.mean():.6f}")
    print(f"Mean PIR from KPI    : {kpi_eval['pir_s'].mean():.6f}")
    print(f"Mean Tb             : {kpi_eval['beacon_interval_s'].mean():.6f}")
    print_optional_mean(kpi_eval, "active_vehicle_count_core", "Mean active vehicles")
    print_optional_mean(kpi_eval, "density_veh_per_km_core", "Mean density veh/km ")
    print_optional_mean(kpi_eval, "mean_neighbors_150m", "Mean neighbors 150m ")
    print_optional_mean(kpi_eval, "mean_neighbors_300m", "Mean neighbors 300m ")
    print(f"Mean PHY CBR        : {kpi_eval['cbr'].mean():.6f}")
    print_optional_mean(kpi_eval, "sensing_exclusion_ratio", "Mean sensing exclusion ratio")

    plt.figure()
    plt.scatter(kpi_eval["beacon_interval_s"], kpi_eval["pir_s"], alpha=0.5)
    plt.xlabel("Tb (s)")
    plt.ylabel("PIR (s)")
    plt.title("PIR vs beacon interval")
    plt.grid()
    save_or_show(
        args.out_dir / "pir_vs_beacon_interval.png" if args.out_dir is not None else None,
        args.show,
    )

    # =========================
    # 4. EXACT PRR vs DISTANCE
    # =========================
    centers = []
    prr_dist = []
    for col, low, high in eligible_bin_columns(tx_eval):
        denom = tx_eval[col].sum()
        num = unique_rx_count(
            rx_eval[(rx_eval["distance_m"] >= low) & (rx_eval["distance_m"] < high)]
        )
        if denom > 0:
            centers.append((low + high) / 2.0)
            prr_dist.append(num / denom)

    plt.figure()
    plt.plot(centers, prr_dist, marker="o")
    plt.xlabel("TX-time distance bin center (m)")
    plt.ylabel("PRR")
    plt.title("Exact PRR vs TX-time distance")
    plt.grid()
    save_or_show(
        args.out_dir / "prr_vs_distance.png" if args.out_dir is not None else None,
        args.show,
    )


if __name__ == "__main__":
    main()
