import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


KPI_PATH = Path("data/output/kpi_timeseries_run01.csv")
TX_PATH = Path("data/output/kpi_timeseries_run01_tx_packet_log.csv")
RX_PATH = Path("data/output/kpi_timeseries_run01_rx_packet_log.csv")

AWARENESS_RANGE_M = 150.0
KPI_WINDOW_S = 1.0
WARMUP_S = 5.0


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


def main() -> None:
    kpi = pd.read_csv(KPI_PATH)
    tx = pd.read_csv(TX_PATH)
    rx = pd.read_csv(RX_PATH)

    kpi_eval = kpi[kpi["time_s"] >= WARMUP_S].copy()
    tx_eval = tx[tx["time_s"] >= WARMUP_S].copy()
    rx_eval = rx[rx["time_s"] >= WARMUP_S].copy()

    # =========================
    # 1. GLOBAL PRR VALIDATION
    # =========================
    denominator = tx_eval["eligible_rx_count_150m"].sum()
    numerator = unique_rx_count(rx_eval[rx_eval["distance_m"] <= AWARENESS_RANGE_M])
    prr_raw = numerator / denominator if denominator else 0.0

    print("\n===== PRR VALIDATION =====")
    print(f"Warm-up cutoff       : {WARMUP_S:.1f} s")
    print(f"PRR from raw logs    : {prr_raw:.6f}")
    print(f"Mean PRR KPI samples : {kpi_eval['prr_150m'].mean():.6f}")

    # =========================
    # 2. TIME-WINDOW PRR VALIDATION
    # =========================
    time_bins = np.arange(kpi_eval["time_s"].min(), kpi_eval["time_s"].max(), KPI_WINDOW_S)
    prr_time = []
    time_centers = []

    for t1 in time_bins:
        t0 = t1 - KPI_WINDOW_S
        tx_w = tx[(tx["time_s"] >= t0) & (tx["time_s"] <= t1)]
        rx_w = rx[(rx["time_s"] >= t0) & (rx["time_s"] <= t1)]

        denom = tx_w["eligible_rx_count_150m"].sum()
        num = unique_rx_count(rx_w[rx_w["distance_m"] <= AWARENESS_RANGE_M])

        if denom > 0:
            prr_time.append(num / denom)
            time_centers.append(t1)

    plt.figure()
    plt.plot(time_centers, prr_time, label="PRR recomputed from raw logs", linewidth=2)
    plt.plot(kpi_eval["time_s"], kpi_eval["prr_150m"], "--", label="PRR from KPI logger")
    plt.xlabel("Time (s)")
    plt.ylabel("PRR")
    plt.title("PRR validation")
    plt.legend()
    plt.grid()
    plt.show()

    # =========================
    # 3. PIR AND CBR SANITY
    # =========================
    pir_valid = rx_eval["pir_s"].dropna()

    print("\n===== PIR / CBR SANITY =====")
    print(f"Mean PIR from RX log : {pir_valid.mean():.6f}")
    print(f"Mean PIR from KPI    : {kpi_eval['pir_s'].mean():.6f}")
    print(f"Mean Tb             : {kpi_eval['beacon_interval_s'].mean():.6f}")
    print(f"Mean CBR            : {kpi_eval['cbr'].mean():.6f}")

    plt.figure()
    plt.scatter(kpi_eval["beacon_interval_s"], kpi_eval["pir_s"], alpha=0.5)
    plt.xlabel("Tb (s)")
    plt.ylabel("PIR (s)")
    plt.title("PIR vs beacon interval")
    plt.grid()
    plt.show()

    # =========================
    # 4. EXACT PRR vs DISTANCE
    # =========================
    centers = []
    prr_dist = []
    for col, low, high in eligible_bin_columns(tx_eval):
        denom = tx_eval[col].sum()
        num = unique_rx_count(rx_eval[(rx_eval["distance_m"] >= low) & (rx_eval["distance_m"] < high)])
        if denom > 0:
            centers.append((low + high) / 2.0)
            prr_dist.append(num / denom)

    plt.figure()
    plt.plot(centers, prr_dist, marker="o")
    plt.xlabel("TX-time distance bin center (m)")
    plt.ylabel("PRR")
    plt.title("Exact PRR vs TX-time distance")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
