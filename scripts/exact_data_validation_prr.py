import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TX_PATH = Path("data/output/kpi_timeseries_run01_tx_packet_log.csv")
RX_PATH = Path("data/output/kpi_timeseries_run01_rx_packet_log.csv")
AWARENESS_RANGE_M = 300.0
WARMUP_S = 5.0


def eligible_bin_columns(tx: pd.DataFrame) -> list[tuple[str, float, float]]:
    cols: list[tuple[str, float, float]] = []
    pattern = re.compile(r"^eligible_(\d+)_(\d+)m$")
    for col in tx.columns:
        match = pattern.match(col)
        if match:
            cols.append((col, float(match.group(1)), float(match.group(2))))
    return sorted(cols, key=lambda item: item[1])


def main() -> None:
    tx = pd.read_csv(TX_PATH)
    rx = pd.read_csv(RX_PATH)
    if "eligible_rx_count_awareness" not in tx and "eligible_rx_count_150m" in tx:
        tx["eligible_rx_count_awareness"] = tx["eligible_rx_count_150m"]

    tx = tx[tx["time_s"] >= WARMUP_S].copy()
    rx = rx[rx["time_s"] >= WARMUP_S].copy()

    denom_awareness = tx["eligible_rx_count_awareness"].sum()
    numer_awareness = rx[rx["distance_m"] <= AWARENESS_RANGE_M][
        ["tx_node_id", "rx_node_id", "seq"]
    ].drop_duplicates().shape[0]
    prr_awareness = numer_awareness / denom_awareness if denom_awareness else 0.0

    print("===== Exact PRR validation =====")
    print(f"Warm-up cutoff       : {WARMUP_S:.1f} s")
    print(f"Awareness range      : {AWARENESS_RANGE_M:.1f} m")
    print(f"Eligible awareness   : {denom_awareness}")
    print(f"Unique RX awareness  : {numer_awareness}")
    print(f"Exact PRR awareness  : {prr_awareness:.6f}")

    bins = eligible_bin_columns(tx)
    if not bins:
        raise ValueError("No eligible distance-bin columns found in TX log")

    centers = []
    prr_by_distance = []
    for col, low, high in bins:
        denom = tx[col].sum()
        numer = rx[(rx["distance_m"] >= low) & (rx["distance_m"] < high)][
            ["tx_node_id", "rx_node_id", "seq"]
        ].drop_duplicates().shape[0]
        if denom > 0:
            centers.append((low + high) / 2.0)
            prr_by_distance.append(numer / denom)
            print(f"PRR [{low:.0f}, {high:.0f}) m : {numer}/{denom} = {numer / denom:.6f}")

    plt.figure()
    plt.plot(centers, prr_by_distance, marker="o")
    plt.xlabel("TX-time distance bin center (m)")
    plt.ylabel("PRR")
    plt.title("Exact PRR vs TX-time distance")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
