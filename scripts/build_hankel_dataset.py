from pathlib import Path
import pandas as pd


IN_PATH = Path("data/output/kpi_timeseries_run01.csv")
OUT_PATH = Path("data/output/deepc_dataset_run01.csv")
WARMUP_S = 5.0


def main() -> None:
    df = pd.read_csv(IN_PATH)
    df = df[df["time_s"] >= WARMUP_S].copy()

    keep = [
        "time_s",
        "tx_power_dbm",
        "beacon_interval_s",
        "active_vehicle_count_core",
        "density_veh_per_km_core",
        "mean_neighbors_150m",
        "mean_neighbors_300m",
        "prr_150m",
        "pir_s",
        "cbr",
        "sensing_exclusion_ratio",
    ]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    out = df[keep].dropna().copy()
    out.to_csv(OUT_PATH, index=False)
    print(f"Saved DeePC dataset to: {OUT_PATH}")
    print(f"Warm-up cutoff: {WARMUP_S} s")
    print(f"Rows kept: {len(out)}")


if __name__ == "__main__":
    main()
