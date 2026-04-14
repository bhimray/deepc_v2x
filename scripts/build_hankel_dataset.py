from pathlib import Path
import pandas as pd


IN_PATH = Path("data/output/kpi_timeseries_run01.csv")
OUT_PATH = Path("data/output/deepc_dataset_run01.csv")


def main() -> None:
    df = pd.read_csv(IN_PATH)

    keep = [
        "time_s",
        "tx_power_dbm",
        "beacon_interval_s",
        "prr_150m",
        "cbr",
        "pir_s",
    ]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    out = df[keep].copy()
    out.to_csv(OUT_PATH, index=False)
    print(f"Saved DeePC dataset to: {OUT_PATH}")


if __name__ == "__main__":
    main()