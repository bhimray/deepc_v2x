import json
from pathlib import Path

import numpy as np
import pandas as pd

# ==============================
# PATHS
# ==============================
RAW_PATH = Path("data/raw/us101/us101_raw.csv")
OUT_PATH = Path("data/processed/ngsim_us101_mainline_0p1s.csv")
CFG_PATH = Path("data/processed/run_config.json")

# ==============================
# PARAMETERS 
# ==============================
TARGET_DT = 0.1  # 100 msv2x_env
MIN_TRACK_DURATION = 20.0  # seconds
MAX_ABS_SPEED_MPS = 60.0
MAX_POSITION_JUMP_M = 20.0
MAX_VEHICLES = 100  # start small, increase later

# TIME WINDOW (VERY IMPORTANT)
TIME_START = 100.0
TIME_END = 400.0

# MAINLINE LANES (adjust after inspection)
MAINLINE_LANES = {2, 3, 4, 5}


# ==============================
# UTILS
# ==============================
def feet_to_m(x):
    return x * 0.3048


def smooth(x, window=5):
    if len(x) < window:
        return x
    return pd.Series(x).rolling(window=window, center=True, min_periods=1).mean().to_numpy()


def interpolate_track(df_track):
    t0 = df_track["time_s"].min()
    t1 = df_track["time_s"].max()

    t_new = np.arange(t0, t1 + 1e-9, TARGET_DT)

    x = np.interp(t_new, df_track["time_s"], df_track["x_m"])
    y = np.interp(t_new, df_track["time_s"], df_track["y_m"])

    x = smooth(x)
    y = smooth(y)

    vx = np.gradient(x, TARGET_DT)
    vy = np.gradient(y, TARGET_DT)
    speed = np.sqrt(vx**2 + vy**2)

    return pd.DataFrame({
        "time_s": t_new,
        "vehicle_id": int(df_track["vehicle_id"].iloc[0]),
        "x_m": x,
        "y_m": y,
        "vx_mps": vx,
        "vy_mps": vy,
        "speed_mps": speed,
        "lane_id": int(df_track["lane_id"].mode().iloc[0]),
    })


# ==============================
# MAIN
# ==============================
def main():
    print("Loading raw NGSIM data...")

    # ------------------------------
    # Column mapping for the raw CSV header.
    # ------------------------------
    rename_map = {
        "Vehicle_ID": "vehicle_id",
        "Global_Time": "global_time_ms",
        "Local_X": "x_ft",
        "Local_Y": "y_ft",
        "Lane_ID": "lane_id",
    }

    df = pd.read_csv(RAW_PATH, usecols=list(rename_map), low_memory=False)
    df = df.rename(columns=rename_map)

    required = ["vehicle_id", "global_time_ms", "x_ft", "y_ft", "lane_id"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    # ------------------------------
    # Numeric cleanup
    # ------------------------------
    before_numeric = df[required].copy()
    original_rows = len(df)
    for col in required:
        if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
            df[col] = df[col].astype("string").str.replace(",", "", regex=False).str.strip()
        df[col] = pd.to_numeric(df[col], errors="coerce")
  
#----debug
    bad_mask = df[required].isna().any(axis=1)

    print("NaN count by column:")
    print(df[required].isna().sum())

    print("Original values that failed conversion:")
    print(before_numeric[bad_mask].head(20).to_string())

    print("Converted values:")
    print(df.loc[bad_mask, required].head(20).to_string())
# ------------------------------

    df = df.dropna(subset=required).copy()
    dropped_rows = original_rows - len(df)
    if dropped_rows:
        print(f"Dropped malformed rows: {dropped_rows}")

    df["vehicle_id"] = df["vehicle_id"].astype(int)
    df["lane_id"] = df["lane_id"].astype(int)

    # ------------------------------
    # Time normalization
    # ------------------------------
    df["time_s"] = (df["global_time_ms"] - df["global_time_ms"].min()) / 1000.0

    # ------------------------------
    # Unit conversion
    # ------------------------------
    df["x_m"] = feet_to_m(df["x_ft"])
    df["y_m"] = feet_to_m(df["y_ft"])

    # ------------------------------
    # Time window filtering
    # ------------------------------
    df = df[(df["time_s"] >= TIME_START) & (df["time_s"] <= TIME_END)].copy()

    # ------------------------------
    # Lane filtering (mainline only)
    # ------------------------------
    df = df[df["lane_id"].isin(MAINLINE_LANES)].copy()

    # ------------------------------
    # Limit number of vehicles
    # ------------------------------
    vehicle_ids = df["vehicle_id"].unique()
    selected_ids = vehicle_ids[:MAX_VEHICLES]
    df = df[df["vehicle_id"].isin(selected_ids)]

    print(f"Selected vehicles: {len(selected_ids)}")

    # ------------------------------
    # Process tracks
    # ------------------------------
    processed_tracks = []
    kept = []

    for vid, g in df.groupby("vehicle_id"):
        g = g.sort_values("time_s").drop_duplicates("time_s")

        duration = g["time_s"].iloc[-1] - g["time_s"].iloc[0]
        if duration < MIN_TRACK_DURATION:
            continue

        track = interpolate_track(g)

        # --------------------------
        # Sanity checks
        # --------------------------
        dx = np.diff(track["x_m"], prepend=track["x_m"].iloc[0])
        dy = np.diff(track["y_m"], prepend=track["y_m"].iloc[0])
        jump = np.sqrt(dx**2 + dy**2)

        if np.any(track["speed_mps"] > MAX_ABS_SPEED_MPS):
            continue

        if np.any(jump > MAX_POSITION_JUMP_M):
            continue

        processed_tracks.append(track)
        kept.append(vid)

    if not processed_tracks:
        raise RuntimeError("No valid tracks after filtering.")

    # ------------------------------
    # Merge all tracks
    # ------------------------------
    out = pd.concat(processed_tracks, ignore_index=True)
    out = out.sort_values(["time_s", "vehicle_id"]).reset_index(drop=True)

    # ------------------------------
    # Save output
    # ------------------------------
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    # ------------------------------
    # Save config (for paper reproducibility)
    # ------------------------------
    cfg = {
        "dataset": "NGSIM US-101",
        "time_window": [TIME_START, TIME_END],
        "sample_time": TARGET_DT,
        "vehicles_kept": len(set(kept)),
        "lane_filter": sorted(MAINLINE_LANES),
        "notes": "Mainline only, resampled to 0.1s, smoothed, filtered",
    }

    CFG_PATH.write_text(json.dumps(cfg, indent=2))

    print("====================================")
    print(f"Saved: {OUT_PATH}")
    print(f"Vehicles kept: {len(set(kept))}")
    print("====================================")


if __name__ == "__main__":
    main()
