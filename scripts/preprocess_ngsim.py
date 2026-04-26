import json
import argparse
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
MAX_VEHICLES = 100  # default cap; override with --max-vehicles

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
def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess NGSIM US-101 tracks for ns-3 V2X runs.")
    parser.add_argument("--raw-path", type=Path, default=RAW_PATH, help="Input raw NGSIM CSV.")
    parser.add_argument("--out-path", type=Path, default=OUT_PATH, help="Output processed mobility CSV.")
    parser.add_argument("--cfg-path", type=Path, default=CFG_PATH, help="Output preprocessing config JSON.")
    parser.add_argument("--max-vehicles", type=int, default=MAX_VEHICLES, help="Maximum valid tracks to keep.")
    parser.add_argument("--time-start", type=float, default=TIME_START, help="Start of normalized time window [s].")
    parser.add_argument("--time-end", type=float, default=TIME_END, help="End of normalized time window [s].")
    parser.add_argument(
        "--min-track-duration",
        type=float,
        default=MIN_TRACK_DURATION,
        help="Minimum vehicle track duration to keep [s].",
    )
    parser.add_argument(
        "--lanes",
        type=str,
        default=",".join(str(lane) for lane in sorted(MAINLINE_LANES)),
        help="Comma-separated lane IDs to keep.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_vehicles <= 0:
        raise ValueError("--max-vehicles must be positive")
    lanes = {int(lane.strip()) for lane in args.lanes.split(",") if lane.strip()}
    if not lanes:
        raise ValueError("--lanes must include at least one lane ID")

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

    df = pd.read_csv(args.raw_path, usecols=list(rename_map), low_memory=False)
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
# ------

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
    df = df[(df["time_s"] >= args.time_start) & (df["time_s"] <= args.time_end)].copy()

    # ------------------------------
    # Lane filtering (mainline only)
    # ------------------------------
    df = df[df["lane_id"].isin(lanes)].copy()

    # ------------------------------
    # Process tracks
    # ------------------------------
    processed_tracks = []
    kept = []

    candidate_ids = df["vehicle_id"].drop_duplicates().to_numpy()
    print(f"Candidate vehicles after time/lane filters: {len(candidate_ids)}")

    grouped = {vid: g for vid, g in df.groupby("vehicle_id")}
    for vid in candidate_ids:
        g = grouped[vid]
        g = g.sort_values("time_s").drop_duplicates("time_s")

        duration = g["time_s"].iloc[-1] - g["time_s"].iloc[0]
        if duration < args.min_track_duration:
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
        if len(kept) >= args.max_vehicles:
            break

    if not processed_tracks:
        raise RuntimeError("No valid tracks after filtering.")
    if len(kept) < args.max_vehicles:
        raise RuntimeError(
            f"Requested {args.max_vehicles} vehicles, but only {len(kept)} valid tracks "
            "remained after filtering. Relax filters or use a larger raw data window."
        )

    # ------------------------------
    # Merge all tracks
    # ------------------------------
    out = pd.concat(processed_tracks, ignore_index=True)
    out = out.sort_values(["time_s", "vehicle_id"]).reset_index(drop=True)

    # ------------------------------
    # Save output
    # ------------------------------
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_path, index=False)

    # ------------------------------
    # Save config (for paper reproducibility)
    # ------------------------------
    lane_note = "All requested lanes" if lanes != MAINLINE_LANES else "Mainline only"
    cfg = {
        "dataset": "NGSIM US-101",
        "raw_path": str(args.raw_path),
        "mobility_csv": str(args.out_path),
        "time_window": [args.time_start, args.time_end],
        "sample_time": TARGET_DT,
        "min_track_duration_s": args.min_track_duration,
        "vehicles_kept": len(set(kept)),
        "lane_filter": sorted(lanes),
        "notes": f"{lane_note}, resampled to 0.1s, smoothed, filtered",
    }

    args.cfg_path.parent.mkdir(parents=True, exist_ok=True)
    args.cfg_path.write_text(json.dumps(cfg, indent=2))

    print("====================================")
    print(f"Saved: {args.out_path}")
    print(f"Config: {args.cfg_path}")
    print(f"Vehicles kept: {len(set(kept))}")
    print("====================================")


if __name__ == "__main__":
    main()
