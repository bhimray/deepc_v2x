import json
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ==============================
# PATHS
# ==============================
RAW_PATH = Path("data/raw/us101/us101_raw.csv")
OUT_PATH = Path("data/processed/ngsim_us101_mainline_0p1s.csv")
CFG_PATH = Path("data/processed/run_config.json")
ACTIVE_OUT_PATH = Path("data/processed/ngsim_us101_mainline_active20_250_densest_600s.csv")
ACTIVE_CFG_PATH = Path("data/processed/run_config_active20_250_densest_600s.json")

# ==============================
# PARAMETERS 
# ==============================
TARGET_DT = 0.1  # 100 ms
MIN_TRACK_DURATION = 20.0  # seconds
ACTIVE_MIN_TRACK_DURATION = 1.0  # seconds
MAX_ABS_SPEED_MPS = 60.0
MAX_POSITION_JUMP_M = 20.0
MAX_VEHICLES = 100  # default cap; override with --max-vehicles
ACTIVE_MIN_VEHICLES = 20
ACTIVE_MAX_VEHICLES = 250
ACTIVE_DURATION_S = 600.0
ACTIVE_MAX_WINDOW_ATTEMPTS = 50
CSV_CHUNK_SIZE = 500_000

# TIME WINDOW (VERY IMPORTANT)
TIME_START = 0.0
TIME_END = 600.0

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

    t_new = np.round(np.arange(t0, t1 + 1e-9, TARGET_DT), 10)

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


def parse_lanes(lanes_arg):
    lanes = {int(lane.strip()) for lane in lanes_arg.split(",") if lane.strip()}
    if not lanes:
        raise ValueError("--lanes must include at least one lane ID")
    return lanes


def sample_step_ms():
    step_ms = TARGET_DT * 1000.0
    rounded = int(round(step_ms))
    if not np.isclose(step_ms, rounded):
        raise ValueError(f"TARGET_DT={TARGET_DT} cannot be represented as an integer millisecond step")
    return rounded


def sample_count_for_duration(duration_s):
    steps = duration_s / TARGET_DT
    rounded = int(round(steps))
    if not np.isclose(steps, rounded):
        raise ValueError(f"Duration {duration_s} s is not an integer multiple of TARGET_DT={TARGET_DT}")
    return rounded + 1


def raw_column_map():
    return {
        "Vehicle_ID": "vehicle_id",
        "Global_Time": "global_time_ms",
        "Local_X": "x_ft",
        "Local_Y": "y_ft",
        "Lane_ID": "lane_id",
    }


def raw_required_columns():
    return ["vehicle_id", "global_time_ms", "x_ft", "y_ft", "lane_id"]


def parse_raw_int(value):
    return int(str(value).replace(",", "").strip())


def clean_raw_dataframe(df):
    required = raw_required_columns()
    original_rows = len(df)
    for col in required:
        if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
            df[col] = df[col].astype("string").str.replace(",", "", regex=False).str.strip()
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required).copy()
    dropped_rows = original_rows - len(df)
    if dropped_rows:
        print(f"Dropped malformed rows: {dropped_rows}")

    df["vehicle_id"] = df["vehicle_id"].astype(int)
    df["global_time_ms"] = df["global_time_ms"].astype("int64")
    df["lane_id"] = df["lane_id"].astype(int)
    df["x_m"] = feet_to_m(df["x_ft"])
    df["y_m"] = feet_to_m(df["y_ft"])

    return df


def load_raw_dataframe(raw_path):
    # Column mapping for the raw CSV header.
    rename_map = raw_column_map()

    df = pd.read_csv(raw_path, usecols=list(rename_map), low_memory=False)
    df = df.rename(columns=rename_map)

    required = raw_required_columns()
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    return clean_raw_dataframe(df)


def scan_active_counts(raw_path, lanes):
    ids_by_time = defaultdict(set)
    rows = 0
    kept_rows = 0
    with raw_path.open(newline="") as raw_file:
        reader = csv.DictReader(raw_file)
        for row in reader:
            rows += 1
            try:
                lane_id = parse_raw_int(row["Lane_ID"])
                if lane_id not in lanes:
                    continue
                global_time_ms = parse_raw_int(row["Global_Time"])
                vehicle_id = parse_raw_int(row["Vehicle_ID"])
            except (KeyError, TypeError, ValueError):
                continue
            ids_by_time[global_time_ms].add(vehicle_id)
            kept_rows += 1

    print(f"Scanned raw rows: {rows}")
    print(f"Rows in selected lanes: {kept_rows}")
    if not ids_by_time:
        raise RuntimeError("No raw rows found for the selected lane filter.")

    return pd.Series({time_ms: len(ids) for time_ms, ids in ids_by_time.items()}).sort_index()


def load_active_window_dataframe(raw_path, lanes, start_global_ms, end_global_ms):
    rename_map = raw_column_map()
    frames = []
    for chunk in pd.read_csv(
        raw_path,
        usecols=list(rename_map),
        low_memory=False,
        chunksize=CSV_CHUNK_SIZE,
    ):
        chunk = chunk.rename(columns=rename_map)
        chunk = clean_raw_dataframe(chunk)
        chunk = chunk[
            chunk["lane_id"].isin(lanes) &
            (chunk["global_time_ms"] >= start_global_ms) &
            (chunk["global_time_ms"] <= end_global_ms)
        ].copy()
        if not chunk.empty:
            frames.append(chunk)

    if not frames:
        raise RuntimeError("Selected active window produced no raw rows.")

    return pd.concat(frames, ignore_index=True)


def active_count_stats(values):
    values = np.asarray(values, dtype=np.int64)
    return {
        "min": int(values.min()),
        "max": int(values.max()),
        "mean": float(values.mean()),
    }


def find_active_window_candidates(counts, min_active, max_active, duration_s):
    expected_samples = sample_count_for_duration(duration_s)
    step_ms = sample_step_ms()

    counts = counts.sort_index()
    times = counts.index.to_numpy(dtype=np.int64)
    values = counts.to_numpy(dtype=np.int64)

    candidates = []

    def add_run(run_times, run_counts):
        if len(run_counts) < expected_samples:
            return
        run_counts = np.asarray(run_counts, dtype=np.int64)
        prefix = np.concatenate(([0], np.cumsum(run_counts, dtype=np.int64)))
        totals = prefix[expected_samples:] - prefix[:-expected_samples]
        for i, total in enumerate(totals):
            start = int(run_times[i])
            end = int(run_times[i + expected_samples - 1])
            candidates.append({
                "start_global_ms": start,
                "end_global_ms": end,
                "active_count_sum": int(total),
                "sample_count": expected_samples,
            })

    run_times = []
    run_counts = []
    previous_time = None
    for time_ms, count in zip(times, values):
        is_valid_count = min_active <= count <= max_active
        is_contiguous = previous_time is None or time_ms == previous_time + step_ms
        if is_valid_count and (not run_times or is_contiguous):
            run_times.append(time_ms)
            run_counts.append(count)
        else:
            add_run(run_times, run_counts)
            run_times = [time_ms] if is_valid_count else []
            run_counts = [count] if is_valid_count else []
        previous_time = time_ms

    add_run(run_times, run_counts)

    candidates.sort(key=lambda c: (c["active_count_sum"], -c["start_global_ms"]), reverse=True)
    return candidates, counts


def build_window_dataframe(df, start_global_ms, end_global_ms):
    step_ms = sample_step_ms()
    window = df[
        (df["global_time_ms"] >= start_global_ms) &
        (df["global_time_ms"] <= end_global_ms)
    ].copy()
    sample_index = np.rint((window["global_time_ms"] - start_global_ms) / step_ms).astype(int)
    window["time_s"] = np.round(sample_index * TARGET_DT, 10)
    return window


def process_tracks(df, min_track_duration, vehicle_limit=None):
    processed_tracks = []
    kept = []
    rejected = {
        "short_duration": 0,
        "speed": 0,
        "position_jump": 0,
    }

    candidate_ids = df.sort_values(["global_time_ms", "vehicle_id"])["vehicle_id"].drop_duplicates().to_numpy()
    print(f"Candidate vehicles after time/lane filters: {len(candidate_ids)}")

    grouped = {vid: g for vid, g in df.groupby("vehicle_id")}
    for vid in candidate_ids:
        g = grouped[vid]
        g = g.sort_values("time_s").drop_duplicates("time_s")

        duration = g["time_s"].iloc[-1] - g["time_s"].iloc[0]
        if duration < min_track_duration:
            rejected["short_duration"] += 1
            continue

        track = interpolate_track(g)

        dx = np.diff(track["x_m"], prepend=track["x_m"].iloc[0])
        dy = np.diff(track["y_m"], prepend=track["y_m"].iloc[0])
        jump = np.sqrt(dx**2 + dy**2)

        if np.any(track["speed_mps"] > MAX_ABS_SPEED_MPS):
            rejected["speed"] += 1
            continue

        if np.any(jump > MAX_POSITION_JUMP_M):
            rejected["position_jump"] += 1
            continue

        processed_tracks.append(track)
        kept.append(vid)
        if vehicle_limit is not None and len(kept) >= vehicle_limit:
            break

    if not processed_tracks:
        return pd.DataFrame(), kept, rejected

    out = pd.concat(processed_tracks, ignore_index=True)
    out = out.sort_values(["time_s", "vehicle_id"]).reset_index(drop=True)
    return out, kept, rejected


def validate_active_output(out, min_active, max_active, duration_s):
    expected_samples = sample_count_for_duration(duration_s)
    expected_index = pd.Index(range(expected_samples), name="sample_index")
    sample_index = np.rint(out["time_s"] / TARGET_DT).astype(int)

    out_with_sample = out.assign(sample_index=sample_index)
    counts = out_with_sample.groupby("sample_index")["vehicle_id"].nunique()
    counts = counts.reindex(expected_index, fill_value=0)
    duplicates = int(out_with_sample.duplicated(["sample_index", "vehicle_id"]).sum())

    stats = active_count_stats(counts.to_numpy())
    time_min = float(out["time_s"].min()) if not out.empty else np.nan
    time_max = float(out["time_s"].max()) if not out.empty else np.nan
    expected_time_max = (expected_samples - 1) * TARGET_DT

    ok = (
        stats["min"] >= min_active and
        stats["max"] <= max_active and
        duplicates == 0 and
        np.isclose(time_min, 0.0) and
        np.isclose(time_max, expected_time_max)
    )

    return {
        "ok": bool(ok),
        "sample_count": int(len(counts)),
        "min": stats["min"],
        "max": stats["max"],
        "mean": stats["mean"],
        "duplicate_time_vehicle_rows": duplicates,
        "time_min_s": time_min,
        "time_max_s": time_max,
    }


def raw_window_active_stats(counts, start_global_ms, end_global_ms):
    expected_times = np.arange(start_global_ms, end_global_ms + sample_step_ms(), sample_step_ms())
    window_counts = counts.reindex(expected_times, fill_value=0).to_numpy(dtype=np.int64)
    return active_count_stats(window_counts)


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
        default=None,
        help=(
            "Minimum vehicle track duration to keep [s]. Defaults to 20.0 in legacy mode "
            "and 1.0 in active-window mode."
        ),
    )
    parser.add_argument(
        "--lanes",
        type=str,
        default=",".join(str(lane) for lane in sorted(MAINLINE_LANES)),
        help="Comma-separated lane IDs to keep.",
    )
    parser.add_argument(
        "--active-window",
        action="store_true",
        help="Select a raw time window whose per-sample active unique vehicle count satisfies bounds.",
    )
    parser.add_argument(
        "--active-min-vehicles",
        type=int,
        default=ACTIVE_MIN_VEHICLES,
        help="Minimum active unique vehicles required at every sample in active-window mode.",
    )
    parser.add_argument(
        "--active-max-vehicles",
        type=int,
        default=ACTIVE_MAX_VEHICLES,
        help="Maximum active unique vehicles allowed at every sample in active-window mode.",
    )
    parser.add_argument(
        "--active-duration",
        type=float,
        default=ACTIVE_DURATION_S,
        help="Selected active-window duration [s].",
    )
    parser.add_argument(
        "--active-selection",
        choices=["densest"],
        default="densest",
        help="Policy for choosing among valid active windows.",
    )
    parser.add_argument(
        "--active-max-window-attempts",
        type=int,
        default=ACTIVE_MAX_WINDOW_ATTEMPTS,
        help="Maximum raw active-window candidates to try after post-processing validation failures; <=0 means all.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    lanes = parse_lanes(args.lanes)
    min_track_duration = (
        ACTIVE_MIN_TRACK_DURATION if args.active_window else MIN_TRACK_DURATION
    ) if args.min_track_duration is None else args.min_track_duration

    if min_track_duration < 0:
        raise ValueError("--min-track-duration must be non-negative")
    if args.active_window:
        if args.out_path == OUT_PATH:
            args.out_path = ACTIVE_OUT_PATH
        if args.cfg_path == CFG_PATH:
            args.cfg_path = ACTIVE_CFG_PATH
        if args.active_min_vehicles <= 0:
            raise ValueError("--active-min-vehicles must be positive")
        if args.active_max_vehicles < args.active_min_vehicles:
            raise ValueError("--active-max-vehicles must be >= --active-min-vehicles")
    elif args.max_vehicles <= 0:
        raise ValueError("--max-vehicles must be positive")

    lane_note = "All requested lanes" if lanes != MAINLINE_LANES else "Mainline only"

    active_metadata = None
    if args.active_window:
        print(
            "Selecting active window: "
            f"{args.active_min_vehicles}-{args.active_max_vehicles} active vehicles, "
            f"{args.active_duration:.1f} s, {args.active_selection}"
        )
        counts = scan_active_counts(args.raw_path, lanes)
        candidates, counts = find_active_window_candidates(
            counts,
            args.active_min_vehicles,
            args.active_max_vehicles,
            args.active_duration,
        )
        if not candidates:
            raise RuntimeError(
                "No active-window candidate satisfies the requested per-sample vehicle bounds."
            )

        max_attempts = len(candidates)
        if args.active_max_window_attempts > 0:
            max_attempts = min(max_attempts, args.active_max_window_attempts)

        out = pd.DataFrame()
        kept = []
        rejected = {}
        validation = None
        selected_candidate = None
        selected_window = None
        raw_stats = None
        for attempt, candidate in enumerate(candidates[:max_attempts], start=1):
            print(
                "Trying active-window candidate "
                f"{attempt}/{max_attempts}: "
                f"{candidate['start_global_ms']}..{candidate['end_global_ms']}"
            )
            window = load_active_window_dataframe(
                args.raw_path,
                lanes,
                candidate["start_global_ms"],
                candidate["end_global_ms"],
            )
            window = build_window_dataframe(
                window,
                candidate["start_global_ms"],
                candidate["end_global_ms"],
            )
            out, kept, rejected = process_tracks(
                window,
                min_track_duration=min_track_duration,
                vehicle_limit=None,
            )
            if out.empty:
                print("Candidate rejected: no valid tracks after preprocessing.")
                continue

            validation = validate_active_output(
                out,
                args.active_min_vehicles,
                args.active_max_vehicles,
                args.active_duration,
            )
            if validation["ok"]:
                selected_candidate = candidate
                selected_window = window
                raw_stats = raw_window_active_stats(
                    counts,
                    candidate["start_global_ms"],
                    candidate["end_global_ms"],
                )
                break

            print(
                "Candidate rejected after preprocessing: "
                f"min={validation['min']} max={validation['max']} "
                f"duplicates={validation['duplicate_time_vehicle_rows']}"
            )

        if selected_candidate is None:
            raise RuntimeError(
                f"No active-window candidate passed final validation after {max_attempts} attempts."
            )

        active_metadata = {
            "active_window_mode": True,
            "active_selection": args.active_selection,
            "selected_global_time_window_ms": [
                selected_candidate["start_global_ms"],
                selected_candidate["end_global_ms"],
            ],
            "active_vehicle_bounds_per_sample": {
                "min": args.active_min_vehicles,
                "max": args.active_max_vehicles,
            },
            "active_duration_s": args.active_duration,
            "raw_window_active_unique": raw_stats,
            "final_active_unique": {
                "min": validation["min"],
                "max": validation["max"],
                "mean": validation["mean"],
            },
            "final_validation": validation,
            "raw_unique_vehicle_ids_in_window": int(selected_window["vehicle_id"].nunique()),
            "vehicle_cap_applied": False,
            "rejected_tracks": rejected,
        }
    else:
        print("Loading raw NGSIM data...")
        df = load_raw_dataframe(args.raw_path)
        df["time_s"] = np.round((df["global_time_ms"] - df["global_time_ms"].min()) / 1000.0, 10)
        df = df[(df["time_s"] >= args.time_start) & (df["time_s"] <= args.time_end)].copy()
        df = df[df["lane_id"].isin(lanes)].copy()

        out, kept, rejected = process_tracks(
            df,
            min_track_duration=min_track_duration,
            vehicle_limit=args.max_vehicles,
        )

        if out.empty:
            raise RuntimeError("No valid tracks after filtering.")
        if len(kept) < args.max_vehicles:
            raise RuntimeError(
                f"Requested {args.max_vehicles} vehicles, but only {len(kept)} valid tracks "
                "remained after filtering. Relax filters or use a larger raw data window."
            )

    # ------------------------------
    # Save output
    # ------------------------------
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_path, index=False)

    # ------------------------------
    # Save config (for paper reproducibility)
    # ------------------------------
    cfg = {
        "dataset": "NGSIM US-101",
        "raw_path": str(args.raw_path),
        "mobility_csv": str(args.out_path),
        "time_window": [args.time_start, args.time_end],
        "sample_time": TARGET_DT,
        "min_track_duration_s": min_track_duration,
        "vehicles_kept": len(set(kept)),
        "lane_filter": sorted(lanes),
        "notes": f"{lane_note}, resampled to 0.1s, smoothed, filtered",
    }
    if active_metadata:
        cfg.update(active_metadata)
        cfg["time_window"] = [0.0, args.active_duration]

    args.cfg_path.parent.mkdir(parents=True, exist_ok=True)
    args.cfg_path.write_text(json.dumps(cfg, indent=2))

    print("====================================")
    print(f"Saved: {args.out_path}")
    print(f"Config: {args.cfg_path}")
    print(f"Vehicles kept: {len(set(kept))}")
    if active_metadata:
        print(
            "Final active unique vehicles per sample: "
            f"min={active_metadata['final_active_unique']['min']} "
            f"max={active_metadata['final_active_unique']['max']} "
            f"mean={active_metadata['final_active_unique']['mean']:.2f}"
        )
    print("====================================")


if __name__ == "__main__":
    main()
