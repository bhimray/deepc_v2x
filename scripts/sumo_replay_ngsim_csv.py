#!/usr/bin/env python3
"""Replay processed NGSIM CSV trajectories in SUMO GUI using TraCI.

This is a visual/debug replay tool. SUMO does not simulate car-following here;
the CSV positions are imposed with traci.vehicle.moveToXY().
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
import hashlib

import pandas as pd


ROAD_MARGIN_M = 50.0
LATERAL_MARGIN_M = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/processed/ngsim_us101_mainline_active20_250_densest_600s.csv"),
        help="Processed mobility CSV with time_s,vehicle_id,x_m,y_m columns.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/output/sumo_replay/ngsim_us101_replay"),
        help="Directory for generated SUMO files.",
    )
    parser.add_argument("--begin", type=float, default=0.0, help="Replay start time in seconds.")
    parser.add_argument("--end", type=float, default=60.0, help="Replay end time in seconds.")
    parser.add_argument("--step-length", type=float, default=0.1, help="SUMO step length.")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only generate SUMO network/config files; do not start TraCI replay.",
    )
    parser.add_argument(
        "--sumo-binary",
        default="sumo-gui",
        help="SUMO binary to launch, usually sumo-gui or sumo.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="GUI delay in seconds per step. Use 0 for fastest.",
    )
    parser.add_argument(
        "--orientation",
        choices=["horizontal", "vertical"],
        default="horizontal",
        help=(
            "Replay orientation. horizontal maps CSV y_m to SUMO x and CSV x_m to SUMO y, "
            "which fits the NGSIM road across the screen."
        ),
    )
    parser.add_argument(
        "--view-zoom",
        type=float,
        default=2.0,
        help="Initial SUMO GUI viewport zoom. Lower values show more of the road.",
    )
    parser.add_argument(
        "--lane-width",
        type=float,
        default=3.7,
        help="SUMO lane width in meters. 3.7 m matches the processed US-101 lateral spacing.",
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame, path: Path) -> None:
    required = ["time_s", "vehicle_id", "x_m", "y_m"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise SystemExit(f"{path} is missing required columns: {missing}")


def transform_xy(row, orientation: str, transform: dict[str, float]) -> tuple[float, float]:
    if orientation == "horizontal":
        return (
            float(row.y_m) - transform["csv_y_min"] + ROAD_MARGIN_M,
            float(row.x_m) - transform["csv_x_center"],
        )
    return (
        float(row.x_m) - transform["csv_x_center"],
        float(row.y_m) - transform["csv_y_min"] + ROAD_MARGIN_M,
    )


def vehicle_color(veh_id: str) -> tuple[int, int, int, int]:
    digest = hashlib.blake2s(veh_id.encode("utf-8"), digest_size=3).digest()
    # Keep colors bright enough to distinguish on SUMO's dark/green background.
    return tuple(80 + int(channel) % 176 for channel in digest) + (255,)


def make_transform(df: pd.DataFrame) -> dict[str, float]:
    return {
        "csv_x_min": float(df["x_m"].min()),
        "csv_x_max": float(df["x_m"].max()),
        "csv_y_min": float(df["y_m"].min()),
        "csv_y_max": float(df["y_m"].max()),
        "csv_x_center": 0.5 * (float(df["x_m"].min()) + float(df["x_m"].max())),
    }


def transformed_bounds(
    df: pd.DataFrame,
    orientation: str,
    transform: dict[str, float],
) -> tuple[float, float, float, float]:
    road_length = transform["csv_y_max"] - transform["csv_y_min"] + 2.0 * ROAD_MARGIN_M
    lateral_half_span = 0.5 * (transform["csv_x_max"] - transform["csv_x_min"])
    if orientation == "horizontal":
        return (0.0, road_length, -lateral_half_span, lateral_half_span)
    return (
        -lateral_half_span,
        lateral_half_span,
        0.0,
        road_length,
    )


def transformed_data_bounds(
    df: pd.DataFrame,
    orientation: str,
    transform: dict[str, float],
) -> tuple[float, float, float, float]:
    points = [transform_xy(row, orientation, transform) for row in df.itertuples(index=False)]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), max(xs), min(ys), max(ys)


def write_plain_network_inputs(
    out_dir: Path,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    orientation: str,
    lane_width: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    x0 = x_min
    x1 = x_max
    y0 = y_min
    y1 = y_max
    x_center = 0.5 * (x_min + x_max)
    y_center = 0.0

    if orientation == "horizontal":
        start_x, start_y = x0, 0.0
        end_x, end_y = x1, 0.0
    else:
        start_x, start_y = 0.0, y0
        end_x, end_y = 0.0, y1

    (out_dir / "nodes.nod.xml").write_text(
        f"""<nodes>
    <node id="start" x="{start_x:.3f}" y="{start_y:.3f}" type="priority"/>
    <node id="end" x="{end_x:.3f}" y="{end_y:.3f}" type="priority"/>
</nodes>
"""
    )
    (out_dir / "edges.edg.xml").write_text(
        f"""<edges>
    <edge id="main" from="start" to="end" priority="1" numLanes="4" speed="40.0" width="{lane_width:.3f}" spreadType="center"/>
</edges>
"""
    )
    (out_dir / "routes.rou.xml").write_text(
        """<routes>
    <vType id="car" accel="2.6" decel="4.5" sigma="0.0" length="5.0" minGap="2.5" maxSpeed="40.0"/>
    <route id="route0" edges="main"/>
</routes>
"""
    )


def write_view_settings(
    out_dir: Path,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    zoom: float,
) -> Path:
    view_path = out_dir / "view.xml"
    center_x = 0.5 * (x_min + x_max)
    center_y = 0.5 * (y_min + y_max)
#     view_path.write_text(
#         f"""<viewsettings>
#     <scheme name="real world"/>
#     <viewport centerX="{center_x:.3f}" centerY="{center_y:.3f}" zoom="{zoom:.3f}"/>
# </viewsettings>
# """
#     )
    return view_path


def build_sumo_net(out_dir: Path) -> Path:
    netconvert = shutil.which("netconvert")
    if netconvert is None:
        raise SystemExit(
            "netconvert was not found. Install SUMO and ensure netconvert is on PATH, "
            "or run this script on a machine with SUMO installed."
        )

    net_path = out_dir / "ngsim_replay.net.xml"
    subprocess.run(
        [
            netconvert,
            "--node-files",
            str(out_dir / "nodes.nod.xml"),
            "--edge-files",
            str(out_dir / "edges.edg.xml"),
            "--output-file",
            str(net_path),
        ],
        check=True,
    )
    return net_path


def write_config(
    out_dir: Path,
    net_path: Path,
    view_path: Path,
    begin: float,
    end: float,
    step_length: float,
) -> Path:
    cfg_path = out_dir / "ngsim_replay.sumocfg"
    cfg_path.write_text(
        f"""<configuration>
    <input>
        <net-file value="{net_path.name}"/>
        <route-files value="routes.rou.xml"/>
    </input>
    <time>
        <begin value="{begin:.3f}"/>
        <end value="{end:.3f}"/>
        <step-length value="{step_length:.3f}"/>
    </time>
    <gui_only>
        <start value="true"/>
        <gui-settings-file value="{view_path.name}"/>
    </gui_only>
</configuration>
"""
    )
    return cfg_path


def prepare_files(args: argparse.Namespace, df: pd.DataFrame) -> Path:
    transform = make_transform(df)
    x_min, x_max, y_min, y_max = transformed_bounds(df, args.orientation, transform)
    view_df = df[(df["time_s"] >= args.begin) & (df["time_s"] <= args.end)]
    if view_df.empty:
        view_df = df
    view_x_min, view_x_max, view_y_min, view_y_max = transformed_data_bounds(
        view_df,
        args.orientation,
        transform,
    )
    write_plain_network_inputs(
        args.out_dir,
        x_min,
        x_max,
        y_min,
        y_max,
        args.orientation,
        args.lane_width,
    )
    view_path = write_view_settings(
        args.out_dir,
        view_x_min,
        view_x_max,
        view_y_min,
        view_y_max,
        args.view_zoom,
    )
    net_path = build_sumo_net(args.out_dir)
    cfg_path = write_config(args.out_dir, net_path, view_path, args.begin, args.end, args.step_length)
    return cfg_path


def replay(
    args: argparse.Namespace,
    cfg_path: Path,
    df: pd.DataFrame,
    transform: dict[str, float],
) -> None:
    try:
        import traci  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Python module 'traci' was not found. Install SUMO tools for this Python environment. "
            "Example: export PYTHONPATH=$SUMO_HOME/tools:$PYTHONPATH"
        ) from exc

    sumo_binary = shutil.which(args.sumo_binary)
    if sumo_binary is None:
        raise SystemExit(f"{args.sumo_binary} was not found on PATH.")

    df = df[(df["time_s"] >= args.begin) & (df["time_s"] <= args.end)].copy()
    df["vehicle_id"] = df["vehicle_id"].astype(str)
    grouped = {round(float(t), 6): group for t, group in df.groupby("time_s", sort=True)}

    traci.start(
        [
            sumo_binary,
            "-c",
            str(cfg_path),
            "--step-length",
            str(args.step_length),
            "--delay",
            str(args.speed),
            "--quit-on-end",
        ]
    )

    active: set[str] = set()
    try:
        current = args.begin
        eps = 1e-9
        while current <= args.end + eps:
            key = round(current, 6)
            frame = grouped.get(key)
            present = set(frame["vehicle_id"]) if frame is not None else set()

            for veh_id in sorted(active - present):
                if veh_id in traci.vehicle.getIDList():
                    traci.vehicle.remove(veh_id)
            active -= active - present

            if frame is not None:
                for row in frame.itertuples(index=False):
                    veh_id = str(row.vehicle_id)
                    if veh_id not in active:
                        traci.vehicle.add(veh_id, "route0", typeID="car", depart=str(current))
                        traci.vehicle.setColor(veh_id, vehicle_color(veh_id))
                        active.add(veh_id)
                    # keepRoute=2 permits placement by absolute coordinates for replay/debug.
                    x, y = transform_xy(row, args.orientation, transform)
                    traci.vehicle.moveToXY(
                        veh_id,
                        edgeID="main",
                        lane=-1,
                        x=x,
                        y=y,
                        angle=-1001.0,
                        keepRoute=2,
                    )

            traci.simulationStep(current)
            current = round(current + args.step_length, 6)
    finally:
        traci.close()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    require_columns(df, args.csv)
    transform = make_transform(df)
    cfg_path = prepare_files(args, df)
    print(f"Generated SUMO replay files in: {args.out_dir}")
    print(f"SUMO config: {cfg_path}")
    if args.prepare_only:
        return
    replay(args, cfg_path, df, transform)


if __name__ == "__main__":
    main()
