import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_DENSITIES = [50, 100, 150, 200, 250, 300, 400]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect NR-V2X KPI datasets for a sweep of vehicle densities."
    )
    parser.add_argument(
        "--densities",
        type=int,
        nargs="+",
        default=DEFAULT_DENSITIES,
        help="Vehicle counts to simulate.",
    )
    parser.add_argument(
        "--mobility-csv",
        type=Path,
        default=Path("data/processed/ngsim_us101_all_lanes_0p1s_400veh.csv"),
        help="Processed mobility CSV to create or reuse.",
    )
    parser.add_argument(
        "--preprocess-config",
        type=Path,
        default=Path("data/processed/run_config_all_lanes_400veh.json"),
        help="Preprocessing config JSON path.",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Reuse --mobility-csv instead of regenerating it.",
    )
    parser.add_argument(
        "--preprocess-lanes",
        default="1,2,3,4,5,6,7,8",
        help="Comma-separated lane IDs used when preprocessing.",
    )
    parser.add_argument(
        "--min-track-duration",
        type=float,
        default=1.0,
        help="Minimum track duration used when preprocessing [s].",
    )
    parser.add_argument(
        "--preprocess-time-start",
        type=float,
        default=0.0,
        help="Start of normalized time window used when preprocessing [s].",
    )
    parser.add_argument(
        "--preprocess-time-end",
        type=float,
        default=400.0,
        help="End of normalized time window used when preprocessing [s].",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/output/density_sweep"),
        help="Directory for KPI/CBR CSVs.",
    )
    parser.add_argument("--sim-time", type=float, default=100.0, help="Simulation time [s].")
    parser.add_argument("--warmup", type=float, default=10.0, help="Warm-up duration [s].")
    parser.add_argument("--cooldown", type=float, default=10.0, help="Cool-down duration [s].")
    parser.add_argument("--seed", type=int, default=12345, help="RNG seed.")
    parser.add_argument("--run", type=int, default=1, help="Base RNG run number.")
    parser.add_argument(
        "--extra-ns3-arg",
        action="append",
        default=[],
        help="Extra simulator argument, for example --extra-ns3-arg=--mcs=6.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a density if its KPI CSV, CBR CSV, and metadata JSON already exist.",
    )
    return parser.parse_args()


def run_command(command: list[str], dry_run: bool) -> None:
    print("+ " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    densities = sorted(set(args.densities))
    max_density = max(densities)

    if not args.skip_preprocess:
        run_command(
            [
                sys.executable,
                "scripts/preprocess_ngsim.py",
                f"--max-vehicles={max_density}",
                f"--out-path={args.mobility_csv}",
                f"--cfg-path={args.preprocess_config}",
                f"--lanes={args.preprocess_lanes}",
                f"--min-track-duration={args.min_track_duration}",
                f"--time-start={args.preprocess_time_start}",
                f"--time-end={args.preprocess_time_end}",
            ],
            args.dry_run,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for offset, density in enumerate(densities):
        suffix = f"{density:03d}"
        kpi_csv = args.output_dir / f"kpi_density_{suffix}.csv"
        cbr_csv = args.output_dir / f"cbr_density_{suffix}.csv"
        metadata_json = args.output_dir / f"kpi_density_{suffix}_metadata.json"
        if args.skip_existing and kpi_csv.exists() and cbr_csv.exists() and metadata_json.exists():
            print(f"Skipping {density} vehicles; outputs already exist.", flush=True)
            continue

        sim_args = [
            "scratch/nr_v2x_ngsim_deepc",
            f"--mobilityCsv={args.mobility_csv}",
            f"--maxVehicles={density}",
            f"--kpiCsv={kpi_csv}",
            f"--cbrCsv={cbr_csv}",
            f"--simTime={args.sim_time}",
            f"--warmup={args.warmup}",
            f"--cooldown={args.cooldown}",
            f"--seed={args.seed}",
            f"--run={args.run + offset}",
            *args.extra_ns3_arg,
        ]
        run_command(["./ns3", "run", " ".join(sim_args)], args.dry_run)

    plot_inputs = [
        str(args.output_dir / f"kpi_density_{density:03d}.csv") for density in densities
    ]
    run_command(
        [
            sys.executable,
            "scripts/plot_cbr_validation.py",
            *plot_inputs,
            f"--plots-dir={args.output_dir / 'plots'}",
        ],
        args.dry_run,
    )


if __name__ == "__main__":
    main()
