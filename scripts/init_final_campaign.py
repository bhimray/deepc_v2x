#!/usr/bin/env python3
"""Initialize a modular final-result campaign folder."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_CAMPAIGN_DIR = Path("data/output/final/ieee_250veh_deepc_campaign_01")
DEFAULT_MOBILITY_CSV = Path("data/processed/ngsim_us101_mainline_active20_250_densest_600s.csv")
METHODS = ["prbs_open_loop", "fixed_baseline", "threshold_dcc", "deepc_matlab"]
RUNS = [101, 102, 103, 104, 105]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the final IEEE campaign skeleton.")
    parser.add_argument("--campaign-dir", type=Path, default=DEFAULT_CAMPAIGN_DIR)
    parser.add_argument(
        "--vehicle-count",
        type=int,
        default=250,
        help="Nominal active-vehicle cap represented by the processed active-window file.",
    )
    parser.add_argument(
        "--max-vehicles-arg",
        type=int,
        default=0,
        help="ns-3 maxVehicles value; 0 uses every trajectory track in the processed active-window file.",
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--sim-time", type=float, default=300.0)
    parser.add_argument("--warmup", type=float, default=10.0)
    parser.add_argument("--cooldown", type=float, default=10.0)
    parser.add_argument("--mobility-csv", type=Path, default=DEFAULT_MOBILITY_CSV)
    parser.add_argument("--overwrite", action="store_true", help="Rewrite existing tracking files.")
    return parser.parse_args()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_text(path: Path, content: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def write_json(path: Path, payload: dict, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def run_dir(campaign_dir: Path, method: str, run: int) -> Path:
    return campaign_dir / "02_runs" / method / f"run_{run}"


def run_manifest(args: argparse.Namespace, method: str, run: int) -> dict:
    controller_type = {
        "prbs_open_loop": "none",
        "fixed_baseline": "open_loop_fixed_inputs",
        "threshold_dcc": "threshold_file_bridge",
        "deepc_matlab": "matlab_yalmip_file_bridge",
    }[method]
    return {
        "method": method,
        "run": run,
        "seed": args.seed,
        "vehicle_count": args.vehicle_count,
        "mobility_csv": str(args.mobility_csv),
        "mobility_selection": {
            "max_vehicles_arg": args.max_vehicles_arg,
            "max_vehicles_arg_meaning": "0 means use all trajectory tracks in mobility_csv",
            "active_vehicle_bounds_per_sample": {
                "min": 20,
                "max": 250,
                "source": "data/processed/run_config_active20_250_densest_600s.json",
            },
        },
        "sim_time_s": args.sim_time,
        "warmup_s": args.warmup,
        "cooldown_s": args.cooldown,
        "controller_type": controller_type,
        "status": "planned",
        "command_file": f"commands/run_{method}.sh",
        "outputs": {
            "kpi_csv": "kpi_timeseries.csv",
            "cbr_csv": "cbr_timeseries.csv",
            "metadata_json": "kpi_timeseries_metadata.json",
            "tx_packet_log_csv": "kpi_timeseries_tx_packet_log.csv",
            "rx_packet_log_csv": "kpi_timeseries_rx_packet_log.csv",
        },
        "validation": {
            "status": "not_run",
            "report": "validation/data_validation.txt",
        },
    }


def campaign_manifest(args: argparse.Namespace) -> dict:
    return {
        "campaign_name": args.campaign_dir.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "ns3_commit": git_commit(),
        "scenario": "NR-V2X CAM DeePC control over NGSIM US-101 mainline active-window mobility",
        "campaign_dir": str(args.campaign_dir),
        "vehicle_count": args.vehicle_count,
        "mobility_csv": str(args.mobility_csv),
        "mobility_run_config": "data/processed/run_config_active20_250_densest_600s.json",
        "mobility_selection": {
            "max_vehicles_arg": args.max_vehicles_arg,
            "max_vehicles_arg_meaning": "0 means use all trajectory tracks in mobility_csv",
            "active_vehicle_bounds_per_sample": {
                "min": 20,
                "max": 250,
                "source": "data/processed/run_config_active20_250_densest_600s.json",
            },
        },
        "seed": args.seed,
        "runs": RUNS,
        "methods": METHODS,
        "simulation": {
            "sim_time_s": args.sim_time,
            "warmup_s": args.warmup,
            "cooldown_s": args.cooldown,
            "sample_time_s": 0.1,
            "control_interval_s": 0.5,
            "awareness_range_m": 300.0,
        },
        "metric_definitions": {
            "mean_prr_awareness": "Mean KPI PRR over the evaluation window.",
            "mean_pir_s": "Mean packet inter-reception time over the evaluation window.",
            "mean_cbr": "Mean PHY channel busy ratio over the evaluation window.",
            "cbr_gt_0p6_rate": "Fraction of KPI samples with CBR greater than 0.6.",
            "p95_delay_s": "95th percentile packet delay from RX packet logs when available.",
        },
        "cleanup": {
            "archive_dir": "data/output/final/archive",
            "archive_before_delete": True,
            "deletion_unit": str(args.campaign_dir),
        },
    }


def readme(args: argparse.Namespace) -> str:
    return f"""# {args.campaign_dir.name}

Self-contained final-result campaign for the IEEE NR-V2X DeePC paper attempt.

## Status

Planned. Fill `02_runs/<method>/run_<id>/` with final ns-3 outputs only.

## Mobility Dataset

All campaign commands use
`{args.mobility_csv}`.
The processed file is an active-window dataset: `--maxVehicles=0` is intentional
and means ns-3 should use every trajectory track in the file, while the
simultaneous active vehicles are bounded by the preprocessing run config.

## Folder Contract

- `00_preflight/`: build checks, smoke tests, and environment notes.
- `01_training/`: PRBS training data, Hankel dataset, scaler, and MATLAB export.
- `02_runs/`: final repeated runs only.
- `03_analysis/`: aggregate metrics, confidence intervals, tables, and paper plots.
- `04_tests/`: validation reports and final acceptance checklist.
- `commands/`: exact commands used to reproduce the campaign.
- `logs/`: campaign-level terminal logs.
- `tmp/`: disposable scratch space.

## Cleanup

Archive this whole folder into `data/output/final/archive/` before deletion.
Nothing outside this campaign folder should be required to interpret final results,
except source code, raw NGSIM input data, and documented environment dependencies.
"""


def acceptance_checklist() -> str:
    checks = [
        "[ ] All 20 final runs completed: 4 methods x 5 runs.",
        "[ ] Every run has KPI, CBR, metadata, TX log, RX log, and run manifest.",
        "[ ] `scripts/data_validation.py` passed for every run.",
        "[ ] Controller-based runs have bridge logs and success rate >= 95%.",
        "[ ] `03_analysis/run_metrics.csv` generated.",
        "[ ] `03_analysis/aggregate_metrics.csv` generated with 95% CIs.",
        "[ ] Final paper plots generated under `03_analysis/plots/`.",
        "[ ] Campaign archived before any cleanup/delete operation.",
    ]
    return "# Acceptance Checklist\n\n" + "\n".join(checks) + "\n"


def command_templates(campaign_dir: Path, mobility_csv_path: Path, max_vehicles: int) -> dict[str, str]:
    cd = str(campaign_dir)
    mobility_csv = str(mobility_csv_path)
    max_vehicles_arg = str(max_vehicles)
    return {
        "README.md": """# Command Templates

Edit these templates only when the experiment definition changes. Capture exact
commands used for each run in the corresponding `run_manifest.json`.
""",
        "00_preflight.sh": f"""#!/usr/bin/env bash
set -euo pipefail

export MPLCONFIGDIR="${{MPLCONFIGDIR:-/tmp/matplotlib}}"
export CCACHE_DISABLE="${{CCACHE_DISABLE:-1}}"
MOBILITY_CSV="${{MOBILITY_CSV:-{mobility_csv}}}"

./ns3 build nr_v2x_ngsim_deepc_data_set_generation
./ns3 build nr_v2x_ngsim_deepc_closed_loop_deepc

./ns3 run "nr_v2x_ngsim_deepc_closed_loop_deepc --validateOnly=true --mobilityCsv=$MOBILITY_CSV --simTime=300 --maxVehicles={max_vehicles_arg} --kpiCsv={cd}/tmp/preflight_kpi.csv --cbrCsv={cd}/tmp/preflight_cbr.csv --deepcBridgeDir={cd}/tmp/preflight_bridge"
""",
        "00_fixed_baseline_open_loop_smoke.sh": f"""#!/usr/bin/env bash
set -euo pipefail

PYTHON="${{PYTHON:-./v2x_env/bin/python}}"
export MPLCONFIGDIR="${{MPLCONFIGDIR:-/tmp/matplotlib}}"
export CCACHE_DISABLE="${{CCACHE_DISABLE:-1}}"
MOBILITY_CSV="${{MOBILITY_CSV:-{mobility_csv}}}"

SIM_TIME="${{SIM_TIME:-30}}"
MAX_VEHICLES="${{MAX_VEHICLES:-{max_vehicles_arg}}}"
RUN="${{RUN:-901}}"
TX_POWER_DBM="${{TX_POWER_DBM:-20}}"
BEACON_INTERVAL_S="${{BEACON_INTERVAL_S:-0.1}}"
WARMUP="${{WARMUP:-2}}"
COOLDOWN="${{COOLDOWN:-2}}"
OUT="{cd}/tmp/fixed_baseline_open_loop_smoke"
mkdir -p "$OUT/validation" "$OUT/plots"

./ns3 run "nr_v2x_ngsim_deepc_data_set_generation --mobilityCsv=$MOBILITY_CSV --simTime=$SIM_TIME --warmup=$WARMUP --cooldown=$COOLDOWN --maxVehicles=$MAX_VEHICLES --seed=12345 --run=$RUN --useInputSchedule=false --txPower=$TX_POWER_DBM --fixedBeaconInterval=$BEACON_INTERVAL_S --kpiCsv=$OUT/kpi_timeseries.csv" | tee "$OUT/ns3.log"
"$PYTHON" scripts/data_validation.py "$OUT/kpi_timeseries.csv" --out-dir "$OUT/validation" > "$OUT/validation/data_validation.txt"
"$PYTHON" scripts/plot_deepc_bridge_run.py "$OUT" --out-dir "$OUT/plots"
""",
        "01_training.sh": f"""#!/usr/bin/env bash
set -euo pipefail

PYTHON="${{PYTHON:-./v2x_env/bin/python}}"
export MPLCONFIGDIR="${{MPLCONFIGDIR:-/tmp/matplotlib}}"
export CCACHE_DISABLE="${{CCACHE_DISABLE:-1}}"
MOBILITY_CSV="${{MOBILITY_CSV:-{mobility_csv}}}"
TRAIN_DIR="{cd}/01_training/prbs_training_250veh_run001"
mkdir -p "$TRAIN_DIR"

./ns3 run "nr_v2x_ngsim_deepc_data_set_generation --mobilityCsv=$MOBILITY_CSV --simTime=600 --maxVehicles={max_vehicles_arg} --seed=12345 --run=1 --kpiCsv=$TRAIN_DIR/kpi_timeseries.csv"
"$PYTHON" scripts/data_validation.py "$TRAIN_DIR/kpi_timeseries.csv" --out-dir "$TRAIN_DIR/validation"
"$PYTHON" scripts/build_hankel_dataset.py --kpi-csv "$TRAIN_DIR/kpi_timeseries.csv" --out-dir "{cd}/01_training/deepc_dataset_dt0p1"
"$PYTHON" scripts/export_deepc_matlab_data.py --dataset-dir "{cd}/01_training/deepc_dataset_dt0p1" --out "{cd}/01_training/deepc_dataset_dt0p1/matlab_deepc_data.mat"
""",
        "run_prbs_open_loop.sh": f"""#!/usr/bin/env bash
set -euo pipefail

PYTHON="${{PYTHON:-./v2x_env/bin/python}}"
export MPLCONFIGDIR="${{MPLCONFIGDIR:-/tmp/matplotlib}}"
export CCACHE_DISABLE="${{CCACHE_DISABLE:-1}}"
MOBILITY_CSV="${{MOBILITY_CSV:-{mobility_csv}}}"
RUN="${{1:?Usage: $0 RUN_NUMBER}}"
OUT="{cd}/02_runs/prbs_open_loop/run_$RUN"
mkdir -p "$OUT/logs" "$OUT/validation" "$OUT/plots"

./ns3 run "nr_v2x_ngsim_deepc_data_set_generation --mobilityCsv=$MOBILITY_CSV --simTime=300 --maxVehicles={max_vehicles_arg} --seed=12345 --run=$RUN --kpiCsv=$OUT/kpi_timeseries.csv" | tee "$OUT/logs/ns3.log"
"$PYTHON" scripts/data_validation.py "$OUT/kpi_timeseries.csv" --out-dir "$OUT/validation" > "$OUT/validation/data_validation.txt"
""",
        "run_fixed_baseline.sh": f"""#!/usr/bin/env bash
set -euo pipefail

PYTHON="${{PYTHON:-./v2x_env/bin/python}}"
export MPLCONFIGDIR="${{MPLCONFIGDIR:-/tmp/matplotlib}}"
export CCACHE_DISABLE="${{CCACHE_DISABLE:-1}}"
MOBILITY_CSV="${{MOBILITY_CSV:-{mobility_csv}}}"
RUN="${{1:?Usage: $0 RUN_NUMBER}}"
OUT="{cd}/02_runs/fixed_baseline/run_$RUN"
mkdir -p "$OUT/logs" "$OUT/validation" "$OUT/plots"

./ns3 run "nr_v2x_ngsim_deepc_data_set_generation --mobilityCsv=$MOBILITY_CSV --simTime=300 --maxVehicles={max_vehicles_arg} --seed=12345 --run=$RUN --useInputSchedule=false --txPower=20 --fixedBeaconInterval=0.1 --kpiCsv=$OUT/kpi_timeseries.csv" | tee "$OUT/logs/ns3.log"
"$PYTHON" scripts/data_validation.py "$OUT/kpi_timeseries.csv" --out-dir "$OUT/validation" > "$OUT/validation/data_validation.txt"
"$PYTHON" scripts/plot_deepc_bridge_run.py "$OUT"
""",
        "run_threshold_dcc.sh": f"""#!/usr/bin/env bash
set -euo pipefail

PYTHON="${{PYTHON:-./v2x_env/bin/python}}"
export MPLCONFIGDIR="${{MPLCONFIGDIR:-/tmp/matplotlib}}"
export CCACHE_DISABLE="${{CCACHE_DISABLE:-1}}"
MOBILITY_CSV="${{MOBILITY_CSV:-{mobility_csv}}}"
RUN="${{1:?Usage: $0 RUN_NUMBER}}"
OUT="{cd}/02_runs/threshold_dcc/run_$RUN"
mkdir -p "$OUT/logs" "$OUT/validation" "$OUT/plots" "$OUT/bridge"

# Replace this controller command with the threshold DCC bridge controller once implemented.
"$PYTHON" scripts/dummy_deepc_file_controller.py --bridge-dir "$OUT/bridge" --mode hold > "$OUT/logs/controller.log" 2>&1 &
CTRL_PID=$!
trap 'kill "$CTRL_PID" 2>/dev/null || true' EXIT
./ns3 run "nr_v2x_ngsim_deepc_closed_loop_deepc --mobilityCsv=$MOBILITY_CSV --simTime=300 --maxVehicles={max_vehicles_arg} --seed=12345 --run=$RUN --kpiCsv=$OUT/kpi_timeseries.csv --cbrCsv=$OUT/cbr_timeseries.csv --deepcBridgeDir=$OUT/bridge --deepcControlInterval=0.5 --deepcPastHorizon=20 --deepcFutureHorizon=10" | tee "$OUT/logs/ns3.log"
"$PYTHON" scripts/data_validation.py "$OUT/kpi_timeseries.csv" --out-dir "$OUT/validation" > "$OUT/validation/data_validation.txt"
"$PYTHON" scripts/plot_deepc_bridge_run.py "$OUT"
""",
        "run_deepc_matlab.sh": f"""#!/usr/bin/env bash
set -euo pipefail

PYTHON="${{PYTHON:-./v2x_env/bin/python}}"
export MPLCONFIGDIR="${{MPLCONFIGDIR:-/tmp/matplotlib}}"
export CCACHE_DISABLE="${{CCACHE_DISABLE:-1}}"
MOBILITY_CSV="${{MOBILITY_CSV:-{mobility_csv}}}"
RUN="${{1:?Usage: $0 RUN_NUMBER}}"
OUT="{cd}/02_runs/deepc_matlab/run_$RUN"
mkdir -p "$OUT/logs" "$OUT/validation" "$OUT/plots" "$OUT/bridge"

echo "Start MATLAB controller on Windows against this WSL bridge path before running ns-3:"
wslpath -w "$OUT/bridge"

./ns3 run "nr_v2x_ngsim_deepc_closed_loop_deepc --mobilityCsv=$MOBILITY_CSV --simTime=300 --maxVehicles={max_vehicles_arg} --seed=12345 --run=$RUN --kpiCsv=$OUT/kpi_timeseries.csv --cbrCsv=$OUT/cbr_timeseries.csv --deepcBridgeDir=$OUT/bridge --deepcControlInterval=0.5 --deepcPastHorizon=20 --deepcFutureHorizon=10 --deepcResponseTimeout=900" | tee "$OUT/logs/ns3.log"
"$PYTHON" scripts/data_validation.py "$OUT/kpi_timeseries.csv" --out-dir "$OUT/validation" > "$OUT/validation/data_validation.txt"
"$PYTHON" scripts/plot_deepc_bridge_run.py "$OUT"
""",
        "03_analyze.sh": f"""#!/usr/bin/env bash
set -euo pipefail

PYTHON="${{PYTHON:-./v2x_env/bin/python}}"
export MPLCONFIGDIR="${{MPLCONFIGDIR:-/tmp/matplotlib}}"

"$PYTHON" scripts/aggregate_final_campaign.py --campaign-dir "{cd}"
"$PYTHON" scripts/plot_cbr_validation.py --campaign-dir "{cd}"
""",
        "04_archive_before_delete.sh": f"""#!/usr/bin/env bash
set -euo pipefail

ARCHIVE_DIR="data/output/final/archive"
mkdir -p "$ARCHIVE_DIR"
tar -czf "$ARCHIVE_DIR/{campaign_dir.name}_$(date +%Y%m%d_%H%M%S).tar.gz" -C "{campaign_dir.parent}" "{campaign_dir.name}"
""",
    }


def main() -> None:
    args = parse_args()
    campaign_dir = args.campaign_dir
    for subdir in [
        "commands",
        "00_preflight",
        "01_training",
        "02_runs",
        "03_analysis/plots",
        "03_analysis/tables",
        "04_tests",
        "logs",
        "tmp",
    ]:
        (campaign_dir / subdir).mkdir(parents=True, exist_ok=True)

    write_text(campaign_dir / "README.md", readme(args), args.overwrite)
    write_json(campaign_dir / "campaign_manifest.json", campaign_manifest(args), args.overwrite)
    write_text(campaign_dir / "04_tests" / "acceptance_checklist.md", acceptance_checklist(), args.overwrite)

    for name, content in command_templates(
        campaign_dir, args.mobility_csv, args.max_vehicles_arg
    ).items():
        write_text(campaign_dir / "commands" / name, content, args.overwrite)

    for method in METHODS:
        for run in RUNS:
            rd = run_dir(campaign_dir, method, run)
            for subdir in ["validation", "plots", "logs"]:
                (rd / subdir).mkdir(parents=True, exist_ok=True)
            if method in {"threshold_dcc", "deepc_matlab"}:
                (rd / "bridge").mkdir(parents=True, exist_ok=True)
            write_json(rd / "run_manifest.json", run_manifest(args, method, run), args.overwrite)

    print(f"Initialized campaign: {campaign_dir}")


if __name__ == "__main__":
    main()
