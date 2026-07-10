# Minimal Matched Campaign Commands

These commands produce the smallest defendable result set for the conference-paper draft.

Run from the repository root.

## Environment

```bash
export MPLCONFIGDIR=/tmp/matplotlib
export CCACHE_DISABLE=1
export PYTHON=./v2x_env/bin/python
```

## Fixed Baselines

The primary fair baseline is fixed 10 Hz with 20 dBm transmit power.

```bash
for run in 101 102 103 104 105; do
  data/output/final/ieee_250veh_deepc_campaign_01/commands/run_fixed_10hz.sh "$run"
done
```

The fixed 5 Hz baseline is contextual only because it changes the offered CAM load.

```bash
for run in 101 102 103 104 105; do
  data/output/final/ieee_250veh_deepc_campaign_01/commands/run_fixed_5hz.sh "$run"
done
```

## DeePC Main Method

The main DeePC method uses the count/context, 0.1 s beacon-interval dataset and keeps CAM generation fixed at 10 Hz.

```bash
export METHOD=deepc_count_tb0p1_prr_only
export DATASET_DIR=data/output/final/ieee_250veh_deepc_campaign_01/01_training/deepc_dataset_txpower_count_tb0p1_dt0p5_n10
export SIM_TIME=300
export WARMUP=10
export COOLDOWN=10
export DEEPC_CONTROLLER_ARGS='--q-prr 40 --q-pir 0 --q-cbr 0'

for run in 101 102 103 104 105; do
  data/output/final/ieee_250veh_deepc_campaign_01/commands/run_deepc_python_txpower.sh "$run"
done
```

## Regenerate Paper Tables

```bash
./v2x_env/bin/python scripts/aggregate_final_campaign.py
```

Check:

```bash
sed -n '1,120p' data/output/final/ieee_250veh_deepc_campaign_01/03_analysis/paper_readiness_report.md
```

The paper is ready for numerical claims only when `fixed_10hz`, `fixed_5hz`, and `deepc_count_tb0p1_prr_only` all show complete runs for 101-105.

## Important Caution

The baseline scripts can overwrite existing run CSVs. Archive any partial or exploratory run directory that must be preserved before rerunning the same run id.
