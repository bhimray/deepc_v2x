# DeePC / RoKDeePC V2X Research Progress Log

This file records the DeePC research workflow step by step so the work remains
recoverable, auditable, and easy to turn into paper methodology.

## Current Goal

Use the existing ns-3 NR-V2X NGSIM open-loop dataset to build and validate
offline DeePC and kernelized DeePC predictors before attempting closed-loop
control inside ns-3.

Primary control objective:

- balance reliability and congestion by predicting/improving PRR while keeping
  CBR below the congestion-control target of 0.6.

## Step 0: Existing Data Checkpoint

Status: completed before this log was created.

Important existing artifacts:

- `data/output/kpi_timeseries_10min_250veh_run01.csv`
- `data/output/kpi_timeseries_10min_250veh_run01_metadata.json`
- `data/output/kpi_timeseries_10min_250veh_run01_tx_packet_log.csv`
- `data/output/kpi_timeseries_10min_250veh_run01_rx_packet_log.csv`
- `data/output/cbr_timeseries_10min_250veh_run01.csv`

Observed dataset facts:

- simulation duration: 600 s
- sample time: 0.1 s
- KPI samples: 5999
- evaluation window from metadata: 10 s to 590 s
- vehicle count: 250
- input schedule: PRBS over Tx power and beacon interval
- DeePC-ready inputs: `tx_power_dbm`, `beacon_interval_s`
- DeePC-ready outputs: `prr_150m`, `pir_s`, `cbr`
- traffic/context variables: `active_vehicle_count_core`,
  `density_veh_per_km_core`, `mean_neighbors_150m`,
  `mean_neighbors_300m`, `sensing_exclusion_ratio`

Initial KPI summary from the existing data:

- mean CBR: about 0.326
- mean PRR at 150 m: about 0.555
- mean PIR: about 0.293 s

Interpretation:

- The data is valuable and should be reused before running another expensive
  ns-3 experiment.
- The first DeePC milestone should be open-loop prediction and validation,
  not closed-loop ns-3 control.

Not done yet:

- DeePC train/validation/test dataset construction.
- Hankel matrix export.
- Linear DeePC-style open-loop prediction.
- Kernelized/RoKDeePC-style open-loop prediction.
- Prediction plots and quantitative metrics.

## Step 1: Offline DeePC Tooling Implementation

Status: completed.

Changes made:

- Replaced `scripts/build_hankel_dataset.py` with a configurable builder.
- Added `scripts/open_loop_deepc_predict.py` for held-out open-loop prediction.
- Wrote generated DeePC artifacts under `data/output/deepc_open_loop_250veh/`.

Commands run:

```bash
./v2x_env/bin/python scripts/build_hankel_dataset.py
./v2x_env/bin/python scripts/open_loop_deepc_predict.py
```

Dataset builder result:

- clean evaluation-window rows: 5801
- train rows: 3480
- validation rows: 1160
- test rows: 1161
- train Hankel windows: 3451
- default past horizon: 20 samples = 2.0 s
- default future horizon: 10 samples = 1.0 s

Generated dataset files:

- `data/output/deepc_open_loop_250veh/deepc_clean.csv`
- `data/output/deepc_open_loop_250veh/deepc_train.csv`
- `data/output/deepc_open_loop_250veh/deepc_val.csv`
- `data/output/deepc_open_loop_250veh/deepc_test.csv`
- `data/output/deepc_open_loop_250veh/deepc_*_normalized.csv`
- `data/output/deepc_open_loop_250veh/scaler.json`
- `data/output/deepc_open_loop_250veh/dataset_summary.json`
- `data/output/deepc_open_loop_250veh/hankel_train_normalized.npz`

Open-loop prediction result on held-out test data:

| Model | Output | RMSE | MAE | R2 |
|---|---:|---:|---:|---:|
| linear_deepc | PRR | 0.044752 | 0.033825 | 0.716919 |
| linear_deepc | PIR | 0.042447 | 0.030457 | 0.635272 |
| linear_deepc | CBR | 0.013252 | 0.009960 | 0.792872 |
| kernel_rokdeepc | PRR | 0.046903 | 0.035616 | 0.689058 |
| kernel_rokdeepc | PIR | 0.046259 | 0.032920 | 0.566820 |
| kernel_rokdeepc | CBR | 0.014893 | 0.011430 | 0.738389 |

Generated prediction files:

- `data/output/deepc_open_loop_250veh/open_loop_predictions/metrics.json`
- `data/output/deepc_open_loop_250veh/open_loop_predictions/first_step_predictions.csv`
- `data/output/deepc_open_loop_250veh/open_loop_predictions/plots/`

Interpretation:

- The existing 18-hour PRBS dataset is sufficient for a first open-loop DeePC
  result.
- Linear DeePC-style prediction currently performs slightly better than the
  tuned RBF kernel predictor on this held-out split.
- CBR prediction is strong enough to support the next stage of control-oriented
  analysis.
- PRR prediction is promising but should be stress-tested across different
  horizons, density regimes, and train/test splits before making a paper claim.

Not done yet:

- Closed-loop DeePC or RoKDeePC control in ns-3.
- Hyperparameter sweep over DeePC horizons.
- Separate metrics by density regime.
- Comparison to threshold-DCC and fixed-input baselines using the same outputs.

## Step 2: acados Installation for C/C++ Closed-Loop Control

Status: completed locally.

Reason:

- The closed-loop DeePC implementation should move toward acados/C++ rather than
  MATLAB/YALMIP, because the long-term target is fast controller execution near
  the ns-3 simulation loop.

Installed local paths:

- acados source: `external/acados`
- acados install prefix: `external/acados-install`
- acados shared library: `external/acados-install/lib/libacados.so`
- acados Python interface: editable install from
  `external/acados/interfaces/acados_template`
- Tera renderer: `external/acados/bin/t_renderer`

Commands run:

```bash
mkdir -p external
git clone https://github.com/acados/acados.git external/acados
git submodule update --init --recursive
cmake -S external/acados -B external/acados/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/home/bim/ns-3-dev/external/acados-install \
  -DACADOS_WITH_QPOASES=ON \
  -DACADOS_WITH_OSQP=ON
cmake --build external/acados/build --target install -j 4
./v2x_env/bin/python -m pip install casadi
./v2x_env/bin/python -m pip install -e external/acados/interfaces/acados_template
env MPLCONFIGDIR=/tmp/matplotlib \
  ACADOS_SOURCE_DIR=/home/bim/ns-3-dev/external/acados \
  ./v2x_env/bin/python -c "from acados_template import get_tera; print(get_tera(force_download=True))"
```

Verification:

```bash
external/acados/bin/t_renderer --help
env MPLCONFIGDIR=/tmp/matplotlib \
  ACADOS_SOURCE_DIR=/home/bim/ns-3-dev/external/acados \
  LD_LIBRARY_PATH=/home/bim/ns-3-dev/external/acados-install/lib \
  ./v2x_env/bin/python - <<'PY'
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import casadi as ca
print(ca.__version__)
print("imports ok")
PY
```

Verification result:

- `t_renderer` is executable.
- `acados_template` imports successfully.
- CasADi version installed: 3.7.2.
- acados libraries were installed under `external/acados-install/lib`.

Important environment variables for future acados scripts:

```bash
export ACADOS_SOURCE_DIR=/home/bim/ns-3-dev/external/acados
export LD_LIBRARY_PATH=/home/bim/ns-3-dev/external/acados-install/lib:$LD_LIBRARY_PATH
export MPLCONFIGDIR=/tmp/matplotlib
```

Not done yet:

- generate a DeePC-specific acados solver.
- build a C++ wrapper around the generated solver.
- connect the generated controller to ns-3.

## Step 3: Closed-Loop Linearized DeePC with acados

Status: completed as an offline surrogate closed-loop rollout.

Implemented file:

- `scripts/closed_loop_linear_deepc_acados.py`

Method:

- Fit a one-step linearized DeePC predictor from normalized train+validation
  data.
- Use the DeePC history buffer as the acados discrete-time state:
  past inputs, past outputs, and past traffic/context variables.
- Use physical controls as acados inputs:
  `tx_power_dbm` and `beacon_interval_s`.
- Use measured held-out future context as stage parameters.
- Optimize a receding-horizon objective for PRR/PIR/CBR tracking, input level,
  and input movement.
- Roll out the controller on the held-out test segment without running ns-3.

Command run:

```bash
env MPLCONFIGDIR=/tmp/matplotlib \
  ACADOS_SOURCE_DIR=/home/bim/ns-3-dev/external/acados \
  LD_LIBRARY_PATH=/home/bim/ns-3-dev/external/acados-install/lib:$LD_LIBRARY_PATH \
  ./v2x_env/bin/python scripts/closed_loop_linear_deepc_acados.py
```

Generated files:

- `data/output/deepc_open_loop_250veh/closed_loop_linear_deepc_acados/closed_loop_rollout.csv`
- `data/output/deepc_open_loop_250veh/closed_loop_linear_deepc_acados/closed_loop_control_schedule.csv`
- `data/output/deepc_open_loop_250veh/closed_loop_linear_deepc_acados/closed_loop_metrics.json`
- `data/output/deepc_open_loop_250veh/closed_loop_linear_deepc_acados/one_step_linear_deepc_beta.npy`
- `data/output/deepc_open_loop_250veh/closed_loop_linear_deepc_acados/acados_generated/`
- `data/output/deepc_open_loop_250veh/closed_loop_linear_deepc_acados/plots/closed_loop_timeseries.png`

400-step surrogate rollout metrics:

| Metric | Value |
|---|---:|
| predicted closed-loop PRR mean | 0.790244 |
| measured open-loop PRR mean on same segment | 0.628639 |
| predicted closed-loop PIR mean | 0.340756 |
| measured open-loop PIR mean on same segment | 0.252873 |
| predicted closed-loop CBR mean | 0.253013 |
| measured open-loop CBR mean on same segment | 0.335500 |
| predicted closed-loop CBR > 0.6 rate | 0.000000 |
| measured open-loop CBR > 0.6 rate | 0.000000 |
| mean selected Tx power | 17.165739 dBm |
| mean selected beacon interval | 0.500000 s |

Interpretation:

- The acados controller is operational and produces a closed-loop schedule.
- On the learned surrogate, the controller improves PRR and lowers CBR compared
  with the held-out measured open-loop segment.
- The optimizer drives `beacon_interval_s` to its upper bound, so the next
  research step must check whether this is physically desirable or whether the
  cost should penalize large PIR / low awareness more strongly.
- These results are not yet ns-3 closed-loop results. They are surrogate
  closed-loop results and must be replayed in ns-3 before making final claims.

Validation:

```bash
./v2x_env/bin/python -m py_compile scripts/closed_loop_linear_deepc_acados.py
```

Follow-up runtime fix:

- Running the script as plain `python scripts/closed_loop_linear_deepc_acados.py`
  initially failed with `OSError: libqpOASES_e.so: cannot open shared object
  file`.
- Cause: `LD_LIBRARY_PATH` must be visible before Python starts; changing it
  inside an already running Python process is too late for the dynamic linker.
- Fix: the script now re-executes itself once with `LD_LIBRARY_PATH`,
  `ACADOS_SOURCE_DIR`, and `MPLCONFIGDIR` set before importing acados.

Verification command:

```bash
./v2x_env/bin/python scripts/closed_loop_linear_deepc_acados.py --no-build
```

Verification result:

- plain venv Python invocation succeeds.
- generated solver reuse works.
- full 400-step rollout artifacts were restored after the smoke test.

Not done yet:

- tune the objective to avoid trivial upper-bound `Tb` behavior if needed.
- replay `closed_loop_control_schedule.csv` in ns-3.
- compare against fixed-input, PRBS, and threshold-DCC baselines on identical
  evaluation windows.
- build the C++ wrapper around the generated acados solver.

## ns-3 File-Bridge Closed-Loop Scaffold

Goal:

- Prepare `scratch/nr_v2x_ngsim_deepc_copy.cc` for true controller-in-the-loop
  simulation without overwriting the open-loop data used for Hankel matrices.
- Implement the first bridge slice before MATLAB/YALMIP integration:
  ns-3 writes DeePC request JSON, an external controller writes response JSON,
  and ns-3 applies the returned control.

Implemented:

- Added safe default output paths in the copy scenario:
  - `data/output/kpi_timeseries_deepc_copy_run01.csv`
  - `data/output/cbr_timeseries_deepc_copy_run01.csv`
- Added protected-output guard that refuses to write to:
  - `data/output/kpi_timeseries_10min_250veh_run01.csv`
  - `data/output/cbr_timeseries_10min_250veh_run01.csv`
  - `data/output/deepc_open_loop_250veh/`
- Added `KpiSample` storage in `KpiLogger`.
- Added optional file bridge controlled by:
  - `--enableDeepcBridge`
  - `--deepcBridgeDir`
  - `--deepcControlInterval`
  - `--deepcPastHorizon`
  - `--deepcFutureHorizon`
  - `--deepcResponseTimeout`
- Added request/response bridge artifacts:
  - `bridge/requests/request_XXXXXX.json`
  - `bridge/responses/response_XXXXXX.json`
  - `bridge/applied_controls.csv`
  - `bridge/controller_solve_times.csv`
- Added `scripts/dummy_deepc_file_controller.py` for bridge smoke tests.

Important behavior:

- Bridge mode is disabled by default.
- Bridge mode currently requires `--useInputSchedule=false` to avoid PRBS
  updates fighting with closed-loop controller updates.
- Future context `d_future` is currently a repeated-current-context forecast.
  A later research version can replace this with NGSIM oracle mobility context.

Validation:

```bash
./v2x_env/bin/python -m py_compile scripts/dummy_deepc_file_controller.py
CCACHE_DISABLE=1 ./ns3 build nr_v2x_ngsim_deepc_copy
CCACHE_DISABLE=1 ./ns3 run "nr_v2x_ngsim_deepc_copy --validateOnly=true --enableDeepcBridge=true --useInputSchedule=false --maxVehicles=25 --simTime=290"
```

Bridge/PRBS conflict rejection test:

```bash
CCACHE_DISABLE=1 ./ns3 run "nr_v2x_ngsim_deepc_copy --validateOnly=true --enableDeepcBridge=true --maxVehicles=25 --simTime=290"
```

Protected-path rejection test:

```bash
CCACHE_DISABLE=1 ./ns3 run "nr_v2x_ngsim_deepc_copy --validateOnly=true --kpiCsv=data/output/kpi_timeseries_10min_250veh_run01.csv"
```

Results:

- The run aborts because bridge mode cannot run with the PRBS input schedule.
- The protected-output test aborts when `kpiCsv` points at the original
  open-loop KPI file.

Bridge smoke test:

```bash
./v2x_env/bin/python scripts/dummy_deepc_file_controller.py \
  --bridge-dir data/output/deepc_bridge_smoke_20260507_01/bridge \
  --tx-power-dbm 16 \
  --beacon-interval-s 0.12 \
  --idle-timeout-s 45

CCACHE_DISABLE=1 ./ns3 run "nr_v2x_ngsim_deepc_copy --enableDeepcBridge=true --useInputSchedule=false --deepcBridgeDir=data/output/deepc_bridge_smoke_20260507_01/bridge --deepcPastHorizon=3 --deepcFutureHorizon=2 --deepcResponseTimeout=20 --simTime=15 --cooldown=1 --maxVehicles=10 --kpiCsv=data/output/deepc_bridge_smoke_20260507_01/kpi_timeseries.csv --cbrCsv=data/output/deepc_bridge_smoke_20260507_01/cbr_timeseries.csv"
```

Smoke-test result:

- ns-3 completed normally.
- 47 bridge responses were applied.
- `applied_controls.csv` confirms controller outputs were applied:
  `tx_power_dbm=16`, `beacon_interval_s=0.12`.
- Existing Hankel artifacts in `data/output/deepc_open_loop_250veh/` were not
  overwritten.

Next step:

- Replace the dummy controller with a MATLAB/YALMIP process that reads the same
  request JSON files and writes the same response JSON schema.
