# Progress Evaluation Report: DeePC-Based NR-V2X Sidelink Transmit-Power Adaptation

Author: Bim  
Project: ns-3 NR-V2X / NGSIM US-101 DeePC study  
Report purpose: professor evaluation of completed research and engineering progress  
Date: July 2026

## Executive Summary

This report summarizes the current state of my research on Data-Enabled
Predictive Control (DeePC) for adaptive NR-V2X sidelink communication. The
central goal is to test whether measured vehicle-to-everything input-output
data can support online transmit-power control in a realistic ns-3 NR-V2X
simulation, with the controller balancing communication reliability,
timeliness, and congestion. Reliability is measured using awareness-range
packet reception ratio (PRR), timeliness is measured using packet
inter-reception time (PIR), and congestion is measured using channel busy ratio
(CBR).

The main completed milestone is an end-to-end DeePC research pipeline. I have
constructed DeePC-ready datasets from ns-3 KPI traces, generated normalized
Hankel matrices, implemented open-loop DeePC predictors, built closed-loop
controller infrastructure through an ns-3 JSON file bridge, added a Python OSQP
controller for online transmit-power control, prepared MATLAB/YALMIP export
support, and generated campaign-level analysis tables and readiness reports.
The current controller can receive recent ns-3 measurements, solve the DeePC
optimization problem, and return a transmit-power command during the simulation.

The strongest current result is feasibility rather than final performance
superiority. In the available preliminary closed-loop run, the DeePC controller
completed 285 online control updates with a 100% controller success rate and a
mean solve time of about 0.0199 s, which is far below the 0.5 s control
interval. The preliminary KPI values are close to the fixed 10 Hz baseline:
mean PRR is 0.5739 for DeePC and 0.5748 for fixed 10 Hz in the currently
available partial evidence. Therefore, the work currently supports the claim
that the closed-loop DeePC control pipeline is implemented and real-time
compatible in this simulation setup. It does not yet support a final claim that
DeePC improves PRR over fixed 10 Hz operation.

The remaining major evaluation step is to complete matched repeated ns-3 runs
for the main methods. The current paper-readiness report shows that the
intended repeated campaign is not complete: fixed 10 Hz and fixed 5 Hz have
partial run-101 evidence, the DeePC run used for orientation is an extra
non-matched run, and run IDs 101-105 are still missing for the main DeePC
method. For this reason, all closed-loop comparison results in this report are
treated as preliminary feasibility evidence.

## Problem and Motivation

NR-V2X sidelink communication enables vehicles to exchange cooperative
awareness messages directly, without routing every message through cellular
infrastructure. This is important for cooperative and automated driving because
nearby vehicles need timely information about position, speed, and motion state.
In dense traffic, however, periodic message generation can create resource
contention and interference. If transmit power is too low, vehicles may miss
important packets. If transmit power is unnecessarily high, the channel can
become more congested and neighboring transmissions may interfere more strongly.

Fixed communication settings are easy to reproduce and are useful as baselines,
but they do not react to measured network conditions. A fixed 10 Hz CAM
configuration may be reasonable in moderate conditions, yet the best
transmit-power choice can depend on the number of active vehicles, recent PRR,
recent CBR, and other measured state. The research question for this stage is:

**Can measured NR-V2X input-output data support real-time predictive
transmit-power control using DeePC?**

This question is deliberately narrower than global optimization of NR-V2X. The
current controller does not change SPS behavior, MCS, sensing parameters, CAM
payload structure, or the reservation period. It focuses on transmit power as
the controlled input while keeping the main comparison at fixed 10 Hz CAM
generation. This makes the comparison with a fixed 10 Hz baseline cleaner: both
methods generate CAMs at the same nominal rate, while DeePC is allowed to adapt
transmit power.

The control objective is to maintain or improve PRR while avoiding excessive
CBR and keeping PIR reasonable. PRR and PIR measure application-facing awareness
quality, while CBR measures congestion at the PHY layer. These metrics create a
natural tradeoff: a lower message rate or lower transmit power can reduce CBR,
but it may also increase PIR or reduce awareness reliability. DeePC is
attractive because it can use collected input-output data directly rather than
requiring a closed-form analytical model of the full NR-V2X network.

## System and Dataset

The simulation environment is ns-3 with an NR-V2X sidelink scenario driven by
processed NGSIM US-101 highway mobility. The final campaign folder identifies
the scenario as "NR-V2X CAM DeePC control over NGSIM US-101 mainline
active-window mobility" and uses the mobility file
`data/processed/ngsim_us101_mainline_active20_250_densest_600s.csv`. The
campaign configuration uses a 250-vehicle physical UE pool and dynamic active
assignment. The processed mobility metadata reports 852 logical vehicle tracks,
a peak of 179 simultaneous active vehicles, and a peak of 118 active core
vehicles.

The main evaluation configuration is:

| Item | Value |
|---|---:|
| Simulator | ns-3 NR-V2X sidelink |
| Mobility | NGSIM US-101 processed highway trajectories |
| Physical UE pool | 250 UEs |
| Simulation target | 300 s |
| Warmup / cooldown | 10 s / 10 s |
| KPI sample time | 0.5 s for final campaign data |
| Control interval | 0.5 s |
| Awareness range | 200 m |
| Main CAM interval | 0.1 s, or 10 Hz |
| Carrier frequency | 5.9 GHz |
| MCS | 6 |
| HARQ | disabled in the current campaign |
| Sidelink sensing | enabled |
| DeePC transmit-power bounds | 10 to 23 dBm |

The main DeePC signals are:

| Role | Signals |
|---|---|
| Controlled input | `tx_power_dbm` |
| Fixed/context input | `beacon_interval_s` |
| Outputs | `prr_awareness`, `pir_s`, `cbr` |
| Traffic context | `active_vehicle_count_core` and related density variables |

The primary closed-loop dataset for the final campaign is stored under
`data/output/final/ieee_250veh_deepc_campaign_01/01_training/`. The
Tx-power-only, fixed-10-Hz DeePC dataset is
`deepc_dataset_txpower_count_tb0p1_dt0p5_n10`. Its source KPI file is
`kpi_timeseries_tb0p1_for_deepc.csv`, and it uses a sample time of 0.5 s. The
clean evaluation window contains 561 rows from 10.0 s to 290.0 s. The split is
336 training rows, 112 validation rows, and 113 test rows. The DeePC past
horizon is 20 samples, or 10 s, and the future horizon is 10 samples, or 5 s.

The resulting Hankel blocks for the Tx-power-only final-campaign controller are:

| Hankel block | Shape |
|---|---:|
| `u_p` | 20 x 307 |
| `u_f` | 10 x 307 |
| `y_p` | 60 x 307 |
| `y_f` | 30 x 307 |
| `d_p` | 40 x 307 |
| `d_f` | 20 x 307 |

## Implemented Work

### DeePC Data Pipeline

The first completed milestone was the offline DeePC data pipeline. I created a
configurable script that can clean the KPI series, select input/output/context
columns, split data into train/validation/test segments, normalize each signal,
and export Hankel matrices. The generated artifacts include clean CSVs,
normalized CSVs, `scaler.json`, `dataset_summary.json`, and
`hankel_train_normalized.npz`.

### Open-Loop Prediction

I implemented held-out open-loop prediction with a linear DeePC-style
predictor using the current final-campaign training dataset. The held-out
metrics are:

| Model | Output | RMSE | MAE | R2 |
|---|---|---:|---:|---:|
| Linear DeePC | PRR | 0.049929 | 0.038993 | -1.063849 |
| Linear DeePC | PIR | 0.039214 | 0.032651 | 0.113547 |
| Linear DeePC | CBR | 0.019695 | 0.016011 | 0.356386 |

These results show that the current final-campaign dataset supports preliminary
data-driven prediction, but the prediction quality is mixed. CBR prediction is
the strongest of the three outputs, while PRR prediction requires additional
data and matched-run validation before it should be used as a final paper
claim.

### Closed-Loop Infrastructure

The ns-3 scenario has been prepared for true controller-in-the-loop simulation
through a JSON file bridge. In bridge mode, ns-3 writes request files containing
recent control, KPI, and context histories. An external controller reads those
requests, solves an optimization problem, writes response files, and ns-3
applies the returned control action.

Implemented bridge behavior includes:

- request files under `bridge/requests/request_XXXXXX.json`;
- response files under `bridge/responses/response_XXXXXX.json`;
- applied-control logging in `bridge/applied_controls.csv`;
- controller timing in `bridge/controller_solve_times.csv`;
- protection against overwriting the original open-loop dataset;
- rejection of bridge mode when the PRBS input schedule is still enabled.

The dummy controller smoke test verified that the file bridge works: ns-3
completed normally, 47 bridge responses were applied, and the applied-controls
log confirmed that the externally supplied transmit power and beacon interval
were used.

### Controllers

Several controller paths have been implemented or prepared.

First, a MATLAB/YALMIP path was prepared by exporting the normalized Hankel
matrices and metadata to `matlab_deepc_data.mat`. The MATLAB file controller can
watch the same bridge directory, solve a classical DeePC optimization in
YALMIP, and write response JSON files using the same schema. This path has not
been executed inside the Linux workspace because MATLAB is available separately
on Windows.

Second, a Python OSQP file controller was implemented in
`scripts/deepc_osqp_file_controller.py`. This controller loads
`dataset_summary.json`, `scaler.json`, and `hankel_train_normalized.npz`, checks
that the request horizons and signal columns match the dataset, builds a
CVXPY/OSQP optimization problem, enforces transmit-power bounds from 10 to
23 dBm, and writes predicted PRR, PIR, CBR, future transmit-power trajectories,
solver status, solve time, and equality-residual diagnostics.

Third, an acados-based offline surrogate controller was implemented. The acados
work verifies that a receding-horizon controller can be generated and rolled out
on a learned surrogate. The surrogate rollout improved predicted PRR and
lowered predicted CBR relative to the measured held-out open-loop segment, but
this is not yet an ns-3 closed-loop result and should not be presented as final
network performance evidence.

### Campaign Tooling

The final campaign folder is
`data/output/final/ieee_250veh_deepc_campaign_01/`. It contains a campaign
manifest, training artifacts, repeated-run directories, command scripts,
analysis tables, plots, and an acceptance checklist. The aggregation script
`scripts/aggregate_final_campaign.py` produces paper-facing outputs including
`paper_readiness_report.md`, `paper_main_table.csv`, `aggregate_metrics.csv`,
`run_metrics.csv`, density-bin tables, and comparison plots.

This is important because the project is now reproducible enough to evaluate:
there is a documented folder contract, exact command scripts for major stages,
and generated readiness reports that identify which claims are safe and which
runs are still missing.

## Results and Evidence

### Open-Loop Predictor Evidence

The open-loop prediction results show that the collected ns-3 KPI data contains
some input-output structure for DeePC-style prediction. On the current
final-campaign training dataset, the linear predictor achieves overall RMSE
values of 0.049929 for PRR, 0.039214 for PIR, and 0.019695 for CBR.

These values support the first research milestone: before attempting full
closed-loop control, the data-driven model can predict the measured
communication outputs on held-out data, but the mixed R2 values mean the
open-loop predictor should be treated as preliminary evidence rather than a
final model-quality claim.

### Preliminary Closed-Loop Feasibility

The current closed-loop comparison is preliminary because the repeated matched
campaign is not finished. The available aggregate evidence is:

| Method | Mean PRR | Mean PIR (s) | Mean CBR | CBR > 0.6 rate | Controller success | Mean solve time (s) |
|---|---:|---:|---:|---:|---:|---:|
| Fixed 10 Hz, 20 dBm | 0.5748 | 0.1601 | 0.4103 | 0.0000 | -- | -- |
| Fixed 5 Hz, 20 dBm | 0.5527 | 0.3028 | 0.3238 | 0.0000 | -- | -- |
| DeePC Tx-power control | 0.5739 | 0.1603 | 0.4104 | 0.0000 | 1.0000 | 0.0199 |

The fairest direct comparison is DeePC versus fixed 10 Hz because both use 10 Hz
CAM generation. Fixed 5 Hz is included only as a contextual lower-load baseline:
it lowers CBR, but it also increases PIR and is not the same operating point.

The preliminary DeePC result is very close to fixed 10 Hz in PRR, PIR, and CBR.
This means the current evidence should not be described as a demonstrated PRR
gain. The positive result is that DeePC actively changes transmit power while
maintaining comparable KPI behavior and solving fast enough for the online
control interval. The available run reports 285 control updates, 100% response
success, a mean solve time of 0.0199 s, and a maximum solve time of 0.167194 s.
Both timing values are below the 0.5 s control interval.

### Evidence Boundary

The paper-readiness report explicitly marks the repeated campaign as incomplete.
For the intended matched run IDs 101-105, fixed 10 Hz currently has only a
partial run 101 with KPI data ending at 161.0 s instead of the 290 s evaluation
end. Fixed 5 Hz has only a partial run 101 with KPI data ending at 119.5 s.
The main DeePC method has missing run IDs 101-105, and the current DeePC result
comes from an extra non-matched run, run 931.

Therefore, the correct interpretation is:

> The collected results validate the feasibility of DeePC-based online
> transmit-power adaptation for NR-V2X and motivate matched repeated-run
> evaluation.

The report should not claim:

> DeePC improves PRR over fixed 10 Hz.

## Defensibility and Limitations

The work is defensible as a research progress milestone because it has moved
from raw ns-3 traces to an operational closed-loop control pipeline. The
implemented pieces include data extraction, DeePC dataset construction,
open-loop validation, bridge-based ns-3 integration, online optimization,
diagnostic logging, and campaign aggregation. These are substantial engineering
and research contributions even before the final statistical campaign is
complete.

The current defensible claims are:

1. DeePC-ready NR-V2X datasets have been built from ns-3 NGSIM mobility traces.
2. Linear DeePC-style prediction has been evaluated on held-out PRR, PIR, and
   CBR using the current final-campaign training dataset.
3. The ns-3 file bridge can apply external controller commands during a run.
4. The Python OSQP DeePC controller can solve online transmit-power commands
   with solve times well below the 0.5 s control interval in the preliminary
   run.
5. The current closed-loop evidence is comparable to fixed 10 Hz in the partial
   run but does not yet prove improvement.

The main limitations are:

1. The matched repeated campaign is incomplete, so confidence intervals across
   repeated runs are not meaningful yet.
2. The work currently uses one processed NGSIM US-101 active-window mobility
   scenario.
3. The validation is simulation-based and does not include field experiments.
4. The current controller adapts transmit power only; it does not control SPS
   parameters, MCS, or resource-selection settings.
5. Threshold-DCC should remain excluded from the main result table until the
   placeholder controller is replaced by a validated implementation.

## Next Steps

The most important next step is to complete the matched repeated ns-3 campaign.
The minimum professor- and paper-ready campaign should include the same run IDs
for fixed 10 Hz, fixed 5 Hz, and the selected DeePC method:
`fixed_10hz`, `fixed_5hz`, and `deepc_count_tb0p1_prr_only` for run IDs
101-105. Each run should reach the evaluation end time, normally 290 s for the
300 s simulation with 10 s cooldown.

After completing the runs, I should regenerate the campaign analysis with:

```bash
./v2x_env/bin/python scripts/aggregate_final_campaign.py
```

The regenerated outputs should include:

- `data/output/final/ieee_250veh_deepc_campaign_01/03_analysis/paper_readiness_report.md`;
- `data/output/final/ieee_250veh_deepc_campaign_01/03_analysis/tables/paper_main_table.csv`;
- `data/output/final/ieee_250veh_deepc_campaign_01/03_analysis/tables/aggregate_metrics.csv`;
- comparison plots under `data/output/final/ieee_250veh_deepc_campaign_01/03_analysis/plots/`.

Once the matched campaign is complete, the results should be checked against the
defensibility checklist before any paper-style claim is made. The main table
should report mean PRR, mean PIR, mean CBR, CBR > 0.6 rate, controller success
rate, and mean solve time with 95% confidence intervals across runs. The
analysis should also include density-bin metrics and prediction-error
diagnostics, because DeePC may be most useful in particular traffic-density
regimes rather than uniformly across the whole scenario.

If the completed matched runs show a statistically meaningful improvement or
stability advantage, this progress report can be converted into a conference
paper. If the completed runs show similar performance to fixed 10 Hz, the paper
can still be framed around feasibility, reproducibility, and lessons learned
for data-driven NR-V2X control.

## Evidence and Reproducibility Pointers

The numerical claims in this report trace to the following artifacts:

| Evidence | Artifact |
|---|---|
| Research history and milestones | `docs/deepc_research_progress.md` |
| Conservative paper narrative | `docs/deepc_nr_v2x_conference_paper_draft.md` |
| Claim safety checklist | `docs/deepc_nr_v2x_defensibility_checklist.md` |
| Open-loop prediction metrics | `data/output/final/ieee_250veh_deepc_campaign_01/01_training/open_loop_predictions/metrics.json` |
| Final campaign manifest | `data/output/final/ieee_250veh_deepc_campaign_01/campaign_manifest.json` |
| Tx-power DeePC dataset summary | `data/output/final/ieee_250veh_deepc_campaign_01/01_training/deepc_dataset_txpower_count_tb0p1_dt0p5_n10/dataset_summary.json` |
| Preliminary closed-loop table | `data/output/final/ieee_250veh_deepc_campaign_01/03_analysis/tables/paper_main_table.csv` |
| Run-level preliminary metrics | `data/output/final/ieee_250veh_deepc_campaign_01/03_analysis/tables/run_metrics.csv` |
| Paper readiness status | `data/output/final/ieee_250veh_deepc_campaign_01/03_analysis/paper_readiness_report.md` |
| Acceptance checklist | `data/output/final/ieee_250veh_deepc_campaign_01/04_tests/acceptance_checklist.md` |

Suggested figures for the final report or presentation are:

- Final-campaign PRR comparison:
  `data/output/final/ieee_250veh_deepc_campaign_01/03_analysis/plots/mean_prr_awareness_comparison.png`
- Final-campaign PIR comparison:
  `data/output/final/ieee_250veh_deepc_campaign_01/03_analysis/plots/mean_pir_s_comparison.png`
- Final-campaign CBR comparison:
  `data/output/final/ieee_250veh_deepc_campaign_01/03_analysis/plots/mean_cbr_comparison.png`

## Conclusion

This project has reached a meaningful implementation and feasibility milestone.
I have built the DeePC data pipeline, validated open-loop prediction, prepared
the ns-3 closed-loop bridge, implemented online transmit-power optimization,
and generated campaign-level analysis tooling. The preliminary closed-loop run
shows that the controller can operate within the required control interval and
produce KPI behavior comparable to fixed 10 Hz operation.

The correct current conclusion is conservative: the work demonstrates a working
end-to-end DeePC control pipeline and preliminary real-time feasibility for
NR-V2X transmit-power adaptation. The next phase is to complete matched repeated
runs so the final paper can determine whether this feasibility result also
becomes a statistically defensible reliability or congestion-control
improvement.
