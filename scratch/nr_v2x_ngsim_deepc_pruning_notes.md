# NR-V2X NGSIM DeePC Dataset Pruning Notes

Date: 2026-05-31

## Purpose

This scratch scenario was pruned to reduce computation time and file I/O for open-loop DeePC dataset generation.

## Outputs Kept

The KPI CSV keeps only:

```text
time_s,tx_power_dbm,beacon_interval_s,active_vehicle_count_core,prr_awareness,pir_s,cbr
```

Standalone CBR CSV output was removed because `cbr` is already sampled in the KPI CSV.

## Removed Outputs And Computation

- Removed `density_veh_per_km_core`.
- Removed `mean_neighbors_150m` and `mean_neighbors_300m`.
- Removed `sensing_exclusion_ratio`.
- Removed pairwise neighbor scans from KPI sampling.
- Removed sensing-exclusion trace storage and computation from `CbrLogger`.

## Removed Logging

- Removed per-packet TX side log generation.
- Removed per-packet RX side log generation.
- Removed periodic `flush()` calls from KPI and CBR sampling.
- Streams now flush naturally when closed.
- Removed standalone CBR CSV generation.

## Default Behavior Changes

- `sampleTimeS` default changed from `0.1 s` to `0.5 s`.
- Sidelink sensing default changed from enabled to disabled.
- `--enableSensing=true` can still be used to re-enable NR sidelink sensing-based resource selection.
- CBR remains computed from `NrSpectrumPhy::ChannelOccupied` busy-time traces.
- The simulator prints sparse progress as `t=<seconds>, v=<active_core_count>`.
- Plotting now consumes KPI CSV files only.

## Expected Runtime Impact

The largest expected savings come from removing packet side-log I/O, reducing sample frequency, avoiding pairwise neighbor counting, and disabling sidelink sensing by default.

## How To Re-Enable Sensing

Run the scratch program with:

```text
--enableSensing=true
```

This restores sensing-based resource selection behavior, but the pruned sensing-exclusion metric remains absent from output.
