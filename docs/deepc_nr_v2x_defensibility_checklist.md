# DeePC NR-V2X Paper Defensibility Checklist

Use this checklist before trusting a table, plot, or abstract claim.

## Experiment Readiness

- Main methods are fixed: `fixed_10hz`, `fixed_5hz`, and `deepc_count_tb0p1_prr_only`.
- Main fair comparison is DeePC versus `fixed_10hz`, because both use 10 Hz CAM generation.
- `fixed_5hz` is described only as a contextual lower-load baseline.
- Every method in the main table uses the same declared run ids.
- Every main-table run reaches the evaluation end time, normally 290 s.
- Warmup and cooldown are consistently 10 s each.
- Awareness range is consistently 200 m.
- Tx-power bounds for DeePC are consistently 10 to 23 dBm.
- CAM interval remains 0.1 s for DeePC.

## Metric Readiness

- PRR is described as awareness-range PRR within 200 m.
- PIR is described as packet inter-reception time.
- CBR is described as PHY busy-time fraction from `NrSpectrumPhy::ChannelOccupied`.
- Tables report 95% confidence intervals across runs.
- The abstract does not include exact percentages until matched repeated runs are complete.
- Runtime metrics include controller success rate and mean solve time.

## Baseline Readiness

- `fixed_10hz` is complete for the matched run set.
- `fixed_5hz` command and produced runs both use 300 s simulation time.
- Threshold DCC is excluded unless the placeholder bridge controller is replaced.
- Any partial or extra DeePC run, such as a one-off tuning run, is described as preliminary evidence only.

## Writing Readiness

- The contribution is stated as DeePC-based closed-loop transmit-power adaptation, not global optimization of NR-V2X.
- The related-work section distinguishes the paper from SPS modeling, DCC, RRI adaptation, and learning-based scheduling.
- The limitations section explicitly mentions one mobility scenario, simulation-only validation, and Tx-power-only control.
- The conclusion says "promising" or "suggests" unless the final matched campaign is complete.

## Commands

Generate paper-facing tables and readiness report:

```bash
./v2x_env/bin/python scripts/aggregate_final_campaign.py
```

Use the exact minimal campaign run commands in:

- `docs/deepc_nr_v2x_minimal_campaign_commands.md`

Primary outputs:

- `data/output/final/ieee_250veh_deepc_campaign_01/03_analysis/paper_readiness_report.md`
- `data/output/final/ieee_250veh_deepc_campaign_01/03_analysis/tables/paper_main_table.csv`
- `data/output/final/ieee_250veh_deepc_campaign_01/03_analysis/tables/aggregate_metrics.csv`
