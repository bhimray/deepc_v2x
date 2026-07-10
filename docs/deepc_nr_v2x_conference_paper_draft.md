# DeePC-Based Closed-Loop Transmit-Power Adaptation for NR-V2X Sidelink

Working draft for a focused conference paper.

## Abstract

Reliable vehicle-to-everything communication is essential for cooperative and automated driving, particularly in dense highway scenarios where sidelink resource contention and channel variations can degrade packet reception. This paper investigates data-enabled predictive control (DeePC) for closed-loop transmit-power adaptation in NR-V2X sidelink communication. The proposed controller is trained from input-output measurements collected in ns-3 and is evaluated using realistic NGSIM US-101 vehicle trajectories in a 250-vehicle active-window highway scenario. Performance is measured using awareness-range packet reception ratio (PRR), packet inter-reception time (PIR), and channel busy ratio (CBR) within a 200 m communication-awareness range. The currently collected partial outputs show that the DeePC controller completed 285 online control updates with 100% response success and a mean solve time of 0.0199 s, while producing PRR, PIR, and CBR values comparable to fixed 10 Hz operation. These findings validate the feasibility of closed-loop DeePC for adaptive NR-V2X transmit-power tuning, although matched-duration repeated runs are required before final reliability-gain claims are made.

## 1. Introduction

NR-V2X sidelink enables direct vehicle-to-vehicle communication without relying on network infrastructure. This is important for cooperative awareness, safety messaging, and automated driving support. In dense traffic, however, periodic cooperative awareness messages can create contention, interference, and uneven packet reception. Fixed communication parameters are simple and reproducible, but they do not react to changing vehicle density and radio conditions.

This paper studies whether data-enabled predictive control can adapt NR-V2X transmit power online using measured communication outcomes rather than an explicit analytical network model. The central idea is to learn predictive behavior from open-loop simulation data and then use recent PRR, PIR, CBR, and vehicle-density context to choose transmit-power actions in closed loop.

The contributions are:

1. An ns-3/NGSIM evaluation pipeline for closed-loop NR-V2X CAM dissemination with packet-level KPI logging.
2. A DeePC controller that adapts transmit power every 0.5 s while keeping CAM generation fixed at 10 Hz.
3. A matched experimental comparison against fixed 10 Hz operation and contextual lower-rate baselines under realistic US-101 highway mobility.

## 2. Related Work

NR-V2X sidelink and its Release 16 design are summarized by Garcia et al. Their tutorial motivates sidelink as a key component of advanced V2X use cases and provides the standards background for physical-layer and resource-allocation assumptions.

Congestion-control work in C-V2X often studies DCC, packet dropping, resource reservation interval adaptation, and sensing-based semi-persistent scheduling. McCarthy and O'Driscoll evaluate standardized DCC behavior and propose RRI-adaptive approaches for sidelink congestion. Cao et al. model NR sidelink Mode 2 PRR under SPS behavior and validate the model with ns-3. Dayal et al. study adaptive RRI selection for cooperative awareness in decentralized V2X.

Compared with these works, this paper does not propose a new SPS model or standards-level DCC rule. Instead, it evaluates a data-driven closed-loop controller that uses measured input-output behavior to tune transmit power in a realistic ns-3 NR-V2X scenario. DeePC has been used in connected-vehicle traffic control contexts, such as the data-driven predictive control work of Wang et al., but here the controlled system is the communication layer rather than longitudinal vehicle motion.

## 3. System Model and Scenario

The scenario uses NGSIM US-101 highway trajectories processed into an active-window mobility file:

- Mobility: `data/processed/ngsim_us101_mainline_active20_250_densest_600s.csv`
- Logical vehicle tracks: 250 selected active-window tracks
- Physical UE pool: 250
- Simulation time: 300 s
- Warmup and cooldown: 10 s each
- Awareness range: 200 m
- CAM generation: fixed periodic generation
- Main CAM interval: 0.1 s
- Carrier frequency: 5.9 GHz
- HARQ: disabled
- Sidelink sensing: enabled

The primary metrics are:

- PRR: successful unique Tx-Rx-sequence receptions divided by eligible receiver opportunities within 200 m.
- PIR: packet inter-reception time over received CAMs.
- CBR: PHY busy-time fraction measured from `NrSpectrumPhy::ChannelOccupied`.

## 4. DeePC Method

The controller uses transmit power as the controlled input. Beacon interval is held fixed at 0.1 s in the main comparison so the DeePC result is compared fairly against fixed 10 Hz operation.

Inputs and outputs:

- Controlled input: `tx_power_dbm`
- Outputs: `prr_awareness`, `pir_s`, `cbr`
- Context: `active_vehicle_count_core`, `beacon_interval_s`

The DeePC dataset uses:

- Sample time: 0.5 s
- Past horizon: 20 samples, or 10 s
- Future horizon: 10 samples, or 5 s
- Tx-power bounds: 10 to 23 dBm
- Online control interval: 0.5 s

At each control update, the controller receives recent measured behavior, solves the DeePC optimization problem, and applies the first predicted transmit-power action to the ns-3 scenario through the file bridge.

## 5. Evaluation Plan

The minimum conference-ready result set is:

| Method | Role | Final-use condition |
|---|---|---|
| `fixed_10hz` | Primary baseline | Runs 101-105 complete to 300 s |
| `fixed_5hz` | Contextual lower-load baseline | Runs 101-105 complete to 300 s |
| `deepc_count_tb0p1_prr_only` | Proposed method | Runs 101-105 or another declared matched seed set complete to 300 s |

Threshold DCC should be excluded from the main table until the placeholder controller is replaced by a real threshold-DCC implementation and validated.

The main paper table should report:

- Mean PRR with 95% confidence interval
- Mean PIR with 95% confidence interval
- Mean CBR with 95% confidence interval
- CBR > 0.6 rate
- Controller success rate
- Mean controller solve time

The main plots should show:

- PRR over time for fixed 10 Hz and DeePC
- DeePC transmit power over time
- PRR, PIR, and CBR comparison bars with confidence intervals
- Optional PRR by active-vehicle-density bin

## 6. Current Collected Results

The current collected results should be used as **preliminary feasibility evidence**, not as final proof that DeePC improves reliability. The available campaign aggregation reports the following orientation metrics:

| Method | Mean PRR | Mean PIR (s) | Mean CBR | CBR > 0.6 | Controller success | Mean solve time (s) |
|---|---:|---:|---:|---:|---:|---:|
| Fixed 10 Hz, 20 dBm | 0.5748 | 0.1601 | 0.4103 | 0.0000 | -- | -- |
| Fixed 5 Hz, 20 dBm | 0.5527 | 0.3028 | 0.3238 | 0.0000 | -- | -- |
| DeePC Tx-power control | 0.5739 | 0.1603 | 0.4104 | 0.0000 | 1.0000 | 0.0199 |

Interpretation:

1. DeePC is not yet shown to outperform fixed 10 Hz on PRR using the currently collected partial data.
2. DeePC gives nearly identical PRR, PIR, and CBR to the partial fixed 10 Hz baseline while actively changing transmit power.
3. The strongest current result is closed-loop feasibility: 285 control updates, 100% controller response success, and solve time far below the 0.5 s control interval.
4. The fixed 5 Hz baseline lowers CBR but increases PIR, which is useful as a contextual tradeoff but not the fairest direct comparison.

This means the current paper should say:

> The collected results validate the feasibility of DeePC-based online transmit-power adaptation for NR-V2X and motivate matched repeated-run evaluation.

It should not yet say:

> DeePC improves PRR over fixed 10 Hz.

## 7. Results Narrative Template

Use the following structure once matched runs are complete:

1. Reliability: DeePC improves awareness-range PRR compared with the fixed 10 Hz baseline.
2. Timeliness: DeePC reduces or maintains PIR relative to fixed operation.
3. Congestion: DeePC does not increase CBR beyond the fixed 10 Hz operating level and keeps CBR > 0.6 rare or absent.
4. Runtime: Controller solve time remains below the 0.5 s control interval with high controller success rate.
5. Density sensitivity: DeePC remains beneficial or stable under higher active-vehicle-count bins.

Avoid exact percentage claims until all methods use matched completed runs.

## 8. Limitations

This study is currently limited to one processed NGSIM US-101 active-window mobility scenario. The controller adapts transmit power only; CAM generation rate, SPS parameters, MCS, and resource-selection parameters are not controlled. The results are simulation-based and do not include field validation. Broader density cases, alternate traces, and a real DCC baseline are important extensions.

## 9. Conclusion

This paper studies DeePC as a model-free mechanism for adaptive NR-V2X sidelink transmit-power control. Using realistic NGSIM mobility and ns-3 packet-level measurements, the currently collected results validate the feasibility of closed-loop data-driven transmit-power adaptation and show real-time-compatible controller execution. The final contribution will be strongest after matched repeated runs determine whether this feasibility result also becomes a statistically defendable reliability improvement.

## Working References

- Mario H. C. Garcia et al., "A Tutorial on 5G NR V2X Communications." https://arxiv.org/abs/2102.04538
- Brian McCarthy and Aisling O'Driscoll, "Congestion Control in the Cellular-V2X Sidelink." https://arxiv.org/abs/2106.04871
- Liu Cao, Sumit Roy, and Collin Brady, "Semi-Persistent Scheduling in NR Sidelink Mode 2: MAC Packet Reception Ratio Model and ns-3 Validation." https://arxiv.org/abs/2309.16680
- Avik Dayal et al., "Adaptive RRI Selection Algorithms for Improved Cooperative Awareness in Decentralized NR-V2X." https://arxiv.org/abs/2307.12473
- Jiawei Wang et al., "Data-Driven Predictive Control for Connected and Autonomous Vehicles in Mixed Traffic." https://arxiv.org/abs/2110.10097
