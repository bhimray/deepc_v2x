from pathlib import Path
import argparse
import csv
import numpy as np


OUT_PATH = Path("data/processed/prbs_schedule.csv")

SIM_DURATION_S = 600.0
UPDATE_PERIOD_S = 0.5

P_LEVELS_DBM = np.array([10.0, 13.0, 16.0, 19.0, 23.0])
TB_LEVELS_S = np.array([0.05, 0.1, 0.2, 0.3, 0.5])


def prbs_bits(n: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=n, endpoint=False)


def levels_from_bits(bits: np.ndarray, levels: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(levels), size=len(bits))
    # force switching pattern but allow multi-level mapping
    return levels[idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate DeePC PRBS input schedule CSV.")
    parser.add_argument("--out-path", type=Path, default=OUT_PATH, help="Output PRBS CSV path.")
    parser.add_argument(
        "--duration",
        type=float,
        default=SIM_DURATION_S,
        help="Schedule duration in seconds. Use 600 for a 10 minute simulation.",
    )
    parser.add_argument(
        "--update-period",
        type=float,
        default=UPDATE_PERIOD_S,
        help="Input update period in seconds.",
    )
    parser.add_argument("--power-seed", type=int, default=101, help="Seed for Tx power levels.")
    parser.add_argument("--tb-seed", type=int, default=202, help="Seed for beacon interval levels.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.duration <= 0.0:
        raise ValueError("--duration must be positive")
    if args.update_period <= 0.0:
        raise ValueError("--update-period must be positive")

    n_steps = int(np.floor(args.duration / args.update_period)) + 1
    t = np.arange(n_steps) * args.update_period

    bits_p = prbs_bits(n_steps, seed=11)
    bits_tb = prbs_bits(n_steps, seed=29)

    p = levels_from_bits(bits_p, P_LEVELS_DBM, seed=args.power_seed)
    tb = levels_from_bits(bits_tb, TB_LEVELS_S, seed=args.tb_seed)

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    with args.out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "tx_power_dbm", "beacon_interval_s"])
        writer.writerows(zip(t, p, tb))
    print(f"Saved PRBS schedule to: {args.out_path}")
    print(f"Duration: {t[-1]:.1f} s, update period: {args.update_period:.3f} s, rows: {len(t)}")


if __name__ == "__main__":
    main()
