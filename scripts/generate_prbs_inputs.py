from pathlib import Path
import numpy as np
import pandas as pd


OUT_PATH = Path("data/processed/prbs_schedule.csv")

SIM_DURATION_S = 300.0
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


def main() -> None:
    n_steps = int(np.floor(SIM_DURATION_S / UPDATE_PERIOD_S)) + 1
    t = np.arange(n_steps) * UPDATE_PERIOD_S

    bits_p = prbs_bits(n_steps, seed=11)
    bits_tb = prbs_bits(n_steps, seed=29)

    p = levels_from_bits(bits_p, P_LEVELS_DBM, seed=101)
    tb = levels_from_bits(bits_tb, TB_LEVELS_S, seed=202)

    df = pd.DataFrame(
        {
            "time_s": t,
            "tx_power_dbm": p,
            "beacon_interval_s": tb,
        }
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved PRBS schedule to: {OUT_PATH}")


if __name__ == "__main__":
    main()