#!/usr/bin/env python3
"""Dummy file-based DeePC controller for ns-3 bridge smoke tests.

The script watches a bridge request directory, reads request_XXXXXX.json files,
and writes matching response_XXXXXX.json files. It is intentionally simple: it
does not solve DeePC. Its job is to prove that ns-3 can pause, wait, read a
controller response, and apply closed-loop controls through the same interface
that MATLAB/YALMIP will use later.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dummy controller for DeePC file bridge.")
    parser.add_argument(
        "--bridge-dir",
        type=Path,
        default=Path("data/output/deepc_matlab_bridge_run01/bridge"),
    )
    parser.add_argument("--tx-power-dbm", type=float, default=16.0)
    parser.add_argument("--beacon-interval-s", type=float, default=0.10)
    parser.add_argument(
        "--mode",
        choices=["fixed", "hold"],
        default="fixed",
        help="fixed writes configured controls; hold returns request previous_u.",
    )
    parser.add_argument("--poll-s", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means run forever.")
    parser.add_argument("--idle-timeout-s", type=float, default=0.0, help="0 means no timeout.")
    return parser.parse_args()


def response_path_for(request_path: Path, response_dir: Path) -> Path:
    suffix = request_path.name.removeprefix("request_")
    return response_dir / f"response_{suffix}"


def write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    tmp.replace(path)


def main() -> None:
    args = parse_args()
    request_dir = args.bridge_dir / "requests"
    response_dir = args.bridge_dir / "responses"
    response_dir.mkdir(parents=True, exist_ok=True)

    processed: set[str] = set()
    completed = 0
    last_activity = time.monotonic()
    print(f"Dummy DeePC controller watching {request_dir}")

    while True:
        requests = sorted(request_dir.glob("request_*.json"))
        did_work = False
        for request_path in requests:
            if request_path.name in processed:
                continue
            response_path = response_path_for(request_path, response_dir)
            if response_path.exists():
                processed.add(request_path.name)
                continue

            start = time.monotonic()
            with request_path.open() as f:
                request = json.load(f)

            if args.mode == "hold":
                previous_u = request.get("previous_u", [args.tx_power_dbm, args.beacon_interval_s])
                tx_power_dbm = float(previous_u[0])
                beacon_interval_s = float(previous_u[1])
            else:
                tx_power_dbm = args.tx_power_dbm
                beacon_interval_s = args.beacon_interval_s

            payload = {
                "step": int(request["step"]),
                "time_s": float(request["time_s"]),
                "success": True,
                "tx_power_dbm": tx_power_dbm,
                "beacon_interval_s": beacon_interval_s,
                "solve_time_s": time.monotonic() - start,
                "controller": "dummy_fixed_file_controller",
            }
            write_json_atomic(response_path, payload)
            processed.add(request_path.name)
            completed += 1
            did_work = True
            last_activity = time.monotonic()
            print(
                f"step={payload['step']} response={response_path} "
                f"u=[{tx_power_dbm:.3f}, {beacon_interval_s:.3f}]"
            )

            if args.max_steps > 0 and completed >= args.max_steps:
                return

        if args.idle_timeout_s > 0 and time.monotonic() - last_activity > args.idle_timeout_s:
            return
        if not did_work:
            time.sleep(args.poll_s)


if __name__ == "__main__":
    main()
