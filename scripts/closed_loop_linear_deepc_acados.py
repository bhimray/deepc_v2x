#!/usr/bin/env python3
"""Closed-loop linearized DeePC controller using acados.

The script uses the existing ns-3 open-loop DeePC dataset, fits a one-step
linearized DeePC predictor, generates an acados discrete-time OCP solver, and
rolls out a receding-horizon controller on the held-out test context.

It does not run ns-3. The generated control schedule is the bridge artifact that
can later be replayed in ns-3 or wrapped from C++.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("ACADOS_SOURCE_DIR", "/home/bim/ns-3-dev/external/acados")
os.environ["LD_LIBRARY_PATH"] = (
    "/home/bim/ns-3-dev/external/acados-install/lib:"
    + os.environ.get("LD_LIBRARY_PATH", "")
)

import matplotlib

matplotlib.use("Agg")

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver


DEFAULT_DATASET_DIR = Path("data/output/deepc_open_loop_250veh")
DEFAULT_OUT_DIR = DEFAULT_DATASET_DIR / "closed_loop_linear_deepc_acados"
ACADOS_INSTALL_DIR = Path("/home/bim/ns-3-dev/external/acados-install")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and run a closed-loop linearized DeePC acados controller."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--past-horizon", type=int, default=20, help="Tini in samples.")
    parser.add_argument("--control-horizon", type=int, default=10, help="acados horizon N.")
    parser.add_argument("--rollout-steps", type=int, default=400)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--sample-time", type=float, default=0.1)
    parser.add_argument("--target-prr", type=float, default=0.95)
    parser.add_argument(
        "--target-cbr",
        type=float,
        default=0.35,
        help="Soft CBR tracking target. Constraint target remains a later ns-3 validation step.",
    )
    parser.add_argument("--target-pir", type=float, default=0.10)
    parser.add_argument("--target-power", type=float, default=16.0)
    parser.add_argument("--target-tb", type=float, default=0.10)
    parser.add_argument("--weight-prr", type=float, default=40.0)
    parser.add_argument("--weight-pir", type=float, default=3.0)
    parser.add_argument("--weight-cbr", type=float, default=8.0)
    parser.add_argument("--weight-power", type=float, default=0.002)
    parser.add_argument("--weight-tb", type=float, default=0.5)
    parser.add_argument("--weight-du-power", type=float, default=0.05)
    parser.add_argument("--weight-du-tb", type=float, default=15.0)
    parser.add_argument("--min-power", type=float, default=10.0)
    parser.add_argument("--max-power", type=float, default=23.0)
    parser.add_argument("--min-tb", type=float, default=0.05)
    parser.add_argument("--max-tb", type=float, default=0.50)
    parser.add_argument(
        "--qp-solver",
        default="PARTIAL_CONDENSING_HPIPM",
        help="acados QP solver.",
    )
    parser.add_argument(
        "--nlp-solver-type",
        default="SQP_RTI",
        choices=["SQP", "SQP_RTI"],
    )
    parser.add_argument("--build", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def table_to_array(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return df[cols].to_numpy(dtype=float)


def row_flat(values: np.ndarray) -> np.ndarray:
    return values.reshape(-1)


def make_one_step_windows(
    df: pd.DataFrame,
    input_cols: list[str],
    output_cols: list[str],
    context_cols: list[str],
    past_horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    u = table_to_array(df, input_cols)
    y = table_to_array(df, output_cols)
    d = table_to_array(df, context_cols)
    n_windows = len(df) - past_horizon
    if n_windows <= 0:
        raise ValueError("Not enough rows to create one-step windows")

    x_rows = []
    y_rows = []
    for i in range(n_windows):
        x_rows.append(
            np.concatenate(
                [
                    row_flat(u[i : i + past_horizon]),
                    row_flat(y[i : i + past_horizon]),
                    row_flat(d[i : i + past_horizon]),
                    u[i + past_horizon],
                    d[i + past_horizon],
                ]
            )
        )
        y_rows.append(y[i + past_horizon])
    return np.asarray(x_rows), np.asarray(y_rows)


def fit_linear_ridge(x_train: np.ndarray, y_train: np.ndarray, ridge: float) -> np.ndarray:
    x_aug = np.column_stack([x_train, np.ones(len(x_train))])
    lhs = x_aug.T @ x_aug
    reg = ridge * np.eye(lhs.shape[0])
    reg[-1, -1] = 0.0
    rhs = x_aug.T @ y_train
    return np.linalg.solve(lhs + reg, rhs)


def scaler_arrays(cols: list[str], scaler: dict) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray([scaler["mean"][col] for col in cols], dtype=float)
    std = np.asarray([scaler["std"][col] for col in cols], dtype=float)
    return mean, std


def normalize_row(row: np.ndarray, cols: list[str], scaler: dict) -> np.ndarray:
    mean, std = scaler_arrays(cols, scaler)
    return (row - mean) / std


def denormalize_rows(values: np.ndarray, cols: list[str], scaler: dict) -> np.ndarray:
    mean, std = scaler_arrays(cols, scaler)
    return values * std + mean


def shift_append(vec: ca.SX, block_size: int, new_block: ca.SX) -> ca.SX:
    if vec.size1() == block_size:
        return new_block
    return ca.vertcat(vec[block_size:], new_block)


def sx_denormalize(vec: ca.SX, cols: list[str], scaler: dict) -> ca.SX:
    values = []
    for i, col in enumerate(cols):
        values.append(vec[i] * scaler["std"][col] + scaler["mean"][col])
    return ca.vertcat(*values)


def sx_normalize_control(u_phys: ca.SX, input_cols: list[str], scaler: dict) -> ca.SX:
    values = []
    for i, col in enumerate(input_cols):
        values.append((u_phys[i] - scaler["mean"][col]) / scaler["std"][col])
    return ca.vertcat(*values)


def build_acados_solver(
    args: argparse.Namespace,
    beta: np.ndarray,
    scaler: dict,
    input_cols: list[str],
    output_cols: list[str],
    context_cols: list[str],
) -> AcadosOcpSolver:
    nu = len(input_cols)
    ny = len(output_cols)
    nd = len(context_cols)
    nx = args.past_horizon * (nu + ny + nd)
    np_stage = nd + nu

    x = ca.SX.sym("x", nx)
    u = ca.SX.sym("u", nu)
    p = ca.SX.sym("p", np_stage)

    u_buf = x[: args.past_horizon * nu]
    y_buf_start = args.past_horizon * nu
    y_buf = x[y_buf_start : y_buf_start + args.past_horizon * ny]
    d_buf = x[y_buf_start + args.past_horizon * ny :]

    d_next_norm = p[:nd]
    prev_u_phys = p[nd:]
    u_norm = sx_normalize_control(u, input_cols, scaler)
    feature = ca.vertcat(u_buf, y_buf, d_buf, u_norm, d_next_norm)

    beta_no_bias = beta[:-1, :]
    beta_bias = beta[-1, :]
    y_next_norm = ca.DM(beta_no_bias).T @ feature + ca.DM(beta_bias)

    x_next = ca.vertcat(
        shift_append(u_buf, nu, u_norm),
        shift_append(y_buf, ny, y_next_norm),
        shift_append(d_buf, nd, d_next_norm),
    )

    y_next_phys = sx_denormalize(y_next_norm, output_cols, scaler)
    du = u - prev_u_phys

    model = AcadosModel()
    model.name = "linearized_deepc_v2x"
    model.x = x
    model.u = u
    model.p = p
    model.disc_dyn_expr = x_next
    model.cost_y_expr = ca.vertcat(y_next_phys, u, du)
    model.cost_y_expr_e = x

    ocp = AcadosOcp()
    ocp.model = model
    ocp.solver_options.N_horizon = args.control_horizon
    ocp.solver_options.tf = args.control_horizon * args.sample_time
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.qp_solver = args.qp_solver
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.nlp_solver_type = args.nlp_solver_type
    ocp.solver_options.print_level = 0

    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.yref = np.asarray(
        [
            args.target_prr,
            args.target_pir,
            args.target_cbr,
            args.target_power,
            args.target_tb,
            0.0,
            0.0,
        ],
        dtype=float,
    )
    ocp.cost.W = np.diag(
        [
            args.weight_prr,
            args.weight_pir,
            args.weight_cbr,
            args.weight_power,
            args.weight_tb,
            args.weight_du_power,
            args.weight_du_tb,
        ]
    )
    ocp.cost.yref_e = np.zeros(nx)
    ocp.cost.W_e = np.zeros((nx, nx))

    ocp.constraints.x0 = np.zeros(nx)
    ocp.constraints.lbu = np.asarray([args.min_power, args.min_tb], dtype=float)
    ocp.constraints.ubu = np.asarray([args.max_power, args.max_tb], dtype=float)
    ocp.constraints.idxbu = np.asarray([0, 1], dtype=np.int64)
    ocp.parameter_values = np.zeros(np_stage)

    code_export_dir = args.out_dir / "acados_generated"
    code_export_dir.mkdir(parents=True, exist_ok=True)
    ocp.code_gen_opts.code_export_directory = str(code_export_dir)
    ocp.code_gen_opts.acados_include_path = str(ACADOS_INSTALL_DIR / "include")
    ocp.code_gen_opts.acados_lib_path = str(ACADOS_INSTALL_DIR / "lib")
    json_file = args.out_dir / "linearized_deepc_acados_ocp.json"
    return AcadosOcpSolver(ocp, json_file=str(json_file), build=args.build, generate=True)


def initial_state_from_test(
    test_norm: pd.DataFrame,
    input_cols: list[str],
    output_cols: list[str],
    context_cols: list[str],
    past_horizon: int,
) -> np.ndarray:
    return np.concatenate(
        [
            row_flat(table_to_array(test_norm.iloc[:past_horizon], input_cols)),
            row_flat(table_to_array(test_norm.iloc[:past_horizon], output_cols)),
            row_flat(table_to_array(test_norm.iloc[:past_horizon], context_cols)),
        ]
    )


def predict_one_step_norm(
    state: np.ndarray,
    u_phys: np.ndarray,
    d_norm: np.ndarray,
    beta: np.ndarray,
    scaler: dict,
    input_cols: list[str],
) -> np.ndarray:
    u_norm = normalize_row(u_phys, input_cols, scaler)
    feature = np.concatenate([state, u_norm, d_norm])
    return feature @ beta[:-1, :] + beta[-1, :]


def update_state(
    state: np.ndarray,
    u_phys: np.ndarray,
    y_norm: np.ndarray,
    d_norm: np.ndarray,
    scaler: dict,
    input_cols: list[str],
    nu: int,
    ny: int,
    nd: int,
    past_horizon: int,
) -> np.ndarray:
    u_norm = normalize_row(u_phys, input_cols, scaler)
    u_len = past_horizon * nu
    y_len = past_horizon * ny
    d_len = past_horizon * nd
    u_buf = state[:u_len]
    y_buf = state[u_len : u_len + y_len]
    d_buf = state[u_len + y_len : u_len + y_len + d_len]
    return np.concatenate(
        [
            np.concatenate([u_buf[nu:], u_norm]),
            np.concatenate([y_buf[ny:], y_norm]),
            np.concatenate([d_buf[nd:], d_norm]),
        ]
    )


def set_initial_state(solver: AcadosOcpSolver, x0: np.ndarray) -> None:
    solver.set(0, "lbx", x0)
    solver.set(0, "ubx", x0)


def compute_metrics(result: pd.DataFrame) -> dict:
    return {
        "rollout_rows": int(len(result)),
        "mean_deepc_prr_150m": float(result["deepc_prr_150m"].mean()),
        "mean_deepc_pir_s": float(result["deepc_pir_s"].mean()),
        "mean_deepc_cbr": float(result["deepc_cbr"].mean()),
        "mean_measured_open_loop_prr_150m": float(result["measured_open_loop_prr_150m"].mean()),
        "mean_measured_open_loop_pir_s": float(result["measured_open_loop_pir_s"].mean()),
        "mean_measured_open_loop_cbr": float(result["measured_open_loop_cbr"].mean()),
        "deepc_cbr_gt_0p6_rate": float((result["deepc_cbr"] > 0.6).mean()),
        "measured_open_loop_cbr_gt_0p6_rate": float(
            (result["measured_open_loop_cbr"] > 0.6).mean()
        ),
        "mean_tx_power_dbm": float(result["deepc_tx_power_dbm"].mean()),
        "mean_beacon_interval_s": float(result["deepc_beacon_interval_s"].mean()),
    }


def json_safe_args(args: argparse.Namespace) -> dict:
    safe = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            safe[key] = str(value)
        else:
            safe[key] = value
    return safe


def plot_results(result: pd.DataFrame, out_dir: Path) -> None:
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(result["time_s"], result["deepc_prr_150m"], label="acados DeePC surrogate")
    axes[0].plot(
        result["time_s"],
        result["measured_open_loop_prr_150m"],
        "--",
        label="measured open-loop",
    )
    axes[0].set_ylabel("PRR")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(result["time_s"], result["deepc_cbr"], label="acados DeePC surrogate")
    axes[1].plot(result["time_s"], result["measured_open_loop_cbr"], "--", label="measured open-loop")
    axes[1].axhline(0.6, color="k", linestyle=":", linewidth=1)
    axes[1].set_ylabel("CBR")
    axes[1].grid(alpha=0.3)

    axes[2].plot(result["time_s"], result["deepc_pir_s"], label="acados DeePC surrogate")
    axes[2].plot(result["time_s"], result["measured_open_loop_pir_s"], "--", label="measured open-loop")
    axes[2].set_ylabel("PIR (s)")
    axes[2].grid(alpha=0.3)

    axes[3].plot(result["time_s"], result["deepc_tx_power_dbm"], label="Tx power")
    axes[3].set_ylabel("P (dBm)")
    axes[3].grid(alpha=0.3)
    ax2 = axes[3].twinx()
    ax2.plot(result["time_s"], result["deepc_beacon_interval_s"], color="tab:orange", label="Tb")
    ax2.set_ylabel("Tb (s)")
    axes[3].set_xlabel("Time (s)")

    fig.tight_layout()
    fig.savefig(plots / "closed_loop_timeseries.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = load_json(args.dataset_dir / "dataset_summary.json")
    scaler = load_json(args.dataset_dir / "scaler.json")
    input_cols = summary["input_cols"]
    output_cols = summary["output_cols"]
    context_cols = summary["context_cols"]

    nu = len(input_cols)
    ny = len(output_cols)
    nd = len(context_cols)

    train = pd.read_csv(args.dataset_dir / "deepc_train_normalized.csv")
    val = pd.read_csv(args.dataset_dir / "deepc_val_normalized.csv")
    test_norm = pd.read_csv(args.dataset_dir / "deepc_test_normalized.csv")
    test_phys = pd.read_csv(args.dataset_dir / "deepc_test.csv")

    train_final = pd.concat([train, val], ignore_index=True)
    x_train, y_train = make_one_step_windows(
        train_final, input_cols, output_cols, context_cols, args.past_horizon
    )
    beta = fit_linear_ridge(x_train, y_train, args.ridge)
    np.save(args.out_dir / "one_step_linear_deepc_beta.npy", beta)

    solver = build_acados_solver(args, beta, scaler, input_cols, output_cols, context_cols)

    state = initial_state_from_test(
        test_norm, input_cols, output_cols, context_cols, args.past_horizon
    )
    prev_u = table_to_array(test_phys.iloc[[args.past_horizon - 1]], input_cols)[0]
    max_steps = min(args.rollout_steps, len(test_norm) - args.past_horizon - args.control_horizon)
    if max_steps <= 0:
        raise ValueError("Test dataset is too short for requested rollout")

    rows = []
    for k in range(max_steps):
        stage0 = args.past_horizon + k
        set_initial_state(solver, state)
        for i in range(args.control_horizon):
            d_i = table_to_array(test_norm.iloc[[stage0 + i]], context_cols)[0]
            p_i = np.concatenate([d_i, prev_u])
            solver.set(i, "p", p_i)
            solver.set(i, "u", prev_u)
        solver.set(args.control_horizon, "p", np.concatenate([d_i, prev_u]))

        status = solver.solve()
        if status != 0:
            print(f"Warning: acados returned status {status} at rollout step {k}; holding input")
            u_apply = prev_u.copy()
        else:
            u_apply = solver.get(0, "u")

        d_now_norm = table_to_array(test_norm.iloc[[stage0]], context_cols)[0]
        y_pred_norm = predict_one_step_norm(
            state, u_apply, d_now_norm, beta, scaler, input_cols
        )
        y_pred = denormalize_rows(y_pred_norm.reshape(1, -1), output_cols, scaler)[0]
        measured = table_to_array(test_phys.iloc[[stage0]], output_cols)[0]
        context = table_to_array(test_phys.iloc[[stage0]], context_cols)[0]
        rows.append(
            [
                float(test_phys["time_s"].iloc[stage0]),
                float(u_apply[0]),
                float(u_apply[1]),
                float(y_pred[0]),
                float(y_pred[1]),
                float(y_pred[2]),
                float(measured[0]),
                float(measured[1]),
                float(measured[2]),
                *[float(v) for v in context],
                int(status),
            ]
        )

        state = update_state(
            state,
            u_apply,
            y_pred_norm,
            d_now_norm,
            scaler,
            input_cols,
            nu,
            ny,
            nd,
            args.past_horizon,
        )
        prev_u = u_apply

    columns = [
        "time_s",
        "deepc_tx_power_dbm",
        "deepc_beacon_interval_s",
        "deepc_prr_150m",
        "deepc_pir_s",
        "deepc_cbr",
        "measured_open_loop_prr_150m",
        "measured_open_loop_pir_s",
        "measured_open_loop_cbr",
        *context_cols,
        "acados_status",
    ]
    result = pd.DataFrame(rows, columns=columns)
    result.to_csv(args.out_dir / "closed_loop_rollout.csv", index=False)
    result[["time_s", "deepc_tx_power_dbm", "deepc_beacon_interval_s"]].to_csv(
        args.out_dir / "closed_loop_control_schedule.csv", index=False
    )
    metrics = {
        "args": json_safe_args(args),
        "input_cols": input_cols,
        "output_cols": output_cols,
        "context_cols": context_cols,
        "one_step_train_windows": int(len(x_train)),
        "metrics": compute_metrics(result),
    }
    with (args.out_dir / "closed_loop_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    plot_results(result, args.out_dir)

    print(f"Saved acados closed-loop artifacts to: {args.out_dir}")
    print(json.dumps(metrics["metrics"], indent=2))


if __name__ == "__main__":
    main()
