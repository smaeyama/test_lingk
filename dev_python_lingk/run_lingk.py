#!/usr/bin/env python
# coding: utf-8

"""Reference linear gyrokinetic solver with Fortran-like outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import xarray as xr
from tqdm import tqdm

from linear_gyrokinetic import (
    GKParameters,
    RungeKuttaGillStepper,
    build_geometry,
    compute_time_step_control,
    complex_to_parts,
    compute_density_moment,
    init_state,
)


EPS = 1.0e-10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the reference linear gyrokinetic solver")
    parser.add_argument("--output-dir", default="lingk_output")
    parser.add_argument("--fkinzv-out", default=None)
    parser.add_argument("--mominz-out", default=None)
    parser.add_argument("--frq-out", default=None)
    parser.add_argument("--param-namelist", default=None)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--dt-out", type=float, default=0.1)
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--max-steps", type=int, default=1_000_000)
    parser.add_argument("--nz", type=int, default=24 * 5)
    parser.add_argument("--nv", type=int, default=32)
    parser.add_argument("--nm", type=int, default=31)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--im", type=int, action="append", default=None)
    parser.add_argument("--ky", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--disable-dtc", action="store_true")
    parser.add_argument("--disable-progress", action="store_true")
    return parser.parse_args()


def _parse_namelist(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}

    values: dict[str, Any] = {}
    for raw_line in Path(path).read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("&") or line == "/":
            continue
        if "=" not in line:
            continue
        key, value = [part.strip() for part in line.split("=", 1)]
        values[key] = _parse_namelist_value(value)
    return values


def _parse_namelist_value(value: str) -> Any:
    token = value.strip().rstrip(",")
    lower = token.lower()
    if lower in {".true.", ".false."}:
        return lower == ".true."
    if lower.endswith("d0"):
        return float(lower[:-2])
    try:
        return int(token)
    except ValueError:
        return float(token.replace("d", "e"))


def _build_parameters(args: argparse.Namespace) -> GKParameters:
    params = GKParameters(nz=args.nz, nv=args.nv, nm=args.nm, dt=args.dt, dt_out=args.dt_out, seed=args.seed)
    namelist = _parse_namelist(args.param_namelist)

    scalar_keys = ("kx", "ky", "eps_r", "q_0", "s_hat", "lambda", "beta")
    vector_keys = ("R0_Ln", "R0_Lt", "nu", "Anum", "Znum", "fcs", "sgn", "tau")

    for key in scalar_keys:
        attr = "lambda_" if key == "lambda" else key
        if key in namelist:
            setattr(params, attr, namelist[key])

    for key in vector_keys:
        if key in namelist:
            getattr(params, key)[:] = namelist[key]

    if args.ky is not None:
        params.ky = args.ky
    if args.beta is not None:
        params.beta = args.beta
    return params


def _output_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fkinzv = Path(args.fkinzv_out) if args.fkinzv_out else out_dir / "fkinzv.nc"
    mominz = Path(args.mominz_out) if args.mominz_out else out_dir / "mominzt.nc"
    frq = Path(args.frq_out) if args.frq_out else out_dir / "frq.txt"
    return fkinzv, mominz, frq


def _mu_indices(args: argparse.Namespace, params: GKParameters) -> np.ndarray:
    if args.im:
        indices = np.array(sorted(set(args.im)), dtype=int)
    else:
        indices = np.array([params.nm // 4], dtype=int)
    if np.any(indices < 0) or np.any(indices > params.nm):
        raise ValueError(f"mu indices must satisfy 0 <= im <= {params.nm}")
    return indices


def _linfreq_rows(times: np.ndarray, pk_series: np.ndarray) -> tuple[list[str], bool]:
    header = "#            time           growth        frequency       diff(grow)       diff(freq)         1-Ineq."
    lines = [header]
    stop_signal = False
    time0 = None
    pk0 = None
    omega0 = 0.0 + 0.0j
    pk0_norm2 = None

    for time, pk in zip(times, pk_series):
        if time < times[0] + (times[1] - times[0] if len(times) > 1 else 0.0) - EPS:
            time0 = time
            pk0 = pk.copy()
            pk0_norm2 = float(np.sum(np.abs(pk0[1:-1]) ** 2))
            continue

        assert time0 is not None and pk0 is not None and pk0_norm2 is not None
        pk0pk = np.sum(np.conjugate(pk0[1:-1]) * pk[1:-1])
        pk_norm2 = float(np.sum(np.abs(pk[1:-1]) ** 2))
        omega = np.log(pk0pk / pk0_norm2) / (1j * (time0 - time))

        diff_real = abs(np.real(omega - omega0) / np.real(omega)) if not np.isclose(np.real(omega), 0.0) else 0.0
        diff_imag = abs(np.imag(omega - omega0) / np.imag(omega)) if not np.isclose(np.imag(omega), 0.0) else 0.0
        ineq = abs(pk0pk) ** 2 / (pk0_norm2 * pk_norm2) if pk_norm2 > 0.0 else 1.0

        lines.append(
            f"{time:17.7e}{np.imag(omega):17.7e}{np.real(omega):17.7e}{diff_imag:17.7e}{diff_real:17.7e}{(1.0 - ineq):17.7e}"
        )

        if diff_real < 1.0e-4 and diff_imag < 1.0e-4 and (1.0 - ineq) < 1.0e-4:
            lines.append("# Well converged.")
            lines.append("#              kx               ky       Growthrate        Frequency       Diff(grow)       Diff(freq)         1-Ineq")
            stop_signal = True
            break

        time0 = time
        pk0 = pk.copy()
        omega0 = omega
        pk0_norm2 = pk_norm2

    return lines, stop_signal


def main() -> None:
    total_start = perf_counter()
    init_elapsed = 0.0
    rkg_elapsed = 0.0
    sample_elapsed = 0.0
    output_elapsed = 0.0

    init_start = perf_counter()
    args = parse_args()
    params = _build_parameters(args)
    geom = build_geometry(params)
    dt_control = compute_time_step_control(geom)
    if not args.disable_dtc:
        params.dt = dt_control["dt"]
    mu_indices = _mu_indices(args, params)
    fkinzv_path, mominz_path, frq_path = _output_paths(args)

    print(" # Time step size control")
    print("")
    print(f" # courant num. = {dt_control['courant_num']:20.15f}")
    print(f" # dt_perp      = {dt_control['dt_perp']:23.15E}")
    print(f" # dt_zz        = {dt_control['dt_zz']:23.15E}")
    print(f" # dt_vl        = {dt_control['dt_vl']:23.15E}")
    print(f" # dt_col       = {dt_control['dt_col']:23.15E}")
    print(f" # dt           = {dt_control['dt']:23.15E}")
    print("")

    hk, fk, pk, ak = init_state(params, geom)
    stepper = RungeKuttaGillStepper(geom)
    init_elapsed += perf_counter() - init_start

    fkinzv_times: list[float] = []
    fkinzv_series: list[np.ndarray] = []
    mominz_times: list[float] = []
    phi_series: list[np.ndarray] = []
    a_series: list[np.ndarray] = []
    dens_series: list[np.ndarray] = []
    pk_for_freq: list[np.ndarray] = []

    time = 0.0
    time_out = time + params.dt_out - EPS

    def record_state(current_time: float, current_fk: np.ndarray, current_pk: np.ndarray, current_ak: np.ndarray) -> None:
        density = compute_density_moment(current_fk, geom)
        fkinzv_times.append(current_time)
        fkinzv_series.append(current_fk[:, :, mu_indices, :])
        mominz_times.append(current_time)
        phi_series.append(current_pk.copy())
        a_series.append(current_ak.copy())
        dens_series.append(density)
        pk_for_freq.append(current_pk.copy())

    sample_start = perf_counter()
    record_state(time, fk, pk, ak)
    sample_elapsed += perf_counter() - sample_start

    iterator = range(args.max_steps + 1)
    if not args.disable_progress:
        iterator = tqdm(iterator, desc="lingk-reference", unit="step")

    for istep in iterator:
        if time > args.time_limit:
            break

        rkg_start = perf_counter()
        hk, fk, pk, ak = stepper.step(hk, params.dt)
        rkg_elapsed += perf_counter() - rkg_start
        time += params.dt

        if not args.disable_progress:
            assert isinstance(iterator, tqdm)
            iterator.set_postfix(step=istep + 1, time=f"{time:.3f}", next_out=f"{time_out + EPS:.3f}")

        if time > time_out:
            sample_start = perf_counter()
            record_state(time, fk, pk, ak)
            sample_elapsed += perf_counter() - sample_start
            time_out += params.dt_out

    fkinzv_arr = np.stack(fkinzv_series, axis=0)
    phi_arr = np.stack(phi_series, axis=0)
    a_arr = np.stack(a_series, axis=0)
    dens_arr = np.stack(dens_series, axis=0)
    freq_lines, _ = _linfreq_rows(np.asarray(mominz_times), np.stack(pk_for_freq, axis=0))

    f_real, f_imag = complex_to_parts(fkinzv_arr)
    phi_real, phi_imag = complex_to_parts(phi_arr)
    a_real, a_imag = complex_to_parts(a_arr)
    dens_real, dens_imag = complex_to_parts(dens_arr)

    out_start = perf_counter()
    ds_fkinzv = xr.Dataset(
        data_vars={
            "f_real": (("time", "mu_index", "z", "vl", "species"), np.transpose(f_real, (0, 3, 1, 2, 4))),
            "f_imag": (("time", "mu_index", "z", "vl", "species"), np.transpose(f_imag, (0, 3, 1, 2, 4))),
        },
        coords={
            "time": np.asarray(fkinzv_times),
            "mu_index": mu_indices,
            "mu": ("mu_index", geom.mu[mu_indices]),
            "z": geom.zz,
            "vl": geom.vl,
            "species": np.arange(params.ns),
        },
        attrs={"source": "reference_lingk_sim.py", "dt": params.dt, "dt_out": params.dt_out, "seed": params.seed},
    )
    ds_fkinzv.to_netcdf(fkinzv_path)

    ds_mominz = xr.Dataset(
        data_vars={
            "phi_real": (("time", "z"), phi_real),
            "phi_imag": (("time", "z"), phi_imag),
            "A_real": (("time", "z"), a_real),
            "A_imag": (("time", "z"), a_imag),
            "dens_real": (("time", "z", "species"), dens_real),
            "dens_imag": (("time", "z", "species"), dens_imag),
        },
        coords={"time": np.asarray(mominz_times), "z": geom.zz, "species": np.arange(params.ns)},
        attrs={"source": "reference_lingk_sim.py", "dt": params.dt, "dt_out": params.dt_out},
    )
    ds_mominz.to_netcdf(mominz_path)

    frq_path.write_text("\n".join(freq_lines) + "\n")
    output_elapsed += perf_counter() - out_start

    total_elapsed = perf_counter() - total_start
    other_elapsed = total_elapsed - (init_elapsed + rkg_elapsed + sample_elapsed + output_elapsed)
    nsteps_done = int(np.floor(time / params.dt + 1.0e-12))

    print(f"[lingk-reference] wrote fkinzv dataset: {fkinzv_path}")
    print(f"[lingk-reference] wrote mominzt dataset: {mominz_path}")
    print(f"[lingk-reference] wrote frq table: {frq_path}")
    print(f"[lingk-reference] parameters: nz={params.nz}, nv={params.nv}, nm={params.nm}, dt={params.dt}, dt_out={params.dt_out}")
    print("")
    print(" ### Elapsed time ###")
    print(f" # Time steps = {nsteps_done:12d}")
    print(" #")
    print(f" #      Total = {total_elapsed:18.15f}")
    print(f" #       Init = {init_elapsed:18.15E}")
    print(f" #        RKG = {rkg_elapsed:18.15f}")
    print(f" #     Sample = {sample_elapsed:18.15f}")
    print(f" #     Output = {output_elapsed:18.15f}")
    print(f" #      Other = {other_elapsed:18.15E}")
    print(" End program.")


if __name__ == "__main__":
    main()
