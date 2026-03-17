#!/usr/bin/env python

"""Compare Fortran and Python linear gyrokinetic outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


FRQ_COLUMNS = (
    "time",
    "growth",
    "frequency",
    "diff(grow)",
    "diff(freq)",
    "1-Ineq",
)

MOMINZT_COLUMNS = (
    "z",
    "time",
    "phi_real",
    "phi_imag",
    "A_real",
    "A_imag",
    "dens_real",
    "dens_imag",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Fortran and Python lingk outputs")
    parser.add_argument("--python-frq", default="lingk_output/frq.txt")
    parser.add_argument("--fortran-frq", default="../data/frq.001")
    parser.add_argument("--python-mominzt", default="lingk_output/mominzt.nc")
    parser.add_argument("--fortran-mominzt", default="../data/mominzt.001")
    parser.add_argument("--python-fkinzv", default="lingk_output/fkinzv.nc")
    parser.add_argument("--fortran-fkinzv-dir", default="../data")
    parser.add_argument("--max-abs", type=float, default=1.0e-5)
    parser.add_argument("--max-rel-rms", type=float, default=1.0e-6)
    parser.add_argument("--assert-match", action="store_true")
    return parser.parse_args()


def load_frq(path: str | Path) -> np.ndarray:
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] != len(FRQ_COLUMNS):
        raise ValueError(f"{path} has {data.shape[1]} columns, expected {len(FRQ_COLUMNS)}")
    return data


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values**2)))


def relative_rms(error: np.ndarray, reference: np.ndarray) -> float:
    denom = rms(reference)
    if np.isclose(denom, 0.0):
        return float("nan")
    return rms(error) / denom


def print_metric_table(title: str, metrics: list[tuple[str, float, float, float]], context: list[str]) -> None:
    print(title)
    for line in context:
        print(line)
    print("")
    print(f"{'column':>12} {'max':>14} {'rms':>14} {'relative_rms':>14}")
    print("-" * 60)
    for name, max_err, rms_err, rel_rms_err in metrics:
        rel_rms_str = f"{rel_rms_err:.7e}" if np.isfinite(rel_rms_err) else "nan"
        print(f"{name:>12} {max_err:14.7e} {rms_err:14.7e} {rel_rms_str:>14}")


def compare_arrays(names: tuple[str, ...], py_data: np.ndarray, ft_data: np.ndarray) -> list[tuple[str, float, float, float]]:
    metrics: list[tuple[str, float, float, float]] = []
    for icol, name in enumerate(names):
        err_col = py_data[..., icol] - ft_data[..., icol]
        ref_col = ft_data[..., icol]
        metrics.append((name, float(np.max(np.abs(err_col))), rms(err_col), relative_rms(err_col, ref_col)))
    return metrics


def compare_frq(py_path: Path, ft_path: Path) -> list[tuple[str, float, float, float]]:
    py_data = load_frq(py_path)
    ft_data = load_frq(ft_path)
    nrows = min(len(py_data), len(ft_data))
    py_data = py_data[:nrows]
    ft_data = ft_data[:nrows]
    metrics = compare_arrays(FRQ_COLUMNS, py_data, ft_data)
    context = [
        f"python frq : {py_path}",
        f"fortran frq: {ft_path}",
        f"rows compared: {nrows}",
    ]
    if len(py_data) != len(ft_data):
        context.append(f"[warn] row count differs: python={len(py_data)}, fortran={len(ft_data)}")
    print_metric_table("FRQ Comparison", metrics, context)
    return metrics


def load_hdf5_dataset(path: Path, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        for name in names:
            out[name] = f[name][()]
    return out


def load_fortran_mominzt(path: Path) -> dict[str, np.ndarray]:
    blocks: list[np.ndarray] = []
    current: list[list[float]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            if current:
                blocks.append(np.asarray(current, dtype=float))
                current = []
            continue
        if line.lstrip().startswith("#"):
            continue
        current.append([float(x.replace("D", "E")) for x in line.split()])
    if current:
        blocks.append(np.asarray(current, dtype=float))

    arr = np.stack(blocks, axis=0)
    return {name: arr[..., i] for i, name in enumerate(MOMINZT_COLUMNS)}


def compare_mominzt(py_path: Path, ft_path: Path) -> list[tuple[str, float, float, float]]:
    py = load_hdf5_dataset(py_path, ("time", "z", "phi_real", "phi_imag", "A_real", "A_imag", "dens_real", "dens_imag"))
    ft = load_fortran_mominzt(ft_path)

    nt = min(py["time"].shape[0], ft["time"].shape[0])
    nz = min(py["z"].shape[0], ft["z"].shape[1])

    py_table = np.stack(
        (
            np.broadcast_to(py["z"][None, :nz], (nt, nz)),
            np.broadcast_to(py["time"][:nt, None], (nt, nz)),
            py["phi_real"][:nt, :nz],
            py["phi_imag"][:nt, :nz],
            py["A_real"][:nt, :nz],
            py["A_imag"][:nt, :nz],
            py["dens_real"][:nt, :nz, 0],
            py["dens_imag"][:nt, :nz, 0],
        ),
        axis=-1,
    )
    ft_table = np.stack(
        (
            ft["z"][:nt, :nz],
            ft["time"][:nt, :nz],
            ft["phi_real"][:nt, :nz],
            ft["phi_imag"][:nt, :nz],
            ft["A_real"][:nt, :nz],
            ft["A_imag"][:nt, :nz],
            ft["dens_real"][:nt, :nz],
            ft["dens_imag"][:nt, :nz],
        ),
        axis=-1,
    )
    metrics = compare_arrays(MOMINZT_COLUMNS, py_table, ft_table)
    print_metric_table(
        "MOMINZT Comparison",
        metrics,
        [
            f"python mominzt : {py_path}",
            f"fortran mominzt: {ft_path}",
            f"time slices compared: {nt}",
            f"z points compared: {nz}",
        ],
    )
    return metrics


def load_fortran_fkinzv(path: Path, nz: int, nv: int) -> dict[str, np.ndarray]:
    arr = np.fromfile(path, dtype=np.float64).reshape(nv * nz, 4)
    arr = arr.reshape(nv, nz, 4)
    return {
        "z": arr[:, :, 0].T,
        "vl": arr[:, :, 1].T,
        "f_real": arr[:, :, 2].T,
        "f_imag": arr[:, :, 3].T,
    }


def compare_fkinzv(py_path: Path, ft_dir: Path) -> list[tuple[str, float, float, float]]:
    py = load_hdf5_dataset(py_path, ("time", "z", "vl", "mu_index", "f_real", "f_imag"))
    nz = py["z"].shape[0]
    nv = py["vl"].shape[0]

    ft_files = sorted(ft_dir.glob("fkinzv_im*.dat"))
    im = int(py["mu_index"][0])
    ft_files = [f for f in ft_files if f.name.startswith(f"fkinzv_im{im:04d}_")]
    nt = min(py["time"].shape[0], len(ft_files))

    ft_real = np.zeros((nt, nz, nv))
    ft_imag = np.zeros((nt, nz, nv))
    ft_z = np.zeros((nt, nz, nv))
    ft_vl = np.zeros((nt, nz, nv))
    for it, path in enumerate(ft_files[:nt]):
        rec = load_fortran_fkinzv(path, nz=nz, nv=nv)
        ft_z[it] = rec["z"]
        ft_vl[it] = rec["vl"]
        ft_real[it] = rec["f_real"]
        ft_imag[it] = rec["f_imag"]

    py_real = py["f_real"][:nt, 0, :, :, 0]
    py_imag = py["f_imag"][:nt, 0, :, :, 0]
    py_z = np.broadcast_to(py["z"][None, :, None], (nt, nz, nv))
    py_vl = np.broadcast_to(py["vl"][None, None, :], (nt, nz, nv))

    py_table = np.stack((py_z, py_vl, py_real, py_imag), axis=-1)
    ft_table = np.stack((ft_z, ft_vl, ft_real, ft_imag), axis=-1)
    metrics = compare_arrays(("z", "vl", "f_real", "f_imag"), py_table, ft_table)
    print_metric_table(
        "FKINZV Comparison",
        metrics,
        [
            f"python fkinzv : {py_path}",
            f"fortran fkinzv dir: {ft_dir}",
            f"mu index compared: {im}",
            f"time slices compared: {nt}",
            f"grid compared: nz={nz}, nv={nv}",
        ],
    )
    return metrics


def assert_metrics(metrics: list[tuple[str, float, float, float]], max_abs: float, max_rel_rms: float) -> None:
    failures: list[str] = []
    for name, max_err, _rms_err, rel_rms_err in metrics:
        if max_err > max_abs:
            failures.append(f"{name}: max_abs={max_err:.7e} exceeds {max_abs:.7e}")
        if np.isfinite(rel_rms_err) and rel_rms_err > max_rel_rms:
            failures.append(f"{name}: relative_rms={rel_rms_err:.7e} exceeds {max_rel_rms:.7e}")
    if failures:
        raise SystemExit("Comparison thresholds failed:\n" + "\n".join(failures))


def main() -> None:
    args = parse_args()
    metrics: list[tuple[str, float, float, float]] = []
    metrics.extend(compare_frq(Path(args.python_frq), Path(args.fortran_frq)))
    print("")
    metrics.extend(compare_mominzt(Path(args.python_mominzt), Path(args.fortran_mominzt)))
    print("")
    metrics.extend(compare_fkinzv(Path(args.python_fkinzv), Path(args.fortran_fkinzv_dir)))
    if args.assert_match:
        assert_metrics(metrics, max_abs=args.max_abs, max_rel_rms=args.max_rel_rms)


if __name__ == "__main__":
    main()
