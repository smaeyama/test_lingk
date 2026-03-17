#!/usr/bin/env python

"""Python reference solver for the linear gyrokinetic `lingk` code."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import h5py
import numpy as np
from scipy.special import i0, j0
from tqdm import tqdm


PI = np.pi
CI = 1j
EPS = 1.0e-10


@dataclass
class GKParameters:
    nz: int = 24 * 5
    nv: int = 32
    nm: int = 31
    ns: int = 1
    nzb: int = 2
    nvb: int = 2
    lz: float = 5.0 * np.pi
    lv: float = 4.0
    lm: float = 8.0
    dt: float = 0.01
    dt_out: float = 0.1
    time_limit: float = 10.0
    max_steps: int = 1_000_000
    kx: float = 0.0
    ky: float = 0.2
    eps_r: float = 0.18
    q_0: float = 1.4
    s_hat: float = 0.8
    lambda_: float = 0.0
    beta: float = 0.0
    seed: int = 0
    R0_Ln: np.ndarray = field(init=False)
    R0_Lt: np.ndarray = field(init=False)
    nu: np.ndarray = field(init=False)
    Anum: np.ndarray = field(init=False)
    Znum: np.ndarray = field(init=False)
    fcs: np.ndarray = field(init=False)
    sgn: np.ndarray = field(init=False)
    tau: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.R0_Ln = np.full(self.ns, 2.2, dtype=np.float64)
        self.R0_Lt = np.full(self.ns, 6.9, dtype=np.float64)
        self.nu = np.zeros(self.ns, dtype=np.float64)
        self.Anum = np.ones(self.ns, dtype=np.float64)
        self.Znum = np.ones(self.ns, dtype=np.float64)
        self.fcs = np.ones(self.ns, dtype=np.float64)
        self.sgn = np.ones(self.ns, dtype=np.float64)
        self.tau = np.ones(self.ns, dtype=np.float64)


@dataclass
class GKGeometry:
    params: GKParameters
    dz: float
    dv: float
    dm: float
    zz: np.ndarray
    vl: np.ndarray
    mu: np.ndarray
    omg: np.ndarray
    rootg: np.ndarray
    ksq: np.ndarray
    dpara: np.ndarray
    vp: np.ndarray
    mir: np.ndarray
    dvp: np.ndarray
    fmx: np.ndarray
    kvd: np.ndarray
    kvs: np.ndarray
    j0: np.ndarray
    g0: np.ndarray
    fct_poisson: np.ndarray
    fct_ampere: np.ndarray

    @property
    def nz_tot(self) -> int:
        return 2 * self.params.nz

    @property
    def nv_tot(self) -> int:
        return 2 * self.params.nv

    @property
    def nm_tot(self) -> int:
        return self.params.nm + 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Python reference implementation of lingk")
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


def build_parameters(args: argparse.Namespace) -> GKParameters:
    params = GKParameters(
        nz=args.nz,
        nv=args.nv,
        nm=args.nm,
        dt=args.dt,
        dt_out=args.dt_out,
        time_limit=args.time_limit,
        max_steps=args.max_steps,
        seed=args.seed,
    )
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


def build_geometry(params: GKParameters) -> GKGeometry:
    dz = params.lz / float(params.nz)
    dv = 2.0 * params.lv / float(2 * params.nv - 1)
    dm = np.sqrt(2.0 * params.lm) / float(params.nm)

    zz = dz * np.arange(-params.nz, params.nz, dtype=np.float64)
    omg = 1.0 - params.eps_r * np.cos(zz)
    rootg = params.q_0 / omg
    ksq = (params.kx + params.s_hat * zz * params.ky) ** 2 + params.ky**2
    dpara = dz * params.q_0 * np.ones_like(zz)

    vl = np.linspace(-params.lv, params.lv, 2 * params.nv, dtype=np.float64)
    mu = 0.5 * (dm * np.arange(params.nm + 1, dtype=np.float64)) ** 2

    vp = np.sqrt(2.0 * omg[:, None] * mu[None, :])
    mir = mu[None, :] * params.eps_r * np.sin(zz)[:, None] / params.q_0
    dvp = vp[:, 1].copy()

    fmx = np.exp(-0.5 * vl[None, :, None] ** 2 - mu[None, None, :] * omg[:, None, None]) / np.sqrt(2.0 * PI) ** 3

    kvd = np.empty((2 * params.nz, 2 * params.nv, params.nm + 1, params.ns), dtype=np.float64)
    kvs = np.empty_like(kvd)
    j0_arr = np.empty((2 * params.nz, params.nm + 1, params.ns), dtype=np.float64)
    g0 = np.empty((2 * params.nz, params.ns), dtype=np.float64)

    for ispec in range(params.ns):
        cs = params.sgn[ispec] * params.tau[ispec] / params.Znum[ispec]
        geom_factor = params.ky * np.cos(zz) + (params.kx + params.s_hat * zz * params.ky) * np.sin(zz)
        kvd[..., ispec] = -(vl[None, :, None] ** 2 + omg[:, None, None] * mu[None, None, :]) * geom_factor[:, None, None] * cs
        kvs[..., ispec] = -cs * params.ky * (
            params.R0_Ln[ispec]
            + params.R0_Lt[ispec] * (0.5 * vl[None, :, None] ** 2 + omg[:, None, None] * mu[None, None, :] - 1.5)
        )

        x_g0 = ksq * params.tau[ispec] * params.Anum[ispec] / (params.Znum[ispec] ** 2 * omg**2)
        g0[:, ispec] = np.exp(-x_g0) * i0(x_g0)

        x_j0 = np.sqrt(2.0 * ksq[:, None] * mu[None, :] / omg[:, None])
        x_j0 *= np.sqrt(params.tau[ispec] * params.Anum[ispec]) / params.Znum[ispec]
        j0_arr[:, :, ispec] = j0(x_j0)

    wr = params.lambda_ * ksq.copy()
    for ispec in range(params.ns):
        wr += params.Znum[ispec] * params.fcs[ispec] / params.tau[ispec] * (1.0 - g0[:, ispec])
    fct_poisson = 1.0 / wr

    if params.beta > 0.0:
        wr_amp = ksq.copy()
        for ispec in range(params.ns):
            integrand = (
                params.Znum[ispec]
                * params.fcs[ispec]
                / params.Anum[ispec]
                * vl[None, :, None] ** 2
                * j0_arr[:, None, :, ispec] ** 2
                * fmx
            )
            wr_amp += params.beta * vintegral_real(integrand, vp=vp, dv=dv, dvp=dvp)
        fct_ampere = 1.0 / wr_amp
    else:
        fct_ampere = np.zeros_like(ksq)

    return GKGeometry(
        params=params,
        dz=dz,
        dv=dv,
        dm=dm,
        zz=zz,
        vl=vl,
        mu=mu,
        omg=omg,
        rootg=rootg,
        ksq=ksq,
        dpara=dpara,
        vp=vp,
        mir=mir,
        dvp=dvp,
        fmx=fmx,
        kvd=kvd,
        kvs=kvs,
        j0=j0_arr,
        g0=g0,
        fct_poisson=fct_poisson,
        fct_ampere=fct_ampere,
    )


def vintegral_real(wf: np.ndarray, vp: np.ndarray, dv: float, dvp: np.ndarray) -> np.ndarray:
    wn = np.sum(wf[:, :, 1:-1] * vp[:, None, 1:-1], axis=(1, 2)) * (2.0 * PI * dv * dvp)
    wfvp = wf[:, :, 1] * vp[:, None, 1]
    wfvp1 = wf[:, :, 2] * vp[:, None, 2]
    corr = (-wfvp / 12.0 + (wfvp1 - 2.0 * wfvp) * 11.0 / 720.0) * (2.0 * PI * dv * dvp[:, None])
    return wn - np.sum(corr, axis=1)


def vintegral_species(wf: np.ndarray, geom: GKGeometry) -> np.ndarray:
    wn = np.sum(wf[:, :, 1:-1, :] * geom.vp[:, None, 1:-1, None], axis=(1, 2)) * (2.0 * PI * geom.dv * geom.dvp[:, None])
    wfvp = wf[:, :, 1, :] * geom.vp[:, None, 1, None]
    wfvp1 = wf[:, :, 2, :] * geom.vp[:, None, 2, None]
    corr = (-wfvp / 12.0 + (wfvp1 - 2.0 * wfvp) * 11.0 / 720.0) * (2.0 * PI * geom.dv * geom.dvp[:, None, None])
    return wn - np.sum(corr, axis=1)


def vintegral_z(wf: np.ndarray, geom: GKGeometry) -> np.ndarray:
    wn = np.sum(wf[:, :, 1:-1] * geom.vp[:, None, 1:-1], axis=(1, 2)) * (2.0 * PI * geom.dv * geom.dvp)
    wfvp = wf[:, :, 1] * geom.vp[:, None, 1]
    wfvp1 = wf[:, :, 2] * geom.vp[:, None, 2]
    corr = (-wfvp / 12.0 + (wfvp1 - 2.0 * wfvp) * 11.0 / 720.0) * (2.0 * PI * geom.dv * geom.dvp[:, None])
    return wn - np.sum(corr, axis=1)


def solve_electrostatic_field(fk: np.ndarray, geom: GKGeometry) -> np.ndarray:
    wf = fk * geom.j0[:, None, :, :] * geom.params.sgn[None, None, None, :] * geom.params.fcs[None, None, None, :]
    nk = np.sum(vintegral_species(wf, geom), axis=1)
    if geom.params.ns == 1:
        return nk / ((1.0 - geom.g0[:, 0]) / geom.params.tau[0] + 1.0)
    return nk * geom.fct_poisson


def solve_magnetic_field_from_f(fk: np.ndarray, geom: GKGeometry) -> np.ndarray:
    coeff = geom.params.sgn * geom.params.fcs * np.sqrt(geom.params.tau / geom.params.Anum)
    wf = fk * geom.j0[:, None, :, :] * geom.vl[None, :, None, None] * coeff[None, None, None, :]
    nk = np.sum(vintegral_species(wf, geom), axis=1)
    return nk * geom.params.beta / geom.ksq


def solve_magnetic_field_from_h(hk: np.ndarray, geom: GKGeometry) -> np.ndarray:
    coeff = geom.params.sgn * geom.params.fcs * np.sqrt(geom.params.tau / geom.params.Anum)
    wf = hk * geom.j0[:, None, :, :] * geom.vl[None, :, None, None] * coeff[None, None, None, :]
    nk = np.sum(vintegral_species(wf, geom), axis=1)
    return nk * geom.params.beta * geom.fct_ampere


def hh_to_ff(hk: np.ndarray, ak: np.ndarray, geom: GKGeometry) -> np.ndarray:
    coeff = geom.params.sgn * geom.params.Znum / np.sqrt(geom.params.Anum * geom.params.tau)
    correction = (
        coeff[None, None, None, :]
        * geom.fmx[:, :, :, None]
        * geom.vl[None, :, None, None]
        * geom.j0[:, None, :, :]
        * ak[:, None, None, None]
    )
    return hk - correction


def ff_to_hh(fk: np.ndarray, ak: np.ndarray, geom: GKGeometry) -> np.ndarray:
    coeff = geom.params.sgn * geom.params.Znum / np.sqrt(geom.params.Anum * geom.params.tau)
    correction = (
        coeff[None, None, None, :]
        * geom.fmx[:, :, :, None]
        * geom.vl[None, :, None, None]
        * geom.j0[:, None, :, :]
        * ak[:, None, None, None]
    )
    return fk + correction


def state_fields_from_h(hk: np.ndarray, geom: GKGeometry) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ak = solve_magnetic_field_from_h(hk, geom)
    fk = hh_to_ff(hk, ak, geom)
    pk = solve_electrostatic_field(fk, geom)
    return fk, pk, ak


def init_state(params: GKParameters, geom: GKGeometry) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    initval = 1.0e-3
    profile = initval * (1.0 + geom.zz[:, None, None] + geom.vl[None, :, None]) ** 2
    profile *= np.exp(-(geom.zz[:, None, None] ** 2) / (0.2 * PI) ** 2)
    fk = np.repeat((profile * geom.fmx)[..., None], params.ns, axis=3).astype(np.complex128)
    pk = solve_electrostatic_field(fk, geom)
    ak = solve_magnetic_field_from_f(fk, geom)
    hk = ff_to_hh(fk, ak, geom)
    return hk, fk, pk, ak


def extend_distribution(field: np.ndarray, geom: GKGeometry) -> np.ndarray:
    ff = np.zeros(
        (
            geom.nz_tot + 2 * geom.params.nzb,
            geom.nv_tot + 2 * geom.params.nvb,
            geom.nm_tot + 2 * geom.params.nvb,
            geom.params.ns,
        ),
        dtype=np.complex128,
    )

    z0 = geom.params.nzb
    v0 = geom.params.nvb
    m0 = geom.params.nvb
    ff[z0 : z0 + geom.nz_tot, v0 : v0 + geom.nv_tot, m0 : m0 + geom.nm_tot, :] = field

    for ispec in range(geom.params.ns):
        for im in range(geom.nm_tot):
            for iv in range(geom.nv_tot):
                if geom.vl[iv] > 0.0:
                    ff[z0 + geom.nz_tot, v0 + iv, m0 + im, ispec] = ff[z0 + geom.nz_tot - 1, v0 + iv, m0 + im, ispec]
                    ff[z0 + geom.nz_tot + 1, v0 + iv, m0 + im, ispec] = (
                        -ff[z0 + geom.nz_tot - 2, v0 + iv, m0 + im, ispec]
                        + 2.0 * ff[z0 + geom.nz_tot - 1, v0 + iv, m0 + im, ispec]
                    )
                else:
                    ff[z0 - 1, v0 + iv, m0 + im, ispec] = ff[z0, v0 + iv, m0 + im, ispec]
                    ff[z0 - 2, v0 + iv, m0 + im, ispec] = -ff[z0 + 1, v0 + iv, m0 + im, ispec] + 2.0 * ff[z0, v0 + iv, m0 + im, ispec]
    return ff


def extend_field_zero_z(field: np.ndarray, geom: GKGeometry) -> np.ndarray:
    ext = np.zeros((geom.nz_tot + 2 * geom.params.nzb, field.shape[1], field.shape[2]), dtype=np.complex128)
    z0 = geom.params.nzb
    ext[z0 : z0 + geom.nz_tot, :, :] = field
    return ext


def collision_term(ff_ext: np.ndarray, geom: GKGeometry) -> np.ndarray:
    dh = np.zeros((geom.nz_tot, geom.nv_tot, geom.nm_tot, geom.params.ns), dtype=np.complex128)
    z0 = geom.params.nzb
    v0 = geom.params.nvb
    m0 = geom.params.nvb

    for ispec in range(geom.params.ns):
        nu_s = (
            geom.params.nu[ispec]
            * np.sqrt(geom.params.tau[ispec] / geom.params.Anum[ispec])
            * geom.params.fcs[ispec]
            * geom.params.Znum[ispec] ** 3
            / geom.params.tau[ispec] ** 2
        )
        if np.isclose(nu_s, 0.0):
            continue

        cef1 = nu_s / (12.0 * geom.dv * geom.dv)
        cef2 = nu_s / (12.0 * geom.dv)
        cef3 = nu_s / (12.0 * geom.dvp * geom.dvp)
        cef4 = nu_s / (12.0 * geom.dvp)

        for im in range(geom.nm_tot):
            me = im + m0
            if im == 0:
                diff_mu = (
                    -ff_ext[z0 : z0 + geom.nz_tot, v0 : v0 + geom.nv_tot, me + 2, ispec]
                    + 16.0 * ff_ext[z0 : z0 + geom.nz_tot, v0 : v0 + geom.nv_tot, me + 1, ispec]
                    - 30.0 * ff_ext[z0 : z0 + geom.nz_tot, v0 : v0 + geom.nv_tot, me, ispec]
                    + 16.0 * ff_ext[z0 : z0 + geom.nz_tot, v0 : v0 + geom.nv_tot, me + 1, ispec]
                    - ff_ext[z0 : z0 + geom.nz_tot, v0 : v0 + geom.nv_tot, me + 2, ispec]
                ) * (2.0 * cef3[:, None])
                drift_mu = 0.0
            elif im == 1:
                diff_mu = (
                    -ff_ext[z0 : z0 + geom.nz_tot, v0 : v0 + geom.nv_tot, me + 2, ispec]
                    + 16.0 * ff_ext[z0 : z0 + geom.nz_tot, v0 : v0 + geom.nv_tot, me + 1, ispec]
                    - 30.0 * ff_ext[z0 : z0 + geom.nz_tot, v0 : v0 + geom.nv_tot, me, ispec]
                    + 16.0 * ff_ext[z0 : z0 + geom.nz_tot, v0 : v0 + geom.nv_tot, me - 1, ispec]
                    - ff_ext[z0 : z0 + geom.nz_tot, v0 : v0 + geom.nv_tot, me, ispec]
                ) * cef3[:, None]
                drift_mu = (
                    -ff_ext[z0 : z0 + geom.nz_tot, v0 : v0 + geom.nv_tot, me + 2, ispec]
                    + 8.0 * ff_ext[z0 : z0 + geom.nz_tot, v0 : v0 + geom.nv_tot, me + 1, ispec]
                    - 8.0 * ff_ext[z0 : z0 + geom.nz_tot, v0 : v0 + geom.nv_tot, me - 1, ispec]
                    + ff_ext[z0 : z0 + geom.nz_tot, v0 : v0 + geom.nv_tot, me, ispec]
                ) * (cef4[:, None] * (geom.vp[:, im, None] + 1.0 / geom.vp[:, im, None]))
            else:
                diff_mu = (
                    -ff_ext[z0 : z0 + geom.nz_tot, v0 : v0 + geom.nv_tot, me + 2, ispec]
                    + 16.0 * ff_ext[z0 : z0 + geom.nz_tot, v0 : v0 + geom.nv_tot, me + 1, ispec]
                    - 30.0 * ff_ext[z0 : z0 + geom.nz_tot, v0 : v0 + geom.nv_tot, me, ispec]
                    + 16.0 * ff_ext[z0 : z0 + geom.nz_tot, v0 : v0 + geom.nv_tot, me - 1, ispec]
                    - ff_ext[z0 : z0 + geom.nz_tot, v0 : v0 + geom.nv_tot, me - 2, ispec]
                ) * cef3[:, None]
                drift_mu = (
                    -ff_ext[z0 : z0 + geom.nz_tot, v0 : v0 + geom.nv_tot, me + 2, ispec]
                    + 8.0 * ff_ext[z0 : z0 + geom.nz_tot, v0 : v0 + geom.nv_tot, me + 1, ispec]
                    - 8.0 * ff_ext[z0 : z0 + geom.nz_tot, v0 : v0 + geom.nv_tot, me - 1, ispec]
                    + ff_ext[z0 : z0 + geom.nz_tot, v0 : v0 + geom.nv_tot, me - 2, ispec]
                ) * (cef4[:, None] * (geom.vp[:, im, None] + 1.0 / geom.vp[:, im, None]))

            diff_v = (
                -ff_ext[z0 : z0 + geom.nz_tot, v0 + 2 : v0 + 2 + geom.nv_tot, me, ispec]
                + 16.0 * ff_ext[z0 : z0 + geom.nz_tot, v0 + 1 : v0 + 1 + geom.nv_tot, me, ispec]
                - 30.0 * ff_ext[z0 : z0 + geom.nz_tot, v0 : v0 + geom.nv_tot, me, ispec]
                + 16.0 * ff_ext[z0 : z0 + geom.nz_tot, v0 - 1 : v0 - 1 + geom.nv_tot, me, ispec]
                - ff_ext[z0 : z0 + geom.nz_tot, v0 - 2 : v0 - 2 + geom.nv_tot, me, ispec]
            ) * cef1
            drift_v = (
                -ff_ext[z0 : z0 + geom.nz_tot, v0 + 2 : v0 + 2 + geom.nv_tot, me, ispec]
                + 8.0 * ff_ext[z0 : z0 + geom.nz_tot, v0 + 1 : v0 + 1 + geom.nv_tot, me, ispec]
                - 8.0 * ff_ext[z0 : z0 + geom.nz_tot, v0 - 1 : v0 - 1 + geom.nv_tot, me, ispec]
                + ff_ext[z0 : z0 + geom.nz_tot, v0 - 2 : v0 - 2 + geom.nv_tot, me, ispec]
            ) * (cef2 * geom.vl[None, :])

            dh[:, :, im, ispec] += diff_v + drift_v + diff_mu + drift_mu
            dh[:, :, im, ispec] += nu_s * 3.0 * ff_ext[z0 : z0 + geom.nz_tot, v0 : v0 + geom.nv_tot, me, ispec]
            dh[:, :, im, ispec] -= (
                nu_s
                * geom.ksq[:, None]
                * geom.params.Anum[ispec]
                * geom.params.tau[ispec]
                / (geom.params.Znum[ispec] * geom.omg[:, None]) ** 2
                * ff_ext[z0 : z0 + geom.nz_tot, v0 : v0 + geom.nv_tot, me, ispec]
            )
    return dh


def rhs_h(hk: np.ndarray, geom: GKGeometry, fk: np.ndarray, pk: np.ndarray, ak: np.ndarray) -> np.ndarray:
    ff_ext = extend_distribution(fk, geom)
    z0 = geom.params.nzb
    v0 = geom.params.nvb
    m0 = geom.params.nvb

    dh = np.zeros_like(hk)
    psi = geom.j0[:, :, :] * pk[:, None, None]
    chi = geom.j0[:, :, :] * ak[:, None, None]
    psi_ext = extend_field_zero_z(psi, geom)

    for ispec in range(geom.params.ns):
        cs1 = geom.params.sgn[ispec] * geom.params.Znum[ispec] / geom.params.tau[ispec]
        cs2 = np.sqrt(geom.params.tau[ispec] / geom.params.Anum[ispec])

        dh_ispec = (
            -CI * geom.kvd[..., ispec] * fk[..., ispec]
            - cs1
            * geom.fmx
            * (
                CI * geom.kvd[..., ispec] * psi[:, :, ispec][:, None, :]
                - CI * geom.kvs[..., ispec] * (psi[:, :, ispec][:, None, :] - cs2 * geom.vl[None, :, None] * chi[:, :, ispec][:, None, :])
            )
        )

        dfdz = (
            -ff_ext[z0 + 2 : z0 + 2 + geom.nz_tot, v0 : v0 + geom.nv_tot, m0 : m0 + geom.nm_tot, ispec]
            + 8.0 * ff_ext[z0 + 1 : z0 + 1 + geom.nz_tot, v0 : v0 + geom.nv_tot, m0 : m0 + geom.nm_tot, ispec]
            - 8.0 * ff_ext[z0 - 1 : z0 - 1 + geom.nz_tot, v0 : v0 + geom.nv_tot, m0 : m0 + geom.nm_tot, ispec]
            + ff_ext[z0 - 2 : z0 - 2 + geom.nz_tot, v0 : v0 + geom.nv_tot, m0 : m0 + geom.nm_tot, ispec]
        ) / (12.0 * geom.dpara[:, None, None])
        dfdv = (
            -ff_ext[z0 : z0 + geom.nz_tot, v0 + 2 : v0 + 2 + geom.nv_tot, m0 : m0 + geom.nm_tot, ispec]
            + 8.0 * ff_ext[z0 : z0 + geom.nz_tot, v0 + 1 : v0 + 1 + geom.nv_tot, m0 : m0 + geom.nm_tot, ispec]
            - 8.0 * ff_ext[z0 : z0 + geom.nz_tot, v0 - 1 : v0 - 1 + geom.nv_tot, m0 : m0 + geom.nm_tot, ispec]
            + ff_ext[z0 : z0 + geom.nz_tot, v0 - 2 : v0 - 2 + geom.nv_tot, m0 : m0 + geom.nm_tot, ispec]
        ) / (12.0 * geom.dv)
        dpsidz = (
            -psi_ext[z0 + 2 : z0 + 2 + geom.nz_tot, :, ispec]
            + 8.0 * psi_ext[z0 + 1 : z0 + 1 + geom.nz_tot, :, ispec]
            - 8.0 * psi_ext[z0 - 1 : z0 - 1 + geom.nz_tot, :, ispec]
            + psi_ext[z0 - 2 : z0 - 2 + geom.nz_tot, :, ispec]
        ) / (12.0 * geom.dpara[:, None])

        dh_ispec -= geom.vl[None, :, None] * cs2 * dfdz
        dh_ispec += geom.mir[:, None, :] * cs2 * dfdv
        dh_ispec -= cs1 * geom.fmx * geom.vl[None, :, None] * cs2 * dpsidz[:, None, :]
        dh[:, :, :, ispec] = dh_ispec

    dh += collision_term(ff_ext, geom)
    return dh


class RungeKuttaGillStepper:
    def __init__(self, geom: GKGeometry):
        self.geom = geom
        self.q = np.zeros((geom.nz_tot, geom.nv_tot, geom.nm_tot, geom.params.ns), dtype=np.complex128)

    def _update_active_mu(self, hk: np.ndarray, k: np.ndarray, factor: float, q_factor: float) -> np.ndarray:
        r = factor * (k[:, :, :-1, :] - q_factor * self.q[:, :, :-1, :])
        hk[:, :, :-1, :] += r
        return r

    def step(self, hk: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        hk_work = hk.copy()

        fk, pk, ak = state_fields_from_h(hk_work, self.geom)
        dh = rhs_h(hk_work, self.geom, fk=fk, pk=pk, ak=ak)
        k = dt * dh
        r = self._update_active_mu(hk_work, k, 0.5, 2.0)
        self.q[:, :, :-1, :] += 3.0 * r - 0.5 * k[:, :, :-1, :]

        fk, pk, ak = state_fields_from_h(hk_work, self.geom)
        dh = rhs_h(hk_work, self.geom, fk=fk, pk=pk, ak=ak)
        k = dt * dh
        factor = 1.0 - np.sqrt(0.5)
        r = self._update_active_mu(hk_work, k, factor, 1.0)
        self.q[:, :, :-1, :] += 3.0 * r - factor * k[:, :, :-1, :]

        fk, pk, ak = state_fields_from_h(hk_work, self.geom)
        dh = rhs_h(hk_work, self.geom, fk=fk, pk=pk, ak=ak)
        k = dt * dh
        factor = 1.0 + np.sqrt(0.5)
        r = self._update_active_mu(hk_work, k, factor, 1.0)
        self.q[:, :, :-1, :] += 3.0 * r - factor * k[:, :, :-1, :]

        fk, pk, ak = state_fields_from_h(hk_work, self.geom)
        dh = rhs_h(hk_work, self.geom, fk=fk, pk=pk, ak=ak)
        k = dt * dh
        r = self._update_active_mu(hk_work, k, 1.0 / 6.0, 2.0)
        self.q[:, :, :-1, :] += 3.0 * r - 0.5 * k[:, :, :-1, :]

        fk, pk, ak = state_fields_from_h(hk_work, self.geom)
        return hk_work, fk, pk, ak


def compute_time_step_control(geom: GKGeometry, courant_num: float = 0.5) -> dict[str, float]:
    kvd_max = float(np.max(geom.kvd))
    dt_perp = courant_num * PI / kvd_max

    vl_max = 0.0
    for ispec in range(geom.params.ns):
        cs = float(np.sqrt(geom.params.tau[ispec] / geom.params.Anum[ispec]))
        vl_max = max(vl_max, float(np.max(cs * geom.params.lv / geom.dpara)))
    dt_zz = courant_num / vl_max

    mir_max = 0.0
    for ispec in range(geom.params.ns):
        cs = float(np.sqrt(geom.params.tau[ispec] / geom.params.Anum[ispec]))
        mir_max = max(mir_max, float(np.max(cs * geom.mir)))
    dt_vl = np.inf if np.isclose(mir_max, 0.0) else courant_num * geom.dv / mir_max

    nu_max = 0.0
    for ispec in range(geom.params.ns):
        nu_temp = (
            geom.params.nu[ispec]
            * np.sqrt(geom.params.tau[ispec] / geom.params.Anum[ispec])
            * geom.params.fcs[ispec]
            * geom.params.Znum[ispec] ** 3
            / geom.params.tau[ispec] ** 2
            * (2.0 / geom.dv**2)
        )
        nu_max = max(nu_max, float(nu_temp))
        nu_temp_dvp = (
            geom.params.nu[ispec]
            * np.sqrt(geom.params.tau[ispec] / geom.params.Anum[ispec])
            * geom.params.fcs[ispec]
            * geom.params.Znum[ispec] ** 3
            / geom.params.tau[ispec] ** 2
            * (2.0 / geom.dvp**2)
        )
        nu_max = max(nu_max, float(np.max(nu_temp_dvp)))

    dt_col = 99999.9999 if np.isclose(nu_max, 0.0) else courant_num / nu_max
    dt = min(dt_perp, dt_zz, dt_vl, dt_col)
    return {
        "courant_num": float(courant_num),
        "dt_perp": float(dt_perp),
        "dt_zz": float(dt_zz),
        "dt_vl": float(dt_vl),
        "dt_col": float(dt_col),
        "dt": float(dt),
    }


def output_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fkinzv = Path(args.fkinzv_out) if args.fkinzv_out else out_dir / "fkinzv.nc"
    mominz = Path(args.mominz_out) if args.mominz_out else out_dir / "mominzt.nc"
    frq = Path(args.frq_out) if args.frq_out else out_dir / "frq.txt"
    return fkinzv, mominz, frq


def mu_indices(args: argparse.Namespace, params: GKParameters) -> np.ndarray:
    if args.im:
        indices = np.array(sorted(set(args.im)), dtype=int)
    else:
        indices = np.array([params.nm // 4], dtype=int)
    if np.any(indices < 0) or np.any(indices > params.nm):
        raise ValueError(f"mu indices must satisfy 0 <= im <= {params.nm}")
    return indices


def linfreq_rows(times: np.ndarray, pk_series: np.ndarray, dt_out: float, kx: float, ky: float) -> tuple[list[str], bool]:
    header = "#            time           growth        frequency       diff(grow)       diff(freq)         1-Ineq."
    lines = [header]
    stop_signal = False
    time0 = None
    pk0 = None
    omega0 = 0.0 + 0.0j
    pk0_norm2 = None

    for time, pk in zip(times, pk_series):
        if time < dt_out:
            time0 = time
            pk0 = pk.copy()
            omega0 = 0.0 + 0.0j
            pk0_norm2 = float(np.sum(np.abs(pk[1:-1]) ** 2))
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
            lines.append("#              kx               ky       Growthrate        Frequency       Diff(grow)       Diff(freq)         1 - Ineq")
            lines.append(
                f"# {kx:17.7e}{ky:17.7e}{np.imag(omega):17.7e}{np.real(omega):17.7e}{diff_imag:17.7e}{diff_real:17.7e}{(1.0 - ineq):17.7e}"
            )
            stop_signal = True
            break

        time0 = time
        pk0 = pk.copy()
        omega0 = omega
        pk0_norm2 = pk_norm2

    return lines, stop_signal


def write_fkinzv(path: Path, times: np.ndarray, mu_idx: np.ndarray, geom: GKGeometry, params: GKParameters, series: np.ndarray) -> None:
    f_real = np.real(series)
    f_imag = np.imag(series)
    with h5py.File(path, "w") as f:
        f.create_dataset("time", data=times)
        f.create_dataset("mu_index", data=mu_idx)
        f.create_dataset("mu", data=geom.mu[mu_idx])
        f.create_dataset("z", data=geom.zz)
        f.create_dataset("vl", data=geom.vl)
        f.create_dataset("species", data=np.arange(params.ns))
        f.create_dataset("f_real", data=np.transpose(f_real, (0, 3, 1, 2, 4)))
        f.create_dataset("f_imag", data=np.transpose(f_imag, (0, 3, 1, 2, 4)))


def write_mominz(path: Path, times: np.ndarray, geom: GKGeometry, params: GKParameters, phi: np.ndarray, a: np.ndarray, dens: np.ndarray) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset("time", data=times)
        f.create_dataset("z", data=geom.zz)
        f.create_dataset("species", data=np.arange(params.ns))
        f.create_dataset("phi_real", data=np.real(phi))
        f.create_dataset("phi_imag", data=np.imag(phi))
        f.create_dataset("A_real", data=np.real(a))
        f.create_dataset("A_imag", data=np.imag(a))
        f.create_dataset("dens_real", data=np.real(dens))
        f.create_dataset("dens_imag", data=np.imag(dens))


def main() -> None:
    total_start = perf_counter()
    init_elapsed = 0.0
    rkg_elapsed = 0.0
    sample_elapsed = 0.0
    output_elapsed = 0.0

    init_start = perf_counter()
    args = parse_args()
    params = build_parameters(args)
    geom = build_geometry(params)
    dt_control = compute_time_step_control(geom)
    if not args.disable_dtc:
        params.dt = dt_control["dt"]
    selected_mu = mu_indices(args, params)
    fkinzv_path, mominz_path, frq_path = output_paths(args)

    print(" # Time step size control")
    print("")
    print(f" # courant num. = {dt_control['courant_num']:20.15f}")
    print(f" # dt_perp      = {dt_control['dt_perp']:23.15E}")
    print(f" # dt_zz        = {dt_control['dt_zz']:23.15E}")
    print(f" # dt_vl        = {dt_control['dt_vl']:23.15E}")
    print(f" # dt_col       = {dt_control['dt_col']:23.15E}")
    print(f" # dt           = {params.dt:23.15E}")
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
        density = np.stack([vintegral_z(current_fk[:, :, :, ispec], geom) for ispec in range(params.ns)], axis=-1)
        fkinzv_times.append(current_time)
        fkinzv_series.append(current_fk[:, :, selected_mu, :])
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
        if time > params.time_limit:
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
    freq_lines, _ = linfreq_rows(np.asarray(mominz_times), np.stack(pk_for_freq, axis=0), params.dt_out, params.kx, params.ky)

    out_start = perf_counter()
    write_fkinzv(fkinzv_path, np.asarray(fkinzv_times), selected_mu, geom, params, fkinzv_arr)
    write_mominz(mominz_path, np.asarray(mominz_times), geom, params, phi_arr, a_arr, dens_arr)
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
