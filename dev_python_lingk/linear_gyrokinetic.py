#!/usr/bin/env python
# coding: utf-8

"""Shared utilities for the linear gyrokinetic reference and DLRA solvers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import i0, j0
try:
    from jax import config as jax_config
    jax_config.update("jax_enable_x64", True)
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except Exception:
    jax = None
    jnp = np
    JAX_AVAILABLE = False


PI = np.pi
CI = 1j


def _backend_array(array):
    return jnp.asarray(array) if JAX_AVAILABLE else np.asarray(array)


def _is_jax_array(array) -> bool:
    return JAX_AVAILABLE and isinstance(array, jax.Array)


def _array_set(arr, index, value):
    if _is_jax_array(arr):
        return arr.at[index].set(value)
    arr[index] = value
    return arr


def _array_add(arr, index, value):
    if _is_jax_array(arr):
        return arr.at[index].add(value)
    arr[index] += value
    return arr


def _weighted_qr_left_backend(matrix: np.ndarray, weight: float) -> tuple[np.ndarray, np.ndarray]:
    q, r = jnp.linalg.qr(matrix, mode="reduced") if JAX_AVAILABLE else np.linalg.qr(matrix, mode="reduced")
    sqrt_weight = np.sqrt(weight)
    return q / sqrt_weight, r * sqrt_weight


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
        self.R0_Ln = np.full(self.ns, 2.2, dtype=float)
        self.R0_Lt = np.full(self.ns, 6.9, dtype=float)
        self.nu = np.zeros(self.ns, dtype=float)
        self.Anum = np.ones(self.ns, dtype=float)
        self.Znum = np.ones(self.ns, dtype=float)
        self.fcs = np.ones(self.ns, dtype=float)
        self.sgn = np.ones(self.ns, dtype=float)
        self.tau = np.ones(self.ns, dtype=float)


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
    pos_cols: np.ndarray
    neg_cols: np.ndarray

    @property
    def nz_tot(self) -> int:
        return 2 * self.params.nz

    @property
    def nv_tot(self) -> int:
        return 2 * self.params.nv

    @property
    def nm_tot(self) -> int:
        return self.params.nm + 1

    @property
    def nvm(self) -> int:
        return self.nv_tot * self.nm_tot * self.params.ns

    @property
    def vm_weight(self) -> float:
        return self.dv * self.dm


def build_geometry(params: GKParameters) -> GKGeometry:
    nz = params.nz
    nv = params.nv
    nm = params.nm
    ns = params.ns

    dz = params.lz / float(nz)
    dv = 2.0 * params.lv / float(2 * nv - 1)
    dm = np.sqrt(2.0 * params.lm) / float(nm)

    zz = dz * np.arange(-nz, nz, dtype=float)
    omg = 1.0 - params.eps_r * np.cos(zz)
    rootg = params.q_0 / omg
    ksq = (params.kx + params.s_hat * zz * params.ky) ** 2 + params.ky**2
    dpara = dz * params.q_0 * np.ones_like(zz)

    vl = np.linspace(-params.lv, params.lv, 2 * nv)
    mu = 0.5 * (dm * np.arange(nm + 1, dtype=float)) ** 2

    vp = np.sqrt(2.0 * omg[:, None] * mu[None, :])
    mir = mu[None, :] * params.eps_r * np.sin(zz)[:, None] / params.q_0
    dvp = vp[:, 1].copy()

    fmx = np.exp(-0.5 * vl[None, :, None] ** 2 - mu[None, None, :] * omg[:, None, None]) / np.sqrt(2.0 * PI) ** 3

    kvd = np.empty((2 * nz, 2 * nv, nm + 1, ns), dtype=float)
    kvs = np.empty((2 * nz, 2 * nv, nm + 1, ns), dtype=float)
    j0_arr = np.empty((2 * nz, nm + 1, ns), dtype=float)
    g0 = np.empty((2 * nz, ns), dtype=float)

    for ispec in range(ns):
        cs = params.sgn[ispec] * params.tau[ispec] / params.Znum[ispec]
        geom_factor = params.ky * np.cos(zz) + (params.kx + params.s_hat * zz * params.ky) * np.sin(zz)
        kvd[..., ispec] = -(
            vl[None, :, None] ** 2 + omg[:, None, None] * mu[None, None, :]
        ) * geom_factor[:, None, None] * cs
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
    for ispec in range(ns):
        wr += params.Znum[ispec] * params.fcs[ispec] / params.tau[ispec] * (1.0 - g0[:, ispec])
    fct_poisson = 1.0 / wr

    if params.beta > 0.0:
        wr_amp = ksq.copy()
        for ispec in range(ns):
            integrand = (
                params.Znum[ispec]
                * params.fcs[ispec]
                / params.Anum[ispec]
                * vl[None, :, None] ** 2
                * j0_arr[:, None, :, ispec] ** 2
                * fmx
            )
            wr_amp += params.beta * _vintegral_core(integrand, vp=vp, dv=dv, dvp=dvp)
        fct_ampere = 1.0 / wr_amp
    else:
        fct_ampere = np.zeros_like(ksq)

    return GKGeometry(
        params=params,
        dz=dz,
        dv=dv,
        dm=dm,
        zz=_backend_array(zz),
        vl=_backend_array(vl),
        mu=_backend_array(mu),
        omg=_backend_array(omg),
        rootg=_backend_array(rootg),
        ksq=_backend_array(ksq),
        dpara=_backend_array(dpara),
        vp=_backend_array(vp),
        mir=_backend_array(mir),
        dvp=_backend_array(dvp),
        fmx=_backend_array(fmx),
        kvd=_backend_array(kvd),
        kvs=_backend_array(kvs),
        j0=_backend_array(j0_arr),
        g0=_backend_array(g0),
        fct_poisson=_backend_array(fct_poisson),
        fct_ampere=_backend_array(fct_ampere),
        pos_cols=np.flatnonzero(vl > 0.0) + params.nvb,
        neg_cols=np.flatnonzero(vl <= 0.0) + params.nvb,
    )


def flatten_vm(field: np.ndarray) -> np.ndarray:
    return field.reshape(field.shape[0], -1)


def unflatten_vm(matrix: np.ndarray, geom: GKGeometry) -> np.ndarray:
    return matrix.reshape(geom.nz_tot, geom.nv_tot, geom.nm_tot, geom.params.ns)


def weighted_gram(matrix: np.ndarray, weight: float) -> np.ndarray:
    return np.asarray(matrix).conj().T @ np.asarray(matrix) * weight


def complex_to_parts(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(array)
    return np.real(arr), np.imag(arr)


def parts_to_complex(real_part: np.ndarray, imag_part: np.ndarray) -> np.ndarray:
    return np.asarray(real_part) + 1j * np.asarray(imag_part)


def init_state(params: GKParameters, geom: GKGeometry) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    initval = 1.0e-3
    profile = initval * (1.0 + geom.zz[:, None, None] + geom.vl[None, :, None]) ** 2
    profile *= jnp.exp(-(geom.zz[:, None, None] ** 2) / (0.2 * PI) ** 2)
    fk = jnp.repeat((profile * geom.fmx)[..., None], params.ns, axis=3).astype(jnp.complex128)
    pk = solve_electrostatic_field(fk, geom)
    ak = solve_magnetic_field_from_f(fk, geom)
    hk = ff_to_hh(fk, ak, geom)
    return hk, fk, pk, ak


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


def _vintegral_core(wf: np.ndarray, vp: np.ndarray, dv: float, dvp: np.ndarray) -> np.ndarray:
    wn = jnp.zeros(wf.shape[0], dtype=wf.dtype)
    weighted = wf[:, :, 1:-1] * vp[:, None, 1:-1]
    wn = wn + jnp.sum(weighted, axis=(1, 2)) * (2.0 * PI * dv * dvp)

    wfvp = wf[:, :, 1] * vp[:, None, 1]
    wfvp1 = wf[:, :, 2] * vp[:, None, 2]
    corr = (-wfvp / 12.0 + (wfvp1 - 2.0 * wfvp) * 11.0 / 720.0) * (2.0 * PI * dv * dvp[:, None])
    wn = wn - jnp.sum(corr, axis=1)
    return wn


def vintegral_z(wf: np.ndarray, geom: GKGeometry) -> np.ndarray:
    return _vintegral_core(wf, vp=geom.vp, dv=geom.dv, dvp=geom.dvp)


def _vintegral_species(wf: np.ndarray, geom: GKGeometry) -> np.ndarray:
    weighted = wf[:, :, 1:-1, :] * geom.vp[:, None, 1:-1, None]
    wn = jnp.sum(weighted, axis=(1, 2)) * (2.0 * PI * geom.dv * geom.dvp[:, None])

    wfvp = wf[:, :, 1, :] * geom.vp[:, None, 1, None]
    wfvp1 = wf[:, :, 2, :] * geom.vp[:, None, 2, None]
    corr = (-wfvp / 12.0 + (wfvp1 - 2.0 * wfvp) * 11.0 / 720.0) * (2.0 * PI * geom.dv * geom.dvp[:, None, None])
    wn = wn - jnp.sum(corr, axis=1)
    return wn


def solve_electrostatic_field(fk: np.ndarray, geom: GKGeometry) -> np.ndarray:
    wf = fk * geom.j0[:, None, :, :] * geom.params.sgn[None, None, None, :] * geom.params.fcs[None, None, None, :]
    nk = jnp.sum(_vintegral_species(wf, geom), axis=1)

    if geom.params.ns == 1:
        return nk / ((1.0 - geom.g0[:, 0]) / geom.params.tau[0] + 1.0)
    return nk * geom.fct_poisson


def solve_magnetic_field_from_f(fk: np.ndarray, geom: GKGeometry) -> np.ndarray:
    coeff = geom.params.sgn * geom.params.fcs * np.sqrt(geom.params.tau / geom.params.Anum)
    wf = fk * geom.j0[:, None, :, :] * geom.vl[None, :, None, None] * coeff[None, None, None, :]
    nk = np.sum(_vintegral_species(wf, geom), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return jnp.where(geom.ksq != 0.0, nk * geom.params.beta / geom.ksq, 0.0)


def solve_magnetic_field_from_h(hk: np.ndarray, geom: GKGeometry) -> np.ndarray:
    coeff = geom.params.sgn * geom.params.fcs * np.sqrt(geom.params.tau / geom.params.Anum)
    wf = hk * geom.j0[:, None, :, :] * geom.vl[None, :, None, None] * coeff[None, None, None, :]
    nk = jnp.sum(_vintegral_species(wf, geom), axis=1)
    return nk * geom.params.beta * geom.fct_ampere


def state_fields_from_h(hk: np.ndarray, geom: GKGeometry) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ak = solve_magnetic_field_from_h(hk, geom)
    fk = hh_to_ff(hk, ak, geom)
    pk = solve_electrostatic_field(fk, geom)
    return fk, pk, ak


def compute_density_moment(fk: np.ndarray, geom: GKGeometry) -> np.ndarray:
    return _vintegral_species(fk, geom)


def _mu_integration_weights(geom: GKGeometry) -> np.ndarray:
    weights = jnp.zeros((geom.nz_tot, geom.nm_tot), dtype=geom.vp.dtype)
    weights = _array_set(weights, (slice(None), slice(1, -1)), geom.vp[:, 1:-1])
    if geom.nm_tot > 1:
        weights = _array_add(weights, (slice(None), 1), (82.0 / 720.0) * geom.vp[:, 1])
    if geom.nm_tot > 2:
        weights = _array_add(weights, (slice(None), 2), (-11.0 / 720.0) * geom.vp[:, 2])
    weights *= 2.0 * PI * geom.dv * geom.dvp[:, None]
    return weights


def _low_rank_basis_scalar(
    right_factor: np.ndarray,
    kernel: np.ndarray,
    geom: GKGeometry,
) -> np.ndarray:
    v_tensor = right_factor.reshape(geom.nv_tot, geom.nm_tot, geom.params.ns, -1)
    return jnp.einsum("zvms,vmsr,zm->zr", kernel, v_tensor, _mu_integration_weights(geom))


def _low_rank_basis_species(
    right_factor: np.ndarray,
    kernel: np.ndarray,
    geom: GKGeometry,
) -> np.ndarray:
    v_tensor = right_factor.reshape(geom.nv_tot, geom.nm_tot, geom.params.ns, -1)
    return jnp.einsum("zvms,vmsr,zm->zsr", kernel, v_tensor, _mu_integration_weights(geom))


def _field_kernel_cache(geom: GKGeometry) -> dict[str, np.ndarray]:
    cache = getattr(geom, "_field_kernel_cache", None)
    if cache is not None:
        return cache

    coeff_amp = geom.params.sgn * geom.params.fcs * np.sqrt(geom.params.tau / geom.params.Anum)
    coeff_h2f = geom.params.sgn * geom.params.Znum / np.sqrt(geom.params.Anum * geom.params.tau)
    cache = {
        "kernel_amp": geom.j0[:, None, :, :] * geom.vl[None, :, None, None] * coeff_amp[None, None, None, :],
        "kernel_phi_h": geom.j0[:, None, :, :] * geom.params.sgn[None, None, None, :] * geom.params.fcs[None, None, None, :],
        "phi_corr_kernel": (
            geom.fmx[:, :, :, None]
            * geom.vl[None, :, None, None]
            * geom.j0[:, None, :, :] ** 2
            * (geom.params.sgn * geom.params.fcs * coeff_h2f)[None, None, None, :]
        ),
        "dens_corr_kernel": geom.fmx[:, :, :, None] * geom.vl[None, :, None, None] * geom.j0[:, None, :, :] * coeff_h2f[None, None, None, :],
    }
    setattr(geom, "_field_kernel_cache", cache)
    return cache


def _low_rank_operator_cache(geom: GKGeometry) -> dict[str, np.ndarray]:
    cache = getattr(geom, "_low_rank_operator_cache", None)
    if cache is not None:
        return cache

    ns = geom.params.ns
    nv = geom.nv_tot
    nm = geom.nm_tot

    cs2 = np.sqrt(geom.params.tau / geom.params.Anum)
    cs_drift = geom.params.sgn * geom.params.tau / geom.params.Znum

    species_idx = np.arange(ns, dtype=float)[None, None, :]
    vl_grid = np.broadcast_to(np.asarray(geom.vl)[:, None, None], (nv, nm, ns))
    mu_grid = np.broadcast_to(np.asarray(geom.mu)[None, :, None], (nv, nm, ns))
    cs2_grid = np.broadcast_to(cs2[None, None, :], (nv, nm, ns))
    cs_drift_grid = np.broadcast_to(cs_drift[None, None, :], (nv, nm, ns))

    stream_vm = (-vl_grid * cs2_grid).reshape(-1)
    mirror_vm = (mu_grid * cs2_grid).reshape(-1)
    kvd_v2_vm = (cs_drift_grid * vl_grid**2).reshape(-1)
    kvd_mu_vm = (cs_drift_grid * mu_grid).reshape(-1)

    pos_mask = np.broadcast_to((np.asarray(geom.vl) > 0.0)[:, None, None], (nv, nm, ns)).reshape(-1)
    neg_mask = ~pos_mask

    cache = {
        "stream_vm": _backend_array(stream_vm),
        "mirror_vm": _backend_array(mirror_vm),
        "kvd_v2_vm": _backend_array(kvd_v2_vm),
        "kvd_mu_vm": _backend_array(kvd_mu_vm),
        "pos_mask": _backend_array(pos_mask.astype(float)),
        "neg_mask": _backend_array(neg_mask.astype(float)),
        "geom_factor": _backend_array(-(geom.params.ky * np.cos(np.asarray(geom.zz)) + (geom.params.kx + geom.params.s_hat * np.asarray(geom.zz) * geom.params.ky) * np.sin(np.asarray(geom.zz)))),
        "geom_factor_omg": _backend_array(
            -(
                geom.params.ky * np.cos(np.asarray(geom.zz))
                + (geom.params.kx + geom.params.s_hat * np.asarray(geom.zz) * geom.params.ky) * np.sin(np.asarray(geom.zz))
            )
            * np.asarray(geom.omg)
        ),
        "mir_z": _backend_array(geom.params.eps_r * np.sin(np.asarray(geom.zz)) / geom.params.q_0),
    }
    setattr(geom, "_low_rank_operator_cache", cache)
    return cache


def solve_fields_from_h_factors(left_factor: np.ndarray, right_factor: np.ndarray, geom: GKGeometry) -> tuple[np.ndarray, np.ndarray]:
    kernels = _field_kernel_cache(geom)
    kernel_amp = kernels["kernel_amp"]
    amp_basis = _low_rank_basis_scalar(right_factor, kernel_amp, geom)
    amp_src = jnp.sum(left_factor * amp_basis, axis=1)
    ak = amp_src * geom.params.beta * geom.fct_ampere

    kernel_phi_h = kernels["kernel_phi_h"]
    phi_basis_h = _low_rank_basis_scalar(right_factor, kernel_phi_h, geom)
    phi_src_h = jnp.sum(left_factor * phi_basis_h, axis=1)

    kernel_phi_corr = kernels["phi_corr_kernel"]
    phi_corr = jnp.sum(kernel_phi_corr * _mu_integration_weights(geom)[:, None, :, None], axis=(1, 2, 3))
    phi_src = phi_src_h - ak * phi_corr

    if geom.params.ns == 1:
        pk = phi_src / ((1.0 - geom.g0[:, 0]) / geom.params.tau[0] + 1.0)
    else:
        pk = phi_src * geom.fct_poisson
    return pk, ak


def compute_density_moment_from_h_factors(left_factor: np.ndarray, right_factor: np.ndarray, ak: np.ndarray, geom: GKGeometry) -> np.ndarray:
    kernel_density_h = jnp.ones((geom.nz_tot, geom.nv_tot, geom.nm_tot, geom.params.ns), dtype=right_factor.dtype)
    dens_basis_h = _low_rank_basis_species(right_factor, kernel_density_h, geom)
    dens_h = jnp.einsum("zr,zsr->zs", left_factor, dens_basis_h)

    kernel_density_corr = _field_kernel_cache(geom)["dens_corr_kernel"]
    dens_corr = jnp.sum(kernel_density_corr * _mu_integration_weights(geom)[:, None, :, None], axis=(1, 2))
    return dens_h - ak[:, None] * dens_corr


def _five_point_first_derivative(arr: np.ndarray, spacing: np.ndarray, axis: int) -> np.ndarray:
    if axis == 0:
        return (
            -jnp.roll(arr, -2, axis=0)
            + 8.0 * jnp.roll(arr, -1, axis=0)
            - 8.0 * jnp.roll(arr, 1, axis=0)
            + jnp.roll(arr, 2, axis=0)
        ) / (12.0 * spacing[:, None, None])
    raise ValueError("Only axis=0 is supported")


def _derivative_v_right_factor(right_factor: np.ndarray, geom: GKGeometry) -> np.ndarray:
    vt = right_factor.reshape(geom.nv_tot, geom.nm_tot, geom.params.ns, -1)
    ext = jnp.pad(vt, ((geom.params.nvb, geom.params.nvb), (0, 0), (0, 0), (0, 0)), mode="constant")
    dvt = (
        -ext[geom.params.nvb + 2 : geom.params.nvb + 2 + geom.nv_tot]
        + 8.0 * ext[geom.params.nvb + 1 : geom.params.nvb + 1 + geom.nv_tot]
        - 8.0 * ext[geom.params.nvb - 1 : geom.params.nvb - 1 + geom.nv_tot]
        + ext[geom.params.nvb - 2 : geom.params.nvb - 2 + geom.nv_tot]
    ) / (12.0 * geom.dv)
    return dvt.reshape(geom.nvm, -1)


def _derivative_z_split_matrices(left_factor: np.ndarray, right_factor: np.ndarray, geom: GKGeometry) -> np.ndarray:
    ops = _low_rank_operator_cache(geom)
    right_pos = right_factor * ops["pos_mask"][:, None]
    right_neg = right_factor * ops["neg_mask"][:, None]

    zeros = jnp.zeros((geom.params.nzb, left_factor.shape[1]), dtype=left_factor.dtype)
    upper1 = left_factor[-1:, :]
    upper2 = -left_factor[-2:-1, :] + 2.0 * left_factor[-1:, :]
    lower1 = left_factor[:1, :]
    lower2 = -left_factor[1:2, :] + 2.0 * left_factor[:1, :]

    ext_pos = jnp.concatenate((zeros, left_factor, upper1, upper2), axis=0)
    ext_neg = jnp.concatenate((lower2, lower1, left_factor, zeros), axis=0)

    dleft_pos = (
        -ext_pos[geom.params.nzb + 2 : geom.params.nzb + 2 + geom.nz_tot]
        + 8.0 * ext_pos[geom.params.nzb + 1 : geom.params.nzb + 1 + geom.nz_tot]
        - 8.0 * ext_pos[geom.params.nzb - 1 : geom.params.nzb - 1 + geom.nz_tot]
        + ext_pos[geom.params.nzb - 2 : geom.params.nzb - 2 + geom.nz_tot]
    ) / (12.0 * geom.dpara[:, None])
    dleft_neg = (
        -ext_neg[geom.params.nzb + 2 : geom.params.nzb + 2 + geom.nz_tot]
        + 8.0 * ext_neg[geom.params.nzb + 1 : geom.params.nzb + 1 + geom.nz_tot]
        - 8.0 * ext_neg[geom.params.nzb - 1 : geom.params.nzb - 1 + geom.nz_tot]
        + ext_neg[geom.params.nzb - 2 : geom.params.nzb - 2 + geom.nz_tot]
    ) / (12.0 * geom.dpara[:, None])
    return dleft_pos @ right_pos.T + dleft_neg @ right_neg.T


def _collisionless_beta0_fast_path_enabled(geom: GKGeometry) -> bool:
    return np.isclose(geom.params.beta, 0.0) and np.allclose(geom.params.nu, 0.0)


def rhs_h_collisionless_beta0_factors(left_factor: np.ndarray, right_factor: np.ndarray, geom: GKGeometry) -> np.ndarray:
    ops = _low_rank_operator_cache(geom)
    pk, _ = solve_fields_from_h_factors(left_factor, right_factor, geom)

    rhs_matrix = jnp.zeros((geom.nz_tot, geom.nvm), dtype=left_factor.dtype)

    rhs_matrix = rhs_matrix + ((-CI * ops["geom_factor"])[:, None] * left_factor) @ (right_factor * ops["kvd_v2_vm"][:, None]).T
    rhs_matrix = rhs_matrix + ((-CI * ops["geom_factor_omg"])[:, None] * left_factor) @ (right_factor * ops["kvd_mu_vm"][:, None]).T

    dfdz_matrix = _derivative_z_split_matrices(left_factor, right_factor, geom)
    rhs_matrix = rhs_matrix + dfdz_matrix * ops["stream_vm"][None, :]

    dv_right = _derivative_v_right_factor(right_factor, geom)
    rhs_matrix = rhs_matrix + (ops["mir_z"][:, None] * left_factor) @ (dv_right * ops["mirror_vm"][:, None]).T

    psi = geom.j0 * pk[:, None, None]
    dpsidz = _five_point_first_derivative(psi, geom.dpara, axis=0)

    source = jnp.zeros((geom.nz_tot, geom.nv_tot, geom.nm_tot, geom.params.ns), dtype=left_factor.dtype)
    for ispec in range(geom.params.ns):
        cs1 = geom.params.sgn[ispec] * geom.params.Znum[ispec] / geom.params.tau[ispec]
        cs2 = np.sqrt(geom.params.tau[ispec] / geom.params.Anum[ispec])
        source = _array_set(
            source,
            (slice(None), slice(None), slice(None), ispec),
            -cs1
            * geom.fmx
            * (
                CI * (geom.kvd[..., ispec] - geom.kvs[..., ispec]) * psi[..., ispec][:, None, :]
                + geom.vl[None, :, None] * cs2 * dpsidz[..., ispec][:, None, :]
            ),
        )
    return rhs_matrix + flatten_vm(source)


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


def _extend_state(field: np.ndarray, geom: GKGeometry) -> np.ndarray:
    base = jnp.pad(
        field,
        ((0, 0), (geom.params.nvb, geom.params.nvb), (geom.params.nvb, geom.params.nvb), (0, 0)),
        mode="constant",
    )

    vsize = geom.nv_tot + 2 * geom.params.nvb
    pos_mask = np.zeros(vsize, dtype=bool)
    neg_mask = np.zeros(vsize, dtype=bool)
    pos_mask[geom.pos_cols] = True
    neg_mask[geom.neg_cols] = True
    pos_mask_arr = _backend_array(pos_mask[None, :, None, None])
    neg_mask_arr = _backend_array(neg_mask[None, :, None, None])

    upper_1 = jnp.where(pos_mask_arr, base[-1:, :, :, :], 0.0)
    upper_2 = jnp.where(pos_mask_arr, -base[-2:-1, :, :, :] + 2.0 * base[-1:, :, :, :], 0.0)
    lower_1 = jnp.where(neg_mask_arr, base[:1, :, :, :], 0.0)
    lower_2 = jnp.where(neg_mask_arr, -base[1:2, :, :, :] + 2.0 * base[:1, :, :, :], 0.0)
    return jnp.concatenate((lower_2, lower_1, base, upper_1, upper_2), axis=0)


def collision_term(ff_ext: np.ndarray, geom: GKGeometry) -> np.ndarray:
    dh = jnp.zeros((geom.nz_tot, geom.nv_tot, geom.nm_tot, geom.params.ns), dtype=jnp.complex128)
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

            dh = _array_add(dh, (slice(None), slice(None), im, ispec), diff_v + drift_v + diff_mu + drift_mu)
            dh = _array_add(
                dh,
                (slice(None), slice(None), im, ispec),
                nu_s * 3.0 * ff_ext[z0 : z0 + geom.nz_tot, v0 : v0 + geom.nv_tot, me, ispec],
            )
            dh = _array_add(
                dh,
                (slice(None), slice(None), im, ispec),
                -(
                nu_s
                * geom.ksq[:, None]
                * geom.params.Anum[ispec]
                * geom.params.tau[ispec]
                / (geom.params.Znum[ispec] * geom.omg[:, None]) ** 2
                * ff_ext[z0 : z0 + geom.nz_tot, v0 : v0 + geom.nv_tot, me, ispec]
                ),
            )
    return dh


def rhs_h(
    hk: np.ndarray,
    geom: GKGeometry,
    fk: np.ndarray | None = None,
    pk: np.ndarray | None = None,
    ak: np.ndarray | None = None,
) -> np.ndarray:
    if fk is None or pk is None or ak is None:
        fk, pk, ak = state_fields_from_h(hk, geom)
    ff_ext = _extend_state(fk, geom)

    z0 = geom.params.nzb
    v0 = geom.params.nvb
    m0 = geom.params.nvb

    dh = jnp.zeros_like(hk)
    psi = geom.j0[:, None, :, :] * pk[:, None, None, None]
    chi = geom.j0[:, None, :, :] * ak[:, None, None, None]

    for ispec in range(geom.params.ns):
        cs1 = geom.params.sgn[ispec] * geom.params.Znum[ispec] / geom.params.tau[ispec]
        cs2 = np.sqrt(geom.params.tau[ispec] / geom.params.Anum[ispec])

        dh_ispec = (
            -CI * geom.kvd[..., ispec] * fk[..., ispec]
            - cs1
            * geom.fmx
            * (
                CI * geom.kvd[..., ispec] * psi[..., ispec]
                - CI * geom.kvs[..., ispec] * (psi[..., ispec] - cs2 * geom.vl[None, :, None] * chi[..., ispec])
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
            -jnp.roll(psi[..., ispec], -2, axis=0)
            + 8.0 * jnp.roll(psi[..., ispec], -1, axis=0)
            - 8.0 * jnp.roll(psi[..., ispec], 1, axis=0)
            + jnp.roll(psi[..., ispec], 2, axis=0)
        ) / (12.0 * geom.dpara[:, None, None])

        dh_ispec = dh_ispec - geom.vl[None, :, None] * np.sqrt(geom.params.tau[ispec] / geom.params.Anum[ispec]) * dfdz
        dh_ispec = dh_ispec + geom.mir[:, None, :] * np.sqrt(geom.params.tau[ispec] / geom.params.Anum[ispec]) * dfdv
        dh_ispec = dh_ispec - (
            cs1
            * geom.fmx
            * geom.vl[None, :, None]
            * np.sqrt(geom.params.tau[ispec] / geom.params.Anum[ispec])
            * dpsidz
        )
        dh = _array_set(dh, (slice(None), slice(None), slice(None), ispec), dh_ispec)

    dh += collision_term(ff_ext, geom)
    return dh


def rk4_step_full(hk: np.ndarray, dt: float, geom: GKGeometry) -> np.ndarray:
    k1 = rhs_h(hk, geom)
    k2 = rhs_h(hk + 0.5 * dt * k1, geom)
    k3 = rhs_h(hk + 0.5 * dt * k2, geom)
    k4 = rhs_h(hk + dt * k3, geom)
    return hk + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


class RungeKuttaGillStepper:
    def __init__(self, geom: GKGeometry):
        self.geom = geom
        self.q = jnp.zeros((geom.nz_tot, geom.nv_tot, geom.nm_tot, geom.params.ns), dtype=jnp.complex128)
        if JAX_AVAILABLE:
            self._compiled_step = jax.jit(lambda hk, q, dt: _rkg_step_impl(hk, q, dt, self.geom))
        else:
            self._compiled_step = None

    def step(self, hk: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._compiled_step is not None:
            hk_work, q, fk, pk, ak = self._compiled_step(hk, self.q, dt)
            self.q = q
            return hk_work, fk, pk, ak

        hk_work = hk.copy()
        q = self.q

        fk, pk, ak = state_fields_from_h(hk_work, self.geom)
        dh = rhs_h(hk_work, self.geom, fk=fk, pk=pk, ak=ak)
        k = dt * dh
        r = 0.5 * (k - 2.0 * q)
        hk_work = hk_work + r
        q = q + 3.0 * r - 0.5 * k

        fk, pk, ak = state_fields_from_h(hk_work, self.geom)
        dh = rhs_h(hk_work, self.geom, fk=fk, pk=pk, ak=ak)
        k = dt * dh
        r = (1.0 - np.sqrt(0.5)) * (k - q)
        hk_work = hk_work + r
        q = q + 3.0 * r - (1.0 - np.sqrt(0.5)) * k

        fk, pk, ak = state_fields_from_h(hk_work, self.geom)
        dh = rhs_h(hk_work, self.geom, fk=fk, pk=pk, ak=ak)
        k = dt * dh
        r = (1.0 + np.sqrt(0.5)) * (k - q)
        hk_work = hk_work + r
        q = q + 3.0 * r - (1.0 + np.sqrt(0.5)) * k

        fk, pk, ak = state_fields_from_h(hk_work, self.geom)
        dh = rhs_h(hk_work, self.geom, fk=fk, pk=pk, ak=ak)
        k = dt * dh
        r = (k - 2.0 * q) / 6.0
        hk_work = hk_work + r
        q = q + 3.0 * r - 0.5 * k

        self.q = q
        fk, pk, ak = state_fields_from_h(hk_work, self.geom)
        return hk_work, fk, pk, ak


def _rkg_step_impl(hk: np.ndarray, q: np.ndarray, dt: float, geom: GKGeometry):
    hk_work = hk

    fk, pk, ak = state_fields_from_h(hk_work, geom)
    dh = rhs_h(hk_work, geom, fk=fk, pk=pk, ak=ak)
    k = dt * dh
    r = 0.5 * (k - 2.0 * q)
    hk_work = hk_work + r
    q = q + 3.0 * r - 0.5 * k

    fk, pk, ak = state_fields_from_h(hk_work, geom)
    dh = rhs_h(hk_work, geom, fk=fk, pk=pk, ak=ak)
    k = dt * dh
    r = (1.0 - jnp.sqrt(0.5)) * (k - q)
    hk_work = hk_work + r
    q = q + 3.0 * r - (1.0 - jnp.sqrt(0.5)) * k

    fk, pk, ak = state_fields_from_h(hk_work, geom)
    dh = rhs_h(hk_work, geom, fk=fk, pk=pk, ak=ak)
    k = dt * dh
    r = (1.0 + jnp.sqrt(0.5)) * (k - q)
    hk_work = hk_work + r
    q = q + 3.0 * r - (1.0 + jnp.sqrt(0.5)) * k

    fk, pk, ak = state_fields_from_h(hk_work, geom)
    dh = rhs_h(hk_work, geom, fk=fk, pk=pk, ak=ak)
    k = dt * dh
    r = (k - 2.0 * q) / 6.0
    hk_work = hk_work + r
    q = q + 3.0 * r - 0.5 * k

    fk, pk, ak = state_fields_from_h(hk_work, geom)
    return hk_work, q, fk, pk, ak


def _weighted_qr_left(matrix: np.ndarray, weight: float) -> tuple[np.ndarray, np.ndarray]:
    q, r = np.linalg.qr(matrix, mode="reduced")
    return q / np.sqrt(weight), r * np.sqrt(weight)


def _rk4_integrate(state: np.ndarray, dt: float, rhs_fn) -> np.ndarray:
    k1 = rhs_fn(state)
    k2 = rhs_fn(state + 0.5 * dt * k1)
    k3 = rhs_fn(state + 0.5 * dt * k2)
    k4 = rhs_fn(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _projector_splitting_step_impl(x_basis: np.ndarray, s_coeff: np.ndarray, v_basis: np.ndarray, dt: float, geom: GKGeometry):
    dvm = geom.vm_weight
    dz = geom.dz

    def full_rhs_from_factors(left_factor: np.ndarray, right_factor: np.ndarray) -> np.ndarray:
        if _collisionless_beta0_fast_path_enabled(geom):
            return rhs_h_collisionless_beta0_factors(left_factor, right_factor, geom)
        hk = unflatten_vm(left_factor @ right_factor.T, geom)
        pk, ak = solve_fields_from_h_factors(left_factor, right_factor, geom)
        fk = hh_to_ff(hk, ak, geom)
        return flatten_vm(rhs_h(hk, geom, fk=fk, pk=pk, ak=ak))

    k_mat0 = x_basis @ s_coeff

    def rhs_k(k_mat: np.ndarray) -> np.ndarray:
        rhs_full = full_rhs_from_factors(k_mat, v_basis)
        return rhs_full @ v_basis * dvm

    k_mat = _rk4_integrate(k_mat0, dt, rhs_k)
    x_new, s_hat = _weighted_qr_left_backend(k_mat, dz)

    def rhs_s(s_mat: np.ndarray) -> np.ndarray:
        rhs_full = full_rhs_from_factors(x_new @ s_mat, v_basis)
        return -(jnp.conjugate(x_new).T @ rhs_full @ v_basis) * dz * dvm

    s_tilde = _rk4_integrate(s_hat, dt, rhs_s)
    l_mat0 = v_basis @ s_tilde.T

    def rhs_l(l_mat: np.ndarray) -> np.ndarray:
        rhs_full = full_rhs_from_factors(x_new, l_mat)
        return rhs_full.T @ jnp.conjugate(x_new) * dz

    l_mat = _rk4_integrate(l_mat0, dt, rhs_l)
    v_new, s_final_t = _weighted_qr_left_backend(l_mat, dvm)
    return x_new, s_final_t.T, v_new


