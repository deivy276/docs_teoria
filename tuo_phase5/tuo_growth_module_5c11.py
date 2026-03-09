
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import math
import numpy as np
import pandas as pd

# Hybrid As -> sigma8 normalization constants
A_S_REF = 2.1e-9
SIGMA8_REF = 0.811  # Planck-like LCDM reference at z=0
N_S_REF = 0.965
H_REF = 0.674
OMEGA_B_REF = 0.02237
OMEGA_C_REF = 0.1200
C_LIGHT = 299792.458  # km/s
T_CMB = 2.7255
K_PIVOT_MPC = 0.05   # primordial pivot in 1/Mpc


@dataclass
class GrowthDataset:
    name: str
    z: np.ndarray
    fs8: np.ndarray
    sigma: np.ndarray
    cov: Optional[np.ndarray] = None
    k_eff: Optional[np.ndarray] = None

    def __post_init__(self):
        self.z = np.asarray(self.z, dtype=float)
        self.fs8 = np.asarray(self.fs8, dtype=float)
        self.sigma = np.asarray(self.sigma, dtype=float)
        if self.k_eff is None:
            self.k_eff = np.full_like(self.z, 0.10, dtype=float)
        else:
            self.k_eff = np.asarray(self.k_eff, dtype=float)
            if self.k_eff.shape != self.z.shape:
                raise ValueError('k_eff must have same shape as z')
        if self.cov is not None:
            self.cov = np.asarray(self.cov, dtype=float)
            if self.cov.shape != (self.z.size, self.z.size):
                raise ValueError('covariance shape mismatch')
            self.icov = np.linalg.inv(self.cov)
        else:
            self.icov = None


def builtin_rsd_diag_minimal(k_pivot_hmpc: float = 0.10) -> GrowthDataset:
    z = np.array([0.15, 0.38, 0.51, 0.70, 0.85, 1.48], dtype=float)
    fs8 = np.array([0.53, 0.50, 0.46, 0.44, 0.39, 0.30], dtype=float)
    sigma = np.array([0.08, 0.05, 0.05, 0.06, 0.09, 0.08], dtype=float)
    return GrowthDataset('RSD_diag_minimal_builtin', z, fs8, sigma, k_eff=np.full_like(z, float(k_pivot_hmpc)))


def builtin_boss_eboss_diag(k_pivot_hmpc: float = 0.10) -> GrowthDataset:
    z = np.array([0.38, 0.51, 0.706, 0.85, 1.48], dtype=float)
    fs8 = np.array([0.497, 0.458, 0.473, 0.315, 0.462], dtype=float)
    sigma = np.array([0.045, 0.038, 0.041, 0.113, 0.045], dtype=float)
    return GrowthDataset('BOSS_eBOSS_diag_builtin', z, fs8, sigma, k_eff=np.full_like(z, float(k_pivot_hmpc)))


def builtin_eboss_dr16_compressed(k_pivot_hmpc: float = 0.10) -> GrowthDataset:
    z = np.array([0.38, 0.51, 0.706, 0.85, 1.48], dtype=float)
    fs8 = np.array([0.497, 0.458, 0.473, 0.315, 0.462], dtype=float)
    sigma = np.array([0.045, 0.038, 0.041, 0.113, 0.045], dtype=float)
    cov = np.array([
        [0.002025, 0.000810, 0.0,      0.0,      0.0     ],
        [0.000810, 0.001444, 0.0,      0.0,      0.0     ],
        [0.0,      0.0,      0.001681, 0.0,      0.0     ],
        [0.0,      0.0,      0.0,      0.012769, 0.0     ],
        [0.0,      0.0,      0.0,      0.0,      0.002025],
    ], dtype=float)
    return GrowthDataset('eBOSS_DR16_compressed_builtin', z, fs8, sigma, cov=cov, k_eff=np.full_like(z, float(k_pivot_hmpc)))


def _load_covariance(path: Path, n: int) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        arr = np.loadtxt(path, delimiter=',')
        if arr.ndim == 0:
            arr = np.array([arr], dtype=float)
    except Exception:
        arr = np.loadtxt(path)
        if arr.ndim == 0:
            arr = np.array([arr], dtype=float)
    if arr.ndim == 1:
        if arr.size == n * n:
            return arr.reshape((n, n))
        if arr.size == 1 + n * n:
            return arr[1:].reshape((n, n))
        raise ValueError(f'Cannot reshape covariance vector of length {arr.size} into ({n},{n})')
    if arr.shape == (n, n):
        return arr
    if arr.size == n * n:
        return arr.reshape((n, n))
    raise ValueError(f'Unexpected covariance shape {arr.shape}, expected ({n},{n})')


def load_growth_csv(path: Path, cov_path: Optional[Path] = None, k_pivot_hmpc: float = 0.10) -> GrowthDataset:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        df = pd.read_csv(path)
        if len(df.columns) == 1:
            raise ValueError('fallback to whitespace/comment')
    except Exception:
        df = pd.read_csv(path, sep=r'\s+', comment='#', header=None)
        names = ['z', 'fs8', 'sigma', 'k_eff'][:df.shape[1]]
        df = df.iloc[:, :len(names)]
        df.columns = names

    cols = {c.lower(): c for c in df.columns}
    if 'z' not in cols or 'fs8' not in cols:
        raise ValueError('growth file must contain z and fs8 columns')
    if cov_path is not None:
        cov = _load_covariance(Path(cov_path), len(df))
        sigma = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    else:
        if 'sigma' not in cols:
            raise ValueError('without growth-cov, the growth file must contain sigma column')
        sigma = df[cols['sigma']].to_numpy(dtype=float)
        cov = None
    if 'k_eff' in cols:
        k_eff = df[cols['k_eff']].to_numpy(dtype=float)
    else:
        k_eff = np.full(len(df), float(k_pivot_hmpc), dtype=float)
    return GrowthDataset(str(path), df[cols['z']].to_numpy(dtype=float), df[cols['fs8']].to_numpy(dtype=float), sigma, cov=cov, k_eff=k_eff)


def build_lcdm_reference_background(*, h: float, Omega_b0: float, Omega_c0: float, Omega_r0: float,
                                    z_max: float, npts: int) -> Dict[str, np.ndarray]:
    N = np.linspace(-np.log1p(float(z_max)), 0.0, int(npts))
    z = np.exp(-N) - 1.0
    om = float(Omega_b0) + float(Omega_c0)
    orad = float(Omega_r0)
    ov = 1.0 - om - orad
    E = np.sqrt(om * (1.0 + z) ** 3 + orad * (1.0 + z) ** 4 + ov)
    e2 = E * E
    b = float(Omega_b0) * np.exp(-3.0 * N)
    c = float(Omega_c0) * np.exp(-3.0 * N)
    r = float(Omega_r0) * np.exp(-4.0 * N)
    q = -1.0 - np.gradient(np.log(E), N)
    return {
        'N': N, 'z': z, 'E': E, 'E2': e2, 'q': q,
        'x': np.zeros_like(N), 'y': np.zeros_like(N),
        'Omega_b': b / e2, 'Omega_c': c / e2, 'Omega_m': (b + c) / e2,
        'Omega_r': r / e2, 'Omega_phi': ov / e2,
        'b': b, 'c': c, 'r': r,
    }


def _interp(Nq: float, N: np.ndarray, arr: np.ndarray) -> float:
    return float(np.interp(Nq, N, arr))


def solve_linear_growth_reference(bg: Dict[str, np.ndarray], *, Omega_b0: float, Omega_c0: float,
                                  z_start: float = 100.0, delta_norm: float = 1.0) -> Dict[str, np.ndarray]:
    N_all = np.asarray(bg['N'], dtype=float)
    z_all = np.asarray(bg['z'], dtype=float)
    Om_b_all = np.asarray(bg['Omega_b'], dtype=float)
    Om_c_all = np.asarray(bg['Omega_c'], dtype=float)
    q_all = np.asarray(bg['q'], dtype=float)

    idx0 = int(np.argmin(np.abs(z_all - float(z_start))))
    N = N_all[idx0:].copy()
    z = z_all[idx0:].copy()
    Om_b = Om_b_all[idx0:].copy()
    Om_c = Om_c_all[idx0:].copy()
    q = q_all[idx0:].copy()
    dlnE = -1.0 - q

    d0 = float(delta_norm) * math.exp(float(N[0]))
    S = np.zeros((4, N.size), dtype=float)
    S[:, 0] = np.array([d0, d0, d0, d0], dtype=float)

    def rhs(i: int, state):
        db, ub, dc, uc = state
        ob = Om_b[i]; oc = Om_c[i]; dl = dlnE[i]
        src = 1.5 * (ob * db + oc * dc)
        return ub, -(2.0 + dl) * ub + src, uc, -(2.0 + dl) * uc + src

    for i in range(1, N.size):
        hN = float(N[i] - N[i - 1])
        s0 = S[:, i - 1]
        im = i - 1
        k1 = np.array(rhs(im, s0))
        k2 = np.array(rhs(im, s0 + 0.5 * hN * k1))
        k3 = np.array(rhs(im, s0 + 0.5 * hN * k2))
        k4 = np.array(rhs(i, s0 + hN * k3))
        S[:, i] = s0 + (hN / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    db, ub, dc, uc = S
    wsum = max(Omega_b0 + Omega_c0, 1e-30)
    D_raw = (Omega_b0 * db + Omega_c0 * dc) / wsum
    Dp_raw = (Omega_b0 * ub + Omega_c0 * uc) / wsum
    D0 = float(D_raw[-1])
    D = D_raw / max(D0, 1e-30)
    f = Dp_raw / np.maximum(D_raw, 1e-30)
    return {
        'N': N, 'z': z,
        'delta_b': db, 'delta_c': dc,
        'D_raw': D_raw, 'D': D, 'f': f
    }


def _top_hat_window(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-4
    xs = x[small]
    out[small] = 1.0 - xs * xs / 10.0 + xs ** 4 / 280.0
    xt = x[~small]
    out[~small] = 3.0 * (np.sin(xt) - xt * np.cos(xt)) / np.maximum(xt ** 3, 1e-30)
    return out


def transfer_eh_nw(k_mpc: np.ndarray, *, h: float, omega_b: float, omega_c: float, Tcmb: float = T_CMB) -> np.ndarray:
    k_mpc = np.asarray(k_mpc, dtype=float)
    omega_m = float(omega_b + omega_c)
    fb = float(omega_b / omega_m) if omega_m > 0.0 else 0.0
    theta = Tcmb / 2.7
    s = 44.5 * np.log(9.83 / max(omega_m, 1e-30)) / np.sqrt(1.0 + 10.0 * omega_b ** 0.75)
    alpha_gamma = 1.0 - 0.328 * np.log(431.0 * max(omega_m, 1e-30)) * fb + 0.38 * np.log(22.3 * max(omega_m, 1e-30)) * fb * fb
    Gamma_eff = omega_m * h * (alpha_gamma + (1.0 - alpha_gamma) / (1.0 + (0.43 * k_mpc * s) ** 4))
    q = k_mpc * theta * theta / np.maximum(Gamma_eff, 1e-30)
    L0 = np.log(2.0 * math.e + 1.8 * q)
    C0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
    return L0 / (L0 + C0 * q * q)


def _default_k_grid() -> np.ndarray:
    return np.logspace(-3.0, 0.7, 48)


def _sigma8_raw_transfer_only(*, h: float, omega_b: float, omega_c: float, As: float, ns: float, k_grid_hmpc: Optional[np.ndarray] = None) -> float:
    if k_grid_hmpc is None:
        k_grid_hmpc = _default_k_grid()
    k_h = np.asarray(k_grid_hmpc, dtype=float)
    k_mpc = h * k_h
    T = transfer_eh_nw(k_mpc, h=h, omega_b=omega_b, omega_c=omega_c)
    # Use a calibrated shape-only proxy; absolute amplitude will be set by calibration factor.
    P = As * (k_mpc / K_PIVOT_MPC) ** (ns - 1.0) * T * T * np.maximum(k_h, 1e-30)
    R8 = 8.0 / h
    W = _top_hat_window(k_mpc * R8)
    integrand = (k_h ** 3 * P / (2.0 * np.pi ** 2)) * W * W
    sigma2 = np.trapz(integrand, x=np.log(np.maximum(k_h, 1e-30)))
    return float(np.sqrt(max(sigma2, 0.0)))


_SIGMA8_CAL = None


def sigma8_0_from_As_hybrid(*, h: float, omega_b: float, omega_c: float, As: float = A_S_REF, ns: float = N_S_REF, k_grid_hmpc: Optional[np.ndarray] = None) -> float:
    """Stable hybrid normalization.

    We anchor the absolute amplitude to a Planck-like LCDM reference through A_s,
    and deliberately avoid using the uncalibrated transfer-based absolute amplitude as
    a direct predictor. The shape dependence enters in the growth sector, not here.
    """
    return float(SIGMA8_REF * math.sqrt(max(float(As), 1e-30) / A_S_REF))


def solve_linear_growth_5c11(bg: Dict[str, np.ndarray], *,
                             Omega_b0: float, Omega_c0: float,
                             h: float, omega_b: float, omega_c: float,
                             beta0: float, nu2: float, lam4: float, xi_r: float,
                             As: float = A_S_REF, ns: float = N_S_REF,
                             k_eff_hmpc: float = 0.10,
                             k_values_hmpc: Optional[np.ndarray] = None,
                             z_start: float = 100.0,
                             delta_norm: float = 1.0) -> Dict[str, np.ndarray]:
    N_all = np.asarray(bg['N'], dtype=float)
    z_all = np.asarray(bg['z'], dtype=float)
    Om_b_all = np.asarray(bg['Omega_b'], dtype=float)
    Om_c_all = np.asarray(bg['Omega_c'], dtype=float)
    q_all = np.asarray(bg['q'], dtype=float)
    x_all = np.asarray(bg['x'], dtype=float)
    y_all = np.asarray(bg['y'], dtype=float)
    r_all = np.asarray(bg['r'], dtype=float)

    idx0 = int(np.argmin(np.abs(z_all - float(z_start))))
    N = N_all[idx0:].copy()
    z = z_all[idx0:].copy()
    Om_b = Om_b_all[idx0:].copy()
    Om_c = Om_c_all[idx0:].copy()
    q = q_all[idx0:].copy()
    x_arr = x_all[idx0:].copy()
    y_arr = y_all[idx0:].copy()
    r_arr = r_all[idx0:].copy()
    a_arr = np.exp(N)
    dlnE = -1.0 - q

    # Dynamic coupling from field motion
    qhat_arr = float(beta0) * y_arr

    # Effective mass for Yukawa-type screening
    m_eff2 = np.maximum(float(nu2) + 3.0 * float(lam4) * x_arr * x_arr + float(xi_r) * r_arr, 1e-12)

    if k_values_hmpc is None:
        k_values = np.array([float(k_eff_hmpc)], dtype=float)
    else:
        k_values = np.asarray(k_values_hmpc, dtype=float)
    if np.all(np.abs(k_values - float(k_eff_hmpc)) > 1e-12):
        k_values = np.sort(np.unique(np.concatenate([k_values, [float(k_eff_hmpc)]])))
    else:
        k_values = np.sort(np.unique(k_values))
    nk = k_values.size
    ktilde2 = (k_values * C_LIGHT / 100.0) ** 2  # dimensionless approximation

    mu = 1.0 + (2.0 * float(beta0) ** 2) * (ktilde2[None, :] / (ktilde2[None, :] + (a_arr[:, None] ** 2) * m_eff2[:, None]))

    d0 = float(delta_norm) * math.exp(float(N[0]))
    db = np.full(nk, d0)
    ub = np.full(nk, d0)
    dc = np.full(nk, d0)
    uc = np.full(nk, d0)
    Dk_raw = np.empty((N.size, nk))
    fk = np.empty((N.size, nk))
    Dk_raw[0] = (Omega_b0 * db + Omega_c0 * dc) / max(Omega_b0 + Omega_c0, 1e-30)
    fk[0] = np.ones(nk)

    for i in range(1, N.size):
        hN = float(N[i] - N[i - 1])
        im = i - 1
        ob = Om_b[im]; oc = Om_c[im]; dl = dlnE[im]; qh = qhat_arr[im]; muv = mu[im]

        def rhs(dbv, ubv, dcv, ucv):
            src_b = 1.5 * (ob * dbv + oc * dcv)
            src_c = 1.5 * (ob * dbv + oc * muv * dcv)
            return ubv, -(2.0 + dl) * ubv + src_b, ucv, -(2.0 + dl - qh) * ucv + src_c

        k1 = rhs(db, ub, dc, uc)
        k2 = rhs(db + 0.5 * hN * k1[0], ub + 0.5 * hN * k1[1], dc + 0.5 * hN * k1[2], uc + 0.5 * hN * k1[3])
        k3 = rhs(db + 0.5 * hN * k2[0], ub + 0.5 * hN * k2[1], dc + 0.5 * hN * k2[2], uc + 0.5 * hN * k2[3])

        ob2 = Om_b[i]; oc2 = Om_c[i]; dl2 = dlnE[i]; qh2 = qhat_arr[i]; muv2 = mu[i]
        def rhs2(dbv, ubv, dcv, ucv):
            src_b = 1.5 * (ob2 * dbv + oc2 * dcv)
            src_c = 1.5 * (ob2 * dbv + oc2 * muv2 * dcv)
            return ubv, -(2.0 + dl2) * ubv + src_b, ucv, -(2.0 + dl2 - qh2) * ucv + src_c

        k4 = rhs2(db + hN * k3[0], ub + hN * k3[1], dc + hN * k3[2], uc + hN * k3[3])

        db = db + (hN / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        ub = ub + (hN / 6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        dc = dc + (hN / 6.0) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        uc = uc + (hN / 6.0) * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])

        dm = (Omega_b0 * db + Omega_c0 * dc) / max(Omega_b0 + Omega_c0, 1e-30)
        dmp = (Omega_b0 * ub + Omega_c0 * uc) / max(Omega_b0 + Omega_c0, 1e-30)
        Dk_raw[i] = dm
        fk[i] = dmp / np.maximum(dm, 1e-30)

    Dk = Dk_raw / np.maximum(Dk_raw[-1], 1e-30)

    # Hybrid normalization from As:
    g_ref = solve_linear_growth_reference(build_lcdm_reference_background(
        h=h, Omega_b0=Omega_b0, Omega_c0=Omega_c0, Omega_r0=float(np.asarray(bg['r'], dtype=float)[-1]), z_max=max(z_start, float(np.max(z))+1.0), npts=max(len(z_all), len(z)+1)
    ), Omega_b0=Omega_b0, Omega_c0=Omega_c0, z_start=z_start, delta_norm=delta_norm)
    sigma8_ref_like = sigma8_0_from_As_hybrid(h=h, omega_b=omega_b, omega_c=omega_c, As=As, ns=ns, k_grid_hmpc=k_values)
    D_ref_start = float(g_ref['D'][0])
    # pivot D for sigma8 evolution
    ipiv = int(np.argmin(np.abs(k_values - float(k_eff_hmpc))))
    D_sigma = Dk[:, ipiv]
    D_sigma_start = max(float(D_sigma[0]), 1e-30)
    sigma8_ini = sigma8_ref_like * D_ref_start
    sigma8_0_pred = sigma8_ini / D_sigma_start
    sigma8_z = sigma8_0_pred * D_sigma

    return {
        'N': N,
        'z': z,
        'k_values_hmpc': k_values,
        'D_k': Dk,
        'f_k': fk,
        'D_sigma': D_sigma,
        'sigma8': sigma8_z,
        'sigma8_0_pred': float(sigma8_0_pred),
        'S8_0_pred': float(sigma8_0_pred * math.sqrt((Omega_b0 + Omega_c0) / 0.3)),
        'qhat': qhat_arr,
        'mu_grid': mu,
        'm_eff2': m_eff2,
        'As': float(As),
        'ns': float(ns),
        'k_eff_hmpc': float(k_eff_hmpc),
    }


def _predict_fs8_at_dataset(growth: Dict[str, np.ndarray], z_points: np.ndarray, k_points: np.ndarray) -> np.ndarray:
    z_grid = np.asarray(growth['z'], dtype=float)
    D_sigma = np.asarray(growth['D_sigma'], dtype=float)
    sigma8_0 = float(growth['sigma8_0_pred'])
    k_values = np.asarray(growth['k_values_hmpc'], dtype=float)
    f_k = np.asarray(growth['f_k'], dtype=float)  # shape (nz, nk)
    pred = np.empty_like(z_points, dtype=float)
    # sort z increasing for interp
    order_z = np.argsort(z_grid)
    z_sorted = z_grid[order_z]
    D_sorted = D_sigma[order_z]
    f_sorted = f_k[order_z, :]
    for i, (zv, kv) in enumerate(zip(z_points, k_points)):
        Dz = float(np.interp(float(zv), z_sorted, D_sorted))
        # interpolate f in z for each k-channel, then in k
        f_vs_k = np.array([np.interp(float(zv), z_sorted, f_sorted[:, j]) for j in range(k_values.size)], dtype=float)
        fz = float(np.interp(float(kv), k_values, f_vs_k))
        pred[i] = fz * sigma8_0 * Dz
    return pred


class GrowthLikelihood5C11:
    def __init__(self, dataset: GrowthDataset):
        self.dataset = dataset

    def evaluate(self, growth: Dict[str, np.ndarray]) -> Dict[str, object]:
        pred = _predict_fs8_at_dataset(growth, self.dataset.z, self.dataset.k_eff)
        diff = pred - self.dataset.fs8
        if self.dataset.icov is not None:
            chi2 = float(diff @ self.dataset.icov @ diff)
        else:
            chi2 = float(np.sum((diff / self.dataset.sigma) ** 2))
        return {
            'chi2_growth': chi2,
            'z': self.dataset.z.copy(),
            'fs8_obs': self.dataset.fs8.copy(),
            'sigma_obs': self.dataset.sigma.copy(),
            'k_eff': self.dataset.k_eff.copy(),
            'fs8_th': pred,
        }


def validate_lcdm_limit(bg: Dict[str, np.ndarray], *, Omega_b0: float, Omega_c0: float,
                        h: float, omega_b: float, omega_c: float,
                        z_start: float = 100.0, k_pivot_hmpc: float = 0.10) -> Dict[str, float]:
    # Old reference module
    from tuo_growth_module import solve_linear_growth as solve_old
    g_old = solve_old(bg, Omega_b0=Omega_b0, Omega_c0=Omega_c0, sigma8_0=1.0, z_start=z_start,
                      beta_eff=np.zeros_like(bg['N']), y_field=np.zeros_like(bg['N']))
    g_new = solve_linear_growth_5c11(bg, Omega_b0=Omega_b0, Omega_c0=Omega_c0,
                                     h=h, omega_b=omega_b, omega_c=omega_c,
                                     beta0=0.0, nu2=0.0, lam4=0.0, xi_r=0.0,
                                     As=A_S_REF, ns=N_S_REF,
                                     k_eff_hmpc=k_pivot_hmpc,
                                     k_values_hmpc=np.array([k_pivot_hmpc], dtype=float),
                                     z_start=z_start)
    z = np.asarray(g_old['z'], dtype=float)
    D_old = np.asarray(g_old['D'], dtype=float)
    f_old = np.asarray(g_old['f'], dtype=float)
    D_new = np.interp(z, np.asarray(g_new['z'])[::-1], np.asarray(g_new['D_sigma'])[::-1])
    # compute f at pivot from new
    f_new = np.interp(z, np.asarray(g_new['z'])[::-1], np.asarray(g_new['f_k'])[::-1, 0])

    return {
        'max_rel_D': float(np.max(np.abs(D_new - D_old) / np.maximum(np.abs(D_old), 1e-12))),
        'max_rel_f': float(np.max(np.abs(f_new - f_old) / np.maximum(np.abs(f_old), 1e-12))),
        'sigma8_0_pred_5c11': float(g_new['sigma8_0_pred']),
        'S8_0_pred_5c11': float(g_new['S8_0_pred']),
    }
