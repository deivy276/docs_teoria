from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class GrowthDataset:
    name: str
    z: np.ndarray
    fs8: np.ndarray
    sigma: np.ndarray
    cov: Optional[np.ndarray] = None

    def __post_init__(self):
        self.z = np.asarray(self.z, dtype=float)
        self.fs8 = np.asarray(self.fs8, dtype=float)
        self.sigma = np.asarray(self.sigma, dtype=float)
        if self.cov is not None:
            self.cov = np.asarray(self.cov, dtype=float)
            self.icov = np.linalg.inv(self.cov)
        else:
            self.icov = None


def builtin_rsd_diag_minimal() -> GrowthDataset:
    z = np.array([0.15, 0.38, 0.51, 0.70, 0.85, 1.48], dtype=float)
    fs8 = np.array([0.53, 0.50, 0.46, 0.44, 0.39, 0.30], dtype=float)
    sigma = np.array([0.08, 0.05, 0.05, 0.06, 0.09, 0.08], dtype=float)
    return GrowthDataset(name='RSD_diag_minimal_builtin', z=z, fs8=fs8, sigma=sigma)


def builtin_boss_eboss_diag() -> GrowthDataset:
    z = np.array([0.38, 0.51, 0.70, 1.48], dtype=float)
    fs8 = np.array([0.497, 0.458, 0.437, 0.462], dtype=float)
    sigma = np.array([0.045, 0.038, 0.052, 0.045], dtype=float)
    return GrowthDataset(name='BOSS_eBOSS_diag_builtin', z=z, fs8=fs8, sigma=sigma)


def _load_numeric_matrix(path: Path) -> np.ndarray:
    # Try comma first, then whitespace.
    for delim in [',', None]:
        try:
            arr = np.loadtxt(path, delimiter=delim)
            if arr.size > 0:
                return np.asarray(arr, dtype=float)
        except Exception:
            pass
    raise ValueError(f'Could not read covariance matrix from {path}')


def load_growth_cov(path: Path, n_expected: Optional[int] = None) -> np.ndarray:
    arr = _load_numeric_matrix(path)
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 2:
        if n_expected is not None and arr.shape != (n_expected, n_expected):
            raise ValueError(f'Growth covariance shape {arr.shape} incompatible with n={n_expected}')
        return arr
    # 1D flattened cases
    flat = arr.ravel()
    if n_expected is None:
        n = int(round(np.sqrt(flat.size)))
        if n * n != flat.size:
            raise ValueError('Cannot infer square covariance matrix from 1D data')
        return flat.reshape((n, n))
    if flat.size == n_expected * n_expected:
        return flat.reshape((n_expected, n_expected))
    if flat.size == 1 + n_expected * n_expected:
        # Pantheon-like format with leading element.
        return flat[1:].reshape((n_expected, n_expected))
    raise ValueError(f'Cannot reshape covariance of length {flat.size} to ({n_expected},{n_expected})')


def load_growth_csv(path: Path, cov_path: Optional[Path] = None) -> GrowthDataset:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    required = {'z', 'fs8'}
    if not required.issubset(cols):
        raise ValueError('growth csv must contain columns z and fs8; optional sigma')
    z = df[cols['z']].to_numpy(dtype=float)
    fs8 = df[cols['fs8']].to_numpy(dtype=float)
    if 'sigma' in cols:
        sigma = df[cols['sigma']].to_numpy(dtype=float)
    else:
        sigma = np.full_like(z, np.nan, dtype=float)
    cov = None
    if cov_path is not None:
        cov = load_growth_cov(cov_path, n_expected=len(z))
        if np.any(~np.isfinite(sigma)):
            sigma = np.sqrt(np.diag(cov))
    else:
        if np.any(~np.isfinite(sigma)):
            raise ValueError('growth csv without covariance must include sigma column')
    return GrowthDataset(name=str(path), z=z, fs8=fs8, sigma=sigma, cov=cov)


def _interp(Nq: float, N: np.ndarray, arr: np.ndarray) -> float:
    return float(np.interp(Nq, N, arr))


def solve_linear_growth(bg: Dict[str, np.ndarray], *, Omega_b0: float, Omega_c0: float,
                        sigma8_0: float = 0.811, z_start: float = 100.0,
                        beta_eff: Optional[np.ndarray] = None,
                        y_field: Optional[np.ndarray] = None,
                        delta_norm: float = 1.0) -> Dict[str, np.ndarray]:
    N_all = np.asarray(bg['N'], dtype=float)
    z_all = np.asarray(bg['z'], dtype=float)
    Om_b_all = np.asarray(bg['Omega_b'], dtype=float)
    Om_c_all = np.asarray(bg['Omega_c'], dtype=float)
    q_all = np.asarray(bg['q'], dtype=float)

    if beta_eff is None:
        beta_eff_all = np.zeros_like(N_all)
    else:
        beta_eff_all = np.asarray(beta_eff, dtype=float)
    if y_field is None:
        y_all = np.zeros_like(N_all)
    else:
        y_all = np.asarray(y_field, dtype=float)

    idx0 = int(np.argmin(np.abs(z_all - float(z_start))))
    N = N_all[idx0:].copy()
    z = z_all[idx0:].copy()
    Om_b = Om_b_all[idx0:].copy()
    Om_c = Om_c_all[idx0:].copy()
    q = q_all[idx0:].copy()
    beta_eff_arr = beta_eff_all[idx0:].copy()
    y_arr = y_all[idx0:].copy()

    dlnE = -1.0 - q

    d0 = float(delta_norm) * np.exp(N[0])
    S = np.zeros((4, N.size), dtype=float)
    S[:, 0] = np.array([d0, d0, d0, d0], dtype=float)

    def rhs(Nq: float, state: np.ndarray) -> np.ndarray:
        db, ub, dc, uc = map(float, state)
        ob = _interp(Nq, N, Om_b)
        oc = _interp(Nq, N, Om_c)
        dl = _interp(Nq, N, dlnE)
        be = _interp(Nq, N, beta_eff_arr)
        yy = _interp(Nq, N, y_arr)
        src_b = 1.5 * (ob * db + oc * dc)
        src_c = 1.5 * (ob * db + oc * (1.0 + 2.0 * be * be) * dc)
        return np.array([
            ub,
            -(2.0 + dl) * ub + src_b,
            uc,
            -(2.0 + dl - be * yy) * uc + src_c,
        ], dtype=float)

    for i in range(1, N.size):
        h = float(N[i] - N[i - 1])
        s0 = S[:, i - 1]
        n0 = float(N[i - 1])
        k1 = rhs(n0, s0)
        k2 = rhs(n0 + 0.5 * h, s0 + 0.5 * h * k1)
        k3 = rhs(n0 + 0.5 * h, s0 + 0.5 * h * k2)
        k4 = rhs(n0 + h, s0 + h * k3)
        S[:, i] = s0 + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    delta_b = S[0]
    ddelta_b = S[1]
    delta_c = S[2]
    ddelta_c = S[3]

    wb = float(Omega_b0)
    wc = float(Omega_c0)
    wsum = wb + wc
    D_raw = (wb * delta_b + wc * delta_c) / wsum
    Dp_raw = (wb * ddelta_b + wc * ddelta_c) / wsum
    D0 = float(D_raw[-1])
    D = D_raw / D0
    f = Dp_raw / D_raw
    sigma8 = float(sigma8_0) * D
    fs8 = f * sigma8

    return {
        'N': N,
        'z': z,
        'delta_b': delta_b,
        'delta_c': delta_c,
        'D': D,
        'f': f,
        'sigma8': sigma8,
        'fs8': fs8,
        'beta_eff': beta_eff_arr,
        'y_field': y_arr,
    }


class GrowthLikelihood:
    def __init__(self, dataset: GrowthDataset):
        self.dataset = dataset

    def evaluate(self, growth: Dict[str, np.ndarray]) -> Dict[str, object]:
        z_pred_grid = np.asarray(growth['z'], dtype=float)
        fs8_grid = np.asarray(growth['fs8'], dtype=float)
        order = np.argsort(z_pred_grid)
        z_sorted = z_pred_grid[order]
        fs8_sorted = fs8_grid[order]
        pred = np.interp(self.dataset.z, z_sorted, fs8_sorted)
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
            'fs8_th': pred,
        }
