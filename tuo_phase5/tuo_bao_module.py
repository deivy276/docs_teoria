import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.linalg import cho_factor, cho_solve

C_KM_S = 299792.458


@dataclass
class BAODataSet:
    name: str
    z: np.ndarray
    obs: List[str]
    y: np.ndarray
    cov: np.ndarray
    meta: Optional[Dict] = None

    @property
    def size(self) -> int:
        return int(len(self.y))


# ------------------------
# Drag epoch / sound horizon
# ------------------------
def z_drag_eh98(omega_b: float, omega_m: float) -> float:
    """Eisenstein & Hu 1998 fitting formula for the drag redshift z_d.

    Parameters
    ----------
    omega_b : float
        Physical baryon density Ω_b h^2.
    omega_m : float
        Physical matter density Ω_m h^2.
    """
    b1 = 0.313 * omega_m ** (-0.419) * (1.0 + 0.607 * omega_m ** 0.674)
    b2 = 0.238 * omega_m ** 0.223
    return (1291.0 * omega_m ** 0.251 / (1.0 + 0.659 * omega_m ** 0.828)) * (1.0 + b1 * omega_b ** b2)


def sound_speed_fraction(z: np.ndarray, Omega_b0: float, Omega_gamma0: float) -> np.ndarray:
    """c_s/c in the tightly-coupled photon-baryon fluid."""
    z = np.asarray(z, dtype=float)
    Rb = (3.0 * Omega_b0 / (4.0 * Omega_gamma0)) / (1.0 + z)
    return 1.0 / np.sqrt(3.0 * (1.0 + Rb))


def numerical_r_d_from_background(bg: Dict[str, np.ndarray], *, h: float, Omega_b0: float,
                                  Omega_gamma0: float, z_drag: float,
                                  z_max: Optional[float] = None) -> float:
    """Numerical sound horizon at the drag epoch using a precomputed background.

    Returns r_d in Mpc.
    """
    z = np.asarray(bg['z'], dtype=float)
    E = np.asarray(bg['E'], dtype=float)
    order = np.argsort(z)
    z = z[order]
    E = E[order]
    if z_max is not None:
        mask = z <= float(z_max)
        z = z[mask]
        E = E[mask]
    if z_drag < z[0] or z_drag > z[-1]:
        raise ValueError(f"z_drag={z_drag:.2f} outside background grid [{z[0]:.3g}, {z[-1]:.3g}]")

    # integrate from z_d to z_max on the existing z-grid plus exact z_d insertion
    i0 = int(np.searchsorted(z, z_drag, side='left'))
    if i0 == 0:
        z_int = z.copy()
        E_int = E.copy()
    else:
        z_left, z_right = z[i0 - 1], z[i0]
        E_drag = E[i0 - 1] + (E[i0] - E[i0 - 1]) * (z_drag - z_left) / (z_right - z_left)
        z_int = np.concatenate(([z_drag], z[i0:]))
        E_int = np.concatenate(([E_drag], E[i0:]))

    cs_frac = sound_speed_fraction(z_int, Omega_b0=Omega_b0, Omega_gamma0=Omega_gamma0)
    integrand = cs_frac / E_int
    chi = np.trapz(integrand, z_int)
    Dh0 = C_KM_S / (100.0 * h)
    return Dh0 * chi


# ------------------------
# Distance measures
# ------------------------
def build_distance_cache(bg: Dict[str, np.ndarray], *, h: float) -> Dict[str, np.ndarray]:
    z = np.asarray(bg['z'], dtype=float)
    E = np.asarray(bg['E'], dtype=float)
    order = np.argsort(z)
    z = z[order]
    E = E[order]
    invE = 1.0 / E
    chi_dimless = cumulative_trapezoid(invE, z, initial=0.0)
    Dh0 = C_KM_S / (100.0 * h)
    DM = Dh0 * chi_dimless
    DH = Dh0 * invE
    return {
        'z': z,
        'E': E,
        'DH': DH,
        'DM': DM,
        'DH_interp': interp1d(z, DH, kind='cubic', bounds_error=False, fill_value='extrapolate'),
        'DM_interp': interp1d(z, DM, kind='cubic', bounds_error=False, fill_value='extrapolate'),
    }


def DH_of_z(z: np.ndarray, cache: Dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray(cache['DH_interp'](np.asarray(z, dtype=float)), dtype=float)


def DM_of_z(z: np.ndarray, cache: Dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray(cache['DM_interp'](np.asarray(z, dtype=float)), dtype=float)


def DV_of_z(z: np.ndarray, cache: Dict[str, np.ndarray]) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    DH = DH_of_z(z, cache)
    DM = DM_of_z(z, cache)
    return (z * DH * DM * DM) ** (1.0 / 3.0)


# ------------------------
# Built-in BAO compilations
# ------------------------
def builtin_desi_dr2_bao() -> BAODataSet:
    """DESI DR2 BAO background-level compilation.

    Included points:
      - BGS isotropic DV/rd at z=0.295
      - Anisotropic (DM/rd, DH/rd) at z=0.510, 0.706, 0.934, 1.321, 1.484, 2.330

    Covariance: block diagonal, with per-redshift 2x2 blocks built from quoted 1σ errors and
    the published correlation coefficient r_{M,H} for each anisotropic bin.
    """
    entries = [
        (0.295, 'DV_over_rd', 7.942, 0.075, None),
        (0.510, 'DM_over_rd', 13.587, 0.169, -0.475),
        (0.510, 'DH_over_rd', 21.863, 0.427, -0.475),
        (0.706, 'DM_over_rd', 17.347, 0.180, -0.423),
        (0.706, 'DH_over_rd', 19.458, 0.332, -0.423),
        (0.934, 'DM_over_rd', 21.574, 0.153, -0.425),
        (0.934, 'DH_over_rd', 17.641, 0.193, -0.425),
        (1.321, 'DM_over_rd', 27.605, 0.320, -0.437),
        (1.321, 'DH_over_rd', 14.178, 0.217, -0.437),
        (1.484, 'DM_over_rd', 30.519, 0.758, -0.489),
        (1.484, 'DH_over_rd', 12.816, 0.513, -0.489),
        (2.330, 'DM_over_rd', 38.990, 0.530, -0.431),
        (2.330, 'DH_over_rd', 8.632, 0.101, -0.431),
    ]
    z = np.array([e[0] for e in entries], dtype=float)
    obs = [e[1] for e in entries]
    y = np.array([e[2] for e in entries], dtype=float)
    sig = np.array([e[3] for e in entries], dtype=float)
    cov = np.zeros((len(entries), len(entries)), dtype=float)
    i = 0
    while i < len(entries):
        cov[i, i] = sig[i] ** 2
        if obs[i] == 'DV_over_rd':
            i += 1
            continue
        rho = entries[i][4]
        cov[i + 1, i + 1] = sig[i + 1] ** 2
        cov[i, i + 1] = cov[i + 1, i] = float(rho) * sig[i] * sig[i + 1]
        i += 2
    return BAODataSet(
        name='DESI_DR2_builtin',
        z=z,
        obs=obs,
        y=y,
        cov=cov,
        meta={
            'source': 'DESI DR2 BAO (BGS + LRG/ELG/QSO + Lyα)',
            'covariance': 'block diagonal from quoted σ and r_MH',
        },
    )


def builtin_eboss_dr16_diagonal() -> BAODataSet:
    """Lightweight diagonal compilation of eBOSS/SDSS DR16 BAO points.

    This is provided as a quick cross-check only. If full covariance tables are available,
    prefer loading them through `load_bao_from_files`.
    """
    entries = [
        (0.150, 'DV_over_rd', 4.51, 0.14),
        (0.380, 'DM_over_rd', 10.27, 0.15),
        (0.380, 'DH_over_rd', 24.89, 0.58),
        (0.510, 'DM_over_rd', 13.38, 0.18),
        (0.510, 'DH_over_rd', 22.43, 0.48),
        (0.698, 'DM_over_rd', 17.65, 0.30),
        (0.698, 'DH_over_rd', 19.78, 0.46),
        (0.850, 'DM_over_rd', 19.6, 2.1),
        (0.850, 'DH_over_rd', 19.5, 1.0),
        (1.480, 'DM_over_rd', 30.21, 0.79),
        (1.480, 'DH_over_rd', 13.23, 0.47),
        (2.330, 'DM_over_rd', 37.6, 1.9),
        (2.330, 'DH_over_rd', 8.93, 0.28),
    ]
    z = np.array([e[0] for e in entries], dtype=float)
    obs = [e[1] for e in entries]
    y = np.array([e[2] for e in entries], dtype=float)
    sig = np.array([e[3] for e in entries], dtype=float)
    cov = np.diag(sig ** 2)
    return BAODataSet('eBOSS_DR16_diag', z, obs, y, cov, meta={'warning': 'diagonal approximation'})


def load_bao_from_files(data_csv: Path, cov_csv: Optional[Path] = None) -> BAODataSet:
    df = pd.read_csv(data_csv)
    required = {'z', 'obs', 'value'}
    if not required.issubset(df.columns):
        raise ValueError(f'BAO CSV must contain columns {required}')
    z = df['z'].to_numpy(dtype=float)
    obs = df['obs'].astype(str).tolist()
    y = df['value'].to_numpy(dtype=float)
    if cov_csv is not None:
        cov = np.loadtxt(cov_csv, delimiter=',')
    elif 'sigma' in df.columns:
        sig = df['sigma'].to_numpy(dtype=float)
        cov = np.diag(sig ** 2)
    else:
        raise ValueError('Provide cov_csv or a sigma column in data_csv')
    return BAODataSet(name=data_csv.stem, z=z, obs=obs, y=y, cov=np.asarray(cov, dtype=float))


# ------------------------
# Likelihood
# ------------------------
class BAOLikelihood:
    def __init__(self, dataset: BAODataSet, *, h: float = 0.674,
                 use_numeric_rd: bool = True, z_drag_mode: str = 'eh98'):
        self.dataset = dataset
        self.h = float(h)
        self.use_numeric_rd = bool(use_numeric_rd)
        self.z_drag_mode = z_drag_mode
        self.cho = cho_factor(dataset.cov, lower=True, check_finite=False)

    def z_drag(self, *, Omega_b0: float, Omega_c0: float) -> float:
        omega_b = Omega_b0 * self.h ** 2
        omega_m = (Omega_b0 + Omega_c0) * self.h ** 2
        return z_drag_eh98(omega_b, omega_m)

    def r_d(self, bg: Dict[str, np.ndarray], *, Omega_b0: float, Omega_c0: float,
            Omega_gamma0: float) -> Tuple[float, float]:
        zd = self.z_drag(Omega_b0=Omega_b0, Omega_c0=Omega_c0)
        if self.use_numeric_rd:
            rd = numerical_r_d_from_background(bg, h=self.h, Omega_b0=Omega_b0,
                                               Omega_gamma0=Omega_gamma0, z_drag=zd)
        else:
            # fallback: still compute numerically since this is the physically clean baseline
            rd = numerical_r_d_from_background(bg, h=self.h, Omega_b0=Omega_b0,
                                               Omega_gamma0=Omega_gamma0, z_drag=zd)
        return float(rd), float(zd)

    def theory_vector(self, bg: Dict[str, np.ndarray], *, Omega_b0: float, Omega_c0: float,
                      Omega_gamma0: float) -> Tuple[np.ndarray, Dict[str, float]]:
        cache = build_distance_cache(bg, h=self.h)
        rd, zd = self.r_d(bg, Omega_b0=Omega_b0, Omega_c0=Omega_c0, Omega_gamma0=Omega_gamma0)
        out = np.empty(self.dataset.size, dtype=float)
        for i, (z, obs) in enumerate(zip(self.dataset.z, self.dataset.obs)):
            if obs == 'DH_over_rd':
                out[i] = float(DH_of_z(z, cache) / rd)
            elif obs == 'DM_over_rd':
                out[i] = float(DM_of_z(z, cache) / rd)
            elif obs == 'DV_over_rd':
                out[i] = float(DV_of_z(z, cache) / rd)
            else:
                raise ValueError(f'Unknown BAO observable type: {obs}')
        return out, {'r_d': rd, 'z_drag': zd}

    def chi2(self, theory_vec: np.ndarray) -> float:
        delta = self.dataset.y - np.asarray(theory_vec, dtype=float)
        Cinv_delta = cho_solve(self.cho, delta, check_finite=False)
        return float(delta @ Cinv_delta)

    def evaluate(self, bg: Dict[str, np.ndarray], *, Omega_b0: float, Omega_c0: float,
                 Omega_gamma0: float) -> Dict[str, float]:
        vec, meta = self.theory_vector(bg, Omega_b0=Omega_b0, Omega_c0=Omega_c0, Omega_gamma0=Omega_gamma0)
        chi2 = self.chi2(vec)
        return {
            'chi2_bao': float(chi2),
            'r_d_Mpc': float(meta['r_d']),
            'z_drag': float(meta['z_drag']),
        }
