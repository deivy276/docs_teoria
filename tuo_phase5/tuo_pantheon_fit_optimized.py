import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize_scalar

from tuo_phase4_core_optimized import TUOPhase4CoreOptimized

C_KM_S = 299792.458


def load_pantheon(data_path: Path, cov_path: Path):
    df = pd.read_csv(data_path, sep=r"\s+", comment="#")
    with open(cov_path, "r") as f:
        n = int(f.readline().strip())
        flat = np.loadtxt(f)
    cov = flat.reshape((n, n))
    if len(df) != n:
        raise ValueError(f"Data/covariance mismatch: len(df)={len(df)} but cov={n}")
    return df, cov


def apply_selection(df, cov, zmin=0.01, drop_calibrators=True, use_hubble_flow_flag=False):
    mask = np.ones(len(df), dtype=bool)
    mask &= np.isfinite(df["zHD"].to_numpy())
    mask &= np.isfinite(df["m_b_corr"].to_numpy())
    mask &= df["zHD"].to_numpy() > zmin
    if drop_calibrators:
        mask &= (df["IS_CALIBRATOR"].to_numpy() == 0)
    if use_hubble_flow_flag:
        mask &= (df["USED_IN_SH0ES_HF"].to_numpy() == 1)
    idx = np.where(mask)[0]
    return df.iloc[idx].reset_index(drop=True), cov[np.ix_(idx, idx)], idx


def prepare_covariance(cov):
    jitter = 0.0
    try:
        cho = cho_factor(cov, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        jitter = 1e-12 * np.mean(np.diag(cov))
        cho = cho_factor(cov + jitter * np.eye(cov.shape[0]), lower=True, check_finite=False)
    return cho, jitter


def marginalised_chi2(residual, cho):
    ones = np.ones_like(residual)
    Cinvr = cho_solve(cho, residual, check_finite=False)
    Cinv1 = cho_solve(cho, ones, check_finite=False)
    A = float(residual @ Cinvr)
    B = float(ones @ Cinvr)
    C = float(ones @ Cinv1)
    Mhat = B / C
    chi2 = A - B * B / C
    return chi2, Mhat


def lcdm_E(z, omega_m):
    om = float(omega_m)
    return np.sqrt(om * (1.0 + z) ** 3 + (1.0 - om))


def dimensionless_dl_from_E(z_eval, E_func, zmax=None, ngrid=20000):
    z_eval = np.asarray(z_eval, dtype=float)
    z_hi = float(np.max(z_eval) if zmax is None else max(zmax, np.max(z_eval)))
    z_grid = np.linspace(0.0, z_hi, int(ngrid))
    E_grid = E_func(z_grid)
    invE = 1.0 / E_grid
    chi = cumulative_trapezoid(invE, z_grid, initial=0.0)
    dL = (1.0 + z_grid) * chi
    interp = interp1d(z_grid, dL, kind='cubic', bounds_error=False, fill_value='extrapolate')
    return interp(z_eval)


def mu_from_dimensionless_dl(dl_dimless):
    arr = np.asarray(dl_dimless, dtype=float)
    if np.any(arr <= 0.0):
        raise ValueError('Luminosity distance must be positive')
    return 5.0 * np.log10(arr)


class PantheonLikelihoodOptimized:
    def __init__(self, df_sel, cov_sel):
        self.df = df_sel
        self.z = df_sel['zHD'].to_numpy(dtype=float)
        self.mb = df_sel['m_b_corr'].to_numpy(dtype=float)
        self.cho, self.jitter = prepare_covariance(cov_sel)

    def chi2_from_mu(self, mu_model):
        res = self.mb - mu_model
        return marginalised_chi2(res, self.cho)

    def fit_lcdm(self):
        def obj(om):
            if om <= 0.0 or om >= 1.0:
                return np.inf
            dl = dimensionless_dl_from_E(self.z, lambda zz: lcdm_E(zz, om), zmax=float(np.max(self.z)))
            mu = mu_from_dimensionless_dl(dl)
            chi2, _ = self.chi2_from_mu(mu)
            return chi2
        opt = minimize_scalar(obj, bounds=(0.05, 0.6), method='bounded', options={'xatol':1e-4})
        om_best = float(opt.x)
        dl = dimensionless_dl_from_E(self.z, lambda zz: lcdm_E(zz, om_best), zmax=float(np.max(self.z)))
        mu = mu_from_dimensionless_dl(dl)
        chi2, Mhat = self.chi2_from_mu(mu)
        return {'omega_m': om_best, 'chi2': float(chi2), 'Mhat': float(Mhat)}

    def evaluate_tuo(self, tuo_params, *, core=None, bg_npts=8000,
                     final_backend='numba', use_numba_shoot=True):
        if core is None:
            core = TUOPhase4CoreOptimized(tuo_params)
        else:
            core.update(**tuo_params)
        Omega_V0, sol = core.solve(final_backend=final_backend, use_numba_shoot=use_numba_shoot,
                                   grid_npts=bg_npts if final_backend == 'numba' else None)
        bg = core.sample_background(Omega_V0, sol, z_max=max(core.z_ini, float(np.max(self.z)) + 10.0), npts=bg_npts)
        z_bg = bg['z']
        E_bg = bg['E']
        order = np.argsort(z_bg)
        z_sorted = z_bg[order]
        E_sorted = E_bg[order]
        dL = dimensionless_dl_from_E(self.z, lambda zz: np.interp(zz, z_sorted, E_sorted), zmax=float(np.max(self.z)), ngrid=min(int(bg_npts), 20000))
        mu = mu_from_dimensionless_dl(dL)
        chi2, Mhat = self.chi2_from_mu(mu)
        diag = core.diagnostics(bg)
        return {
            'Omega_V0': float(Omega_V0),
            'chi2': float(chi2),
            'Mhat': float(Mhat),
            'shoot_evals': int(core.last_shoot_evals),
            **{k: (None if v is None else float(v)) for k, v in diag.items()},
        }


def main():
    ap = argparse.ArgumentParser(description='Optimized TUO Phase 4 vs Pantheon+')
    ap.add_argument('--data', type=Path, default=Path('Pantheon+SH0ES.dat'))
    ap.add_argument('--cov', type=Path, default=Path('Pantheon+SH0ES_STAT+SYS.cov'))
    ap.add_argument('--zmin', type=float, default=0.01)
    ap.add_argument('--keep-calibrators', action='store_true')
    ap.add_argument('--use-hubble-flow-flag', action='store_true')
    ap.add_argument('--beta0', type=float, default=0.02)
    ap.add_argument('--nu2', type=float, default=0.02)
    ap.add_argument('--lam4', type=float, default=0.0)
    ap.add_argument('--xi-r', type=float, default=10.0, dest='xi_r')
    ap.add_argument('--z-c', type=float, default=100.0, dest='z_c')
    ap.add_argument('--s', type=float, default=4.0)
    ap.add_argument('--x-ini', type=float, default=0.2, dest='x_ini')
    ap.add_argument('--y-ini', type=float, default=0.0, dest='y_ini')
    ap.add_argument('--z-ini', type=float, default=1e5, dest='z_ini')
    ap.add_argument('--Omega-b0', type=float, default=0.05, dest='Omega_b0')
    ap.add_argument('--Omega-c0', type=float, default=0.25, dest='Omega_c0')
    ap.add_argument('--Omega-r0', type=float, default=9.2e-5, dest='Omega_r0')
    ap.add_argument('--bg-npts', type=int, default=8000)
    ap.add_argument('--final-backend', choices=['scipy', 'numba'], default='numba')
    ap.add_argument('--no-numba-shoot', action='store_true')
    ap.add_argument('--json-out', type=Path, default=None)
    args = ap.parse_args()

    df, cov = load_pantheon(args.data, args.cov)
    df_sel, cov_sel, _ = apply_selection(df, cov, zmin=args.zmin,
                                         drop_calibrators=not args.keep_calibrators,
                                         use_hubble_flow_flag=args.use_hubble_flow_flag)
    like = PantheonLikelihoodOptimized(df_sel, cov_sel)
    lcdm = like.fit_lcdm()
    tuo_params = {
        'Omega_b0': args.Omega_b0,
        'Omega_c0': args.Omega_c0,
        'Omega_r0': args.Omega_r0,
        'beta0': args.beta0,
        'nu2': args.nu2,
        'lam4': args.lam4,
        'xi_r': args.xi_r,
        'z_c': args.z_c,
        's': args.s,
        'x_ini': args.x_ini,
        'y_ini': args.y_ini,
        'z_ini': args.z_ini,
    }
    core = TUOPhase4CoreOptimized(tuo_params)
    tuo = like.evaluate_tuo(tuo_params, core=core, bg_npts=args.bg_npts,
                            final_backend=args.final_backend, use_numba_shoot=not args.no_numba_shoot)
    out = {
        'selection': {
            'n_total': int(len(df)),
            'n_selected': int(len(df_sel)),
            'zmin': args.zmin,
            'drop_calibrators': not args.keep_calibrators,
            'use_hubble_flow_flag': args.use_hubble_flow_flag,
            'cov_jitter': float(like.jitter),
        },
        'lcdm': lcdm,
        'tuo': tuo,
        'tuo_params': tuo_params,
        'delta_chi2_tuo_minus_lcdm': float(tuo['chi2'] - lcdm['chi2']),
    }
    txt = json.dumps(out, indent=2)
    print(txt)
    if args.json_out is not None:
        args.json_out.write_text(txt)


if __name__ == '__main__':
    main()
