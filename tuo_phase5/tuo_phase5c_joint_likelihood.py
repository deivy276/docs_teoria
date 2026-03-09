
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from tuo_phase4_core_optimized import TUOPhase4CoreOptimized
from tuo_pantheon_fit_optimized import PantheonLikelihoodOptimized, apply_selection, load_pantheon
from tuo_bao_module import builtin_desi_dr2_bao, builtin_eboss_dr16_diagonal
from tuo_phase5b_joint_likelihood import PantheonBAOJointLikelihood5B, derive_background_densities
from tuo_growth_module import (
    GrowthLikelihood,
    builtin_rsd_diag_minimal,
    builtin_boss_eboss_diag,
    load_growth_csv,
    load_growth_cov,
    solve_linear_growth,
)


def _beta_eff_array(N: np.ndarray, beta0: float, z_c: float, s: float) -> np.ndarray:
    N = np.asarray(N, dtype=float)
    N_c = -np.log1p(float(z_c))
    return 0.5 * float(beta0) * (1.0 + np.tanh(float(s) * (N - N_c)))


def _interp_basis(growth: Dict[str, np.ndarray], z_eval: np.ndarray) -> np.ndarray:
    z_pred = np.asarray(growth['z'], dtype=float)
    fs8 = np.asarray(growth['fs8'], dtype=float)
    order = np.argsort(z_pred)
    return np.interp(np.asarray(z_eval, dtype=float), z_pred[order], fs8[order])


def fit_sigma8_linear(dataset, fs8_basis: np.ndarray,
                      sigma8_mean: Optional[float] = None,
                      sigma8_sigma: Optional[float] = None,
                      bounds: Tuple[float, float] = (0.2, 1.5)) -> Tuple[float, float, float]:
    """Analytic best fit of sigma8_0 when fs8_th = sigma8_0 * fs8_basis."""
    g = np.asarray(fs8_basis, dtype=float)
    y = np.asarray(dataset.fs8, dtype=float)
    if dataset.icov is not None:
        A = float(g @ dataset.icov @ g)
        B = float(g @ dataset.icov @ y)
    else:
        w = 1.0 / np.asarray(dataset.sigma, dtype=float) ** 2
        A = float(np.sum(w * g * g))
        B = float(np.sum(w * g * y))

    prior_weight = 0.0
    prior_target = 0.0
    if sigma8_mean is not None and sigma8_sigma is not None and sigma8_sigma > 0.0:
        prior_weight = 1.0 / float(sigma8_sigma) ** 2
        prior_target = float(sigma8_mean) * prior_weight

    A_eff = A + prior_weight
    B_eff = B + prior_target
    if A_eff <= 0.0:
        return np.nan, np.inf, np.inf

    sigma8_best = float(np.clip(B_eff / A_eff, bounds[0], bounds[1]))
    diff = sigma8_best * g - y
    if dataset.icov is not None:
        chi2_growth = float(diff @ dataset.icov @ diff)
    else:
        chi2_growth = float(np.sum((diff / dataset.sigma) ** 2))

    chi2_prior = 0.0
    if prior_weight > 0.0:
        chi2_prior = ((sigma8_best - float(sigma8_mean)) / float(sigma8_sigma)) ** 2

    return sigma8_best, float(chi2_growth), float(chi2_prior)


class PantheonBAOGrowthJointLikelihood5C:
    def __init__(self, pantheon_like: PantheonLikelihoodOptimized, bao_dataset, growth_dataset):
        self.base = PantheonBAOJointLikelihood5B(pantheon_like, bao_dataset)
        self.growth_dataset = growth_dataset
        self.growth_like = GrowthLikelihood(growth_dataset)

    def priors_chi2(self, *, h: float, omega_b: float, omega_c: float,
                    h_mean: float, h_sigma: float,
                    omega_b_mean: float, omega_b_sigma: float,
                    omega_c_mean: float, omega_c_sigma: float,
                    sigma8_0: Optional[float] = None,
                    sigma8_mean: Optional[float] = None,
                    sigma8_sigma: Optional[float] = None) -> float:
        chi2 = self.base.priors_chi2(h=h, omega_b=omega_b, omega_c=omega_c,
                                     h_mean=h_mean, h_sigma=h_sigma,
                                     omega_b_mean=omega_b_mean, omega_b_sigma=omega_b_sigma,
                                     omega_c_mean=omega_c_mean, omega_c_sigma=omega_c_sigma)
        if sigma8_0 is not None and sigma8_mean is not None and sigma8_sigma is not None and sigma8_sigma > 0.0:
            chi2 += ((float(sigma8_0) - float(sigma8_mean)) / float(sigma8_sigma)) ** 2
        return float(chi2)

    def _lcdm_growth(self, *, h: float, omega_b: float, omega_c: float,
                     bg_npts: int = 12000, z_growth_start: float = 100.0):
        dens = derive_background_densities(h=h, omega_b=omega_b, omega_c=omega_c)
        om_b = dens['Omega_b0']; om_c = dens['Omega_c0']
        om = om_b + om_c
        orad = dens['Omega_r0']
        ov = 1.0 - om - orad
        zmax = max(float(z_growth_start), float(np.max(self.growth_dataset.z)) + 5.0)
        z_bg = np.expm1(np.linspace(0.0, np.log1p(zmax), bg_npts))
        N_bg = -np.log1p(z_bg)
        E = np.sqrt(om * (1.0 + z_bg) ** 3 + orad * (1.0 + z_bg) ** 4 + ov)
        dlnE = np.gradient(np.log(E), N_bg)
        q = -1.0 - dlnE
        b = om_b * np.exp(-3.0 * N_bg)
        c = om_c * np.exp(-3.0 * N_bg)
        e2 = E * E
        bg = {
            'N': N_bg, 'z': z_bg, 'E': E, 'q': q, 'y': np.zeros_like(N_bg),
            'Omega_b': b / e2, 'Omega_c': c / e2,
            'b': b, 'c': c, 'r': orad * np.exp(-4.0 * N_bg),
        }
        growth = solve_linear_growth(bg, Omega_b0=om_b, Omega_c0=om_c, sigma8_0=1.0,
                                     z_start=z_growth_start, beta_eff=np.zeros_like(N_bg), y_field=np.zeros_like(N_bg))
        return dens, bg, growth

    def fit_lcdm(self, *, h_mean: float = 0.674, h_sigma: float = 0.005,
                 omega_b_mean: float = 0.02237, omega_b_sigma: float = 0.00015,
                 omega_c_mean: float = 0.1200, omega_c_sigma: float = 0.0012,
                 sigma8_mean: Optional[float] = None, sigma8_sigma: Optional[float] = None,
                 bg_npts: int = 8000, z_growth_start: float = 100.0,
                 bounds: Optional[Tuple[Tuple[float, float], ...]] = None):
        base = self.base.fit_lcdm(h_mean=h_mean, h_sigma=h_sigma,
                                  omega_b_mean=omega_b_mean, omega_b_sigma=omega_b_sigma,
                                  omega_c_mean=omega_c_mean, omega_c_sigma=omega_c_sigma,
                                  bg_npts=bg_npts, bounds=bounds)
        dens, bg, growth = self._lcdm_growth(h=base['h'], omega_b=base['omega_b'], omega_c=base['omega_c'],
                                             bg_npts=bg_npts, z_growth_start=z_growth_start)
        fs8_basis = _interp_basis(growth, self.growth_dataset.z)
        sigma8_best, chi2_growth, chi2_sigma8_prior = fit_sigma8_linear(
            self.growth_dataset, fs8_basis, sigma8_mean=sigma8_mean, sigma8_sigma=sigma8_sigma
        )
        chi2_prior = self.priors_chi2(h=base['h'], omega_b=base['omega_b'], omega_c=base['omega_c'],
                                      sigma8_0=sigma8_best, sigma8_mean=sigma8_mean, sigma8_sigma=sigma8_sigma,
                                      h_mean=h_mean, h_sigma=h_sigma,
                                      omega_b_mean=omega_b_mean, omega_b_sigma=omega_b_sigma,
                                      omega_c_mean=omega_c_mean, omega_c_sigma=omega_c_sigma)
        out = dict(base)
        out.update({
            'sigma8_0_hat': float(sigma8_best),
            'chi2_growth': float(chi2_growth),
            'chi2_total': float(base['chi2_total'] + chi2_growth),
            'chi2_prior': float(chi2_prior),
            'chi2_posterior_effective': float(base['chi2_total'] + chi2_growth + chi2_prior),
        })
        return out

    def evaluate_tuo(self, tuo_params: Dict, *, h: float, omega_b: float, omega_c: float,
                     sigma8_mean: Optional[float] = None, sigma8_sigma: Optional[float] = None,
                     core: Optional[TUOPhase4CoreOptimized] = None,
                     bg_npts: int = 12000, zmax_bg: float = 1e5, z_growth_start: float = 100.0,
                     final_backend: str = 'numba', use_numba_shoot: bool = True) -> Dict[str, float]:
        res = self.base.evaluate_tuo(tuo_params, h=h, omega_b=omega_b, omega_c=omega_c,
                                     core=core, bg_npts=bg_npts, zmax_bg=zmax_bg,
                                     final_backend=final_backend, use_numba_shoot=use_numba_shoot)
        dens = derive_background_densities(h=h, omega_b=omega_b, omega_c=omega_c)
        params = dict(tuo_params)
        params.update(dens)
        if core is None:
            core = TUOPhase4CoreOptimized(params)
        else:
            core.update(**params)
        OmV, sol = core.solve(final_backend=final_backend, use_numba_shoot=use_numba_shoot,
                              grid_npts=bg_npts if final_backend == 'numba' else None)
        bg = core.sample_background(
            OmV, sol,
            z_max=max(zmax_bg, float(np.max(self.growth_dataset.z)) + 5.0, z_growth_start),
            npts=bg_npts,
        )
        beta_eff = _beta_eff_array(bg['N'], float(tuo_params['beta0']), float(tuo_params['z_c']), float(tuo_params['s']))
        growth = solve_linear_growth(bg, Omega_b0=dens['Omega_b0'], Omega_c0=dens['Omega_c0'], sigma8_0=1.0,
                                     z_start=z_growth_start, beta_eff=beta_eff, y_field=bg['y'])
        fs8_basis = _interp_basis(growth, self.growth_dataset.z)
        sigma8_best, chi2_growth, chi2_sigma8_prior = fit_sigma8_linear(
            self.growth_dataset, fs8_basis, sigma8_mean=sigma8_mean, sigma8_sigma=sigma8_sigma
        )
        out = dict(res)
        out.update({
            'sigma8_0_hat': float(sigma8_best),
            'chi2_growth': float(chi2_growth),
            'chi2_total': float(res['chi2_total'] + chi2_growth),
        })
        return out


def pick_bao_dataset(name: str):
    if name == 'desi_dr2':
        return builtin_desi_dr2_bao()
    if name == 'eboss_dr16_diag':
        return builtin_eboss_dr16_diagonal()
    raise ValueError(f'Unknown BAO dataset: {name}')


def pick_growth_dataset(name: str, growth_csv: Optional[Path], growth_cov: Optional[Path] = None):
    if growth_csv is not None:
        return load_growth_csv(growth_csv, cov_path=growth_cov)
    if name == 'minimal_diag':
        return builtin_rsd_diag_minimal()
    if name == 'boss_eboss_diag':
        return builtin_boss_eboss_diag()
    raise ValueError(f'Unknown growth dataset: {name}')


def main():
    ap = argparse.ArgumentParser(description='TUO Fase 5C joint likelihood (Pantheon+ + BAO + Growth)')
    ap.add_argument('--data', type=Path, default=Path('Pantheon+SH0ES.dat'))
    ap.add_argument('--cov', type=Path, default=Path('Pantheon+SH0ES_STAT+SYS.cov'))
    ap.add_argument('--bao', choices=['desi_dr2', 'eboss_dr16_diag'], default='desi_dr2')
    ap.add_argument('--growth', choices=['minimal_diag', 'boss_eboss_diag'], default='minimal_diag')
    ap.add_argument('--growth-csv', type=Path, default=None)
    ap.add_argument('--growth-cov', type=Path, default=None, help='Optional covariance matrix for growth CSV')
    ap.add_argument('--zmin', type=float, default=0.01)
    ap.add_argument('--keep-calibrators', action='store_true')
    ap.add_argument('--use-hubble-flow-flag', action='store_true')
    ap.add_argument('--bg-npts', type=int, default=8000)
    ap.add_argument('--final-backend', choices=['scipy', 'numba'], default='numba')
    ap.add_argument('--no-numba-shoot', action='store_true')
    ap.add_argument('--h-mean', type=float, default=0.674)
    ap.add_argument('--h-sigma', type=float, default=0.005)
    ap.add_argument('--omega-b-mean', dest='omega_b_mean', type=float, default=0.02237)
    ap.add_argument('--omega-b-sigma', dest='omega_b_sigma', type=float, default=0.00015)
    ap.add_argument('--omega-c-mean', dest='omega_c_mean', type=float, default=0.1200)
    ap.add_argument('--omega-c-sigma', dest='omega_c_sigma', type=float, default=0.0012)
    ap.add_argument('--sigma8-mean', dest='sigma8_mean', type=float, default=None)
    ap.add_argument('--sigma8-sigma', dest='sigma8_sigma', type=float, default=None)
    # smoke-point params
    ap.add_argument('--beta0', type=float, default=0.011)
    ap.add_argument('--nu2', type=float, default=0.008)
    ap.add_argument('--lam4', type=float, default=0.0)
    ap.add_argument('--xi-r', dest='xi_r', type=float, default=10.0)
    ap.add_argument('--z-c', dest='z_c', type=float, default=100.0)
    ap.add_argument('--s', type=float, default=4.0)
    ap.add_argument('--x-ini', dest='x_ini', type=float, default=0.03)
    ap.add_argument('--y-ini', dest='y_ini', type=float, default=0.0)
    ap.add_argument('--z-ini', dest='z_ini', type=float, default=1e5)
    ap.add_argument('--h', type=float, default=0.679)
    ap.add_argument('--omega-b', dest='omega_b', type=float, default=0.02237)
    ap.add_argument('--omega-c', dest='omega_c', type=float, default=0.1198)
    ap.add_argument('--json-out', type=Path, default=None)
    args = ap.parse_args()

    df, cov = load_pantheon(args.data, args.cov)
    df_sel, cov_sel, _ = apply_selection(df, cov, zmin=args.zmin,
                                         drop_calibrators=not args.keep_calibrators,
                                         use_hubble_flow_flag=args.use_hubble_flow_flag)
    sn_like = PantheonLikelihoodOptimized(df_sel, cov_sel)
    bao_dataset = pick_bao_dataset(args.bao)
    growth_dataset = pick_growth_dataset(args.growth, args.growth_csv, args.growth_cov)
    joint = PantheonBAOGrowthJointLikelihood5C(sn_like, bao_dataset, growth_dataset)

    lcdm = joint.fit_lcdm(h_mean=args.h_mean, h_sigma=args.h_sigma,
                          omega_b_mean=args.omega_b_mean, omega_b_sigma=args.omega_b_sigma,
                          omega_c_mean=args.omega_c_mean, omega_c_sigma=args.omega_c_sigma,
                          sigma8_mean=args.sigma8_mean, sigma8_sigma=args.sigma8_sigma,
                          bg_npts=args.bg_npts)

    tuo_params = {
        'beta0': args.beta0, 'nu2': args.nu2, 'lam4': args.lam4, 'xi_r': args.xi_r,
        'z_c': args.z_c, 's': args.s, 'x_ini': args.x_ini, 'y_ini': args.y_ini, 'z_ini': args.z_ini,
    }
    tuo = joint.evaluate_tuo(tuo_params, h=args.h, omega_b=args.omega_b, omega_c=args.omega_c,
                             sigma8_mean=args.sigma8_mean, sigma8_sigma=args.sigma8_sigma,
                             bg_npts=args.bg_npts, final_backend=args.final_backend,
                             use_numba_shoot=not args.no_numba_shoot)
    out = {
        'selection': {
            'n_total': int(len(df)),
            'n_selected': int(len(df_sel)),
            'bao_dataset': bao_dataset.name,
            'growth_dataset': growth_dataset.name,
        },
        'lcdm': lcdm,
        'tuo_point': tuo,
        'delta_chi2_tuo_minus_lcdm': float(tuo['chi2_total'] - lcdm['chi2_total']),
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
