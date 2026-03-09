
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
from tuo_growth_module_5c11 import (
    GrowthLikelihood5C11,
    builtin_rsd_diag_minimal,
    builtin_boss_eboss_diag,
    builtin_eboss_dr16_compressed,
    load_growth_csv,
    build_lcdm_reference_background,
    solve_linear_growth_5c11,
    validate_lcdm_limit,
)


class PantheonBAOGrowthJointLikelihood5C11:
    def __init__(self, pantheon_like: PantheonLikelihoodOptimized, bao_dataset, growth_dataset):
        self.base = PantheonBAOJointLikelihood5B(pantheon_like, bao_dataset)
        self.growth_dataset = growth_dataset
        self.growth_like = GrowthLikelihood5C11(growth_dataset)

    def priors_chi2(self, *, h: float, omega_b: float, omega_c: float,
                    h_mean: float, h_sigma: float,
                    omega_b_mean: float, omega_b_sigma: float,
                    omega_c_mean: float, omega_c_sigma: float,
                    beta0: Optional[float] = None,
                    beta0_mean: Optional[float] = None,
                    beta0_sigma: Optional[float] = None) -> float:
        chi2 = self.base.priors_chi2(
            h=h, omega_b=omega_b, omega_c=omega_c,
            h_mean=h_mean, h_sigma=h_sigma,
            omega_b_mean=omega_b_mean, omega_b_sigma=omega_b_sigma,
            omega_c_mean=omega_c_mean, omega_c_sigma=omega_c_sigma,
        )
        if beta0 is not None and beta0_mean is not None and beta0_sigma is not None and beta0_sigma > 0.0:
            chi2 += ((float(beta0) - float(beta0_mean)) / float(beta0_sigma)) ** 2
        return float(chi2)

    def _lcdm_growth(self, *, h: float, omega_b: float, omega_c: float,
                     bg_npts: int = 12000, z_growth_start: float = 100.0,
                     As: float = 2.1e-9, ns: float = 0.965,
                     k_pivot_hmpc: float = 0.1):
        dens = derive_background_densities(h=h, omega_b=omega_b, omega_c=omega_c)
        zmax = max(float(z_growth_start), float(np.max(self.growth_dataset.z)) + 5.0, 1e3)
        bg = build_lcdm_reference_background(
            h=h, Omega_b0=dens['Omega_b0'], Omega_c0=dens['Omega_c0'], Omega_r0=dens['Omega_r0'],
            z_max=zmax, npts=bg_npts
        )
        growth = solve_linear_growth_5c11(
            bg,
            Omega_b0=dens['Omega_b0'], Omega_c0=dens['Omega_c0'],
            h=h, omega_b=omega_b, omega_c=omega_c,
            beta0=0.0, nu2=0.0, lam4=0.0, xi_r=0.0,
            As=As, ns=ns,
            k_eff_hmpc=k_pivot_hmpc,
            k_values_hmpc=np.unique(np.concatenate([self.growth_dataset.k_eff, np.array([k_pivot_hmpc])])).astype(float),
            z_start=z_growth_start,
        )
        return dens, bg, growth

    def fit_lcdm(self, *, h_mean: float = 0.674, h_sigma: float = 0.005,
                 omega_b_mean: float = 0.02237, omega_b_sigma: float = 0.00015,
                 omega_c_mean: float = 0.1200, omega_c_sigma: float = 0.0012,
                 bg_npts: int = 8000, z_growth_start: float = 100.0,
                 bounds: Optional[Tuple[Tuple[float, float], ...]] = None,
                 As: float = 2.1e-9, ns: float = 0.965,
                 k_pivot_hmpc: float = 0.1):
        base = self.base.fit_lcdm(
            h_mean=h_mean, h_sigma=h_sigma,
            omega_b_mean=omega_b_mean, omega_b_sigma=omega_b_sigma,
            omega_c_mean=omega_c_mean, omega_c_sigma=omega_c_sigma,
            bg_npts=bg_npts, bounds=bounds
        )
        dens, bg, growth = self._lcdm_growth(
            h=base['h'], omega_b=base['omega_b'], omega_c=base['omega_c'],
            bg_npts=bg_npts, z_growth_start=z_growth_start,
            As=As, ns=ns, k_pivot_hmpc=k_pivot_hmpc
        )
        grow = self.growth_like.evaluate(growth)
        chi2_growth = float(grow['chi2_growth'])
        chi2_prior = self.priors_chi2(
            h=base['h'], omega_b=base['omega_b'], omega_c=base['omega_c'],
            h_mean=h_mean, h_sigma=h_sigma,
            omega_b_mean=omega_b_mean, omega_b_sigma=omega_b_sigma,
            omega_c_mean=omega_c_mean, omega_c_sigma=omega_c_sigma
        )
        out = dict(base)
        out.update({
            'chi2_growth': chi2_growth,
            'chi2_total': float(base['chi2_total'] + chi2_growth),
            'chi2_prior': float(chi2_prior),
            'chi2_posterior_effective': float(base['chi2_total'] + chi2_growth + chi2_prior),
            'sigma8_0_pred': float(growth['sigma8_0_pred']),
            'S8_0_pred': float(growth['S8_0_pred']),
            'fs8_pred': grow['fs8_th'].tolist(),
        })
        return out

    def evaluate_tuo(self, tuo_params: Dict, *, h: float, omega_b: float, omega_c: float,
                     core: Optional[TUOPhase4CoreOptimized] = None,
                     bg_npts: int = 12000, zmax_bg: float = 1e5, z_growth_start: float = 100.0,
                     final_backend: str = 'numba', use_numba_shoot: bool = True,
                     As: float = 2.1e-9, ns: float = 0.965,
                     k_pivot_hmpc: float = 0.1) -> Dict[str, float]:
        res = self.base.evaluate_tuo(
            tuo_params, h=h, omega_b=omega_b, omega_c=omega_c,
            core=core, bg_npts=bg_npts, zmax_bg=zmax_bg,
            final_backend=final_backend, use_numba_shoot=use_numba_shoot
        )
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
            npts=bg_npts
        )
        growth = solve_linear_growth_5c11(
            bg,
            Omega_b0=dens['Omega_b0'], Omega_c0=dens['Omega_c0'],
            h=h, omega_b=omega_b, omega_c=omega_c,
            beta0=float(tuo_params['beta0']), nu2=float(tuo_params['nu2']),
            lam4=float(tuo_params['lam4']), xi_r=float(tuo_params['xi_r']),
            As=As, ns=ns,
            k_pivot_hmpc=k_pivot_hmpc,
            k_values_hmpc=np.unique(np.concatenate([self.growth_dataset.k_eff, np.array([k_pivot_hmpc])])).astype(float),
            z_start=z_growth_start
        )
        grow = self.growth_like.evaluate(growth)
        out = dict(res)
        out.update({
            'chi2_growth': float(grow['chi2_growth']),
            'chi2_total': float(res['chi2_total'] + grow['chi2_growth']),
            'sigma8_0_pred': float(growth['sigma8_0_pred']),
            'S8_0_pred': float(growth['S8_0_pred']),
            'fs8_pred': grow['fs8_th'].tolist(),
        })
        return out


def pick_bao_dataset(name: str):
    if name == 'desi_dr2':
        return builtin_desi_dr2_bao()
    if name == 'eboss_dr16_diag':
        return builtin_eboss_dr16_diagonal()
    raise ValueError(f'Unknown BAO dataset: {name}')


def pick_growth_dataset(name: str, growth_csv: Optional[Path], growth_cov: Optional[Path], k_pivot_hmpc: float = 0.10):
    if growth_csv is not None:
        return load_growth_csv(growth_csv, cov_path=growth_cov, k_pivot_hmpc=k_pivot_hmpc)
    if name == 'minimal_diag':
        return builtin_rsd_diag_minimal(k_pivot_hmpc=k_pivot_hmpc)
    if name == 'boss_eboss_diag':
        return builtin_boss_eboss_diag(k_pivot_hmpc=k_pivot_hmpc)
    if name == 'eboss_dr16_compressed':
        return builtin_eboss_dr16_compressed(k_pivot_hmpc=k_pivot_hmpc)
    raise ValueError(f'Unknown growth dataset: {name}')


def main():
    ap = argparse.ArgumentParser(description='TUO Fase 5C.1.1 joint likelihood (Pantheon+ + BAO + Growth)')
    ap.add_argument('--data', type=Path, default=Path('Pantheon+SH0ES.dat'))
    ap.add_argument('--cov', type=Path, default=Path('Pantheon+SH0ES_STAT+SYS.cov'))
    ap.add_argument('--bao', choices=['desi_dr2', 'eboss_dr16_diag'], default='desi_dr2')
    ap.add_argument('--growth', choices=['minimal_diag', 'boss_eboss_diag', 'eboss_dr16_compressed'], default='minimal_diag')
    ap.add_argument('--growth-csv', type=Path, default=None)
    ap.add_argument('--growth-cov', type=Path, default=None)
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
    ap.add_argument('--As', type=float, default=2.1e-9)
    ap.add_argument('--ns', type=float, default=0.965)
    ap.add_argument('--k-pivot', dest='k_pivot', type=float, default=0.10)
    # smoke TUO point
    ap.add_argument('--beta0', type=float, default=0.011)
    ap.add_argument('--nu2', type=float, default=0.008)
    ap.add_argument('--lam4', type=float, default=0.0)
    ap.add_argument('--xi-r', dest='xi_r', type=float, default=10.0)
    ap.add_argument('--z-c', dest='z_c', type=float, default=100.0)
    ap.add_argument('--s', type=float, default=4.0)
    ap.add_argument('--x-ini', dest='x_ini', type=float, default=0.03)
    ap.add_argument('--y-ini', dest='y_ini', type=float, default=0.0)
    ap.add_argument('--validate-lcdm', action='store_true')
    args = ap.parse_args()

    df, covm = load_pantheon(args.data, args.cov)
    df_sel, cov_sel, _ = apply_selection(df, covm, zmin=args.zmin,
                                         drop_calibrators=not args.keep_calibrators,
                                         use_hubble_flow_flag=args.use_hubble_flow_flag)
    sn_like = PantheonLikelihoodOptimized(df_sel, cov_sel)
    bao_dataset = pick_bao_dataset(args.bao)
    growth_dataset = pick_growth_dataset(args.growth, args.growth_csv, args.growth_cov, k_pivot_hmpc=args.k_pivot)
    joint = PantheonBAOGrowthJointLikelihood5C11(sn_like, bao_dataset, growth_dataset)

    lcdm = joint.fit_lcdm(
        h_mean=args.h_mean, h_sigma=args.h_sigma,
        omega_b_mean=args.omega_b_mean, omega_b_sigma=args.omega_b_sigma,
        omega_c_mean=args.omega_c_mean, omega_c_sigma=args.omega_c_sigma,
        bg_npts=args.bg_npts, As=args.As, ns=args.ns, k_pivot_hmpc=args.k_pivot
    )

    tuo_params = {
        'beta0': args.beta0, 'nu2': args.nu2, 'lam4': args.lam4, 'xi_r': args.xi_r,
        'z_c': args.z_c, 's': args.s, 'x_ini': args.x_ini, 'y_ini': args.y_ini,
    }
    core = TUOPhase4CoreOptimized({
        **tuo_params,
        **derive_background_densities(h=args.h_mean, omega_b=args.omega_b_mean, omega_c=args.omega_c_mean)
    })
    tuo = joint.evaluate_tuo(
        tuo_params, h=args.h_mean, omega_b=args.omega_b_mean, omega_c=args.omega_c_mean,
        core=core, bg_npts=args.bg_npts, final_backend=args.final_backend,
        use_numba_shoot=not args.no_numba_shoot,
        As=args.As, ns=args.ns, k_pivot_hmpc=args.k_pivot
    )

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

    if args.validate_lcdm:
        dens = derive_background_densities(h=args.h_mean, omega_b=args.omega_b_mean, omega_c=args.omega_c_mean)
        bg_ref = build_lcdm_reference_background(
            h=args.h_mean, Omega_b0=dens['Omega_b0'], Omega_c0=dens['Omega_c0'], Omega_r0=dens['Omega_r0'],
            z_max=max(1e3, float(np.max(growth_dataset.z)) + 5.0), npts=max(args.bg_npts, 4000)
        )
        out['lcdm_validation'] = validate_lcdm_limit(
            bg_ref,
            Omega_b0=dens['Omega_b0'], Omega_c0=dens['Omega_c0'],
            h=args.h_mean, omega_b=args.omega_b_mean, omega_c=args.omega_c_mean,
            z_start=100.0, k_pivot_hmpc=args.k_pivot,
        )

    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
