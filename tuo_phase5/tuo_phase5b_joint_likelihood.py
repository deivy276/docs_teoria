import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from tuo_phase4_core_optimized import TUOPhase4CoreOptimized
from tuo_pantheon_fit_optimized import (
    PantheonLikelihoodOptimized,
    apply_selection,
    load_pantheon,
    lcdm_E,
    dimensionless_dl_from_E,
    mu_from_dimensionless_dl,
)
from tuo_bao_module import BAOLikelihood, builtin_desi_dr2_bao, builtin_eboss_dr16_diagonal

# Physical radiation constants (Tcmb ~ 2.7255 K, Neff fixed)
OMEGA_GAMMA_PHYSICAL = 2.469e-5
NEFF_DEFAULT = 3.046
RADIATION_FACTOR = 1.0 + 0.22710731766 * NEFF_DEFAULT
OMEGA_R_PHYSICAL = OMEGA_GAMMA_PHYSICAL * RADIATION_FACTOR


def derive_background_densities(*, h: float, omega_b: float, omega_c: float,
                                omega_gamma_physical: float = OMEGA_GAMMA_PHYSICAL,
                                omega_r_physical: float = OMEGA_R_PHYSICAL) -> Dict[str, float]:
    h2 = float(h) ** 2
    if h2 <= 0.0:
        raise ValueError('h must be positive')
    return {
        'Omega_b0': float(omega_b) / h2,
        'Omega_c0': float(omega_c) / h2,
        'Omega_gamma0': float(omega_gamma_physical) / h2,
        'Omega_r0': float(omega_r_physical) / h2,
    }


class PantheonBAOJointLikelihood5B:
    """Joint likelihood Pantheon+ + BAO with h, ω_b, ω_c free.

    The data χ² is always separated from Gaussian prior penalties. The posterior used by the
    sampler can then be built as logpost = -0.5*(chi2_total + chi2_prior + optional extra priors).
    """

    def __init__(self, pantheon_like: PantheonLikelihoodOptimized, bao_dataset,
                 *, omega_gamma_physical: float = OMEGA_GAMMA_PHYSICAL,
                 omega_r_physical: float = OMEGA_R_PHYSICAL):
        self.sn = pantheon_like
        self.bao_dataset = bao_dataset
        self.omega_gamma_physical = float(omega_gamma_physical)
        self.omega_r_physical = float(omega_r_physical)

    def _bao_like(self, h: float) -> BAOLikelihood:
        return BAOLikelihood(self.bao_dataset, h=float(h))

    def priors_chi2(self, *, h: float, omega_b: float, omega_c: float,
                    h_mean: float, h_sigma: float,
                    omega_b_mean: float, omega_b_sigma: float,
                    omega_c_mean: float, omega_c_sigma: float) -> float:
        out = 0.0
        out += ((float(h) - h_mean) / h_sigma) ** 2
        out += ((float(omega_b) - omega_b_mean) / omega_b_sigma) ** 2
        out += ((float(omega_c) - omega_c_mean) / omega_c_sigma) ** 2
        return float(out)

    def fit_lcdm(self, *, h_mean: float = 0.674, h_sigma: float = 0.005,
                 omega_b_mean: float = 0.02237, omega_b_sigma: float = 0.00015,
                 omega_c_mean: float = 0.1200, omega_c_sigma: float = 0.0012,
                 bg_npts: int = 8000, zmax_bg: float = 1e5,
                 bounds: Optional[Tuple[Tuple[float, float], ...]] = None):
        """Fit a baseline flat-ΛCDM with free (h, ω_b, ω_c) and Gaussian priors.

        Ω_m is derived from ω_b + ω_c and h.
        """
        if bounds is None:
            bounds = ((0.55, 0.85), (0.020, 0.0255), (0.09, 0.15))

        zmax_sn = float(np.max(self.sn.z))
        zmax_bg = max(float(zmax_bg), zmax_sn, float(np.max(self.bao_dataset.z)) + 5.0)

        def obj(x):
            h, omega_b, omega_c = map(float, x)
            try:
                dens = derive_background_densities(h=h, omega_b=omega_b, omega_c=omega_c,
                                                   omega_gamma_physical=self.omega_gamma_physical,
                                                   omega_r_physical=self.omega_r_physical)
            except Exception:
                return np.inf
            om = dens['Omega_b0'] + dens['Omega_c0']
            orad = dens['Omega_r0']
            ov = 1.0 - om - orad
            if om <= 0.0 or ov <= 0.0:
                return np.inf

            dl = dimensionless_dl_from_E(self.sn.z,
                                         lambda zz: np.sqrt(om * (1.0 + zz) ** 3 + orad * (1.0 + zz) ** 4 + ov),
                                         zmax=zmax_sn)
            mu = mu_from_dimensionless_dl(dl)
            chi2_sn, _ = self.sn.chi2_from_mu(mu)

            z_bg = np.expm1(np.linspace(0.0, np.log1p(zmax_bg), bg_npts))
            E_bg = np.sqrt(om * (1.0 + z_bg) ** 3 + orad * (1.0 + z_bg) ** 4 + ov)
            bg = {'z': z_bg, 'E': E_bg}
            bao_like = self._bao_like(h)
            bao_eval = bao_like.evaluate(bg, Omega_b0=dens['Omega_b0'], Omega_c0=dens['Omega_c0'],
                                         Omega_gamma0=dens['Omega_gamma0'])
            chi2_prior = self.priors_chi2(h=h, omega_b=omega_b, omega_c=omega_c,
                                          h_mean=h_mean, h_sigma=h_sigma,
                                          omega_b_mean=omega_b_mean, omega_b_sigma=omega_b_sigma,
                                          omega_c_mean=omega_c_mean, omega_c_sigma=omega_c_sigma)
            return float(chi2_sn + bao_eval['chi2_bao'] + chi2_prior)

        x0 = np.array([h_mean, omega_b_mean, omega_c_mean], dtype=float)
        opt = minimize(obj, x0=x0, method='L-BFGS-B', bounds=bounds)
        h_best, omega_b_best, omega_c_best = map(float, opt.x)
        dens = derive_background_densities(h=h_best, omega_b=omega_b_best, omega_c=omega_c_best,
                                           omega_gamma_physical=self.omega_gamma_physical,
                                           omega_r_physical=self.omega_r_physical)
        om = dens['Omega_b0'] + dens['Omega_c0']
        orad = dens['Omega_r0']
        ov = 1.0 - om - orad
        dl = dimensionless_dl_from_E(self.sn.z,
                                     lambda zz: np.sqrt(om * (1.0 + zz) ** 3 + orad * (1.0 + zz) ** 4 + ov),
                                     zmax=zmax_sn)
        mu = mu_from_dimensionless_dl(dl)
        chi2_sn, Mhat = self.sn.chi2_from_mu(mu)
        z_bg = np.expm1(np.linspace(0.0, np.log1p(zmax_bg), bg_npts))
        E_bg = np.sqrt(om * (1.0 + z_bg) ** 3 + orad * (1.0 + z_bg) ** 4 + ov)
        bg = {'z': z_bg, 'E': E_bg}
        bao_like = self._bao_like(h_best)
        bao_eval = bao_like.evaluate(bg, Omega_b0=dens['Omega_b0'], Omega_c0=dens['Omega_c0'],
                                     Omega_gamma0=dens['Omega_gamma0'])
        chi2_prior = self.priors_chi2(h=h_best, omega_b=omega_b_best, omega_c=omega_c_best,
                                      h_mean=h_mean, h_sigma=h_sigma,
                                      omega_b_mean=omega_b_mean, omega_b_sigma=omega_b_sigma,
                                      omega_c_mean=omega_c_mean, omega_c_sigma=omega_c_sigma)
        return {
            'h': h_best,
            'omega_b': omega_b_best,
            'omega_c': omega_c_best,
            'Omega_m': om,
            'chi2_total': float(chi2_sn + bao_eval['chi2_bao']),
            'chi2_sn': float(chi2_sn),
            'chi2_bao': float(bao_eval['chi2_bao']),
            'chi2_prior': float(chi2_prior),
            'chi2_posterior_effective': float(chi2_sn + bao_eval['chi2_bao'] + chi2_prior),
            'Mhat': float(Mhat),
            'r_d_Mpc': float(bao_eval['r_d_Mpc']),
            'z_drag': float(bao_eval['z_drag']),
            'Omega_b0': dens['Omega_b0'],
            'Omega_c0': dens['Omega_c0'],
            'Omega_r0': dens['Omega_r0'],
        }

    def evaluate_tuo(self, tuo_params: Dict, *, h: float, omega_b: float, omega_c: float,
                     core: Optional[TUOPhase4CoreOptimized] = None,
                     bg_npts: int = 12000, zmax_bg: float = 1e5,
                     final_backend: str = 'numba', use_numba_shoot: bool = True) -> Dict[str, float]:
        dens = derive_background_densities(h=h, omega_b=omega_b, omega_c=omega_c,
                                           omega_gamma_physical=self.omega_gamma_physical,
                                           omega_r_physical=self.omega_r_physical)
        params = dict(tuo_params)
        params.update(dens)
        if core is None:
            core = TUOPhase4CoreOptimized(params)
        else:
            core.update(**params)
        OmV, sol = core.solve(final_backend=final_backend, use_numba_shoot=use_numba_shoot,
                              grid_npts=bg_npts if final_backend == 'numba' else None)
        bg = core.sample_background(OmV, sol,
                                    z_max=max(zmax_bg, float(np.max(self.sn.z)), float(np.max(self.bao_dataset.z)) + 5.0),
                                    npts=bg_npts)
        z_bg = bg['z']; E_bg = bg['E']
        order = np.argsort(z_bg)
        z_sorted = z_bg[order]
        E_sorted = E_bg[order]
        dL = dimensionless_dl_from_E(self.sn.z,
                                     lambda zz: np.interp(zz, z_sorted, E_sorted),
                                     zmax=float(np.max(self.sn.z)),
                                     ngrid=min(int(bg_npts), 20000))
        mu = mu_from_dimensionless_dl(dL)
        chi2_sn, Mhat = self.sn.chi2_from_mu(mu)
        bao_like = self._bao_like(h)
        bao_eval = bao_like.evaluate(bg, Omega_b0=dens['Omega_b0'], Omega_c0=dens['Omega_c0'],
                                     Omega_gamma0=dens['Omega_gamma0'])
        diag = core.diagnostics(bg)
        out = {
            'Omega_V0': float(OmV),
            'chi2_sn': float(chi2_sn),
            'chi2_bao': float(bao_eval['chi2_bao']),
            'chi2_total': float(chi2_sn + bao_eval['chi2_bao']),
            'Mhat': float(Mhat),
            'r_d_Mpc': float(bao_eval['r_d_Mpc']),
            'z_drag': float(bao_eval['z_drag']),
            'h': float(h),
            'omega_b': float(omega_b),
            'omega_c': float(omega_c),
            'Omega_b0': dens['Omega_b0'],
            'Omega_c0': dens['Omega_c0'],
            'Omega_r0': dens['Omega_r0'],
            'Omega_gamma0': dens['Omega_gamma0'],
            **{k: (None if v is None else float(v)) for k, v in diag.items()},
        }
        return out


def main():
    ap = argparse.ArgumentParser(description='Phase 5B joint Pantheon+ + BAO likelihood with free h, ω_b, ω_c')
    ap.add_argument('--data', type=Path, default=Path('Pantheon+SH0ES.dat'))
    ap.add_argument('--cov', type=Path, default=Path('Pantheon+SH0ES_STAT+SYS.cov'))
    ap.add_argument('--bao', choices=['desi_dr2', 'eboss_dr16_diag'], default='desi_dr2')
    ap.add_argument('--zmin', type=float, default=0.01)
    ap.add_argument('--keep-calibrators', action='store_true')
    ap.add_argument('--use-hubble-flow-flag', action='store_true')
    ap.add_argument('--bg-npts', type=int, default=12000)
    ap.add_argument('--final-backend', choices=['scipy', 'numba'], default='numba')
    ap.add_argument('--no-numba-shoot', action='store_true')
    # TUO params
    ap.add_argument('--beta0', type=float, default=0.02)
    ap.add_argument('--nu2', type=float, default=0.02)
    ap.add_argument('--lam4', type=float, default=0.0)
    ap.add_argument('--xi-r', type=float, default=10.0, dest='xi_r')
    ap.add_argument('--z-c', type=float, default=100.0, dest='z_c')
    ap.add_argument('--s', type=float, default=4.0)
    ap.add_argument('--x-ini', type=float, default=0.2, dest='x_ini')
    ap.add_argument('--y-ini', type=float, default=0.0, dest='y_ini')
    ap.add_argument('--z-ini', type=float, default=1e5, dest='z_ini')
    # free physical densities + Hubble
    ap.add_argument('--h', type=float, default=0.674)
    ap.add_argument('--omega-b', type=float, default=0.02237, dest='omega_b')
    ap.add_argument('--omega-c', type=float, default=0.1200, dest='omega_c')
    # Gaussian priors
    ap.add_argument('--h-mean', type=float, default=0.674)
    ap.add_argument('--h-sigma', type=float, default=0.005)
    ap.add_argument('--omega-b-mean', type=float, default=0.02237, dest='omega_b_mean')
    ap.add_argument('--omega-b-sigma', type=float, default=0.00015, dest='omega_b_sigma')
    ap.add_argument('--omega-c-mean', type=float, default=0.1200, dest='omega_c_mean')
    ap.add_argument('--omega-c-sigma', type=float, default=0.0012, dest='omega_c_sigma')
    ap.add_argument('--json-out', type=Path, default=None)
    args = ap.parse_args()

    df, cov = load_pantheon(args.data, args.cov)
    df_sel, cov_sel, _ = apply_selection(df, cov, zmin=args.zmin,
                                         drop_calibrators=not args.keep_calibrators,
                                         use_hubble_flow_flag=args.use_hubble_flow_flag)
    sn_like = PantheonLikelihoodOptimized(df_sel, cov_sel)
    bao_dataset = builtin_desi_dr2_bao() if args.bao == 'desi_dr2' else builtin_eboss_dr16_diagonal()
    joint = PantheonBAOJointLikelihood5B(sn_like, bao_dataset)

    lcdm = joint.fit_lcdm(h_mean=args.h_mean, h_sigma=args.h_sigma,
                          omega_b_mean=args.omega_b_mean, omega_b_sigma=args.omega_b_sigma,
                          omega_c_mean=args.omega_c_mean, omega_c_sigma=args.omega_c_sigma,
                          bg_npts=args.bg_npts)

    tuo_params = {
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
    core = TUOPhase4CoreOptimized({
        'Omega_b0': args.omega_b / args.h ** 2,
        'Omega_c0': args.omega_c / args.h ** 2,
        'Omega_r0': OMEGA_R_PHYSICAL / args.h ** 2,
        **tuo_params,
    })
    tuo = joint.evaluate_tuo(tuo_params, h=args.h, omega_b=args.omega_b, omega_c=args.omega_c,
                             core=core, bg_npts=args.bg_npts, final_backend=args.final_backend,
                             use_numba_shoot=not args.no_numba_shoot)
    chi2_prior = joint.priors_chi2(h=args.h, omega_b=args.omega_b, omega_c=args.omega_c,
                                   h_mean=args.h_mean, h_sigma=args.h_sigma,
                                   omega_b_mean=args.omega_b_mean, omega_b_sigma=args.omega_b_sigma,
                                   omega_c_mean=args.omega_c_mean, omega_c_sigma=args.omega_c_sigma)
    out = {
        'selection': {
            'n_total': int(len(df)),
            'n_selected': int(len(df_sel)),
            'zmin': args.zmin,
            'drop_calibrators': not args.keep_calibrators,
            'use_hubble_flow_flag': args.use_hubble_flow_flag,
            'bao_dataset': bao_dataset.name,
        },
        'priors': {
            'h': [args.h_mean, args.h_sigma],
            'omega_b': [args.omega_b_mean, args.omega_b_sigma],
            'omega_c': [args.omega_c_mean, args.omega_c_sigma],
        },
        'lcdm': lcdm,
        'tuo': {
            **tuo,
            'chi2_prior': float(chi2_prior),
            'chi2_posterior_effective': float(tuo['chi2_total'] + chi2_prior),
        },
        'tuo_params': {**tuo_params, 'h': args.h, 'omega_b': args.omega_b, 'omega_c': args.omega_c},
        'delta_chi2_tuo_minus_lcdm': float(tuo['chi2_total'] - lcdm['chi2_total']),
        'delta_chi2eff_tuo_minus_lcdm': float((tuo['chi2_total'] + chi2_prior) - lcdm['chi2_posterior_effective']),
    }
    txt = json.dumps(out, indent=2)
    print(txt)
    if args.json_out is not None:
        args.json_out.write_text(txt)


if __name__ == '__main__':
    main()
