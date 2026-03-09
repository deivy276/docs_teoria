import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from scipy.optimize import minimize_scalar

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


class PantheonBAOJointLikelihood:
    def __init__(self, pantheon_like: PantheonLikelihoodOptimized,
                 bao_like: BAOLikelihood):
        self.sn = pantheon_like
        self.bao = bao_like

    def fit_lcdm(self, *, h: float = 0.674, Omega_b0: float = 0.05, Omega_c0_guess: float = 0.25,
                 Omega_gamma0: float = 5.38e-5, zmax_bg: float = 1e5, bg_npts: int = 8000):
        dataset_zmax = float(np.max(self.bao.dataset.z))

        def obj(om):
            if om <= 0.05 or om >= 0.6:
                return np.inf
            # SN piece
            dl = dimensionless_dl_from_E(self.sn.z, lambda zz: lcdm_E(zz, om), zmax=float(np.max(self.sn.z)))
            mu = mu_from_dimensionless_dl(dl)
            chi2_sn, _ = self.sn.chi2_from_mu(mu)
            # BAO piece from background grid
            Egrid_model = TUOPhase4CoreOptimized({
                'Omega_b0': Omega_b0,
                'Omega_c0': max(om - Omega_b0, 1e-6),
                'Omega_r0': 9.2e-5,
                'beta0': 0.0,
                'nu2': 0.01,
                'lam4': 0.0,
                'xi_r': 0.0,
                'z_c': 100.0,
                's': 4.0,
                'x_ini': 0.0,
                'y_ini': 0.0,
                'z_ini': zmax_bg,
            })
            # For LCDM the scalar sector is frozen; set Omega_V0 by closure today analytically
            OmV = 1.0 - om - 9.2e-5
            # Build synthetic bg directly from E(z)
            z_bg = np.expm1(np.linspace(0.0, np.log1p(zmax_bg), bg_npts))
            E_bg = np.sqrt(om * (1.0 + z_bg) ** 3 + 9.2e-5 * (1.0 + z_bg) ** 4 + OmV)
            bg = {'z': z_bg, 'E': E_bg}
            chi2_bao = self.bao.evaluate(bg, Omega_b0=Omega_b0, Omega_c0=max(om - Omega_b0, 1e-6), Omega_gamma0=Omega_gamma0)['chi2_bao']
            return chi2_sn + chi2_bao

        opt = minimize_scalar(obj, bounds=(0.1, 0.5), method='bounded', options={'xatol':1e-4})
        om_best = float(opt.x)
        # recompute summary
        dl = dimensionless_dl_from_E(self.sn.z, lambda zz: lcdm_E(zz, om_best), zmax=float(np.max(self.sn.z)))
        mu = mu_from_dimensionless_dl(dl)
        chi2_sn, Mhat = self.sn.chi2_from_mu(mu)
        OmV = 1.0 - om_best - 9.2e-5
        z_bg = np.expm1(np.linspace(0.0, np.log1p(zmax_bg), bg_npts))
        E_bg = np.sqrt(om_best * (1.0 + z_bg) ** 3 + 9.2e-5 * (1.0 + z_bg) ** 4 + OmV)
        bg = {'z': z_bg, 'E': E_bg}
        bao_eval = self.bao.evaluate(bg, Omega_b0=Omega_b0, Omega_c0=max(om_best - Omega_b0, 1e-6), Omega_gamma0=Omega_gamma0)
        return {
            'omega_m': om_best,
            'chi2_total': float(chi2_sn + bao_eval['chi2_bao']),
            'chi2_sn': float(chi2_sn),
            'chi2_bao': float(bao_eval['chi2_bao']),
            'Mhat': float(Mhat),
            'r_d_Mpc': bao_eval['r_d_Mpc'],
            'z_drag': bao_eval['z_drag'],
        }

    def evaluate_tuo(self, tuo_params: Dict, *, core: Optional[TUOPhase4CoreOptimized] = None,
                     bg_npts: int = 12000, zmax_bg: float = 1e5,
                     final_backend: str = 'numba', use_numba_shoot: bool = True):
        if core is None:
            core = TUOPhase4CoreOptimized(tuo_params)
        else:
            core.update(**tuo_params)
        OmV, sol = core.solve(final_backend=final_backend, use_numba_shoot=use_numba_shoot,
                              grid_npts=bg_npts if final_backend == 'numba' else None)
        bg = core.sample_background(OmV, sol, z_max=max(zmax_bg, float(np.max(self.sn.z)), float(np.max(self.bao.dataset.z)) + 5.0), npts=bg_npts)
        # SN
        z_bg = bg['z']; E_bg = bg['E']
        order = np.argsort(z_bg)
        z_sorted = z_bg[order]
        E_sorted = E_bg[order]
        dL = dimensionless_dl_from_E(self.sn.z, lambda zz: np.interp(zz, z_sorted, E_sorted), zmax=float(np.max(self.sn.z)), ngrid=min(int(bg_npts), 20000))
        mu = mu_from_dimensionless_dl(dL)
        chi2_sn, Mhat = self.sn.chi2_from_mu(mu)
        # BAO
        bao_eval = self.bao.evaluate(bg, Omega_b0=core.Omega_b0, Omega_c0=core.Omega_c0, Omega_gamma0=core.Omega_gamma0)
        diag = core.diagnostics(bg)
        return {
            'Omega_V0': float(OmV),
            'chi2_sn': float(chi2_sn),
            'chi2_bao': float(bao_eval['chi2_bao']),
            'chi2_total': float(chi2_sn + bao_eval['chi2_bao']),
            'Mhat': float(Mhat),
            'r_d_Mpc': float(bao_eval['r_d_Mpc']),
            'z_drag': float(bao_eval['z_drag']),
            **{k: (None if v is None else float(v)) for k, v in diag.items()},
        }


def main():
    ap = argparse.ArgumentParser(description='Phase 5 joint Pantheon+ + BAO likelihood for TUO')
    ap.add_argument('--data', type=Path, default=Path('Pantheon+SH0ES.dat'))
    ap.add_argument('--cov', type=Path, default=Path('Pantheon+SH0ES_STAT+SYS.cov'))
    ap.add_argument('--bao', choices=['desi_dr2', 'eboss_dr16_diag'], default='desi_dr2')
    ap.add_argument('--zmin', type=float, default=0.01)
    ap.add_argument('--keep-calibrators', action='store_true')
    ap.add_argument('--use-hubble-flow-flag', action='store_true')
    ap.add_argument('--h', type=float, default=0.674)
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
    ap.add_argument('--Omega-b0', type=float, default=0.05, dest='Omega_b0')
    ap.add_argument('--Omega-c0', type=float, default=0.25, dest='Omega_c0')
    ap.add_argument('--Omega-r0', type=float, default=9.2e-5, dest='Omega_r0')
    ap.add_argument('--bg-npts', type=int, default=12000)
    ap.add_argument('--final-backend', choices=['scipy', 'numba'], default='numba')
    ap.add_argument('--no-numba-shoot', action='store_true')
    ap.add_argument('--json-out', type=Path, default=None)
    args = ap.parse_args()

    df, cov = load_pantheon(args.data, args.cov)
    df_sel, cov_sel, _ = apply_selection(df, cov, zmin=args.zmin,
                                         drop_calibrators=not args.keep_calibrators,
                                         use_hubble_flow_flag=args.use_hubble_flow_flag)
    sn_like = PantheonLikelihoodOptimized(df_sel, cov_sel)
    bao_like = BAOLikelihood(builtin_desi_dr2_bao() if args.bao == 'desi_dr2' else builtin_eboss_dr16_diagonal(), h=args.h)
    joint = PantheonBAOJointLikelihood(sn_like, bao_like)

    lcdm = joint.fit_lcdm(h=args.h, Omega_b0=args.Omega_b0, bg_npts=args.bg_npts)
    p = {
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
    core = TUOPhase4CoreOptimized(p)
    tuo = joint.evaluate_tuo(p, core=core, bg_npts=args.bg_npts, final_backend=args.final_backend, use_numba_shoot=not args.no_numba_shoot)
    out = {
        'selection': {
            'n_total': int(len(df)),
            'n_selected': int(len(df_sel)),
            'zmin': args.zmin,
            'drop_calibrators': not args.keep_calibrators,
            'use_hubble_flow_flag': args.use_hubble_flow_flag,
            'bao_dataset': bao_like.dataset.name,
        },
        'lcdm': lcdm,
        'tuo': tuo,
        'tuo_params': p,
        'delta_chi2_tuo_minus_lcdm': float(tuo['chi2_total'] - lcdm['chi2_total']),
    }
    txt = json.dumps(out, indent=2)
    print(txt)
    if args.json_out is not None:
        args.json_out.write_text(txt)


if __name__ == '__main__':
    main()
