
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import eigh

from tuo_phase4_core_optimized import TUOPhase4CoreOptimized
from tuo_pantheon_fit_optimized import load_pantheon, apply_selection, PantheonLikelihoodOptimized
from tuo_phase5c11_joint_likelihood import (
    PantheonBAOGrowthJointLikelihood5C11,
    pick_bao_dataset,
    pick_growth_dataset,
)
from tuo_phase5b_joint_likelihood import derive_background_densities

_WORKER: Dict[str, Any] = {}
_BLOB_COLS = [
    'chi2_total', 'chi2_sn', 'chi2_bao', 'chi2_growth', 'chi2_prior', 'Omega_V0',
    'z_eq', 'z_t', 'f_TUO_zstar', 'r_d_Mpc', 'z_drag', 'sigma8_0_pred', 'S8_0_pred',
    'h', 'omega_b', 'omega_c', 'Omega_b0', 'Omega_c0', 'Omega_r0'
]


def _reset_core_warm(core: TUOPhase4CoreOptimized):
    core._warm_root = None
    core._warm_bracket = None
    core.last_shoot_evals = 0


def init_worker(worker_cfg: Dict[str, Any]):
    global _WORKER
    data = Path(worker_cfg['data']); cov = Path(worker_cfg['cov'])
    zmin = float(worker_cfg['zmin'])
    drop_cal = bool(worker_cfg['drop_calibrators'])
    use_hf = bool(worker_cfg['use_hubble_flow_flag'])
    fixed = dict(worker_cfg['fixed'])
    bg_npts = int(worker_cfg['bg_npts'])
    final_backend = worker_cfg.get('final_backend', 'numba')
    use_numba_shoot = bool(worker_cfg.get('use_numba_shoot', True))
    f_tuo_max = float(worker_cfg.get('f_tuo_max', 1e-2))
    use_zeq_prior = bool(worker_cfg.get('use_zeq_prior', False))
    zeq_mean = float(worker_cfg.get('zeq_mean', 3402.0))
    zeq_sigma = float(worker_cfg.get('zeq_sigma', 26.0))
    bao_name = worker_cfg.get('bao_dataset', 'desi_dr2')
    growth_name = worker_cfg.get('growth_dataset', 'minimal_diag')
    growth_csv = worker_cfg.get('growth_csv', None)
    growth_cov = worker_cfg.get('growth_cov', None)
    k_pivot = float(worker_cfg.get('k_pivot', 0.10))
    priors = dict(worker_cfg['priors'])
    growth_cfg = dict(worker_cfg.get('growth_cfg', {}))

    df, covm = load_pantheon(data, cov)
    df_sel, cov_sel, _ = apply_selection(df, covm, zmin=zmin,
                                         drop_calibrators=drop_cal,
                                         use_hubble_flow_flag=use_hf)
    sn_like = PantheonLikelihoodOptimized(df_sel, cov_sel)
    bao_dataset = pick_bao_dataset(bao_name)
    growth_dataset = pick_growth_dataset(growth_name,
                                         Path(growth_csv) if growth_csv else None,
                                         Path(growth_cov) if growth_cov else None,
                                         k_pivot_hmpc=k_pivot)
    joint = PantheonBAOGrowthJointLikelihood5C11(sn_like, bao_dataset, growth_dataset)

    h0 = priors['h_mean']; ob0 = priors['omega_b_mean']; oc0 = priors['omega_c_mean']
    dens0 = derive_background_densities(h=h0, omega_b=ob0, omega_c=oc0)
    core = TUOPhase4CoreOptimized({
        'Omega_b0': dens0['Omega_b0'],
        'Omega_c0': dens0['Omega_c0'],
        'Omega_r0': dens0['Omega_r0'],
        **fixed,
    })

    _WORKER = {
        'joint': joint,
        'core': core,
        'fixed': fixed,
        'bg_npts': bg_npts,
        'final_backend': final_backend,
        'use_numba_shoot': use_numba_shoot,
        'f_tuo_max': f_tuo_max,
        'use_zeq_prior': use_zeq_prior,
        'zeq_mean': zeq_mean,
        'zeq_sigma': zeq_sigma,
        'selection': {
            'n_total': int(len(df)),
            'n_selected': int(len(df_sel)),
            'bao_dataset': bao_dataset.name,
            'growth_dataset': growth_dataset.name,
        },
        'priors': priors,
        'growth_cfg': growth_cfg,
        'k_pivot': k_pivot,
    }


def worker_logpost(theta: np.ndarray) -> Tuple[float, np.ndarray]:
    global _WORKER
    beta0, nu2, x_ini, h, omega_b, omega_c = map(float, theta)

    if not (-0.10 <= beta0 <= 0.10):
        return -np.inf, np.array([np.inf] + [np.nan] * (len(_BLOB_COLS) - 1), dtype=float)
    if not (0.005 <= nu2 <= 0.06):
        return -np.inf, np.array([np.inf] + [np.nan] * (len(_BLOB_COLS) - 1), dtype=float)
    if not (0.0 <= x_ini <= 0.6):
        return -np.inf, np.array([np.inf] + [np.nan] * (len(_BLOB_COLS) - 1), dtype=float)
    if not (0.55 <= h <= 0.85):
        return -np.inf, np.array([np.inf] + [np.nan] * (len(_BLOB_COLS) - 1), dtype=float)
    if not (0.020 <= omega_b <= 0.0255):
        return -np.inf, np.array([np.inf] + [np.nan] * (len(_BLOB_COLS) - 1), dtype=float)
    if not (0.09 <= omega_c <= 0.15):
        return -np.inf, np.array([np.inf] + [np.nan] * (len(_BLOB_COLS) - 1), dtype=float)

    pri = _WORKER['priors']
    fixed = _WORKER['fixed']
    params = {
        'beta0': beta0, 'nu2': nu2, 'x_ini': x_ini,
        'lam4': fixed['lam4'], 'xi_r': fixed['xi_r'],
        'z_c': fixed['z_c'], 's': fixed['s'], 'y_ini': fixed['y_ini'],
    }

    joint = _WORKER['joint']
    core = _WORKER['core']
    dens = derive_background_densities(h=h, omega_b=omega_b, omega_c=omega_c)
    core.update(Omega_b0=dens['Omega_b0'], Omega_c0=dens['Omega_c0'], Omega_r0=dens['Omega_r0'],
                beta0=beta0, nu2=nu2, x_ini=x_ini,
                lam4=fixed['lam4'], xi_r=fixed['xi_r'], z_c=fixed['z_c'], s=fixed['s'], y_ini=fixed['y_ini'])

    try:
        res = joint.evaluate_tuo(
            params, h=h, omega_b=omega_b, omega_c=omega_c,
            core=core,
            bg_npts=_WORKER['bg_npts'],
            final_backend=_WORKER['final_backend'],
            use_numba_shoot=_WORKER['use_numba_shoot'],
            As=_WORKER['growth_cfg']['As'], ns=_WORKER['growth_cfg']['ns'],
            k_pivot_hmpc=_WORKER['k_pivot'],
        )
    except Exception:
        return -np.inf, np.array([np.inf] + [np.nan] * (len(_BLOB_COLS) - 1), dtype=float)

    if not np.isfinite(res['chi2_total']):
        return -np.inf, np.array([np.inf] + [np.nan] * (len(_BLOB_COLS) - 1), dtype=float)
    if float(res.get('f_TUO_zstar', np.inf)) > _WORKER['f_tuo_max']:
        return -np.inf, np.array([np.inf] + [np.nan] * (len(_BLOB_COLS) - 1), dtype=float)

    chi2_prior = joint.priors_chi2(
        h=h, omega_b=omega_b, omega_c=omega_c,
        h_mean=pri['h_mean'], h_sigma=pri['h_sigma'],
        omega_b_mean=pri['omega_b_mean'], omega_b_sigma=pri['omega_b_sigma'],
        omega_c_mean=pri['omega_c_mean'], omega_c_sigma=pri['omega_c_sigma'],
        beta0=beta0 if pri.get('beta0_sigma') is not None else None,
        beta0_mean=pri.get('beta0_mean', None),
        beta0_sigma=pri.get('beta0_sigma', None),
    )

    if _WORKER['use_zeq_prior']:
        chi2_prior += ((float(res['z_eq']) - _WORKER['zeq_mean']) / _WORKER['zeq_sigma']) ** 2

    logpost = -0.5 * (float(res['chi2_total']) + float(chi2_prior))

    blob = np.array([
        float(res['chi2_total']), float(res['chi2_sn']), float(res['chi2_bao']), float(res['chi2_growth']),
        float(chi2_prior), float(res['Omega_V0']), float(res['z_eq']), float(res['z_t']),
        float(res['f_TUO_zstar']), float(res['r_d_Mpc']), float(res['z_drag']),
        float(res['sigma8_0_pred']), float(res['S8_0_pred']),
        float(h), float(omega_b), float(omega_c),
        float(res['Omega_b0']), float(res['Omega_c0']), float(res['Omega_r0']),
    ], dtype=float)
    return float(logpost), blob


def _run_chain_serial(args_tuple) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    rng_seed, init, nsteps, burn, adapt_every, proposal_cov = args_tuple
    rng = np.random.default_rng(int(rng_seed))
    d = init.size
    samples = np.empty((nsteps, d), dtype=float)
    logps = np.empty(nsteps, dtype=float)
    blobs = np.empty((nsteps, len(_BLOB_COLS)), dtype=float)

    theta = np.array(init, dtype=float)
    lp, blob = worker_logpost(theta)
    if not np.isfinite(lp):
        raise RuntimeError('Initial point has -inf posterior')
    accepted = 0
    proposal = np.array(proposal_cov, dtype=float)

    chain_hist = [theta.copy()]
    t0 = time.time()
    for i in range(nsteps):
        prop = rng.multivariate_normal(theta, proposal)
        lp_prop, blob_prop = worker_logpost(prop)
        if np.isfinite(lp_prop) and np.log(rng.random()) < (lp_prop - lp):
            theta = prop
            lp = lp_prop
            blob = blob_prop
            accepted += 1
        samples[i] = theta
        logps[i] = lp
        blobs[i] = blob
        chain_hist.append(theta.copy())
        if adapt_every > 0 and (i + 1) % adapt_every == 0 and (i + 1) > max(20, burn // 2):
            arr = np.array(chain_hist[max(0, len(chain_hist) - 200):], dtype=float)
            if arr.shape[0] > d + 5:
                proposal = np.cov(arr.T) + 1e-6 * np.eye(d)
                proposal *= (2.38 ** 2) / d
    runtime = time.time() - t0
    return samples, logps, blobs, accepted / float(nsteps), runtime


def gelman_rubin(chains: np.ndarray) -> np.ndarray:
    m, n, d = chains.shape
    means = np.mean(chains, axis=1)
    W = np.mean(np.var(chains, axis=1, ddof=1), axis=0)
    B = n * np.var(means, axis=0, ddof=1)
    var_hat = ((n - 1) / n) * W + (1 / n) * B
    Rhat = np.sqrt(var_hat / np.maximum(W, 1e-30))
    return Rhat


def main():
    ap = argparse.ArgumentParser(description='TUO Fase 5C.1.1 MCMC (Pantheon+ + BAO + Growth)')
    ap.add_argument('--data', type=Path, default=Path('Pantheon+SH0ES.dat'))
    ap.add_argument('--cov', type=Path, default=Path('Pantheon+SH0ES_STAT+SYS.cov'))
    ap.add_argument('--bao-dataset', choices=['desi_dr2', 'eboss_dr16_diag'], default='desi_dr2')
    ap.add_argument('--growth-dataset', choices=['minimal_diag', 'boss_eboss_diag', 'eboss_dr16_compressed'], default='minimal_diag')
    ap.add_argument('--growth-csv', type=Path, default=None)
    ap.add_argument('--growth-cov', type=Path, default=None)
    ap.add_argument('--outdir', type=Path, default=Path('tuo_phase5c11_run'))
    ap.add_argument('--zmin', type=float, default=0.01)
    ap.add_argument('--keep-calibrators', action='store_true')
    ap.add_argument('--use-hubble-flow-flag', action='store_true')
    ap.add_argument('--nchains', type=int, default=8)
    ap.add_argument('--nsteps', type=int, default=1000)
    ap.add_argument('--burn', type=int, default=300)
    ap.add_argument('--adapt-every', type=int, default=50)
    ap.add_argument('--nprocs', type=int, default=0)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--bg-npts', type=int, default=10000)
    ap.add_argument('--Omega-b0', dest='Omega_b0_fixed', type=float, default=None)
    ap.add_argument('--Omega-c0', dest='Omega_c0_fixed', type=float, default=None)
    ap.add_argument('--lam4', type=float, default=0.0)
    ap.add_argument('--xi-r', dest='xi_r', type=float, default=10.0)
    ap.add_argument('--z-c', dest='z_c', type=float, default=100.0)
    ap.add_argument('--s', type=float, default=4.0)
    ap.add_argument('--y-ini', dest='y_ini', type=float, default=0.0)
    ap.add_argument('--beta0-init', dest='beta0_init', type=float, default=0.02)
    ap.add_argument('--nu2-init', dest='nu2_init', type=float, default=0.02)
    ap.add_argument('--x-ini-init', dest='x_ini_init', type=float, default=0.05)
    ap.add_argument('--h-init', type=float, default=0.674)
    ap.add_argument('--omega-b-init', dest='omega_b_init', type=float, default=0.02237)
    ap.add_argument('--omega-c-init', dest='omega_c_init', type=float, default=0.1200)
    ap.add_argument('--sig-beta0', dest='sig_beta0', type=float, default=0.004)
    ap.add_argument('--sig-nu2', dest='sig_nu2', type=float, default=0.003)
    ap.add_argument('--sig-xini', dest='sig_xini', type=float, default=0.01)
    ap.add_argument('--sig-h', dest='sig_h', type=float, default=0.002)
    ap.add_argument('--sig-omega-b', dest='sig_omega_b', type=float, default=0.00005)
    ap.add_argument('--sig-omega-c', dest='sig_omega_c', type=float, default=0.0004)
    ap.add_argument('--beta0-mean', dest='beta0_mean', type=float, default=None)
    ap.add_argument('--beta0-sigma', dest='beta0_sigma', type=float, default=None)
    ap.add_argument('--h-mean', type=float, default=0.674)
    ap.add_argument('--h-sigma', type=float, default=0.005)
    ap.add_argument('--omega-b-mean', dest='omega_b_mean', type=float, default=0.02237)
    ap.add_argument('--omega-b-sigma', dest='omega_b_sigma', type=float, default=0.00015)
    ap.add_argument('--omega-c-mean', dest='omega_c_mean', type=float, default=0.1200)
    ap.add_argument('--omega-c-sigma', dest='omega_c_sigma', type=float, default=0.0012)
    ap.add_argument('--As', type=float, default=2.1e-9)
    ap.add_argument('--ns', type=float, default=0.965)
    ap.add_argument('--k-pivot', dest='k_pivot', type=float, default=0.10)
    ap.add_argument('--no-zeq-prior', action='store_true')
    ap.add_argument('--zeq-mean', type=float, default=3402.0)
    ap.add_argument('--zeq-sigma', type=float, default=26.0)
    ap.add_argument('--f-tuo-max', dest='f_tuo_max', type=float, default=1e-2)
    ap.add_argument('--final-backend', choices=['numba', 'scipy'], default='numba')
    ap.add_argument('--no-numba-shoot', action='store_true')
    ap.add_argument('--serial', action='store_true')
    args = ap.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    fixed = {
        'lam4': float(args.lam4),
        'xi_r': float(args.xi_r),
        'z_c': float(args.z_c),
        's': float(args.s),
        'y_ini': float(args.y_ini),
    }
    priors = {
        'h_mean': float(args.h_mean), 'h_sigma': float(args.h_sigma),
        'omega_b_mean': float(args.omega_b_mean), 'omega_b_sigma': float(args.omega_b_sigma),
        'omega_c_mean': float(args.omega_c_mean), 'omega_c_sigma': float(args.omega_c_sigma),
        'beta0_mean': args.beta0_mean, 'beta0_sigma': args.beta0_sigma,
    }
    growth_cfg = {'As': float(args.As), 'ns': float(args.ns)}

    worker_cfg = {
        'data': str(args.data), 'cov': str(args.cov),
        'zmin': float(args.zmin),
        'drop_calibrators': not args.keep_calibrators,
        'use_hubble_flow_flag': bool(args.use_hubble_flow_flag),
        'fixed': fixed,
        'bg_npts': int(args.bg_npts),
        'final_backend': args.final_backend,
        'use_numba_shoot': not args.no_numba_shoot,
        'f_tuo_max': float(args.f_tuo_max),
        'use_zeq_prior': not args.no_zeq_prior,
        'zeq_mean': float(args.zeq_mean), 'zeq_sigma': float(args.zeq_sigma),
        'bao_dataset': args.bao_dataset,
        'growth_dataset': args.growth_dataset,
        'growth_csv': str(args.growth_csv) if args.growth_csv is not None else None,
        'growth_cov': str(args.growth_cov) if args.growth_cov is not None else None,
        'priors': priors,
        'growth_cfg': growth_cfg,
        'k_pivot': float(args.k_pivot),
    }

    init_worker(worker_cfg)
    selection = dict(_WORKER['selection'])
    joint = _WORKER['joint']

    lcdm = joint.fit_lcdm(
        h_mean=priors['h_mean'], h_sigma=priors['h_sigma'],
        omega_b_mean=priors['omega_b_mean'], omega_b_sigma=priors['omega_b_sigma'],
        omega_c_mean=priors['omega_c_mean'], omega_c_sigma=priors['omega_c_sigma'],
        bg_npts=int(args.bg_npts), As=float(args.As), ns=float(args.ns), k_pivot_hmpc=float(args.k_pivot)
    )

    init = np.array([args.beta0_init, args.nu2_init, args.x_ini_init, args.h_init, args.omega_b_init, args.omega_c_init], dtype=float)
    d = init.size
    proposal_diag = np.array([args.sig_beta0, args.sig_nu2, args.sig_xini, args.sig_h, args.sig_omega_b, args.sig_omega_c], dtype=float)
    proposal_cov = np.diag(proposal_diag ** 2)

    rng = np.random.default_rng(int(args.seed))
    chain_inits = []
    for _ in range(args.nchains):
        th = init + rng.normal(scale=proposal_diag, size=d)
        th[0] = np.clip(th[0], -0.10, 0.10)
        th[1] = np.clip(th[1], 0.005, 0.06)
        th[2] = np.clip(th[2], 0.0, 0.6)
        th[3] = np.clip(th[3], 0.55, 0.85)
        th[4] = np.clip(th[4], 0.020, 0.0255)
        th[5] = np.clip(th[5], 0.09, 0.15)
        lp, _ = worker_logpost(th)
        tries = 0
        while not np.isfinite(lp) and tries < 100:
            th = init + rng.normal(scale=proposal_diag, size=d)
            th[0] = np.clip(th[0], -0.10, 0.10)
            th[1] = np.clip(th[1], 0.005, 0.06)
            th[2] = np.clip(th[2], 0.0, 0.6)
            th[3] = np.clip(th[3], 0.55, 0.85)
            th[4] = np.clip(th[4], 0.020, 0.0255)
            th[5] = np.clip(th[5], 0.09, 0.15)
            lp, _ = worker_logpost(th)
            tries += 1
        if not np.isfinite(lp):
            raise RuntimeError('Could not initialize finite posterior for one chain')
        chain_inits.append(th)

    run_args = []
    for j in range(args.nchains):
        run_args.append((int(args.seed) + 1000 + j, chain_inits[j], int(args.nsteps), int(args.burn), int(args.adapt_every), proposal_cov))

    results = []
    if args.serial or int(args.nprocs) == 1:
        for ra in run_args:
            _reset_core_warm(_WORKER['core'])
            results.append(_run_chain_serial(ra))
    else:
        nprocs = int(args.nprocs) if int(args.nprocs) > 0 else None
        with Pool(processes=nprocs, initializer=init_worker, initargs=(worker_cfg,)) as pool:
            results = pool.map(_run_chain_serial, run_args)

    samples_list, logps_list, blobs_list, accs, runtimes = zip(*results)
    nch = len(samples_list)
    nsteps = samples_list[0].shape[0]
    chains = np.stack(samples_list, axis=0)
    logps = np.stack(logps_list, axis=0)
    blobs = np.stack(blobs_list, axis=0)

    for j in range(nch):
        pd.DataFrame(samples_list[j], columns=['beta0', 'nu2', 'x_ini', 'h', 'omega_b', 'omega_c']).to_csv(outdir / f'chain_{j}.csv', index=False)
        np.save(outdir / f'chain_{j}_samples.npy', samples_list[j])
        np.save(outdir / f'chain_{j}_logpost.npy', logps_list[j])
        np.save(outdir / f'chain_{j}_blobs.npy', blobs_list[j])

    keep = slice(int(args.burn), None)
    flat_samples = chains[:, keep, :].reshape(-1, d)
    flat_logps = logps[:, keep].reshape(-1)
    flat_blobs = blobs[:, keep, :].reshape(-1, blobs.shape[-1])

    cols = ['beta0', 'nu2', 'x_ini', 'h', 'omega_b', 'omega_c', 'logpost'] + _BLOB_COLS
    flat_df = pd.DataFrame(np.column_stack([flat_samples, flat_logps, flat_blobs]), columns=cols)
    flat_df.to_csv(outdir / 'flat_samples_with_blobs.csv', index=False)

    best_idx = int(np.argmin(flat_blobs[:, 0]))
    best_theta = flat_samples[best_idx]
    best_blob = flat_blobs[best_idx]
    Rhat = gelman_rubin(chains[:, keep, :])

    summary = {
        'selection': selection,
        'priors': priors,
        'lcdm': lcdm,
        'nchains': int(args.nchains),
        'nsteps': int(args.nsteps),
        'burn': int(args.burn),
        'acceptance_mean': float(np.mean(accs)),
        'acceptance_per_chain': [float(a) for a in accs],
        'runtime_per_chain_sec': [float(t) for t in runtimes],
        'Rhat': {
            'beta0': float(Rhat[0]), 'nu2': float(Rhat[1]), 'x_ini': float(Rhat[2]),
            'h': float(Rhat[3]), 'omega_b': float(Rhat[4]), 'omega_c': float(Rhat[5]),
        },
        'best_fit_tuo': {
            'beta0': float(best_theta[0]), 'nu2': float(best_theta[1]), 'x_ini': float(best_theta[2]),
            'h': float(best_theta[3]), 'omega_b': float(best_theta[4]), 'omega_c': float(best_theta[5]),
            'logpost': float(flat_logps[best_idx]),
            'chi2_total': float(best_blob[0]), 'chi2_sn': float(best_blob[1]), 'chi2_bao': float(best_blob[2]),
            'chi2_growth': float(best_blob[3]), 'chi2_prior': float(best_blob[4]), 'Omega_V0': float(best_blob[5]),
            'z_eq': float(best_blob[6]), 'z_t': float(best_blob[7]), 'f_TUO_zstar': float(best_blob[8]),
            'r_d_Mpc': float(best_blob[9]), 'z_drag': float(best_blob[10]),
            'sigma8_0_pred': float(best_blob[11]), 'S8_0_pred': float(best_blob[12]),
            'Omega_b0': float(best_blob[16]), 'Omega_c0': float(best_blob[17]), 'Omega_r0': float(best_blob[18]),
        },
        'delta_chi2_tuo_minus_lcdm': float(best_blob[0] - lcdm['chi2_total']),
        'delta_chi2eff_tuo_minus_lcdm': float((best_blob[0] + best_blob[4]) - lcdm['chi2_posterior_effective']),
        'posterior_quantiles_16_50_84': {
            'beta0': [float(x) for x in np.quantile(flat_samples[:, 0], [0.16, 0.50, 0.84])],
            'nu2': [float(x) for x in np.quantile(flat_samples[:, 1], [0.16, 0.50, 0.84])],
            'x_ini': [float(x) for x in np.quantile(flat_samples[:, 2], [0.16, 0.50, 0.84])],
            'h': [float(x) for x in np.quantile(flat_samples[:, 3], [0.16, 0.50, 0.84])],
            'omega_b': [float(x) for x in np.quantile(flat_samples[:, 4], [0.16, 0.50, 0.84])],
            'omega_c': [float(x) for x in np.quantile(flat_samples[:, 5], [0.16, 0.50, 0.84])],
        },
    }

    with open(outdir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
