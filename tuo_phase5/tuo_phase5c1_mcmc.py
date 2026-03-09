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
from tuo_phase5c1_joint_likelihood import (
    PantheonBAOGrowthJointLikelihood5C1,
    pick_bao_dataset,
    pick_growth_dataset,
)

_WORKER: Dict[str, Any] = {}
_BLOB_COLS = [
    'chi2_total', 'chi2_sn', 'chi2_bao', 'chi2_growth', 'chi2_prior', 'Omega_V0',
    'z_eq', 'z_t', 'f_TUO_zstar', 'r_d_Mpc', 'z_drag', 'sigma8_0_pred',
    'h', 'omega_b', 'omega_c', 'Omega_b0', 'Omega_c0', 'Omega_r0'
]


def _reset_core_warm(core: TUOPhase4CoreOptimized):
    core._warm_root = None
    core._warm_bracket = None
    core.last_shoot_evals = 0


def init_worker(worker_cfg: Dict[str, Any]):
    global _WORKER
    data = Path(worker_cfg['data']); cov = Path(worker_cfg['cov'])
    zmin = float(worker_cfg['zmin']); drop_cal = bool(worker_cfg['drop_calibrators']); use_hf = bool(worker_cfg['use_hubble_flow_flag'])
    fixed = dict(worker_cfg['fixed'])
    bg_npts = int(worker_cfg['bg_npts']); final_backend = worker_cfg.get('final_backend', 'numba'); use_numba_shoot = bool(worker_cfg.get('use_numba_shoot', True))
    f_tuo_max = float(worker_cfg.get('f_tuo_max', 1e-2))
    use_zeq_prior = bool(worker_cfg.get('use_zeq_prior', False)); zeq_mean = float(worker_cfg.get('zeq_mean', 3402.0)); zeq_sigma = float(worker_cfg.get('zeq_sigma', 26.0))
    bao_name = worker_cfg.get('bao_dataset', 'desi_dr2'); growth_name = worker_cfg.get('growth_dataset', 'boss_eboss_diag'); growth_csv = worker_cfg.get('growth_csv', None); growth_cov = worker_cfg.get('growth_cov', None); k_pivot = float(worker_cfg.get('k_pivot', 0.1))
    priors = dict(worker_cfg['priors'])

    df, covm = load_pantheon(data, cov)
    df_sel, cov_sel, _ = apply_selection(df, covm, zmin=zmin, drop_calibrators=drop_cal, use_hubble_flow_flag=use_hf)
    sn_like = PantheonLikelihoodOptimized(df_sel, cov_sel)
    bao_dataset = pick_bao_dataset(bao_name)
    growth_dataset = pick_growth_dataset(growth_name, Path(growth_csv) if growth_csv else None, Path(growth_cov) if growth_cov else None, k_pivot_hmpc=k_pivot)
    joint = PantheonBAOGrowthJointLikelihood5C1(sn_like, bao_dataset, growth_dataset)

    from tuo_phase5b_joint_likelihood import derive_background_densities
    h0 = priors['h_mean']; ob0 = priors['omega_b_mean']; oc0 = priors['omega_c_mean']
    dens0 = derive_background_densities(h=h0, omega_b=ob0, omega_c=oc0)
    core = TUOPhase4CoreOptimized({'Omega_b0': dens0['Omega_b0'], 'Omega_c0': dens0['Omega_c0'], 'Omega_r0': dens0['Omega_r0'], **fixed})

    _WORKER = {
        'joint': joint, 'core': core, 'fixed': fixed, 'bg_npts': bg_npts,
        'final_backend': final_backend, 'use_numba_shoot': use_numba_shoot,
        'f_tuo_max': f_tuo_max, 'use_zeq_prior': use_zeq_prior, 'zeq_mean': zeq_mean, 'zeq_sigma': zeq_sigma,
        'selection': {'n_total': int(len(df)), 'n_selected': int(len(df_sel)), 'bao_dataset': bao_dataset.name, 'growth_dataset': growth_dataset.name},
        'priors': priors,
    }


def worker_logpost(theta: np.ndarray) -> Tuple[float, np.ndarray]:
    global _WORKER
    beta0, nu2, x_ini, h, omega_b, omega_c = map(float, theta)
    if not (0.0 <= beta0 <= 0.1):
        return -np.inf, np.array([np.inf] + [np.nan]*(len(_BLOB_COLS)-1))
    if not (0.0 <= nu2 <= 0.08):
        return -np.inf, np.array([np.inf] + [np.nan]*(len(_BLOB_COLS)-1))
    if not (0.0 <= x_ini <= 0.8):
        return -np.inf, np.array([np.inf] + [np.nan]*(len(_BLOB_COLS)-1))
    if not (0.55 <= h <= 0.85):
        return -np.inf, np.array([np.inf] + [np.nan]*(len(_BLOB_COLS)-1))
    if not (0.020 <= omega_b <= 0.0255):
        return -np.inf, np.array([np.inf] + [np.nan]*(len(_BLOB_COLS)-1))
    if not (0.09 <= omega_c <= 0.15):
        return -np.inf, np.array([np.inf] + [np.nan]*(len(_BLOB_COLS)-1))

    p = dict(_WORKER['fixed']); p.update({'beta0': beta0, 'nu2': nu2, 'x_ini': x_ini})
    joint = _WORKER['joint']; core = _WORKER['core']; pri = _WORKER['priors']
    try:
        res = joint.evaluate_tuo(p, h=h, omega_b=omega_b, omega_c=omega_c,
                                 As=pri['As_mean'], ns=pri['ns'],
                                 core=core, bg_npts=_WORKER['bg_npts'],
                                 final_backend=_WORKER['final_backend'], use_numba_shoot=_WORKER['use_numba_shoot'])
        f_tuo = res.get('f_TUO_zstar')
        if (f_tuo is None) or (not np.isfinite(f_tuo)) or (f_tuo > _WORKER['f_tuo_max']):
            blob = np.array([np.inf, res.get('chi2_sn', np.nan), res.get('chi2_bao', np.nan), res.get('chi2_growth', np.nan), np.nan,
                             res.get('Omega_V0', np.nan), res.get('z_eq', np.nan), res.get('z_t', np.nan), f_tuo if f_tuo is not None else np.nan,
                             res.get('r_d_Mpc', np.nan), res.get('z_drag', np.nan), res.get('sigma8_0_pred', np.nan),
                             h, omega_b, omega_c, res.get('Omega_b0', np.nan), res.get('Omega_c0', np.nan), res.get('Omega_r0', np.nan)], dtype=float)
            return -np.inf, blob
        chi2_total = float(res['chi2_total'])
        chi2_prior = joint.priors_chi2(h=h, omega_b=omega_b, omega_c=omega_c,
                                       As=pri['As_mean'], As_mean=pri['As_mean'], As_sigma=pri.get('As_sigma'),
                                       h_mean=pri['h_mean'], h_sigma=pri['h_sigma'],
                                       omega_b_mean=pri['omega_b_mean'], omega_b_sigma=pri['omega_b_sigma'],
                                       omega_c_mean=pri['omega_c_mean'], omega_c_sigma=pri['omega_c_sigma'])
        logpost = -0.5 * (chi2_total + chi2_prior)
        if _WORKER['use_zeq_prior']:
            z_eq = res.get('z_eq')
            if z_eq is None or (not np.isfinite(z_eq)):
                return -np.inf, np.array([chi2_total, res.get('chi2_sn', np.nan), res.get('chi2_bao', np.nan), res.get('chi2_growth', np.nan), chi2_prior,
                                          res.get('Omega_V0', np.nan), np.nan, res.get('z_t', np.nan), f_tuo, res.get('r_d_Mpc', np.nan), res.get('z_drag', np.nan), res.get('sigma8_0_pred', np.nan), h, omega_b, omega_c,
                                          res.get('Omega_b0', np.nan), res.get('Omega_c0', np.nan), res.get('Omega_r0', np.nan)], dtype=float)
            logpost += -0.5 * ((float(z_eq) - _WORKER['zeq_mean']) / _WORKER['zeq_sigma']) ** 2
        blob = np.array([chi2_total, float(res.get('chi2_sn', np.nan)), float(res.get('chi2_bao', np.nan)), float(res.get('chi2_growth', np.nan)), float(chi2_prior),
                         float(res.get('Omega_V0', np.nan)), float(res.get('z_eq', np.nan)) if res.get('z_eq') is not None else np.nan,
                         float(res.get('z_t', np.nan)) if res.get('z_t') is not None else np.nan, float(f_tuo),
                         float(res.get('r_d_Mpc', np.nan)), float(res.get('z_drag', np.nan)), float(res.get('sigma8_0_pred', np.nan)),
                         h, omega_b, omega_c, float(res.get('Omega_b0', np.nan)), float(res.get('Omega_c0', np.nan)), float(res.get('Omega_r0', np.nan))], dtype=float)
        return float(logpost), blob
    except Exception:
        return -np.inf, np.array([np.inf] + [np.nan]*(len(_BLOB_COLS)-1))


def adaptive_metropolis_chain(chain_cfg: Dict[str, Any]) -> Dict[str, Any]:
    global _WORKER
    chain_id = int(chain_cfg['chain_id']); nsteps = int(chain_cfg['nsteps']); burn = int(chain_cfg['burn']); adapt_every = int(chain_cfg.get('adapt_every', 25)); seed = int(chain_cfg['seed'])
    x0 = np.array(chain_cfg['x0'], dtype=float); prop_scales = np.array(chain_cfg['prop_scales'], dtype=float)
    rng = np.random.default_rng(seed); d = len(x0)
    _reset_core_warm(_WORKER['core'])
    lp, blob = worker_logpost(x0)
    tries = 0
    while not np.isfinite(lp):
        x0 = x0 + prop_scales * rng.normal(size=d)
        lp, blob = worker_logpost(x0)
        tries += 1
        if tries > 300:
            raise RuntimeError(f'Chain {chain_id}: unable to find finite start')
    x = x0.copy(); samples = np.zeros((nsteps, d)); logp = np.full(nsteps, -np.inf); blobs = np.full((nsteps, len(_BLOB_COLS)), np.nan); accepted = 0
    prop_cov = np.diag(prop_scales**2); chol = np.linalg.cholesky(prop_cov); history=[x.copy()]; scale_fac=(2.38**2)/d; eps=1e-12; t0=time.time()
    for i in range(nsteps):
        if i > burn and (i % adapt_every == 0) and len(history) > d + 10:
            arr=np.array(history); C=np.cov(arr.T, ddof=1); vals, vecs = eigh(C); vals=np.clip(vals,1e-14,None); C=(vecs*vals)@vecs.T; prop_cov=scale_fac*C+eps*np.eye(d); chol=np.linalg.cholesky(prop_cov)
        proposal = x + chol @ rng.normal(size=d)
        lp_p, blob_p = worker_logpost(proposal)
        if np.isfinite(lp_p) and np.log(rng.uniform()) < (lp_p - lp):
            x=proposal; lp=lp_p; blob=blob_p; accepted += 1
        history.append(x.copy()); samples[i]=x; logp[i]=lp; blobs[i]=blob
    return {'chain_id': chain_id, 'samples': samples, 'logp': logp, 'blobs': blobs, 'acceptance': accepted/float(nsteps), 'runtime_sec': time.time()-t0}


def gelman_rubin(chains: np.ndarray) -> float:
    m, n = chains.shape[:2]
    chain_means = np.mean(chains, axis=1)
    grand = np.mean(chain_means, axis=0)
    B = n * np.var(chain_means, axis=0, ddof=1)
    W = np.mean(np.var(chains, axis=1, ddof=1), axis=0)
    var_hat = ((n - 1) / n) * W + (1 / n) * B
    Rhat = np.sqrt(var_hat / W)
    return Rhat


def main():
    ap = argparse.ArgumentParser(description='Phase 5C.1 MCMC: dynamic coupling, k-dependent Geff, A_s anchoring')
    ap.add_argument('--data', type=Path, default=Path('Pantheon+SH0ES.dat'))
    ap.add_argument('--cov', type=Path, default=Path('Pantheon+SH0ES_STAT+SYS.cov'))
    ap.add_argument('--bao-dataset', choices=['desi_dr2', 'eboss_dr16_diag'], default='desi_dr2')
    ap.add_argument('--growth-dataset', choices=['minimal_diag', 'boss_eboss_diag'], default='boss_eboss_diag')
    ap.add_argument('--growth-csv', type=str, default=None)
    ap.add_argument('--growth-cov', type=str, default=None)
    ap.add_argument('--k-pivot', type=float, default=0.1)
    ap.add_argument('--h', type=float, default=0.674)
    ap.add_argument('--zmin', type=float, default=0.01)
    ap.add_argument('--keep-calibrators', action='store_true')
    ap.add_argument('--use-hubble-flow-flag', action='store_true')
    ap.add_argument('--outdir', type=Path, default=Path('phase5c1_run'))
    ap.add_argument('--nchains', type=int, default=8)
    ap.add_argument('--nsteps', type=int, default=1000)
    ap.add_argument('--burn', type=int, default=300)
    ap.add_argument('--adapt-every', type=int, default=25)
    ap.add_argument('--nprocs', type=int, default=1)
    ap.add_argument('--seed', type=int, default=20250308)
    ap.add_argument('--bg-npts', type=int, default=10000)
    ap.add_argument('--lam4', type=float, default=0.0)
    ap.add_argument('--xi-r', type=float, default=10.0, dest='xi_r')
    ap.add_argument('--z-c', type=float, default=100.0, dest='z_c')
    ap.add_argument('--s', type=float, default=4.0)
    ap.add_argument('--y-ini', type=float, default=0.0, dest='y_ini')
    ap.add_argument('--z-ini', type=float, default=1e5, dest='z_ini')
    ap.add_argument('--beta0-init', type=float, default=0.02)
    ap.add_argument('--nu2-init', type=float, default=0.02)
    ap.add_argument('--x-ini-init', type=float, default=0.05)
    ap.add_argument('--sig-beta0', type=float, default=0.01)
    ap.add_argument('--sig-nu2', type=float, default=0.01)
    ap.add_argument('--sig-xini', type=float, default=0.03)
    ap.add_argument('--sig-h', type=float, default=0.003)
    ap.add_argument('--sig-omegab', type=float, default=1.5e-4)
    ap.add_argument('--sig-omegac', type=float, default=1.2e-3)
    ap.add_argument('--h-mean', type=float, default=0.674)
    ap.add_argument('--h-sigma', type=float, default=0.005)
    ap.add_argument('--omega-b-mean', type=float, default=0.02237, dest='omega_b_mean')
    ap.add_argument('--omega-b-sigma', type=float, default=0.00015, dest='omega_b_sigma')
    ap.add_argument('--omega-c-mean', type=float, default=0.1200, dest='omega_c_mean')
    ap.add_argument('--omega-c-sigma', type=float, default=0.0012, dest='omega_c_sigma')
    ap.add_argument('--As-mean', type=float, default=2.1e-9, dest='As_mean')
    ap.add_argument('--As-sigma', type=float, default=None, dest='As_sigma')
    ap.add_argument('--f-tuo-max', type=float, default=1e-2)
    ap.add_argument('--no-zeq-prior', action='store_true')
    ap.add_argument('--zeq-mean', type=float, default=3402.0)
    ap.add_argument('--zeq-sigma', type=float, default=26.0)
    ap.add_argument('--final-backend', choices=['numba', 'scipy'], default='numba')
    ap.add_argument('--no-numba-shoot', action='store_true')
    ap.add_argument('--serial', action='store_true')
    args = ap.parse_args()

    outdir = args.outdir; outdir.mkdir(parents=True, exist_ok=True)
    fixed = {'lam4': float(args.lam4), 'xi_r': float(args.xi_r), 'z_c': float(args.z_c), 's': float(args.s), 'y_ini': float(args.y_ini), 'z_ini': float(args.z_ini)}
    priors = {'h_mean': float(args.h_mean), 'h_sigma': float(args.h_sigma), 'omega_b_mean': float(args.omega_b_mean), 'omega_b_sigma': float(args.omega_b_sigma), 'omega_c_mean': float(args.omega_c_mean), 'omega_c_sigma': float(args.omega_c_sigma), 'As_mean': float(args.As_mean), 'As_sigma': None if args.As_sigma is None else float(args.As_sigma), 'ns': 0.965}
    worker_cfg = {'data': str(args.data), 'cov': str(args.cov), 'zmin': float(args.zmin), 'drop_calibrators': not args.keep_calibrators, 'use_hubble_flow_flag': bool(args.use_hubble_flow_flag), 'fixed': fixed,
                  'bg_npts': int(args.bg_npts), 'final_backend': args.final_backend, 'use_numba_shoot': not args.no_numba_shoot,
                  'f_tuo_max': float(args.f_tuo_max), 'use_zeq_prior': not args.no_zeq_prior, 'zeq_mean': float(args.zeq_mean), 'zeq_sigma': float(args.zeq_sigma),
                  'bao_dataset': args.bao_dataset, 'growth_dataset': args.growth_dataset, 'growth_csv': args.growth_csv, 'growth_cov': args.growth_cov, 'k_pivot': float(args.k_pivot), 'priors': priors}

    init_worker(worker_cfg)
    lcdm = _WORKER['joint'].fit_lcdm(h_mean=priors['h_mean'], h_sigma=priors['h_sigma'],
                                     omega_b_mean=priors['omega_b_mean'], omega_b_sigma=priors['omega_b_sigma'],
                                     omega_c_mean=priors['omega_c_mean'], omega_c_sigma=priors['omega_c_sigma'],
                                     As_mean=priors['As_mean'], As_sigma=priors['As_sigma'], ns=priors['ns'], bg_npts=args.bg_npts)
    selection = dict(_WORKER['selection'])

    rng = np.random.default_rng(args.seed)
    p0_center = np.array([args.beta0_init, args.nu2_init, args.x_ini_init, priors['h_mean'], priors['omega_b_mean'], priors['omega_c_mean']], dtype=float)
    sig0 = np.array([args.sig_beta0, args.sig_nu2, args.sig_xini, args.sig_h, args.sig_omegab, args.sig_omegac], dtype=float)
    chain_cfgs = []
    for i in range(args.nchains):
        x0 = p0_center + sig0 * rng.normal(size=6)
        chain_cfgs.append({'chain_id': i, 'nsteps': int(args.nsteps), 'burn': int(args.burn), 'adapt_every': int(args.adapt_every), 'seed': int(args.seed + 100*i + 17), 'x0': x0, 'prop_scales': sig0})

    t_all = time.time()
    if args.serial or int(args.nprocs) <= 1:
        results = [adaptive_metropolis_chain(cfg) for cfg in chain_cfgs]
    else:
        with Pool(processes=int(args.nprocs), initializer=init_worker, initargs=(worker_cfg,)) as pool:
            results = pool.map(adaptive_metropolis_chain, chain_cfgs)
    total_runtime = time.time() - t_all

    results = sorted(results, key=lambda r: r['chain_id'])
    samples = np.stack([r['samples'] for r in results], axis=0)
    logp = np.stack([r['logp'] for r in results], axis=0)
    blobs = np.stack([r['blobs'] for r in results], axis=0)
    acc = [float(r['acceptance']) for r in results]; runtimes = [float(r['runtime_sec']) for r in results]

    burn = int(args.burn)
    post_samples = samples[:, burn:, :]
    post_logp = logp[:, burn:]
    post_blobs = blobs[:, burn:, :]

    Rhat = gelman_rubin(post_samples)
    names = ['beta0', 'nu2', 'x_ini', 'h', 'omega_b', 'omega_c']
    flat_s = post_samples.reshape((-1, post_samples.shape[-1]))
    flat_lp = post_logp.reshape(-1)
    flat_b = post_blobs.reshape((-1, post_blobs.shape[-1]))
    best_idx = int(np.nanargmax(flat_lp))
    best_theta = flat_s[best_idx]
    best_blob = flat_b[best_idx]
    quant = {name: np.nanpercentile(flat_s[:, i], [16,50,84]).tolist() for i, name in enumerate(names)}

    summary = {
        'selection': selection,
        'priors': priors,
        'nchains': int(args.nchains), 'nsteps': int(args.nsteps), 'burn': int(args.burn),
        'acceptance_mean': float(np.mean(acc)), 'acceptance_per_chain': acc,
        'runtime_per_chain_sec': runtimes, 'runtime_total_sec': float(total_runtime),
        'Rhat': {name: float(Rhat[i]) for i, name in enumerate(names)},
        'lcdm': lcdm,
        'best_fit_tuo': {name: float(best_theta[i]) for i, name in enumerate(names)} | {col: float(best_blob[j]) if np.isfinite(best_blob[j]) else None for j, col in enumerate(_BLOB_COLS)},
        'delta_chi2_tuo_minus_lcdm': float(best_blob[0] - lcdm['chi2_total']),
        'delta_chi2eff_tuo_minus_lcdm': float((best_blob[0] + best_blob[4]) - lcdm['chi2_posterior_effective']),
        'posterior_quantiles_16_50_84': quant,
    }
    (outdir / 'summary.json').write_text(json.dumps(summary, indent=2, default=lambda x: x.tolist() if hasattr(x,'tolist') else x))
    for r in results:
        cid = r['chain_id']
        np.save(outdir / f'chain_{cid}_samples.npy', r['samples']); np.save(outdir / f'chain_{cid}_logpost.npy', r['logp']); np.save(outdir / f'chain_{cid}_blobs.npy', r['blobs'])
        df = pd.DataFrame(r['samples'], columns=names); df['logpost']=r['logp'];
        for j, col in enumerate(_BLOB_COLS): df[col]=r['blobs'][:,j]
        df.to_csv(outdir / f'chain_{cid}.csv', index=False)
    flat_df = pd.DataFrame(flat_s, columns=names); flat_df['logpost']=flat_lp
    for j, col in enumerate(_BLOB_COLS): flat_df[col]=flat_b[:,j]
    flat_df.to_csv(outdir / 'flat_samples_with_blobs.csv', index=False)
    print(json.dumps(summary, indent=2, default=lambda x: x.tolist() if hasattr(x,'tolist') else x))


if __name__ == '__main__':
    main()
