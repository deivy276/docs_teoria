import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from multiprocessing import Pool, cpu_count

from tuo_phase4_core_optimized import TUOPhase4CoreOptimized
from tuo_pantheon_fit_optimized import load_pantheon, apply_selection, PantheonLikelihoodOptimized
from tuo_bao_module import builtin_desi_dr2_bao, builtin_eboss_dr16_diagonal
from tuo_phase5b_joint_likelihood import PantheonBAOJointLikelihood5B, derive_background_densities, OMEGA_R_PHYSICAL

_WORKER: Dict[str, Any] = {}
_BLOB_COLS = [
    "chi2_total", "chi2_sn", "chi2_bao", "chi2_prior", "Omega_V0",
    "z_eq", "z_t", "f_TUO_zstar", "r_d_Mpc", "z_drag",
    "h", "omega_b", "omega_c", "Omega_b0", "Omega_c0", "Omega_r0"
]


def _reset_core_warm(core: TUOPhase4CoreOptimized):
    core._warm_root = None
    core._warm_bracket = None
    core.last_shoot_evals = 0


def init_worker(worker_cfg: Dict[str, Any]):
    global _WORKER
    data = Path(worker_cfg['data'])
    cov = Path(worker_cfg['cov'])
    zmin = float(worker_cfg['zmin'])
    drop_calibrators = bool(worker_cfg['drop_calibrators'])
    use_hubble_flow_flag = bool(worker_cfg['use_hubble_flow_flag'])
    fixed = dict(worker_cfg['fixed'])
    bg_npts = int(worker_cfg['bg_npts'])
    final_backend = worker_cfg.get('final_backend', 'numba')
    use_numba_shoot = bool(worker_cfg.get('use_numba_shoot', True))
    f_tuo_max = float(worker_cfg.get('f_tuo_max', 1e-2))
    use_zeq_prior = bool(worker_cfg.get('use_zeq_prior', False))
    zeq_mean = float(worker_cfg.get('zeq_mean', 3402.0))
    zeq_sigma = float(worker_cfg.get('zeq_sigma', 26.0))
    bao_name = worker_cfg.get('bao_dataset', 'desi_dr2')

    prior_cfg = dict(worker_cfg['priors'])

    df, covm = load_pantheon(data, cov)
    df_sel, cov_sel, _ = apply_selection(df, covm, zmin=zmin,
                                         drop_calibrators=drop_calibrators,
                                         use_hubble_flow_flag=use_hubble_flow_flag)
    sn_like = PantheonLikelihoodOptimized(df_sel, cov_sel)
    bao_dataset = builtin_desi_dr2_bao() if bao_name == 'desi_dr2' else builtin_eboss_dr16_diagonal()
    joint = PantheonBAOJointLikelihood5B(sn_like, bao_dataset)
    # core is seeded at prior means; update() will overwrite per evaluation
    h0 = prior_cfg['h_mean']
    ob0 = prior_cfg['omega_b_mean']
    oc0 = prior_cfg['omega_c_mean']
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
        },
        'priors': prior_cfg,
    }


def worker_logpost(theta: np.ndarray) -> Tuple[float, np.ndarray]:
    global _WORKER
    beta0, nu2, x_ini, h, omega_b, omega_c = map(float, theta)

    # Broad physical box priors
    if not (0.0 <= beta0 <= 0.1):
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

    p = dict(_WORKER['fixed'])
    p.update({'beta0': beta0, 'nu2': nu2, 'x_ini': x_ini})
    joint = _WORKER['joint']
    core = _WORKER['core']
    pri = _WORKER['priors']

    try:
        res = joint.evaluate_tuo(p, h=h, omega_b=omega_b, omega_c=omega_c,
                                 core=core, bg_npts=_WORKER['bg_npts'],
                                 final_backend=_WORKER['final_backend'],
                                 use_numba_shoot=_WORKER['use_numba_shoot'])
        f_tuo = res.get('f_TUO_zstar')
        if (f_tuo is None) or (not np.isfinite(f_tuo)) or (f_tuo > _WORKER['f_tuo_max']):
            blob = np.array([
                np.inf,
                res.get('chi2_sn', np.nan),
                res.get('chi2_bao', np.nan),
                np.nan,
                res.get('Omega_V0', np.nan),
                res.get('z_eq', np.nan),
                res.get('z_t', np.nan),
                f_tuo if f_tuo is not None else np.nan,
                res.get('r_d_Mpc', np.nan),
                res.get('z_drag', np.nan),
                h, omega_b, omega_c,
                res.get('Omega_b0', np.nan),
                res.get('Omega_c0', np.nan),
                res.get('Omega_r0', np.nan),
            ], dtype=float)
            return -np.inf, blob

        chi2_total = float(res['chi2_total'])
        chi2_prior = joint.priors_chi2(h=h, omega_b=omega_b, omega_c=omega_c,
                                       h_mean=pri['h_mean'], h_sigma=pri['h_sigma'],
                                       omega_b_mean=pri['omega_b_mean'], omega_b_sigma=pri['omega_b_sigma'],
                                       omega_c_mean=pri['omega_c_mean'], omega_c_sigma=pri['omega_c_sigma'])
        logpost = -0.5 * (chi2_total + chi2_prior)
        if _WORKER['use_zeq_prior']:
            z_eq = res.get('z_eq')
            if z_eq is None or (not np.isfinite(z_eq)):
                return -np.inf, np.array([chi2_total, res.get('chi2_sn', np.nan), res.get('chi2_bao', np.nan), chi2_prior, res.get('Omega_V0', np.nan), np.nan, res.get('z_t', np.nan), f_tuo, res.get('r_d_Mpc', np.nan), res.get('z_drag', np.nan), h, omega_b, omega_c, res.get('Omega_b0', np.nan), res.get('Omega_c0', np.nan), res.get('Omega_r0', np.nan)], dtype=float)
            logpost += -0.5 * ((float(z_eq) - _WORKER['zeq_mean']) / _WORKER['zeq_sigma']) ** 2
        blob = np.array([
            chi2_total,
            float(res.get('chi2_sn', np.nan)),
            float(res.get('chi2_bao', np.nan)),
            float(chi2_prior),
            float(res.get('Omega_V0', np.nan)),
            float(res.get('z_eq', np.nan)) if res.get('z_eq') is not None else np.nan,
            float(res.get('z_t', np.nan)) if res.get('z_t') is not None else np.nan,
            float(f_tuo),
            float(res.get('r_d_Mpc', np.nan)),
            float(res.get('z_drag', np.nan)),
            float(h), float(omega_b), float(omega_c),
            float(res.get('Omega_b0', np.nan)),
            float(res.get('Omega_c0', np.nan)),
            float(res.get('Omega_r0', np.nan)),
        ], dtype=float)
        return float(logpost), blob
    except Exception:
        return -np.inf, np.array([np.inf] + [np.nan] * (len(_BLOB_COLS) - 1), dtype=float)


def adaptive_metropolis_chain(chain_cfg: Dict[str, Any]) -> Dict[str, Any]:
    global _WORKER
    chain_id = int(chain_cfg['chain_id'])
    nsteps = int(chain_cfg['nsteps'])
    burn = int(chain_cfg['burn'])
    adapt_every = int(chain_cfg.get('adapt_every', 25))
    seed = int(chain_cfg['seed'])
    x0 = np.array(chain_cfg['x0'], dtype=float)
    prop_scales = np.array(chain_cfg['prop_scales'], dtype=float)

    rng = np.random.default_rng(seed)
    d = len(x0)
    _reset_core_warm(_WORKER['core'])

    lp, blob = worker_logpost(x0)
    tries = 0
    while not np.isfinite(lp):
        x0 = x0 + prop_scales * rng.normal(size=d)
        lp, blob = worker_logpost(x0)
        tries += 1
        if tries > 300:
            raise RuntimeError(f'Chain {chain_id}: unable to find finite start')

    x = x0.copy()
    samples = np.zeros((nsteps, d), dtype=float)
    logp = np.full(nsteps, -np.inf, dtype=float)
    blobs = np.full((nsteps, len(_BLOB_COLS)), np.nan, dtype=float)
    accepted = 0

    prop_cov = np.diag(prop_scales ** 2)
    chol = np.linalg.cholesky(prop_cov)
    history = [x.copy()]
    scale_fac = (2.38 ** 2) / d
    eps = 1e-12

    t0 = time.time()
    for i in range(nsteps):
        if i > burn and (i % adapt_every == 0) and len(history) > d + 10:
            arr = np.array(history)
            C = np.cov(arr.T, ddof=1)
            vals, vecs = eigh(C)
            vals = np.clip(vals, 1e-14, None)
            C = (vecs * vals) @ vecs.T
            prop_cov = scale_fac * C + eps * np.eye(d)
            chol = np.linalg.cholesky(prop_cov)

        proposal = x + chol @ rng.normal(size=d)
        lp_prop, blob_prop = worker_logpost(proposal)
        if np.isfinite(lp_prop) and np.log(rng.uniform()) < (lp_prop - lp):
            x = proposal
            lp = lp_prop
            blob = blob_prop
            accepted += 1

        history.append(x.copy())
        samples[i] = x
        logp[i] = lp
        blobs[i] = blob

    runtime = time.time() - t0
    return {
        'chain_id': chain_id,
        'samples': samples,
        'logpost': logp,
        'blobs': blobs,
        'acceptance': accepted / max(nsteps, 1),
        'runtime_sec': runtime,
    }


def gelman_rubin(chains: np.ndarray) -> np.ndarray:
    chains = np.asarray(chains, dtype=float)
    m, n, d = chains.shape
    if n < 2:
        return np.full(d, np.nan)
    chain_means = chains.mean(axis=1)
    grand_mean = chain_means.mean(axis=0)
    B = n / (m - 1) * np.sum((chain_means - grand_mean) ** 2, axis=0) if m > 1 else np.zeros(d)
    W = np.mean(np.var(chains, axis=1, ddof=1), axis=0)
    var_hat = (n - 1) / n * W + B / n
    return np.sqrt(var_hat / W)


def main():
    ap = argparse.ArgumentParser(description='Phase 5B optimized AM-MCMC for TUO + Pantheon+ + BAO with free h, ω_b, ω_c')
    ap.add_argument('--data', type=Path, default=Path('Pantheon+SH0ES.dat'))
    ap.add_argument('--cov', type=Path, default=Path('Pantheon+SH0ES_STAT+SYS.cov'))
    ap.add_argument('--bao-dataset', choices=['desi_dr2', 'eboss_dr16_diag'], default='desi_dr2')
    ap.add_argument('--outdir', type=Path, default=Path('tuo_phase5b_run'))
    ap.add_argument('--zmin', type=float, default=0.01)
    ap.add_argument('--keep-calibrators', action='store_true')
    ap.add_argument('--use-hubble-flow-flag', action='store_true')
    ap.add_argument('--nchains', type=int, default=4)
    ap.add_argument('--nsteps', type=int, default=120)
    ap.add_argument('--burn', type=int, default=40)
    ap.add_argument('--adapt-every', type=int, default=20)
    ap.add_argument('--nprocs', type=int, default=0)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--bg-npts', type=int, default=8000)
    # fixed TUO params aside from sampled ones
    ap.add_argument('--lam4', type=float, default=0.0)
    ap.add_argument('--xi-r', type=float, default=10.0, dest='xi_r')
    ap.add_argument('--z-c', type=float, default=100.0, dest='z_c')
    ap.add_argument('--s', type=float, default=4.0)
    ap.add_argument('--y-ini', type=float, default=0.0, dest='y_ini')
    ap.add_argument('--z-ini', type=float, default=1e5, dest='z_ini')
    # initial centers and proposal widths
    ap.add_argument('--beta0-init', type=float, default=0.02)
    ap.add_argument('--nu2-init', type=float, default=0.02)
    ap.add_argument('--x-ini-init', type=float, default=0.05)
    ap.add_argument('--h-init', type=float, default=0.674)
    ap.add_argument('--omega-b-init', type=float, default=0.02237, dest='omega_b_init')
    ap.add_argument('--omega-c-init', type=float, default=0.1200, dest='omega_c_init')
    ap.add_argument('--sig-beta0', type=float, default=0.006)
    ap.add_argument('--sig-nu2', type=float, default=0.004)
    ap.add_argument('--sig-xini', type=float, default=0.03)
    ap.add_argument('--sig-h', type=float, default=0.003)
    ap.add_argument('--sig-omega-b', type=float, default=0.00012, dest='sig_omega_b')
    ap.add_argument('--sig-omega-c', type=float, default=0.0010, dest='sig_omega_c')
    # optional zeq prior (off by default in 5B)
    ap.add_argument('--use-zeq-prior', action='store_true')
    ap.add_argument('--zeq-mean', type=float, default=3402.0)
    ap.add_argument('--zeq-sigma', type=float, default=26.0)
    ap.add_argument('--f-tuo-max', type=float, default=1e-2)
    # Gaussian priors on free early-time parameters
    ap.add_argument('--h-mean', type=float, default=0.674)
    ap.add_argument('--h-sigma', type=float, default=0.005)
    ap.add_argument('--omega-b-mean', type=float, default=0.02237, dest='omega_b_mean')
    ap.add_argument('--omega-b-sigma', type=float, default=0.00015, dest='omega_b_sigma')
    ap.add_argument('--omega-c-mean', type=float, default=0.1200, dest='omega_c_mean')
    ap.add_argument('--omega-c-sigma', type=float, default=0.0012, dest='omega_c_sigma')
    ap.add_argument('--final-backend', choices=['numba', 'scipy'], default='numba')
    ap.add_argument('--no-numba-shoot', action='store_true')
    ap.add_argument('--serial', action='store_true')
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    fixed = {
        'lam4': args.lam4,
        'xi_r': args.xi_r,
        'z_c': args.z_c,
        's': args.s,
        'y_ini': args.y_ini,
        'z_ini': args.z_ini,
        'beta0': args.beta0_init,
        'nu2': args.nu2_init,
        'x_ini': args.x_ini_init,
    }
    priors = {
        'h_mean': args.h_mean,
        'h_sigma': args.h_sigma,
        'omega_b_mean': args.omega_b_mean,
        'omega_b_sigma': args.omega_b_sigma,
        'omega_c_mean': args.omega_c_mean,
        'omega_c_sigma': args.omega_c_sigma,
    }
    worker_cfg = {
        'data': str(args.data),
        'cov': str(args.cov),
        'zmin': args.zmin,
        'drop_calibrators': not args.keep_calibrators,
        'use_hubble_flow_flag': args.use_hubble_flow_flag,
        'fixed': fixed,
        'bg_npts': args.bg_npts,
        'final_backend': args.final_backend,
        'use_numba_shoot': not args.no_numba_shoot,
        'f_tuo_max': args.f_tuo_max,
        'use_zeq_prior': args.use_zeq_prior,
        'zeq_mean': args.zeq_mean,
        'zeq_sigma': args.zeq_sigma,
        'bao_dataset': args.bao_dataset,
        'priors': priors,
    }

    # Baseline LCDM with same priors and BAO dataset
    init_worker(worker_cfg)
    lcdm = _WORKER['joint'].fit_lcdm(h_mean=args.h_mean, h_sigma=args.h_sigma,
                                     omega_b_mean=args.omega_b_mean, omega_b_sigma=args.omega_b_sigma,
                                     omega_c_mean=args.omega_c_mean, omega_c_sigma=args.omega_c_sigma,
                                     bg_npts=args.bg_npts)
    selection = dict(_WORKER['selection'])

    center = np.array([
        args.beta0_init,
        args.nu2_init,
        args.x_ini_init,
        args.h_init,
        args.omega_b_init,
        args.omega_c_init,
    ], dtype=float)
    scales = np.array([
        args.sig_beta0,
        args.sig_nu2,
        args.sig_xini,
        args.sig_h,
        args.sig_omega_b,
        args.sig_omega_c,
    ], dtype=float)

    rng = np.random.default_rng(args.seed)
    x0s = []
    for _ in range(args.nchains):
        for __ in range(500):
            cand = center + scales * rng.normal(size=center.size)
            lp, _ = worker_logpost(cand)
            if np.isfinite(lp):
                x0s.append(cand)
                break
        else:
            raise RuntimeError('Unable to seed all chains with finite starts')

    chain_cfgs = []
    for i in range(args.nchains):
        chain_cfgs.append({
            'chain_id': i + 1,
            'nsteps': args.nsteps,
            'burn': args.burn,
            'adapt_every': args.adapt_every,
            'seed': int(args.seed + 1009 * i),
            'x0': x0s[i],
            'prop_scales': scales,
        })

    results = []
    if args.serial or args.nchains == 1:
        for cfg in chain_cfgs:
            results.append(adaptive_metropolis_chain(cfg))
    else:
        nprocs = args.nprocs if args.nprocs > 0 else min(args.nchains, cpu_count())
        with Pool(processes=nprocs, initializer=init_worker, initargs=(worker_cfg,)) as pool:
            results = pool.map(adaptive_metropolis_chain, chain_cfgs)

    # Save per-chain files
    for res in results:
        cid = res['chain_id']
        np.save(args.outdir / f'chain_{cid}_samples.npy', res['samples'])
        np.save(args.outdir / f'chain_{cid}_logpost.npy', res['logpost'])
        np.save(args.outdir / f'chain_{cid}_blobs.npy', res['blobs'])
        df_chain = pd.DataFrame(res['samples'], columns=['beta0', 'nu2', 'x_ini', 'h', 'omega_b', 'omega_c'])
        df_chain['logpost'] = res['logpost']
        for i, col in enumerate(_BLOB_COLS):
            df_chain[col] = res['blobs'][:, i]
        df_chain.to_csv(args.outdir / f'chain_{cid}.csv', index=False)

    # Flatten post burn-in
    chains = np.stack([r['samples'][args.burn:] for r in results], axis=0)
    logps = np.concatenate([r['logpost'][args.burn:] for r in results])
    blobs = np.concatenate([r['blobs'][args.burn:] for r in results], axis=0)
    flat = chains.reshape((-1, chains.shape[-1]))
    df_flat = pd.DataFrame(flat, columns=['beta0', 'nu2', 'x_ini', 'h', 'omega_b', 'omega_c'])
    df_flat['logpost'] = logps
    for i, col in enumerate(_BLOB_COLS):
        df_flat[col] = blobs[:, i]
    df_flat.to_csv(args.outdir / 'flat_samples_with_blobs.csv', index=False)

    # Diagnostics
    Rhat = gelman_rubin(chains)
    accs = [r['acceptance'] for r in results]
    runtimes = [r['runtime_sec'] for r in results]
    idx_best = int(np.argmax(logps))
    best = df_flat.iloc[idx_best]
    # 16-50-84 quantiles on sampled parameters
    quants = {}
    for col in ['beta0', 'nu2', 'x_ini', 'h', 'omega_b', 'omega_c']:
        quants[col] = [float(x) for x in np.quantile(df_flat[col].to_numpy(dtype=float), [0.16, 0.5, 0.84])]

    summary = {
        'selection': selection,
        'priors': priors,
        'lcdm': lcdm,
        'nchains': args.nchains,
        'nsteps': args.nsteps,
        'burn': args.burn,
        'acceptance_mean': float(np.mean(accs)),
        'acceptance_per_chain': [float(a) for a in accs],
        'runtime_per_chain_sec': [float(t) for t in runtimes],
        'Rhat': {
            'beta0': float(Rhat[0]),
            'nu2': float(Rhat[1]),
            'x_ini': float(Rhat[2]),
            'h': float(Rhat[3]),
            'omega_b': float(Rhat[4]),
            'omega_c': float(Rhat[5]),
        },
        'best_fit_tuo': {
            'beta0': float(best['beta0']),
            'nu2': float(best['nu2']),
            'x_ini': float(best['x_ini']),
            'h': float(best['h']),
            'omega_b': float(best['omega_b']),
            'omega_c': float(best['omega_c']),
            'logpost': float(best['logpost']),
            'chi2_total': float(best['chi2_total']),
            'chi2_sn': float(best['chi2_sn']),
            'chi2_bao': float(best['chi2_bao']),
            'chi2_prior': float(best['chi2_prior']),
            'Omega_V0': float(best['Omega_V0']),
            'z_eq': float(best['z_eq']),
            'z_t': float(best['z_t']),
            'f_TUO_zstar': float(best['f_TUO_zstar']),
            'r_d_Mpc': float(best['r_d_Mpc']),
            'z_drag': float(best['z_drag']),
            'Omega_b0': float(best['Omega_b0']),
            'Omega_c0': float(best['Omega_c0']),
            'Omega_r0': float(best['Omega_r0']),
        },
        'delta_chi2_tuo_minus_lcdm': float(best['chi2_total'] - lcdm['chi2_total']),
        'delta_chi2eff_tuo_minus_lcdm': float((best['chi2_total'] + best['chi2_prior']) - lcdm['chi2_posterior_effective']),
        'posterior_quantiles_16_50_84': quants,
    }
    txt = json.dumps(summary, indent=2)
    print(txt)
    (args.outdir / 'summary.json').write_text(txt)


if __name__ == '__main__':
    main()
