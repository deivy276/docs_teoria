#!/usr/bin/env python3
"""
Postprocesado para corridas TUO (Fase 4/Fase 5).

Lee flat_samples_with_blobs.csv y opcionalmente summary.json, y genera:
- corner plot de parametros
- histogramas de observables derivados
- CSV con puntos fisicamente aceptables rankeados por chi2
- resumen JSON con cuantiles y mejor muestra

No depende de corner; usa matplotlib puro para facilitar portabilidad.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_PARAM_COLS = ["beta0", "nu2", "x_ini"]
DEFAULT_OBS_COLS = [
    "chi2_total",
    "chi2",
    "delta_chi2",
    "chi2_sn",
    "chi2_bao",
    "Omega_V0",
    "z_eq",
    "z_t",
    "f_TUO_zstar",
    "r_d_Mpc",
    "z_drag",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Postprocesado para flat_samples_with_blobs.csv de la TUO")
    p.add_argument("--samples", required=True, help="Ruta a flat_samples_with_blobs.csv")
    p.add_argument("--summary", default=None, help="Ruta a summary.json (opcional)")
    p.add_argument("--outdir", default=None, help="Directorio de salida (default: junto al CSV)")
    p.add_argument("--params", nargs="*", default=None, help="Columnas de parámetros a usar")
    p.add_argument("--obs", nargs="*", default=None, help="Columnas de observables a graficar")
    p.add_argument("--bins", type=int, default=40, help="Número de bins para histogramas")
    p.add_argument("--max-filtered", type=int, default=5000, help="Máximo de filas a guardar en el CSV filtrado")
    p.add_argument("--zt-min", type=float, default=0.5)
    p.add_argument("--zt-max", type=float, default=0.8)
    p.add_argument("--zeq-min", type=float, default=3000.0)
    p.add_argument("--zeq-max", type=float, default=3600.0)
    p.add_argument("--ftuo-max", type=float, default=1e-2)
    p.add_argument("--delta-chi2-max", type=float, default=20.0, help="Corte blando para puntos fisicamente viables")
    return p.parse_args()



def bestfit_from_summary(summary: Dict) -> Dict:
    if not summary:
        return {}
    for key in ["best_fit_tuo", "best_fit"]:
        if key in summary and isinstance(summary[key], dict):
            return summary[key]
    if "mcmc" in summary and isinstance(summary["mcmc"], dict):
        bf = summary["mcmc"].get("best_fit", {})
        if isinstance(bf, dict):
            return bf
    return {}



def lcdm_chi2_from_summary(summary: Dict) -> float | None:
    if not summary:
        return None
    lcdm = summary.get("lcdm", {})
    if isinstance(lcdm, dict):
        if "chi2_total" in lcdm:
            return float(lcdm["chi2_total"])
        if "chi2" in lcdm:
            return float(lcdm["chi2"])
    return None



def infer_chi2_column(df: pd.DataFrame) -> str | None:
    for c in ["chi2_total", "chi2"]:
        if c in df.columns:
            return c
    return None



def add_delta_chi2(df: pd.DataFrame, summary: Dict | None) -> pd.DataFrame:
    if "delta_chi2" in df.columns:
        return df
    chi2_col = infer_chi2_column(df)
    if chi2_col is None:
        return df
    lcdm_chi2 = lcdm_chi2_from_summary(summary or {})
    if lcdm_chi2 is None:
        # Usa el mínimo de la cadena como referencia interna si no hay summary.
        ref = float(df[chi2_col].min())
    else:
        ref = lcdm_chi2
    df = df.copy()
    df["delta_chi2"] = df[chi2_col] - ref
    return df



def weighted_quantiles(values: np.ndarray, probs: Sequence[float]) -> np.ndarray:
    return np.quantile(values, probs)



def density_levels(H: np.ndarray, probs: Sequence[float] = (0.68, 0.95)) -> List[float]:
    flat = np.asarray(H, dtype=float).ravel()
    flat = flat[np.isfinite(flat)]
    flat = flat[flat > 0]
    if flat.size == 0:
        return []
    order = np.sort(flat)[::-1]
    csum = np.cumsum(order)
    csum /= csum[-1]
    levels = []
    for p in probs:
        idx = np.searchsorted(csum, p)
        idx = min(idx, order.size - 1)
        levels.append(order[idx])
    # contour requiere niveles crecientes
    return sorted(set(levels))



def robust_range(arr: np.ndarray, qlo: float = 0.001, qhi: float = 0.999, pad: float = 0.05) -> Tuple[float, float]:
    lo, hi = np.quantile(arr, [qlo, qhi])
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo, hi = np.nanmin(arr), np.nanmax(arr)
    if hi <= lo:
        eps = abs(lo) * 1e-3 + 1e-6
        return lo - eps, hi + eps
    width = hi - lo
    return lo - pad * width, hi + pad * width





def hist_safe(ax, arr: np.ndarray, bins: int, xr: Tuple[float, float], **kwargs) -> None:
    lo, hi = xr
    width = hi - lo
    if not np.isfinite(width) or width <= max(1e-14, 1e-12 * max(1.0, abs(lo), abs(hi))):
        center = float(np.nanmedian(arr))
        span = max(1e-6, 1e-6 * max(1.0, abs(center)))
        ax.hist(arr, bins=1, range=(center - span, center + span), **kwargs)
    else:
        ax.hist(arr, bins=bins, range=xr, **kwargs)

def make_corner_plot(df: pd.DataFrame, params: List[str], best: Dict, outpath: Path, bins: int = 40) -> None:
    n = len(params)
    fig, axes = plt.subplots(n, n, figsize=(3.4 * n, 3.4 * n))
    if n == 1:
        axes = np.array([[axes]])

    arrays = {p: df[p].to_numpy(dtype=float) for p in params}
    ranges = {p: robust_range(arrays[p]) for p in params}

    for i, py in enumerate(params):
        for j, px in enumerate(params):
            ax = axes[i, j]
            if i < j:
                ax.axis("off")
                continue
            x = arrays[px]
            if i == j:
                q16, q50, q84 = weighted_quantiles(x, [0.16, 0.5, 0.84])
                xr = ranges[px]
                hist_safe(ax, x, bins, xr, color="#4c72b0", alpha=0.75, density=True)
                ax.axvline(q50, color="black", lw=1.5)
                ax.axvline(q16, color="black", lw=1, ls="--")
                ax.axvline(q84, color="black", lw=1, ls="--")
                if px in best:
                    ax.axvline(float(best[px]), color="#dd8452", lw=1.5)
                ax.set_xlim(*ranges[px])
                title = f"{px}\n{q50:.4g} (+{q84-q50:.3g}/-{q50-q16:.3g})"
                ax.set_title(title, fontsize=10)
            else:
                y = arrays[py]
                xr = ranges[px]
                yr = ranges[py]
                H, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[xr, yr])
                xc = 0.5 * (xedges[1:] + xedges[:-1])
                yc = 0.5 * (yedges[1:] + yedges[:-1])
                X, Y = np.meshgrid(xc, yc)
                ax.hist2d(x, y, bins=bins, range=[xr, yr], cmap="Blues")
                levels = density_levels(H.T)
                if len(levels) >= 1:
                    ax.contour(X, Y, H.T, levels=levels, colors=["#1f1f1f"] * len(levels), linewidths=1.0)
                ax.plot(np.median(x), np.median(y), marker="s", ms=4, color="black")
                if px in best and py in best:
                    ax.plot(float(best[px]), float(best[py]), marker="*", ms=9, color="#dd8452")
                ax.set_xlim(*xr)
                ax.set_ylim(*yr)
            if i == n - 1:
                ax.set_xlabel(px)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(py)
            elif i > 0:
                ax.set_yticklabels([])
    fig.suptitle("TUO — Corner plot de parámetros", y=0.995, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(outpath, dpi=180)
    plt.close(fig)



def make_observables_plot(df: pd.DataFrame, obs_cols: List[str], best: Dict, outpath: Path, bins: int = 40) -> None:
    cols = [c for c in obs_cols if c in df.columns]
    if not cols:
        return
    n = len(cols)
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, cols):
        arr = df[c].to_numpy(dtype=float)
        q16, q50, q84 = weighted_quantiles(arr, [0.16, 0.5, 0.84])
        xr = robust_range(arr)
        hist_safe(ax, arr, bins, xr, color="#55a868", alpha=0.8, density=True)
        ax.axvline(q50, color="black", lw=1.5)
        ax.axvline(q16, color="black", lw=1, ls="--")
        ax.axvline(q84, color="black", lw=1, ls="--")
        if c in best:
            ax.axvline(float(best[c]), color="#c44e52", lw=1.5)
        ax.set_title(f"{c}\n{q50:.4g} (+{q84-q50:.3g}/-{q50-q16:.3g})", fontsize=10)
        ax.set_xlim(*xr)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("TUO — Histogramas de observables", y=0.995, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(outpath, dpi=180)
    plt.close(fig)



def apply_physical_filter(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    filt = np.ones(len(df), dtype=bool)
    if "z_t" in df.columns:
        filt &= (df["z_t"] >= args.zt_min) & (df["z_t"] <= args.zt_max)
    if "z_eq" in df.columns:
        filt &= (df["z_eq"] >= args.zeq_min) & (df["z_eq"] <= args.zeq_max)
    if "f_TUO_zstar" in df.columns:
        filt &= df["f_TUO_zstar"] <= args.ftuo_max
    if "delta_chi2" in df.columns:
        filt &= df["delta_chi2"] <= args.delta_chi2_max
    out = df.loc[filt].copy()
    chi2_col = infer_chi2_column(out)
    if chi2_col is not None:
        out = out.sort_values(chi2_col, ascending=True)
    return out.head(args.max_filtered)



def build_summary(df: pd.DataFrame, params: List[str], obs_cols: List[str], best: Dict, summary_in: Dict | None) -> Dict:
    out: Dict = {}
    out["nsamples"] = int(len(df))
    out["param_cols"] = params
    out["obs_cols"] = [c for c in obs_cols if c in df.columns]
    out["best_fit_from_summary"] = best
    chi2_col = infer_chi2_column(df)
    if chi2_col is not None:
        idx = int(df[chi2_col].astype(float).idxmin())
        out["best_sample_row_index"] = idx
        out["best_sample"] = {k: float(df.loc[idx, k]) if np.issubdtype(df[k].dtype, np.number) else df.loc[idx, k] for k in df.columns}
    quantiles: Dict[str, Dict[str, float]] = {}
    for c in params + [x for x in obs_cols if x in df.columns]:
        arr = df[c].to_numpy(dtype=float)
        q16, q50, q84 = weighted_quantiles(arr, [0.16, 0.5, 0.84])
        quantiles[c] = {
            "q16": float(q16),
            "q50": float(q50),
            "q84": float(q84),
        }
    out["quantiles"] = quantiles
    if summary_in:
        out["input_summary_excerpt"] = {
            k: summary_in.get(k) for k in ["lcdm", "mcmc", "Rhat", "posterior_quantiles_16_50_84", "best_fit_tuo"] if k in summary_in
        }
    return out



def main() -> None:
    args = parse_args()
    samples_path = Path(args.samples)
    if not samples_path.exists():
        raise FileNotFoundError(samples_path)
    outdir = Path(args.outdir) if args.outdir else samples_path.parent / "postprocess"
    outdir.mkdir(parents=True, exist_ok=True)

    summary = None
    if args.summary:
        with open(args.summary, "r", encoding="utf-8") as f:
            summary = json.load(f)

    df = pd.read_csv(samples_path)
    df = add_delta_chi2(df, summary)

    param_cols = args.params if args.params else [c for c in DEFAULT_PARAM_COLS if c in df.columns]
    if not param_cols:
        raise ValueError("No se encontraron columnas de parámetros en el CSV.")
    obs_cols = args.obs if args.obs else [c for c in DEFAULT_OBS_COLS if c in df.columns]

    best = bestfit_from_summary(summary or {})

    corner_path = outdir / "tuo_phase5_corner.png"
    obs_path = outdir / "tuo_phase5_observables.png"
    filt_csv = outdir / "tuo_phase5_physically_acceptable_ranked.csv"
    summary_out = outdir / "tuo_phase5_postprocess_summary.json"

    make_corner_plot(df, param_cols, best, corner_path, bins=args.bins)
    make_observables_plot(df, obs_cols, best, obs_path, bins=args.bins)

    filt = apply_physical_filter(df, args)
    filt.to_csv(filt_csv, index=False)

    summary_obj = build_summary(df, param_cols, obs_cols, best, summary)
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(summary_obj, f, indent=2)

    print(json.dumps({
        "samples": str(samples_path),
        "outdir": str(outdir),
        "corner": str(corner_path),
        "observables": str(obs_path),
        "filtered_csv": str(filt_csv),
        "summary": str(summary_out),
        "n_filtered": int(len(filt)),
    }, indent=2))


if __name__ == "__main__":
    main()
