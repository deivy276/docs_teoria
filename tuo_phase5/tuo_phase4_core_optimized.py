import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def dec(fn):
            return fn
        return dec


@njit(cache=True)
def _beta_eff_numba(N: float, beta0: float, N_c: float, s: float) -> float:
    return 0.5 * beta0 * (1.0 + math.tanh(s * (N - N_c)))


@njit(cache=True)
def _rhs_numba_scalar(N: float, state: np.ndarray,
                      beta0: float, N_c: float, s: float,
                      nu2: float, lam4: float, xi_r: float,
                      Omega_V0: float) -> Tuple[bool, np.ndarray]:
    x = state[0]
    y = state[1]
    b = state[2]
    c = state[3]
    r = state[4]

    x2 = x * x
    y2 = y * y
    denom = 1.0 - y2 / 6.0
    if denom <= 0.0:
        return False, np.zeros(5)

    U = Omega_V0 + 0.5 * nu2 * x2 + 0.25 * lam4 * x2 * x2 + 0.5 * xi_r * r * x2
    num = b + c + r + U
    if num <= 0.0:
        return False, np.zeros(5)

    e2 = num / denom
    inv_e2 = 1.0 / e2
    om_m = (b + c) * inv_e2
    om_r = r * inv_e2
    dlnE = -0.5 * y2 - 1.5 * om_m - 2.0 * om_r

    beff = _beta_eff_numba(N, beta0, N_c, s)
    dUdx = nu2 * x + lam4 * x2 * x + xi_r * r * x

    out = np.empty(5)
    out[0] = y
    out[1] = -(3.0 + dlnE) * y - 3.0 * dUdx * inv_e2 + 3.0 * beff * c * inv_e2
    out[2] = -3.0 * b
    out[3] = -(3.0 + beff * y) * c
    out[4] = -4.0 * r
    return True, out


@njit(cache=True)
def _rk4_step_numba(N: float, state: np.ndarray, h: float,
                    beta0: float, N_c: float, s: float,
                    nu2: float, lam4: float, xi_r: float,
                    Omega_V0: float) -> Tuple[bool, np.ndarray]:
    ok1, k1 = _rhs_numba_scalar(N, state, beta0, N_c, s, nu2, lam4, xi_r, Omega_V0)
    if not ok1:
        return False, state
    s2 = state + 0.5 * h * k1
    ok2, k2 = _rhs_numba_scalar(N + 0.5 * h, s2, beta0, N_c, s, nu2, lam4, xi_r, Omega_V0)
    if not ok2:
        return False, state
    s3 = state + 0.5 * h * k2
    ok3, k3 = _rhs_numba_scalar(N + 0.5 * h, s3, beta0, N_c, s, nu2, lam4, xi_r, Omega_V0)
    if not ok3:
        return False, state
    s4 = state + h * k3
    ok4, k4 = _rhs_numba_scalar(N + h, s4, beta0, N_c, s, nu2, lam4, xi_r, Omega_V0)
    if not ok4:
        return False, state
    new_state = state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return True, new_state


@njit(cache=True)
def _integrate_final_numba(N_ini: float, state_ini: np.ndarray,
                           beta0: float, N_c: float, s: float,
                           nu2: float, lam4: float, xi_r: float,
                           Omega_V0: float, nsteps: int) -> Tuple[bool, np.ndarray]:
    state = state_ini.copy()
    N = N_ini
    h_base = (0.0 - N_ini) / nsteps
    for _ in range(nsteps):
        h = h_base
        if N + h > 0.0:
            h = -N
        ok, state = _rk4_step_numba(N, state, h, beta0, N_c, s, nu2, lam4, xi_r, Omega_V0)
        if not ok:
            return False, state
        N += h
    return True, state


@njit(cache=True)
def _integrate_grid_numba(N_ini: float, state_ini: np.ndarray,
                          beta0: float, N_c: float, s: float,
                          nu2: float, lam4: float, xi_r: float,
                          Omega_V0: float, npts: int) -> Tuple[bool, np.ndarray, np.ndarray]:
    N_arr = np.empty(npts)
    Y_arr = np.empty((5, npts))
    state = state_ini.copy()
    N = N_ini
    h = (0.0 - N_ini) / (npts - 1)
    N_arr[0] = N
    for j in range(5):
        Y_arr[j, 0] = state[j]
    for i in range(1, npts):
        ok, state = _rk4_step_numba(N, state, h, beta0, N_c, s, nu2, lam4, xi_r, Omega_V0)
        if not ok:
            return False, N_arr[:i], Y_arr[:, :i]
        N += h
        N_arr[i] = N
        for j in range(5):
            Y_arr[j, i] = state[j]
    return True, N_arr, Y_arr


@dataclass
class ScalarBackground:
    N: np.ndarray
    Y: np.ndarray
    backend: str


class TUOPhase4CoreOptimized:
    """
    Núcleo optimizado Fase 4 para TUO:
      - helpers escalares (math, no numpy overhead en RHS)
      - warm-start para Omega_V0 (persiste entre update/solve si se reutiliza el objeto)
      - ruta numba-friendly para integraciones repetidas del shooting
      - base lista para usar en likelihoods y samplers largos
    Para aprovechar warm-start en MCMC, conviene REUTILIZAR la misma instancia por cadena/worker.
    """

    __slots__ = (
        'Omega_b0', 'Omega_c0', 'Omega_r0', 'beta0', 'nu2', 'lam4', 'xi_r', 'z_c', 's',
        'x_ini', 'y_ini', 'z_ini', 'rtol', 'atol', 'max_step', 'method', 'z_star', 'Omega_gamma0',
        'N_c', '_N_ini', '_Y_ini', '_warm_root', '_warm_bracket', 'last_shoot_evals',
        'numba_shoot_steps', 'numba_grid_npts'
    )

    def __init__(self, params=None):
        self.Omega_b0 = 0.05
        self.Omega_c0 = 0.25
        self.Omega_r0 = 9.2e-5
        self.beta0 = 0.05
        self.nu2 = 0.02
        self.lam4 = 0.0
        self.xi_r = 10.0
        self.z_c = 100.0
        self.s = 4.0
        self.x_ini = 0.2
        self.y_ini = 0.0
        self.z_ini = 1.0e5
        self.rtol = 1e-8
        self.atol = 1e-10
        self.max_step = 0.1
        self.method = 'DOP853'
        self.z_star = 1100.0
        self.Omega_gamma0 = 5.38e-5
        self.N_c = -math.log1p(self.z_c)
        self._N_ini = -math.log1p(self.z_ini)
        self._Y_ini = None
        self._warm_root = None
        self._warm_bracket = None
        self.last_shoot_evals = 0
        self.numba_shoot_steps = 4096
        self.numba_grid_npts = 5000
        if params:
            self.update(**params)
        else:
            self._refresh_cache()

    def update(self, **params):
        for k, v in params.items():
            if not hasattr(self, k):
                raise AttributeError(f'Unknown parameter: {k}')
            setattr(self, k, float(v) if isinstance(v, (int, float, np.floating)) else v)
        self._refresh_cache()
        return self

    def _refresh_cache(self):
        self.N_c = -math.log1p(self.z_c)
        self._N_ini = -math.log1p(self.z_ini)
        b_ini = self.Omega_b0 * math.exp(-3.0 * self._N_ini)
        c_ini = self.Omega_c0 * math.exp(-3.0 * self._N_ini)
        r_ini = self.Omega_r0 * math.exp(-4.0 * self._N_ini)
        self._Y_ini = np.array([self.x_ini, self.y_ini, b_ini, c_ini, r_ini], dtype=np.float64)

    # ----- scalar helpers -----
    def beta_eff(self, N: float) -> float:
        return 0.5 * self.beta0 * (1.0 + math.tanh(self.s * (N - self.N_c)))

    def U_tilde(self, x: float, r: float, Omega_V0: float) -> float:
        x2 = x * x
        return Omega_V0 + 0.5 * self.nu2 * x2 + 0.25 * self.lam4 * x2 * x2 + 0.5 * self.xi_r * r * x2

    def dUdx(self, x: float, r: float) -> float:
        x2 = x * x
        return self.nu2 * x + self.lam4 * x2 * x + self.xi_r * r * x

    def E2_scalar(self, x: float, y: float, b: float, c: float, r: float, Omega_V0: float) -> float:
        y2 = y * y
        denom = 1.0 - y2 / 6.0
        if denom <= 0.0:
            raise ValueError('No físico: 1 - y^2/6 <= 0')
        num = b + c + r + self.U_tilde(x, r, Omega_V0)
        if num <= 0.0:
            raise ValueError('No físico: numerador de E^2 <= 0')
        return num / denom

    def rhs_scalar(self, N: float, Y: np.ndarray, Omega_V0: float):
        x = float(Y[0]); y = float(Y[1]); b = float(Y[2]); c = float(Y[3]); r = float(Y[4])
        x2 = x * x
        y2 = y * y
        denom = 1.0 - y2 / 6.0
        if denom <= 0.0:
            raise ValueError('No físico: 1 - y^2/6 <= 0')
        U = Omega_V0 + 0.5 * self.nu2 * x2 + 0.25 * self.lam4 * x2 * x2 + 0.5 * self.xi_r * r * x2
        num = b + c + r + U
        if num <= 0.0:
            raise ValueError('No físico: numerador de E^2 <= 0')
        e2 = num / denom
        inv_e2 = 1.0 / e2
        om_m = (b + c) * inv_e2
        om_r = r * inv_e2
        dlnE = -0.5 * y2 - 1.5 * om_m - 2.0 * om_r
        beff = 0.5 * self.beta0 * (1.0 + math.tanh(self.s * (N - self.N_c)))
        dUdx = self.nu2 * x + self.lam4 * x2 * x + self.xi_r * r * x
        return (
            y,
            -(3.0 + dlnE) * y - 3.0 * dUdx * inv_e2 + 3.0 * beff * c * inv_e2,
            -3.0 * b,
            -(3.0 + beff * y) * c,
            -4.0 * r,
        )

    # ----- scipy path -----
    def _rhs_scipy(self, N, Y, Omega_V0):
        return self.rhs_scalar(N, Y, Omega_V0)

    def _solve_with_OmegaV0_scipy(self, Omega_V0: float, dense_output: bool = True):
        sol = solve_ivp(
            self._rhs_scipy,
            (self._N_ini, 0.0),
            self._Y_ini,
            args=(Omega_V0,),
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
            dense_output=dense_output,
            max_step=self.max_step,
        )
        return sol

    # ----- numba path for repeated objective evaluations -----
    def _solve_final_numba(self, Omega_V0: float):
        ok, state = _integrate_final_numba(
            self._N_ini, self._Y_ini,
            self.beta0, self.N_c, self.s,
            self.nu2, self.lam4, self.xi_r,
            Omega_V0, int(self.numba_shoot_steps)
        )
        return ok, state

    def _solve_grid_numba(self, Omega_V0: float, npts: Optional[int] = None):
        if npts is None:
            npts = int(self.numba_grid_npts)
        ok, N, Y = _integrate_grid_numba(
            self._N_ini, self._Y_ini,
            self.beta0, self.N_c, self.s,
            self.nu2, self.lam4, self.xi_r,
            Omega_V0, int(npts)
        )
        return ok, ScalarBackground(N=np.asarray(N), Y=np.asarray(Y), backend='numba-rk4')

    # ----- objective / warm-start root finding -----
    def _objective_E0_fast(self, Omega_V0: float, use_numba: bool = True) -> float:
        self.last_shoot_evals += 1
        if use_numba and NUMBA_AVAILABLE:
            ok, state = self._solve_final_numba(Omega_V0)
            if not ok:
                return math.nan
            x, y, b, c, r = map(float, state)
        else:
            try:
                sol = self._solve_with_OmegaV0_scipy(Omega_V0, dense_output=False)
                if sol.status < 0:
                    return math.nan
                x, y, b, c, r = map(float, sol.y[:, -1])
            except Exception:
                return math.nan
        try:
            return self.E2_scalar(x, y, b, c, r, Omega_V0) - 1.0
        except Exception:
            return math.nan

    def shoot_OmegaV0(self, bracket: Tuple[float, float] = (-2.0, 2.0), nscan: int = 17,
                       use_numba: bool = True, xtol: float = 1e-10) -> float:
        self.last_shoot_evals = 0
        memo = {}

        def f(v: float) -> float:
            key = round(float(v), 14)
            if key not in memo:
                memo[key] = self._objective_E0_fast(float(v), use_numba=use_numba)
            return memo[key]

        # 1) Try previous bracket directly
        if self._warm_bracket is not None:
            a, b = self._warm_bracket
            fa = f(a); fb = f(b)
            if math.isfinite(fa) and math.isfinite(fb) and fa * fb < 0.0:
                root = brentq(f, a, b, xtol=xtol, rtol=xtol, maxiter=80)
                self._warm_root = root
                self._warm_bracket = (a, b)
                return float(root)

        # 2) Local expanding bracket around last root (warm-start)
        if self._warm_root is not None:
            center = float(self._warm_root)
            fc = f(center)
            if math.isfinite(fc) and abs(fc) < 1e-12:
                return center
            delta = max(1e-4, 0.02 * max(1.0, abs(center)))
            for _ in range(10):
                a = center - delta
                b = center + delta
                fa = f(a); fb = f(b)
                if math.isfinite(fa) and math.isfinite(fb) and fa * fb < 0.0:
                    root = brentq(f, a, b, xtol=xtol, rtol=xtol, maxiter=80)
                    self._warm_root = root
                    self._warm_bracket = (a, b)
                    return float(root)
                delta *= 2.0

        # 3) Short global scan fallback
        grid = np.linspace(bracket[0], bracket[1], int(nscan))
        vals = [f(float(g)) for g in grid]
        root_bracket = None
        for a, fa, b, fb in zip(grid[:-1], vals[:-1], grid[1:], vals[1:]):
            if math.isfinite(fa) and math.isfinite(fb) and fa * fb < 0.0:
                root_bracket = (float(a), float(b))
                break
        if root_bracket is None:
            raise RuntimeError('No se encontró un bracket para Omega_V0')
        root = brentq(f, root_bracket[0], root_bracket[1], xtol=xtol, rtol=xtol, maxiter=120)
        self._warm_root = float(root)
        self._warm_bracket = root_bracket
        return float(root)

    def solve(self, final_backend: str = 'scipy', use_numba_shoot: bool = True,
              grid_npts: Optional[int] = None):
        Omega_V0 = self.shoot_OmegaV0(use_numba=use_numba_shoot)
        if final_backend == 'scipy':
            sol = self._solve_with_OmegaV0_scipy(Omega_V0, dense_output=True)
        elif final_backend == 'numba':
            ok, sol = self._solve_grid_numba(Omega_V0, npts=grid_npts)
            if not ok:
                raise RuntimeError('Numba grid solve failed for final backend')
        else:
            raise ValueError("final_backend must be 'scipy' or 'numba'")
        return Omega_V0, sol

    def sample_background(self, Omega_V0: float, sol: Union[ScalarBackground, object],
                          z_max: Optional[float] = None, npts: int = 4000):
        z_max = self.z_ini if z_max is None else z_max
        if isinstance(sol, ScalarBackground):
            N = sol.N
            x, y, b, c, r = sol.Y
        else:
            N_min = -math.log1p(z_max)
            N = np.linspace(N_min, 0.0, int(npts))
            x, y, b, c, r = sol.sol(N)
        # vector post-processing outside tight RHS loop is fine with numpy
        x = np.asarray(x); y = np.asarray(y); b = np.asarray(b); c = np.asarray(c); r = np.asarray(r)
        x2 = x * x
        U = Omega_V0 + 0.5 * self.nu2 * x2 + 0.25 * self.lam4 * x2 * x2 + 0.5 * self.xi_r * r * x2
        e2 = (b + c + r + U) / (1.0 - y * y / 6.0)
        E = np.sqrt(e2)
        z = np.exp(-N) - 1.0
        Om_b = b / e2
        Om_c = c / e2
        Om_m = (b + c) / e2
        Om_r = r / e2
        Om_phi = (e2 * y * y / 6.0 + U) / e2
        dlnE_dN = -0.5 * y * y - 1.5 * Om_m - 2.0 * Om_r
        q_dec = -1.0 - dlnE_dN
        w_eff = -1.0 - (2.0 / 3.0) * dlnE_dN
        return {
            'N': N, 'z': z, 'x': x, 'y': y, 'E': E, 'E2': e2,
            'Omega_b': Om_b, 'Omega_c': Om_c, 'Omega_m': Om_m, 'Omega_r': Om_r,
            'Omega_phi': Om_phi, 'q': q_dec, 'w_eff': w_eff,
            'b': b, 'c': c, 'r': r,
        }

    @staticmethod
    def _zero_crossing_z(z, f):
        idx = np.where(f[:-1] * f[1:] <= 0.0)[0]
        if len(idx) == 0:
            return None
        i = idx[0]
        z1, z2 = z[i], z[i + 1]
        f1, f2 = f[i], f[i + 1]
        if f2 == f1:
            return float(z1)
        return float(z1 + (0.0 - f1) * (z2 - z1) / (f2 - f1))

    def diagnostics(self, bg):
        z = bg['z']; q = bg['q']; b = bg['b']; c = bg['c']; r = bg['r']; Om_phi = bg['Omega_phi']
        z_t = self._zero_crossing_z(z, q)
        z_eq = self._zero_crossing_z(z, b + c - r)
        j = int(np.argmin(np.abs(z - self.z_star)))
        return {
            'z_t': z_t,
            'z_eq': z_eq,
            'f_TUO_zstar': float(Om_phi[j]),
            'Omega_m_zstar': float(bg['Omega_m'][j]),
            'Omega_r_zstar': float(bg['Omega_r'][j]),
        }


def benchmark_once(params=None, z_max=1e3, npts=2000):
    import time
    params = params or {
        'Omega_b0': 0.05, 'Omega_c0': 0.25, 'Omega_r0': 9.2e-5,
        'beta0': 0.05, 'nu2': 0.02, 'lam4': 0.0, 'xi_r': 10.0,
        'z_c': 100.0, 's': 4.0, 'x_ini': 0.2, 'y_ini': 0.0, 'z_ini': 1e5,
    }
    model = TUOPhase4CoreOptimized(params)
    t0 = time.perf_counter()
    OmV, sol = model.solve(final_backend='scipy', use_numba_shoot=True)
    t1 = time.perf_counter()
    bg = model.sample_background(OmV, sol, z_max=z_max, npts=npts)
    diag = model.diagnostics(bg)
    return {
        'Omega_V0': OmV,
        'solve_seconds': t1 - t0,
        'shoot_evals': model.last_shoot_evals,
        **diag,
    }


if __name__ == '__main__':
    import json
    out = benchmark_once()
    print(json.dumps(out, indent=2))
