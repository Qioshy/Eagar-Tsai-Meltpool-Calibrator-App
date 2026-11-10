# ======================================================================
# SECTION 1-8 of et_modeler.py  (engine, geometry, materials, calibration,
# plots, export, self-tests).  UI (section 9) is appended afterwards.
# ======================================================================
from __future__ import annotations
import io, json, math, os, re, sys, time, warnings
from dataclasses import dataclass, asdict, field, replace
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import roots_legendre

# ----------------------------------------------------------------------
# 1. Constants, units
# ----------------------------------------------------------------------
def _pick_data_dir() -> Path:
    """et_data next to the script if writable (NOT the case inside C:\\Program Files), else ~/.et_modeler."""
    base = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    try:
        d = base / "et_data"; d.mkdir(exist_ok=True)
        (d / ".w").write_text("x"); (d / ".w").unlink()
        return d
    except OSError:
        d = Path.home() / ".et_modeler"; d.mkdir(exist_ok=True)
        return d


DATA_DIR = _pick_data_dir()
POWER_UNITS = {"W": 1.0, "kW": 1e3}
SPEED_UNITS = {"mm/s": 1e-3, "m/s": 1.0}
LENGTH_UNITS = {"um": 1e-6, "mm": 1e-3, "m": 1.0}
BEAM_DEFS = {
    "D4sigma diameter": 0.25,      # sigma = D/4
    "1/e^2 diameter": 0.25,        # D = 2w, sigma = w/2 = D/4
    "1/e^2 radius": 0.5,           # sigma = w/2
    "Gaussian sigma": 1.0,
}


def to_si(value: float, unit: str, table: dict) -> float:
    return float(value) * table[unit]


# ----------------------------------------------------------------------
# 2. Data classes
# ----------------------------------------------------------------------
@dataclass
class Material:
    name: str = "Custom"
    k: float = 25.0            # W/m/K
    rho: float = 7600.0        # kg/m3
    cp: float = 700.0          # J/kg/K
    T_liq: float = 1600.0      # K
    T_sol: Optional[float] = None
    T_boil: Optional[float] = None
    latent_heat: float = 0.0   # J/kg, only used for normalized enthalpy
    source: str = "user input - VERIFY"      # general note
    sources: dict = field(default_factory=dict)   # per-property source: k, rho, cp, T_liq, T_sol, T_boil, latent_heat

    @property
    def alpha(self) -> float:
        return self.k / (self.rho * self.cp)

    def validate(self, T0: float) -> list[str]:
        errs = []
        for n in ("k", "rho", "cp", "T_liq"):
            if not getattr(self, n) > 0:
                errs.append(f"{n} must be > 0")
        if self.T_liq <= T0:
            errs.append("T_liquidus must be above substrate/preheat temperature")
        if self.T_sol is not None and self.T_sol > self.T_liq:
            errs.append("T_solidus must not exceed T_liquidus")
        if self.T_boil is not None and self.T_boil <= self.T_liq:
            errs.append("T_boil must exceed T_liquidus")
        return errs


@dataclass
class Process:
    P: float = 200.0           # W (incident)
    v: float = 1.0             # m/s
    beam: float = 80e-6        # in the unit given by beam_def (metres)
    beam_def: str = "D4sigma diameter"
    eta: float = 0.4
    T0: float = 298.0          # K
    layer: float = 40e-6
    hatch: float = 100e-6
    beam_scale: float = 1.0    # calibration factor on sigma

    @property
    def sigma(self) -> float:
        return self.beam * BEAM_DEFS[self.beam_def] * self.beam_scale

    def validate(self) -> list[str]:
        errs = []
        if self.P <= 0: errs.append("power must be > 0")
        if self.v <= 0: errs.append("scan speed must be > 0")
        if self.beam <= 0: errs.append("beam size must be > 0")
        if not (0 < self.eta <= 1): errs.append("absorptivity must be in (0, 1]")
        if self.T0 <= 0: errs.append("T0 must be > 0 K")
        return errs


@dataclass
class ModelSettings:
    npan: int = 48             # quadrature panels (accuracy ~1e-6 vs 160 panels)
    ng: int = 8                # Gauss nodes per panel
    nx_scan: int = 17          # x samples per pass for max width/depth search
    keyhole_dw_transition: float = 0.5   # editable heuristics, NOT universal
    keyhole_dw_keyhole: float = 1.0
    keyhole_norm_enthalpy: float = 6.0   # informational; verify definition (King et al. 2014)
    peak_T_warn_factor: float = 20.0     # flag peak T above this multiple of T_liq
    use_enthalpy_flag: bool = False
    k_scale: float = 1.0


# ----------------------------------------------------------------------
# 3. E-T engine
# ----------------------------------------------------------------------
_GL_CACHE: dict = {}


def _gl(ng: int):
    if ng not in _GL_CACHE:
        _GL_CACHE[ng] = roots_legendre(ng)
    return _GL_CACHE[ng]


def _nodes(S: float, s_lo: float, npan: int, ng: int):
    """Composite Gauss-Legendre nodes on [0, S] with geometric panel spacing.

    Geometric spacing (first main-grid edge s_lo) resolves both the near-source structure (scale s_lo) and the
    far-wake bump (scale sqrt(|x|/v)) when the two differ by orders of magnitude.
    """
    xg, wg = _gl(ng)
    # 8 extra geometric sub-panels in [s_lo/1000, s_lo] resolve the exp(-z^2/(4 alpha s^2)) switch-on when z is
    # small but non-zero (it sits below s_lo); the main grid then follows the s_lo rule above.
    edges = np.concatenate([[0.0], np.geomspace(s_lo * 1e-3, s_lo, 9), np.geomspace(s_lo, S, npan)[1:]])
    a, b = edges[:-1], edges[1:]
    s = (0.5 * (b - a)[:, None] * xg[None, :] + 0.5 * (a + b)[:, None]).ravel()
    w = (0.5 * (b - a)[:, None] * wg[None, :]).ravel()
    return s, w


def _prep(xmax: float, v: float, sigma: float, alpha: float, npan: int, ng: int):
    """Quadrature nodes/weights for all points whose |x| <= xmax."""
    u_dec = (180.0 * alpha + math.sqrt((180.0 * alpha) ** 2 + 360.0 * (v * sigma) ** 2)) / (2.0 * v * v)
    S = 1.1 * math.sqrt(xmax / v + u_dec)
    s_lo = 0.05 * min(sigma / math.sqrt(2.0 * alpha), math.sqrt(sigma / v))
    s, w = _nodes(S, min(s_lo, S / 50), npan, ng)
    s2 = s * s
    var = sigma ** 2 + 2.0 * alpha * s2
    return s2, 2.0 * w / var, var


def temperature_rise(x, y, z, q: float, v: float, sigma: float, k: float,
                     alpha: float, npan: int = 48, ng: int = 8, prep=None) -> np.ndarray:
    """T - T0 [K] for a Gaussian surface source on a semi-infinite solid.

    Beam frame: x<0 behind the beam, z>=0 into the solid.
    q = absorbed power [W]; sigma = Gaussian std of the source [m]
    (flux = q/(2 pi sigma^2) exp(-r^2/(2 sigma^2))).
    With t = s^2 (removes the t^-1/2 singularity):
      T-T0 = q sqrt(alpha)/(2 pi^1.5 k) * 2 * INT_0^inf ds/(sigma^2+2 alpha s^2)
             * exp(-z^2/(4 alpha s^2) - ((x+v s^2)^2 + y^2)/(2 (sigma^2+2 alpha s^2)))
    Verified against Rosenthal (sigma->0), scipy.quad and the eagar-tsai library.
    """
    x, y, z = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float), np.asarray(z, float))
    shape = x.shape
    xf, yf, zf = x.ravel(), y.ravel(), z.ravel()
    if prep is None:
        xmax = float(np.max(np.abs(xf))) if xf.size else 0.0
        prep = _prep(xmax, v, sigma, alpha, npan, ng)
    s2, wv, var = prep
    out = np.empty(xf.size)
    chunk = max(1, int(4_000_000 // s2.size))
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        for i0 in range(0, xf.size, chunk):
            sl = slice(i0, i0 + chunk)
            E = (-(zf[sl, None] ** 2) / (4.0 * alpha * s2[None, :])
                 - ((xf[sl, None] + v * s2[None, :]) ** 2 + yf[sl, None] ** 2) / (2.0 * var[None, :]))
            E = np.nan_to_num(E, nan=-np.inf)
            out[i0:i0 + chunk] = np.exp(E) @ wv
    pref = q * math.sqrt(alpha) / (2.0 * math.pi ** 1.5 * k)
    return (pref * out).reshape(shape)


class ETCase:
    """One E-T evaluation context (process + material + settings)."""

    def __init__(self, proc: Process, mat: Material, st: ModelSettings | None = None):
        self.p, self.m, self.st = proc, mat, st or ModelSettings()
        self.k = mat.k * self.st.k_scale
        self.alpha = self.k / (mat.rho * mat.cp)
        self.q = proc.eta * proc.P
        self.sig = proc.sigma
        self.theta_l = mat.T_liq - proc.T0

    def dT(self, x, y, z, prep=None):
        return temperature_rise(x, y, z, self.q, self.p.v, self.sig, self.k, self.alpha,
                                self.st.npan, self.st.ng, prep=prep)

    def prep(self, xmax: float):
        return _prep(xmax, self.p.v, self.sig, self.alpha, self.st.npan, self.st.ng)

    # --- 1-D root finding helpers -------------------------------------
    def _bisect_edge(self, xs: np.ndarray, axis: str, hi0: float, nit: int = 30) -> np.ndarray:
        """Edge of the liquidus isotherm along y (z=0) or z (y=0) for each x."""
        pre = self.prep(float(np.max(np.abs(xs))))
        def T(c):
            if axis == "y":
                return self.dT(xs, c, 0.0, prep=pre)
            return self.dT(xs, 0.0, c, prep=pre)
        hi = np.full(xs.shape, hi0)
        for _ in range(30):                       # grow bracket if needed
            if np.all(T(hi) < self.theta_l):
                break
            hi = np.where(T(hi) >= self.theta_l, hi * 2.0, hi)
        lo = np.zeros_like(hi)
        inside0 = T(lo) >= self.theta_l
        for _ in range(nit):
            mid = 0.5 * (lo + hi)
            hot = T(mid) >= self.theta_l
            lo = np.where(hot, mid, lo)
            hi = np.where(hot, hi, mid)
        return np.where(inside0, 0.5 * (lo + hi), 0.0)

    def _max_edge(self, x_lo: float, x_hi: float, axis: str, hi0: float):
        st = self.st
        xs = np.linspace(x_lo, x_hi, st.nx_scan)
        e = self._bisect_edge(xs, axis, hi0)
        i = int(np.argmax(e))
        for _ in range(2):                         # two refinement passes
            a = xs[max(i - 1, 0)]; b = xs[min(i + 1, xs.size - 1)]
            xs = np.linspace(a, b, st.nx_scan)
            e = self._bisect_edge(xs, axis, max(float(e.max()) * 1.5, hi0 * 0.1))
            i = int(np.argmax(e))
        return float(e[i]), float(xs[i])

    def _root_x(self, x_in: float, x_far: float) -> float:
        lo, hi = x_in, x_far
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if float(self.dT(mid, 0.0, 0.0)) >= self.theta_l:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def melt_pool(self) -> dict:
        """Melt pool geometry from the liquidus isotherm."""
        sig, v = self.sig, self.p.v
        # peak temperature on the surface centreline
        xs = np.concatenate([-np.geomspace(sig * 1e-3, 60 * sig, 220)[::-1], [0.0],
                             np.geomspace(sig * 1e-3, 6 * sig, 40)])
        Tc = self.dT(xs, 0.0, 0.0)
        ip = int(np.argmax(Tc)); xpk = float(xs[ip])
        # refine peak
        xx = np.linspace(xs[max(ip - 1, 0)], xs[min(ip + 1, xs.size - 1)], 41)
        Tp = self.dT(xx, 0.0, 0.0); j = int(np.argmax(Tp)); xpk = float(xx[j]); Tpk = float(Tp[j])
        out = dict(peak_T=self.p.T0 + Tpk, melted=bool(Tpk >= self.theta_l))
        nan = float("nan")
        if not out["melted"]:
            out.update(width=0.0, depth=0.0, length=0.0, head=0.0, tail=0.0, x_width=nan,
                       x_depth=nan, G=nan, cooling_rate=nan)
            return out
        xb = -2 * sig
        for _ in range(40):
            if float(self.dT(min(xb, xpk - 1e-12), 0, 0)) < self.theta_l: break
            xb *= 2
        xf = 2 * sig
        for _ in range(40):
            if float(self.dT(max(xf, xpk + 1e-12), 0, 0)) < self.theta_l: break
            xf *= 2
        x_rear = self._root_x(xpk, min(xb, xpk - 1e-12) if xb < xpk else xpk - 1e-12)
        x_front = self._root_x(xpk, max(xf, xpk + 1e-12))
        half_w, x_w = self._max_edge(x_rear, x_front, "y", 2 * sig)
        depth, x_d = self._max_edge(x_rear, x_front, "z", 2 * sig)
        h = 1e-3 * abs(x_rear)
        dTdx = (float(self.dT(x_rear - h, 0, 0)) - float(self.dT(x_rear + h, 0, 0))) / (2 * h)
        G = abs(dTdx)
        out.update(width=2 * half_w, depth=depth, length=x_front - x_rear,
                   head=x_front, tail=-x_rear, x_width=x_w, x_depth=x_d,
                   G=G, cooling_rate=v * G, x_rear=x_rear, x_front=x_front)
        return out


# ----------------------------------------------------------------------
# 4. Derived metrics / regime flags
# ----------------------------------------------------------------------
def energy_metrics(proc: Process, mat: Material) -> dict:
    lin = proc.P / proc.v                                       # J/m
    ved = proc.P / (proc.v * proc.hatch * proc.layer) if proc.hatch > 0 and proc.layer > 0 else float("nan")
    hs = mat.cp * (mat.T_liq - proc.T0) + mat.latent_heat
    sig_b = proc.beam * BEAM_DEFS[proc.beam_def] * 2.0           # beam radius ~ 2 sigma
    alpha = mat.alpha
    nh = proc.eta * proc.P / (math.pi * mat.rho * hs * math.sqrt(alpha * proc.v * sig_b ** 3))
    return dict(linear_energy_J_per_m=lin, VED_J_per_m3=ved, norm_enthalpy=nh)


def regime_flag(res: dict, em: dict, st: ModelSettings, mat: Material) -> str:
    if not res["melted"]:
        return "no melting"
    if res["width"] <= 0:
        return "no melting"
    dw = res["depth"] / res["width"]
    flags = []
    if dw >= st.keyhole_dw_keyhole or (st.use_enthalpy_flag and em["norm_enthalpy"] >= st.keyhole_norm_enthalpy):
        flags.append("KEYHOLE-suspect (depth unreliable)")
    elif dw >= st.keyhole_dw_transition:
        flags.append("transition (depth may be underpredicted)")
    else:
        flags.append("conduction")
    if mat.T_boil and res["peak_T"] > mat.T_boil:
        flags.append("T_peak > T_boil")
    if res["peak_T"] > st.peak_T_warn_factor * mat.T_liq:
        flags.append("extreme peak T: outside conduction-model validity")
    return "; ".join(flags)


def run_case(proc: Process, mat: Material, st: ModelSettings | None = None) -> dict:
    st = st or ModelSettings()
    r = ETCase(proc, mat, st).melt_pool()
    em = energy_metrics(proc, mat)
    r.update(em)
    r["dw_ratio"] = r["depth"] / r["width"] if r["width"] > 0 else float("nan")
    r["lw_ratio"] = r["length"] / r["width"] if r["width"] > 0 else float("nan")
    r["regime"] = regime_flag(r, em, st, mat)
    return r


# ----------------------------------------------------------------------
# 5. Materials
# ----------------------------------------------------------------------
# Deliberately NO built-in values for real alloys: every number must be supplied
# and sourced by the user. 'Generic example' exists only so the app starts.
def builtin_materials() -> dict[str, Material]:
    return {"Generic example (NOT a real alloy)": Material(
        name="Generic example (NOT a real alloy)", k=25.0, rho=7600.0, cp=700.0, T_liq=1600.0,
        source="placeholder for testing only",
        sources={n: "placeholder for testing only" for n in ("k", "rho", "cp", "T_liq", "T_sol", "T_boil", "latent_heat")})}


def load_user_materials() -> dict[str, Material]:
    f = DATA_DIR / "materials.json"
    if not f.exists():
        return {}
    try:
        return {n: Material(**d) for n, d in json.loads(f.read_text()).items()}
    except Exception:
        return {}


def save_user_material(m: Material) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    cur = load_user_materials()
    cur[m.name] = m
    (DATA_DIR / "materials.json").write_text(json.dumps({n: asdict(x) for n, x in cur.items()}, indent=2))


# ----------------------------------------------------------------------
# 6. Calibration
# ----------------------------------------------------------------------
EXP_COLS = ["power_W", "speed_m_s", "beam_um", "width_um", "depth_um", "length_um", "exclude"]


def predict_table(df: pd.DataFrame, base: Process, mat: Material, st: ModelSettings,
                  eta: float, beam_scale: float = 1.0) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        p = replace(base, P=float(r["power_W"]), v=float(r["speed_m_s"]), eta=eta, beam_scale=beam_scale)
        if "beam_um" in r and pd.notna(r["beam_um"]) and float(r["beam_um"]) > 0:
            p = replace(p, beam=float(r["beam_um"]) * 1e-6, beam_def=base.beam_def)
        g = run_case(p, mat, st)
        rows.append(dict(pred_width_um=g["width"] * 1e6, pred_depth_um=g["depth"] * 1e6,
                         pred_length_um=g["length"] * 1e6, regime=g["regime"]))
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


@dataclass
class CalResult:
    params: dict
    std_err: dict
    corr: Optional[np.ndarray]
    names: list
    rmse: dict
    mape: dict
    r2: dict
    table: pd.DataFrame
    cost: float
    n_used: int
    warnings: list
    opts: dict = field(default_factory=dict)     # weights, bounds, dw_max (needed by run_mcmc)


def _metrics(tab: pd.DataFrame) -> tuple[dict, dict, dict]:
    rmse, mape, r2 = {}, {}, {}
    for c in ("width", "depth", "length"):
        m, p = f"{c}_um", f"pred_{c}_um"
        if m in tab and tab[m].notna().sum() >= 1:
            s_ = tab[[m, p]].dropna()
            e = s_[p] - s_[m]
            rmse[c] = float(np.sqrt((e ** 2).mean()))
            mape[c] = float((e.abs() / s_[m]).mean() * 100)
            ss = ((s_[m] - s_[m].mean()) ** 2).sum()
            r2[c] = float(1 - (e ** 2).sum() / ss) if ss > 0 and len(s_) > 1 else float("nan")
    return rmse, mape, r2


def _filter_tracks(df: pd.DataFrame, dw_max: Optional[float]):
    """Drop user-excluded tracks (blank -> False) and, optionally, tracks above a depth/width cut-off."""
    d = df.copy()
    for c in EXP_COLS:
        if c not in d:
            d[c] = np.nan if c != "exclude" else False
    d["exclude"] = d["exclude"].map(_truthy)
    warn, use = [], ~d["exclude"]
    if dw_max is not None:
        kh = (d["depth_um"] / d["width_um"] >= dw_max).fillna(False)
        if kh.any():
            warn.append(f"{int(kh.sum())} track(s) with measured depth/width >= {dw_max} excluded (E-T depth unreliable).")
        use &= ~kh
    return d[use].reset_index(drop=True), warn


def _setup(names, x, st):
    kw = dict(zip(names, x))
    return replace(st, k_scale=kw.get("k_scale", 1.0)), kw["eta"], kw.get("beam_scale", 1.0)


def _residual_fn(d, base, mat, st, names, weights):
    """Relative-error residuals (pred - meas)/meas for the weighted width/depth/length columns."""
    def resid(x):
        st2, eta, bs = _setup(names, x, st)
        t = predict_table(d, base, mat, st2, eta, bs)
        r = []
        for c in ("width", "depth", "length"):
            wgt = weights.get(c, 0.0)
            if wgt <= 0 or f"{c}_um" not in t:
                continue
            m = t[f"{c}_um"]; p = t[f"pred_{c}_um"]
            ok = m.notna() & (m > 0)
            r.extend((wgt * (p[ok] - m[ok]) / m[ok]).tolist())
        return np.array(r)
    return resid


def calibrate(df: pd.DataFrame, base: Process, mat: Material, st: ModelSettings, *,
              fit_beam_scale: bool = False, fit_k_scale: bool = False,
              weights: dict | None = None, eta0: float = 0.4,
              bounds_eta=(0.05, 1.0), dw_max: Optional[float] = 1.0,
              log: Callable | None = None) -> CalResult:
    """Least-squares fit of an EFFECTIVE absorptivity (optionally beam scale, k scale) to measured pools."""
    weights = weights or {"width": 1.0, "depth": 1.0, "length": 0.0}
    d, warn = _filter_tracks(df, dw_max)
    if len(d) < 2:
        raise ValueError("Need at least 2 usable tracks to calibrate.")
    names = ["eta"]; x0 = [eta0]; lo = [bounds_eta[0]]; hi = [bounds_eta[1]]
    if fit_beam_scale: names.append("beam_scale"); x0.append(1.0); lo.append(0.5); hi.append(2.0)
    if fit_k_scale: names.append("k_scale"); x0.append(1.0); lo.append(0.5); hi.append(2.0)
    n_obs = sum(int(d[f"{c}_um"].notna().sum()) for c in ("width", "depth", "length") if weights.get(c, 0) > 0)
    if n_obs < len(names) + 1:
        raise ValueError("Not enough measurements for the number of fitted parameters.")
    resid = _residual_fn(d, base, mat, st, names, weights)
    sol = least_squares(resid, x0, bounds=(lo, hi), x_scale=[1.0] * len(x0), diff_step=1e-3)
    st2, eta, bs = _setup(names, sol.x, st)
    tab = predict_table(d, base, mat, st2, eta, bs)
    J = sol.jac; n, p = J.shape
    s2 = 2 * sol.cost / max(n - p, 1)
    try:
        cov = np.linalg.inv(J.T @ J) * s2
        se = np.sqrt(np.diag(cov))
        corr = cov / np.outer(se, se)
    except np.linalg.LinAlgError:
        se = np.full(p, np.nan); corr = None
        warn.append("Jacobian singular: parameters not identifiable from this data.")
    if corr is not None and p > 1 and np.nanmax(np.abs(corr - np.eye(p))) > 0.9:
        warn.append("Fitted parameters are strongly correlated (|r|>0.9): they cannot be separated with this data. "
                    "Fit absorptivity only, or add tracks with different beam sizes.")
    for nme, val, s_ in zip(names, sol.x, se):
        if np.isfinite(s_) and abs(val) > 0 and s_ / abs(val) > 0.25:
            warn.append(f"Parameter {nme} is poorly determined (std. error {s_ / abs(val) * 100:.0f}% of its value): "
                        "the data cannot support fitting it. Remove it or add more varied tracks.")
    for nme, val, (a_, b_) in zip(names, sol.x, zip(lo, hi)):
        if val <= a_ * 1.001 or val >= b_ * 0.999:
            warn.append(f"Parameter {nme} sits on its bound ({val:.3f}); the model structure may not fit this data.")
    rmse, mape, r2 = _metrics(tab)
    return CalResult(params=dict(zip(names, map(float, sol.x))), std_err=dict(zip(names, map(float, se))),
                     corr=corr, names=names, rmse=rmse, mape=mape, r2=r2, table=tab,
                     cost=float(sol.cost), n_used=len(d), warnings=warn,
                     opts=dict(weights=weights, lo=lo, hi=hi, dw_max=dw_max))


def run_mcmc(df, base, mat, st, res: CalResult, nwalk: int = 8, nsteps: int = 80, seed: int = 1,
             progress: Callable | None = None) -> dict:
    """OPTIONAL emcee posterior around the least-squares fit (flat priors inside the bounds, Gaussian
    relative-error likelihood with sigma from the fit residuals). Reflects measurement scatter only,
    not model discrepancy, so the intervals are optimistic."""
    import emcee
    o = res.opts
    d, _ = _filter_tracks(df, o["dw_max"])
    names = res.names; nd = len(names)
    resid = _residual_fn(d, base, mat, st, names, o["weights"])
    x0 = np.array([res.params[n] for n in names]); lo = np.array(o["lo"]); hi = np.array(o["hi"])
    r0 = resid(x0)
    sig = max(float(np.sqrt(np.sum(r0 ** 2) / max(len(r0) - nd, 1))), 0.01)

    def lp(x):
        if np.any(x <= lo) or np.any(x >= hi):
            return -np.inf
        r = resid(x)
        return -0.5 * float(np.sum(r * r)) / sig ** 2

    nwalk = max(int(nwalk), 2 * nd + 2); nwalk += nwalk % 2
    se = np.array([res.std_err[n] for n in names])
    step = np.where(np.isfinite(se) & (se > 0), se, 0.01 * np.abs(x0)) * 0.3
    rng = np.random.default_rng(seed)
    p0 = np.clip(x0 + step * rng.standard_normal((nwalk, nd)), lo + 1e-6 * (hi - lo), hi - 1e-6 * (hi - lo))
    sam = emcee.EnsembleSampler(nwalk, nd, lp)
    for i, _ in enumerate(sam.sample(p0, iterations=int(nsteps))):
        if progress:
            progress(i, int(nsteps))
    ch = sam.get_chain(discard=int(nsteps) // 3, flat=True)
    try:
        tau = float(np.max(sam.get_autocorr_time(tol=0)))
    except Exception:
        tau = float("nan")
    q = np.percentile(ch, [16, 50, 84], axis=0)
    tab = pd.DataFrame({"parameter": names, "median": q[1], "p16": q[0], "p84": q[2]})
    return dict(table=tab, chain=ch, sigma_rel=sig, accept=float(np.mean(sam.acceptance_fraction)),
                tau=tau, nsteps=int(nsteps), nwalk=nwalk)


def cross_validate(df, base, mat, st, mode: str = "loo", frac_test: float = 0.3, seed: int = 0,
                   progress: Callable | None = None, **kw):
    """Leave-one-out or random hold-out. Returns (table of test predictions, metrics)."""
    d = df.reset_index(drop=True)
    idx = np.arange(len(d))
    rows = []
    if mode == "loo":
        folds = [([j for j in idx if j != i], [i]) for i in idx]
    else:
        rng = np.random.default_rng(seed); perm = rng.permutation(idx)
        nt = max(1, int(round(frac_test * len(d))))
        folds = [(sorted(perm[nt:]), sorted(perm[:nt]))]
    for i_f, (tr, te) in enumerate(folds):
        if progress: progress(i_f, len(folds))
        res = calibrate(d.iloc[tr], base, mat, st, **kw)
        pr = res.params
        st2 = replace(st, k_scale=pr.get("k_scale", 1.0))
        rows.append(predict_table(d.iloc[te], base, mat, st2, pr["eta"], pr.get("beam_scale", 1.0)))
    tab = pd.concat(rows, ignore_index=True)
    return tab, _metrics(tab)


def library_crosscheck(proc: Process, mat: Material, st: ModelSettings | None = None,
                       domain=(480, 75, 65, 1.0)):
    """Compare with the optional eagar-tsai library on a SMALL domain (um; length x half-width x depth, grid).
    The library fixes T0 = 298 K and takes beam_diameter = 2*sigma. Raises ImportError if not installed.
    Returns (table, in_domain)."""
    from eagar_tsai import compute_single_point, BeamParameters, MaterialProperties, SimulationDomain
    st = st or ModelSettings()
    p = replace(proc, T0=298.0)
    mine = run_case(p, mat, st)
    r = compute_single_point(BeamParameters(beam_diameter=2 * p.sigma, power=p.P, velocity=p.v, absorptivity=p.eta),
                             MaterialProperties(mat.T_liq, mat.k * st.k_scale, mat.rho, mat.cp),
                             SimulationDomain(*domain), full_field=False)
    rows = []
    for n, a_, b_ in (("width", mine["width"] * 1e6, r.width_um), ("depth", mine["depth"] * 1e6, r.depth_um),
                      ("length", mine["length"] * 1e6, r.length_um)):
        tol = max(2.5, 0.03 * b_)
        rows.append(dict(quantity=f"{n} [um]", this_app=a_, library=b_, abs_diff=abs(a_ - b_), tolerance=tol, ok=bool(abs(a_ - b_) <= tol)))
    inside = bool(mine["melted"] and mine["width"] * 5e5 <= 0.95 * domain[1] and mine["depth"] * 1e6 <= 0.95 * domain[2]
                  and mine["length"] * 1e6 <= 0.95 * domain[0])
    return pd.DataFrame(rows), inside


# ----------------------------------------------------------------------
# 7. Sweep, sensitivity, Goldak export
# ----------------------------------------------------------------------
def sweep(base: Process, mat: Material, st: ModelSettings, P_vals, v_vals) -> pd.DataFrame:
    rows = []
    for P in P_vals:
        for v in v_vals:
            r = run_case(replace(base, P=float(P), v=float(v)), mat, st)
            rows.append(dict(power_W=P, speed_m_s=v, width_um=r["width"] * 1e6, depth_um=r["depth"] * 1e6,
                             length_um=r["length"] * 1e6, dw_ratio=r["dw_ratio"], peak_T_K=r["peak_T"],
                             linear_energy_J_mm=r["linear_energy_J_per_m"] / 1e3, regime=r["regime"]))
    return pd.DataFrame(rows)


def sensitivity(base: Process, mat: Material, st: ModelSettings, frac: float = 0.10) -> pd.DataFrame:
    """% change of width/depth for +frac and -frac change of each input."""
    ref = run_case(base, mat, st)
    items = {
        "power": lambda s: (replace(base, P=base.P * s), mat, st),
        "speed": lambda s: (replace(base, v=base.v * s), mat, st),
        "beam size": lambda s: (replace(base, beam=base.beam * s), mat, st),
        "absorptivity": lambda s: (replace(base, eta=min(1.0, base.eta * s)), mat, st),
        "conductivity": lambda s: (base, replace(mat, k=mat.k * s), st),
        "density": lambda s: (base, replace(mat, rho=mat.rho * s), st),
        "specific heat": lambda s: (base, replace(mat, cp=mat.cp * s), st),
        "T_liquidus rise": lambda s: (base, replace(mat, T_liq=base.T0 + (mat.T_liq - base.T0) * s), st),
    }
    rows = []
    for name, f in items.items():
        up, dn = run_case(*f(1 + frac)), run_case(*f(1 - frac))
        def pc(r, key): return (r[key] - ref[key]) / ref[key] * 100 if ref[key] > 0 else float("nan")
        rows.append(dict(parameter=name, width_up_pct=pc(up, "width"), width_dn_pct=pc(dn, "width"),
                         depth_up_pct=pc(up, "depth"), depth_dn_pct=pc(dn, "depth")))
    return pd.DataFrame(rows)


def goldak_start(res: dict, convention: str = "a=half-width, b=depth, c=length") -> dict:
    cf, cr = res["head"], res["tail"]
    return {"a_half_width_m": res["width"] / 2, "b_depth_m": res["depth"], "c_front_m": cf, "c_rear_m": cr,
            "f_front": 2 * cf / (cf + cr) if cf + cr > 0 else float("nan"),
            "f_rear": 2 * cr / (cf + cr) if cf + cr > 0 else float("nan"),
            "note": "Heuristic starting guess from E-T geometry. Calibrate in Abaqus; check each paper's a/b/c naming."}


# ----------------------------------------------------------------------
# Temperature-dependent property tables -> effective constants
# ----------------------------------------------------------------------
PROP_LABEL = {"k": "conductivity k [W/m/K]", "rho": "density rho [kg/m3]", "cp": "specific heat cp [J/kg/K]",
              "H": "enthalpy H [J/kg]", "fl": "liquid fraction fl [-]"}


TP_COLS = ["T", "k", "rho", "cp", "H", "fl"]


def map_prop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map a table with arbitrary headers (e.g. a JMatPro export) onto the canonical columns T, k, rho, cp, H, fl."""
    low = {c: str(c).lower().strip() for c in df.columns}

    def find(sub=(), exact=()):
        for c, l in low.items():
            if l in exact or any(k in l for k in sub):
                return c
    cT = find(("temp",), ("t", "t_k", "t [k]", "t (k)", "t/k", "t [c]", "t (c)"))
    if cT is None:
        raise ValueError("no temperature column found (name it 'T' or 'Temperature')")
    cols = {"k": find(("conduct",), ("k",)), "rho": find(("dens",), ("rho",)), "cp": find(("specific", "heat cap"), ("cp",)),
            "H": find(("enthalpy",), ("h",)), "fl": find(("liquid", "fraction"), ("fl",))}
    if all(c is None for c in cols.values()):
        raise ValueError("no property columns recognised (use names like conductivity, density, specific heat, enthalpy, liquid fraction)")
    out = pd.DataFrame({"T": pd.to_numeric(df[cT], errors="coerce")})
    for p, c in cols.items():
        out[p] = pd.to_numeric(df[c], errors="coerce") if c is not None else np.nan
    return out[TP_COLS].astype(float)


def tables_from_df(df: pd.DataFrame, T_unit: str = "K", mults: dict | None = None) -> tuple[dict, list]:
    """Turn the editable property table into {property: DataFrame(T_K, value)} in SI units; rows with a blank cell are ignored."""
    mults, errs, out = mults or {}, [], {}
    T = pd.to_numeric(df["T"], errors="coerce") + (273.15 if T_unit == "degC" else 0.0)
    for p in ("k", "rho", "cp", "H", "fl"):
        if p not in df:
            continue
        d = pd.DataFrame({"T_K": T, "value": pd.to_numeric(df[p], errors="coerce") * float(mults.get(p, 1.0))}).dropna()
        if d.empty:
            continue
        if len(d) < 2:
            errs.append(f"{PROP_LABEL[p]}: enter at least 2 rows (temperature and value)"); continue
        d = d.sort_values("T_K").reset_index(drop=True)
        if d["T_K"].duplicated().any():
            errs.append(f"{PROP_LABEL[p]}: two rows have the same temperature"); continue
        if (d["T_K"] <= 0).any():
            errs.append(f"{PROP_LABEL[p]}: temperatures must be above 0 K (check the temperature unit)"); continue
        out[p] = d
    if "fl" in out and out["fl"]["value"].max() > 1.5:            # percent -> fraction
        out["fl"]["value"] = out["fl"]["value"] / 100.0
    return out, errs


def effective_props(tabs: dict, T0: float, T_liq: float, mean: bool = True, T_ref: float | None = None,
                    latent: float = 0.0) -> tuple[dict, list]:
    """Effective CONSTANT k, rho, cp for the E-T model from T-dependent tables.
    mean=True: trapezoid average over [T0, T_liq]; else value at T_ref. cp from an enthalpy table is
    (H(T_liq)-H(T0))/(T_liq-T0), which includes latent heat only if H does. With a cp table, latent>0 adds L/(T_liq-T0)."""
    if T_liq <= T0:
        raise ValueError("T_liquidus must exceed T0")
    if not mean and not T_ref:
        raise ValueError("reference temperature required")
    grid = np.linspace(T0, T_liq, 400)
    f = lambda d, T: np.interp(T, d["T_K"].values, d["value"].values)

    def avg(d):
        if not mean:
            return float(f(d, T_ref))
        y = f(d, grid)
        return float(np.sum((y[1:] + y[:-1]) * np.diff(grid)) / 2.0 / (T_liq - T0))
    out, notes = {}, []
    for p in ("k", "rho", "cp", "H"):
        if p in tabs and (tabs[p]["T_K"].min() > T0 or tabs[p]["T_K"].max() < T_liq):
            notes.append(f"{p}: table covers {tabs[p]['T_K'].min():.0f}-{tabs[p]['T_K'].max():.0f} K only; end values are held constant outside it.")
    for p in ("k", "rho"):
        if p in tabs:
            out[p] = avg(tabs[p])
    if "H" in tabs:
        out["cp"] = float((f(tabs["H"], T_liq) - f(tabs["H"], T0)) / (T_liq - T0))
        notes.append("cp = (H(T_liq) - H(T0)) / (T_liq - T0): includes latent heat only if the H table does. A separate cp table and the latent-heat field are ignored.")
    elif "cp" in tabs:
        out["cp"] = avg(tabs["cp"])
        if "rho" in tabs and mean:        # energy per volume is the mean of rho(T)*cp(T), not (mean rho)*(mean cp)
            prod = f(tabs["rho"], grid) * f(tabs["cp"], grid)
            out["cp"] = float(np.sum((prod[1:] + prod[:-1]) * np.diff(grid)) / 2.0 / (T_liq - T0)) / out["rho"]
        if latent > 0:
            out["cp"] += latent / (T_liq - T0)
            notes.append(f"cp includes L/(T_liq-T0) = {latent / (T_liq - T0):.1f} J/kg/K from the latent-heat field. Do not use this if your cp table already contains the melting peak (apparent cp).")
    return out, notes


def property_warnings(tabs: dict, T_unit: str, T0: float, T_liq: float, latent: float, inclat: bool) -> list:
    """Plain-language warnings about the temperature table (unit mix-ups, missing latent heat)."""
    w = []
    if tabs and T_unit == "K" and min(d["T_K"].min() for d in tabs.values()) < 0.5 * T0:
        w.append(f"Your table starts at {min(d['T_K'].min() for d in tabs.values()):.0f} K. If these temperatures are in degC, set 'Temperature unit' to degC "
                 "(otherwise the averaging window covers the wrong part of the table).")
    if "cp" in tabs and "H" not in tabs and not inclat:
        if latent > 0:
            w.append(f"Latent heat ({latent:.3g} J/kg) is entered but NOT added to cp, so the pool will be too large. Tick the latent-heat box below "
                     "(unless your cp column already contains the melting peak).")
        else:
            w.append("No latent heat is included (latent-heat field = 0 and no enthalpy column): the model will overpredict the melt pool, typically by roughly 10 %. "
                     "Enter the latent heat of fusion above and tick the box below, or give an enthalpy column.")
    return w


def solve_eta_for_width(proc: Process, mat: Material, stg: ModelSettings, width_um: float, bounds=(0.01, 1.0)) -> dict | None:
    """Absorptivity for which the E-T melt pool width equals width_um (width grows monotonically with eta). None if unreachable."""
    from scipy.optimize import brentq

    def f(eta):
        r = run_case(replace(proc, eta=eta), mat, stg)
        return (r["width"] * 1e6 if r["melted"] else 0.0) - width_um
    lo, hi = bounds
    if width_um < 1.0 or f(lo) > 0 or f(hi) < 0:
        return None
    eta = brentq(f, lo, hi, xtol=1e-5)
    r = run_case(replace(proc, eta=eta), mat, stg)
    if not r["melted"] or abs(r["width"] * 1e6 - width_um) > 0.01 * width_um:       # converged onto the melting threshold, not a real pool
        return None
    return dict(eta=float(eta), width_um=r["width"] * 1e6, depth_um=r["depth"] * 1e6, dw_ratio=r["dw_ratio"], regime=r["regime"])


def solidus_liquidus_from_fl(d: pd.DataFrame, tol: float = 1e-3) -> tuple[float, float]:
    """Solidus = highest T with fl <= tol, liquidus = lowest T with fl >= 1-tol."""
    T, y = d["T_K"].values, d["value"].values
    lo, hi = T[y <= tol], T[y >= 1 - tol]
    if lo.size == 0 or hi.size == 0:
        raise ValueError("the liquid-fraction table must span 0 to 1")
    Ts, Tl = float(lo.max()), float(hi.min())
    if Tl <= Ts:
        raise ValueError("liquid fraction table is not monotonic")
    return Ts, Tl


GOLDAK_CONVENTIONS = {
    "Goldak 1984: a=half-width (y), b=depth (z), c_f/c_r=front/rear length (x)": "goldak",
    "Common AM-FEM: a_f/a_r=front/rear length (x), b=half-width (y), c=depth (z)": "am",
}


GOLDAK_INFO = {
    "a (half-width)": "Semi-axis of the heat-source ellipsoid ACROSS the track (y direction) = half of the melt pool width.",
    "b (depth)": "Semi-axis INTO the part (z direction) = melt pool depth.",
    "c_f (front length)": "Semi-axis of the FRONT half, ahead of the laser = melt pool head length.",
    "c_r (rear length)": "Semi-axis of the REAR half, behind the laser = melt pool tail length (longer than the front).",
    "a_f (front length)": "Semi-axis of the FRONT half, ahead of the laser = melt pool head length.",
    "a_r (rear length)": "Semi-axis of the REAR half, behind the laser = melt pool tail length (longer than the front).",
    "b (half-width)": "Semi-axis ACROSS the track (y direction) = half of the melt pool width.",
    "c (depth)": "Semi-axis INTO the part (z direction) = melt pool depth.",
    "f_f": "Fraction of the heat deposited in the front half: f_f = 2 c_f / (c_f + c_r).",
    "f_r": "Fraction of the heat deposited in the rear half: f_r = 2 c_r / (c_f + c_r). Always f_f + f_r = 2.",
    "Q = eta*P (effective absorbed power)": "Total absorbed heat power to give the FE heat source (e.g. Abaqus DFLUX). eta is the effective (calibrated) absorptivity.",
}


def goldak_table(gs: dict, eta: float, P: float, convention: str = "goldak") -> pd.DataFrame:
    """Goldak starting values in the chosen naming convention, with units and a plain-language meaning."""
    if convention == "goldak":
        rows = [("a (half-width)", gs["a_half_width_m"]), ("b (depth)", gs["b_depth_m"]),
                ("c_f (front length)", gs["c_front_m"]), ("c_r (rear length)", gs["c_rear_m"])]
    else:
        rows = [("a_f (front length)", gs["c_front_m"]), ("a_r (rear length)", gs["c_rear_m"]),
                ("b (half-width)", gs["a_half_width_m"]), ("c (depth)", gs["b_depth_m"])]
    out = [(n, v * 1e6, "um", v, GOLDAK_INFO[n]) for n, v in rows]
    out += [("f_f", gs["f_front"], "-", gs["f_front"], GOLDAK_INFO["f_f"]), ("f_r", gs["f_rear"], "-", gs["f_rear"], GOLDAK_INFO["f_r"]),
            ("Q = eta*P (effective absorbed power)", eta * P, "W", eta * P, GOLDAK_INFO["Q = eta*P (effective absorbed power)"])]
    return pd.DataFrame(out, columns=["parameter", "value", "unit", "value in SI (m, W)", "what it is"])


def inputs_table(proc: Process, mat: Material, st: ModelSettings) -> pd.DataFrame:
    s_ = mat.sources or {}
    rows = [("laser power P", "W", proc.P, "process"), ("scan speed v", "m/s", proc.v, "process"),
            (f"beam size ({proc.beam_def})", "m", proc.beam, "process"), ("Gaussian sigma used", "m", proc.sigma, "derived"),
            ("absorptivity eta (effective)", "-", proc.eta, "process / calibration"), ("T0", "K", proc.T0, "process"),
            ("beam-size scale factor", "-", proc.beam_scale, "calibration"), ("conductivity scale factor", "-", st.k_scale, "calibration"),
            ("k", "W/m/K", mat.k, s_.get("k", "")), ("rho", "kg/m3", mat.rho, s_.get("rho", "")),
            ("cp", "J/kg/K", mat.cp, s_.get("cp", "")), ("T_liquidus", "K", mat.T_liq, s_.get("T_liq", "")),
            ("T_solidus", "K", mat.T_sol if mat.T_sol else float("nan"), s_.get("T_sol", "")),
            ("T_boil", "K", mat.T_boil if mat.T_boil else float("nan"), s_.get("T_boil", "")),
            ("latent heat", "J/kg", mat.latent_heat, s_.get("latent_heat", "")), ("material", "", mat.name, mat.source)]
    return pd.DataFrame(rows, columns=["quantity", "unit", "value", "source"])


# ----------------------------------------------------------------------
# 8. Fields and plots (matplotlib, simple)
# ----------------------------------------------------------------------
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def field_planes(proc: Process, mat: Material, st: ModelSettings, res: dict, n: int = 90):
    """T on x-y (z=0), x-z (y=0) and y-z (at x of max width) planes, in micrometres / K."""
    c = ETCase(proc, mat, st)
    L = max(res["length"], 4 * c.sig); W = max(res["width"], 4 * c.sig); D = max(res["depth"], 2 * c.sig)
    xs = np.linspace(-(res["tail"] + 0.35 * L), res["head"] + 0.35 * L, n * 2)
    ys = np.linspace(-0.9 * W, 0.9 * W, n)
    zs = np.linspace(0, 1.6 * D, n)
    pre = c.prep(float(np.max(np.abs(xs))))
    X, Y = np.meshgrid(xs, ys)
    Txy = c.dT(X, Y, 0.0, prep=pre) + proc.T0
    X2, Z2 = np.meshgrid(xs, zs)
    Txz = c.dT(X2, 0.0, Z2, prep=pre) + proc.T0
    xw = res.get("x_width", 0.0)
    Y3, Z3 = np.meshgrid(ys, zs)
    Tyz = c.dT(xw, Y3, Z3) + proc.T0
    return dict(xs=xs * 1e6, ys=ys * 1e6, zs=zs * 1e6, Txy=Txy, Txz=Txz, Tyz=Tyz)


def fig_sections(proc, mat, st, res, exp_width_um=None, exp_depth_um=None):
    plt = _mpl()
    f = field_planes(proc, mat, st, res)
    Tl = mat.T_liq
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    # cross-section
    a = ax[0]
    a.contour(f["ys"], f["zs"], f["Tyz"], levels=[Tl], colors="k")
    a.contourf(f["ys"], f["zs"], f["Tyz"], levels=[Tl, 1e9], colors=["#f4a582"], alpha=0.6)
    a.invert_yaxis(); a.set_aspect("equal"); a.set_xlabel("y [um]"); a.set_ylabel("depth z [um]")
    a.set_title(f"Cross-section  W={res['width']*1e6:.0f} um, D={res['depth']*1e6:.0f} um")
    if exp_width_um: a.plot([-exp_width_um / 2, exp_width_um / 2], [0, 0], "b-", lw=4, label="exp. width")
    if exp_depth_um: a.plot([0, 0], [0, exp_depth_um], "g-", lw=4, label="exp. depth")
    if exp_width_um or exp_depth_um: a.legend(loc="lower right")
    # top view
    a = ax[1]
    a.contourf(f["xs"], f["ys"], f["Txy"], levels=[Tl, 1e9], colors=["#f4a582"], alpha=0.6)
    a.contour(f["xs"], f["ys"], f["Txy"], levels=[Tl], colors="k")
    a.set_aspect("equal"); a.set_xlabel("x [um]  (beam moves to +x)"); a.set_ylabel("y [um]")
    a.set_title(f"Top view  L={res['length']*1e6:.0f} um")
    # longitudinal
    a = ax[2]
    a.contourf(f["xs"], f["zs"], f["Txz"], levels=[Tl, 1e9], colors=["#f4a582"], alpha=0.6)
    a.contour(f["xs"], f["zs"], f["Txz"], levels=[Tl], colors="k")
    a.invert_yaxis(); a.set_aspect("equal"); a.set_xlabel("x [um]"); a.set_ylabel("depth z [um]")
    a.set_title("Longitudinal section")
    fig.tight_layout()
    return fig


def fig_heatmap(proc, mat, st, res, vmax_factor: float = 2.0):
    plt = _mpl()
    f = field_planes(proc, mat, st, res)
    fig, a = plt.subplots(figsize=(8, 4))
    vmax = mat.T_liq * vmax_factor
    im = a.contourf(f["xs"], f["ys"], np.minimum(f["Txy"], vmax), levels=30, cmap="inferno")
    a.contour(f["xs"], f["ys"], f["Txy"], levels=[mat.T_liq], colors="c", linewidths=1.5)
    fig.colorbar(im, ax=a, label=f"T [K] (clipped at {vmax:.0f})")
    a.set_aspect("equal"); a.set_xlabel("x [um]"); a.set_ylabel("y [um]"); a.set_title("Surface temperature, liquidus in cyan")
    fig.tight_layout(); return fig


def fig_map(sw: pd.DataFrame, value: str = "depth_um", kh_dw: float = 1.0):
    plt = _mpl()
    Ps = np.sort(sw["power_W"].unique()); Vs = np.sort(sw["speed_m_s"].unique())
    Z = sw.pivot(index="power_W", columns="speed_m_s", values=value).reindex(index=Ps, columns=Vs).values
    fig, a = plt.subplots(figsize=(7, 5))
    im = a.pcolormesh(Vs, Ps, Z, shading="nearest", cmap="viridis")
    fig.colorbar(im, ax=a, label=value)
    DW = sw.pivot(index="power_W", columns="speed_m_s", values="dw_ratio").reindex(index=Ps, columns=Vs).values
    if len(Ps) > 1 and len(Vs) > 1 and np.nanmax(DW) >= kh_dw >= np.nanmin(DW):
        a.contour(Vs, Ps, DW, levels=[kh_dw], colors="r", linewidths=2)
        a.plot([], [], "r-", label=f"depth/width = {kh_dw}")
        a.legend(loc="upper left")
    a.set_xlabel("scan speed [m/s]"); a.set_ylabel("power [W]"); a.set_title(f"{value} (E-T)")
    fig.tight_layout(); return fig


def fig_parity(tab: pd.DataFrame, title: str = "Calibration"):
    plt = _mpl()
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.4))
    for a, c in zip(ax, ("width", "depth")):
        m, p = f"{c}_um", f"pred_{c}_um"
        if m not in tab or tab[m].dropna().empty:
            a.set_visible(False); continue
        s = tab[[m, p]].dropna(); lim = [0, max(s.max()) * 1.1]
        a.plot(lim, lim, "k-", lw=1); a.plot(lim, [l * 1.1 for l in lim], "k:", lw=0.8); a.plot(lim, [l * 0.9 for l in lim], "k:", lw=0.8)
        a.scatter(s[m], s[p], c="tab:blue"); a.set_xlim(lim); a.set_ylim(lim)
        a.set_xlabel(f"measured {c} [um]"); a.set_ylabel(f"E-T {c} [um]"); a.set_title(f"{title}: {c} (dotted = +/-10%)")
    fig.tight_layout(); return fig


def fig_residuals(tab: pd.DataFrame):
    plt = _mpl()
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a, xcol, lab in zip(ax, ("power_W", "speed_m_s"), ("power [W]", "speed [m/s]")):
        for c, mk in (("width", "o"), ("depth", "s")):
            m, p = f"{c}_um", f"pred_{c}_um"
            if m in tab and tab[m].notna().any():
                a.scatter(tab[xcol], (tab[p] / tab[m] - 1) * 100, marker=mk, label=c)
        a.axhline(0, color="k", lw=0.8); a.set_xlabel(lab); a.set_ylabel("(pred/meas - 1) [%]"); a.legend()
    fig.tight_layout(); return fig


def fig_sensitivity(sens: pd.DataFrame):
    plt = _mpl()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    y = np.arange(len(sens))
    for a, c in zip(ax, ("width", "depth")):
        a.barh(y + 0.2, sens[f"{c}_up_pct"], 0.4, label="+10% input")
        a.barh(y - 0.2, sens[f"{c}_dn_pct"], 0.4, label="-10% input")
        a.set_yticks(y); a.set_yticklabels(sens["parameter"]); a.axvline(0, color="k", lw=0.8)
        a.set_xlabel(f"change in {c} [%]"); a.legend()
    fig.tight_layout(); return fig


def fig_props(tabs: dict, eff: dict, T0: float, T_liq: float):
    plt = _mpl()
    names = [p for p in ("k", "rho", "cp", "H", "fl") if p in tabs]
    ncol = min(3, len(names)); nrow = math.ceil(len(names) / ncol)
    fig, ax = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.3 * nrow), squeeze=False)
    for a in ax.ravel():
        a.set_visible(False)
    for a, p in zip(ax.ravel(), names):
        a.set_visible(True); d = tabs[p]
        a.plot(d["T_K"], d["value"], "o-", ms=3)
        a.axvspan(T0, T_liq, color="#f4a582", alpha=0.25, label="T0 ... T_liq")
        if p in eff:
            a.axhline(eff[p], color="k", ls="--", label=f"effective = {eff[p]:.4g}")
        a.set_xlabel("T [K]"); a.set_ylabel(PROP_LABEL[p]); a.legend(fontsize=7)
    fig.tight_layout(); return fig


def fig_posterior(chain: np.ndarray, names: list):
    plt = _mpl()
    fig, ax = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3.4), squeeze=False)
    for a, i, n in zip(ax[0], range(len(names)), names):
        a.hist(chain[:, i], bins=30, color="tab:blue", alpha=0.8)
        for q_ in np.percentile(chain[:, i], [16, 50, 84]): a.axvline(q_, color="k", lw=0.8, ls="--")
        a.set_xlabel(n); a.set_ylabel("samples")
    fig.suptitle("emcee posterior (median, 16/84 %)"); fig.tight_layout(); return fig


RESULT_ROWS = [
    ("Melt pool width", "um", lambda r: r["width"] * 1e6, "0.0", "Full width of the molten region on the top surface (liquidus isotherm). Compare with the width you measure on a top view or cross-section."),
    ("Melt pool depth", "um", lambda r: r["depth"] * 1e6, "0.0", "Maximum depth of the molten region below the surface. Compare with the cross-section depth. Underpredicted for deep, keyhole-like pools."),
    ("Melt pool length", "um", lambda r: r["length"] * 1e6, "0.0", "Length of the molten region along the scan direction on the surface (head + tail)."),
    ("  head length (ahead of beam centre)", "um", lambda r: r["head"] * 1e6, "0.0", "Distance from the beam centre to the front edge of the pool."),
    ("  tail length (behind beam centre)", "um", lambda r: r["tail"] * 1e6, "0.0", "Distance from the beam centre to the rear edge of the pool; the metal solidifies at this rear edge."),
    ("Length / width", "-", lambda r: r["lw_ratio"], "0.00", "Elongation of the pool. Very long, thin pools tend to become unstable (balling). Informational."),
    ("Depth / width", "-", lambda r: r["dw_ratio"], "0.00", "Shape indicator: small = shallow conduction-mode pool; near or above ~1 = deep, keyhole-like pool. Thresholds are editable heuristics (Model settings tab)."),
    ("Peak surface temperature", "K", lambda r: r["peak_T"], "0", "Highest temperature under the beam in the E-T solution. Unrealistically high at high power (no evaporation or convection in the model); use it only to see whether the case is far outside the model's validity."),
    ("Thermal gradient G at trailing liquidus point", "K/m", lambda r: r["G"], "0.00E+00", "How steeply temperature falls along the scan direction at the rear edge of the pool, where the metal solidifies. Together with the solidification rate it sets grain morphology (G/R) and scale (G x R). E-T estimate, not an FEM result."),
    ("Cooling rate at trailing liquidus point", "K/s", lambda r: r["cooling_rate"], "0.00E+00", "G x scan speed: how fast the metal cools while solidifying. Faster cooling gives finer cells/dendrites. E-T estimate."),
    ("Linear energy density P/v", "J/mm", lambda r: r["linear_energy_J_per_m"] / 1e3, "0.00", "Laser energy delivered per millimetre of track (power / speed). Simple way to compare parameter sets."),
    ("Volumetric energy density P/(v h t)", "J/mm3", lambda r: r["VED_J_per_m3"] / 1e9, "0.0", "Energy per unit volume of powder bed: power / (speed x hatch x layer). Uses hatch and layer from the Process tab."),
    ("Normalized enthalpy", "-", lambda r: r["norm_enthalpy"], "0.0", "Absorbed power relative to the enthalpy needed to bring the metal to melting (King et al. 2014 type). Larger = more keyhole-prone. Informational unless the enthalpy flag is enabled."),
    ("Regime", "", lambda r: r["regime"], None, "Automatic flag from the editable thresholds: conduction / transition / keyhole-suspect, plus warnings for boiling and extreme temperatures. E-T is only valid in conduction mode."),
]


def results_table(res: dict, numeric: bool = False) -> pd.DataFrame:
    """Results with units and a plain-language meaning. numeric=False gives display strings (for st.table)."""
    rows = []
    for name, unit, fn, fmt, meaning in RESULT_ROWS:
        v = fn(res)
        if not numeric and isinstance(v, (int, float, np.floating)):
            v = f"{v:.4g}"
        rows.append((name, v, unit, meaning))
    return pd.DataFrame(rows, columns=["quantity", "value", "unit", "what it means"])


REFERENCES = [
    ("Physics of the model", "T. W. Eagar and N.-S. Tsai, Temperature fields produced by traveling distributed heat sources, Welding Journal 62(12), 346s-355s (1983).",
     "The moving Gaussian-source solution on a semi-infinite solid that this app evaluates (all of tabs 5-8).", "Checked online this session (cited in the eagar-tsai library documentation)."),
    ("Physics of the model", "D. Rosenthal, The theory of moving sources of heat and its application to metal treatments, Trans. ASME 68, 849-866 (1946).",
     "Point-source limit used as a self-test (the E-T result must approach Rosenthal for a tiny beam).", "From memory; verify volume/pages before citing."),
    ("Physics of the model", "H. S. Carslaw and J. C. Jaeger, Conduction of Heat in Solids, 2nd ed., Oxford University Press (1959).",
     "Background for the semi-infinite-solid conduction solutions.", "From memory; verify edition details before citing."),
    ("Software cross-check", "eagar-tsai Python library, ArroyaveLab (github.com/ArroyaveLab/eagar-tsai; pip install eagar-tsai; GPL-3.0).",
     "Optional independent cross-check (Model settings tab, and the self-tests). It fixes T0 = 298 K and takes beam_diameter = 2 sigma.", "Checked online this session (PyPI / GitHub)."),
    ("Heat source for FE models", "J. Goldak, A. Chakravarti and M. Bibby, A new finite element model for welding heat sources, Metall. Trans. B 15(2), 299-305 (1984).",
     "Double-ellipsoid heat source whose starting dimensions are derived in the Goldak tab (f_f = 2 c_f/(c_f + c_r)).", "From memory; verify volume/pages before citing."),
    ("Heat source for FE models", "Dassault Systemes, Abaqus Analysis User's Guide: material data blocks *CONDUCTIVITY, *DENSITY, *SPECIFIC HEAT, *LATENT HEAT (data lines 'value, temperature') and the DFLUX user subroutine.",
     "Layout of the temperature-dependent property table (Material tab) and the intended use of the Goldak values.", "From memory of the Abaqus documentation; check the manual of your Abaqus version."),
    ("Melting regimes and thresholds", "W. E. King et al., Observation of keyhole-mode laser melting in laser powder-bed fusion additive manufacturing, J. Mater. Process. Technol. 214(12), 2915-2925 (2014).",
     "Normalized enthalpy and the keyhole-mode idea behind the 'Normalized enthalpy' result and flag.", "Existence/title/journal checked online this session; page numbers from memory."),
    ("Melting regimes and thresholds", "Depth/width thresholds (0.5 transition, 1.0 keyhole), peak-T warning at 20 x T_liquidus, enthalpy threshold 6.",
     "Defaults of the editable flags in Model settings.", "NOT taken from one specific paper: rough rules of thumb chosen when writing this app. Set your own from your alloy's literature."),
    ("Beam and absorptivity", "ISO 11146-1:2005, Lasers and laser-related equipment - Test methods for laser beam widths, divergence angles and beam propagation ratios, Part 1.",
     "Definition of the D4sigma beam diameter (sigma = D/4 for Gaussian beams).", "From memory; verify the standard's current edition."),
    ("Beam and absorptivity", "J. Trapp et al., In situ absorptivity measurements of metallic powders during laser powder-bed fusion additive manufacturing, Appl. Mater. Today 9, 341-349 (2017).",
     "Context for why the 'absorptivity' used here is an effective fitted value, not a material constant.", "From memory; verify details before citing."),
    ("Solidification and microstructure", "W. Kurz and D. J. Fisher, Fundamentals of Solidification, 4th ed., Trans Tech Publications (1998).",
     "Meaning of G (gradient), G x R (cooling rate) and G/R (morphology) used in the Results descriptions.", "From memory; verify edition details before citing."),
    ("Solidification and microstructure", "T. DebRoy et al., Additive manufacturing of metallic components - Process, structure and properties, Prog. Mater. Sci. 92, 112-224 (2018).",
     "General background on melt pool physics, keyhole formation and the limits of conduction models.", "From memory; verify details before citing."),
    ("Numerics and software", "P. Virtanen et al. (SciPy 1.0 contributors), SciPy 1.0: fundamental algorithms for scientific computing in Python, Nature Methods 17, 261-272 (2020).",
     "Gauss-Legendre nodes, scipy.quad (self-tests) and scipy.optimize.least_squares (calibration).", "From memory; verify before citing."),
    ("Numerics and software", "M. A. Branch, T. F. Coleman and Y. Li, A subspace, interior, and conjugate gradient method for large-scale bound-constrained minimization problems, SIAM J. Sci. Comput. 21(1), 1-23 (1999).",
     "Trust-region-reflective algorithm behind least_squares with bounds (calibration).", "From memory; verify before citing."),
    ("Numerics and software", "D. Foreman-Mackey, D. W. Hogg, D. Lang and J. Goodman, emcee: The MCMC Hammer, PASP 125, 306-312 (2013).",
     "Optional posterior sampling in the Calibration tab.", "From memory; verify before citing."),
    ("Advanced modules (named in the specification)", "Hu et al. 2024: profile-distance metric D for melt pool calibration, with E-T as the fast forward model (citation as named in the specification; NOT looked up).",
     "Profile calibration tab (D = sqrt((2/pi) INT |r_exp - r_sim|^2 dphi) / experimental depth).", "Named by the requester; full reference not verified."),
    ("Advanced modules (named in the specification)", "Hybrid depth relation R = A P / (2 a k (T_b - T0) g(Pe)) with a cubic g(Pe) (as given in the specification; original source and coefficients not available to me).",
     "Hybrid depth tab; the coefficients are fitted from your own data.", "Formula taken from the specification; source not verified."),
    ("Advanced modules (named in the specification)", "Goldak axes as 5%-density radii and the published example 102.7 um vs about 63 um; lack-of-fusion (H/w)^2 + (L/d)^2 <= 1 and keyhole limit w/d > 1.5; "
     "benchmark of 13% versus 1.5% error on the scan-direction axis with a longitudinal section (all as given in the specification).",
     "Warnings in tabs 8, 10, 11 and the process-window overlay.", "Numbers taken from the specification; sources not verified."),
    ("Numerics and software", "R. Storn and K. Price, Differential evolution - a simple and efficient heuristic for global optimization over continuous spaces, J. Global Optim. 11, 341-359 (1997).",
     "Optimiser behind scipy.optimize.differential_evolution (Goldak inversion).", "From memory; verify before citing."),
    ("Material data", "JMatPro (Sente Software), Abaqus data sheets, handbooks or your own DSC/CALPHAD results.",
     "NOT built in. You supply every material value and its source in the Material tab.", "By design: the app contains no real-alloy values."),
]


# ======================================================================
# Advanced modules: hybrid depth, profile calibration, FE correction / Goldak surrogate, process map
# ======================================================================
def _trapz(y, x) -> float:
    y = np.asarray(y, float); x = np.asarray(x, float)
    return float(np.sum((y[1:] + y[:-1]) * np.diff(x)) / 2.0)


def _poly3(c, x):
    return c[0] + c[1] * x + c[2] * x ** 2 + c[3] * x ** 3


# ---------------------------- 4. hybrid depth -------------------------------------------------
HYBRID_THRESH = (0.8, 1.2)          # R = d/w: conduction < 0.8, transition 0.8-1.2, keyhole > 1.2 (editable in the UI)


def _row_process(base: Process, r) -> Process:
    p = replace(base, P=float(r["power_W"]), v=float(r["speed_m_s"]))
    if "beam_um" in r and pd.notna(r["beam_um"]) and float(r["beam_um"]) > 0:
        p = replace(p, beam=float(r["beam_um"]) * 1e-6)
    return p


def hybrid_needed(df: pd.DataFrame) -> bool:
    """The hybrid depth model is only relevant if some measured depth/width exceeds the E-T limit of about 0.5."""
    if df is None or not len(df) or "depth_um" not in df or "width_um" not in df:
        return False
    dw = (df["depth_um"] / df["width_um"]).replace([np.inf, -np.inf], np.nan)
    return bool((dw > 0.5).any())


def _hybrid_S_Pe(p: Process, mat: Material, st: ModelSettings):
    """S = A P / (2 a k (Tb - T0))  and  Pe = v a / D, with a = 1/e^2 beam radius = 2 sigma and D = k/(rho cp)."""
    k = mat.k * st.k_scale
    a = 2.0 * p.sigma
    D = k / (mat.rho * mat.cp)
    return p.eta * p.P / (2.0 * a * k * (mat.T_boil - p.T0)), p.v * a / D


@dataclass
class HybridFit:
    coef: np.ndarray
    pe_min: float
    pe_max: float
    r2_g: float
    n: int
    table: pd.DataFrame
    metrics: pd.DataFrame
    warnings: list


def hybrid_fit(df: pd.DataFrame, base: Process, mat: Material, st: ModelSettings, min_tracks: int = 8) -> HybridFit:
    """Calibrate g(Pe) = c0 + c1 Pe + c2 Pe^2 + c3 Pe^3 by LINEAR least squares from g_meas = S / R_meas (S as above)."""
    if not mat.T_boil or mat.T_boil <= base.T0:
        raise ValueError("The hybrid depth model needs the boiling temperature (Material tab).")
    d, _ = _filter_tracks(df, None)
    d = d.dropna(subset=["power_W", "speed_m_s", "width_um", "depth_um"])
    d = d[(d["width_um"] > 0) & (d["depth_um"] > 0)].reset_index(drop=True)
    if len(d) < min_tracks:
        raise ValueError(f"Need at least {min_tracks} tracks with measured width and depth (found {len(d)}).")
    rows = []
    for _, r in d.iterrows():
        p = _row_process(base, r); S, Pe = _hybrid_S_Pe(p, mat, st); g = run_case(p, mat, st)
        R = float(r["depth_um"] / r["width_um"])
        rows.append(dict(power_W=p.P, speed_m_s=p.v, Pe=Pe, S=S, R_meas=R, g_meas=S / R, width_um=float(r["width_um"]), depth_um=float(r["depth_um"]),
                         et_width_um=g["width"] * 1e6, et_depth_um=g["depth"] * 1e6))
    t = pd.DataFrame(rows)
    X = np.vander(t["Pe"].values, 4, increasing=True); gm = t["g_meas"].values
    coef = np.linalg.lstsq(X, gm, rcond=None)[0]
    ghat = X @ coef
    ss_tot = float(np.sum((gm - gm.mean()) ** 2)); r2 = 1.0 - float(np.sum((gm - ghat) ** 2)) / ss_tot if ss_tot > 0 else float("nan")
    warn = []
    if np.any(ghat <= 0):
        warn.append("g(Pe) is not positive for some tracks: the cubic does not describe this data.")
    if t["Pe"].max() / max(t["Pe"].min(), 1e-12) < 2.0:
        warn.append("The tracks cover a narrow Peclet range: the cubic is poorly constrained.")
    loo = np.full(len(t), np.nan)
    for i in range(len(t)):
        m = np.arange(len(t)) != i
        c_i = np.linalg.lstsq(X[m], gm[m], rcond=None)[0]; g_i = float(X[i] @ c_i)
        if g_i > 0:
            loo[i] = t["S"][i] / g_i * t["et_width_um"][i]
    t["hybrid_depth_um"] = np.where(ghat > 0, t["S"].values / np.where(ghat > 0, ghat, 1.0) * t["et_width_um"].values, np.nan)
    t["hybrid_loo_depth_um"] = loo

    def met(pred):
        e = pred - t["depth_um"].values; ok = np.isfinite(e)
        return float(np.sqrt(np.mean(e[ok] ** 2))), float(np.mean(np.abs(e[ok]) / t["depth_um"].values[ok]) * 100)
    m = pd.DataFrame([("plain E-T depth", *met(t["et_depth_um"].values)), ("hybrid depth, in-sample", *met(t["hybrid_depth_um"].values)),
                      ("hybrid depth, leave-one-out", *met(loo))], columns=["depth model", "RMSE [um]", "MAPE [%]"])
    return HybridFit(coef=coef, pe_min=float(t["Pe"].min()), pe_max=float(t["Pe"].max()), r2_g=r2, n=len(t), table=t, metrics=m, warnings=warn)


def hybrid_predict(fit: HybridFit, proc: Process, mat: Material, st: ModelSettings, width_et: float, thresholds=HYBRID_THRESH) -> dict:
    """depth = R * width_ET with R = S / g(Pe); Pe is clamped to the calibrated range (flagged)."""
    S, Pe = _hybrid_S_Pe(proc, mat, st)
    pe_c = min(max(Pe, fit.pe_min), fit.pe_max)
    g = float(_poly3(fit.coef, pe_c))
    if g <= 0:
        return dict(R=float("nan"), depth=float("nan"), regime="invalid (g <= 0)", Pe=Pe, clamped=pe_c != Pe)
    R = S / g
    regime = "conduction" if R < thresholds[0] else "transition" if R <= thresholds[1] else "keyhole"
    return dict(R=R, depth=R * width_et, regime=regime, Pe=Pe, clamped=bool(pe_c != Pe))


# ---------------------------- 5. profile calibration ------------------------------------------------
PHI_CS = np.linspace(0.0, np.pi / 2, 61)       # 60 rays in the cross-section quadrant
PHI_LG = np.linspace(0.0, np.pi, 91)           # longitudinal half plane: 0 = ahead of the beam, pi = behind it


@dataclass
class ProfileTrack:
    name: str
    P: float
    v: float
    y_um: np.ndarray
    z_um: np.ndarray
    x_um: Optional[np.ndarray] = None          # optional longitudinal profile (x ahead of the beam centre positive, z down)
    zl_um: Optional[np.ndarray] = None


def polar_profile(h, z, phi_grid, mirror: bool = True) -> np.ndarray:
    """Boundary points (horizontal coordinate h, depth z down, any common unit) -> r(phi) on a common phi grid."""
    h = np.abs(np.asarray(h, float)) if mirror else np.asarray(h, float)
    z = np.asarray(z, float)
    ph = np.arctan2(z, h); r = np.hypot(h, z)
    u, inv = np.unique(np.round(ph, 12), return_inverse=True)
    rm = np.bincount(inv, weights=r) / np.bincount(inv)
    return np.interp(phi_grid, u, rm)


def profile_distance(r_exp, r_sim, phi, depth_ref: float) -> float:
    """D = sqrt( (1/span) * INT |r_exp - r_sim|^2 dphi ) / depth_ref   (span = pi/2 gives the (2/pi) factor)."""
    span = float(phi[-1] - phi[0])
    return math.sqrt(_trapz((np.asarray(r_exp) - np.asarray(r_sim)) ** 2, phi) / span) / depth_ref


def _phi_weights(phi) -> np.ndarray:
    w = np.empty(len(phi)); d = np.diff(phi)
    w[0], w[-1] = d[0] / 2, d[-1] / 2; w[1:-1] = (d[:-1] + d[1:]) / 2
    return w


def envelope(case: "ETCase", phi, plane: str = "cs", n_x: int = 25, n_iter: int = 14) -> np.ndarray:
    """Liquidus boundary r(phi) [m] by vectorised bisection along rays from the origin (track centreline, surface).
    plane 'cs': (y, z) cross-section, ENVELOPE over x = max over n_x x-samples of T >= T_liq (not a single x-plane).
    plane 'lg': (x, z) longitudinal section at y = 0 (instantaneous pool outline in the beam frame)."""
    g = case.melt_pool()
    if not g["melted"]:
        return np.zeros_like(phi)
    c, sn = np.cos(phi), np.sin(phi)
    if plane == "cs":
        xs = np.linspace(-1.05 * g["tail"], 0.6 * g["head"], n_x)
        r_hi = 1.6 * max(g["width"] / 2, g["depth"], 1e-6); pre = case.prep(float(np.max(np.abs(xs))))
        hot = lambda r: case.dT(xs[None, :], (r * c)[:, None], (r * sn)[:, None], prep=pre).max(axis=1) >= case.theta_l
    else:
        r_hi = 1.25 * max(g["head"], g["tail"], g["depth"], 1e-6); pre = case.prep(r_hi * 2.0)
        hot = lambda r: case.dT(r * c, 0.0, r * sn, prep=pre) >= case.theta_l
    hi = np.full(phi.shape, r_hi)
    for _ in range(8):
        h = hot(hi)
        if not h.any():
            break
        hi = np.where(h, hi * 1.5, hi)
    lo = np.zeros_like(hi); inside0 = hot(lo)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi); h = hot(mid)
        lo = np.where(h, mid, lo); hi = np.where(h, hi, mid)
    return np.where(inside0, 0.5 * (lo + hi), 0.0)


def _track_case(base: Process, t: ProfileTrack, eta: float, beam_scale: float, mat: Material, st: ModelSettings) -> "ETCase":
    return ETCase(replace(base, P=t.P, v=t.v, eta=eta, beam_scale=beam_scale), mat, st)


def profile_residuals(tracks, base, mat, st, eta, beam_scale: float = 1.0, n_x: int = 25):
    """Residual vector whose squared norm per track equals D_CS^2 + D_MF^2; also per-track details."""
    res, info = [], []
    wcs, wlg = _phi_weights(PHI_CS), _phi_weights(PHI_LG)
    for t in tracks:
        depth_exp = float(np.max(t.z_um)) * 1e-6
        case = _track_case(base, t, eta, beam_scale, mat, st)
        r_exp = polar_profile(t.y_um * 1e-6, t.z_um * 1e-6, PHI_CS); r_sim = envelope(case, PHI_CS, "cs", n_x)
        e = (r_exp - r_sim) / depth_exp * np.sqrt(wcs / (PHI_CS[-1] - PHI_CS[0])); res.append(e)
        d_cs = float(np.linalg.norm(e)); d_mf = float("nan"); lg = None
        if t.x_um is not None and t.zl_um is not None and len(t.x_um) >= 5:
            rl_exp = polar_profile(t.x_um * 1e-6, t.zl_um * 1e-6, PHI_LG, mirror=False); rl_sim = envelope(case, PHI_LG, "lg")
            el = (rl_exp - rl_sim) / depth_exp * np.sqrt(wlg / (PHI_LG[-1] - PHI_LG[0])); res.append(el)
            d_mf = float(np.linalg.norm(el)); lg = (rl_exp, rl_sim)
        info.append(dict(track=t.name, D_CS=d_cs, D_MF=d_mf, D_total=math.sqrt(d_cs ** 2 + (0.0 if math.isnan(d_mf) else d_mf ** 2)),
                         r_exp=r_exp, r_sim=r_sim, lg=lg))
    return np.concatenate(res), info


def profile_calibrate(tracks, base: Process, mat: Material, st: ModelSettings, fit_beam_scale: bool = False,
                      eta0: float = 0.4, bounds_eta=(0.05, 1.0), n_x: int = 25) -> dict:
    """Least-squares fit of eta (optionally beam scale) on the profile distance D (Hu et al. 2024 type metric)."""
    if not tracks:
        raise ValueError("No tracks with boundary data.")
    names = ["eta"] + (["beam_scale"] if fit_beam_scale else [])
    x0 = [eta0] + ([1.0] if fit_beam_scale else []); lo = [bounds_eta[0]] + ([0.5] if fit_beam_scale else []); hi = [bounds_eta[1]] + ([2.0] if fit_beam_scale else [])
    fun = lambda x: profile_residuals(tracks, base, mat, st, x[0], x[1] if fit_beam_scale else 1.0, n_x)[0]
    sol = least_squares(fun, x0, bounds=(lo, hi), diff_step=1e-3)
    n, p = sol.jac.shape; s2 = 2 * sol.cost / max(n - p, 1)
    warn = []
    try:
        se = np.sqrt(np.diag(np.linalg.inv(sol.jac.T @ sol.jac) * s2))
    except np.linalg.LinAlgError:
        se = np.full(p, np.nan); warn.append("Jacobian singular: parameters not identifiable from this data.")
    for nme, val, s_ in zip(names, sol.x, se):
        if np.isfinite(s_) and val and s_ / abs(val) > 0.25:
            warn.append(f"{nme} is poorly determined (std. error {s_ / abs(val) * 100:.0f}% of its value).")
    if not any(t.x_um is not None and t.zl_um is not None for t in tracks):
        warn.append("Cross-sections only: the scan-direction extent of the pool is NOT constrained by this data (in a published benchmark the error on that Goldak axis "
                    "was about 13% with cross-sections only and about 1.5% once a longitudinal section was added). Please add a longitudinal section. "
                    "The E-T length shown below is only a labelled PRIOR, not a result of the fit.")
    eta_f, bs_f = float(sol.x[0]), float(sol.x[1]) if fit_beam_scale else 1.0
    _, info = profile_residuals(tracks, base, mat, st, eta_f, bs_f, n_x)
    prior = []
    for t in tracks:
        g = run_case(replace(base, P=t.P, v=t.v, eta=eta_f, beam_scale=bs_f), mat, st)
        prior.append(dict(track=t.name, head_um_ET_prior=g["head"] * 1e6, tail_um_ET_prior=g["tail"] * 1e6, length_um_ET_prior=g["length"] * 1e6))
    tab = pd.DataFrame([{k: v for k, v in i.items() if k in ("track", "D_CS", "D_MF", "D_total")} for i in info])
    return dict(params=dict(zip(names, map(float, sol.x))), std_err=dict(zip(names, map(float, se))), table=tab, prior=pd.DataFrame(prior),
                warnings=warn, info=info, names=names, n_tracks=len(tracks), D_rms=float(np.sqrt(np.mean(tab["D_total"] ** 2))))


def profile_demo(base: Process, mat: Material, st: ModelSettings, conds, eta_true: float = 0.30, noise: float = 0.02,
                 longitudinal: bool = True, seed: int = 0):
    """Synthetic digitised boundaries from the model itself (known eta), for trying the module out."""
    rng = np.random.default_rng(seed); tracks = []
    for i, (P, v) in enumerate(conds, 1):
        case = ETCase(replace(base, P=P, v=v, eta=eta_true), mat, st)
        r = envelope(case, PHI_CS, "cs") * (1 + noise * rng.standard_normal(len(PHI_CS)))
        t = ProfileTrack(f"T{i}", P, v, r * np.cos(PHI_CS) * 1e6, r * np.sin(PHI_CS) * 1e6)
        if longitudinal:
            rl = envelope(case, PHI_LG, "lg") * (1 + noise * rng.standard_normal(len(PHI_LG)))
            t.x_um, t.zl_um = rl * np.cos(PHI_LG) * 1e6, rl * np.sin(PHI_LG) * 1e6
        tracks.append(t)
    return tracks


def tracks_to_tables(tracks):
    tr = pd.DataFrame([(t.name, t.P, t.v) for t in tracks], columns=["track", "power_W", "speed_m_s"])
    cs = pd.concat([pd.DataFrame({"track": t.name, "y_um": t.y_um, "z_um": t.z_um}) for t in tracks], ignore_index=True)
    lg = [pd.DataFrame({"track": t.name, "x_um": t.x_um, "z_um": t.zl_um}) for t in tracks if t.x_um is not None]
    return tr, cs, (pd.concat(lg, ignore_index=True) if lg else pd.DataFrame({"track": pd.Series(dtype=str), "x_um": pd.Series(dtype=float), "z_um": pd.Series(dtype=float)}))


def tables_to_tracks(tr: pd.DataFrame, cs: pd.DataFrame, lg: Optional[pd.DataFrame]) -> tuple[list, list]:
    out, errs = [], []
    for _, r in tr.dropna(subset=["power_W", "speed_m_s"]).iterrows():
        nm = str(r["track"])
        c = cs[cs["track"].astype(str) == nm].dropna(subset=["y_um", "z_um"])
        if len(c) < 5 or c["z_um"].max() <= 0:
            errs.append(f"Track {nm}: need at least 5 cross-section boundary points with positive z."); continue
        t = ProfileTrack(nm, float(r["power_W"]), float(r["speed_m_s"]), c["y_um"].values.astype(float), c["z_um"].values.astype(float))
        if lg is not None and len(lg):
            l = lg[lg["track"].astype(str) == nm].dropna(subset=["x_um", "z_um"])
            if len(l) >= 5:
                t.x_um, t.zl_um = l["x_um"].values.astype(float), l["z_um"].values.astype(float)
        out.append(t)
    return out, errs


def fig_profiles(tracks, info):
    plt = _mpl()
    n = len(tracks); ncol = 2 if any(t.x_um is not None for t in tracks) else 1
    fig, ax = plt.subplots(n, ncol, figsize=(5.2 * ncol, 3.2 * n), squeeze=False)
    for i, (t, d) in enumerate(zip(tracks, info)):
        a = ax[i, 0]
        a.plot(np.r_[-t.y_um[::-1], t.y_um], np.r_[t.z_um[::-1], t.z_um], "o", ms=3, label="measured")
        ph = PHI_CS; rs = d["r_sim"] * 1e6
        a.plot(np.r_[-(rs * np.cos(ph))[::-1], rs * np.cos(ph)], np.r_[(rs * np.sin(ph))[::-1], rs * np.sin(ph)], "r-", label="E-T envelope")
        a.invert_yaxis(); a.set_aspect("equal"); a.set_title(f"{t.name}: cross-section, D = {d['D_CS']:.3f}", fontsize=9); a.set_xlabel("y [um]"); a.set_ylabel("z [um]"); a.legend(fontsize=7)
        if ncol == 2:
            b = ax[i, 1]
            if d["lg"] is not None:
                b.plot(t.x_um, t.zl_um, "o", ms=3); rl = d["lg"][1] * 1e6
                b.plot(rl * np.cos(PHI_LG), rl * np.sin(PHI_LG), "r-"); b.invert_yaxis(); b.set_aspect("equal")
                b.set_title(f"{t.name}: longitudinal, D = {d['D_MF']:.3f}", fontsize=9); b.set_xlabel("x [um] (+ ahead of beam)")
            else:
                b.axis("off")
    fig.tight_layout(); return fig


# ---------------------------- 6. FE-informed correction and Goldak surrogate ----------------------------
FE_OUTS = ("width", "depth", "length")


def fe_fingerprint(base: Process, mat: Material, st: ModelSettings) -> tuple:
    return (round(base.eta, 6), round(base.beam, 9), base.beam_def, round(base.beam_scale, 6), round(mat.k * st.k_scale, 6), round(mat.rho, 3), round(mat.cp, 3), round(mat.T_liq, 3))


def fe_fit(df_fe: pd.DataFrame, base: Process, mat: Material, st: ModelSettings, ridge: float = 1e-3, min_rows: int = 6) -> dict:
    """6a) y_FE = y_ET * (1 + c0 + c1 Pn + c2 vn) per output by ridge least squares (ridge on c1, c2 only);
    Pn, vn in [-1, 1] over the DOE range. Leave-one-out via the hat matrix. Refuses to extrapolate outside the DOE range."""
    d = df_fe.dropna(subset=["power_W", "speed_m_s"]).reset_index(drop=True)
    if len(d) < min_rows:
        raise ValueError(f"Need at least {min_rows} Abaqus results (10-25 recommended); found {len(d)}.")
    P, v = d["power_W"].values.astype(float), d["speed_m_s"].values.astype(float)
    rng_ = dict(P=(P.min(), P.max()), v=(v.min(), v.max()))
    norm = lambda x, lh: np.zeros_like(x) if lh[1] <= lh[0] else 2 * (x - lh[0]) / (lh[1] - lh[0]) - 1
    X = np.column_stack([np.ones(len(d)), norm(P, rng_["P"]), norm(v, rng_["v"])]); Lam = np.diag([0.0, ridge, ridge])
    et = [run_case(replace(base, P=float(a), v=float(b)), mat, st) for a, b in zip(P, v)]
    coef, mets, tabs, warn = {}, [], pd.DataFrame({"power_W": P, "speed_m_s": v}), []
    for o in FE_OUTS:
        col = f"{o}_um"
        if col not in d or d[col].notna().sum() < min_rows:
            continue
        y_fe = d[col].values.astype(float); y_et = np.array([e[o] * 1e6 for e in et]); ok = np.isfinite(y_fe) & (y_et > 0)
        if ok.sum() < min_rows:
            continue
        Xo, ratio = X[ok], y_fe[ok] / y_et[ok] - 1.0
        A = Xo.T @ Xo + Lam; Ai = np.linalg.inv(A); c = Ai @ Xo.T @ ratio
        h = np.einsum("ij,jk,ik->i", Xo, Ai, Xo); fitted = Xo @ c; loo_ratio = ratio - (ratio - fitted) / (1 - h)
        yf, ye = y_fe[ok], y_et[ok]; y_corr, y_loo = ye * (1 + fitted), ye * (1 + loo_ratio)
        rm = lambda a: float(np.sqrt(np.mean((a - yf) ** 2))); mp = lambda a: float(np.mean(np.abs(a - yf) / yf) * 100)
        coef[o] = c
        mets.append(dict(output=o, n=int(ok.sum()), raw_ET_RMSE_um=rm(ye), raw_ET_MAPE_pct=mp(ye), corrected_in_sample_RMSE_um=rm(y_corr),
                         corrected_LOO_RMSE_um=rm(y_loo), corrected_LOO_MAPE_pct=mp(y_loo)))
        tabs.loc[ok, f"{o}_ET_um"] = ye; tabs.loc[ok, f"{o}_FE_um"] = yf; tabs.loc[ok, f"{o}_corrected_LOO_um"] = y_loo
        if mets[-1]["corrected_LOO_RMSE_um"] >= mets[-1]["raw_ET_RMSE_um"]:
            warn.append(f"{o}: the correction does NOT beat raw E-T in leave-one-out; do not use it for {o}.")
    if not coef:
        raise ValueError("No output column (width_um, depth_um, length_um) has enough values.")
    if len(d) < 10:
        warn.append("Fewer than 10 Abaqus results: the correction is poorly constrained.")
    fp = fe_fingerprint(base, mat, st)
    return dict(coef=coef, ranges=rng_, table=tabs, metrics=pd.DataFrame(mets), warnings=warn, fingerprint=fp, n=len(d))


def fe_predict(model: dict, P: float, v: float, et: dict) -> dict:
    """Corrected width/depth/length [um] from E-T values [um]; raises ValueError outside the DOE range (no extrapolation)."""
    (p0, p1), (v0, v1) = model["ranges"]["P"], model["ranges"]["v"]
    tol = 1e-9
    if not (p0 - tol <= P <= p1 + tol and v0 - tol <= v <= v1 + tol):
        raise ValueError(f"(P, v) = ({P:g} W, {v:g} m/s) is outside the Abaqus DOE range P {p0:g}-{p1:g} W, v {v0:g}-{v1:g} m/s: no extrapolation.")
    pn = 0.0 if p1 <= p0 else 2 * (P - p0) / (p1 - p0) - 1; vn = 0.0 if v1 <= v0 else 2 * (v - v0) / (v1 - v0) - 1
    return {o: et[o] * (1 + c[0] + c[1] * pn + c[2] * vn) for o, c in model["coef"].items()}


def lhs_doe(ranges: dict, n: int = 36, seed: int = 0) -> pd.DataFrame:
    """6b) Latin-hypercube design over {name: (lo, hi)}; lo == hi fixes that parameter."""
    from scipy.stats import qmc
    names = list(ranges); lo = np.array([ranges[k][0] for k in names], float); hi = np.array([ranges[k][1] for k in names], float)
    u = qmc.LatinHypercube(d=len(names), seed=seed).random(int(n))
    return pd.DataFrame(lo + u * (hi - lo), columns=names)


class Surrogate:
    """RBF (thin-plate spline) or quadratic-ridge surrogate on the free (varying) inputs, one model for all outputs."""

    def __init__(self, method: str = "rbf", smoothing: float = 1e-6, ridge: float = 1e-6):
        self.method, self.smoothing, self.ridge = method, smoothing, ridge

    def _feat(self, U):
        cols = [np.ones(len(U))] + [U[:, i] for i in range(U.shape[1])] + [U[:, i] * U[:, j] for i in range(U.shape[1]) for j in range(i, U.shape[1])]
        return np.column_stack(cols)

    def _fit_core(self, U, Yn):
        if self.method == "rbf":
            from scipy.interpolate import RBFInterpolator
            return RBFInterpolator(U, Yn, kernel="thin_plate_spline", smoothing=self.smoothing)
        F = self._feat(U); pen = np.eye(F.shape[1]) * self.ridge; pen[0, 0] = 0
        W = np.linalg.solve(F.T @ F + pen, F.T @ Yn)
        return lambda Uq: self._feat(np.atleast_2d(Uq)) @ W

    def fit(self, X, Y, in_names, out_names):
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        self.in_names, self.out_names = list(in_names), list(out_names)
        self.lo, self.hi = X.min(0), X.max(0); self.var = self.hi > self.lo
        self.ylo, self.yhi = Y.min(0), Y.max(0); self.ym, self.ys = Y.mean(0), np.where(Y.std(0) > 0, Y.std(0), 1.0)
        self.X, self.Y = X, Y
        self._model = self._fit_core(self._scale(X), (Y - self.ym) / self.ys)
        return self

    def _scale(self, X):
        X = np.atleast_2d(X)
        return (X[:, self.var] - self.lo[self.var]) / (self.hi[self.var] - self.lo[self.var])

    def predict(self, X):
        return np.asarray(self._model(self._scale(X))) * self.ys + self.ym

    def loo(self) -> pd.DataFrame:
        n = len(self.X); P = np.empty_like(self.Y)
        for i in range(n):
            m = np.arange(n) != i
            sub = Surrogate(self.method, self.smoothing, self.ridge); sub.in_names, sub.out_names = self.in_names, self.out_names
            sub.lo, sub.hi, sub.var = self.lo, self.hi, self.var
            sub.ym, sub.ys = self.Y[m].mean(0), np.where(self.Y[m].std(0) > 0, self.Y[m].std(0), 1.0)
            core = sub._fit_core(self._scale(self.X[m]), (self.Y[m] - sub.ym) / sub.ys)
            P[i] = np.asarray(core(self._scale(self.X[i:i + 1]))) * sub.ys + sub.ym
        e = P - self.Y
        return pd.DataFrame({"output": self.out_names, "LOO_RMSE": np.sqrt((e ** 2).mean(0)), "LOO_MAPE_pct": (np.abs(e) / np.abs(self.Y)).mean(0) * 100,
                             "training_range_lo": self.ylo, "training_range_hi": self.yhi})


def inverse_de(sur: Surrogate, targets: dict, weights: dict | None = None, seed: int = 1, popsize: int = 15, maxiter: int = 200, n_cand: int = 3,
               fixed: dict | None = None) -> dict:
    """Minimise the weighted relative mismatch to the measured targets over the surrogate's input box (scipy differential_evolution).
    targets: {output name: value}; an output named 'D' (profile metric) is driven toward 0."""
    from scipy.optimize import differential_evolution
    w = {o: (weights or {}).get(o, 1.0) for o in targets}
    idx = {o: sur.out_names.index(o) for o in targets}
    warn = []
    fixed = fixed or {}
    free = int(sum(1 for i, n in enumerate(sur.in_names) if sur.var[i] and n not in fixed))
    nt = sum(1 for o in targets if o != "D") + (1 if "D" in targets else 0)
    if nt < free:
        warn.append(f"{nt} target(s) for {free} free parameter(s): the solution is NOT unique. Fix some parameters (e.g. eta from the E-T calibration) by giving them a single value in the DOE range.")
    for o, t in targets.items():
        if o != "D" and not (sur.ylo[idx[o]] <= t <= sur.yhi[idx[o]]):
            warn.append(f"Target {o} = {t:g} is outside the DOE output range [{sur.ylo[idx[o]]:.4g}, {sur.yhi[idx[o]]:.4g}]: widen the DOE ranges and run more cases.")

    def obj(x):
        y = sur.predict(x)[0]
        return float(sum(w[o] * (((y[idx[o]]) if o == "D" else (y[idx[o]] - t) / t) ** 2) for o, t in targets.items()))
    bnds = [(fixed[n], fixed[n]) if n in fixed else (float(sur.lo[i]), float(sur.hi[i])) for i, n in enumerate(sur.in_names)]
    for n, val in fixed.items():
        i = sur.in_names.index(n)
        if not (sur.lo[i] - 1e-9 <= val <= sur.hi[i] + 1e-9):
            warn.append(f"Fixed {n} = {val:g} is outside the DOE range [{sur.lo[i]:g}, {sur.hi[i]:g}].")
    res = differential_evolution(obj, bnds, seed=seed, popsize=popsize, maxiter=maxiter, tol=1e-10, polish=True, updating="deferred")
    cands = [res.x]
    pop, en = np.asarray(res.population), np.asarray(res.population_energies)
    span = np.where(sur.hi > sur.lo, sur.hi - sur.lo, 1.0)
    for i in np.argsort(en):
        if len(cands) >= n_cand:
            break
        if all(np.max(np.abs(pop[i] - c) / span) > 0.08 for c in cands):
            cands.append(pop[i])
    rows = []
    for j, c in enumerate(cands):
        y = sur.predict(c)[0]; row = {"candidate": "best" if j == 0 else f"alternative {j}", **dict(zip(sur.in_names, c))}
        row.update({f"pred_{o}": y[sur.out_names.index(o)] for o in sur.out_names}); row["weighted_mismatch"] = obj(c)
        rows.append(row)
    return dict(best=dict(zip(sur.in_names, res.x)), table=pd.DataFrame(rows), warnings=warn, objective=float(res.fun))


# ---------------------------- 7. process-map overlays ------------------------------------------------
def process_map_criteria(sw: pd.DataFrame, hatch_um: float, layer_um: float, kh_wd: float = 1.5, depth_col: str = "depth_um") -> pd.DataFrame:
    """Lack of fusion: (H/w)^2 + (L/d)^2 <= 1 is fully fused.  Keyhole avoidance: w/d > kh_wd.  H = hatch, L = layer thickness."""
    out = sw.copy(); w = out["width_um"].values.astype(float); d = out[depth_col].values.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["lof_index"] = np.where((w > 0) & (d > 0), (hatch_um / w) ** 2 + (layer_um / d) ** 2, np.inf)
        out["w_over_d"] = np.where(d > 0, w / d, np.nan)
    out["lof_ok"] = out["lof_index"] <= 1.0
    out["keyhole_ok"] = out["w_over_d"] > kh_wd
    out["window_ok"] = out["lof_ok"] & out["keyhole_ok"]
    return out


def fig_window(sc: pd.DataFrame, kh_wd: float = 1.5):
    plt = _mpl()
    Ps = np.sort(sc["power_W"].unique()); Vs = np.sort(sc["speed_m_s"].unique())
    piv = lambda c: sc.pivot(index="power_W", columns="speed_m_s", values=c).reindex(index=Ps, columns=Vs).values
    fig, a = plt.subplots(figsize=(7, 5))
    a.pcolormesh(Vs, Ps, piv("window_ok").astype(float), shading="nearest", cmap="Greens", vmin=0, vmax=1.6, alpha=0.8)
    L, K = piv("lof_index"), piv("w_over_d")
    if len(Ps) > 1 and len(Vs) > 1:
        Lc = np.where(np.isfinite(L), L, 1e6)
        if Lc.min() <= 1 <= Lc.max():
            a.contour(Vs, Ps, Lc, levels=[1.0], colors="tab:blue", linewidths=2); a.plot([], [], "-", color="tab:blue", label="lack of fusion: (H/w)^2+(L/d)^2 = 1")
        if np.nanmin(K) <= kh_wd <= np.nanmax(K):
            a.contour(Vs, Ps, np.nan_to_num(K), levels=[kh_wd], colors="tab:red", linewidths=2); a.plot([], [], "-", color="tab:red", label=f"keyhole limit: w/d = {kh_wd}")
    a.plot([], [], "s", color="tab:green", alpha=0.8, label="process window")
    a.legend(loc="upper left", fontsize=8); a.set_xlabel("scan speed [m/s]"); a.set_ylabel("power [W]"); a.set_title("Process window")
    fig.tight_layout(); return fig


def fig_hybrid(fit: HybridFit):
    plt = _mpl(); t = fit.table
    fig, a = plt.subplots(figsize=(5.2, 4.6)); m = t["depth_um"]
    a.plot(m, t["et_depth_um"], "s", color="gray", label="plain E-T")
    a.plot(m, t["hybrid_loo_depth_um"], "o", color="tab:blue", label="hybrid (leave-one-out)")
    lim = [0, float(np.nanmax([m.max(), t["et_depth_um"].max(), t["hybrid_loo_depth_um"].max()])) * 1.1]
    a.plot(lim, lim, "k--", lw=1); a.set_xlim(lim); a.set_ylim(lim); a.set_xlabel("measured depth [um]"); a.set_ylabel("predicted depth [um]"); a.legend(fontsize=8)
    fig.tight_layout(); return fig


def _tint(hexcol: str, f: float = 0.88) -> str:
    h = hexcol.lstrip("#"); r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return "#%02X%02X%02X" % tuple(int(c + (255 - c) * f) for c in (r, g, b))


def build_workbook(proc: Process, mat: Material, stg: ModelSettings, goldak_conv: str = "goldak", cal: dict | None = None,
                   sweep_df: pd.DataFrame | None = None, prop_tabs: dict | None = None, prop_eff: dict | None = None,
                   prop_notes: list | None = None, extra: list | None = None) -> bytes:
    """ONE formatted Excel workbook with everything: summary, inputs, Goldak values, calibration, sweep, properties, notes, references."""
    import datetime
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    NAVY, ZEBRA = PatternFill("solid", fgColor="1F3864"), PatternFill("solid", fgColor="EEF3FA")
    thin = Side(style="thin", color="BFBFBF"); BORDER = Border(left=thin, right=thin, top=thin, bottom=thin)

    def title(ws, text, sub, color="1F3864"):
        ws["A1"] = text; ws["A1"].font = Font(bold=True, size=15, color=color)
        ws["A2"] = sub; ws["A2"].font = Font(italic=True, color="595959")

    def put(ws, df, r0, widths, wrap=(), fmt=None, hdr_color="1F3864"):
        for j, c in enumerate(df.columns, 1):
            cell = ws.cell(r0, j, str(c)); cell.font = Font(bold=True, color="FFFFFF"); cell.fill = PatternFill("solid", fgColor=hdr_color)
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True); cell.border = BORDER
        for i, row in enumerate(df.itertuples(index=False), 1):
            for j, v in enumerate(row, 1):
                if isinstance(v, (np.floating, np.integer)):
                    v = v.item()
                if isinstance(v, float) and not math.isfinite(v):
                    v = None
                if isinstance(v, (np.bool_,)):
                    v = bool(v)
                cell = ws.cell(r0 + i, j, v); cell.border = BORDER
                cell.alignment = Alignment(vertical="top", wrap_text=(df.columns[j - 1] in wrap))
                if i % 2 == 0:
                    cell.fill = ZEBRA
                f = fmt(i - 1, j - 1, v) if fmt else None
                if f:
                    cell.number_format = f
        for j, w in enumerate(widths, 1):
            ws.column_dimensions[chr(64 + j)].width = w
        ws.freeze_panes = ws.cell(r0 + 1, 1)
        return r0 + len(df) + 2

    res = run_case(proc, mat, stg)
    wb = Workbook(); ws = wb.active; ws.title = "Summary"
    title(ws, "Eagar-Tsai melt pool report", f"Generated {datetime.date.today().isoformat()}  |  material: {mat.name}  |  P = {proc.P:g} W, v = {proc.v:g} m/s, "
          f"beam {proc.beam * 1e6:g} um ({proc.beam_def}), effective absorptivity {proc.eta:g}, T0 = {proc.T0:g} K")
    if not res["melted"]:
        ws["A4"] = f"NO MELTING: peak temperature {res['peak_T']:.0f} K is below the liquidus {mat.T_liq:.0f} K."
        ws["A4"].font = Font(bold=True, color="C00000"); r = 6
    else:
        rt = results_table(res, numeric=True); fmts = [x[3] for x in RESULT_ROWS]
        ws["A4"] = "Key results"; ws["A4"].font = Font(bold=True, size=12, color="2E7D32")
        r = put(ws, rt, 5, [48, 16, 9, 95], wrap=("what it means", "quantity"), hdr_color="2E7D32",
                fmt=lambda i, j, v: fmts[i] if j == 1 and fmts[i] else None)
        reg = res["regime"]; cell = ws.cell(5 + len(RESULT_ROWS), 2)
        col = "F8CBAD" if ("KEYHOLE" in reg or "extreme" in reg) else "FFE699" if ("transition" in reg or "T_boil" in reg) else "C6E0B4"
        cell.fill = PatternFill("solid", fgColor=col); cell.alignment = Alignment(wrap_text=True, vertical="top")
        ws.row_dimensions[5 + len(RESULT_ROWS)].height = 45
    for line in ("E-T underpredicts keyhole depth; flags are editable heuristics; a calibrated absorptivity is an EFFECTIVE fitting value, not a measured one, and does not transfer between alloys.",):
        ws.cell(r, 1, line).font = Font(italic=True, color="7F6000")

    ws = wb.create_sheet("Inputs"); title(ws, "Inputs used", "Everything the numbers in this workbook were computed from; SI units, temperatures in K.")
    put(ws, inputs_table(proc, mat, stg), 4, [40, 12, 18, 70], wrap=("source",), hdr_color="1565C0")

    if res["melted"]:
        ws = wb.create_sheet("Goldak"); title(ws, "Goldak double-ellipsoid starting values", "Heuristic starting guesses from the E-T melt pool; refine against sections in your FE model.")
        put(ws, goldak_table(goldak_start(res), proc.eta, proc.P, goldak_conv), 4, [38, 14, 8, 20, 90], wrap=("what it is",), hdr_color="6D4C41",
            fmt=lambda i, j, v: "0.00" if j == 1 else "0.000E+00" if j == 3 else None)

    if cal:
        ws = wb.create_sheet("Calibration"); title(ws, "Calibration against measured tracks", f"{cal['n_used']} tracks used; material '{cal['material']}'. The fitted absorptivity is an EFFECTIVE parameter, not a measured absorptivity.", "C62828")
        meaning = {"eta": "Effective absorptivity: fraction of laser power that appears as heat in the model.", "beam_scale": "Multiplier on the beam size.", "k_scale": "Multiplier on thermal conductivity."}
        pt = pd.DataFrame({"parameter": cal["names"], "value": [cal["params"][n] for n in cal["names"]], "std. error": [cal["std_err"][n] for n in cal["names"]],
                           "meaning": [meaning.get(n, "") for n in cal["names"]]})
        r = put(ws, pt, 4, [34, 14, 14, 16, 16, 16, 16, 16, 16, 16], wrap=("meaning",), hdr_color="C62828", fmt=lambda i, j, v: "0.0000" if j in (1, 2) else None)
        for w_ in cal.get("warnings", []):
            ws.cell(r, 1, "WARNING: " + w_).font = Font(bold=True, color="C00000"); r += 1
        mt = pd.DataFrame({"metric": ["RMSE [um]", "MAPE [%]", "R2"], **{c: [cal["rmse"].get(c), cal["mape"].get(c), cal["r2"].get(c)] for c in ("width", "depth", "length") if c in cal["rmse"]}})
        ws.cell(r, 1, "Training metrics").font = Font(bold=True); r = put(ws, mt, r + 1, [34, 14, 14, 16, 16, 16, 16, 16, 16, 16], hdr_color="C62828", fmt=lambda i, j, v: "0.00" if j > 0 else None)
        tab = pd.DataFrame(cal["table"]); ws.cell(r, 1, "Per-track comparison (measured vs predicted)").font = Font(bold=True)
        r = put(ws, tab, r + 1, [34, 14, 14, 16, 16, 16, 16, 16, 16, 16], hdr_color="C62828", fmt=lambda i, j, v: "0.00" if isinstance(v, float) else None)
        if cal.get("cv"):
            ws.cell(r, 1, "Validation on tracks not used in each fit").font = Font(bold=True)
            put(ws, pd.DataFrame(cal["cv"]["table"]), r + 1, [34, 14, 14, 16, 16, 16, 16, 16, 16, 16], hdr_color="C62828", fmt=lambda i, j, v: "0.00" if isinstance(v, float) else None)
        ws.freeze_panes = None

    if sweep_df is not None and len(sweep_df):
        ws = wb.create_sheet("Sweep"); title(ws, "Power-speed sweep", "One row per power/speed pair; same material, beam, absorptivity and T0 as the Inputs sheet.", "283593")
        put(ws, sweep_df, 4, [12, 14, 12, 12, 12, 11, 13, 18, 50], hdr_color="283593",
            fmt=lambda i, j, v: ("0.0" if isinstance(v, float) else None))

    if prop_tabs:
        ws = wb.create_sheet("Properties"); title(ws, "Temperature-dependent property table", "As entered (converted to K and SI units) and the effective constants used by the model.", "EF6C00")
        merged = None
        for p, d in prop_tabs.items():
            d2 = d.rename(columns={"value": PROP_LABEL[p]})
            merged = d2 if merged is None else merged.merge(d2, on="T_K", how="outer")
        merged = merged.sort_values("T_K").rename(columns={"T_K": "temperature [K]"})
        r = put(ws, merged, 4, [20, 24, 24, 24, 24, 24], hdr_color="EF6C00", fmt=lambda i, j, v: "0.000" if isinstance(v, float) else None)
        if prop_eff:
            ws.cell(r, 1, "Effective constants").font = Font(bold=True)
            unit = {"k": "W/m/K", "rho": "kg/m3", "cp": "J/kg/K"}
            r = put(ws, pd.DataFrame([(p, unit[p], v) for p, v in prop_eff.items()], columns=["property", "unit", "effective constant"]), r + 1, [20, 24, 24, 24, 24, 24],
                    hdr_color="EF6C00", fmt=lambda i, j, v: "0.000" if j == 2 else None)
        for n_ in (prop_notes or []):
            ws.cell(r, 1, n_).alignment = Alignment(wrap_text=False); r += 1
        ws.freeze_panes = None

    for nm, ttl, sub, edf, colr in (extra or []):
        if edf is None or not len(edf):
            continue
        ws = wb.create_sheet(nm[:31]); title(ws, ttl, sub, colr.lstrip("#"))
        put(ws, edf, 4, [max(14, min(40, len(str(c)) + 4)) for c in edf.columns], hdr_color=colr.lstrip("#"),
            fmt=lambda i, j, v: "0.000" if isinstance(v, float) else None)

    ws = wb.create_sheet("Notes"); title(ws, "Assumptions and limits", "Read before using the numbers.", "546E7A")
    notes = ["Eagar-Tsai: Gaussian surface source on a semi-infinite solid, CONSTANT k, rho, cp; no latent heat, convection, radiation, Marangoni flow or evaporation.",
             "Conduction mode only. E-T underpredicts keyhole depth; peak temperatures are unrealistically high at high power.",
             "Melt pool = region above the liquidus. Width at the surface, depth on the centreline plane, length along the surface centreline.",
             "G and cooling rate are E-T estimates at the trailing liquidus point, not FEM results.",
             "Depth/width, enthalpy and peak-temperature flags are editable heuristics, not laws.",
             "Calibrated absorptivity is an effective fitting parameter (absorbs powder-bed, property, beam and model errors); it does not transfer between alloys or machines.",
             "Goldak values are starting guesses derived from the melt pool geometry, not calibrated parameters."]
    for i, n_ in enumerate(notes, 4):
        ws.cell(i, 1, f"{i - 3}. {n_}").alignment = Alignment(wrap_text=True, vertical="top"); ws.row_dimensions[i].height = 32
    ws.column_dimensions["A"].width = 150

    ws = wb.create_sheet("References"); title(ws, "References and where things come from", "Status column says how each item was verified.", "AD1457")
    put(ws, pd.DataFrame(REFERENCES, columns=["topic", "reference", "used for", "how it was checked"]), 4, [28, 90, 70, 50],
        wrap=("reference", "used for", "how it was checked", "topic"), hdr_color="AD1457")
    bio = io.BytesIO(); wb.save(bio); return bio.getvalue()


# ----------------------------------------------------------------------
# UI helpers: colours, how-to boxes, lazy downloads
# ----------------------------------------------------------------------
TAB_PALETTE = [("#1565C0", "#E3F2FD"), ("#EF6C00", "#FFF3E0"), ("#6A1B9A", "#F3E5F5"), ("#00838F", "#E0F7FA"), ("#2E7D32", "#E8F5E9"),
               ("#C62828", "#FFEBEE"), ("#283593", "#E8EAF6"), ("#6D4C41", "#EFEBE9"), ("#827717", "#F9FBE7"), ("#00695C", "#E0F2F1"), ("#5E35B1", "#EDE7F6"), ("#546E7A", "#ECEFF1"), ("#AD1457", "#FCE4EC")]
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

TAB_INFO = [
    dict(why="Describes the laser pass you want to simulate: power, speed, spot size and how much of the light heats the metal.",
         steps=["Enter laser <b>power</b> and <b>scan speed</b> and pick their units.",
                "Enter the <b>beam size</b> and choose its definition exactly as your laser data sheet gives it (D4&sigma;, 1/e&sup2; diameter or 1/e&sup2; radius). The app converts it to the Gaussian &sigma;.",
                "<b>Absorptivity &eta;</b> = fraction of laser power that becomes heat. If you do not know it, start near 0.3-0.5 and let the <b>Calibration</b> tab find a value from your measured tracks.",
                "<b>T0</b> = temperature of the plate/powder bed before the pass (room temperature is about 293-300 K; use your preheat if you have one)."],
         notes="Everything is converted to SI internally. Layer thickness and hatch are only used for the energy-density numbers. Results update as soon as you change a value."),
    dict(why="The model needs the thermal properties and melting range of YOUR metal. No alloy is built in, so nothing is silently assumed.",
         steps=["Type conductivity <b>k</b>, density <b>&rho;</b>, specific heat <b>cp</b> and the <b>liquidus</b> temperature (SI units, temperatures in kelvin: K = &deg;C + 273.15).",
                "Write where every value came from in its <b>source</b> box (paper, handbook, JMatPro, your DSC run).",
                "Your properties change with temperature? Fill the <b>temperature table</b> at the bottom of this tab (or load it from a file). The app averages it between T0 and the liquidus and fills k, &rho; and cp for you.",
                "Use <b>Save this material</b> to reuse it in later sessions."],
         notes="This is a constant-property model, so use values representative of the range from T0 up to the liquidus. Latent heat is a single number (J/kg): it only changes the result if you add it to cp."),
    dict(why="Controls numerical accuracy and the thresholds used to warn you when a pool is outside what the model can describe (keyhole mode, extreme temperature).",
         steps=["Leave the numerical settings at their defaults unless you are testing convergence (they are accurate to about 1e-6 for normal cases).",
                "Adjust the <b>depth/width</b> thresholds to your alloy and the literature: they decide when a pool is called <i>transition</i> or <i>keyhole-suspect</i>.",
                "Optional: use the button at the bottom to compare with the independent <b>eagar-tsai</b> library."],
         notes="The thresholds are rules of thumb, not physical constants. The conductivity scale factor is normally filled by the Calibration tab."),
    dict(why="Your measured single-track melt pools are what the model is calibrated against.",
         steps=["Prepare <b>one row per single track</b>: power, speed and the measured width and/or depth (see the format box).",
                "<b>Upload</b> a CSV/XLSX or <b>paste</b> cells from Excel, choose which column is which, pick the units and press <b>Load into table</b>.",
                "Edit, add or delete rows directly in the table. Tick <b>exclude</b> to ignore a track without deleting it.",
                "Optional: type one measured (or Abaqus) width/depth below: it is drawn on the Results plots and the Results tab solves for the matching absorptivity."],
         notes="Deep keyhole-like tracks (depth about equal to or larger than width) are poorly described by E-T; the Calibration tab can leave them out automatically."),
    dict(why="Shows the predicted melt pool for the current inputs, so you can compare with a micrograph before printing.",
         steps=["Read the <b>regime</b> message first: green = conduction mode (model valid), orange/red = treat depth with caution.",
                "Compare <b>width</b> and <b>depth</b> with your cross-sections; the table says what every number means.",
                "Have a measured (or Abaqus) width? Type it in the Experimental data tab (overlay boxes): this tab then shows the <b>absorptivity that reproduces it</b>.",
                "Tick <b>sensitivity</b> to see which input changes the result most.",
                "Download figures here, or get everything in <b>one Excel report</b> (button below and in the sidebar)."],
         notes="G and cooling rate are quick E-T estimates for orientation; use FEM for quantitative thermal histories."),
    dict(why="Absorptivity (and optionally beam size and conductivity) are the least certain inputs. This tab adjusts them so the model reproduces YOUR measured tracks.",
         steps=["Load at least 2 tracks in tab 4 (5 or more is much better).",
                "Keep the default (<b>fit absorptivity only</b>) unless the warnings say your data can support more.",
                "Press <b>Run calibration</b> and read the warnings first, then the parity plot (points on the diagonal = good fit).",
                "Choose a validation mode to test on tracks that were not used for the fit, then <b>Apply</b> the values to the other tabs."],
         notes="The fitted absorptivity is an effective fitting parameter, not a measured material property, and it does not transfer to another alloy or machine."),
    dict(why="Maps how width, depth and length change over a grid of power and speed: a quick process-window overview.",
         steps=["Set the power and speed ranges and the number of steps (9 x 9 takes a few seconds).",
                "Press <b>Run sweep</b> and choose which quantity to colour the map with.",
                "Dashed lines mark the depth/width thresholds from Model settings."],
         notes="Uses the Process, Material and Model-settings values currently entered (beam size, absorptivity, T0)."),
    dict(why="If you will simulate the build with finite elements (e.g. Abaqus DFLUX), this gives starting dimensions for a Goldak double-ellipsoid heat source.",
         steps=["Pick the naming convention your FE model or paper uses.",
                "Copy the values (lengths in micrometres, SI column in metres) into your heat-source definition.",
                "Refine them in the FE model against your micrographs."],
         notes="These are heuristic starting guesses derived from the E-T melt pool geometry, not calibrated Goldak parameters."),
    dict(why="E-T can never give a depth larger than about half the width. For deep, keyhole-like tracks (measured depth/width above 0.5) this tab calibrates a hybrid depth model.",
         steps=["Load at least 8 tracks with measured width AND depth in tab 4 (and enter the boiling temperature in the Material tab). The tab unlocks by itself when some track has depth/width above 0.5.",
                "Calibrate the absorptivity on the conduction-mode tracks first: the model uses it as A.",
                "Press <b>Fit g(Pe)</b>, then check R&sup2; and the leave-one-out error against plain E-T depth.",
                "The Results tab then shows the hybrid depth and a regime from R = depth/width (thresholds editable here)."],
         notes="Width stays the E-T width; only the depth changes: depth = R x width. Outside the calibrated Peclet range the model is clamped and you get a warning."),
    dict(why="Fits the absorptivity (and optionally the beam size) to the whole digitised melt pool outline instead of just a width and a depth.",
         steps=["Enter each track's power and speed, then its digitised half-outline: y (across) and z (depth, positive down) in micrometres, origin at the track centreline on the surface.",
                "Strongly recommended: also add a longitudinal section (x ahead of the beam positive, z down). Cross-sections alone do not constrain the scan-direction length.",
                "Press <b>Run profile calibration</b>; read the distance D per track (0 = identical outlines) and the overlay plot.",
                "<b>Apply</b> the fitted values to the Process tab."],
         notes="The simulated outline is the envelope over x of the liquidus isotherm (widest/deepest section along the track), not a single cross-section. No data yet? Press the demo button to see it work."),
    dict(why="Cuts the number of Abaqus runs: (6a) corrects E-T with a few Abaqus results; (6b) finds Goldak parameters that reproduce your measured melt pool.",
         steps=["<b>6a:</b> paste 10-25 Abaqus results (power, speed, width, depth, length) run with ONE fixed Goldak set, then press fit. Read the leave-one-out error against raw E-T.",
                "<b>6b:</b> set ranges for Goldak a, b, c and eta, generate a Latin-hypercube list of runs, run them in Abaqus and paste the widths/depths/lengths back.",
                "Fit the surrogate, check its leave-one-out error, enter your measured target values and press <b>Find parameters</b>.",
                "Run the 2-3 suggested candidates in Abaqus to verify."],
         notes="Goldak semi-axes are 5%-density radii, not melt pool dimensions, and the calibrated values only hold for the FE assumptions you used. Naming of a, b, c differs between papers."),
    dict(why="Lists what the model assumes and where it stops being valid, so you know how far to trust a result.",
         steps=["Read it once before using results in a report.", "Re-read the flags section if the Results tab shows an orange or red regime."],
         notes="The same notes are included in the Excel report."),
    dict(why="Shows where the equations, methods and default values come from and how each source was checked.",
         steps=["Cite the original papers (not this app) in your work.", "Check items marked 'from memory' against the original before citing them."],
         notes="Defaults that are not from a specific paper are labelled as such."),
]


def inject_css() -> None:
    import streamlit as st
    # two selector families: Streamlit >= 1.5x (div[role=tab] in a tablist) and older releases (button[data-baseweb=tab])
    new_t, old_t = "div[role='tablist'] > div[role='tab']", "button[data-baseweb='tab']"
    css = ["<style>", "div[role='tablist'],div[data-baseweb='tab-list']{gap:6px;flex-wrap:wrap;}",
           f"{new_t},{old_t}{{border-radius:10px 10px 0 0;padding:8px 14px;font-weight:700;}}",
           f"{new_t}:hover,{old_t}:hover{{filter:brightness(0.95);}}", "div[data-baseweb='tab-highlight']{height:4px;}"]
    for i, (main, light) in enumerate(TAB_PALETTE, 1):
        n, o = f"{new_t}:nth-child({i})", f"{old_t}:nth-of-type({i})"
        css.append(f"{n},{o}{{background:{light};color:{main};}}")
        css.append(f"{n} p,{o} p{{color:{main};}}")
        css.append(f"{n}[aria-selected='true'],{o}[aria-selected='true']{{background:{main};color:#fff;}}")
        css.append(f"{n}[aria-selected='true'] p,{o}[aria-selected='true'] p{{color:#fff;}}")
    css += [".et-banner{border-radius:10px;padding:10px 18px;color:#fff;font-weight:700;font-size:1.3rem;margin:4px 0 10px 0;}",
            ".et-how{border-left:6px solid;border-radius:8px;padding:10px 16px;margin:2px 0 6px 0;color:#1b1b1b;font-size:0.95rem;}",
            ".et-how ol{margin:4px 0 8px 0;padding-left:22px;}", ".et-sub{border-left:6px solid;padding:4px 12px;margin:14px 0 6px 0;font-weight:700;font-size:1.05rem;border-radius:4px;}", "</style>"]
    st.markdown("".join(css), unsafe_allow_html=True)


def head(i: int, title: str) -> None:
    import streamlit as st
    main, light = TAB_PALETTE[i]; info = TAB_INFO[i]
    st.markdown(f"<div class='et-banner' style='background:{main}'>{title}</div>", unsafe_allow_html=True)
    with st.expander("How to use this tab (why it is needed, steps, good to know)", expanded=True):
        steps = "".join(f"<li>{x}</li>" for x in info["steps"])
        st.markdown(f"<div class='et-how' style='border-color:{main};background:{light}'><b style='color:{main}'>Why you need it</b><br>{info['why']}<br>"
                    f"<b style='color:{main}'>How to use it</b><ol>{steps}</ol><b style='color:{main}'>Good to know</b><br>{info['notes']}</div>", unsafe_allow_html=True)


def subhead(i: int, text: str) -> None:
    import streamlit as st
    main, light = TAB_PALETTE[i]
    st.markdown(f"<div class='et-sub' style='border-color:{main};background:{light};color:{main}'>{text}</div>", unsafe_allow_html=True)


def dl_button(container, label: str, key: str, make, file_name: str, mime: str, **kw):
    """Download button that builds its file only when clicked (nothing is generated or registered on page render,
    and clicking does not rerun the app). Falls back to eager data on older Streamlit versions."""
    try:
        return container.download_button(label, make, file_name=file_name, mime=mime, key=key, on_click="ignore", **kw)
    except Exception:
        return container.download_button(label, make(), file_name=file_name, mime=mime, key=key, **kw)


# ----------------------------------------------------------------------
# 9. Export
# ----------------------------------------------------------------------
def to_xlsx_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as xw:
        for n, d in sheets.items():
            d.to_excel(xw, sheet_name=n[:31], index=False)
    return bio.getvalue()


def fig_png_bytes(fig, dpi: int = 300) -> bytes:
    bio = io.BytesIO(); fig.savefig(bio, format="png", dpi=dpi); return bio.getvalue()


def docx_report(title: str, tables: dict[str, pd.DataFrame], figures: dict, notes: list[str]) -> Optional[bytes]:
    try:
        import docx
    except ImportError:
        return None
    d = docx.Document(); d.add_heading(title, 0)
    for n in notes: d.add_paragraph(n)
    for name, t in tables.items():
        d.add_heading(name, 2)
        tt = d.add_table(rows=1, cols=len(t.columns)); tt.style = "Table Grid"
        for i, c in enumerate(t.columns): tt.rows[0].cells[i].text = str(c)
        for _, r in t.iterrows():
            cells = tt.add_row().cells
            for i, v in enumerate(r): cells[i].text = f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v)
    for name, fg in figures.items():
        d.add_heading(name, 2); d.add_picture(io.BytesIO(fig_png_bytes(fg, 200)), width=docx.shared.Inches(6.2))
    bio = io.BytesIO(); d.save(bio); return bio.getvalue()


def _truthy(x) -> bool:
    """Robust boolean parsing: blank/NaN -> False, 'x'/'yes'/'true'/'1' -> True."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return False
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    return str(x).strip().lower() in {"1", "true", "yes", "y", "x", "excl", "exclude"}


def read_exp_table(file_or_df, colmap: dict, units: dict) -> pd.DataFrame:
    """Return standard columns (power_W, speed_m_s, beam_um, width_um, depth_um, length_um, exclude)."""
    df = file_or_df.copy()
    out = pd.DataFrame()
    out["power_W"] = pd.to_numeric(df[colmap["power"]], errors="coerce") * POWER_UNITS[units["power"]]
    out["speed_m_s"] = pd.to_numeric(df[colmap["speed"]], errors="coerce") * SPEED_UNITS[units["speed"]]
    lu = LENGTH_UNITS[units["length"]] / 1e-6
    for k, col in (("beam", "beam_um"), ("width", "width_um"), ("depth", "depth_um"), ("length", "length_um")):
        out[col] = pd.to_numeric(df[colmap[k]], errors="coerce") * lu if colmap.get(k) not in (None, "", "(none)") else np.nan
    out["exclude"] = (df[colmap["exclude"]].map(_truthy) if colmap.get("exclude") not in (None, "", "(none)") else False)
    return out.dropna(subset=["power_W", "speed_m_s"]).reset_index(drop=True)


# ----------------------------------------------------------------------
# 10. Self-tests   (python et_modeler.py --selftest)
# ----------------------------------------------------------------------
def _t_rosenthal():
    k, rho, cp = 25.0, 7600.0, 700.0; al = k / (rho * cp); P, v, eta = 200.0, 1.0, 0.4
    x, y, z = -200e-6, 50e-6, 30e-6; R = math.sqrt(x * x + y * y + z * z)
    ros = eta * P / (2 * math.pi * k * R) * math.exp(-v * (R + x) / (2 * al))
    mine = float(temperature_rise(x, y, z, eta * P, v, 1e-7, k, al))
    assert abs(mine / ros - 1) < 1e-3, (mine, ros)


def _t_quad():
    from scipy.integrate import quad
    rng = np.random.default_rng(3); worst = 0.0; n = 0; tries = 0
    while n < 320 and tries < 4000:
        tries += 1
        al = 10 ** rng.uniform(np.log10(6e-7), np.log10(5e-5)); k = rng.uniform(8, 200); v = 10 ** rng.uniform(np.log10(0.05), np.log10(3.0))
        sig = rng.uniform(8e-6, 80e-6); q = rng.uniform(50, 400)
        x = -10 ** rng.uniform(-5.5, -2.7); y = rng.uniform(0, 150e-6); z = rng.uniform(0, 150e-6)
        f = lambda s: 2.0 / (sig ** 2 + 2 * al * s * s) * math.exp(-z * z / (4 * al * s * s + 1e-300) - ((x + v * s * s) ** 2 + y * y) / (2 * (sig ** 2 + 2 * al * s * s)))
        ref = q * math.sqrt(al) / (2 * math.pi ** 1.5 * k) * quad(f, 0, np.inf, limit=800, epsabs=0, epsrel=1e-11)[0]
        if ref > 1.0:                      # ignore points whose temperature rise is < 1 K (irrelevant, round-off dominated)
            worst = max(worst, abs(float(temperature_rise(x, y, z, q, v, sig, k, al)) / ref - 1)); n += 1
    assert n >= 300, f"only {n} valid points"
    assert worst < 1e-4, worst


def _t_units():
    p = Process(beam=80e-6, beam_def="D4sigma diameter"); assert abs(p.sigma - 20e-6) < 1e-12
    assert abs(Process(beam=40e-6, beam_def="1/e^2 radius").sigma - 20e-6) < 1e-12
    assert abs(to_si(500, "mm/s", SPEED_UNITS) - 0.5) < 1e-12


def _t_physics():
    m = builtin_materials()["Generic example (NOT a real alloy)"]
    r1 = run_case(Process(P=150), m); r2 = run_case(Process(P=250), m); r3 = run_case(Process(P=250, v=1.5), m)
    assert r2["width"] > r1["width"] and r2["depth"] > r1["depth"]
    assert r3["width"] < r2["width"]
    r0 = run_case(Process(P=1.0), m); assert not r0["melted"]
    tt = temperature_rise(0, 40e-6, 10e-6, 80, 1, 20e-6, 25, 25 / 5.32e6); tm = temperature_rise(0, -40e-6, 10e-6, 80, 1, 20e-6, 25, 25 / 5.32e6)
    assert abs(float(tt) - float(tm)) < 1e-9 * float(tt)
    assert abs(float(temperature_rise(-1e-4, 0, 0, 160, 1, 20e-6, 25, 25 / 5.32e6)) / float(temperature_rise(-1e-4, 0, 0, 80, 1, 20e-6, 25, 25 / 5.32e6)) - 2) < 1e-9


def _t_library():
    try:
        import eagar_tsai  # noqa: F401
    except Exception:
        print("   (eagar-tsai not installed: skipped)"); return
    m = Material(k=25.0, rho=7600.0, cp=700.0, T_liq=1600.0)
    tab, inside = library_crosscheck(Process(P=200, v=1.0, beam=80e-6, eta=0.4), m)   # small domain, 1 um grid
    assert inside, "test case must fit the small domain"
    assert tab["ok"].all(), tab.to_string()


def _t_speed():
    m = builtin_materials()["Generic example (NOT a real alloy)"]; p = Process(beam=80e-6)
    run_case(p, m)
    t = time.time()
    for i in range(10): run_case(replace(p, P=120 + 15 * i, v=0.5 + 0.1 * i), m)
    per = (time.time() - t) / 10; assert per < 0.1, f"{per:.3f} s per case"
    t = time.time(); sweep(p, m, ModelSettings(), np.linspace(100, 300, 9), np.linspace(0.4, 1.6, 9))
    assert time.time() - t < 10, "9x9 sweep too slow"


def _t_flags():
    m = Material(k=25.0, rho=7600.0, cp=700.0, T_liq=1600.0, T_boil=3000.0); st = ModelSettings()
    r = {"melted": True, "width": 100e-6, "depth": 120e-6, "peak_T": 50000.0}
    f = regime_flag(r, {"norm_enthalpy": 1.0}, st, m)
    assert "KEYHOLE" in f and "extreme" in f and "T_boil" in f, f
    r["depth"] = 60e-6; r["peak_T"] = 2000.0
    assert "transition" in regime_flag(r, {"norm_enthalpy": 1.0}, st, m)
    r["depth"] = 20e-6; assert regime_flag(r, {"norm_enthalpy": 1.0}, st, m) == "conduction"


def _t_goldak_sources():
    m = builtin_materials()["Generic example (NOT a real alloy)"]
    assert all(m.sources.get(n) for n in ("k", "rho", "cp", "T_liq")), "every material value needs a source field"
    assert Material(**json.loads(json.dumps(asdict(m)))) == m
    r = run_case(Process(beam=80e-6), m); gs = goldak_start(r)
    t1 = goldak_table(gs, 0.4, 200.0, "goldak"); t2 = goldak_table(gs, 0.4, 200.0, "am")
    assert t1.iloc[0, 0].startswith("a (half") and t2.iloc[2, 0].startswith("b (half")
    assert abs(gs["f_front"] + gs["f_rear"] - 2) < 1e-9


def _t_mcmc():
    try:
        import emcee  # noqa: F401
    except Exception:
        print("   (emcee not installed: skipped)"); return
    m = builtin_materials()["Generic example (NOT a real alloy)"]; base = Process(beam=80e-6)
    rows = []
    for P, v in [(120, 0.6), (160, 0.8), (200, 1.0), (240, 1.0), (140, 1.2)]:
        g = run_case(replace(base, P=P, v=v, eta=0.3), m)
        rows.append(dict(power_W=P, speed_m_s=v, width_um=g["width"] * 1e6, depth_um=g["depth"] * 1e6))
    df = pd.DataFrame(rows); res = calibrate(df, base, m, ModelSettings(), dw_max=None)
    out = run_mcmc(df, base, m, ModelSettings(), res, nwalk=4, nsteps=12)
    t = out["table"].iloc[0]; assert t["p16"] <= t["median"] <= t["p84"] and abs(t["median"] - 0.3) < 0.05


def _t_calibration():
    m = builtin_materials()["Generic example (NOT a real alloy)"]; base = Process(beam=80e-6)
    rng = np.random.default_rng(5); eta_true = 0.33
    rows = []
    for P, v in [(120, 0.6), (160, 0.8), (200, 1.0), (240, 1.0), (180, 0.6), (140, 1.2), (220, 1.4), (260, 1.2), (100, 0.4), (200, 1.6)]:
        r = run_case(replace(base, P=P, v=v, eta=eta_true), m)
        rows.append(dict(power_W=P, speed_m_s=v, width_um=r["width"] * 1e6 * (1 + 0.05 * rng.standard_normal()),
                         depth_um=r["depth"] * 1e6 * (1 + 0.05 * rng.standard_normal()), exclude=False))
    df = pd.DataFrame(rows)
    res = calibrate(df, base, m, ModelSettings(), eta0=0.5, dw_max=None)
    assert abs(res.params["eta"] / eta_true - 1) < 0.06, res.params
    res2 = calibrate(df, base, m, ModelSettings(), eta0=0.5, dw_max=None, fit_beam_scale=True)
    assert res2.corr is not None and "beam_scale" in res2.params


def _t_io():
    raw = pd.DataFrame({"P (kW)": [0.2, 0.3], "v": [800, 1200], "W": [110, 120], "D": [45, 50], "x": [np.nan, "yes"]})
    out = read_exp_table(raw, {"power": "P (kW)", "speed": "v", "width": "W", "depth": "D", "length": "(none)", "beam": "(none)", "exclude": "x"},
                         {"power": "kW", "speed": "mm/s", "length": "um"})
    assert list(out["power_W"]) == [200.0, 300.0] and abs(out["speed_m_s"][1] - 1.2) < 1e-12
    assert list(out["exclude"]) == [False, True], list(out["exclude"])


def _t_cal_options():
    m = builtin_materials()["Generic example (NOT a real alloy)"]; base = Process(beam=80e-6)
    rows = [dict(power_W=P, speed_m_s=v) for P, v in [(120, 0.6), (160, 0.8), (200, 1.0), (240, 1.0), (180, 0.6), (140, 1.2)]]
    df = pd.DataFrame(rows)
    for i, r in df.iterrows():
        g = run_case(replace(base, P=r.power_W, v=r.speed_m_s, eta=0.3), m)
        df.loc[i, "width_um"] = g["width"] * 1e6; df.loc[i, "depth_um"] = g["depth"] * 1e6
    r2 = calibrate(df, base, m, ModelSettings(), fit_beam_scale=True, dw_max=None)
    assert any("correlated" in w or "bound" in w or "singular" in w for w in r2.warnings) or abs(r2.params["eta"] - 0.3) < 0.05
    df2 = df.copy(); df2.loc[0, "depth_um"] = df2.loc[0, "width_um"] * 1.5      # fake keyhole track
    r3 = calibrate(df2, base, m, ModelSettings(), dw_max=1.0)
    assert r3.n_used == len(df2) - 1 and any("excluded" in w for w in r3.warnings)
    tab, (rm, mp, r2_) = cross_validate(df, base, m, ModelSettings(), mode="holdout", dw_max=None)
    assert len(tab) >= 1 and "width" in rm


def _t_tprops():
    df = pd.DataFrame({"T": [300.0, 1300.0, np.nan], "k": [10.0, 20.0, np.nan], "rho": [np.nan] * 3, "cp": [np.nan] * 3,
                       "H": [0.0, 7.0e5, np.nan], "fl": [np.nan] * 3})
    tabs, errs = tables_from_df(df, "K"); assert not errs and set(tabs) == {"k", "H"}
    t2, _ = tables_from_df(pd.DataFrame({"T": [0.0, 100.0], "k": [5.0, 6.0]}), "degC"); assert abs(t2["k"]["T_K"][0] - 273.15) < 1e-9
    _, e1 = tables_from_df(pd.DataFrame({"T": [300.0, 300.0], "k": [5.0, 6.0]})); assert e1                       # duplicate temperature
    _, e2 = tables_from_df(pd.DataFrame({"T": [300.0, 400.0, 500.0], "k": [5.0, np.nan, np.nan]})); assert e2    # only one usable row
    _, e3 = tables_from_df(pd.DataFrame({"T": [-5.0, 100.0], "k": [5.0, 6.0]})); assert e3                       # negative K
    t4, _ = tables_from_df(pd.DataFrame({"T": [300.0, 400.0], "rho": [8.0, 7.9]}), "K", {"rho": 1000.0}); assert t4["rho"]["value"][0] == 8000.0
    eff, notes = effective_props(tabs, 300.0, 1300.0)
    assert abs(eff["k"] - 15.0) < 1e-6 and abs(eff["cp"] - 700.0) < 1e-9, eff          # linear k -> mean 15; cp = dH/dT
    e5, _ = effective_props({"cp": pd.DataFrame({"T_K": [300.0, 1300.0], "value": [600.0, 800.0]})}, 300.0, 1300.0, latent=2.5e5)
    assert abs(e5["cp"] - (700.0 + 250.0)) < 1e-6, e5
    e6, _ = effective_props(tabs, 300.0, 1300.0, mean=False, T_ref=800.0); assert abs(e6["k"] - 15.0) < 1e-9
    fl = pd.DataFrame({"T_K": [1100.0, 1200.0, 1250.0, 1300.0, 1400.0], "value": [0.0, 0.0, 0.5, 1.0, 1.0]})
    assert solidus_liquidus_from_fl(fl) == (1200.0, 1300.0)
    raw = pd.DataFrame({"T [K]": [300.0, 1300.0], "Conductivity": [10.0, 20.0], "Density": [8000.0, 7500.0],
                        "Specific heat": [500.0, 700.0], "liquid fraction (%)": [0.0, 100.0]})
    m = map_prop_columns(raw); assert list(m.columns) == TP_COLS and m["rho"][1] == 7500.0 and np.isnan(m["H"][0])
    try:
        map_prop_columns(pd.DataFrame({"foo": [1, 2]})); raise AssertionError("should fail")
    except ValueError:
        pass
    assert _pick_data_dir().exists()


def _t_solve_and_warnings():
    m = builtin_materials()["Generic example (NOT a real alloy)"]; st_ = ModelSettings(); p = Process(P=220.0, v=1.2, beam=80e-6, eta=0.33)
    w = run_case(p, m, st_)["width"] * 1e6
    sol = solve_eta_for_width(replace(p, eta=0.9), m, st_, w); assert abs(sol["eta"] - 0.33) < 1e-3 and abs(sol["width_um"] - w) < 1e-2, sol
    assert solve_eta_for_width(p, m, st_, 1e5) is None and solve_eta_for_width(p, m, st_, 0.01) is None
    T = {"cp": pd.DataFrame({"T_K": [20.0, 1355.0], "value": [427.0, 692.0]})}
    ws = property_warnings(T, "K", 298.0, 1610.0, 0.0, False); assert len(ws) == 2 and "degC" in ws[0] and "latent" in ws[1].lower()
    assert property_warnings({"cp": T["cp"].assign(T_K=[300.0, 1600.0])}, "K", 298.0, 1610.0, 2.5e5, True) == []
    # energy per volume must be the mean of rho*cp, not the product of means
    rho = pd.DataFrame({"T_K": [300.0, 1300.0], "value": [8000.0, 7000.0]}); cp = pd.DataFrame({"T_K": [300.0, 1300.0], "value": [500.0, 700.0]})
    eff, _ = effective_props({"rho": rho, "cp": cp}, 300.0, 1300.0)
    g = np.linspace(300.0, 1300.0, 400); y = np.interp(g, rho["T_K"], rho["value"]) * np.interp(g, cp["T_K"], cp["value"])
    ref = float(np.sum((y[1:] + y[:-1]) * np.diff(g)) / 2 / 1000.0); assert abs(eff["rho"] * eff["cp"] / ref - 1) < 1e-9


def _t_et_dw_limit():
    rng = np.random.default_rng(11); worst = 0.0
    for _ in range(120):
        m = Material(k=rng.uniform(8, 120), rho=rng.uniform(2500, 9000), cp=rng.uniform(400, 1000), T_liq=rng.uniform(900, 2000))
        p = Process(P=rng.uniform(40, 500), v=10 ** rng.uniform(-1.3, 0.5), beam=rng.uniform(20, 160) * 1e-6, eta=rng.uniform(0.2, 0.8), T0=298.0)
        r = run_case(p, m)
        if r["melted"] and r["width"] > 0:
            worst = max(worst, r["depth"] / r["width"])
    assert worst <= 0.51, worst


def _t_latent_lowers_width():
    tabs = {"k": pd.DataFrame({"T_K": [300.0, 1700.0], "value": [12.0, 28.0]}), "rho": pd.DataFrame({"T_K": [300.0, 1700.0], "value": [8100.0, 7300.0]}),
            "cp": pd.DataFrame({"T_K": [300.0, 1700.0], "value": [430.0, 700.0]})}
    w = []
    for L in (0.0, 2.5e5):
        e, _ = effective_props(tabs, 298.0, 1610.0, latent=L)
        w.append(run_case(Process(P=220.0, v=1.2, beam=80e-6, eta=0.3), Material(k=e["k"], rho=e["rho"], cp=e["cp"], T_liq=1610.0))["width"])
    assert w[1] < w[0] * 0.95, w


def _t_hybrid():
    m = Material(k=21.0, rho=7800.0, cp=700.0, T_liq=1610.0, T_boil=3100.0); base = Process(beam=80e-6, eta=0.35)
    c_true = np.array([0.6, 0.35, -0.03, 0.002]); rows = []
    rng = np.random.default_rng(5)
    for P, v in zip(rng.uniform(120, 400, 14), rng.uniform(0.3, 2.5, 14)):
        S, Pe = _hybrid_S_Pe(replace(base, P=float(P), v=float(v)), m, ModelSettings())
        R = S / float(_poly3(c_true, Pe)); w_et = run_case(replace(base, P=float(P), v=float(v)), m)["width"] * 1e6
        rows.append(dict(power_W=P, speed_m_s=v, width_um=w_et, depth_um=w_et * R))          # measured width = E-T width: depth = R * width exactly
    df = pd.DataFrame(rows); assert hybrid_needed(df) and not hybrid_needed(df.assign(depth_um=df["width_um"] * 0.3))
    fit = hybrid_fit(df, base, m, ModelSettings())
    assert np.allclose(fit.coef, c_true, rtol=1e-6, atol=1e-9), (fit.coef, c_true)       # cubic g(Pe) recovered from exact data
    assert fit.r2_g > 0.999999
    assert fit.metrics.iloc[2]["RMSE [um]"] < 1e-3
    r = hybrid_predict(fit, replace(base, P=200.0, v=3.5), m, ModelSettings(), 100e-6); assert r["clamped"] and r["regime"] in ("conduction", "transition", "keyhole")
    try:
        hybrid_fit(df.iloc[:5], base, m, ModelSettings()); raise AssertionError("needs >= 8 tracks")
    except ValueError:
        pass
    try:
        hybrid_fit(df, base, replace(m, T_boil=None), ModelSettings()); raise AssertionError("needs Tb")
    except ValueError:
        pass


def _t_profile():
    # circle of radius R gives a constant r(phi); identical profiles give D = 0
    R = 50.0; th = np.linspace(0, np.pi / 2, 25)
    r = polar_profile(R * np.cos(th), R * np.sin(th), PHI_CS); assert np.allclose(r, R, rtol=1e-9)
    assert profile_distance(r, r, PHI_CS, R) == 0.0
    assert abs(profile_distance(r, r + 5.0, PHI_CS, R) - 0.1) < 1e-12            # constant 10 % radial error -> D = 0.1
    m = Material(k=21.0, rho=7800.0, cp=750.0, T_liq=1610.0); st_ = ModelSettings(); base = Process(beam=80e-6)
    case = ETCase(replace(base, P=220.0, v=1.0, eta=0.3), m, st_); g = case.melt_pool()
    ph = np.array([0.0, np.pi / 2]); rr = envelope(case, ph, "cs")
    assert abs(rr[0] * 2 / g["width"] - 1) < 0.03 and abs(rr[1] / g["depth"] - 1) < 0.03, (rr, g["width"], g["depth"])      # envelope = widest/deepest section
    assert abs(envelope(case, np.array([0.0, np.pi]), "lg")[0] / g["head"] - 1) < 0.03
    # dual inversion (cross-section + longitudinal) recovers a known eta
    tr = profile_demo(base, m, st_, [(180.0, 0.8), (240.0, 1.2)], eta_true=0.30, noise=0.0)
    out = profile_calibrate(tr, base, m, st_, eta0=0.5); assert abs(out["params"]["eta"] - 0.30) < 0.005, out["params"]
    assert not any("Cross-sections only" in w for w in out["warnings"])
    for t in tr:
        t.x_um = t.zl_um = None
    out2 = profile_calibrate(tr, base, m, st_, eta0=0.5); assert abs(out2["params"]["eta"] - 0.30) < 0.01 and any("Cross-sections only" in w for w in out2["warnings"])
    assert tables_to_tracks(*tracks_to_tables(tr))[1] == []


def _t_fe_and_surrogate():
    m = Material(k=21.0, rho=7800.0, cp=700.0, T_liq=1610.0); base = Process(beam=80e-6, eta=0.3); st_ = ModelSettings(); rng = np.random.default_rng(2)
    rows = []
    for P, v in zip(rng.uniform(150, 350, 14), rng.uniform(0.6, 1.8, 14)):
        r = run_case(replace(base, P=float(P), v=float(v)), m, st_); f = 0.8 + 0.0004 * (P - 250) - 0.05 * (v - 1.2)         # known multiplicative discrepancy
        rows.append(dict(power_W=P, speed_m_s=v, width_um=r["width"] * 1e6 * f, depth_um=r["depth"] * 1e6 * (f + 0.1), length_um=r["length"] * 1e6 * f))
    fe = fe_fit(pd.DataFrame(rows), base, m, st_)
    mt = fe["metrics"].set_index("output")
    assert (mt["corrected_LOO_RMSE_um"] < 0.2 * mt["raw_ET_RMSE_um"]).all(), mt
    P0, v0 = 250.0, 1.2; e = run_case(replace(base, P=P0, v=v0), m, st_)
    c = fe_predict(fe, P0, v0, {o: e[o] * 1e6 for o in FE_OUTS}); assert abs(c["width"] / (e["width"] * 1e6 * 0.8) - 1) < 0.03
    try:
        fe_predict(fe, 900.0, 1.0, {o: 1.0 for o in FE_OUTS}); raise AssertionError("must refuse to extrapolate")
    except ValueError:
        pass
    only = fe_fit(pd.DataFrame(rows)[["power_W", "speed_m_s", "width_um"]], base, m, st_); assert list(only["coef"]) == ["width"]      # one output only
    # DOE generator: bounds, stratification, fixed parameter
    doe = lhs_doe({"a_um": (30, 60), "b_um": (20, 40), "c_um": (50, 90), "eta": (0.5, 0.5)}, 36, seed=3)
    assert doe.shape == (36, 4) and (doe["eta"] == 0.5).all() and doe["a_um"].between(30, 60).all()
    strata = np.floor((doe["a_um"].values - 30) / 30 * 36).astype(int); assert sorted(strata) == list(range(36))        # one point per stratum
    # surrogate LOO sanity + differential_evolution recovers known parameters (eta fixed -> 3 targets, 3 free parameters)
    f = lambda X: np.column_stack([1.3 * X[:, 0] + 0.1 * X[:, 1], 1.1 * X[:, 1] + 0.05 * X[:, 0] + 0.001 * X[:, 1] ** 2, 1.5 * X[:, 2] + 0.2 * X[:, 0]])
    X = doe[["a_um", "b_um", "c_um", "eta"]].values; Y = f(X)
    for meth in ("rbf", "quad"):
        sur = Surrogate(meth).fit(X, Y, ["a_um", "b_um", "c_um", "eta"], ["width_um", "depth_um", "length_um"])
        assert sur.var.tolist() == [True, True, True, False]
        assert (sur.loo()["LOO_MAPE_pct"] < 1.0).all(), sur.loo()
        truth = np.array([[44.0, 31.0, 77.0, 0.5]]); tg = f(truth)[0]
        inv = inverse_de(sur, {"width_um": tg[0], "depth_um": tg[1], "length_um": tg[2]}, seed=4)
        assert all(abs(inv["best"][k] - truth[0, i]) / truth[0, i] < 0.02 for i, k in enumerate(["a_um", "b_um", "c_um"])), inv["best"]
        assert not inv["warnings"] and 1 <= len(inv["table"]) <= 3
    assert inverse_de(sur, {"width_um": 1e4, "depth_um": tg[1]}, seed=1)["warnings"]            # outside the DOE range + non-unique


def _t_process_map():
    sw = pd.DataFrame({"power_W": [100.0, 100.0, 200.0, 200.0], "speed_m_s": [0.5, 1.0, 0.5, 1.0], "width_um": [100.0, 60.0, 160.0, 120.0], "depth_um": [30.0, 15.0, 120.0, 60.0]})
    c = process_map_criteria(sw, hatch_um=100.0, layer_um=40.0, kh_wd=1.5)
    assert c["lof_index"].round(3).tolist() == [round(1 + (40 / 30) ** 2, 3), round((100 / 60) ** 2 + (40 / 15) ** 2, 3), round((100 / 160) ** 2 + 0.1111, 3), round((100 / 120) ** 2 + (40 / 60) ** 2, 3)]
    assert c["lof_ok"].tolist() == [False, False, True, False] and c["keyhole_ok"].tolist() == [True, True, False, True] and c["window_ok"].tolist() == [False, False, False, False]


def _t_workbook():
    from openpyxl import load_workbook
    m = builtin_materials()["Generic example (NOT a real alloy)"]; p = Process(beam=80e-6); stg_ = ModelSettings()
    sw = sweep(p, m, stg_, np.linspace(150, 250, 3), np.linspace(0.6, 1.2, 3))
    cal = dict(material=m.name, params={"eta": 0.33}, std_err={"eta": 0.01}, names=["eta"], n_used=5, rmse={"width": 1.2}, mape={"width": 1.0}, r2={"width": 0.99},
               warnings=["test warning"], table=[{"power_W": 200.0, "speed_m_s": 1.0, "width_um": 100.0, "pred_width_um": 101.0}], cv=None)
    tabs = {"k": pd.DataFrame({"T_K": [300.0, 1600.0], "value": [10.0, 20.0]})}
    data = build_workbook(p, m, stg_, "am", cal=cal, sweep_df=sw, prop_tabs=tabs, prop_eff={"k": 15.0}, prop_notes=["n"])
    wb = load_workbook(io.BytesIO(data))
    assert {"Summary", "Inputs", "Goldak", "Calibration", "Sweep", "Properties", "Notes", "References"} <= set(wb.sheetnames), wb.sheetnames
    ws = wb["Summary"]; rows = {ws.cell(r, 1).value: r for r in range(1, 40) if ws.cell(r, 1).value}
    r = run_case(p, m, stg_); rw = rows["Melt pool width"]
    assert isinstance(ws.cell(rw, 2).value, float) and abs(ws.cell(rw, 2).value - r["width"] * 1e6) < 1e-6
    assert ws.cell(rw, 4).value and len(ws.cell(rw, 4).value) > 30                  # plain-language meaning present
    assert wb["Goldak"].cell(5, 1).value.startswith("a_f") and wb["Goldak"].max_row >= 11
    assert wb["Sweep"].max_row == 4 + 9
    nomelt = build_workbook(replace(p, P=1.0), m, stg_); assert "NO MELTING" in str(load_workbook(io.BytesIO(nomelt))["Summary"]["A4"].value)


TESTS = [("Rosenthal limit", _t_rosenthal), ("quadrature vs scipy.quad", _t_quad), ("units & beam definitions", _t_units),
         ("physical trends & symmetry", _t_physics), ("cross-check vs eagar-tsai library", _t_library),
         ("speed targets (<0.1 s/case, 9x9 sweep <10 s)", _t_speed), ("regime / peak-T / boiling flags", _t_flags),
         ("material sources, Goldak naming", _t_goldak_sources), ("T-dependent property tables", _t_tprops), ("solve eta from width, table warnings, rho*cp mean", _t_solve_and_warnings), ("E-T depth/width <= 0.51 over a wide sweep", _t_et_dw_limit), ("latent switch lowers width", _t_latent_lowers_width),
         ("hybrid depth: cubic g(Pe) recovery, LOO, clamping", _t_hybrid), ("profile metric, envelope, dual inversion", _t_profile),
         ("FE correction, DOE, surrogate, differential evolution", _t_fe_and_surrogate), ("process-map criteria", _t_process_map),
         ("single formatted Excel workbook", _t_workbook), ("optional emcee posterior", _t_mcmc),
         ("synthetic calibration recovery", _t_calibration),
         ("file reading, units, blank exclude cells", _t_io), ("calibration options (corr., exclusion, hold-out)", _t_cal_options)]


def run_selftests() -> int:
    fails = 0
    for name, fn in TESTS:
        t = time.time()
        try:
            fn(); print(f"PASS  {name}  ({time.time() - t:.1f}s)")
        except Exception as ex:
            fails += 1; print(f"FAIL  {name}: {type(ex).__name__}: {ex}")
    return fails




def run_uitests() -> int:
    """Headless UI tests with streamlit.testing.v1.AppTest (python et_modeler.py --uitest)."""
    from streamlit.testing.v1 import AppTest
    path = str(Path(__file__).resolve())

    def new():
        return AppTest.from_file(path, default_timeout=240).run()

    def width(at):
        return float([m for m in at.metric if m.label.startswith("Width")][0].value)

    def t_default():
        at = new(); assert not at.exception, at.exception
        assert len(at.tabs) == 13 and [t.label for t in at.tabs][0].endswith("Process")
        assert width(at) > 0

    def t_invalid():
        at = new(); at.number_input(key="P").set_value(0.0).run()
        assert not at.exception and len(at.sidebar.error) >= 1
        at = new(); at.number_input(key="m_Tl").set_value(100.0).run()          # liquidus below T0
        assert not at.exception and len(at.sidebar.error) >= 1
        at = new(); at.number_input(key="m_rho").set_value(0.0).run(); assert not at.exception

    def t_nomelt():
        at = new(); at.number_input(key="P").set_value(1.0).run()
        assert not at.exception and any("No melting" in e.value for e in at.error)

    def t_units():
        a = width(new())
        at = new(); at.selectbox(key="P_unit").set_value("kW").run(); at.number_input(key="P").set_value(0.2).run()
        at.selectbox(key="v_unit").set_value("m/s").run(); at.number_input(key="v").set_value(1.0).run()
        assert not at.exception and abs(width(at) / a - 1) < 1e-9, (a, width(at))

    def t_session():
        at = new(); at.session_state["_pending"] = {"inputs": {"P": 123.0, "eta": 0.5}}; at.run()
        assert not at.exception and at.session_state["P"] == 123.0 and at.session_state["eta"] == 0.5

    def t_calibration():
        at = new(); at.button(key="btn_demo").click().run(); assert not at.exception
        at.radio(key="c_vmode").set_value("random hold-out (30%)").run()
        at.button(key="btn_cal").click().run(); assert not at.exception, at.exception
        cal = at.session_state["cal"]; assert abs(cal["params"]["eta"] - 0.33) < 0.06, cal["params"]
        assert cal["cv"] is not None
        at.button(key="btn_apply").click().run(); assert not at.exception
        assert abs(at.session_state["eta"] - cal["params"]["eta"]) < 1e-3
        try:
            import emcee  # noqa: F401
            at.number_input(key="mc_walk").set_value(4).run(); at.number_input(key="mc_steps").set_value(20).run()
            at.button(key="btn_mcmc").click().run(); assert not at.exception, at.exception
            assert at.session_state["mcmc"]["table"].iloc[0]["parameter"] == "eta"
        except ImportError:
            pass

    def t_library_button():
        try:
            import eagar_tsai  # noqa: F401
        except Exception:
            return
        at = new(); at.number_input(key="beam").set_value(80.0).run(); at.button(key="btn_lib").click().run()
        assert not at.exception, at.exception
        tab, inside = at.session_state["_lib"]; assert inside and tab["ok"].all(), tab

    def t_sweep():
        at = new(); at.number_input(key="sw_pn").set_value(3).run(); at.number_input(key="sw_vn").set_value(3).run()
        at.button(key="btn_sweep").click().run(); assert not at.exception and len(at.session_state["sweep"]) == 9

    def t_tprops_ui():
        at = new()
        df = pd.DataFrame(np.nan, index=range(8), columns=TP_COLS); df.loc[0, ["T", "k"]] = [300.0, 10.0]; df.loc[1, ["T", "k"]] = [1700.0, 30.0]
        at.session_state["tp_df"] = df; at.run(); assert not at.exception, at.exception
        exp, _ = effective_props(tables_from_df(df)[0], at.session_state["T0"], at.session_state["m_Tl"])
        at.button(key="btn_tp_apply").click().run(); assert not at.exception, at.exception
        assert abs(at.session_state["m_k"] - exp["k"]) < 1e-6 and "T-dependent" in at.session_state["m_s_k"]
        bad = df.copy(); bad.loc[0, "cp"] = 500.0                                    # a single cp row -> error shown, no crash
        at.session_state["tp_df"] = bad; at.run(); assert not at.exception and len(at.error) >= 1

    def t_solve_eta_ui():
        at = new(); at.number_input(key="ov_w").set_value(60.0).run(); assert not at.exception, at.exception
        assert any("reproduces your measured width" in i.value for i in at.info)
        at.button(key="btn_eta_solve").click().run(); assert not at.exception
        assert abs(width(at) - 60.0) < 0.3, width(at)

    def t_hybrid_ui():
        at = new(); assert any("Not needed yet" in i.value for i in at.info)
        rows = [dict(power_W=120 + 25 * i, speed_m_s=0.5 + 0.12 * i, beam_um=np.nan, width_um=100.0 + 3 * i, depth_um=40.0 + 9 * i, length_um=np.nan, exclude=False) for i in range(10)]
        at.session_state["exp_df"] = pd.DataFrame(rows); at.session_state["m_Tb"] = 3100.0; at.run(); assert not at.exception, at.exception
        at.button(key="btn_hy_fit").click().run(); assert not at.exception, at.exception
        assert at.session_state["hy"] is not None and at.session_state["hy"].n == 10
        assert any("Hybrid depth (calibrated" in i.value for i in at.info)                      # shown in the Results tab

    def t_profile_ui():
        at = new(); at.button(key="btn_pc_demo").click().run(); assert not at.exception, at.exception
        assert len(at.session_state["pc_tr"]) == 3 and len(at.session_state["pc_lg"]) > 0
        at.number_input(key="pc_eta0").set_value(0.5).run(); at.button(key="btn_pc_run").click().run(); assert not at.exception, at.exception
        r = at.session_state["pc_res"]["res"]; assert abs(r["params"]["eta"] - 0.30) < 0.02, r["params"]
        at.button(key="btn_pc_apply").click().run(); assert not at.exception and abs(at.session_state["eta"] - r["params"]["eta"]) < 1e-3

    def t_fe_ui():
        at = new()
        fe = pd.DataFrame([dict(power_W=150 + 20 * i, speed_m_s=0.6 + 0.1 * i, width_um=70 + 4 * i, depth_um=25 + 2 * i, length_um=250 + 10 * i) for i in range(10)])
        at.session_state["fe_df"] = fe; at.run(); at.button(key="btn_fe_fit").click().run(); assert not at.exception, at.exception
        assert at.session_state["fe"] is not None and set(at.session_state["fe"]["coef"]) == {"width", "depth", "length"}
        at.number_input(key="P").set_value(900.0).run(); assert not at.exception and any("outside the Abaqus DOE range" in w.value for w in at.warning)   # no extrapolation

    def t_goldak_surrogate_ui():
        at = new(); at.number_input(key="gk_e1").set_value(0.2).run()                         # eta fixed (low = high)
        at.button(key="btn_gk_doe").click().run(); assert not at.exception, at.exception
        d = at.session_state["gk_tab"]; assert len(d) == 36 and (d["eta"] == 0.2).all()
        d["width_um"] = 1.3 * d["a_um"] + 0.1 * d["b_um"]; d["depth_um"] = 1.1 * d["b_um"] + 0.05 * d["a_um"]; d["length_um"] = 1.5 * d["c_um"] + 0.2 * d["a_um"]
        at.session_state["gk_tab"] = d; at.run(); at.button(key="btn_gk_fit").click().run(); assert not at.exception, at.exception
        at.number_input(key="gk_tw").set_value(1.3 * 70 + 0.1 * 40).run(); at.number_input(key="gk_td").set_value(1.1 * 40 + 0.05 * 70).run(); at.number_input(key="gk_tl").set_value(1.5 * 120 + 0.2 * 70).run()
        at.button(key="btn_gk_inv").click().run(); assert not at.exception, at.exception
        best = at.session_state["gk_inv"]["best"]; assert abs(best["a_um"] - 70) < 2 and abs(best["b_um"] - 40) < 2 and abs(best["c_um"] - 120) < 3, best

    def t_window_ui():
        at = new(); at.number_input(key="sw_pn").set_value(3).run(); at.number_input(key="sw_vn").set_value(3).run(); at.button(key="btn_sweep").click().run()
        at.checkbox(key="sw_ovl").set_value(True).run(); assert not at.exception, at.exception
        assert "window_ok" in at.session_state["_sw_crit"] and len(at.session_state["_sw_crit"]) == 9

    def t_paste_exp():
        at = new(); at.text_area(key="exp_paste").set_value("Power (W)\tSpeed (mm/s)\tWidth (um)\tDepth (um)\n200\t1000\t110\t45\n250\t1000\t130\t60").run()
        assert not at.exception, at.exception
        assert at.session_state["c_p"] == "Power (W)" and at.session_state["c_v"] == "Speed (mm/s)" and at.session_state["c_d"] == "Depth (um)"
        at.button(key="btn_load").click().run(); assert not at.exception and len(at.session_state["exp_df"]) == 2

    def t_goldak_convention():
        at = new(); at.selectbox(key="goldak_conv").set_value(list(GOLDAK_CONVENTIONS)[1]).run(); assert not at.exception

    tests = [("default run, 13 tabs", t_default), ("invalid inputs", t_invalid), ("no-melt case", t_nomelt), ("unit switch kW & m/s", t_units),
             ("session restore (pending dict)", t_session), ("calibration flow + hold-out + apply", t_calibration),
             ("sweep", t_sweep), ("library cross-check button", t_library_button), ("T-dependent tables apply", t_tprops_ui), ("solve absorptivity from a measured width", t_solve_eta_ui), ("hybrid depth tab (gated, fit, Results line)", t_hybrid_ui), ("profile calibration tab (demo, fit, apply)", t_profile_ui),
             ("FE correction tab (fit, no extrapolation)", t_fe_ui), ("Goldak surrogate tab (DOE, fit, inverse)", t_goldak_surrogate_ui),
             ("process window overlay", t_window_ui), ("paste experimental table", t_paste_exp), ("Goldak convention", t_goldak_convention)]
    fails = 0
    for n, fn in tests:
        t = time.time()
        try:
            fn(); print(f"PASS  UI: {n}  ({time.time() - t:.1f}s)")
        except Exception as ex:
            fails += 1; print(f"FAIL  UI: {n}: {type(ex).__name__}: {ex}")
    return fails


# ----------------------------------------------------------------------
# 11. Streamlit UI
# ----------------------------------------------------------------------
INPUT_KEYS = ["P", "P_unit", "v", "v_unit", "beam", "beam_unit", "beam_def", "eta", "T0", "layer", "hatch",
              "beam_scale", "mat_choice", "m_name", "m_k", "m_rho", "m_cp", "m_Tl", "m_Ts", "m_Tb", "m_Lh", "m_src",
              "s_npan", "s_nx", "s_dwt", "s_dwk", "s_enth", "s_use_enth", "k_scale",
              "m_s_k", "m_s_rho", "m_s_cp", "m_s_Tl", "m_s_Ts", "m_s_Tb", "m_s_Lh", "ov_w", "ov_d",
              "c_fit_bs", "c_fit_k", "c_ww", "c_wd", "c_wl", "c_use_dw", "c_dwcut", "c_eta0", "c_vmode", "mc_walk", "mc_steps",
              "sw_p0", "sw_p1", "sw_pn", "sw_v0", "sw_v1", "sw_vn", "sw_var", "sens_on", "goldak_conv", "u_p", "u_v", "u_l",
              "tp_Tunit", "tp_method", "tp_Tref", "tp_inclat", "tp_s_k", "tp_s_rho", "tp_s_cp", "tp_s_H",
              "hy_cond", "hy_key", "pc_fit_bs", "pc_eta0", "fe_ridge", "gk_a0", "gk_a1", "gk_b0", "gk_b1", "gk_c0", "gk_c1", "gk_e0", "gk_e1", "gk_n", "gk_seed",
              "gk_ratio", "gk_method", "gk_tw", "gk_td", "gk_tl", "gk_ww", "gk_wd", "gk_wl", "sw_ovl", "sw_Lmode", "sw_L", "sw_kh", "sw_dsrc"]


def _profiles_path() -> Path:
    return DATA_DIR / "profiles.json"


def _load_profiles() -> dict:
    f = _profiles_path()
    try:
        return json.loads(f.read_text()) if f.exists() else {}
    except Exception:
        return {}


def _save_profile(name: str, prof: dict) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    cur = _load_profiles(); cur[name] = prof
    _profiles_path().write_text(json.dumps(cur, indent=2, default=float))


def _synthetic_example(mat: Material, base: Process, st: ModelSettings) -> pd.DataFrame:
    rng = np.random.default_rng(11); rows = []
    for P, v in [(120, 0.6), (160, 0.8), (200, 1.0), (240, 1.0), (180, 0.6), (140, 1.2), (220, 1.4), (260, 1.2)]:
        r = run_case(replace(base, P=P, v=v, eta=0.33), mat, st)
        rows.append(dict(power_W=float(P), speed_m_s=float(v), beam_um=np.nan,
                         width_um=round(r["width"] * 1e6 * (1 + 0.04 * rng.standard_normal()), 1),
                         depth_um=round(r["depth"] * 1e6 * (1 + 0.04 * rng.standard_normal()), 1),
                         length_um=np.nan, exclude=False))
    return pd.DataFrame(rows, columns=EXP_COLS)


def main() -> None:
    import streamlit as st
    ss = st.session_state
    st.set_page_config(page_title="E-T Melt Pool Modeler", layout="wide")
    try:
        import pyarrow  # noqa: F401   (needed by st.dataframe / st.data_editor)
    except Exception as ex:
        st.error(f"Your Python environment is inconsistent: {ex}")
        st.markdown("Typical cause: NumPy 1.x installed next to a pyarrow built for NumPy 2. Run this in the SAME terminal you start Streamlit from, "
                    "then restart the app:")
        st.code("python -m pip install --upgrade numpy pandas scipy matplotlib pyarrow streamlit openpyxl", language="bash")
        st.stop()

    inject_css()
    rpt = {}                                   # filled at the end of the run; read when a report download is clicked
    make_report = lambda: build_workbook(**rpt)
    # ---- defaults live in session_state (no value= on keyed widgets) ----
    _DEF = {
        "P": 200.0,
        "v": 1000.0,
        "beam": 80.0,
        "eta": 0.40,
        "T0": 298.0,
        "layer": 40.0,
        "hatch": 100.0,
        "beam_scale": 1.0,
        "m_name": "Custom",
        "m_k": 25.0,
        "m_rho": 7600.0,
        "m_cp": 700.0,
        "m_Tl": 1600.0,
        "m_Ts": 0.0,
        "m_Tb": 0.0,
        "m_Lh": 0.0,
        "m_src": "user input - VERIFY",
        "s_npan": 48,
        "s_nx": 17,
        "s_dwt": 0.5,
        "s_dwk": 1.0,
        "s_enth": 6.0,
        "s_use_enth": False,
        "k_scale": 1.0,
        "P_unit": "W",
        "v_unit": "mm/s",
        "beam_unit": "um",
        "beam_def": "D4sigma diameter",
        "mat_choice": "Custom",
        "m_s_k": "user input - VERIFY", "m_s_rho": "user input - VERIFY", "m_s_cp": "user input - VERIFY",
        "m_s_Tl": "user input - VERIFY", "m_s_Ts": "", "m_s_Tb": "", "m_s_Lh": "",
        "ov_w": 0.0, "ov_d": 0.0, "u_p": "W", "u_v": "mm/s", "u_l": "um",
        "c_fit_bs": False, "c_fit_k": False, "c_ww": 1.0, "c_wd": 1.0, "c_wl": 0.0, "c_use_dw": True, "c_dwcut": 1.0,
        "c_eta0": 0.4, "c_vmode": "none", "mc_walk": 8, "mc_steps": 80, "prof_name": "calibration",
        "sw_p0": 100.0, "sw_p1": 300.0, "sw_pn": 9, "sw_v0": 0.4, "sw_v1": 1.6, "sw_vn": 9, "sw_var": "depth_um",
        "sens_on": False, "goldak_conv": list(GOLDAK_CONVENTIONS)[0], "exp_paste": "",
        "hy_cond": HYBRID_THRESH[0], "hy_key": HYBRID_THRESH[1], "pc_fit_bs": False, "pc_eta0": 0.4, "fe_ridge": 1e-3,
        "gk_a0": 40.0, "gk_a1": 100.0, "gk_b0": 20.0, "gk_b1": 60.0, "gk_c0": 60.0, "gk_c1": 200.0, "gk_e0": 0.2, "gk_e1": 0.5, "gk_n": 36, "gk_seed": 0,
        "gk_ratio": 0.5, "gk_method": "rbf", "gk_tw": 0.0, "gk_td": 0.0, "gk_tl": 0.0, "gk_ww": 1.0, "gk_wd": 1.0, "gk_wl": 1.0,
        "sw_ovl": False, "sw_Lmode": "Process-tab layer thickness", "sw_L": 40.0, "sw_kh": 1.5, "sw_dsrc": "E-T depth",
        "tp_Tunit": "K", "tp_method": "mean over T0 ... T_liquidus", "tp_Tref": 1000.0, "tp_inclat": False,
        "tp_s_k": 1.0, "tp_s_rho": 1.0, "tp_s_cp": 1.0, "tp_s_H": 1.0,
    }
    for _k, _v in _DEF.items():
        ss.setdefault(_k, _v)

    # ---- apply a pending session restore BEFORE any widget exists -------
    if "_pending" in ss:
        pend = ss.pop("_pending")
        for k, v in pend.get("inputs", {}).items():
            ss[k] = v
        if "exp" in pend:
            ss["exp_df"] = pd.DataFrame(pend["exp"], columns=EXP_COLS); ss["_exp_gen"] = ss.get("_exp_gen", 0) + 1
    if "pc_tr" not in ss:
        ss["pc_tr"] = pd.DataFrame({"track": pd.Series(dtype=str), "power_W": pd.Series(dtype=float), "speed_m_s": pd.Series(dtype=float)})
        ss["pc_cs"] = pd.DataFrame({"track": pd.Series(dtype=str), "y_um": pd.Series(dtype=float), "z_um": pd.Series(dtype=float)})
        ss["pc_lg"] = pd.DataFrame({"track": pd.Series(dtype=str), "x_um": pd.Series(dtype=float), "z_um": pd.Series(dtype=float)})
    if "fe_df" not in ss:
        ss["fe_df"] = pd.DataFrame(np.nan, index=range(12), columns=["power_W", "speed_m_s", "width_um", "depth_um", "length_um"])
    if "tp_df" not in ss:
        ss["tp_df"] = pd.DataFrame(np.nan, index=range(10), columns=TP_COLS)
    if "exp_df" not in ss:
        ss["exp_df"] = pd.DataFrame({c: pd.Series(dtype=float) for c in EXP_COLS[:-1]}).assign(exclude=pd.Series(dtype=bool))

    st.title("E-T Melt Pool Modeler")
    st.caption("Eagar-Tsai moving Gaussian source - conduction-mode melt pool size for any material, with calibration to your own measurements. "
               "Constant properties, no latent heat, no convection: see the Assumptions tab.")

    tabs = st.tabs(["1 Process", "2 Material", "3 Model settings", "4 Experimental data", "5 Results",
                    "6 Calibration", "7 Sweep / maps", "8 Goldak export", "9 Hybrid depth", "10 Profile calibration", "11 FE correction / Goldak surrogate", "12 Assumptions", "13 References"])

    # ================= 1 Process =================
    with tabs[0]:
        head(0, "Process inputs")
        c1, c2, c3 = st.columns(3)
        with c1:
            a, b = st.columns([3, 1]); a.number_input("Laser power", min_value=0.0, step=10.0, key="P"); b.selectbox("unit", list(POWER_UNITS), key="P_unit")
            a, b = st.columns([3, 1]); a.number_input("Scan speed", min_value=0.0, step=50.0, key="v"); b.selectbox("unit ", list(SPEED_UNITS), key="v_unit")
        with c2:
            a, b = st.columns([3, 1]); a.number_input("Beam size", min_value=0.0, step=5.0, key="beam"); b.selectbox("unit  ", ["um", "mm"], key="beam_unit")
            st.selectbox("Beam size definition", list(BEAM_DEFS), key="beam_def",
                         help="sigma = D/4 for D4sigma or 1/e^2 diameter; sigma = w/2 for 1/e^2 radius")
        with c3:
            st.number_input("Absorptivity eta (effective)", min_value=0.01, max_value=1.0, step=0.01, key="eta")
            st.number_input("Substrate / preheat temperature T0 [K]", min_value=1.0, step=1.0, key="T0")
        with st.expander("Energy-density inputs (only used for energy metrics) and calibration scale factors"):
            d1, d2, d3 = st.columns(3)
            d1.number_input("Layer thickness [um]", min_value=0.0, key="layer")
            d2.number_input("Hatch distance [um]", min_value=0.0, key="hatch")
            d3.number_input("Beam-size scale factor (from calibration)", min_value=0.1, max_value=5.0, step=0.01, key="beam_scale")

    # ================= 2 Material =================
    with tabs[1]:
        head(1, "Material properties (SI units, K)")
        st.warning("No real-alloy values are built in. Enter values for YOUR alloy and note the source of EACH value. Constant properties are used: "
                   "pick values representative of the range between T0 and the liquidus.")
        mats = {**builtin_materials(), **load_user_materials()}

        def _load_mat():
            if ss.mat_choice in mats:
                m = mats[ss.mat_choice]
                ss.m_name, ss.m_k, ss.m_rho, ss.m_cp, ss.m_Tl = m.name, m.k, m.rho, m.cp, m.T_liq
                ss.m_Ts, ss.m_Tb, ss.m_Lh, ss.m_src = (m.T_sol or 0.0), (m.T_boil or 0.0), m.latent_heat, m.source
                for key, n in (("m_s_k", "k"), ("m_s_rho", "rho"), ("m_s_cp", "cp"), ("m_s_Tl", "T_liq"),
                               ("m_s_Ts", "T_sol"), ("m_s_Tb", "T_boil"), ("m_s_Lh", "latent_heat")):
                    ss[key] = m.sources.get(n, m.source)

        st.selectbox("Preset / saved material", ["Custom"] + list(mats), key="mat_choice", on_change=_load_mat)
        st.text_input("Name", key="m_name")
        hdr = st.columns([2, 3])
        hdr[0].markdown("**Value**"); hdr[1].markdown("**Source of this value** (paper, handbook, DSC/CALPHAD run, ...)")
        for lab, key, skey, step in (("Thermal conductivity k [W/m/K]", "m_k", "m_s_k", 1.0), ("Density rho [kg/m3]", "m_rho", "m_s_rho", 50.0),
                                     ("Specific heat cp [J/kg/K]", "m_cp", "m_s_cp", 10.0), ("Liquidus temperature [K]", "m_Tl", "m_s_Tl", 10.0),
                                     ("Solidus temperature [K] (0 = not set)", "m_Ts", "m_s_Ts", 10.0),
                                     ("Boiling temperature [K] (0 = not set)", "m_Tb", "m_s_Tb", 50.0),
                                     ("Latent heat [J/kg] (enthalpy metric only)", "m_Lh", "m_s_Lh", 1000.0)):
            c1, c2 = st.columns([2, 3])
            c1.number_input(lab, min_value=0.0, step=step, key=key)
            c2.text_input("source", key=skey, label_visibility="collapsed", placeholder="source (required)")
        st.text_input("General note on the material source", key="m_src")
        missing = [lab for lab, key, skey in (("k", "m_k", "m_s_k"), ("rho", "m_rho", "m_s_rho"), ("cp", "m_cp", "m_s_cp"), ("T_liquidus", "m_Tl", "m_s_Tl"),
                                             ("T_solidus", "m_Ts", "m_s_Ts"), ("T_boil", "m_Tb", "m_s_Tb"), ("latent heat", "m_Lh", "m_s_Lh"))
                   if ss[key] and (not str(ss[skey]).strip() or "VERIFY" in str(ss[skey]))]
        if missing:
            st.warning("Source missing or still marked VERIFY for: " + ", ".join(missing))
        st.caption(f"Thermal diffusivity alpha = {ss.m_k / (ss.m_rho * ss.m_cp) * 1e6:.3f} mm2/s" if ss.m_rho > 0 and ss.m_cp > 0 else "")
        if st.button("Save this material for future sessions", key="btn_save_mat"):
            save_user_material(Material(name=ss.m_name, k=ss.m_k, rho=ss.m_rho, cp=ss.m_cp, T_liq=ss.m_Tl,
                                        T_sol=ss.m_Ts or None, T_boil=ss.m_Tb or None, latent_heat=ss.m_Lh, source=ss.m_src,
                                        sources={n: ss[k] for n, k in (("k", "m_s_k"), ("rho", "m_s_rho"), ("cp", "m_s_cp"), ("T_liq", "m_s_Tl"),
                                                                         ("T_sol", "m_s_Ts"), ("T_boil", "m_s_Tb"), ("latent_heat", "m_s_Lh"))}))
            st.success(f"Saved '{ss.m_name}' in {DATA_DIR}/materials.json")

        subhead(1, "Temperature-dependent properties (optional table, Abaqus-style)")
        with st.container(border=True):
            st.markdown("One row per temperature; leave a cell empty (it shows as None) if you do not have that property. "
                        "**Tip: copy a block of cells in Excel and paste it into the table with Ctrl+V.** The app averages the table between T0 and the liquidus "
                        "and fills the constants above (the model itself needs constants).")
            fu = st.file_uploader("...or load a table from CSV / XLSX (header row; JMatPro-style column names are recognised)", type=["csv", "xlsx"], key="tp_file")
            if fu is not None and ss.get("_tp_fid") != fu.file_id:
                try:
                    raw_ = pd.read_csv(fu, sep=None, engine="python") if fu.name.lower().endswith(".csv") else pd.read_excel(fu)
                    ss["tp_df"] = map_prop_columns(raw_); ss["_tp_fid"] = fu.file_id; ss["_tp_gen"] = ss.get("_tp_gen", 0) + 1; st.rerun()
                except Exception as ex:
                    st.error(f"Could not read the table: {ex}")
            c1, c2, c3, c4 = st.columns([1, 2, 1, 1])
            c1.selectbox("Temperature unit", ["K", "degC"], key="tp_Tunit")
            c2.selectbox("Averaging", ["mean over T0 ... T_liquidus", "value at reference temperature"], key="tp_method")
            c3.number_input("Reference T [K]", min_value=1.0, key="tp_Tref", help="Only used for 'value at reference temperature'")
            c4.markdown("&nbsp;")
            if c4.button("Clear table", key="btn_tp_clear"):
                ss["tp_df"] = pd.DataFrame(np.nan, index=range(10), columns=TP_COLS); ss["_tp_gen"] = ss.get("_tp_gen", 0) + 1; st.rerun()
            edited = st.data_editor(ss["tp_df"], num_rows="dynamic", width="stretch", key=f"tp_editor_{ss.get('_tp_gen', 0)}",
                                    column_config={"T": st.column_config.NumberColumn("Temperature", format="%.1f"),
                                                   "k": st.column_config.NumberColumn("Conductivity k [W/m/K]"),
                                                   "rho": st.column_config.NumberColumn("Density [kg/m3]"),
                                                   "cp": st.column_config.NumberColumn("Specific heat [J/kg/K]"),
                                                   "H": st.column_config.NumberColumn("Enthalpy [J/kg] (optional)"),
                                                   "fl": st.column_config.NumberColumn("Liquid fraction 0-1 (optional)")})
            with st.expander("Unit conversion factors (only if your numbers are not in the units shown in the column headers)"):
                q1, q2, q3, q4 = st.columns(4)
                q1.number_input("k  x  to W/m/K", min_value=0.0, format="%g", key="tp_s_k")
                q2.number_input("density  x  to kg/m3 (g/cm3: 1000)", min_value=0.0, format="%g", key="tp_s_rho")
                q3.number_input("cp  x  to J/kg/K (kJ: 1000)", min_value=0.0, format="%g", key="tp_s_cp")
                q4.number_input("enthalpy  x  to J/kg", min_value=0.0, format="%g", key="tp_s_H")
            st.checkbox("Add the latent-heat field above to cp as L/(T_liq - T0)  (only if the cp column is NOT an apparent cp and no enthalpy column is given)", key="tp_inclat")
            st.caption("Latent heat is a single number (J/kg), not a curve. Alternatively give an enthalpy column H(T) that already contains it (cp = change in H over T0...T_liq). "
                       "A liquid-fraction column lets the app read the solidus and liquidus for you.")
            tabs_, perrs = tables_from_df(edited, ss.tp_Tunit, {"k": ss.tp_s_k, "rho": ss.tp_s_rho, "cp": ss.tp_s_cp, "H": ss.tp_s_H})
            for e_ in perrs:
                st.error(e_)
            ss["_tp_eff"], ss["_tp_tabs"], ss["_tp_notes"] = {}, tabs_, []
            if not tabs_ and not perrs:
                st.info("Table is empty: the constant values entered above are used as they are.")
            for w_ in property_warnings(tabs_, ss.tp_Tunit, ss.T0, ss.m_Tl, ss.m_Lh, ss.tp_inclat):
                st.warning(w_)
            if tabs_:
                try:
                    eff, enotes = effective_props(tabs_, ss.T0, ss.m_Tl, ss.tp_method.startswith("mean"), ss.tp_Tref, ss.m_Lh if ss.tp_inclat else 0.0)
                except ValueError as ex:
                    eff, enotes = {}, []; st.error(str(ex))
                ss["_tp_eff"], ss["_tp_notes"] = eff, enotes
                if eff:
                    unit = {"k": "W/m/K", "rho": "kg/m3", "cp": "J/kg/K"}
                    st.dataframe(pd.DataFrame([(p, unit[p], v) for p, v in eff.items()], columns=["property", "unit", "effective constant"]), hide_index=True, width="stretch")
                for n_ in enotes:
                    st.caption(n_)
                import matplotlib.pyplot as plt
                fp_ = fig_props(tabs_, eff, ss.T0, ss.m_Tl); st.pyplot(fp_)
                dl_button(st, "Property curves PNG (300 dpi)", "dl_props", lambda f=fp_: fig_png_bytes(f), "et_properties.png", "image/png"); plt.close(fp_)

                def _apply_tp():
                    ef = ss.get("_tp_eff") or {}
                    note = ("T-dependent table, " + ("mean over T0-T_liq" if ss.tp_method.startswith("mean") else f"value at {ss.tp_Tref:.0f} K")
                            + f" (T0={ss.T0:.0f} K, T_liq={ss.m_Tl:.0f} K)")
                    for p, key, skey in (("k", "m_k", "m_s_k"), ("rho", "m_rho", "m_s_rho"), ("cp", "m_cp", "m_s_cp")):
                        if p in ef:
                            ss[key] = float(ef[p]); ss[skey] = note
                st.button("Apply effective values to the constant-property fields above", key="btn_tp_apply", on_click=_apply_tp, disabled=not eff, type="primary")
                if "fl" in tabs_:
                    try:
                        Ts_, Tl_ = solidus_liquidus_from_fl(tabs_["fl"])
                        st.caption(f"From the liquid-fraction column (fl <= 0.001 / >= 0.999): solidus {Ts_:.1f} K, liquidus {Tl_:.1f} K.")

                        def _apply_sl(Ts_=Ts_, Tl_=Tl_):
                            ss["m_Ts"], ss["m_Tl"] = Ts_, Tl_
                            ss["m_s_Ts"] = ss["m_s_Tl"] = "from liquid-fraction column (fl <= 0.001 / >= 0.999)"
                        st.button("Apply solidus / liquidus from this column", key="btn_tp_sl", on_click=_apply_sl)
                    except ValueError as ex:
                        st.warning(str(ex))

    # ================= 3 Model settings =================
    with tabs[2]:
        head(2, "Numerical and regime settings")
        s1, s2 = st.columns(2)
        s1.number_input("Quadrature panels", min_value=16, max_value=400, step=8, key="s_npan",
                        help="48 is accurate to ~1e-6 relative; raise to 96+ for extreme parameters")
        s1.number_input("x samples per width/depth search pass", min_value=9, max_value=81, step=2, key="s_nx")
        s1.number_input("Conductivity scale factor (from calibration)", min_value=0.2, max_value=5.0, step=0.01, key="k_scale")
        s2.number_input("Depth/width: 'transition' threshold", min_value=0.1, max_value=3.0, step=0.05, key="s_dwt",
                        help="Editable heuristic, not a universal law")
        s2.number_input("Depth/width: 'keyhole-suspect' threshold", min_value=0.1, max_value=5.0, step=0.05, key="s_dwk")
        s2.checkbox("Also flag by normalized enthalpy", key="s_use_enth")
        s2.number_input("Normalized-enthalpy threshold", min_value=0.1, step=0.5, key="s_enth",
                        help="Verify the definition and threshold in King et al. 2014 before relying on it")

    # ================= build objects =================
    proc = Process(P=ss.P * POWER_UNITS[ss.P_unit], v=ss.v * SPEED_UNITS[ss.v_unit], beam=ss.beam * LENGTH_UNITS[ss.beam_unit],
                   beam_def=ss.beam_def, eta=ss.eta, T0=ss.T0, layer=ss.layer * 1e-6, hatch=ss.hatch * 1e-6, beam_scale=ss.beam_scale)
    mat = Material(name=ss.m_name, k=ss.m_k, rho=ss.m_rho, cp=ss.m_cp, T_liq=ss.m_Tl, T_sol=ss.m_Ts or None,
                   T_boil=ss.m_Tb or None, latent_heat=ss.m_Lh, source=ss.m_src,
                   sources={n: ss[k] for n, k in (("k", "m_s_k"), ("rho", "m_s_rho"), ("cp", "m_s_cp"), ("T_liq", "m_s_Tl"),
                                                  ("T_sol", "m_s_Ts"), ("T_boil", "m_s_Tb"), ("latent_heat", "m_s_Lh"))})
    stg = ModelSettings(npan=int(ss.s_npan), nx_scan=int(ss.s_nx), keyhole_dw_transition=ss.s_dwt, keyhole_dw_keyhole=ss.s_dwk,
                        keyhole_norm_enthalpy=ss.s_enth, use_enthalpy_flag=ss.s_use_enth, k_scale=ss.k_scale)
    errs = proc.validate() + mat.validate(proc.T0)

    with tabs[2]:
        st.markdown("---")
        st.markdown("**Optional cross-check against the eagar-tsai library**")
        import importlib.util
        if importlib.util.find_spec("eagar_tsai") is None:
            st.caption("The optional `eagar-tsai` package is not installed (pip install eagar-tsai); the built-in engine is self-contained.")
        else:
            st.caption("Runs the library on a SMALL domain (480 x 75 x 65 um, 1 um grid; its default domain is far too slow). The library fixes T0 = 298 K "
                       "and takes beam_diameter = 2*sigma; tolerance max(2.5 um, 3%). Only meaningful for pools that fit in the domain.")
            if st.button("Run library cross-check", key="btn_lib", disabled=bool(errs)):
                with st.spinner("Running eagar-tsai ..."):
                    try:
                        ss["_lib"] = library_crosscheck(proc, mat, stg)
                    except Exception as ex:
                        ss["_lib"] = None; st.error(f"Library run failed: {ex}")
            if ss.get("_lib"):
                lt, inside = ss["_lib"]
                st.dataframe(lt.round(2), hide_index=True, width="stretch")
                if not inside:
                    st.warning("This pool does not fit the small domain (or does not melt): library values are clipped and not comparable.")
                elif lt["ok"].all():
                    st.success("Agreement within tolerance.")
                else:
                    st.error("Outside tolerance.")

    # ---- sidebar: session save / restore ----
    with st.sidebar:
        st.header("Session")
        payload = {"inputs": {k: ss[k] for k in INPUT_KEYS if k in ss}, "exp": ss.get("_exp_current", ss["exp_df"]).to_dict("records")}
        dl_button(st, "Save session (JSON)", "dl_session", lambda: json.dumps(payload, default=float), "et_session.json", "application/json")
        up = st.file_uploader("Restore session", type=["json"], key="sess_file")
        if up is not None and ss.get("_last_sess") != up.file_id:
            try:
                ss["_pending"] = json.loads(up.getvalue().decode()); ss["_last_sess"] = up.file_id; st.rerun()
            except Exception as ex:
                st.error(f"Could not read session file: {ex}")
        if errs:
            st.error("Input problems:\n- " + "\n- ".join(errs))
        st.header("Export")
        export_slot = st.container()

    # ================= 4 Experimental data =================
    with tabs[3]:
        head(3, "Measured melt pool data (single tracks)")
        st.caption("Blank cells in the exclude column mean NOT excluded. Tracks flagged here are never used for fitting or validation.")
        with st.expander("Expected file format (any column names work - you map them below)"):
            st.markdown("""
- **One row per single track**, first row = header, CSV (comma, semicolon or tab) or XLSX (first sheet).
- **Required columns:** laser **power** and scan **speed**. Give **width** and/or **depth** (length optional). Optional: **beam size** per track (overrides the Process tab) and an **exclude** flag (`x`, `yes`, `true`, `1` = exclude; blank = keep).
- Units can be anything listed in the unit boxes below (power W/kW, speed mm/s or m/s, lengths um/mm/m) - one unit per quantity for the whole file, so don't mix units in a column.
- Header names are only used to pre-select the columns; e.g. `Power (W)`, `Speed (mm/s)`, `Width (um)`, `Depth (um)`, `Exclude`.
""")
            dl_button(st, "Download blank CSV template", "dl_exp_tpl", lambda: ",".join(EXP_COLS) + "\n", "et_exp_template.csv", "text/csv")
        f = st.file_uploader("Upload CSV / XLSX", type=["csv", "xlsx"], key="exp_file")
        st.text_area("...or paste a table (copy cells from Excel, header row first)", key="exp_paste", height=110)
        raw, src_id = None, None
        try:
            if f is not None:
                raw = pd.read_csv(f, sep=None, engine="python") if f.name.lower().endswith(".csv") else pd.read_excel(f)
                src_id = ("file", f.file_id)
            elif ss.exp_paste.strip():
                raw = pd.read_csv(io.StringIO(ss.exp_paste), sep=None, engine="python"); src_id = ("paste", hash(ss.exp_paste))
        except Exception as ex:
            st.error(f"Could not read the table: {ex}")
        if raw is not None:
            try:
                raw.columns = raw.columns.astype(str)
                st.dataframe(raw.head(8), width="stretch")
                cols = ["(none)"] + list(raw.columns)

                def guess(*names):
                    for n in names:
                        for c in raw.columns:
                            if n in c.lower():
                                return c
                    return "(none)"
                if ss.get("_exp_fid") != src_id:        # new file: re-guess columns (set before the widgets exist)
                    for key, nm in (("c_p", ("power", "p_")), ("c_v", ("speed", "vel", "v_")), ("c_w", ("width",)), ("c_d", ("depth",)),
                                    ("c_l", ("length",)), ("c_b", ("beam", "spot")), ("c_x", ("exclude",))):
                        ss[key] = guess(*nm)
                    ss["_exp_fid"] = src_id
                u1, u2, u3 = st.columns(3)
                units = {"power": u1.selectbox("Power unit", list(POWER_UNITS), key="u_p"),
                         "speed": u2.selectbox("Speed unit", list(SPEED_UNITS), key="u_v"),
                         "length": u3.selectbox("Length unit (beam/width/depth/length)", list(LENGTH_UNITS), key="u_l")}
                g = st.columns(4)
                cm = {"power": g[0].selectbox("Power column", cols, key="c_p"), "speed": g[1].selectbox("Speed column", cols, key="c_v"),
                      "width": g[2].selectbox("Width column", cols, key="c_w"), "depth": g[3].selectbox("Depth column", cols, key="c_d")}
                g = st.columns(3)
                cm["length"] = g[0].selectbox("Length column", cols, key="c_l")
                cm["beam"] = g[1].selectbox("Beam size column", cols, key="c_b")
                cm["exclude"] = g[2].selectbox("Exclude flag column", cols, key="c_x")
                if st.button("Load into table", key="btn_load"):
                    if "(none)" in (cm["power"], cm["speed"]):
                        st.error("Select power and speed columns.")
                    else:
                        ss["exp_df"] = read_exp_table(raw, cm, units); ss["_exp_gen"] = ss.get("_exp_gen", 0) + 1; st.rerun()
            except Exception as ex:
                st.error(f"Could not read file: {ex}")
        b1, b2 = st.columns(2)
        if b1.button("Load synthetic demo data (eta=0.33 + 4% noise)", key="btn_demo"):
            ss["exp_df"] = _synthetic_example(mat, proc, stg); ss["_exp_gen"] = ss.get("_exp_gen", 0) + 1; st.rerun()
        if b2.button("Clear table", key="btn_clear"):
            ss["exp_df"] = pd.DataFrame({c: pd.Series(dtype=float) for c in EXP_COLS[:-1]}).assign(exclude=pd.Series(dtype=bool)); ss["_exp_gen"] = ss.get("_exp_gen", 0) + 1; st.rerun()
        exp_df = st.data_editor(ss["exp_df"], num_rows="dynamic", width="stretch", key=f"exp_editor_{ss.get('_exp_gen', 0)}",
                                column_config={"power_W": st.column_config.NumberColumn("power [W]"),
                                               "speed_m_s": st.column_config.NumberColumn("speed [m/s]", format="%.3f"),
                                               "beam_um": st.column_config.NumberColumn("beam size [um] (optional)"),
                                               "width_um": st.column_config.NumberColumn("width [um]"),
                                               "depth_um": st.column_config.NumberColumn("depth [um]"),
                                               "length_um": st.column_config.NumberColumn("length [um] (optional)"),
                                               "exclude": st.column_config.CheckboxColumn("exclude")})
        ss["_exp_current"] = exp_df
        st.markdown("**Single-track overlay for the Results tab (optional)**")
        o1, o2 = st.columns(2)
        o1.number_input("Measured width [um] (0 = none)", min_value=0.0, key="ov_w")
        o2.number_input("Measured depth [um] (0 = none)", min_value=0.0, key="ov_d")
        ov_w, ov_d = ss.ov_w, ss.ov_d

    cal = ss.get("cal")
    # ================= 5 Results =================
    with tabs[4]:
        head(4, "Single-case results")
        if errs:
            st.error("Fix the input problems listed in the sidebar.")
        else:
            res = run_case(proc, mat, stg)
            if cal:
                if cal["material"] != mat.name:
                    st.warning(f"Active calibration was made for '{cal['material']}', current material is '{mat.name}'.")
                if not (cal["P_range"][0] * 0.95 <= proc.P <= cal["P_range"][1] * 1.05 and cal["v_range"][0] * 0.95 <= proc.v <= cal["v_range"][1] * 1.05):
                    st.warning("Power/speed are outside the range used for calibration: extrapolation.")
            if not res["melted"]:
                st.error(f"No melting: peak temperature {res['peak_T']:.0f} K is below the liquidus {mat.T_liq:.0f} K.")
            else:
                (st.error if ("KEYHOLE" in res["regime"] or "extreme" in res["regime"]) else st.warning if "transition" in res["regime"] else st.success)(f"Regime: {res['regime']}")
                st.caption("E-T underpredicts keyhole depth; the depth/width and enthalpy flags are editable heuristics; "
                           "a calibration made on one alloy does not transfer to another.")
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Width [um]", f"{res['width'] * 1e6:.1f}"); k2.metric("Depth [um]", f"{res['depth'] * 1e6:.1f}")
                k3.metric("Length [um]", f"{res['length'] * 1e6:.1f}"); k4.metric("Depth / width", f"{res['dw_ratio']:.2f}")
                rdf = results_table(res)
                st.table(rdf.set_index("quantity"))
                if ss.get("hy") is not None and mat.T_boil:
                    hp = hybrid_predict(ss["hy"], proc, mat, stg, res["width"], (ss.hy_cond, ss.hy_key))
                    st.info(f"**Hybrid depth (calibrated in tab 9): {hp['depth'] * 1e6:.1f} um**, R = depth/width = {hp['R']:.2f}, regime: {hp['regime']} "
                            f"(plain E-T depth {res['depth'] * 1e6:.1f} um). Pe = {hp['Pe']:.2f}.")
                    if hp["clamped"]:
                        st.warning(f"Pe = {hp['Pe']:.2f} is outside the calibrated range {ss['hy'].pe_min:.2f}-{ss['hy'].pe_max:.2f}; Pe was clamped, so treat this depth with caution.")
                if ov_w:
                    sol = solve_eta_for_width(proc, mat, stg, ov_w)
                    if sol is None:
                        st.warning(f"No absorptivity between 0.01 and 1 reproduces the measured width of {ov_w:g} um with the current material, speed and beam.")
                    else:
                        st.info(f"**Absorptivity that reproduces your measured width of {ov_w:g} um: eta = {sol['eta']:.3f}** "
                                f"(predicted depth {sol['depth_um']:.1f} um, depth/width {sol['dw_ratio']:.2f}). This is an EFFECTIVE fitting value for this one track: it absorbs every "
                                "model, material and heat-source difference, so check it on other power/speed combinations (Calibration tab).")

                        def _use_eta(e_=sol["eta"]):
                            ss["eta"] = float(min(1.0, max(0.01, round(e_, 4))))
                        st.button("Use this absorptivity in the Process tab", key="btn_eta_solve", on_click=_use_eta)
                if mat.T_boil and res["peak_T"] > mat.T_boil:
                    st.warning("Peak temperature exceeds the boiling temperature: E-T has no evaporation physics here.")
                import matplotlib.pyplot as plt
                fig1 = fig_sections(proc, mat, stg, res, ov_w or None, ov_d or None); st.pyplot(fig1)
                fig2 = fig_heatmap(proc, mat, stg, res); st.pyplot(fig2)
                sens_on = st.checkbox("Compute sensitivity (+/-10% of each input; ~16 extra runs)", key="sens_on")
                figs = {"Sections": fig1, "Surface temperature": fig2}
                if sens_on:
                    sens = sensitivity(proc, mat, stg); st.dataframe(sens.round(2), width="stretch", hide_index=True)
                    fig3 = fig_sensitivity(sens); st.pyplot(fig3); figs["Sensitivity"] = fig3
                    dl_button(st, "Sensitivity PNG (300 dpi)", "dl_sens", lambda f=fig3: fig_png_bytes(f), "et_sensitivity.png", "image/png")
                d1, d2, d3, d4 = st.columns(4)
                dl_button(d1, "Sections PNG (300 dpi)", "dl_sections", lambda f=fig1: fig_png_bytes(f), "et_sections.png", "image/png")
                dl_button(d2, "Heatmap PNG (300 dpi)", "dl_heatmap", lambda f=fig2: fig_png_bytes(f), "et_heatmap.png", "image/png")
                dl_button(d3, "Full Excel report (all tabs)", "dl_xlsx_res", make_report, "et_report.xlsx", XLSX_MIME, type="primary")
                if d4.button("Prepare DOCX report", key="btn_docx"):
                    ss["_docx"] = docx_report("E-T Melt Pool report", {"Results": rdf.drop(columns=["what it means"]), "Inputs": inputs_table(proc, mat, stg)},
                        figs, ["Eagar-Tsai conduction model; constant properties; no latent heat/convection.", f"Material source: {mat.source}"])
                if ss.get("_docx"):
                    dl_button(st, "Download DOCX", "dl_docx", lambda b=ss["_docx"]: b, "et_report.docx", DOCX_MIME)
                elif "_docx" in ss:
                    st.info("python-docx not installed: DOCX export unavailable.")
                for fg in figs.values(): plt.close(fg)

    # ================= 6 Calibration =================
    with tabs[5]:
        head(5, "Calibrate against your measurements")
        usable = exp_df.dropna(subset=["power_W", "speed_m_s"]) if len(exp_df) else exp_df
        st.caption(f"{len(usable)} rows in the experimental table. Needs at least 2 usable tracks; many more tracks than fitted parameters is essential. "
                   "Residuals are relative errors (pred - meas)/meas.")
        c1, c2, c3 = st.columns(3)
        c1.checkbox("Also fit beam-size scale factor", key="c_fit_bs", help="Adds flexibility; keep it only if the std. errors stay small and residuals improve meaningfully")
        c1.checkbox("Also fit conductivity scale factor", key="c_fit_k")
        c2.number_input("Weight: width", 0.0, 10.0, key="c_ww"); c2.number_input("Weight: depth", 0.0, 10.0, key="c_wd"); c2.number_input("Weight: length", 0.0, 10.0, key="c_wl")
        c3.checkbox("Exclude tracks with measured depth/width above:", key="c_use_dw")
        c3.number_input("depth/width cut-off (editable heuristic)", 0.1, 5.0, step=0.05, key="c_dwcut")
        c3.number_input("Initial absorptivity", 0.05, 1.0, step=0.05, key="c_eta0")
        st.radio("Validation", ["none", "random hold-out (30%)", "leave-one-out"], horizontal=True, key="c_vmode")
        if st.button("Run calibration", type="primary", key="btn_cal", disabled=bool(errs)):
            try:
                kw = dict(fit_beam_scale=ss.c_fit_bs, fit_k_scale=ss.c_fit_k, weights={"width": ss.c_ww, "depth": ss.c_wd, "length": ss.c_wl},
                          eta0=ss.c_eta0, dw_max=ss.c_dwcut if ss.c_use_dw else None)
                with st.spinner("Fitting..."):
                    r = calibrate(exp_df, proc, mat, stg, **kw)
                cv = None
                if ss.c_vmode != "none":
                    bar = st.progress(0.0, text="Validating...")
                    keep, _ = _filter_tracks(exp_df, None)
                    cvt, cvm = cross_validate(keep, proc, mat, stg, mode="loo" if ss.c_vmode == "leave-one-out" else "holdout",
                                              progress=lambda i, n: bar.progress((i + 1) / n, text=f"Validation fold {i + 1}/{n}"), **kw)
                    bar.empty(); cv = (cvt, cvm)
                dd = r.table
                ss["cal"] = dict(material=mat.name, params=r.params, std_err=r.std_err, names=r.names, corr=None if r.corr is None else r.corr.tolist(),
                                 P_range=[float(dd["power_W"].min()), float(dd["power_W"].max())], v_range=[float(dd["speed_m_s"].min()), float(dd["speed_m_s"].max())],
                                 n_used=r.n_used, rmse=r.rmse, mape=r.mape, r2=r.r2, warnings=r.warnings, kw=kw,
                                 table=r.table.to_dict("records"), cv=None if cv is None else dict(table=cv[0].to_dict("records"), rmse=cv[1][0], mape=cv[1][1], r2=cv[1][2]))
                ss.pop("mcmc", None)
                st.rerun()
            except Exception as ex:
                st.error(f"Calibration failed: {ex}")
        cal = ss.get("cal")
        if cal:
            import matplotlib.pyplot as plt
            st.markdown(f"**Active calibration** - material '{cal['material']}', {cal['n_used']} tracks, P {cal['P_range'][0]:.0f}-{cal['P_range'][1]:.0f} W, v {cal['v_range'][0]:.2f}-{cal['v_range'][1]:.2f} m/s")
            for w_ in cal["warnings"]: st.warning(w_)
            pt = pd.DataFrame({"parameter": cal["names"], "unit": ["-"] * len(cal["names"]), "value": [cal["params"][n] for n in cal["names"]],
                               "std. error": [cal["std_err"][n] for n in cal["names"]]})
            st.dataframe(pt.round(4), hide_index=True, width="stretch")
            if cal.get("corr") is not None and len(cal["names"]) > 1:
                st.markdown("Parameter correlation (from the Jacobian)")
                st.dataframe(pd.DataFrame(cal["corr"], index=cal["names"], columns=cal["names"]).round(3), width="stretch")
            st.caption("The calibrated absorptivity is an EFFECTIVE fitting parameter for this model, machine and powder - NOT a measured physical absorptivity. "
                       "It absorbs powder-bed, property, beam and model errors, and it does not transfer between alloys.")
            mt = pd.DataFrame({"metric": ["RMSE [um]", "MAPE [%]", "R2"], **{c: [cal["rmse"].get(c), cal["mape"].get(c), cal["r2"].get(c)] for c in ("width", "depth", "length") if c in cal["rmse"]}})
            st.markdown("Training metrics"); st.dataframe(mt.round(3), hide_index=True)
            tab = pd.DataFrame(cal["table"]); st.dataframe(tab.round(2), width="stretch", hide_index=True)
            fg = fig_parity(tab, "Train"); st.pyplot(fg); fr = fig_residuals(tab); st.pyplot(fr)
            if cal.get("cv"):
                st.markdown("**Validation on tracks not used in each fit**")
                cvt = pd.DataFrame(cal["cv"]["table"])
                cvm = pd.DataFrame({"metric": ["RMSE [um]", "MAPE [%]", "R2"], **{c: [cal["cv"]["rmse"].get(c), cal["cv"]["mape"].get(c), cal["cv"]["r2"].get(c)] for c in ("width", "depth") if c in cal["cv"]["rmse"]}})
                st.dataframe(cvm.round(3), hide_index=True)
                fc = fig_parity(cvt, "Validation"); st.pyplot(fc); plt.close(fc)

            def _apply():
                ss["eta"] = float(min(1.0, max(0.01, round(cal["params"]["eta"], 4))))
                ss["beam_scale"] = float(round(cal["params"].get("beam_scale", 1.0), 4)); ss["k_scale"] = float(round(cal["params"].get("k_scale", 1.0), 4))
            a1, a2, a3 = st.columns(3)
            a1.button("Apply calibrated values to Process/Settings tabs", on_click=_apply, key="btn_apply")
            a2.text_input("Profile name", key="prof_name")
            if a3.button("Save calibration profile", key="btn_save_prof"):
                _save_profile(ss.prof_name, cal); st.success("Saved.")
            d1, d2, d3 = st.columns(3)
            dl_button(d1, "Parity PNG (300 dpi)", "dl_parity", lambda f=fg: fig_png_bytes(f), "et_parity.png", "image/png")
            dl_button(d2, "Residuals PNG (300 dpi)", "dl_resid", lambda f=fr: fig_png_bytes(f), "et_residuals.png", "image/png")
            dl_button(d3, "Full Excel report (all tabs)", "dl_xlsx_cal", make_report, "et_report.xlsx", XLSX_MIME, type="primary")
            plt.close(fg); plt.close(fr)

            # ---- optional emcee posterior ----
            import importlib.util
            if importlib.util.find_spec("emcee") is not None and cal.get("kw"):
                with st.expander("Optional: Bayesian posterior with emcee"):
                    st.caption("Flat priors inside the fit bounds, Gaussian relative-error likelihood with sigma from the fit residuals. "
                               "Reflects measurement scatter only, not model discrepancy: the intervals are optimistic. Each step costs ~0.3 s per track.")
                    m1, m2 = st.columns(2)
                    m1.number_input("Walkers", 4, 64, step=2, key="mc_walk"); m2.number_input("Steps", 20, 2000, step=20, key="mc_steps")
                    if st.button("Run MCMC", key="btn_mcmc", disabled=bool(errs)):
                        bar = st.progress(0.0, text="Sampling...")
                        try:
                            r0 = calibrate(exp_df, proc, mat, stg, **cal["kw"])
                            ss["mcmc"] = run_mcmc(exp_df, proc, mat, stg, r0, int(ss.mc_walk), int(ss.mc_steps),
                                                  progress=lambda i, n: bar.progress((i + 1) / n, text=f"Step {i + 1}/{n}"))
                        except Exception as ex:
                            st.error(f"MCMC failed: {ex}")
                        bar.empty()
                    mc = ss.get("mcmc")
                    if mc:
                        st.dataframe(mc["table"].round(4), hide_index=True, width="stretch")
                        st.caption(f"acceptance {mc['accept']:.2f}; sigma_rel {mc['sigma_rel']:.3f}; max autocorrelation time {mc['tau']:.1f} steps; {mc['nwalk']} walkers x {mc['nsteps']} steps.")
                        if not np.isfinite(mc["tau"]) or mc["nsteps"] < 50 * mc["tau"]:
                            st.warning("Chain is shorter than ~50 autocorrelation times: treat the intervals as indicative only; run more steps.")
                        fp = fig_posterior(mc["chain"], cal["names"]); st.pyplot(fp)
                        dl_button(st, "Posterior PNG (300 dpi)", "dl_post", lambda f=fp: fig_png_bytes(f), "et_posterior.png", "image/png"); plt.close(fp)
        profs = _load_profiles()
        if profs:
            sel = st.selectbox("Load a saved profile", ["-"] + list(profs), key="prof_sel")
            if sel != "-" and st.button("Load selected profile", key="btn_loadprof"):
                ss["cal"] = profs[sel]; st.rerun()

    # ================= 7 Sweep =================
    with tabs[6]:
        head(6, "Power-speed sweep")
        st.caption("Uses the current Process (beam, absorptivity, T0), Material and Model settings; power in W, speed in m/s here.")
        w1, w2, w3, w4 = st.columns(4)
        w1.number_input("Power min [W]", 1.0, key="sw_p0"); w1.number_input("Power max [W]", 1.0, key="sw_p1")
        w2.number_input("Power steps", 2, 40, key="sw_pn")
        w3.number_input("Speed min [m/s]", 0.01, key="sw_v0"); w3.number_input("Speed max [m/s]", 0.01, key="sw_v1")
        w4.number_input("Speed steps", 2, 40, key="sw_vn")
        if st.button("Run sweep", key="btn_sweep", disabled=bool(errs)):
            with st.spinner(f"Running {int(ss.sw_pn * ss.sw_vn)} cases..."):
                ss["sweep"] = sweep(proc, mat, stg, np.linspace(ss.sw_p0, ss.sw_p1, int(ss.sw_pn)), np.linspace(ss.sw_v0, ss.sw_v1, int(ss.sw_vn)))
        if "sweep" in ss:
            sw = ss["sweep"]; st.selectbox("Map variable", ["depth_um", "width_um", "length_um", "dw_ratio"], key="sw_var")
            import matplotlib.pyplot as plt
            fm = fig_map(sw, ss.sw_var, stg.keyhole_dw_keyhole); st.pyplot(fm)
            st.dataframe(sw.round(3), width="stretch", hide_index=True)
            e1, e2 = st.columns(2)
            dl_button(e1, "Map PNG (300 dpi)", "dl_map", lambda f=fm: fig_png_bytes(f), "et_map.png", "image/png")
            dl_button(e2, "Full Excel report (all tabs)", "dl_xlsx_sw", make_report, "et_report.xlsx", XLSX_MIME, type="primary")
            plt.close(fm)
            subhead(6, "Process-map overlays (optional): lack of fusion and keyhole limit")
            st.caption("Lack of fusion when (H/w)^2 + (L/d)^2 > 1 (H = hatch spacing from the Process tab, L = layer thickness, w and d = melt pool width and depth). "
                       "Keyhole risk when w/d is below the limit. E-T alone can never give w/d below about 2, so choose the hybrid depth (tab 9) to see keyhole risk.")
            o1, o2, o3, o4 = st.columns(4)
            o1.checkbox("Show process window", key="sw_ovl")
            o2.radio("Layer thickness L used", ["Process-tab layer thickness", "custom value"], key="sw_Lmode",
                     help="Which layer value applies? Use the powder-layer thickness the pool has to penetrate; choose custom if you re-melt consolidated layers.")
            o3.number_input("custom L [um]", 1.0, key="sw_L"); o4.number_input("keyhole limit: w/d >", 0.5, 5.0, step=0.1, key="sw_kh")
            st.radio("Depth used", ["E-T depth", "hybrid depth (tab 9)"], horizontal=True, key="sw_dsrc")
            if ss.sw_ovl:
                dcol = "depth_um"; sw2 = sw.copy()
                if ss.sw_dsrc.startswith("hybrid"):
                    if ss.get("hy") is None or not mat.T_boil:
                        st.warning("Fit the hybrid depth model in tab 9 first (and enter the boiling temperature); showing E-T depth meanwhile.")
                    else:
                        sw2["depth_hybrid_um"] = [hybrid_predict(ss["hy"], replace(proc, P=float(r_.power_W), v=float(r_.speed_m_s)), mat, stg, r_.width_um * 1e-6, (ss.hy_cond, ss.hy_key))["depth"] * 1e6
                                                  for r_ in sw.itertuples()]
                        dcol = "depth_hybrid_um"
                L_um = proc.layer * 1e6 if ss.sw_Lmode.startswith("Process") else ss.sw_L
                sc = process_map_criteria(sw2, proc.hatch * 1e6, L_um, ss.sw_kh, dcol); ss["_sw_crit"] = sc
                fw = fig_window(sc, ss.sw_kh); st.pyplot(fw)
                st.caption(f"H = {proc.hatch * 1e6:g} um, L = {L_um:g} um; {100 * sc['window_ok'].mean():.0f}% of the grid is inside the window.")
                dl_button(st, "Process window PNG (300 dpi)", "dl_win", lambda f=fw: fig_png_bytes(f), "et_process_window.png", "image/png"); plt.close(fw)
                st.dataframe(sc.round(3), width="stretch", hide_index=True)

    # ================= 8 Goldak =================
    with tabs[7]:
        head(7, "Goldak double-ellipsoid starting values (heuristic)")
        st.info("Starting guesses derived from the E-T melt pool geometry. They are NOT calibrated Goldak parameters: refine them in Abaqus against your OM sections. "
                "Goldak semi-axes are 5%-density radii (exp(-3 x^2/a^2)), not melt pool dimensions: in a published example the fitted transverse axis was 102.7 um for a measured "
                "half-width of about 63 um, so expect the calibrated axes to be LARGER than the values below. Tab 11 shows how to calibrate them with few Abaqus runs.")
        if not errs:
            r = run_case(proc, mat, stg)
            if r["melted"]:
                gs = goldak_start(r)
                st.selectbox("Naming convention for the export", list(GOLDAK_CONVENTIONS), key="goldak_conv")
                gdf = goldak_table(gs, proc.eta, proc.P, GOLDAK_CONVENTIONS[ss.goldak_conv])
                st.table(gdf.set_index("parameter"))
                st.caption("Q = eta*P uses the effective (calibrated) absorptivity. Check each paper's a/b/c naming before transferring numbers. "
                           "The Goldak values are also in the Excel report.")
                dl_button(st, "Full Excel report (all tabs)", "dl_xlsx_gk", make_report, "et_report.xlsx", XLSX_MIME, type="primary")
            else:
                st.warning("No melting for the current inputs.")

    # ================= 9 Hybrid depth =================
    with tabs[8]:
        head(8, "Hybrid depth model (deep / keyhole-like tracks)")
        h1, h2 = st.columns(2)
        h1.number_input("Regime: conduction below R = d/w =", 0.1, 3.0, step=0.05, key="hy_cond")
        h2.number_input("Regime: keyhole above R = d/w =", 0.1, 3.0, step=0.05, key="hy_key")
        if not hybrid_needed(exp_df):
            st.info("Not needed yet: none of your measured tracks has depth/width above 0.5, which is the most that E-T can produce. Load tracks with width and depth in tab 4: "
                    "this tab unlocks by itself as soon as one track has depth/width above 0.5.")
        else:
            st.markdown("R = d/w = A P / (2 a k (T_b - T0) g(Pe)),  a = 1/e^2 beam radius (= 2 sigma),  Pe = v a / D,  D = k/(rho cp),  "
                        "g(Pe) = c0 + c1 Pe + c2 Pe^2 + c3 Pe^3.  The g coefficients are not published, so they are fitted here by linear least squares from "
                        "g_meas = A P / (2 a k (T_b - T0) R_meas).  Width = E-T width; depth = R x width.  A is the absorptivity of the Process tab (calibrate it on conduction tracks first).")
            if not mat.T_boil:
                st.error("Enter the boiling temperature in the Material tab: it is part of the model.")
            if st.button("Fit g(Pe)", key="btn_hy_fit", type="primary", disabled=bool(errs) or not mat.T_boil):
                try:
                    with st.spinner("Fitting..."):
                        ss["hy"] = hybrid_fit(exp_df, proc, mat, stg)
                    st.rerun()                      # the Results tab (rendered earlier in the script) must pick up the new fit
                except ValueError as ex:
                    ss["hy"] = None; st.error(str(ex))
            fit = ss.get("hy")
            if fit is not None:
                import matplotlib.pyplot as plt
                for w_ in fit.warnings:
                    st.warning(w_)
                st.dataframe(pd.DataFrame({"coefficient": ["c0", "c1", "c2", "c3"], "value": fit.coef}), hide_index=True, width="stretch")
                st.caption(f"{fit.n} tracks; R2 of g(Pe) = {fit.r2_g:.4f}; calibrated Pe range {fit.pe_min:.3g} - {fit.pe_max:.3g} (outside it Pe is clamped and you get a warning).")
                st.markdown("**Depth error: hybrid (leave-one-out) versus plain E-T**"); st.dataframe(fit.metrics.round(2), hide_index=True, width="stretch")
                fh = fig_hybrid(fit); st.pyplot(fh)
                dl_button(st, "Hybrid parity PNG (300 dpi)", "dl_hy", lambda f=fh: fig_png_bytes(f), "et_hybrid.png", "image/png"); plt.close(fh)
                if not errs and mat.T_boil:
                    r_ = run_case(proc, mat, stg); hp = hybrid_predict(fit, proc, mat, stg, r_["width"], (ss.hy_cond, ss.hy_key))
                    st.success(f"Current Process: hybrid depth {hp['depth'] * 1e6:.1f} um (plain E-T {r_['depth'] * 1e6:.1f} um), R = {hp['R']:.2f}, regime: {hp['regime']}, Pe = {hp['Pe']:.2f}.")
                    if hp["clamped"]:
                        st.warning("Pe is outside the calibrated range and was clamped: treat this depth with caution.")

    # ================= 10 Profile calibration =================
    with tabs[9]:
        head(9, "Profile calibration (melt pool outline)")
        gen = ss.get("_pc_gen", 0)
        b1, b2, b3 = st.columns(3)
        if b1.button("Load synthetic demo (eta = 0.30, 2 % noise)", key="btn_pc_demo", disabled=bool(errs)):
            with st.spinner("Generating demo outlines..."):
                tr_ = profile_demo(proc, mat, stg, [(0.8 * proc.P, 0.9 * proc.v), (proc.P, proc.v), (1.15 * proc.P, 1.1 * proc.v)], 0.30, 0.02, True)
            ss["pc_tr"], ss["pc_cs"], ss["pc_lg"] = tracks_to_tables(tr_); ss["_pc_gen"] = gen + 1; ss.pop("pc_res", None); st.rerun()
        if b2.button("Clear tables", key="btn_pc_clear"):
            for k_, c_ in (("pc_tr", ["track", "power_W", "speed_m_s"]), ("pc_cs", ["track", "y_um", "z_um"]), ("pc_lg", ["track", "x_um", "z_um"])):
                ss[k_] = pd.DataFrame({c: pd.Series(dtype=str if c == "track" else float) for c in c_})
            ss["_pc_gen"] = gen + 1; ss.pop("pc_res", None); st.rerun()
        dl_button(b3, "Boundary CSV template", "dl_pc_tpl", lambda: "track,y_um,z_um\n", "et_profile_template.csv", "text/csv")

        def _load_boundary(fu, key, cols, tgt):
            if fu is not None and ss.get(key) != fu.file_id:
                try:
                    d_ = pd.read_csv(fu, sep=None, engine="python"); d_.columns = [str(c).strip() for c in d_.columns]
                    if not set(cols) <= set(d_.columns):
                        raise ValueError(f"the file needs the columns {cols}")
                    ss[tgt] = d_[cols].copy(); ss[key] = fu.file_id
                    new = [n for n in d_["track"].astype(str).unique() if n not in set(ss["pc_tr"]["track"].astype(str))]
                    if new:
                        ss["pc_tr"] = pd.concat([ss["pc_tr"], pd.DataFrame({"track": new, "power_W": np.nan, "speed_m_s": np.nan})], ignore_index=True)
                    ss["_pc_gen"] = ss.get("_pc_gen", 0) + 1; st.rerun()
                except ValueError as ex:
                    st.error(f"Could not read the file: {ex}")
        u1, u2 = st.columns(2)
        _load_boundary(u1.file_uploader("Cross-section boundary CSV (track, y_um, z_um)", type=["csv"], key="pc_cs_file"), "_pc_cs_fid", ["track", "y_um", "z_um"], "pc_cs")
        _load_boundary(u2.file_uploader("Longitudinal boundary CSV (track, x_um, z_um) - optional", type=["csv"], key="pc_lg_file"), "_pc_lg_fid", ["track", "x_um", "z_um"], "pc_lg")
        st.markdown("**Tracks** (power and speed; the name links the boundary rows)")
        tr_e = st.data_editor(ss["pc_tr"], num_rows="dynamic", width="stretch", key=f"pc_tr_{gen}")
        e1, e2 = st.columns(2)
        e1.markdown("**Cross-section half outline**: y [um] across, z [um] down, origin = track centreline at the surface")
        cs_e = e1.data_editor(ss["pc_cs"], num_rows="dynamic", width="stretch", key=f"pc_cs_{gen}")
        e2.markdown("**Longitudinal outline (optional)**: x [um] (ahead of the beam centre positive), z [um] down")
        lg_e = e2.data_editor(ss["pc_lg"], num_rows="dynamic", width="stretch", key=f"pc_lg_{gen}")
        c1, c2 = st.columns(2)
        c1.checkbox("Also fit the beam-size scale factor", key="pc_fit_bs"); c2.number_input("Initial absorptivity", 0.05, 1.0, step=0.05, key="pc_eta0")
        if st.button("Run profile calibration", key="btn_pc_run", type="primary", disabled=bool(errs)):
            tracks, perr = tables_to_tracks(tr_e, cs_e, lg_e)
            for e_ in perr:
                st.error(e_)
            if tracks:
                with st.spinner(f"Fitting {len(tracks)} outline(s) (about 0.3 s per track and iteration)..."):
                    try:
                        ss["pc_res"] = dict(res=profile_calibrate(tracks, proc, mat, stg, ss.pc_fit_bs, ss.pc_eta0), tracks=tracks)
                    except ValueError as ex:
                        st.error(str(ex))
        pc = ss.get("pc_res")
        if pc:
            import matplotlib.pyplot as plt
            r_ = pc["res"]
            for w_ in r_["warnings"]:
                st.warning(w_)
            st.dataframe(pd.DataFrame({"parameter": r_["names"], "value": [r_["params"][n] for n in r_["names"]], "std. error": [r_["std_err"][n] for n in r_["names"]]}).round(4),
                         hide_index=True, width="stretch")
            st.caption(f"D = RMS radial distance between measured and simulated outline divided by the measured depth (0 = identical). RMS over tracks: {r_['D_rms']:.3f}. "
                       "Total D per track = sqrt(D_CS^2 + D_MF^2). The fitted absorptivity is an EFFECTIVE value, not a measured one.")
            st.dataframe(r_["table"].round(4), hide_index=True, width="stretch")
            st.markdown("**E-T pool length at the fitted parameters: a labelled PRIOR, not constrained by cross-sections**")
            st.dataframe(r_["prior"].round(1), hide_index=True, width="stretch")
            fg_ = fig_profiles(pc["tracks"], r_["info"]); st.pyplot(fg_)
            dl_button(st, "Overlay PNG (300 dpi)", "dl_pc", lambda f=fg_: fig_png_bytes(f), "et_profile_overlay.png", "image/png"); plt.close(fg_)

            def _apply_pc():
                ss["eta"] = float(min(1.0, max(0.01, round(r_["params"]["eta"], 4))))
                if "beam_scale" in r_["params"]:
                    ss["beam_scale"] = float(round(r_["params"]["beam_scale"], 4))
            st.button("Apply fitted values to the Process tab", on_click=_apply_pc, key="btn_pc_apply")

    # ================= 11 FE correction / Goldak surrogate =================
    with tabs[10]:
        head(10, "FE-informed correction and Goldak surrogate")
        st.warning("Goldak semi-axes are 5%-density radii (heat flux ~ exp(-3 x^2/a^2)), NOT melt pool dimensions: in a published example the fitted transverse axis was 102.7 um "
                   "for a measured half-width of about 63 um. Calibrated values are specific to the FE assumptions used, and the naming of a, b, c differs between papers.")
        fgen = ss.get("_fe_gen", 0)
        subhead(10, "6a) Discrepancy model: correct E-T with a few Abaqus runs")
        st.caption("Paste 10-25 Abaqus results computed with ONE fixed Goldak parameter set over a power-speed grid. The app fits  y_FE = y_ET x (1 + c0 + c1 P_norm + c2 v_norm)  "
                   "per output by ridge least squares and refuses to extrapolate outside the Abaqus range. Keep the Process/Material values unchanged after fitting (E-T is evaluated with them).")
        fe_e = st.data_editor(ss["fe_df"], num_rows="dynamic", width="stretch", key=f"fe_ed_{fgen}")
        f1, f2 = st.columns(2)
        f1.number_input("Ridge strength", 0.0, 10.0, format="%g", key="fe_ridge")
        if f2.button("Fit correction", key="btn_fe_fit", type="primary", disabled=bool(errs)):
            try:
                ss["fe"] = fe_fit(fe_e, proc, mat, stg, ss.fe_ridge)
            except ValueError as ex:
                ss["fe"] = None; st.error(str(ex))
        fe = ss.get("fe")
        if fe:
            for w_ in fe["warnings"]:
                st.warning(w_)
            if fe["fingerprint"] != fe_fingerprint(proc, mat, stg):
                st.warning("Process or Material values changed since the correction was fitted: re-fit it.")
            st.markdown("**Leave-one-out error: corrected versus raw E-T**"); st.dataframe(fe["metrics"].round(2), hide_index=True, width="stretch")
            st.dataframe(pd.DataFrame({"output": list(fe["coef"]), "c0": [c[0] for c in fe["coef"].values()], "c1 (power)": [c[1] for c in fe["coef"].values()],
                                       "c2 (speed)": [c[2] for c in fe["coef"].values()]}).round(4), hide_index=True, width="stretch")
            if not errs:
                r_ = run_case(proc, mat, stg)
                try:
                    cp_ = fe_predict(fe, proc.P, proc.v, {o: r_[o] * 1e6 for o in FE_OUTS})
                    st.success("Corrected prediction for the current Process: " + ", ".join(f"{o} {v_:.1f} um (E-T {r_[o] * 1e6:.1f})" for o, v_ in cp_.items()))
                except ValueError as ex:
                    st.warning(str(ex))

        subhead(10, "6b) Inverse Goldak calibration with a surrogate")
        st.caption("Generate a Latin-hypercube list of Goldak parameter sets (30-40 runs), run them in Abaqus, paste the widths/depths/lengths back, fit a surrogate, and let "
                   "differential evolution find the parameters that reproduce your measurement. c is the REAR semi-axis c_r; the front semi-axis is c_f = ratio x c_r. "
                   "A range with equal low and high values fixes that parameter (recommended for eta, taken from the E-T calibration).")
        g1, g2, g3, g4 = st.columns(4)
        g1.number_input("a low [um]", 1.0, key="gk_a0"); g1.number_input("a high [um]", 1.0, key="gk_a1")
        g2.number_input("b low [um]", 1.0, key="gk_b0"); g2.number_input("b high [um]", 1.0, key="gk_b1")
        g3.number_input("c low [um]", 1.0, key="gk_c0"); g3.number_input("c high [um]", 1.0, key="gk_c1")
        g4.number_input("eta low", 0.01, 1.0, step=0.05, key="gk_e0"); g4.number_input("eta high", 0.01, 1.0, step=0.05, key="gk_e1")
        n1, n2, n3 = st.columns(3)
        n1.number_input("Number of runs", 12, 200, key="gk_n"); n2.number_input("Random seed", 0, 10000, key="gk_seed"); n3.number_input("front/rear ratio c_f / c_r", 0.1, 2.0, step=0.05, key="gk_ratio")
        if st.button("Generate Latin-hypercube runs", key="btn_gk_doe"):
            bad = [n_ for n_, lo_, hi_ in (("a", ss.gk_a0, ss.gk_a1), ("b", ss.gk_b0, ss.gk_b1), ("c", ss.gk_c0, ss.gk_c1), ("eta", ss.gk_e0, ss.gk_e1)) if hi_ < lo_]
            if bad:
                st.error("High must be at least low for: " + ", ".join(bad))
            else:
                doe = lhs_doe({"a_um": (ss.gk_a0, ss.gk_a1), "b_um": (ss.gk_b0, ss.gk_b1), "c_um": (ss.gk_c0, ss.gk_c1), "eta": (ss.gk_e0, ss.gk_e1)}, int(ss.gk_n), int(ss.gk_seed))
                doe.insert(0, "run", range(1, len(doe) + 1)); doe["power_W"] = proc.P; doe["speed_m_s"] = proc.v
                for c_ in ("width_um", "depth_um", "length_um"):
                    doe[c_] = np.nan
                ss["gk_tab"] = doe; ss["_gk_gen"] = ss.get("_gk_gen", 0) + 1; ss.pop("gk_sur", None); ss.pop("gk_inv", None); st.rerun()
        if "gk_tab" in ss:
            st.markdown("Run each row in Abaqus and paste the results into the last columns (you can add a column named D for the profile metric). Power and speed may differ per row.")
            gk_e = st.data_editor(ss["gk_tab"], num_rows="dynamic", width="stretch", key=f"gk_ed_{ss.get('_gk_gen', 0)}")
            dl_button(st, "Download runs CSV", "dl_gk_doe", lambda d=ss["gk_tab"]: d.to_csv(index=False), "goldak_doe.csv", "text/csv")
            m1, m2 = st.columns(2)
            m1.selectbox("Surrogate", ["rbf", "quad"], key="gk_method", help="rbf = thin-plate-spline RBF; quad = quadratic ridge regression")
            if m2.button("Fit surrogate", key="btn_gk_fit"):
                outs = [c for c in ("width_um", "depth_um", "length_um", "D") if c in gk_e and gk_e[c].notna().sum() >= 8]
                ins = ["a_um", "b_um", "c_um", "eta"] + [c for c in ("power_W", "speed_m_s") if c in gk_e and gk_e[c].nunique() > 1]
                dd = gk_e.dropna(subset=ins + outs) if outs else gk_e.iloc[0:0]
                if not outs or len(dd) < 8:
                    st.error("Need at least 8 complete rows (inputs and at least one output).")
                else:
                    sur = Surrogate(ss.gk_method).fit(dd[ins].values.astype(float), dd[outs].values.astype(float), ins, outs)
                    ss["gk_sur"], ss["gk_loo"] = sur, sur.loo(); ss.pop("gk_inv", None)
            if ss.get("gk_sur") is not None:
                st.markdown("**Surrogate leave-one-out error**"); st.dataframe(ss["gk_loo"].round(3), hide_index=True, width="stretch")
                st.markdown("**Measured targets** (0 = not used)")
                t1, t2, t3 = st.columns(3)
                t1.number_input("Target width [um]", 0.0, key="gk_tw"); t1.number_input("weight", 0.0, 10.0, key="gk_ww")
                t2.number_input("Target depth [um]", 0.0, key="gk_td"); t2.number_input("weight ", 0.0, 10.0, key="gk_wd")
                t3.number_input("Target length [um]", 0.0, key="gk_tl"); t3.number_input("weight  ", 0.0, 10.0, key="gk_wl")
                if st.button("Find parameters (differential evolution)", key="btn_gk_inv", type="primary"):
                    sur = ss["gk_sur"]; tg = {o: v_ for o, v_ in (("width_um", ss.gk_tw), ("depth_um", ss.gk_td), ("length_um", ss.gk_tl)) if v_ > 0 and o in sur.out_names}
                    if "D" in sur.out_names:
                        tg["D"] = 0.0
                    if not tg:
                        st.error("Enter at least one target that the surrogate predicts.")
                    else:
                        fixed = {n_: v_ for n_, v_ in (("power_W", proc.P), ("speed_m_s", proc.v)) if n_ in sur.in_names}
                        with st.spinner("Searching..."):
                            ss["gk_inv"] = inverse_de(sur, tg, {"width_um": ss.gk_ww, "depth_um": ss.gk_wd, "length_um": ss.gk_wl}, fixed=fixed)
                inv = ss.get("gk_inv")
                if inv:
                    for w_ in inv["warnings"]:
                        st.warning(w_)
                    ct = inv["table"].copy()
                    if "c_um" in ct:
                        ct.insert(ct.columns.get_loc("c_um") + 1, "c_f_um (ratio x c)", ct["c_um"] * ss.gk_ratio)
                    st.markdown("**Suggested parameters: run the best one and the alternatives in Abaqus as verification runs**"); st.dataframe(ct.round(3), hide_index=True, width="stretch")
                    st.caption("Axes are 5%-density radii for the FE assumptions of your runs. If verification runs disagree with the prediction, add them to the table and refit.")

    # ================= 12 Assumptions =================
    with tabs[11]:
        head(11, "Assumptions and limits")
        st.markdown("""
**Physics**
- Eagar-Tsai: Gaussian surface source on a semi-infinite solid, **constant** k, rho, cp (alpha = k/(rho cp)), quasi-steady state, no latent heat, convection, radiation, Marangoni flow or evaporation. **Conduction mode only.**
- Beam frame: x < 0 behind the beam, z >= 0 into the solid. Absorbed power q = eta*P; the substrate temperature T0 is an input (the optional eagar-tsai library fixes 298 K).
- Source standard deviation: sigma = D/4 for a D4sigma or 1/e^2 **diameter**; sigma = w/2 for a 1/e^2 **radius**. The optional library takes beam_diameter = 2*sigma.
- Melt pool = region above the **liquidus**: width from the surface (z = 0), depth from the centreline plane (y = 0), length from the surface centreline (head ahead of, tail behind, the beam centre). Width and depth are maximised over x.
- Thermal gradient and cooling rate are **E-T estimates** at the trailing liquidus point (G = |dT/dx|, cooling rate = v*G), not FEM results.

**Numerics**
- Composite Gauss-Legendre in s (t = s^2), geometric panel spacing (default 48 x 8 nodes, plus 8 sub-panels below the first edge s_lo = 0.05*min(sigma/sqrt(2 alpha), sqrt(sigma/v))); upper limit S = 1.1*sqrt(|x|max/v + u). Checked against Rosenthal, scipy.quad (>= 300 random points, relative error < 1e-4) and the library (`python et_modeler.py --selftest`).

**Flags (all thresholds editable heuristics, not laws)**
- Depth/width >= transition / keyhole thresholds; optional normalized-enthalpy flag; peak T > 20 x liquidus; peak T > T_boil. **E-T underpredicts keyhole depth** and its peak temperatures are unrealistically high at high power.

**Calibration**
- Fits an **effective** absorptivity (optionally beam-size and conductivity scale factors) by least squares on relative width/depth errors. It is not a measured absorptivity, it absorbs every model/machine/powder error, and **it does not transfer between alloys**: re-calibrate with a few tracks of any new material.
- Standard errors and correlations come from the Jacobian (linearised); a warning is raised for |r| > 0.9 or a standard error > 25% of the value. The emcee posterior (optional) reflects measurement scatter only.
- Hold-out is a random split of tracks; leave-one-out refits once per track. Tracks flagged "exclude" (blank = not excluded) and, optionally, tracks above a depth/width cut-off are never used.
- Extrapolating outside the calibrated power/speed range is flagged in Results.

**Advanced modules**
- Hybrid depth: only unlocked when some measured depth/width exceeds 0.5. A in the formula is the Process-tab absorptivity; a = 2 sigma (1/e^2 radius of the Gaussian). g(Pe) is fitted by linear least squares (>= 8 tracks, boiling temperature required), Pe is clamped to the calibrated range, depth = R x E-T width.
- Profile calibration: the simulated cross-section outline is the ENVELOPE over x (max over 25 x-samples of T >= T_liq) found by bisection along 61 rays; a longitudinal outline is the instantaneous y = 0 pool outline in the beam frame (x ahead of the beam centre positive), so its x origin must be the beam position. D uses the experimental depth of the same track as normaliser.
- FE correction: a multiplicative linear-in-(P, v) discrepancy on top of the E-T model with the CURRENT Process/Material values; it never extrapolates outside the pasted (P, v) range. Goldak inversion: c is the rear semi-axis c_r (c_f = ratio x c_r); with fewer targets than free parameters the solution is not unique.
- Process window: lack of fusion (H/w)^2 + (L/d)^2 <= 1 and keyhole limit w/d > 1.5 (editable); H is the Process-tab hatch, L the Process-tab layer thickness unless you choose a custom value.

**Data and units**
- Internally SI (m, W, m/s, K); tables carry units in the column names. Uploaded columns are converted using the selected units. No real-alloy values are built in; every material value has its own source field.
- Goldak values are heuristic starting points derived from the E-T geometry, not calibrated parameters.
""")


    # ================= 10 References =================
    with tabs[12]:
        head(12, "References and sources")
        st.markdown("**How this list was assembled.** The equation and method sources are standard literature that I (the AI that wrote this app) know; "
                    "two items (the eagar-tsai library and the King et al. 2014 paper) were additionally checked online while building it. "
                    "Items marked *from memory* were NOT looked up: confirm authors, volume and pages in the original before you cite them in a thesis or paper.")
        for grp in dict.fromkeys(r_[0] for r_ in REFERENCES):
            subhead(12, grp)
            for g_, cite, used, status in REFERENCES:
                if g_ == grp:
                    st.markdown(f"- {cite}  \n  *Used for:* {used}  \n  *How it was checked:* {status}")
        subhead(12, "Where the default numbers come from")
        st.table(pd.DataFrame([
            ("Depth/width 0.5 (transition) and 1.0 (keyhole)", "Rules of thumb chosen when writing the app, not from one paper. Editable in Model settings."),
            ("Normalized enthalpy threshold 6 (flag off by default)", "Informal value in the spirit of King et al. 2014; verify the definition for your alloy before enabling."),
            ("Peak temperature warning at 20 x T_liquidus", "Rule of thumb: E-T has no evaporation, so very high peaks mean the case is outside its validity."),
            ("Sensitivity step +/-10 %", "Arbitrary small step for a one-at-a-time sensitivity estimate."),
            ("Quadrature 48 panels x 8 nodes (+ 8 sub-panels)", "Self-tested against scipy.quad on >= 300 random points (relative error < 1e-4) and against Rosenthal."),
            ("Initial absorptivity 0.4 in Calibration", "Arbitrary starting guess for the optimiser; the result should not depend on it."),
            ("'Generic example' material", "Placeholder numbers for testing only. It is not a real alloy."),
        ], columns=["default", "origin"]).set_index("default"))

    # ---- single Excel report: everything in one file ----
    if not errs:
        extra = []
        if ss.get("hy") is not None:
            extra += [("Hybrid depth", "Hybrid depth: tracks", "g(Pe) fit per track and leave-one-out depth", ss["hy"].table, "#827717"),
                      ("Hybrid depth errors", "Hybrid depth: errors", "hybrid (LOO) versus plain E-T", ss["hy"].metrics, "#827717")]
        if ss.get("pc_res"):
            pr_ = ss["pc_res"]["res"]
            extra += [("Profile fit", "Profile calibration: parameters", "effective values", pd.DataFrame({"parameter": pr_["names"], "value": [pr_["params"][n] for n in pr_["names"]],
                                                                                                          "std. error": [pr_["std_err"][n] for n in pr_["names"]]}), "#00695C"),
                      ("Profile distances", "Profile calibration: D per track", "0 = identical outlines", pr_["table"], "#00695C"),
                      ("Profile E-T prior", "E-T pool length: labelled PRIOR", "not constrained by cross-sections", pr_["prior"], "#00695C")]
        if ss.get("fe"):
            extra += [("FE correction", "FE correction: leave-one-out error", "corrected versus raw E-T", ss["fe"]["metrics"], "#5E35B1"),
                      ("FE correction data", "FE correction: per-run table", "E-T, Abaqus and leave-one-out corrected values", ss["fe"]["table"], "#5E35B1")]
        if ss.get("gk_inv"):
            extra += [("Goldak inversion", "Suggested Goldak parameters", "verification runs; axes are 5%-density radii", ss["gk_inv"]["table"], "#5E35B1")]
        if ss.get("gk_tab") is not None and len(ss["gk_tab"]):
            extra += [("Goldak DOE", "Latin-hypercube runs and pasted results", "", ss["gk_tab"], "#5E35B1")]
        if ss.get("_sw_crit") is not None and ss.sw_ovl:
            extra += [("Process window", "Process window", "lack of fusion and keyhole criteria", ss["_sw_crit"], "#283593")]
        rpt.update(extra=extra, proc=proc, mat=mat, stg=stg, goldak_conv=GOLDAK_CONVENTIONS[ss.goldak_conv], cal=ss.get("cal"), sweep_df=ss.get("sweep"),
                   prop_tabs=ss.get("_tp_tabs"), prop_eff=ss.get("_tp_eff"), prop_notes=ss.get("_tp_notes"))
    with export_slot:
        st.caption("One Excel file with summary, inputs, Goldak values, calibration, sweep, property table, notes and references.")
        dl_button(st, "Download full Excel report", "dl_xlsx_side", make_report, "et_report.xlsx", XLSX_MIME, type="primary", disabled=bool(errs))


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(1 if run_selftests() else 0)
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        _ctx = get_script_run_ctx()
    except Exception:
        _ctx = None
    if _ctx is None and "--uitest" in sys.argv:
        sys.exit(1 if run_uitests() else 0)
    if _ctx is None:
        print("Run the app with:  streamlit run et_modeler.py      (self-tests: --selftest, headless UI tests: --uitest)")
    else:
        main()
