"""
RIconduit2D_phi_r.py
====================
Reducción II.E de la propuesta: u_r = 0, P = P(z), y
    φ = φ(r,z),  N = N(r,z),  ξ = ξ(r,z),  u_m = u_m(r,z).

Diferencias finitas en r (Thomas + Picard), marcha adaptativa en z
(un paso h frente a dos de h/2, como en el PDF), umbrales suaves
de umbrales_reg.py (tanh). No toca RIconduit2D_FD.py ni los 1D.

El 2D viejo guarda φ(z) plano. Aquí cada nodo radial tiene su φ, N, ξ.
"""

from __future__ import annotations

import math
import warnings

import numpy as np

from calbuco2015d import (
    C1,
    F1,
    F2,
    Fc,
    H,
    Patm,
    R,
    ar1,
    ar2,
    beta,
    f2o,
    g,
    geometry,
    limperh,
    limperl,
    linf,
    lsup,
    model,
    overP1,
    pfinal,
    rcrust,
    tcar,
    xmax,
)
from density import density
from fvrel import fvrel
from umbrales_reg import (
    EPS_FRAG,
    EPS_HENRY,
    EPS_PHI,
    EPS_RB,
    EPS_RE,
    EPS_XI,
    guarda_coalescencia,
    mezclar,
    pesos_regimen,
    s_henry,
    s_re,
    tasa_xi,
)
from viscosity import viscosity
from calbuco2015d import (
    al2o3,
    cao,
    feo,
    k2o,
    mgo,
    mno,
    na2o,
    p2o5,
    sio2,
    tio2,
)

warnings.filterwarnings("ignore")

_G: dict = {}


def _fin(x, fill=0.0):
    x = np.asarray(x, dtype=float)
    out = np.nan_to_num(x, nan=fill, posinf=1e30, neginf=-1e30)
    return float(out) if out.ndim == 0 else out


def _area_weights():
    r, wr = _G["r"], _G["wr"]
    w = 2.0 * math.pi * r
    w = w / max(float(np.trapezoid(w, r)), 1e-30)
    return w


def _fg(P, x_cr):
    """Fracción másica de gas, Henry suave."""
    P = max(float(np.real(P)), 1e4)
    x_cr = float(np.clip(x_cr, 0.0, 0.95))
    test = (1.0 - _G["xi"]) * _G["co"] - (1.0 - x_cr) * C1 * P ** beta
    den = max(1.0 - C1 * P ** beta, 1e-12)
    fg_sat = ((_G["co"] * (1.0 - _G["xi"]) - C1 * (1.0 - x_cr) * P ** beta) / den)
    return max(0.0, float(s_henry(test, _G["eps_henry"])) * fg_sat)


def _phi_alg(P, fg):
    if fg <= 1e-16:
        return 0.0
    P = max(float(P), 1e4)
    phi = 1.0 / (1.0 + (P / (fg * R * _G["T"])) * (1.0 - fg) / _G["rho_m"])
    return float(np.clip(phi, 0.0, 0.99))


def _rho_g(P):
    return max(float(np.real(P)), 1e4) / (R * _G["T"])


def _rb(phi, Nd):
    phi_c = float(np.clip(phi, 0.0, 0.999))
    Nd_s = max(float(Nd), 1.0)
    rb = (max(phi_c, 1e-16) / ((4.0 / 3.0) * math.pi * Nd_s * max(1.0 - phi_c, 1e-12))) ** (1.0 / 3.0)
    return max(float(rb), 1e-8)


def _visc_nodo(P, phi, x_cr, um, Nd):
    """Giordano + fvrel + Llewellin (burbujas), un nodo."""
    wr, Tc = _G["wr"], _G["Tc"]
    fg = _fg(P, x_cr)
    sH = float(s_henry((1.0 - _G["xi"]) * _G["co"] - (1.0 - x_cr) * C1 * max(P, 1e4) ** beta, _G["eps_henry"]))
    h2o_sat = C1 * max(P, 1e4) ** beta * 100.0
    viscl_sat = fvrel(model, x_cr, _G["xi"], ar1, ar2, _G["xmax"], max(um, 1e-9) / wr) * viscosity(
        sio2, tio2, al2o3, feo, mno, mgo, cao, na2o, k2o, p2o5, h2o_sat, f2o, Tc
    )
    viscl_un = fvrel(model, x_cr, _G["xi"], ar1, ar2, _G["xmax"], max(um, 1e-9) / wr) * viscosity(
        sio2, tio2, al2o3, feo, mno, mgo, cao, na2o, k2o, p2o5, _G["h2o"], f2o, Tc
    )
    viscl = float(mezclar(sH, viscl_un, viscl_sat))
    phi_c = float(np.clip(phi, 1e-12, _G["phicrit"] - 1e-6))
    rb = _rb(phi_c, Nd)
    nca = rb * viscl * max(um / wr, 1e-12) / 3.0
    phicritbub = _G["phicrit"] + 0.05
    AA = (1.0 - phi_c / phicritbub) ** (-phicritbub)
    BB = (1.0 - phi_c / phicritbub) ** (5.0 * phicritbub / 3.0)
    c1c = -0.2895 * phi_c + 0.8132
    viscrel = 0.5 * (AA - BB) * (1.0 - math.erf(float(c1c * math.log(max(nca, 1e-30)) + phi_c))) + BB
    return float(np.real(viscrel * viscl))


def _Fmg_suave(um, ug, phi, Nd, rho_g, visc):
    """F_mg = Σ w_k F_mg^{(k)}, C^∞ en φ. Vectorial o escalar."""
    um = np.asarray(um, dtype=float)
    ug = np.asarray(ug, dtype=float)
    phi = np.asarray(phi, dtype=float)
    Nd = np.asarray(Nd, dtype=float)
    visc = np.asarray(visc, dtype=float)
    scalar = um.ndim == 0
    phi_c = np.clip(phi, 0.0, 0.999)
    rb = np.maximum(
        (np.maximum(phi_c, 1e-16) / ((4.0 / 3.0) * math.pi * np.maximum(Nd, 1.0) * np.maximum(1.0 - phi_c, 1e-12)))
        ** (1.0 / 3.0),
        1e-8,
    )
    w1, w2, w3, w4 = pesos_regimen(phi_c, _G["limphi1"], _G["limphi2"], _G["phicrit"], _G["eps_phi"], _G["eps_frag"])
    slip = ug - um
    span = max(_G["limphi2"] - _G["limphi1"], 1e-8)
    tt = np.clip((phi_c - _G["limphi1"]) / span, 0.0, 1.0)
    Re = 2.0 * rb * rho_g * np.abs(slip) / 1e-5
    sRe = s_re(Re, eps=_G["eps_re"])
    kper = np.maximum(0.131 * (rb ** 2) * ((np.maximum(phi_c - _G["limphi1"], 0.0) + 0.05) ** 2.1), 1e-30)
    fac = slip * phi_c * (1.0 - phi_c)
    F1v = 3.0 * visc * fac / (rb ** 2)
    F2_in = ((0.33 / (4.0 * rb)) * rho_g * (np.abs(slip) ** tt)) * ((3.0 * visc / (rb ** 2)) ** (1.0 - tt)) * fac
    F2_st = ((1e-5 / kper) ** tt) * ((3.0 * visc / (rb ** 2)) ** (1.0 - tt)) * fac
    F2v = mezclar(sRe, F2_st, F2_in)
    F3_in = (0.33 / (4.0 * rb)) * rho_g * slip * np.abs(slip) * phi_c * (1.0 - phi_c)
    F3_st = (1e-5 / kper) * fac
    F3v = mezclar(sRe, F3_st, F3_in)
    ra, cd = 1e-3, 0.8
    F4v = 3.0 * cd * rho_g * np.abs(slip) * slip * phi_c * (1.0 - phi_c) / (8.0 * ra)
    mute = (np.abs(w1) >= 1e-8).astype(float)
    Fmg = mute * w1 * _fin(F1v) + (np.abs(w2) >= 1e-8) * w2 * _fin(F2v)
    Fmg = Fmg + (np.abs(w3) >= 1e-8) * w3 * _fin(F3v) + (np.abs(w4) >= 1e-8) * w4 * _fin(F4v)
    Fmg = np.where(phi_c < 1e-10, 0.0, Fmg)
    if scalar:
        return float(_fin(Fmg))
    return _fin(Fmg)


def _thomas(lo, di, up, rhs):
    n = len(di)
    c, d, b, L = up.copy(), rhs.copy(), di.copy(), lo.copy()
    for i in range(1, n):
        w = L[i] / b[i - 1]
        b[i] -= w * c[i - 1]
        d[i] -= w * d[i - 1]
    x = np.zeros(n)
    x[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
    return x


def radial_tridiag(phi_arr, mu_arr):
    """L_r[μ, φ] con φ_i distinto en cada nodo. u[I]=0 no se despeja."""
    r, dr, N = _G["r"], _G["dr"], _G["N_r"]
    a = np.asarray(mu_arr, dtype=float) * (1.0 - np.clip(phi_arr, 0.0, 0.999))
    nunk = N - 1
    lo = np.zeros(nunk)
    di = np.zeros(nunk)
    up = np.zeros(nunk)
    a_h = 0.5 * (a[0] + a[1])
    di[0] = -2.0 * a_h / dr ** 2
    up[0] = 2.0 * a_h / dr ** 2
    for i in range(1, nunk - 1):
        a_p = 0.5 * (a[i] + a[i + 1])
        a_m = 0.5 * (a[i] + a[i - 1])
        lo[i] = (r[i] - dr / 2.0) * a_m / (r[i] * dr ** 2)
        di[i] = -((r[i] + dr / 2.0) * a_p + (r[i] - dr / 2.0) * a_m) / (r[i] * dr ** 2)
        up[i] = (r[i] + dr / 2.0) * a_p / (r[i] * dr ** 2)
    i = nunk - 1
    a_p = 0.5 * (a[i] + a[i + 1])
    a_m = 0.5 * (a[i] + a[i - 1])
    lo[i] = (r[i] - dr / 2.0) * a_m / (r[i] * dr ** 2)
    di[i] = -((r[i] + dr / 2.0) * a_p + (r[i] - dr / 2.0) * a_m) / (r[i] * dr ** 2)
    return lo, di, up


def _stokes_C(phi, Nd, visc):
    """Coeficiente lineal de arrastre: F_mg ≈ C (u_g - u_m). Va a la diagonal."""
    phi_c = np.clip(np.asarray(phi, dtype=float), 0.0, 0.999)
    visc = np.asarray(visc, dtype=float)
    Nd = np.asarray(Nd, dtype=float)
    rb = np.maximum(
        (np.maximum(phi_c, 1e-16) / ((4.0 / 3.0) * math.pi * np.maximum(Nd, 1.0) * np.maximum(1.0 - phi_c, 1e-12)))
        ** (1.0 / 3.0),
        1e-8,
    )
    C = 3.0 * visc * phi_c * (1.0 - phi_c) / (rb ** 2)
    C = np.where(phi_c < 1e-10, 0.0, C)
    return np.clip(_fin(C), 0.0, 1e8)


def solve_radial(P, phi, Nd, x_cr, ug, dpdz, n_picard=4):
    """BVP radial con φ(r), N(r), ξ(r). Arrastre implícito en la diagonal."""
    r, dr, N, wr = _G["r"], _G["dr"], _G["N_r"], _G["wr"]
    phi = np.asarray(phi, dtype=float)
    Nd = np.asarray(Nd, dtype=float)
    x_cr = np.asarray(x_cr, dtype=float)
    ug = np.asarray(ug, dtype=float)
    if ug.ndim == 0:
        ug = np.full(N, float(ug))
    u_ref = max(float(np.mean(np.maximum(ug, 0.0))), 1.0)
    um = 2.0 * u_ref * (1.0 - (r / wr) ** 2)
    um[-1] = 0.0
    um = np.maximum(um, 0.0)
    for _ in range(n_picard):
        mu = np.array([_visc_nodo(P, phi[i], x_cr[i], max(um[i], 1e-9), Nd[i]) for i in range(N)])
        # Viscoso con φ(r). El Stokes C~1/rb² aplasta u_m a u_g (perfil plano);
        # el arrastre suave se usa después, en dφ/dz, no aquí.
        body = (1.0 - phi) * dpdz + _G["rho_m"] * (1.0 - phi) * g
        lo, di, up = radial_tridiag(phi, mu)
        nunk = N - 1
        um_new = np.zeros(N)
        um_new[:nunk] = _thomas(lo, di, up, body[:nunk])
        um_new[-1] = 0.0
        um = 0.5 * um + 0.5 * np.maximum(um_new, 0.0)
    um_avg = float(np.trapezoid(um * 2.0 * math.pi * r, r) / (math.pi * wr ** 2))
    return um, um_avg, 0.0


def _promedio_seccion(phi, Nd, x_cr, um, ug):
    w = _area_weights()
    r, wr = _G["r"], _G["wr"]
    area = math.pi * wr ** 2
    return {
        "phi": float(np.dot(w, phi)),
        "Nd": float(np.dot(w, Nd)),
        "x": float(np.dot(w, x_cr)),
        "um": float(np.trapezoid(um * 2.0 * math.pi * r, r) / area),
        "ug": float(np.trapezoid(np.asarray(ug, dtype=float) * 2.0 * math.pi * r, r) / area),
    }


def _Fmw_seccion(P, av, um):
    """Fricción de pared con φ de sección, no el nodo de la pared.

    Si se usa φ_{I-1} y el 2×2 la manda a 0.8, (1-φ) se va y P no cae.
    """
    mu = _visc_nodo(P, av["phi"], av["x"], max(av["um"], 1e-9), av["Nd"])
    dudr = (um[-1] - um[-2]) / _G["dr"]
    F_grad = -2.0 * (1.0 - av["phi"]) * mu * dudr / _G["wr"]
    F_hp = _G["cg"] * mu * max(av["um"], 0.0) / _G["wr"] ** 2
    if (not np.isfinite(F_grad)) or F_grad < 0.2 * max(F_hp, 1e-6):
        return float(_fin(F_hp)), mu
    return float(_fin(F_grad)), mu


def _dpdz_seccion(P, av, Fmw, Fmg):
    """Lubricación con φ y F_mw de sección (como el 2D viejo).

    El 2×2 1D acá se iba a −10^8 Pa/m y P caía a 0.01 MPa a z=−600 m.
    """
    rg = _rho_g(P)
    phi = float(np.clip(av["phi"], 0.0, 0.99))
    rho_mix = rg * phi + _G["rho_m"] * (1.0 - phi)
    return float(-(rho_mix * g + max(float(Fmw), 0.0)))


def _dN_dx_nodo(P, phi, Nd, x_cr, um):
    """Γ_N / u_m y Γ_ξ / u_m en un nodo (apagados al fragmentar)."""
    w1, w2, w3, w4 = pesos_regimen(phi, _G["limphi1"], _G["limphi2"], _G["phicrit"], _G["eps_phi"], _G["eps_frag"])
    um_s = max(abs(float(um)), 1e-6)
    f2, _xteo, f3, _dx1 = tasa_xi(
        (_G["co"] * (1.0 - _G["xi"]) - (1.0 - x_cr) * C1 * max(P, 1e4) ** beta)
        / max(_G["co"] * (1.0 - _G["xi"]) - (1.0 - _G["xmax"]) * C1 * Patm ** beta, 1e-30),
        x_cr,
        _G["xi"],
        _G["xmax"],
        tcar,
        _G["eps_xi"],
    )
    dxdz = (1.0 - w4) * (w1 + w2 + w3) * (f2 * f3 * (_G["xmax"] - _G["xi"]) / max(tcar * um_s, 1e-30))
    visc = _visc_nodo(P, phi, x_cr, um_s, Nd)
    rg = _rho_g(P)
    rb = _rb(phi, Nd)
    inner = (F1 + F2) * ((3.0 * max(phi, 0.0) * (math.pi / (6.0 * _G["phicrit"])) / (4.0 * math.pi)) ** (1.0 / 3.0))
    den_n = 1.0 - inner
    if abs(den_n) < 1e-8 or phi < 1e-12:
        dNdt_on = 0.0
    else:
        dNdt_on = (
            -(max(Nd, 1.0) ** (2.0 / 3.0))
            * ((1.0 / max(1.0 - phi, 1e-12)) ** (1.0 / 3.0))
            * ((1.0 / 9.0) * (_G["rho_m"] - rg) * 9.81 / max(visc, 1e-30))
            * ((3.0 * max(phi, 0.0) / (4.0 * math.pi)) ** (2.0 / 3.0))
            * (F1 * F1 - F2 * F2)
            / den_n
            * Fc
            * (1.0 - phi / _G["phicrit"])
            * (_G["wr"] - rb)
            / _G["wr"]
        )
    dNdt = float(guarda_coalescencia(rb, _G["wr"], phi, _G["phicrit"], eps_frac=_G["eps_rb"], eps_frag=_G["eps_frag"])) * _fin(
        dNdt_on
    )
    dNdz = (1.0 - w4) * dNdt / um_s
    return float(_fin(dNdz)), float(_fin(dxdz))


def _dphi_nodo(P, phi, Nd, x_cr, um, ug, Fmg, dpdz):
    """dφ/dz local (2×2 de masa+momentum) con dP/dz compartido."""
    fg = _fg(P, x_cr)
    phi_a = _phi_alg(P, fg)
    if fg <= 1e-16:
        return 0.0
    if phi < 1e-8 or um < 1e-3:
        eps_P = max(abs(P) * 1e-6, 100.0)
        phi_ep = _phi_alg(P + eps_P, _fg(P + eps_P, x_cr))
        return float(np.clip((phi_ep - phi_a) / eps_P * dpdz, -20.0, 20.0))
    rg = _rho_g(P)
    um = max(float(um), 1e-9)
    ug = max(float(ug), 1e-9)
    f2, _xteo, f3, dx1 = tasa_xi(
        (_G["co"] * (1.0 - _G["xi"]) - (1.0 - x_cr) * C1 * max(P, 1e4) ** beta)
        / max(_G["co"] * (1.0 - _G["xi"]) - (1.0 - _G["xmax"]) * C1 * Patm ** beta, 1e-30),
        x_cr,
        _G["xi"],
        _G["xmax"],
        tcar,
        _G["eps_xi"],
    )
    dxdp = dx1 / max(um, 1e-6)
    dfgdp = (
        -(-dxdp * C1 * P ** beta + (1.0 - x_cr) * C1 * beta * P ** (beta - 1))
        + (_G["co"] * (1.0 - _G["xi"]) - (1.0 - x_cr) * C1 * P ** beta) * C1 * beta * P ** (beta - 1)
    ) / max((1.0 - C1 * P ** beta) ** 2, 1e-30)
    aco = rg * ug ** 2
    bco = phi - ug ** 2 * phi / (R * _G["T"]) + dfgdp * _G["q"] * ug
    cco = _G["rho_m"] * um ** 2
    dco = dfgdp * _G["q"] * um - (1.0 - phi)
    eco = -_G["rho_m"] * (1.0 - phi) * g + Fmg
    fco = rg * phi * g + Fmg
    den = aco * dco - bco * cco
    if abs(den) < 1e-18 or not np.isfinite(den):
        eps_P = max(abs(P) * 1e-6, 100.0)
        phi_ep = _phi_alg(P + eps_P, _fg(P + eps_P, x_cr))
        return float(np.clip((phi_ep - _phi_alg(P, fg)) / eps_P * dpdz, -20.0, 20.0))
    dphidz = (-eco * bco + dco * fco) / den
    return float(np.clip(_fin(dphidz), -20.0, 20.0))


def derivs(P, phi, Nd, x_cr, dpdz_prev):
    """
    Un nivel z. P es escalar; φ, N, ξ son vectores en r.
    dP/dz sale del promedio de sección (lubricación).
    """
    N = _G["N_r"]
    r, wr = _G["r"], _G["wr"]
    phi = np.clip(np.asarray(phi, dtype=float), 0.0, 0.99)
    Nd = np.maximum(np.asarray(Nd, dtype=float), 1.0)
    x_cr = np.clip(np.asarray(x_cr, dtype=float), _G["xi"], _G["xmax"])
    P = max(float(P), 1e4)
    rg = _rho_g(P)
    fg = np.array([_fg(P, x_cr[i]) for i in range(N)])
    # ug local por masa (q global); si no hay gas, = um
    ug0 = np.zeros(N)
    for i in range(N):
        if fg[i] <= 1e-16 or phi[i] < 1e-10:
            ug0[i] = 0.0
        else:
            ug0[i] = min(_G["q"] * fg[i] / (rg * max(phi[i], 1e-10)), 400.0)

    dpdz = float(dpdz_prev)
    um = np.zeros(N)
    F_mw = 0.0
    w = _area_weights()
    ug_loc = ug0
    for _ in range(4):
        um, um_avg, _ = solve_radial(P, phi, Nd, x_cr, ug0, dpdz)
        ug_loc = np.where((fg > 1e-16) & (phi > 1e-10), ug0, um)
        Q_now = float(
            np.trapezoid(
                (_G["rho_m"] * (1.0 - phi) * um + rg * phi * ug_loc) * 2.0 * math.pi * r,
                r,
            )
        )
        if Q_now > 1e-8 and um_avg > 1e-10:
            um = um * (_G["Q"] / Q_now)
            um[-1] = 0.0
        av = _promedio_seccion(phi, Nd, x_cr, um, ug_loc)
        F_mw, _mu = _Fmw_seccion(P, av, um)
        mu_arr = np.array([_visc_nodo(P, phi[i], x_cr[i], max(um[i], 1e-9), Nd[i]) for i in range(N)])
        Fmg_arr = _Fmg_suave(np.maximum(um, 1e-9), ug_loc, phi, Nd, rg, mu_arr)
        Fmg_avg = float(np.dot(w, Fmg_arr))
        dpdz_new = _dpdz_seccion(P, av, F_mw, Fmg_avg)
        if abs(dpdz_new - dpdz) < 1e-4 * max(abs(dpdz), 1.0):
            dpdz = dpdz_new
            break
        dpdz = dpdz_new

    ug = ug_loc
    mu = np.array([_visc_nodo(P, phi[i], x_cr[i], max(um[i], 1e-9), Nd[i]) for i in range(N)])
    Fmg = _Fmg_suave(np.maximum(um, 1e-9), ug, phi, Nd, rg, mu)

    dphi = np.zeros(N)
    dNd = np.zeros(N)
    dx = np.zeros(N)
    um_typ = max(float(np.max(um)), 1e-6)
    for i in range(N - 1):
        um_d = um[i] if um[i] >= 0.25 * um_typ else 1e-4
        dphi[i] = _dphi_nodo(P, phi[i], Nd[i], x_cr[i], um_d, ug[i], Fmg[i], dpdz)
        dNd[i], dx[i] = _dN_dx_nodo(P, phi[i], Nd[i], x_cr[i], um[i])
    # pared: u_m=0 → se copia el interior (II.F)
    dphi[-1] = dphi[-2]
    dNd[-1] = dNd[-2]
    dx[-1] = dx[-2]
    return float(dpdz), dphi, dNd, dx, um, ug, float(F_mw)


def _clamp_state(P, phi, Nd, x_cr):
    P = float(np.clip(P, 1e4, 2e9))
    phi = np.clip(np.asarray(phi, dtype=float), 0.0, 0.99)
    Nd = np.maximum(np.asarray(Nd, dtype=float), 1.0)
    x_cr = np.clip(np.asarray(x_cr, dtype=float), _G["xi"], _G["xmax"])
    return P, phi, Nd, x_cr


def _proyectar_phi(P, phi, x_cr):
    """Banda de la EOS por nodo; la pared copia el interior."""
    phi = np.asarray(phi, dtype=float).copy()
    x_cr = np.asarray(x_cr, dtype=float)
    for i in range(phi.size):
        pha = _phi_alg(P, _fg(P, x_cr[i]))
        if pha <= 1e-12:
            phi[i] = 0.0
        else:
            phi[i] = float(np.clip(phi[i], 0.4 * pha, min(0.99, 1.6 * pha)))
    phi[-1] = phi[-2]
    return phi


def rk4_step(P, phi, Nd, x_cr, dpdz, h):
    """Un paso RK4 del estado (P, φ(r), N(r), ξ(r))."""

    def f(P_, phi_, Nd_, x_, dpg):
        P_, phi_, Nd_, x_ = _clamp_state(P_, phi_, Nd_, x_)
        return derivs(P_, phi_, Nd_, x_, dpg)

    k1 = f(P, phi, Nd, x_cr, dpdz)
    k2 = f(P + 0.5 * h * k1[0], phi + 0.5 * h * k1[1], Nd + 0.5 * h * k1[2], x_cr + 0.5 * h * k1[3], k1[0])
    k3 = f(P + 0.5 * h * k2[0], phi + 0.5 * h * k2[1], Nd + 0.5 * h * k2[2], x_cr + 0.5 * h * k2[3], k2[0])
    k4 = f(P + h * k3[0], phi + h * k3[1], Nd + h * k3[2], x_cr + h * k3[3], k3[0])
    P_n = P + h / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
    phi_n = phi + h / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    Nd_n = Nd + h / 6.0 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
    x_n = x_cr + h / 6.0 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
    dpdz_n = (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6.0
    P_n, phi_n, Nd_n, x_n = _clamp_state(P_n, phi_n, Nd_n, x_n)
    return P_n, phi_n, Nd_n, x_n, dpdz_n, k1[4], k1[5], k1[6]


def _error_paso(P_a, phi_a, P_b, phi_b):
    eP = abs(P_a - P_b) / (1.0 + abs(P_b))
    ephi = float(np.max(np.abs(phi_a - phi_b)) / (1.0 + np.max(np.abs(phi_b))))
    return max(eP, ephi)


def _configurar(radius, Pressure, wt, Temperature, content_crystal, N_r, eps):
    _G["N_r"] = int(N_r)
    wr = float(radius)
    _G["wr"] = wr
    _G["r"] = np.linspace(1e-6, wr, N_r)
    _G["dr"] = float(_G["r"][1] - _G["r"][0])
    _G["T"] = float(Temperature)
    _G["Tc"] = Temperature - 273.15
    _G["h2o"] = float(wt)
    _G["co"] = wt / 100.0
    _G["xi"] = float(content_crystal)
    _G["xmax"] = float(xmax)
    _G["cg"] = 3 if geometry == "dyke" else 8
    _G["eps_phi"] = float(eps.get("eps_phi", EPS_PHI))
    _G["eps_frag"] = float(eps.get("eps_frag", EPS_FRAG))
    _G["eps_henry"] = float(eps.get("eps_henry", EPS_HENRY))
    _G["eps_re"] = float(eps.get("eps_re", EPS_RE))
    _G["eps_xi"] = float(eps.get("eps_xi", EPS_XI))
    _G["eps_rb"] = float(eps.get("eps_rb", EPS_RB))
    Pi = rcrust * g * abs(H) + float(Pressure)
    _G["Pi"] = Pi
    dis = min(C1 * Pi ** beta, _G["co"])
    _G["rho_m"] = density(sio2, tio2, al2o3, feo, mgo, cao, na2o, k2o, dis * 100, _G["Tc"], Pi / 1e6)
    exi = (1.0 - _G["xi"]) * (_G["co"] - C1 * Pi ** beta) / (1.0 - C1 * Pi ** beta)
    if exi <= 0:
        _G["rho_ti"] = _G["rho_m"]
    else:
        _G["rho_ti"] = Pi * _G["rho_m"] / (_G["rho_m"] * R * _G["T"] * exi + Pi * (1.0 - exi))
    _G["phicrit"] = 0.8
    _G["limphi1"] = limperl
    _G["limphi2"] = limperh
    _G["Nd0"] = 1e8
    return exi, Pi


def _actualizar_umbrales(vinicial):
    """φ1, φ2, φ_crit: la misma cuenta que RIconduitex5_5 (Ca de referencia)."""
    wr = _G["wr"]
    rho_m = _G["rho_m"]
    visc0 = _visc_nodo(_G["Pi"], 0.0, _G["xi"], max(vinicial, 1e-6), 1e8)
    dpdz0 = -_G["rho_ti"] * (g + _G["cg"] * visc0 * vinicial / (wr ** 2 * _G["rho_ti"]))
    dpdt0 = abs(dpdz0 * vinicial)
    Nd0 = float(np.clip(10 ** (1.5 * math.log10(max(dpdt0, 1e-30)) + 5), 1e6, 1e10))
    if C1 * _G["Pi"] ** beta < _G["co"]:
        Nd0 = 1e8
    _G["Nd0"] = Nd0

    phisel = 0.2
    Pmin, Pmax = pfinal, (_G["co"] / C1) ** (1.0 / beta)
    Pcalc = 0.5 * (Pmax + Pmin)
    phicalc = 0.2
    ncalc = 1e-3
    for _ in range(50):
        ncalc = (1.0 - _G["xi"]) * (_G["co"] - C1 * Pcalc ** beta) / max(1.0 - C1 * Pcalc ** beta, 1e-12)
        rhog = Pcalc / (R * _G["T"])
        phicalc = 1.0 / max((1.0 / max(ncalc, 1e-16) - 1.0) * rhog / rho_m + 1.0, 1e-12)
        if abs(phicalc - phisel) < 0.002:
            break
        if phicalc < phisel - 0.002:
            Pmax = Pcalc
        else:
            Pmin = Pcalc
        Pcalc = 0.5 * (Pmax + Pmin)

    dfgdp = (C1 * beta * Pcalc ** (beta - 1)) * (_G["co"] - 1.0) / max((1.0 - C1 * Pcalc ** beta) ** 2, 1e-30)
    drhogdp = 1.0 / (R * _G["T"])
    dphidp = -(
        -dfgdp * rhog / (rho_m * max(ncalc, 1e-16) ** 2)
        + (1.0 / max(ncalc, 1e-16) - 1.0) * drhogdp / rho_m
    ) / max(((1.0 / max(ncalc, 1e-16) - 1.0) * rhog / rho_m + 1.0) ** 2, 1e-30)
    viscalc = _visc_nodo(Pcalc, phicalc, _G["xi"], vinicial, Nd0)
    rho_ti_c = Pcalc * rho_m / (rho_m * R * _G["T"] * ncalc + Pcalc * (1.0 - ncalc))
    velc = math.sqrt(15e9 / rho_m)
    dpdz2 = (-rho_ti_c * (g + _G["cg"] * viscalc * vinicial / (wr ** 2 * rho_ti_c))) / max(
        1.0 - (vinicial ** 2) / (velc ** 2), 1e-8
    )
    dvdz_exp = (
        vinicial
        * (-dfgdp * dpdz2 * (1.0 - phicalc) + (1.0 - ncalc) * dphidp * dpdz2)
        / max((1.0 - phicalc) ** 2, 1e-30)
    )
    dvdz = 0.5 * (dvdz_exp + vinicial / wr)
    rbcalc = _rb(phicalc, Nd0)
    Ca = abs(dvdz * viscalc * rbcalc / 0.3)
    _G["Ca"] = float(Ca)
    _G["phicrit"] = ((lsup - linf) / 2.0) * math.erf(math.log10(max(Ca, 1e-30))) + (lsup + linf) / 2.0
    _G["limphi1"] = ((limperl - limperh) / 2.0) * math.erf(math.log10(max(Ca, 1e-30))) + (limperl + limperh) / 2.0
    _G["limphi2"] = _G["limphi1"] + 0.01


def march(vinicial, max_steps=2500, tol=1e-3, verbose=True):
    """
    Integra de z=H a z=0 para un vinicial. φ, N, ξ son campos en r.
    Paso adaptativo: h vs dos de h/2 (PDF, eq. del error).
    """
    N = _G["N_r"]
    _G["q"] = vinicial * _G["rho_ti"]
    _G["Q"] = _G["q"] * math.pi * _G["wr"] ** 2
    Pi, xi, co, T, rho_m = _G["Pi"], _G["xi"], _G["co"], _G["T"], _G["rho_m"]
    _actualizar_umbrales(vinicial)

    exi = (1.0 - xi) * (co - C1 * Pi ** beta) / (1.0 - C1 * Pi ** beta)
    z_hist, P_hist, phi_hist, Nd_hist, x_hist = [], [], [], [], []
    um_hist, ug_hist, Fmw_hist = [], [], []

    if exi <= 0:
        Pcrit = (co / C1) ** (1.0 / beta)
        visc_ini = _visc_nodo(Pi, 0.0, xi, max(vinicial, 1e-6), _G["Nd0"])
        dpdzcalc = -rho_m * g - _G["cg"] * visc_ini * vinicial / _G["wr"] ** 2
        Hi = H - (Pi - Pcrit) / dpdzcalc
        nH = max(int(abs(Hi - H) / 50.0), 8)
        z_pre = np.linspace(H, Hi, nH)
        for zz in z_pre[:-1]:
            P_k = Pi + (zz - H) * dpdzcalc
            um_k = 2.0 * vinicial * (1.0 - (_G["r"] / _G["wr"]) ** 2)
            um_k[-1] = 0.0
            z_hist.append(zz)
            P_hist.append(P_k)
            phi_hist.append(np.zeros(N))
            Nd_hist.append(np.full(N, _G["Nd0"]))
            x_hist.append(np.full(N, xi))
            um_hist.append(um_k)
            ug_hist.append(np.zeros(N))
            Fmw_hist.append(0.0)
        epsd = max(25.0, abs(Hi) / 200.0)
        z = Hi + epsd
        P = Pcrit + dpdzcalc * epsd
        fgi = max((1.0 - xi) * (co - C1 * P ** beta) / (1.0 - C1 * P ** beta), 1e-12)
        phi0 = np.full(N, max(_phi_alg(P, fgi), 1e-6))
        Nd0 = np.full(N, _G["Nd0"])
        x0 = np.full(N, xi)
        dpdz = dpdzcalc
    else:
        fgi = (1.0 - xi) * (co - C1 * Pi ** beta) / (1.0 - C1 * Pi ** beta)
        z = float(H)
        P = float(Pi)
        phi0 = np.full(N, _phi_alg(Pi, fgi))
        Nd0 = np.full(N, _G["Nd0"])
        x0 = np.full(N, xi)
        dpdz = -rho_m * g

    um0, _, Fmw0 = solve_radial(P, phi0, Nd0, x0, vinicial, dpdz, n_picard=3)
    ug0 = np.full(N, vinicial)
    z_hist.append(z)
    P_hist.append(P)
    phi_hist.append(phi0.copy())
    Nd_hist.append(Nd0.copy())
    x_hist.append(x0.copy())
    um_hist.append(um0)
    ug_hist.append(ug0)
    Fmw_hist.append(Fmw0)

    h = max(abs(z) / 400.0, 2.0)
    h_min, h_max = max(h / 80.0, 0.25), min(h * 8.0, 40.0)
    n_steps = n_rej = 0
    phi, Nd, x_cr = phi0, Nd0, x0

    while z < -1e-3 and n_steps < max_steps:
        if z + h > 0.0:
            h = -z
        P_f, phi_f, Nd_f, x_f, dp_f, _, _, _ = rk4_step(P, phi, Nd, x_cr, dpdz, h)
        P_m, phi_m, Nd_m, x_m, dp_m, _, _, _ = rk4_step(P, phi, Nd, x_cr, dpdz, h / 2.0)
        P_h, phi_h, Nd_h, x_h, dp_h, um_h, ug_h, Fmw_h = rk4_step(P_m, phi_m, Nd_m, x_m, dp_m, h / 2.0)
        P_f, phi_f, Nd_f, x_f = _clamp_state(P_f, phi_f, Nd_f, x_f)
        P_h, phi_h, Nd_h, x_h = _clamp_state(P_h, phi_h, Nd_h, x_h)
        phi_f = _proyectar_phi(P_f, phi_f, x_f)
        phi_h = _proyectar_phi(P_h, phi_h, x_h)
        err = _error_paso(P_f, phi_f, P_h, phi_h)
        if err > tol and h > h_min * 1.01:
            h = max(h * 0.5, h_min)
            n_rej += 1
            continue
        P, phi, Nd, x_cr = P_h, phi_h, Nd_h, x_h
        if float(np.max(phi)) > 1e-3:
            h_min = max(h_min, 2.0)
        dpdz = dp_h
        z = z + h
        n_steps += 1
        z_hist.append(z)
        P_hist.append(P)
        phi_hist.append(phi.copy())
        Nd_hist.append(Nd.copy())
        x_hist.append(x_cr.copy())
        um_hist.append(um_h)
        ug_hist.append(ug_h)
        Fmw_hist.append(Fmw_h)
        if err > 1e-12:
            h = float(np.clip(h * 0.9 * (tol / err) ** 0.5, h_min, h_max))
        else:
            h = min(h * 1.4, h_max)
        if P <= pfinal * 0.6:
            break

    if verbose:
        phi_a = float(phi_hist[-1][0])
        phi_w = float(phi_hist[-1][-2])
        print(
            f"  [phi_r] pasos={n_steps} rechazos={n_rej}  "
            f"z={z:.1f} m ({'boca' if z >= -1.0 else 'aún en el conducto'})  "
            f"P={P:.3e} Pa = {P/1e6:.2f} MPa  (P_atm={pfinal/1e6:.3f} MPa)  "
            f"phi_eje={phi_a:.4f} phi_pared={phi_w:.4f} dphi={phi_a - phi_w:.4e}"
        )
    out = {
        "z": np.asarray(z_hist),
        "P": np.asarray(P_hist),
        "phi": np.vstack(phi_hist),
        "Nd": np.vstack(Nd_hist),
        "xi": np.vstack(x_hist),
        "um": np.vstack(um_hist),
        "ug": np.vstack(ug_hist),
        "Fmw": np.asarray(Fmw_hist),
        "r": _G["r"].copy(),
        "n_steps": n_steps,
        "n_reject": n_rej,
        "vinicial": vinicial,
        "Q": _G["Q"],
        "phicrit": float(_G["phicrit"]),
        "limphi1": float(_G["limphi1"]),
        "limphi2": float(_G["limphi2"]),
        "Ca": float(_G.get("Ca", np.nan)),
        "Nd0": float(_G["Nd0"]),
    }
    return out


def RIconduit2D_phi_r_f(
    radius=16.0,
    Pressure=overP1,
    wt=4.0,
    Temperature=1243.15,
    content_crystal=0.25,
    N_r=12,
    max_steps=2500,
    max_count=1,
    tol=1e-3,
    verbose=True,
    **eps,
):
    """
    Directo 2D con φ(r,z). Por defecto un solo tiro (marcha).
    max_count > 1 activa bisección sobre vinicial (como el 2D viejo).
    """
    _configurar(radius, Pressure, wt, Temperature, content_crystal, N_r, eps)
    vinicial = 10.0
    vmin, vmax = 0.1, 50.0
    best, best_dist = None, 1e30
    for count in range(1, max_count + 1):
        if verbose:
            print(f"\nCount = {count}  vinicial = {vinicial:.5f}")
        try:
            out = march(vinicial, max_steps=max_steps, tol=tol, verbose=verbose)
        except Exception as exc:
            if verbose:
                print(f"  ERROR: {exc}")
            vmax = vinicial
            vinicial = 0.5 * (vmin + vmax)
            continue
        zexit, pexit = float(out["z"][-1]), float(out["P"][-1])
        dist = abs(zexit) + abs(pexit - pfinal) / 1e5
        if dist < best_dist:
            best_dist, best = dist, out
        if max_count == 1:
            break
        vsound = 0.99 * math.sqrt(R * _G["T"])
        ugexit = float(np.max(out["ug"][-1]))
        if zexit >= -15.0 and abs(pexit - pfinal) < 1.5e5:
            if verbose:
                print(">>> tiro aceptado")
            break
        if pexit < pfinal or ugexit > 1.02 * vsound:
            vmax = vinicial
        else:
            vmin = vinicial
        vinicial = 0.5 * (vmin + vmax)
        if vmax - vmin < 1e-4:
            break
    return best if best is not None else out


def radial_grid():
    return np.asarray(_G["r"]), float(_G["wr"]), int(_G["N_r"])


if __name__ == "__main__":
    out = RIconduit2D_phi_r_f(N_r=10, max_steps=80, max_count=1, verbose=True)
    dphi = out["phi"][:, 0] - out["phi"][:, -2]
    print(f"max |φ_eje - φ_pared| = {np.max(np.abs(dphi)):.4e}")
    print(f"nodos z = {out['z'].size}  r = {out['r'].size}")
