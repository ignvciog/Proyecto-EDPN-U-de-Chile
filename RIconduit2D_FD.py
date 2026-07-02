"""
RIconduit2D_FD.py
=================
Modelo de conducto 2D axisimétrico: diferencias finitas en r y z.
Sin solver DAE (sin IDA). Marcha explícita en z con RK4.

Ecuación radial del momento del fundido (cuasi-estática en r):
    (1/r) d/dr [ r mu(1-phi) du_m/dr ] = (1-phi) dP/dz + rho_m(1-phi) g - F_mg

Condiciones de borde: du_m/dr = 0 en r=0 (simetría), u_m(R) = 0 (no-desliz).

La fricción parietal F_mw sale del gradiente real du_m/dr|_{r=R}, no de HP.

Variables verticales: P(z), phi(z), Nd(z), x(z) — avanzadas con RK4.
Variables algebraicas: u_m_avg(z), u_g(z) — de conservación de masa en cada paso.
Campo 2D: u_m(r, z) — de la BVP radial en cada punto z.

Método de tiro: bisección + secante sobre vinicial.
"""

import numpy as np
import math
import warnings
warnings.filterwarnings("ignore")

from density import density
from viscosity import viscosity
from fvrel import fvrel
from calbuco2015d import (
    sio2, tio2, al2o3, feo, mno, mgo, cao, na2o, k2o, p2o5, f2o,
    C1, beta, R, Patm, xmax, lsup, linf, limperh, limperl,
    tcar, model, ar1, ar2, F1, F2, Fc, g, H, geometry, rcrust,
    errtol, pfinal, overP1, radius1, h2o1, T1, xi1,
)

# ── parámetros globales del run ────────────────────────────────────────────────
_G = {
    "wr": None, "T": None, "Tc": None, "rho_m": None, "q": None, "Q": None,
    "cg": None, "h2o": None, "co": None, "xi": None, "xmax_run": xmax,
    "phicrit": 0.8, "limphi1": limperl, "limphi2": limperh,
    "Nd0": 1e8, "xfinal": None, "fragcrit": None,
    "N_r": 20, "N_z": 3000,
    "r": None, "dr": None,
}


# ══════════════════════════════════════════════════════════════════════════════
# Física local (todo explícito, sin globals del módulo externo)
# ══════════════════════════════════════════════════════════════════════════════

def _fg(P, x_cr):
    """Fracción másica de gas exsuelta."""
    P = max(float(np.real(P)), 1e4)   # protección contra P complejo/negativo
    test = (1 - _G["xi"]) * _G["co"] - (1 - x_cr) * C1 * P ** beta
    if test <= 0:
        return 0.0
    return float(((1 - _G["xi"]) * _G["co"] - C1 * (1 - x_cr) * P ** beta) / (1 - C1 * P ** beta))


def _dfgdp(P, x_cr, fg, um_avg):
    """Derivada de fg respecto a P (incluye cristalización cinética)."""
    P = max(float(np.real(P)), 1e4)
    if fg <= 0:
        return 0.0
    f2 = max(0.0, (_G["co"] * (1 - _G["xi"]) - (1 - x_cr) * C1 * P ** beta)
             / (_G["co"] * (1 - _G["xi"]) - (1 - _G["xmax_run"]) * C1 * Patm ** beta))
    xteo = _G["xi"] + (_G["xmax_run"] - _G["xi"]) * f2
    f3 = max(0.0, 1 - x_cr / xteo)
    dxdp = max(0.0, (_G["xmax_run"] - _G["xi"]) * f2 * f3 / (tcar * max(um_avg, 1e-6)))
    return (
        -(-dxdp * C1 * P ** beta + (1 - x_cr) * C1 * beta * P ** (beta - 1))
        + (_G["co"] * (1 - _G["xi"]) - (1 - x_cr) * C1 * P ** beta) * C1 * beta * P ** (beta - 1)
    ) / (1 - C1 * P ** beta) ** 2


def _rho_g(P):
    return max(float(np.real(P)), 1e4) / (R * _G["T"])


def _effective_visc(P, phi, x_cr, um_avg, Nd):
    """Viscosidad efectiva (fundido + burbujas) de Kozono."""
    wr, Tc = _G["wr"], _G["Tc"]
    fg = _fg(P, x_cr)
    h2o_dis = C1 * P ** beta * 100 if fg > 0 else _G["h2o"]
    viscl = fvrel(model, x_cr, _G["xi"], ar1, ar2, _G["xmax_run"], um_avg / wr) * viscosity(
        sio2, tio2, al2o3, feo, mno, mgo, cao, na2o, k2o, p2o5, h2o_dis, f2o, Tc
    )
    phi_c = float(np.clip(phi, 1e-12, _G["phicrit"] - 1e-6))
    Nd_s = max(float(Nd), 1.0)
    rb = (phi_c / ((4.0 / 3) * math.pi * Nd_s * (1 - phi_c))) ** (1.0 / 3)
    nca = rb * viscl * max(um_avg / wr, 1e-12) / 3
    phicritbub = _G["phicrit"] + 0.05
    AA = (1 - phi_c / phicritbub) ** (-phicritbub)
    BB = (1 - phi_c / phicritbub) ** (5 * phicritbub / 3)
    c1c = -0.2895 * phi_c + 0.8132
    c2c = phi_c
    viscrel = 0.5 * (AA - BB) * (1 - math.erf(float(np.real(c1c * math.log(max(nca, 1e-30)) + c2c)))) + BB
    return float(np.real(viscrel * viscl))


def _Fmg(um_avg, ug, phi, Nd, rho_g_val, visc):
    """Arrastre melt-gas (Kozono, régimen según phi)."""
    limphi1, limphi2 = _G["limphi1"], _G["limphi2"]
    phi_c = float(np.clip(phi, 1e-12, _G["phicrit"] + 0.04))
    Nd_s = max(float(Nd), 1.0)
    rb = (phi_c / ((4.0 / 3) * math.pi * Nd_s * (1 - phi_c))) ** (1.0 / 3)
    slip = ug - um_avg

    if phi < limphi1:
        return 3 * visc * slip * phi_c * (1 - phi_c) / (rb ** 2)
    elif phi < limphi2:
        tt = (phi - limphi1) / (limphi2 - limphi1)
        Re = 2 * rb * rho_g_val * abs(slip) / 1e-5
        if Re > 2200:
            return ((0.33 / (4 * rb)) * rho_g_val * abs(slip) ** tt) * (
                (3 * visc / rb ** 2) ** (1 - tt)) * slip * phi_c * (1 - phi_c)
        else:
            kper = 0.131 * rb ** 2 * (phi - limphi1 + 0.05) ** 2.1
            return ((1e-5 / kper) ** tt) * ((3 * visc / rb ** 2) ** (1 - tt)) * slip * phi_c * (1 - phi_c)
    else:
        Re = 2 * rb * rho_g_val * abs(slip) / 1e-5
        if Re > 2200:
            return (0.33 / (4 * rb)) * rho_g_val * slip * abs(slip) * phi_c * (1 - phi_c)
        else:
            kper = 0.131 * rb ** 2 * (phi - limphi1 + 0.05) ** 2.1
            return (1e-5 / kper) * slip * phi_c * (1 - phi_c)


def _Fmg_frag(um_avg, ug, phi, rho_g_val):
    """Arrastre en régimen fragmentado."""
    slip = ug - um_avg
    ra, cd = 1e-3, 0.8
    if phi < _G["phicrit"] + 0.05:
        tt = (phi - _G["phicrit"]) / 0.05
        return ((0.33 / (4 * ra)) ** (1 - tt)) * ((3 * cd / (8 * ra)) ** tt) * rho_g_val * abs(slip) * slip * phi * (1 - phi)
    else:
        return 3 * cd * rho_g_val * abs(slip) * slip * phi * (1 - phi) / (8 * ra)


# ══════════════════════════════════════════════════════════════════════════════
# Solver radial FD (Thomas + Picard)
# ══════════════════════════════════════════════════════════════════════════════

def _thomas(lo, di, up, rhs):
    """Eliminación de Thomas para tridiagonal."""
    n = len(di)
    c = up.copy(); d = rhs.copy(); b = di.copy(); L = lo.copy()
    for i in range(1, n):
        w = L[i] / b[i - 1]
        b[i] -= w * c[i - 1]
        d[i] -= w * d[i - 1]
    x = np.zeros(n)
    x[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
    return x


def _radial_tridiag(phi, mu_arr):
    """Tridiagonal para (1/r) d/dr[r mu(1-phi) du/dr] = RHS con u[N-1]=0."""
    r, dr, N = _G["r"], _G["dr"], _G["N_r"]
    nunk = N - 1
    lo = np.zeros(nunk)
    di = np.zeros(nunk)
    up = np.zeros(nunk)
    fp = 1.0 - phi  # factor (1-phi)

    # nodo central (simetría: du/dr=0 en r[0]≈0)
    mu_h = 0.5 * (mu_arr[0] + mu_arr[1])
    di[0] = -2.0 * fp * mu_h / dr ** 2
    up[0] = 2.0 * fp * mu_h / dr ** 2

    for i in range(1, nunk - 1):
        mu_p = 0.5 * (mu_arr[i] + mu_arr[i + 1])
        mu_m = 0.5 * (mu_arr[i] + mu_arr[i - 1])
        lo[i] = fp * (r[i] - dr / 2) * mu_m / (r[i] * dr ** 2)
        di[i] = -fp * ((r[i] + dr / 2) * mu_p + (r[i] - dr / 2) * mu_m) / (r[i] * dr ** 2)
        up[i] = fp * (r[i] + dr / 2) * mu_p / (r[i] * dr ** 2)

    # nodo junto a la pared
    i = nunk - 1
    mu_p = 0.5 * (mu_arr[i] + mu_arr[i + 1])
    mu_m = 0.5 * (mu_arr[i] + mu_arr[i - 1])
    lo[i] = fp * (r[i] - dr / 2) * mu_m / (r[i] * dr ** 2)
    di[i] = -fp * ((r[i] + dr / 2) * mu_p + (r[i] - dr / 2) * mu_m) / (r[i] * dr ** 2)
    # up[i] * u[N-1] = 0 → ya absorbido en RHS (u[N-1]=0)

    return lo, di, up


def solve_radial_gas(P, dpdz, ug_scale=None):
    """
    Perfil radial u_g(r) del gas en régimen fragmentado.
    Gas newtoniano (μ_g constante) → no necesita Picard.
    Parámetro ug_scale: si se da, escala el perfil para que su media coincida.
    Devuelve (ug_arr, ug_avg, Fgw).
    """
    r, dr, N = _G["r"], _G["dr"], _G["N_r"]
    wr    = _G["wr"]
    mu_g  = 1.8e-5                      # viscosidad vapor de agua ~1000 °C (Pa·s)
    rg    = _rho_g(P)
    drive = -(dpdz + rg * g)            # fuerza motriz neta (> 0 → flujo ascendente)

    if drive <= 0.0:
        # Presión insuficiente para flujo — retornar perfil nulo y Fgw de respaldo
        Fgw_fallback = 0.01 * rg * abs(ug_scale or 1.0) ** 2 / (4 * wr)
        return np.zeros(N), 0.0, float(Fgw_fallback)

    mu_arr = np.full(N, mu_g)
    rhs    = np.full(N - 1, -drive)     # LHS = rhs → signo negativo por convención
    lo, di, up = _radial_tridiag(0.0, mu_arr)   # phi=0: gas ocupa todo el conducto

    ug_arr = np.zeros(N)
    ug_arr[:N - 1] = _thomas(lo, di, up, rhs)
    ug_arr[-1] = 0.0
    ug_arr = np.maximum(ug_arr, 0.0)

    ug_avg = float(np.trapezoid(ug_arr * 2 * math.pi * r, r) / (math.pi * wr ** 2))

    # Escalar para conservar masa (igual que se hace para u_m)
    if ug_scale is not None and ug_avg > 1e-10:
        ug_arr = ug_arr * (ug_scale / ug_avg)
        ug_arr[-1] = 0.0
        ug_avg = float(ug_scale)

    dudr_w = (ug_arr[-1] - ug_arr[-2]) / dr     # < 0
    Fgw    = -2.0 * mu_g * dudr_w / wr           # fricción gas-pared (> 0)

    return ug_arr, ug_avg, float(Fgw)


def solve_radial(P, phi, Nd, x_cr, ug, dpdz, n_picard=5):
    """
    Resuelve la BVP radial del momento del fundido con FD + Picard.
    Devuelve (um_array [N_r], um_avg, F_mw_real).
    """
    r, dr, N = _G["r"], _G["dr"], _G["N_r"]
    wr = _G["wr"]
    rg = _rho_g(P)

    # inicialización con perfil parabólico
    drive = max(-(dpdz + _G["rho_m"] * g), 0.0)
    um = drive / max(4e3, 1e-6) * (wr ** 2 - r ** 2)
    um[-1] = 0.0
    um = np.maximum(um, 0.0)

    for _ in range(n_picard):
        um_avg_loc = float(np.trapezoid(um * 2 * math.pi * r, r) / (math.pi * wr ** 2))
        mu_arr = np.array([
            _effective_visc(P, phi, x_cr, max(um[i], 1e-9), Nd) for i in range(N)
        ])
        Fmg_arr = np.array([
            _Fmg(max(um[i], 1e-9), ug, phi, Nd, rg, mu_arr[i]) for i in range(N)
        ])
        rhs = (1.0 - phi) * dpdz + _G["rho_m"] * (1.0 - phi) * g - Fmg_arr

        lo, di, up = _radial_tridiag(phi, mu_arr)
        # contribución del nodo de pared (u=0) al nodo N-2: up[-1]*0 → 0
        um_new = np.zeros(N)
        um_new[:N - 1] = _thomas(lo, di, up, rhs[:N - 1])
        um_new[-1] = 0.0
        um_new = np.maximum(um_new, 0.0)
        um = um_new

    um_avg = float(np.trapezoid(um * 2 * math.pi * r, r) / (math.pi * wr ** 2))

    # Fricción parietal real desde gradiente en la pared
    mu_wall = _effective_visc(P, phi, x_cr, max(um[-2], 1e-9), Nd)
    dudr_wall = (um[-1] - um[-2]) / dr  # < 0
    F_mw = -2.0 * (1.0 - phi) * mu_wall * dudr_wall / wr   # > 0

    return um, um_avg, float(F_mw)


def _pure_melt_profile(P, dpdz, um_target, Nd, x_cr):
    """
    Perfil radial u_m(r) en fundido puro (φ=0).
    Escala solve_radial para que la media coincida con um_target.
    """
    N = _G["N_r"]
    um_arr, um_avg_fd, _ = solve_radial(P, 0.0, Nd, x_cr, um_target, dpdz, n_picard=3)
    if um_avg_fd > 1e-10:
        um_arr = um_arr * (um_target / um_avg_fd)
        um_arr[-1] = 0.0
    else:
        r, wr = _G["r"], _G["wr"]
        um_arr = 2.0 * um_target * (1.0 - (r / wr) ** 2)
        um_arr[-1] = 0.0
    return um_arr


# ══════════════════════════════════════════════════════════════════════════════
# Derivadas verticales (la "f" del sistema ODE en z)
# ══════════════════════════════════════════════════════════════════════════════

def _derivs(P, phi, Nd, x_cr, dpdz_prev, frag=False):
    """
    Calcula (dP/dz, dphi/dz, dNd/dz, dx/dz, um_avg, ug, um_arr, F_mw).
    Usa Picard interno para consistencia dpdz ↔ u_m(r).
    """
    q = _G["q"]
    rho_m = _G["rho_m"]
    wr = _G["wr"]
    rg = _rho_g(P)

    if frag:
        # ── régimen fragmentado (n_eq=4) ────────────────────────────────────
        fg = _fg(P, _G["xfinal"])
        x_cr_f = _G["xfinal"]
        um_avg = (1 - fg) * q / (rho_m * max(1 - phi, 1e-9))
        ug = fg * q / (rg * max(phi, 1e-9))
        Nd_loc = _G.get("Nd_final", Nd)
        visc = _effective_visc(P, phi, x_cr_f, max(um_avg, 1e-9), Nd_loc)
        F_mw = 0.0  # fundido no toca la pared en régimen fragmentado
        Fmg  = _Fmg_frag(um_avg, ug, phi, rg)

        # ── Perfil radial del gas (FD 2D) — nuevo respecto a Kozono 1D ──────
        # Usamos dpdz_prev como estimación inicial; el gas es newtoniano → no Picard
        ug_arr_frag, _, Fgw = solve_radial_gas(P, dpdz_prev, ug_scale=ug)

        f2 = max(0.0, ((1-_G["xi"])*_G["co"] - (1-x_cr_f)*C1*P**beta)
                 / ((1-_G["xi"])*_G["co"] - (1-_G["xmax_run"])*C1*Patm**beta))
        xteo = _G["xi"] + (_G["xmax_run"] - _G["xi"]) * f2
        f3 = max(0.0, 1 - x_cr_f / max(xteo, 1e-9))
        dxdp = max(0.0, (_G["xmax_run"] - _G["xi"]) * f2 * f3 / tcar)
        dfgdp_val = (
            -(-dxdp * C1 * P**beta + (1-x_cr_f) * C1 * beta * P**(beta-1))
            + ((1-_G["xi"])*_G["co"] - (1-x_cr_f)*C1*P**beta) * C1*beta*P**(beta-1)
        ) / (1 - C1*P**beta)**2

        aco = rg * ug**2
        bco = phi - ug**2 * phi / (R * _G["T"]) + dfgdp_val * q * ug
        cco = rho_m * um_avg**2
        dco = dfgdp_val * q * um_avg - (1 - phi)
        eco = -rho_m * (1 - phi) * g + Fmg - F_mw
        fco = rg * phi * g + Fmg + Fgw
        det = aco * dco - bco * cco
        if abs(det) < 1e-30:
            return 0, 0, 0, 0, um_avg, ug, ug_arr_frag, F_mw
        dpdz = (-eco * aco + fco * cco) / det
        dphidz = (-eco * bco + dco * fco) / det
        return dpdz, dphidz, 0.0, 0.0, um_avg, ug, ug_arr_frag, F_mw

    # ── régimen no fragmentado — HEM + FD radial 2D ─────────────────────────
    # Modelo de equilibrio homogéneo: ug = um_avg (no slip).
    # Evita la inestabilidad ug→∞ del modelo bifásico separado cuando φ→0.
    # La contribución 2D novedosa está en el perfil radial u_m(r) via FD.
    P = max(float(P), 1e4)
    fg = _fg(P, x_cr)

    # Fracción volumétrica de gas desde la ecuación de estado algebraica
    if fg <= 0:
        phi_alg = 0.0
    else:
        phi_alg = 1.0 / (1.0 + (P / (fg * R * _G["T"])) * (1.0 - fg) / rho_m)
    phi_alg = float(np.clip(phi_alg, 0.0, _G["phicrit"] - 1e-6))

    # Densidad de mezcla y velocidad HEM
    rho_mix = rg * phi_alg + rho_m * (1.0 - phi_alg)
    um_mix  = q / max(rho_mix, 1.0)        # = ug = um_avg (no slip)

    # Picard: dpdz ↔ perfil radial FD 2D
    dpdz  = dpdz_prev
    um_arr = None
    F_mw  = 0.0
    dr    = _G["dr"]

    for _ in range(4):
        um_arr, um_avg_fd, F_mw = solve_radial(P, phi_alg, Nd, x_cr, um_mix, dpdz)
        # Escalar para conservar masa
        if um_avg_fd > 1e-10:
            um_arr = um_arr * (um_mix / um_avg_fd)
            um_arr[-1] = 0.0
        # F_mw desde perfil escalado
        mu_wall  = _effective_visc(P, phi_alg, x_cr, max(float(um_arr[-2]), 1e-9), Nd)
        dudr_w   = (um_arr[-1] - um_arr[-2]) / dr   # < 0
        F_mw     = -2.0 * (1.0 - phi_alg) * mu_wall * dudr_w / wr   # > 0
        # Momentum de mezcla HEM: dP/dz = -(rho_mix·g + F_mw)
        dpdz_new = -(rho_mix * g + F_mw)
        if abs(dpdz_new - dpdz) < 1e-4 * max(abs(dpdz), 1.0):
            dpdz = dpdz_new
            break
        dpdz = dpdz_new

    um_avg = um_mix
    ug     = um_mix   # HEM

    # dφ/dz por diferenciación numérica de la ecuación de estado
    eps_P  = max(abs(P) * 1e-6, 100.0)
    fg_ep  = _fg(P + eps_P, x_cr)
    phi_ep = (1.0 / (1.0 + ((P + eps_P) / (fg_ep * R * _G["T"])) * (1.0 - fg_ep) / rho_m)
              if fg_ep > 0 else 0.0)
    phi_ep = float(np.clip(phi_ep, 0.0, _G["phicrit"]))
    dphidz = (phi_ep - phi_alg) / eps_P * dpdz

    visc   = _effective_visc(P, phi_alg, x_cr, max(um_avg, 1e-9), Nd)

    # Criterio de fragmentación
    dvdz   = abs(dpdz) / max(rho_mix, 1.0)
    fragcrit_loc = dvdz * visc / (0.01 * 1e10)
    _G["fragcrit"] = fragcrit_loc

    # dNd/dz  (usar phi_alg consistente con P)
    phi_c = float(np.clip(phi_alg, 1e-12, _G["phicrit"] - 1e-6))
    Nd_s = max(float(Nd), 1.0)
    rb = (phi_c / ((4.0 / 3) * math.pi * Nd_s * (1 - phi_c))) ** (1.0 / 3)
    lim = 0.5
    if rb < lim * _G["wr"] and phi_alg < min(0.5, _G["phicrit"]):
        dNdt = -(Nd_s ** (2/3)) * ((1/(1-phi_c)) ** (1/3)) * ((1/9) * (_G["rho_m"] - rg) * 9.81 / visc) \
               * ((3*phi_c/(4*math.pi)) ** (2/3)) * (F1**2 - F2**2) \
               / (1 - (F1+F2) * ((3*phi_c*(math.pi/(6*_G["phicrit"]))/(4*math.pi)) ** (1/3))) \
               * Fc * (1 - phi_c / _G["phicrit"]) * (_G["wr"] - rb) / _G["wr"]
        dNdz = dNdt / max(um_avg, 1e-9)
    else:
        dNdz = 0.0

    # dx/dz (cristalización cinética)
    f2 = max(0.0, (_G["co"]*(1-_G["xi"]) - (1-x_cr)*C1*P**beta)
             / (_G["co"]*(1-_G["xi"]) - (1-_G["xmax_run"])*C1*Patm**beta))
    xteo = _G["xi"] + (_G["xmax_run"] - _G["xi"]) * f2
    f3 = max(0.0, 1 - x_cr / max(xteo, 1e-9))
    dxdz = max(0.0, (_G["xmax_run"] - _G["xi"]) * f2 * f3 / (tcar * max(um_avg, 1e-9)))

    return float(dpdz), float(dphidz), float(dNdz), float(dxdz), float(um_avg), float(ug), um_arr, float(F_mw)


# ══════════════════════════════════════════════════════════════════════════════
# Marcha en z con RK4 (desde z=H hasta z=0)
# ══════════════════════════════════════════════════════════════════════════════

def _rk4_step(S, dpdz_prev, dz, frag=False):
    """
    Un paso RK4 en z.
    S = (P, phi, Nd, x_cr)
    Devuelve (S_new, dpdz_new, um_arr, um_avg, ug, F_mw, k1_vec).
    k1_vec = (k1_P, k1_ph, k1_Nd, k1_x) — para control de paso adaptativo.
    """
    P, phi, Nd, x_cr = S

    def f(P_, phi_, Nd_, x_, dpg):
        P_ = max(float(np.real(P_)), 1e4)
        phi_ = float(np.clip(np.real(phi_), 1e-12, _G["phicrit"] + 0.1))
        Nd_ = max(float(np.real(Nd_)), 1.0)
        x_ = float(np.clip(np.real(x_), _G["xi"], _G["xmax_run"]))
        res = _derivs(P_, phi_, Nd_, x_, float(np.real(dpg)), frag=frag)
        dP, dph, dNd, dxc = res[0], res[1], res[2], res[3]
        if dP is None:
            return 0.0, 0.0, 0.0, 0.0, res[4], res[5], res[6], res[7]
        return float(np.real(dP)), float(np.real(dph)), float(np.real(dNd)), float(np.real(dxc)), res[4], res[5], res[6], res[7]

    k1_P, k1_ph, k1_Nd, k1_x, um1, ug1, um_arr1, fmw1 = f(P, phi, Nd, x_cr, dpdz_prev)
    k2_P, k2_ph, k2_Nd, k2_x, um2, ug2, um_arr2, fmw2 = f(
        P + 0.5*dz*k1_P, phi + 0.5*dz*k1_ph, Nd + 0.5*dz*k1_Nd, x_cr + 0.5*dz*k1_x, k1_P)
    k3_P, k3_ph, k3_Nd, k3_x, um3, ug3, um_arr3, fmw3 = f(
        P + 0.5*dz*k2_P, phi + 0.5*dz*k2_ph, Nd + 0.5*dz*k2_Nd, x_cr + 0.5*dz*k2_x, k2_P)
    k4_P, k4_ph, k4_Nd, k4_x, um4, ug4, um_arr4, fmw4 = f(
        P + dz*k3_P, phi + dz*k3_ph, Nd + dz*k3_Nd, x_cr + dz*k3_x, k3_P)

    P_new = P + dz / 6 * (k1_P + 2*k2_P + 2*k3_P + k4_P)
    phi_new = phi + dz / 6 * (k1_ph + 2*k2_ph + 2*k3_ph + k4_ph)
    Nd_new = Nd + dz / 6 * (k1_Nd + 2*k2_Nd + 2*k3_Nd + k4_Nd)
    x_new = x_cr + dz / 6 * (k1_x + 2*k2_x + 2*k3_x + k4_x)
    dpdz_new = (k1_P + 2*k2_P + 2*k3_P + k4_P) / 6

    um_arr_out = um_arr1 if um_arr1 is not None else um_arr2
    um_avg_out = (um1 + um2 + um3 + um4) / 4
    ug_out = (ug1 + ug2 + ug3 + ug4) / 4
    F_mw_out = (fmw1 + fmw2 + fmw3 + fmw4) / 4
    k1_vec = (k1_P, k1_ph, k1_Nd, k1_x)

    return (P_new, phi_new, Nd_new, x_new), dpdz_new, um_arr_out, um_avg_out, ug_out, F_mw_out, k1_vec


# ══════════════════════════════════════════════════════════════════════════════
# Integración de una trayectoria desde z=H
# ══════════════════════════════════════════════════════════════════════════════

def _march(vinicial):
    """
    Integra desde z=H hasta z=0 para una velocidad inicial dada.
    Devuelve (zsol, sol) donde sol tiene columnas [P, phi, Nd, x, um_avg, ug, um_r...].
    """
    # Inicializar estado en z=H
    rho_m = _G["rho_m"]
    Pi = _G["Pi"]
    xi = _G["xi"]
    co = _G["co"]
    T = _G["T"]
    wr = _G["wr"]
    N_r = _G["N_r"]
    N_z = _G["N_z"]
    r = _G["r"]

    exi = (1 - xi) * (co - C1 * Pi ** beta) / (1 - C1 * Pi ** beta)

    if exi <= 0:
        # zona sin gas hasta profundidad de saturación
        Pcrit = (co / C1) ** (1 / beta)
        visc_ini = _effective_visc(Pi, 0.0, xi, max(vinicial, 1e-6), _G["Nd0"])
        dpdzcalc = -rho_m * g - _G["cg"] * visc_ini * vinicial / wr**2
        deltH = -(Pi - Pcrit) / dpdzcalc
        Hi = H + deltH  # profundidad de saturación (negativa)

        nH = max(int(abs(deltH) / abs(H) * N_z), 10)
        z_pre = np.linspace(H, Hi, nH)
        sol_pre = np.zeros((nH, N_r + 6))
        for k, zz in enumerate(z_pre):
            P_k = Pi + (zz - H) * dpdzcalc
            um_r = _pure_melt_profile(P_k, dpdzcalc, vinicial, _G["Nd0"], xi)
            sol_pre[k, 0] = P_k
            sol_pre[k, 1] = 0.0
            sol_pre[k, 2] = _G["Nd0"]
            sol_pre[k, 3] = xi
            sol_pre[k, 4] = vinicial
            sol_pre[k, 5] = 0.0          # sin gas en pre-saturación
            sol_pre[k, 6:] = um_r

        # Punto adicional 1 m por encima de Hi para que P < Pcrit → fg > 0
        epsd = max(1.0, abs(Hi) / 1000.0)
        z_start = Hi + epsd
        P_start = Pcrit + dpdzcalc * epsd   # < Pcrit porque dpdzcalc < 0

        fgi = (1 - xi) * (co - C1 * P_start ** beta) / (1 - C1 * P_start ** beta)
        fgi = max(fgi, 1e-12)
        rg_start = P_start / (R * T)
        phi_start = 1.0 / (1.0 + (P_start / (fgi * R * T)) * (1 - fgi) / rho_m)
        phi_start = max(phi_start, 1e-9)

        # Estimar Nd a partir de dp/dt
        visc_s = _effective_visc(P_start, phi_start, xi, max(vinicial, 1e-6), _G["Nd0"])
        dpdt_s = abs(dpdzcalc * vinicial)
        Nd_start = float(np.clip(10 ** (1.5 * math.log10(max(dpdt_s, 1e-30)) + 5), 1e6, 1e10))
        x_start = xi
    else:
        fgi = (1 - xi) * (co - C1 * Pi ** beta) / (1 - C1 * Pi ** beta)
        phi_start = 1.0 / (1.0 + (Pi / (fgi * R * T)) * (1 - fgi) / rho_m)
        P_start = Pi
        Nd_start = _G["Nd0"]
        x_start = xi
        sol_pre = np.zeros((1, N_r + 6))
        dpdz_in = -rho_m * g
        um_r_in, um_avg_fd, _ = solve_radial(Pi, phi_start, Nd_start, x_start, vinicial, dpdz_in, n_picard=3)
        if um_avg_fd > 1e-10:
            um_r_in = um_r_in * (vinicial / um_avg_fd)
            um_r_in[-1] = 0.0
        sol_pre[0, 0] = Pi
        sol_pre[0, 1] = phi_start
        sol_pre[0, 2] = Nd_start
        sol_pre[0, 3] = x_start
        sol_pre[0, 4] = vinicial
        sol_pre[0, 5] = vinicial
        sol_pre[0, 6:] = um_r_in
        z_pre = np.array([H])
        z_start = H

    # Estimar dpdz inicial
    q = _G["q"]
    fg_start = _fg(P_start, x_start)
    um_avg0 = (1 - fg_start) * q / (rho_m * max(1 - phi_start, 1e-6)) if fg_start >= 0 else vinicial
    ug0 = fg_start * q / (_rho_g(P_start) * max(phi_start, 1e-9)) if fg_start > 0 else vinicial
    visc0 = _effective_visc(P_start, phi_start, x_start, max(um_avg0, 1e-6), Nd_start)
    dpdz_cur = -rho_m * g * (1 + _G["cg"] * visc0 / (wr**2 * rho_m * g) * um_avg0)

    S = (P_start, phi_start, Nd_start, x_start)

    # Perfil radial inicial en z_start (continuidad con pre-saturación)
    if phi_start < 1e-8:
        um_r0 = _pure_melt_profile(P_start, dpdz_cur, um_avg0, Nd_start, x_start)
    else:
        um_r0, um_avg_fd0, _ = solve_radial(P_start, phi_start, Nd_start, x_start, ug0, dpdz_cur, n_picard=3)
        if um_avg_fd0 > 1e-10:
            um_r0 = um_r0 * (um_avg0 / um_avg_fd0)
            um_r0[-1] = 0.0
        else:
            um_r0 = np.full(N_r, um_avg0)

    # ── paso adaptativo tipo RK4+Euler embebido ──────────────────────────────
    # dz_min / dz_max: límites de seguridad
    # tol_rel: tolerancia relativa sobre P (la variable más grande en magnitud)
    dz_ini = abs(z_start) / N_z if z_start < 0 else abs(H) / N_z
    dz_min = dz_ini / 100.0
    dz_max = dz_ini * 10.0
    tol_rel = 1e-3   # error relativo permitido entre paso completo y dos medios pasos
    dz = dz_ini

    z_arr = [z_start]
    P_arr = [P_start]; phi_arr = [phi_start]; Nd_arr = [Nd_start]; x_arr = [x_start]
    um_arr_list = [um_r0]; um_avg_arr = [um_avg0]; ug_arr = [ug0]
    Fmw_arr = [0.0]

    frag = False
    _G["xfinal"] = x_start

    z = z_start
    n_steps = 0
    n_reject = 0

    while z < 0:
        P, phi, Nd, x_cr = S
        if phi >= _G["phicrit"] and not frag:
            frag = True
            _G["xfinal"] = x_cr
            _G["Nd_final"] = Nd

        # no sobrepasar z=0
        if z + dz > 0:
            dz = -z + 1e-3

        # ── paso RK4 completo ────────────────────────────────────────────────
        S_full, dpdz_new, um_r, um_avg, ug, F_mw, k1 = _rk4_step(S, dpdz_cur, dz, frag=frag)

        # ── estimación del error: comparar con dos medios pasos ─────────────
        S_h1, _, _, _, _, _, _ = _rk4_step(S, dpdz_cur, dz / 2, frag=frag)
        S_half, _, um_r2, um_avg2, ug2, fmw2, _ = _rk4_step(S_h1, dpdz_cur, dz / 2, frag=frag)

        # error relativo en P (más sensible)
        P_full, P_half = S_full[0], S_half[0]
        scale = max(abs(P_full), abs(P_half), 1e3)
        err = abs(P_full - P_half) / scale

        if err > tol_rel and dz > dz_min * 1.01:
            # rechazar paso y reducir dz
            dz = max(dz * 0.5, dz_min)
            n_reject += 1
            continue

        # aceptar: usar el resultado de dos medios pasos (más preciso, Richardson)
        P_n, phi_n, Nd_n, x_n = S_half
        um_r_out = um_r2 if um_r2 is not None else um_r
        um_avg_out = um_avg2
        ug_out = ug2
        F_mw_out = fmw2

        # sanitizar
        P_n = max(P_n, pfinal * 0.5)
        Nd_n = max(Nd_n, 1.0)
        x_n = float(np.clip(x_n, _G["xi"], _G["xmax_run"]))

        # En régimen no-fragmentado, imponer φ algebraico (HEM) para evitar deriva
        if not frag:
            fg_n = _fg(P_n, x_n)
            if fg_n > 0:
                phi_n = 1.0 / (1.0 + (P_n / (fg_n * R * _G["T"])) * (1.0 - fg_n) / _G["rho_m"])
            else:
                phi_n = 0.0
            phi_n = float(np.clip(phi_n, 0.0, _G["phicrit"] - 1e-6))
        else:
            phi_n = float(np.clip(phi_n, 0.0, 1.0))

        S = (P_n, phi_n, Nd_n, x_n)
        dpdz_cur = dpdz_new
        z = z + dz
        n_steps += 1

        z_arr.append(z)
        P_arr.append(P_n); phi_arr.append(phi_n); Nd_arr.append(Nd_n); x_arr.append(x_n)
        um_arr_list.append(um_r_out if um_r_out is not None else np.zeros(N_r))
        um_avg_arr.append(um_avg_out); ug_arr.append(ug_out)
        Fmw_arr.append(F_mw_out)

        # ajustar dz para el próximo paso (control PI básico)
        if err > 1e-10:
            factor = 0.9 * (tol_rel / err) ** 0.25
            dz = float(np.clip(dz * factor, dz_min, dz_max))
        else:
            dz = min(dz * 1.5, dz_max)

        # Detener si alcanzamos presión atmosférica o fragmentación lejana
        if P_n <= pfinal and phi_n >= _G["phicrit"]:
            break
        if P_n <= pfinal * 0.5 and phi_n < _G["phicrit"]:
            break
        if z >= 0:
            break

    print(f"  [march] pasos={n_steps}  rechazos={n_reject}  dz_final={dz:.2f} m")

    # Ensamblar sol
    n = len(z_arr)
    sol = np.zeros((n, N_r + 6))
    sol[:, 0] = P_arr
    sol[:, 1] = phi_arr
    sol[:, 2] = Nd_arr
    sol[:, 3] = x_arr
    sol[:, 4] = um_avg_arr
    sol[:, 5] = ug_arr
    for k in range(n):
        if um_arr_list[k] is not None:
            sol[k, 6:] = um_arr_list[k]

    zsol = np.array(z_arr)
    if exi <= 0 and sol_pre.shape[0] > 1:
        nc_pre = sol_pre.shape[1]
        nc_sol = sol.shape[1]
        if nc_pre < nc_sol:
            sol_pre = np.hstack([sol_pre, np.zeros((sol_pre.shape[0], nc_sol - nc_pre))])
        zsol = np.concatenate([z_pre[:-1], zsol])
        sol = np.vstack([sol_pre[:-1], sol])

    return zsol, sol


# ══════════════════════════════════════════════════════════════════════════════
# Método de tiro (bisección + secante)
# ══════════════════════════════════════════════════════════════════════════════

def _shoot_objective(vinicial, verbose=True):
    """
    Dispara y devuelve (zexit, pexit, ugexit, phiexit).
    """
    _G["q"] = vinicial * _G["rho_ti"]
    _G["Q"] = _G["q"] * math.pi * _G["wr"] ** 2
    zsol, sol = _march(vinicial)
    zexit = float(zsol[-1])
    pexit = float(sol[-1, 0])
    phiexit = float(sol[-1, 1])
    ugexit = float(sol[-1, 5])
    if verbose:
        print(f"  v={vinicial:.4f}  z={zexit:.1f}  P={pexit:.3e}  ug={ugexit:.3f}  phi={phiexit:.3f}")
    return zsol, sol, zexit, pexit, ugexit, phiexit


def _converged(zexit, pexit, ugexit, phiexit):
    vsound = 0.99 * math.sqrt(R * _G["T"])
    ok1 = zexit >= -5 and pexit >= pfinal and 0.95 * vsound < ugexit <= 1.05 * vsound
    ok2 = zexit >= -5 and abs(pexit - pfinal) < 0.05e5 and phiexit >= _G["phicrit"]
    ok3 = zexit >= -15 and abs(pexit - pfinal) < 1.5e5 and ugexit > 0.5
    return ok1 or ok2 or ok3


# ══════════════════════════════════════════════════════════════════════════════
# Función principal
# ══════════════════════════════════════════════════════════════════════════════

def RIconduit2D_FD_f(radius, Pressure, wt, Temperature, content_crystal, N_r=20, N_z=3000):
    """
    Modelo 2D axisimétrico por diferencias finitas.

    Parámetros
    ----------
    radius : float   Radio del conducto (m)
    Pressure : float Sobrepresión en la entrada (Pa)
    wt : float       Contenido inicial de agua (% peso)
    Temperature : float Temperatura (K)
    content_crystal : float Fracción inicial de cristales
    N_r : int        Nodos radiales
    N_z : int        Pasos en z

    Retorna
    -------
    Lista con misma firma que RIconduitex5_5_f más sol_fullex (índice 16).
    sol_fullex[:, 0:5] = [P, phi, Nd, x, ug]
    sol_fullex[:, 5:]  = u_m(r) en cada z
    """
    # ── configurar globales ─────────────────────────────────────────────────
    _G["N_r"] = int(N_r)
    _G["N_z"] = int(N_z)
    wr = float(radius)
    _G["wr"] = wr
    _G["r"] = np.linspace(1e-6, wr, N_r)
    _G["dr"] = float(_G["r"][1] - _G["r"][0])
    _G["T"] = float(Temperature)
    _G["Tc"] = Temperature - 273.15
    _G["h2o"] = float(wt)
    _G["co"] = wt / 100.0
    _G["xi"] = float(content_crystal)
    _G["xmax_run"] = float(xmax)

    if geometry == "dyke":
        _G["cg"] = 3
    else:
        _G["cg"] = 8

    overP = float(Pressure)
    Pi = rcrust * g * abs(H) + overP
    _G["Pi"] = Pi

    dis = C1 * Pi ** beta
    if dis > _G["co"]:
        dis = _G["co"]
    _G["rho_m"] = density(sio2, tio2, al2o3, feo, mgo, cao, na2o, k2o,
                          dis * 100, _G["Tc"], Pi / 1e6)

    exi = (1 - _G["xi"]) * (_G["co"] - C1 * Pi ** beta) / (1 - C1 * Pi ** beta)
    if exi <= 0:
        _G["rho_ti"] = _G["rho_m"]
    else:
        fgi = exi  # = fg at inlet
        _G["rho_ti"] = Pi * _G["rho_m"] / (_G["rho_m"] * R * _G["T"] * fgi + Pi * (1 - fgi))

    # Estimar Nd inicial
    vinicial0 = 10.0
    visc0 = _effective_visc(Pi, 0.0, _G["xi"], vinicial0, 1e8)
    dpdz0 = -_G["rho_ti"] * (g + _G["cg"] * visc0 * vinicial0 / (wr**2 * _G["rho_ti"]))
    dpdt0 = abs(dpdz0 * vinicial0)
    _G["Nd0"] = float(np.clip(10 ** (1.5 * math.log10(max(dpdt0, 1e-30)) + 5), 1e6, 1e10))
    if C1 * Pi ** beta < _G["co"]:
        _G["Nd0"] = 1e8

    # Actualizar phicrit dinámico (igual que 5_5)
    phisel = 0.2
    Pmin, Pmax = pfinal, (_G["co"] / C1) ** (1 / beta)
    Pcalc = (Pmax + Pmin) / 2
    for _ in range(50):
        ncalc = (1-_G["xi"])*(_G["co"] - C1*Pcalc**beta) / (1 - C1*Pcalc**beta)
        rhogcalc = Pcalc / (R * _G["T"])
        phicalc = 1.0 / (1.0 + (Pcalc / (ncalc * R * _G["T"])) * (1-ncalc) / _G["rho_m"])
        if abs(phicalc - phisel) < 0.002:
            break
        if phicalc < phisel - 0.002:
            Pmax = Pcalc
        else:
            Pmin = Pcalc
        Pcalc = (Pmax + Pmin) / 2

    viscalc = _effective_visc(Pcalc, phicalc, _G["xi"], vinicial0, _G["Nd0"])
    rbcalc = max((phicalc / ((4/3) * math.pi * _G["Nd0"] * (1-phicalc))) ** (1/3), 1e-9)
    dfgdpcalc = (C1 * beta * Pcalc**(beta-1)) * (_G["co"] - 1) / (1 - C1*Pcalc**beta)**2
    drhogdp = 1 / (R * _G["T"])
    rhoticalc = Pcalc * _G["rho_m"] / (_G["rho_m"]*R*_G["T"]*ncalc + Pcalc*(1-ncalc))
    dpdzcalc2 = (-rhoticalc*(g + _G["cg"]*viscalc*vinicial0/(wr**2*rhoticalc))) / (1-(vinicial0**2)/(15e9/_G["rho_m"]))
    dvdz_est = vinicial0 * (-dfgdpcalc*dpdzcalc2*(1-phicalc)) / max((1-phicalc)**2, 1e-9)
    dvdz_est = 0.5 * (dvdz_est + vinicial0 / wr)
    Ca = abs(dvdz_est * viscalc * rbcalc / 0.3)
    _G["phicrit"] = ((lsup - linf) / 2) * math.erf(math.log10(max(Ca, 1e-30))) + (lsup + linf) / 2
    _G["limphi1"] = ((limperl - limperh) / 2) * math.erf(math.log10(max(Ca, 1e-30))) + (limperl + limperh) / 2
    _G["limphi2"] = _G["limphi1"] + 0.01

    # ── método de tiro ──────────────────────────────────────────────────────
    vinicial = 10.0
    vmin, vmax = 0.1, 50.0
    converged = False
    best_zsol, best_sol = None, None
    best_dist = 1e30
    count = 1
    U_prev, P_prev = None, None

    while count <= 60:
        print(f"\nCount = {count}  vinicial = {vinicial:.5f}")
        try:
            zsol, sol, zexit, pexit, ugexit, phiexit = _shoot_objective(vinicial)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            vmax = vinicial
            vinicial = vmin + (vmax - vmin) / 2
            count += 1
            continue

        # guardar mejor trayectoria
        dist = abs(zexit + 0.0) + abs(pexit - pfinal) / 1e5
        if dist < best_dist:
            best_dist = dist
            best_zsol, best_sol = zsol.copy(), sol.copy()

        if _converged(zexit, pexit, ugexit, phiexit):
            print(">>> Solucion de tiro convergida.")
            converged = True
            break

        # secante después de las primeras iteraciones
        v_shot = vinicial
        if count > 15 and U_prev is not None:
            f0 = P_prev - pfinal
            f1 = pexit - pfinal
            if abs(f1 - f0) > 1e2:
                v_sec = v_shot - f1 * (v_shot - U_prev) / (f1 - f0)
                v_sec = float(np.clip(v_sec, vmin + 1e-4, vmax - 1e-4))
                if vmin < v_sec < vmax:
                    vinicial = v_sec
                    U_prev, P_prev = v_shot, pexit
                    count += 1
                    continue

        # bisección
        if pexit < pfinal:
            vmax = v_shot
            vinicial = vmin + (vmax - vmin) / 2
        else:
            if zexit < -5 or ugexit > 1.02 * 0.99 * math.sqrt(R * _G["T"]):
                vmax = v_shot
                vinicial = vmin + (vmax - vmin) / 2
            elif ugexit <= 0.98 * 0.99 * math.sqrt(R * _G["T"]):
                vmin = v_shot
                vinicial = vmin + (vmax - vmin) / 2

        U_prev, P_prev = v_shot, pexit
        print(f"  vmin={vmin:.4f}  vmax={vmax:.4f}")
        if vmax - vmin < 1e-5:
            print("Intervalo de bisección agotado.")
            break
        count += 1

    if not converged:
        print(f"ADVERTENCIA: no convergio. Usando mejor trayectoria (dist={best_dist:.2f}).")
        zsol, sol = best_zsol, best_sol

    # ── post-proceso y empaquetado ──────────────────────────────────────────
    numb = sol.shape[0]
    rho_m = _G["rho_m"]
    wr = _G["wr"]
    visctot = np.zeros(numb)
    rbub = np.zeros(numb)
    for j in range(numb):
        P_j, phi_j, Nd_j, x_j, um_j = sol[j, 0], sol[j, 1], sol[j, 2], sol[j, 3], sol[j, 4]
        visctot[j] = _effective_visc(P_j, float(np.clip(phi_j, 1e-12, _G["phicrit"]-1e-6)),
                                     x_j, max(um_j, 1e-9), max(Nd_j, 1.0))
        phi_c = float(np.clip(phi_j, 1e-12, _G["phicrit"]-1e-6))
        Nd_s = max(Nd_j, 1.0)
        rbub[j] = (phi_c / ((4/3)*math.pi*Nd_s*(1-phi_c)))**(1/3)

    # sol_fullex tiene las mismas columnas que sol (ya incluye u_m(r))
    sol_fullex = sol.copy()

    # compatibilidad con la firma de 5_5 (16 elementos + sol_fullex en [16])
    return [
        zsol,          # 0
        sol[:, :6],    # 1  (P, phi, Nd, x, um_avg, ug) — misma forma que 5_5
        count,         # 2
        vinicial,      # 3
        rho_m,         # 4
        rbub,          # 5
        visctot,       # 6
        _G["rho_ti"],  # 7
        _G.get("fragcrit", 0.0),  # 8
        _G["phicrit"], # 9
        _G["limphi1"], # 10
        _G["limphi2"], # 11
        _G["co"],      # 12
        _G["xi"],      # 13
        _G.get("xfinal", _G["xi"]),  # 14
        _G["cg"],      # 15
        sol_fullex,    # 16: [P, phi, Nd, x, um_avg, ug, um(r)...]
    ]


def radial_grid():
    """Devuelve (r, wr, N_r) para plots."""
    return np.asarray(_G["r"]), float(_G["wr"]), int(_G["N_r"])
