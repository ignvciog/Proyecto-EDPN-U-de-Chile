"""
Conducto axisimétrico con velocidad radial, en diferencias finitas.

En cada paso en z se discretizan, sin anular u_r ni promediar la sección:

    masa del fundido y masa del gas,
    momento axial del fundido y del gas,
    momento radial del fundido y del gas,
    transporte de ξ y de N.

Las derivadas en z van hacia atrás. En r el flujo de masa y el
viscoseo van en forma conservativa; la convección va contra el
flujo. Mientras el agua sigue disuelta se integra solo el fundido
(φ = 0). Cuando Henry suelta agua entran las dos fases.

La viscosidad y el coeficiente de arrastre se actualizan en un
lazo de Picard: dentro de cada iteración esas funciones quedan
congeladas y el sistema de diferencias se resuelve con
least_squares. El lazo es el mismo con el que el tramo anterior
trata μ(u). No sustituye a las ecuaciones.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares

from calbuco2015d import (
    C1,
    F1,
    F2,
    Fc,
    H,
    Patm,
    R as RV,
    T1,
    Tc1,
    al2o3,
    ar1,
    ar2,
    beta,
    cao,
    f2o,
    feo,
    g,
    h2o1,
    k2o,
    mgo,
    mno,
    model,
    na2o,
    overP1,
    p2o5,
    radius1,
    rcrust,
    sio2,
    tcar,
    tio2,
    xi1,
    xmax,
)
from density import density
from fvrel import fvrel
from viscosity import viscosity

R_COND = float(radius1)
XI0 = float(xi1)
CO = float(h2o1) / 100.0
RHO_CRUST = float(rcrust)
P_BASE = RHO_CRUST * g * abs(H) + float(overP1)
RHO_M = float(
    density(sio2, tio2, al2o3, feo, mgo, cao, na2o, k2o, h2o1, Tc1, P_BASE / 1e6)
)
MU_G = 1.0e-5
PHI_CRIT = 0.70
N0 = 1.0e8
CS = math.sqrt(RV * T1)


def _n_henry(P, xi):
    P = np.maximum(np.asarray(P, dtype=float), 1.0e4)
    xi = np.asarray(xi, dtype=float)
    num = (1.0 - XI0) * CO - (1.0 - xi) * C1 * np.power(P, beta)
    den = np.maximum(1.0 - C1 * np.power(P, beta), 1.0e-12)
    return np.maximum(num / den, 0.0)


def _dn_dP(P, xi):
    P = np.maximum(np.asarray(P, dtype=float), 1.0e4)
    xi = np.asarray(xi, dtype=float)
    s = np.power(P, beta)
    ds = beta * np.power(P, beta - 1.0)
    num = (1.0 - XI0) * CO - (1.0 - xi) * C1 * s
    den = np.maximum(1.0 - C1 * s, 1.0e-12)
    dnum = -(1.0 - xi) * C1 * ds
    dden = -C1 * ds
    out = (dnum * den - num * dden) / den ** 2
    return np.where(num > 0.0, out, 0.0)


def _dn_dxi(P, xi):
    P = np.maximum(np.asarray(P, dtype=float), 1.0e4)
    s = np.power(P, beta)
    den = np.maximum(1.0 - C1 * s, 1.0e-12)
    # ∂/∂ξ [a - (1-ξ) C1 s] / den = C1 s / den, solo si hay exsolución
    num = (1.0 - XI0) * CO - (1.0 - np.asarray(xi, dtype=float)) * C1 * s
    return np.where(num > 0.0, (C1 * s) / den, 0.0)


def _rb(phi, Nd):
    phi = np.clip(phi, 0.0, 0.999)
    Nd = np.maximum(Nd, 1.0)
    return np.maximum(
        (np.maximum(phi, 1.0e-16) / ((4.0 / 3.0) * math.pi * Nd * np.maximum(1.0 - phi, 1.0e-12)))
        ** (1.0 / 3.0),
        1.0e-8,
    )


def _shear(umz, dr):
    n = umz.size
    gdot = np.zeros(n)
    gdot[0] = abs(umz[1] - umz[0]) / dr
    gdot[-1] = abs(umz[-1] - umz[-2]) / dr
    gdot[1:-1] = np.abs(umz[2:] - umz[:-2]) / (2.0 * dr)
    return np.maximum(gdot, 1.0e-3)


def _mu_melt(P, phi, xi, umz, Nd, dr):
    """Giordano × cristales × Llewellin, con el corte |∂u_z/∂r|."""
    gdot = _shear(umz, dr)
    P = np.maximum(P, 1.0e4)
    fg = _n_henry(P, xi)
    out = np.empty(P.size)
    for i in range(P.size):
        h2o = C1 * P[i] ** beta * 100.0 if fg[i] > 0.0 else h2o1
        viscl = fvrel(model, float(xi[i]), XI0, ar1, ar2, xmax, float(gdot[i])) * viscosity(
            sio2, tio2, al2o3, feo, mno, mgo, cao, na2o, k2o, p2o5, h2o, f2o, Tc1
        )
        phi_c = float(np.clip(phi[i], 1.0e-8, max(PHI_CRIT - 1.0e-4, 1.0e-3)))
        rb = float(_rb(np.array([phi_c]), np.array([Nd[i]]))[0])
        nca = rb * viscl * max(gdot[i], 1.0e-8) / 3.0
        phistar = PHI_CRIT + 0.05
        if phi_c >= 0.99 * phistar:
            out[i] = max(viscl, 1.0)
            continue
        AA = (1.0 - phi_c / phistar) ** (-phistar)
        BB = (1.0 - phi_c / phistar) ** (5.0 * phistar / 3.0)
        c1 = -0.2895 * phi_c + 0.8132
        rel = 0.5 * (AA - BB) * (1.0 - math.erf(c1 * math.log(max(nca, 1.0e-30)) + phi_c)) + BB
        out[i] = max(float(rel * viscl), 1.0)
    return out


def _K_arrastre(phi, Nd, mu, rho_g, slip_abs):
    """F = K (u_g - u_m), K congelado en el Picard. Mismo régimen que el 1D."""
    phi = np.clip(phi, 0.0, 0.999)
    rb = _rb(phi, Nd)
    fac = phi * (1.0 - phi)
    K = np.zeros(phi.shape)
    for i in range(phi.size):
        if phi[i] < 1.0e-8:
            K[i] = 0.0
            continue
        if phi[i] < 0.15:
            K[i] = 3.0 * mu[i] * fac[i] / rb[i] ** 2
        elif phi[i] < 0.40:
            tt = (phi[i] - 0.15) / (0.40 - 0.15)
            Re = 2.0 * rb[i] * rho_g[i] * slip_abs[i] / MU_G
            kper = max(0.131 * rb[i] ** 2 * (phi[i] - 0.15 + 0.05) ** 2.1, 1.0e-30)
            stokes = 3.0 * mu[i] / rb[i] ** 2
            darcy = (MU_G / kper) if Re <= 2200.0 else 0.33 * rho_g[i] * slip_abs[i] / (4.0 * rb[i])
            K[i] = (darcy ** tt) * (stokes ** (1.0 - tt)) * fac[i]
        elif phi[i] < PHI_CRIT:
            Re = 2.0 * rb[i] * rho_g[i] * slip_abs[i] / MU_G
            kper = max(0.131 * rb[i] ** 2 * (max(phi[i] - 0.15, 0.0) + 0.05) ** 2.1, 1.0e-30)
            K[i] = ((MU_G / kper) if Re <= 2200.0 else 0.33 * rho_g[i] * slip_abs[i] / (4.0 * rb[i])) * fac[i]
        else:
            ra, cd = 1.0e-3, 0.8
            K[i] = 3.0 * cd * rho_g[i] * max(slip_abs[i], 1.0e-6) * fac[i] / (8.0 * ra)
    return K


def _gamma_xi(P, xi):
    den = (1.0 - XI0) * CO - (1.0 - xmax) * C1 * Patm ** beta
    den = max(den, 1.0e-12)
    f2 = np.maximum(
        0.0,
        ((1.0 - XI0) * CO - (1.0 - xi) * C1 * np.maximum(P, 1.0e4) ** beta) / den,
    )
    xeq = XI0 + (xmax - XI0) * f2
    f3 = np.maximum(0.0, 1.0 - xi / np.maximum(xeq, 1.0e-8))
    rate = np.maximum(0.0, (xmax - XI0) * f2 * f3 / tcar)
    # después de fragmentar la tasa se apaga
    return rate


def _gamma_N(phi, Nd, mu, rho_g):
    phi = np.clip(phi, 0.0, 0.999)
    rb = _rb(phi, Nd)
    out = np.zeros(phi.shape)
    for i in range(phi.size):
        if phi[i] < 1.0e-6 or phi[i] >= PHI_CRIT or rb[i] >= 0.5 * R_COND:
            continue
        bracket = 1.0 - (F1 + F2) * ((3.0 * phi[i] * math.pi / (6.0 * PHI_CRIT)) / (4.0 * math.pi)) ** (1.0 / 3.0)
        if bracket <= 0.05:
            continue
        out[i] = (
            -Nd[i] ** (2.0 / 3.0)
            * (1.0 - phi[i]) ** (-1.0 / 3.0)
            * (1.0 / 9.0)
            * (RHO_M - rho_g[i]) * g / max(mu[i], 1.0)
            * (3.0 * phi[i] / (4.0 * math.pi)) ** (2.0 / 3.0)
            * (F1 ** 2 - F2 ** 2)
            / bracket
            * Fc
            * (1.0 - phi[i] / PHI_CRIT)
            * (R_COND - rb[i]) / R_COND
        )
    return out


def malla(n_r):
    r = np.linspace(0.0, R_COND, n_r)
    return r, float(r[1] - r[0])


def estado_inicial(vin, n_r):
    r, dr = malla(n_r)
    umz = 2.0 * vin * (1.0 - (r / R_COND) ** 2)
    umz[-1] = 0.0
    n = r.size
    return {
        "z": float(H),
        "r": r,
        "dr": dr,
        "P": np.full(n, P_BASE),
        "phi": np.zeros(n),
        "umz": umz,
        "umr": np.zeros(n),
        "ugz": umz.copy(),
        "ugr": np.zeros(n),
        "xi": np.full(n, XI0),
        "N": np.full(n, N0),
        "h": None,
    }


def _cara(a):
    return 0.5 * (a[:-1] + a[1:])


def _div_radial(r, dr, flujo_nodo):
    """(1/r) ∂(r f)/∂r con f(0) = f(R) = 0, f en los nodos."""
    n = r.size
    div = np.zeros(n)
    f_half = _cara(flujo_nodo)
    r_half = _cara(r)
    # eje: volumen 0..r_half[0], flujo nulo en 0
    div[0] = 2.0 * f_half[0] / max(r_half[0], 1.0e-8)
    for i in range(1, n - 1):
        div[i] = (r_half[i] * f_half[i] - r_half[i - 1] * f_half[i - 1]) / (r[i] * dr)
    # media celda en la pared: flujo nulo en r=R, flujo en la cara interior
    div[-1] = (-r_half[-1] * f_half[-1]) / (max(r[-1], dr) * 0.5 * dr)
    return div


def _lap_axial(r, dr, mu_eff, u):
    """(1/r) ∂/∂r [ r μ ∂u/∂r ]. μ_eff ya incluye (1-φ) o φ."""
    n = r.size
    L = np.zeros(n)
    mu_h = _cara(mu_eff)
    r_h = _cara(r)
    dudr = np.diff(u) / dr
    flux = r_h * mu_h * dudr
    L[0] = 4.0 * mu_eff[0] * (u[1] - u[0]) / dr ** 2
    for i in range(1, n - 1):
        L[i] = (flux[i] - flux[i - 1]) / (r[i] * dr)
    # pared: derivada hacia atrás, el flujo en r=R usa u conocido
    L[-1] = (flux[-1] - flux[-2]) / (r[-1] * dr) if n > 2 else 0.0
    return L


def _d_dr(r, y, en_el_eje="par"):
    n = y.size
    d = np.zeros(n)
    dr = r[1] - r[0]
    if en_el_eje == "par":
        d[0] = 0.0
    else:
        d[0] = (y[1] - y[0]) / dr
    d[-1] = (y[-1] - y[-2]) / dr
    d[1:-1] = (y[2:] - y[:-2]) / (2.0 * dr)
    return d


def _adv_r(ur, ddr):
    """Convección radial contra el flujo."""
    n = ur.size
    dr_node = None
    return ur * ddr


def _Q(estado):
    r = estado["r"]
    jz = RHO_M * (1.0 - estado["phi"]) * estado["umz"] + (estado["P"] / (RV * T1)) * estado["phi"] * estado["ugz"]
    return float(2.0 * math.pi * np.trapezoid(jz * r, r))


def _poiseuille(vin, mu_ref):
    return -RHO_M * g - 8.0 * mu_ref * vin / R_COND ** 2


def _a_caras(ur, n):
    """Velocidad radial en las caras. Si ya viene en caras, se deja."""
    ur = np.asarray(ur, dtype=float)
    if ur.size == n - 1:
        return ur.copy()
    return 0.5 * (ur[:-1] + ur[1:])


def _empaquetar(st, fases):
    if fases == 1:
        # P, umz (sin pared), umr (sin eje ni pared), xi
        return np.concatenate(
            [st["P"], st["umz"][:-1], st["umr"][1:-1], st["xi"]]
        )
    n = st["r"].size
    umr = _a_caras(st["umr"], n)
    ugr = _a_caras(st["ugr"], n)
    return np.concatenate(
        [st["P"], st["phi"], st["umz"], st["ugz"], st["xi"], st["N"], umr, ugr]
    )


def _desempaquetar(y, plantilla, fases):
    st = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in plantilla.items()}
    n = plantilla["r"].size
    if fases == 1:
        st["P"] = y[:n]
        st["umz"] = np.concatenate([y[n : 2 * n - 1], [0.0]])
        st["umr"] = np.concatenate([[0.0], y[2 * n - 1 : 3 * n - 3], [0.0]])
        st["xi"] = np.clip(y[3 * n - 3 : 4 * n - 3], XI0, xmax)
        st["phi"][:] = 0.0
        st["ugz"] = st["umz"].copy()
        st["ugr"] = st["umr"].copy()
        return st
    nf = n - 1
    st["P"] = y[0:n].copy()
    st["phi"] = np.clip(y[n : 2 * n], 0.0, 0.99)
    st["umz"] = y[2 * n : 3 * n].copy()
    st["ugz"] = y[3 * n : 4 * n].copy()
    st["xi"] = np.clip(y[4 * n : 5 * n], 0.0, xmax)
    st["N"] = np.maximum(y[5 * n : 6 * n], 1.0)
    st["umr"] = y[6 * n : 6 * n + nf].copy()
    st["ugr"] = y[6 * n + nf : 6 * n + 2 * nf].copy()
    st["P"] = np.maximum(st["P"], 1.0e4)
    return st


def _transporte(st, prev, h, fuente, campo):
    """u·∇c - fuente = 0, diferencia hacia atrás en z y contra el flujo en r."""
    r, dr = st["r"], st["dr"]
    c = st[campo]
    c0 = prev[campo]
    ur = st["umr"]
    if ur.size == c.size - 1:
        nodal = np.zeros(c.size)
        nodal[1:-1] = 0.5 * (ur[:-1] + ur[1:])
        ur = nodal
    uz = np.maximum(st["umz"], 0.0)
    ddz = (c - c0) / h
    ddr = _d_dr(r, c)
    # contra el flujo
    adv_r = np.zeros(c.size)
    for i in range(1, c.size - 1):
        if ur[i] >= 0.0:
            adv_r[i] = ur[i] * (c[i] - c[i - 1]) / dr
        else:
            adv_r[i] = ur[i] * (c[i + 1] - c[i]) / dr
    return uz * ddz + adv_r - fuente


def residual_fundido(y, prev, h, mu, prev2, h_prev):
    st = _desempaquetar(y, prev, 1)
    r, dr = st["r"], st["dr"]
    n = r.size
    umz, umr, P, xi = st["umz"], st["umr"], st["P"], st["xi"]
    # masa: ρ ∂u_z/∂z + ρ (1/r)∂(r u_r)/∂r = 0
    ddz = (umz - prev["umz"]) / h
    divr = _div_radial(r, dr, umr)
    masa = RHO_M * (ddz + divr)
    # momento axial
    lap = _lap_axial(r, dr, mu, umz)
    conv_r = _adv_r(umr, _d_dr(r, umz))
    conv_z = umz * ddz
    # esfuerzo axial congelado del paso anterior, como fuerza
    if prev2 is None:
        ax = np.zeros(n)
    else:
        tau = mu * (umz - prev["umz"]) / h
        tau0 = mu * (prev["umz"] - prev2["umz"]) / h_prev
        ax = (tau - tau0) / h
    dPdz = (P - prev["P"]) / h
    mom_z = RHO_M * (conv_z + conv_r) + dPdz + RHO_M * g - lap - ax
    # momento radial
    lap_r = _lap_axial(r, dr, mu, umr)
    geom = np.zeros(n)
    geom[1:] = mu[1:] * umr[1:] / np.maximum(r[1:], dr) ** 2
    conv_rr = umr * _d_dr(r, umr, en_el_eje="impar")
    conv_zr = umz * (umr - prev["umr"]) / h
    dPdr = _d_dr(r, P)
    mom_r = RHO_M * (conv_zr + conv_rr) + dPdr - lap_r + geom
    # transporte de ξ. Sin gas, Γ_ξ usa P.
    gxi = _gamma_xi(P, xi)
    # en la pared se copia el nodo interior
    xi_eq = _transporte(st, prev, h, gxi, "xi")
    xi_eq[-1] = xi[-1] - xi[-2]
    xi_eq[0] = xi[0] - xi[1]
    # escalas
    esc_m = RHO_M * max(abs(float(prev["umz"][0])), 1.0) / max(h, 1.0)
    esc_z = max(RHO_M * g, 1.0)
    esc_r = esc_z
    # en la pared el momento radial cierra P: u_r ya no es incógnita
    res = [masa[:-1] / esc_m, mom_z[:-1] / esc_z, mom_r[1:] / esc_r, xi_eq / 0.05]
    return np.concatenate(res)


def _div_caras(r, dr, F):
    """(1/r) ∂(r F)/∂r con F en las caras y F(R) = 0."""
    n = r.size
    rf = 0.5 * (r[:-1] + r[1:])
    div = np.zeros(n)
    div[0] = 2.0 * F[0] / max(rf[0], 1.0e-8)
    for i in range(1, n - 1):
        div[i] = (rf[i] * F[i] - rf[i - 1] * F[i - 1]) / (r[i] * dr)
    div[-1] = (-rf[-1] * F[-1]) / (max(r[-1], dr) * 0.5 * dr)
    return div


def _ur_en_nodos(ur_f):
    n = ur_f.size + 1
    u = np.zeros(n)
    u[1:-1] = 0.5 * (ur_f[:-1] + ur_f[1:])
    return u


def _lap_ur_caras(r, mu_f, ur):
    """(1/r) ∂/∂r [r μ ∂u_r/∂r] - μ u_r/r^2 en las caras, con u_r(0)=u_r(R)=0."""
    rf = 0.5 * (r[:-1] + r[1:])
    x = np.concatenate([[0.0], rf, [r[-1]]])
    u = np.concatenate([[0.0], ur, [0.0]])
    nf = ur.size
    lap = np.zeros(nf)
    for j in range(nf):
        h1 = x[j + 1] - x[j]
        h2 = x[j + 2] - x[j + 1]
        dudr_r = (u[j + 2] - u[j + 1]) / h2
        dudr_l = (u[j + 1] - u[j]) / h1
        mu_r = mu_f[min(j + 1, nf - 1)]
        mu_l = mu_f[max(j - 1, 0)]
        r_right = 0.5 * (x[j + 1] + x[j + 2])
        r_left = 0.5 * (x[j] + x[j + 1])
        ancho = 0.5 * (h1 + h2)
        centro = max(x[j + 1], 1.0e-8)
        lap[j] = (r_right * mu_r * dudr_r - r_left * mu_l * dudr_l) / (centro * ancho)
        lap[j] -= mu_f[j] * ur[j] / centro ** 2
    return lap


def residual_bifasico(y, prev, h, mu, Kdrag, prev2, h_prev):
    st = _desempaquetar(y, prev, 2)
    r, dr = st["r"], st["dr"]
    n = r.size
    P, phi = st["P"], st["phi"]
    umz, ugz = st["umz"], st["ugz"]
    umr, ugr = st["umr"], st["ugr"]
    xi, Nd = st["xi"], st["N"]
    rho_g = np.maximum(P, Patm) / (RV * T1)
    phi_f = 0.5 * (phi[:-1] + phi[1:])
    rho_f = 0.5 * (rho_g[:-1] + rho_g[1:])
    umz_f = 0.5 * (umz[:-1] + umz[1:])
    ugz_f = 0.5 * (ugz[:-1] + ugz[1:])
    K_f = 0.5 * (Kdrag[:-1] + Kdrag[1:])
    mu_f = 0.5 * (mu[:-1] + mu[1:])
    umr_n = _ur_en_nodos(umr)
    ugr_n = _ur_en_nodos(ugr)
    # Henry y exsolución con el estado nuevo
    n_gas = _n_henry(P, xi)
    dnP, dnxi = _dn_dP(P, xi), _dn_dxi(P, xi)
    jz = RHO_M * (1.0 - phi) * umz + rho_g * phi * ugz
    jr = RHO_M * (1.0 - phi) * umr_n + rho_g * phi * ugr_n
    jz0 = RHO_M * (1.0 - prev["phi"]) * prev["umz"] + (prev["P"] / (RV * T1)) * prev["phi"] * prev["ugz"]
    dn_dz = dnP * (P - prev["P"]) / h + dnxi * (xi - prev["xi"]) / h
    dn_dr = dnP * _d_dr(r, P) + dnxi * _d_dr(r, xi)
    Gamma = jr * dn_dr + jz * dn_dz
    con_gas = (n_gas > 1.0e-8) | (phi > 1.0e-6) | (prev["phi"] > 1.0e-6)
    # masas, con el flujo radial en las caras
    jz_m = RHO_M * (1.0 - phi) * umz
    jz_m0 = RHO_M * (1.0 - prev["phi"]) * prev["umz"]
    jr_m = RHO_M * (1.0 - phi_f) * umr
    masa_m = (jz_m - jz_m0) / h + _div_caras(r, dr, jr_m) + Gamma
    jz_g = rho_g * phi * ugz
    jz_g0 = (prev["P"] / (RV * T1)) * prev["phi"] * prev["ugz"]
    jr_g = rho_f * phi_f * ugr
    masa_g = (jz_g - jz_g0) / h + _div_caras(r, dr, jr_g) - Gamma
    # momentos axiales, por unidad de volumen de fase donde φ>0
    ddz_m = (umz - prev["umz"]) / h
    ddz_g = (ugz - prev["ugz"]) / h
    mu_m = mu * (1.0 - phi)
    mu_g_eff = MU_G * np.maximum(phi, 0.0)
    lap_mz = _lap_axial(r, dr, mu_m, umz)
    lap_gz = _lap_axial(r, dr, mu_g_eff, ugz)
    dPdz = (P - prev["P"]) / h
    Fz = Kdrag * (ugz - umz)
    mom_mz = (
        RHO_M * (1.0 - phi) * (umz * ddz_m + _adv_r(umr_n, _d_dr(r, umz)))
        + (1.0 - phi) * dPdz
        + RHO_M * (1.0 - phi) * g
        - Fz
        - lap_mz
    )
    # gas: la ecuación por unidad de volumen de gas
    phi_s = np.maximum(phi, 1.0e-8)
    mom_gz = (
        rho_g * (ugz * ddz_g + _adv_r(ugr_n, _d_dr(r, ugz)))
        + dPdz
        + rho_g * g
        + Fz / phi_s
        - lap_gz / phi_s
    )
    # momentos radiales, en las caras: la presión entra como (P_{j+1}-P_j)/Δr
    umr0 = _a_caras(prev["umr"], n)
    ugr0 = _a_caras(prev["ugr"], n)
    dPdr = (P[1:] - P[:-1]) / dr
    Fr = K_f * (ugr - umr)
    mu_m_f = mu_f * (1.0 - phi_f)
    mu_g_f = MU_G * np.maximum(phi_f, 0.0)
    lap_mr = _lap_ur_caras(r, mu_m_f, umr)
    lap_gr = _lap_ur_caras(r, mu_g_f, ugr)
    xcar = np.concatenate([[0.0], 0.5 * (r[:-1] + r[1:]), [r[-1]]])
    dur = np.gradient(np.concatenate([[0.0], umr, [0.0]]), xcar)[1:-1]
    dugr = np.gradient(np.concatenate([[0.0], ugr, [0.0]]), xcar)[1:-1]
    phi_fs = np.maximum(phi_f, 1.0e-8)
    mom_mr = (
        RHO_M * (1.0 - phi_f) * (umz_f * (umr - umr0) / h + umr * dur)
        + (1.0 - phi_f) * dPdr
        - Fr
        - lap_mr
    )
    mom_gr = (
        rho_f * (ugz_f * (ugr - ugr0) / h + ugr * dugr)
        + dPdr
        + Fr / phi_fs
        - lap_gr / phi_fs
    )
    # transporte. Tras fragmentar en el nodo, la tasa se anula.
    frag = phi >= PHI_CRIT
    gxi = _gamma_xi(P, xi)
    gN = _gamma_N(phi, Nd, mu, rho_g)
    gxi = np.where(frag, 0.0, gxi)
    gN = np.where(frag, 0.0, gN)
    xi_eq = _transporte(st, prev, h, gxi, "xi")
    # N viaja con el fundido. La ecuación es u·∇N = Γ_N.
    stN = dict(st)
    N_eq = _transporte(stN, prev, h, gN, "N")
    xi_eq[0] = xi[0] - xi[1]
    xi_eq[-1] = xi[-1] - xi[-2]
    N_eq[0] = Nd[0] - Nd[1]
    N_eq[-1] = Nd[-1] - Nd[-2]
    # sin gas la masa y el momento del gas no están: φ = 0 y u_g = u_m
    seco = ~con_gas
    masa_g[seco] = phi[seco] / 1.0e-4
    mom_gz[seco] = (ugz[seco] - umz[seco]) / 0.05
    seco_f = ~(con_gas[:-1] | con_gas[1:])
    mom_gr[seco_f] = (ugr[seco_f] - umr[seco_f]) / 0.05
    # el fundido no desliza en la pared
    mom_mz[-1] = umz[-1] / 1.0e-3
    esc = max(RHO_M * g, 1.0)
    u_ref = max(float(np.max(np.abs(prev["umz"]))), 1.0)
    esc_m = RHO_M * u_ref / max(h, 1.0)
    # la masa de gas pesa ρ_g φ u, no ρ_m u
    rho_ref = max(float(np.mean(np.maximum(prev["P"], Patm) / (RV * T1))), 1.0)
    phi_ref = max(float(np.max(prev["phi"])), float(np.max(phi)), 1.0e-5)
    esc_g = max(rho_ref * phi_ref * u_ref / max(h, 1.0), 1.0e-4)
    return np.concatenate(
        [
            masa_m / esc_m,
            masa_g / esc_g,
            mom_mz / esc,
            mom_gz / esc,
            mom_mr / 100.0,
            mom_gr / esc,
            xi_eq / 0.05,
            N_eq / N0,
        ]
    )


def _cotas(prev, fases):
    n = prev["r"].size
    if fases == 1:
        lo = np.concatenate(
            [
                np.full(n, 1.0e5),
                np.zeros(n - 1),
                np.full(n - 2, -5.0),
                np.full(n, 0.2),
            ]
        )
        hi = np.concatenate(
            [
                np.full(n, P_BASE * 1.05),
                np.full(n - 1, 200.0),
                np.full(n - 2, 5.0),
                np.full(n, xmax),
            ]
        )
        return lo, hi
    nf = n - 1
    lo = np.concatenate(
        [
            np.full(n, 1.0e5),
            np.zeros(n),
            np.zeros(n),
            np.zeros(n),
            np.full(n, 0.2),
            np.full(n, 1.0e6),
            np.full(nf, -30.0),
            np.full(nf, -80.0),
        ]
    )
    hi = np.concatenate(
        [
            np.full(n, P_BASE * 1.05),
            np.full(n, 0.95),
            np.full(n, 500.0),
            np.full(n, 900.0),
            np.full(n, xmax),
            np.full(n, 1.0e12),
            np.full(nf, 30.0),
            np.full(nf, 80.0),
        ]
    )
    return lo, hi


def _escala_resbalamiento(K, phi):
    """Convierte el resbalamiento en una incógnita de orden 1 cuando el arrastre es rígido."""
    esc = max(RHO_M * g, 1.0)
    phi_s = np.maximum(phi, 1.0e-8)
    s0 = np.ones(np.shape(K))
    fuerte = K > esc
    s0[fuerte] = esc * phi_s[fuerte] / np.maximum(K[fuerte], esc)
    return np.maximum(s0, 1.0e-16)


def _semilla(prev, h, prev2, h_prev):
    """Continuación explícita: misma velocidad, ∂P/∂z del paso anterior y φ que pide la masa."""
    st = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in prev.items()}
    if prev2 is not None:
        dPdz = (prev["P"] - prev2["P"]) / max(h_prev, 1.0e-8)
    else:
        dPdz = np.full(prev["r"].size, -RHO_M * g)
    st["P"] = np.maximum(prev["P"] + dPdz * h, Patm)
    rho_g = st["P"] / (RV * T1)
    jz = RHO_M * (1.0 - prev["phi"]) * np.maximum(prev["umz"], 0.0)
    dn = _n_henry(st["P"], st["xi"]) - _n_henry(prev["P"], prev["xi"])
    Gamma = jz * dn / max(h, 1.0e-8)
    suma = Gamma * h / np.maximum(rho_g * np.maximum(prev["umz"], 1.0), 1.0)
    phi = np.clip(prev["phi"] + suma, 0.0, 0.95)
    if phi.size > 2:
        phi[-1] = phi[-2]
        phi[0] = phi[1]
    st["phi"] = phi
    mu = _mu_melt(st["P"], phi, st["xi"], st["umz"], st["N"], st["dr"])
    K = _K_arrastre(phi, st["N"], mu, rho_g, np.abs(prev["ugz"] - prev["umz"]))
    slip = -(dPdz + rho_g * g) * np.maximum(phi, 1.0e-8) / np.maximum(K, 1.0)
    # con arrastre débil el resbalamiento no se cierra así: se conserva el del paso anterior
    debil = K < max(RHO_M * g, 1.0)
    st["ugz"] = np.where(debil, prev["ugz"], prev["umz"] + slip)
    # la semilla radial parte de cero: el modo alternado no se hereda de un paso al otro
    nf = prev["r"].size - 1
    st["umr"] = np.zeros(nf)
    st["ugr"] = np.zeros(nf)
    return st


def _picard(prev, h, prev2, h_prev, fases, guess=None):
    inicio = prev if guess is None else guess
    lo, hi = _cotas(prev, fases)
    if fases == 2:
        n = prev["r"].size
        phi_top = min(0.95, max(float(np.max(prev["phi"])) + 0.02, 0.015))
        hi[n : 2 * n] = phi_top
        # la velocidad radial no puede dispararse en un solo paso
        nf = n - 1
        ur_top = 2.0 if float(np.max(prev["phi"])) < 0.2 else 30.0
        lo[6 * n : 6 * n + nf] = -ur_top
        hi[6 * n : 6 * n + nf] = ur_top
    y = np.minimum(np.maximum(_empaquetar(inicio, fases), lo + 1.0e-12), hi - 1.0e-12)
    sol = None
    n = prev["r"].size
    for _ in range(5):
        st = _desempaquetar(y, prev, fases)
        slip = np.abs(st["ugz"] - st["umz"])
        mu = _mu_melt(st["P"], np.clip(st["phi"], 0.0, 0.95), st["xi"], st["umz"], st["N"], st["dr"])
        if fases == 1:
            fun = lambda z, mu=mu: residual_fundido(z, prev, h, mu, prev2, h_prev)
            sol = least_squares(
                fun, y, bounds=(lo, hi), method="trf",
                ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=40,
            )
            y = sol.x
        else:
            rho_g = np.maximum(st["P"], Patm) / (RV * T1)
            Kdrag = _K_arrastre(np.clip(st["phi"], 0.0, 0.95), st["N"], mu, rho_g, slip)
            s0z = _escala_resbalamiento(Kdrag, np.maximum(st["phi"], prev["phi"]))
            umr_f = _a_caras(st["umr"], n)
            ugr_f = _a_caras(st["ugr"], n)
            K_f = 0.5 * (Kdrag[:-1] + Kdrag[1:])
            phi_f = 0.5 * (np.maximum(st["phi"], 0.0)[:-1] + np.maximum(st["phi"], 0.0)[1:])
            s0r = _escala_resbalamiento(K_f, np.maximum(phi_f, 1.0e-8))
            y_hat = _empaquetar(st, 2)
            y_hat[3 * n : 4 * n] = (st["ugz"] - st["umz"]) / s0z
            nf = n - 1
            y_hat[6 * n + nf :] = (ugr_f - umr_f) / s0r
            lo_h, hi_h = lo.copy(), hi.copy()
            lo_h[3 * n : 4 * n] = -2.0e3
            hi_h[3 * n : 4 * n] = 2.0e3
            lo_h[6 * n + nf :] = -2.0e3
            hi_h[6 * n + nf :] = 2.0e3
            y_hat = np.minimum(np.maximum(y_hat, lo_h + 1.0e-14), hi_h - 1.0e-14)

            def fun(yh, s0z=s0z, s0r=s0r, mu=mu, Kdrag=Kdrag):
                y_fis = yh.copy()
                umz = yh[2 * n : 3 * n]
                umr = yh[6 * n : 6 * n + nf]
                y_fis[3 * n : 4 * n] = umz + yh[3 * n : 4 * n] * s0z
                y_fis[6 * n + nf :] = umr + yh[6 * n + nf :] * s0r
                return residual_bifasico(y_fis, prev, h, mu, Kdrag, prev2, h_prev)

            sol = least_squares(
                fun, y_hat, bounds=(lo_h, hi_h), method="trf",
                ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=100,
            )
            y = sol.x.copy()
            y[3 * n : 4 * n] = sol.x[2 * n : 3 * n] + sol.x[3 * n : 4 * n] * s0z
            y[6 * n + nf :] = sol.x[6 * n : 6 * n + nf] + sol.x[6 * n + nf :] * s0r
        if sol.cost < 1.0e-10:
            break
    st = _desempaquetar(y, prev, fases)
    st["z"] = prev["z"] + h
    st["h"] = h
    return st, float(sol.cost), bool(sol.success)


def marchar(vin=16.133, n_r=9, h_ini=20.0, h_min=0.25, z_tope=0.0, max_pasos=4000):
    """Marcha desde la base. Devuelve la lista de estados aceptados."""
    st = estado_inicial(vin, n_r)
    historia = [st]
    prev2 = None
    h_prev = h_ini
    h = h_ini
    fases_prev = 1
    for _ in range(max_pasos):
        if st["z"] >= z_tope - 1.0e-8:
            break
        h = min(h, z_tope - st["z"])
        hay_gas = float(np.max(_n_henry(st["P"], st["xi"]))) > 1.0e-6 or float(np.max(st["phi"])) > 1.0e-5
        fases = 2 if hay_gas else 1
        if fases != fases_prev:
            h = min(h, 0.5)
            fases_prev = fases
        guess = _semilla(st, h, prev2, h_prev) if fases == 2 else None
        nuevo, costo, ok = _picard(st, h, prev2, h_prev, fases, guess=guess)
        salto = float(np.max(nuevo["phi"]) - np.max(st["phi"]))
        ug_max = float(np.max(nuevo["ugz"]))
        if costo > 8.0e-4 or salto > 0.02 or ug_max > 0.98 * CS or float(np.max(np.abs(nuevo["umr"]))) > 0.05:
            if h <= h_min * 1.01:
                st["corte"] = {
                    "costo": costo,
                    "ok": ok,
                    "salto_phi": salto,
                    "ug_max": ug_max,
                    "cs": CS,
                    "z_intento": st["z"] + h,
                }
                break
            h = max(h * 0.5, h_min)
            continue
        prev2 = st
        h_prev = h
        st = nuevo
        historia.append(st)
        if len(historia) % 25 == 0 or fases == 2 and len(historia) % 10 == 0:
            print(
                f"z={st['z']:.1f} h={h:.3f} fases={fases} costo={costo:.2e} "
                f"P={st['P'][0]/1e6:.2f} phi={float(np.max(st['phi'])):.4f} "
                f"ug={float(np.max(st['ugz'])):.1f} ur={float(np.max(np.abs(st['umr']))):.3e} "
                f"ugr={float(np.max(np.abs(st['ugr']))):.3e}",
                flush=True,
            )
        # paso: se agranda si el residual quedó holgado y no hay gas nuevo
        if costo < 1.0e-8 and salto < 0.01 and fases == 1:
            h = min(h * 1.3, 40.0)
        elif costo < 1.0e-5:
            if fases == 2 and float(np.max(st["phi"])) < 0.05:
                tope_h = 0.4
            elif fases == 2:
                tope_h = 4.0
            else:
                tope_h = 40.0
            h = min(h * 1.2, tope_h)
    return historia


def resumen(historia):
    z = np.array([s["z"] for s in historia])
    P_eje = np.array([s["P"][0] for s in historia])
    P_pared = np.array([s["P"][-1] for s in historia])
    um_eje = np.array([s["umz"][0] for s in historia])
    ug_eje = np.array([s["ugz"][0] for s in historia])
    ur_max = np.array([np.max(np.abs(s["umr"])) for s in historia])
    ugr_max = np.array([np.max(np.abs(s["ugr"])) for s in historia])
    phi_eje = np.array([s["phi"][0] for s in historia])
    phi_media = np.array([np.trapezoid(s["phi"] * s["r"], s["r"]) * 2.0 / R_COND ** 2 for s in historia])
    Q = np.array([_Q(s) for s in historia])
    return {
        "z": z,
        "P_eje": P_eje,
        "P_pared": P_pared,
        "um_eje": um_eje,
        "ug_eje": ug_eje,
        "ur_max": ur_max,
        "ugr_max": ugr_max,
        "phi_eje": phi_eje,
        "phi_media": phi_media,
        "Q": Q,
        "r": historia[0]["r"],
        "ultimo": historia[-1],
        "corte": historia[-1].get("corte"),
    }


def guardar(historia, ruta_npz, ruta_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    info = resumen(historia)
    np.savez(
        ruta_npz,
        z=info["z"],
        P_eje=info["P_eje"],
        P_pared=info["P_pared"],
        um_eje=info["um_eje"],
        ug_eje=info["ug_eje"],
        ur_max=info["ur_max"],
        ugr_max=info["ugr_max"],
        phi_eje=info["phi_eje"],
        phi_media=info["phi_media"],
        Q=info["Q"],
        r=info["r"],
        P_final=info["ultimo"]["P"],
        phi_final=info["ultimo"]["phi"],
        umz_final=info["ultimo"]["umz"],
        umr_final=info["ultimo"]["umr"],
        ugz_final=info["ultimo"]["ugz"],
        ugr_final=info["ultimo"]["ugr"],
    )
    fig, ax = plt.subplots(2, 3, figsize=(11.2, 6.4), sharex=True)
    z = info["z"]
    series = [
        (info["P_eje"] / 1e6, info["P_pared"] / 1e6, "P [MPa]"),
        (info["phi_eje"], info["phi_media"], r"$\phi$"),
        (info["um_eje"], info["ug_eje"], "u eje [m/s]"),
        (info["ur_max"], info["ugr_max"], r"max $|u_r|$ [m/s]"),
        (info["Q"] / info["Q"][0], None, r"$Q/Q_0$"),
    ]
    for k, (a, b, titulo) in enumerate(series):
        eje = ax.flat[k]
        eje.plot(z, a, color="C0", lw=1.4, label="eje" if b is not None else r"$Q/Q_0$")
        if b is not None:
            eje.plot(z, b, color="C1", lw=1.2, label="pared" if k < 2 else "gas")
            eje.legend(frameon=False, fontsize=8)
        eje.set_title(titulo)
        eje.grid(True, alpha=0.3)
    rf = info["r"] if info["ultimo"]["umr"].size == info["r"].size else 0.5 * (info["r"][:-1] + info["r"][1:])
    ax.flat[5].plot(rf, info["ultimo"]["umr"], color="C0", label=r"$u_{m,r}$")
    ax.flat[5].plot(rf, info["ultimo"]["ugr"], color="C1", label=r"$u_{g,r}$")
    ax.flat[5].set_title(f"u_r en z={info['ultimo']['z']:.0f} m")
    ax.flat[5].set_xlabel("r [m]")
    ax.flat[5].legend(frameon=False, fontsize=8)
    ax.flat[5].grid(True, alpha=0.3)
    for eje in ax[1, :2]:
        eje.set_xlabel("z [m]")
    fig.tight_layout()
    fig.savefig(ruta_png, dpi=140)
    plt.close(fig)
    return info


if __name__ == "__main__":
    hist = marchar(vin=16.133, n_r=9, h_ini=25.0, z_tope=0.0, max_pasos=2500)
    info = guardar(
        hist,
        "/tmp/conducto_ur.npz",
        "/tmp/conducto_ur.png",
    )
    ultimo = info["ultimo"]
    print(f"pasos {len(hist)-1}  z_final {ultimo['z']:.2f} m")
    print(f"P eje {info['P_eje'][-1]/1e6:.3f} MPa   phi eje {info['phi_eje'][-1]:.4f}")
    print(f"um eje {info['um_eje'][-1]:.2f}  ug eje {info['ug_eje'][-1]:.2f}  cs {CS:.1f}")
    print(f"|u_mr| max {info['ur_max'][-1]:.4e}   |u_gr| max {info['ugr_max'][-1]:.4e}")
    print(f"Q inicio {info['Q'][0]:.4e}  Q final {info['Q'][-1]:.4e}")
    print("corte", info["corte"])
