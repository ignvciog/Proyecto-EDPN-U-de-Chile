"""
Columna axisimétrica en diferencias finitas, de la base a la fragmentación.

Hasta la saturación solo está el fundido. Desde que Henry suelta agua
se resuelven, en cada paso, las mismas seis ecuaciones:

    masa del fundido, masa del gas,
    momento axial de las dos fases, momento radial de las dos fases,

más el transporte de N y de ξ. No se impone u_g = u_m ni u_g ≠ u_m.
Al cruzar φ_crit cambian los cierres: n pierde (1-ξ), el arrastre pasa
al de fragmentos, Γ_N = Γ_ξ = 0 y el no-deslizamiento de la pared pasa
del fundido al gas. El laplaciano viscoso lleva la derivada radial y
la derivada en z, y el momento radial lleva el término geométrico.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf

from calbuco2015d import (
    C1,
    F1,
    F2,
    Fc,
    Patm,
    R,
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
    p2o5,
    sio2,
    tcar,
    tio2,
    xmax,
)
from fvrel import fvrel
from viscosity import viscosity

from RIconduit2D_sinbub import (
    CO,
    P_BASE,
    P_SAT,
    R_COND,
    XI0,
    _d_dr,
    _div,
    _lap_ur,
    _lap_uz,
    _ur_nodos,
    marchar,
    rho_de,
)

N0 = 1.0e8
RV = float(R)
T_GAS = float(T1)
MU_G = 1.0e-5
RA = 1.0e-3
CD = 0.8
C_J4 = 3.0 * CD / (8.0 * RA)  # 300 m^{-1}
# Corte del fundido ya fragmentado. μ del fundido (~10^5 Pa·s) a escala del
# conducto frena el núcleo contra la pared. Este valor solo regulariza la
# capa junto a la pared.
MU_PIRO = 8.0e3
CS = math.sqrt(RV * T_GAS)
PRES = 2.5e4  # escala de ∂P/∂z, Pa/m, solo para adimensionalizar el resbalamiento


def n_de(P, xi, fragmentado):
    """Fracción másica exsuelta. Después de fragmentar, sin (1-ξ)."""
    P = np.maximum(np.asarray(P, dtype=float), 1.0e4)
    a = C1 * np.power(P, beta)
    den = np.maximum(1.0 - a, 1.0e-12)
    if fragmentado:
        return np.maximum((CO - a) / den, 0.0)
    xi = np.asarray(xi, dtype=float)
    num = (1.0 - XI0) * CO - (1.0 - xi) * a
    return np.maximum(num / den, 0.0)


def phi_henry(P, xi=None):
    """Volumen de equilibrio, solo para el capilar en φ* = 0.2."""
    if xi is None:
        xi = XI0
    n = n_de(P, xi, False)
    rho_m = rho_de(P)
    rho_g = np.maximum(np.asarray(P, dtype=float), 1.0e4) / (RV * T_GAS)
    phi = np.zeros_like(n, dtype=float)
    vivo = n > 1.0e-12
    phi[vivo] = 1.0 / (1.0 + (rho_g[vivo] / rho_m[vivo]) * (1.0 - n[vivo]) / n[vivo])
    return phi


def _umbrales(vin=15.5):
    """φ1, φ2 y φ_crit una vez, en el estado φ* = 0.2."""
    lo, hi = 1.0e5, float(P_SAT)
    Pstar = 0.5 * (lo + hi)
    for _ in range(60):
        phi = float(phi_henry(Pstar))
        if abs(phi - 0.2) < 0.002:
            break
        if phi < 0.2:
            hi = Pstar
        else:
            lo = Pstar
        Pstar = 0.5 * (lo + hi)
    nstar = float(n_de(Pstar, XI0, False))
    rho_m = float(rho_de(Pstar))
    rho_g = Pstar / (RV * T_GAS)
    a = C1 * Pstar ** beta
    dndp = (1.0 - XI0) * (CO - 1.0) * C1 * beta * Pstar ** (beta - 1.0) / max((1.0 - a) ** 2, 1.0e-30)
    razon = (rho_g / rho_m) * (1.0 - nstar) / max(nstar, 1.0e-16)
    drazon = (
        (1.0 / (rho_m * RV * T_GAS)) * (1.0 - nstar) / max(nstar, 1.0e-16)
        - (rho_g / rho_m) * dndp / max(nstar ** 2, 1.0e-30)
    )
    dphidp = -drazon / max((1.0 + razon) ** 2, 1.0e-30)
    phi = float(phi_henry(Pstar))
    mu = _mu_nodo(Pstar, phi, XI0, N0, vin / R_COND)
    rho_c = (1.0 - phi) * rho_m + phi * rho_g
    dpdz = -rho_c * g - 8.0 * mu * vin / R_COND ** 2
    dvdz = vin * (-dndp * (1.0 - phi) + (1.0 - nstar) * dphidp) / max((1.0 - phi) ** 2, 1.0e-30) * dpdz
    gdot = 0.5 * (abs(dvdz) + vin / R_COND)
    rb = (phi / ((4.0 / 3.0) * math.pi * N0 * max(1.0 - phi, 1.0e-6))) ** (1.0 / 3.0)
    Ca = abs(gdot * mu * rb / 0.3)
    phicrit = ((0.785 - 0.525) / 2.0) * erf(math.log10(max(Ca, 1.0e-30))) + (0.785 + 0.525) / 2.0
    phi1 = ((0.15 - 0.40) / 2.0) * erf(math.log10(max(Ca, 1.0e-30))) + (0.15 + 0.40) / 2.0
    return {
        "Pstar": float(Pstar),
        "Ca": float(Ca),
        "phicrit": float(phicrit),
        "phi1": float(phi1),
        "phi2": float(phi1 + 0.01),
        "rb": float(rb),
    }


def _radio_burbuja(phi, Nd):
    phi = np.clip(np.asarray(phi, dtype=float), 0.0, 0.98)
    Nd = np.maximum(np.asarray(Nd, dtype=float), 1.0)
    vivo = phi > 1.0e-10
    rb = np.full(np.shape(phi), 1.0e-6)
    rb[vivo] = (
        phi[vivo] / ((4.0 / 3.0) * math.pi * Nd[vivo] * np.maximum(1.0 - phi[vivo], 1.0e-6))
    ) ** (1.0 / 3.0)
    return np.maximum(rb, 1.0e-8)


def _corte(umz, dr, u_prev=None, h=None):
    n = umz.size
    g_r = np.zeros(n)
    g_r[0] = abs(umz[1] - umz[0]) / dr
    g_r[-1] = abs(umz[-1] - umz[-2]) / dr
    if n > 2:
        g_r[1:-1] = np.abs(umz[2:] - umz[:-2]) / (2.0 * dr)
    if u_prev is None or h is None:
        return np.maximum(g_r, 1.0e-3)
    g_z = np.abs(umz - u_prev) / max(h, 1.0e-6)
    return np.maximum(np.sqrt(g_r ** 2 + g_z ** 2), 1.0e-3)


def _mu_nodo(P, phi, xi, Nd, gdot, phicrit=0.75, fragmentado=False):
    P = float(P)
    if float(n_de(P, xi, False)) <= 0.0 and float(phi) <= 1.0e-8:
        h2o = float(h2o1)
    else:
        h2o = float(C1 * max(P, 1.0e4) ** beta * 100.0)
    vis = float(viscosity(sio2, tio2, al2o3, feo, mno, mgo, cao, na2o, k2o, p2o5, h2o, f2o, Tc1))
    vis *= float(fvrel(model, float(xi), XI0, ar1, ar2, xmax, max(float(gdot), 1.0e-3)))
    # Después de fragmentar el fundido ya no es la fase continua: no va el factor de suspensión.
    if fragmentado:
        return max(vis, 1.0)
    phi = float(phi)
    if phi < 1.0e-8:
        return max(vis, 1.0)
    phistar = float(phicrit) + 0.05
    if phi >= 0.99 * phistar:
        return max(vis, 1.0)
    phi_c = min(phi, 0.9 * phistar)
    rb = float(_radio_burbuja(np.array([phi_c]), np.array([float(Nd)]))[0])
    nca = max(rb * vis * max(float(gdot), 1.0e-6) / 3.0, 1.0e-30)
    AA = (1.0 - phi_c / phistar) ** (-phistar)
    BB = (1.0 - phi_c / phistar) ** (5.0 * phistar / 3.0)
    c1 = -0.2895 * phi_c + 0.8132
    factor = 0.5 * (AA - BB) * (1.0 - math.erf(c1 * math.log(nca) + phi_c)) + BB
    return max(float(factor) * vis, 1.0)


def viscosidad(P, phi, xi, umz, Nd, dr, phicrit, u_prev=None, h=None, fragmentado=False):
    gdot = _corte(umz, dr, u_prev, h)
    return np.array([
        _mu_nodo(
            float(P[i]), float(phi[i]), float(xi[i]), float(Nd[i]),
            float(gdot[i]), phicrit, fragmentado,
        )
        for i in range(P.size)
    ])


def coef_arrastre(phi, rb, mu, rho_g, speed, phi1, phi2, phicrit, phi_regimen=None):
    """C tal que F = C (u_g - u_m), con el régimen que pide el φ local.

    phi_regimen, si viene, elige la rama (Stokes, empalme, permeabilidad,
    fragmentos) sin cambiar el factor φ(1-φ). Sirve para cruzar el empalme
    con continuación: el último peso usa el φ de verdad.
    """
    phi = np.clip(np.asarray(phi, dtype=float), 0.0, 0.98)
    reg = phi if phi_regimen is None else np.clip(np.asarray(phi_regimen, dtype=float), 0.0, 0.98)
    rb = np.maximum(np.asarray(rb, dtype=float), 1.0e-8)
    mu = np.maximum(np.asarray(mu, dtype=float), 1.0)
    rho_g = np.maximum(np.asarray(rho_g, dtype=float), 1.0e-6)
    speed = np.maximum(np.asarray(speed, dtype=float), 0.0)
    fac = phi * (1.0 - phi)
    stokes = 3.0 * mu / rb ** 2
    speed_f = np.maximum(speed, 1.0e-8)
    Re = 2.0 * rb * rho_g * speed_f / MU_G
    kper = 0.131 * rb ** 2 * np.maximum(phi - phi1 + 0.05, 0.05) ** 2.1
    darcy = MU_G / np.maximum(kper, 1.0e-30)
    inercial = 0.33 * rho_g * speed_f / (4.0 * rb)
    perm = np.where(Re > 2200.0, inercial, darcy)
    # Transiciones suaves: el empalme a trozos dejaba un salto de C en φ_crit
    # y el Newton se detenía sin raíz. Los extremos siguen siendo Stokes,
    # permeabilidad y el arrastre de fragmentos.
    tt = _suave((reg - phi1) / max(phi2 - phi1, 1.0e-8))
    base = np.where(reg < phi2, perm ** tt * stokes ** (1.0 - tt), perm)
    puro = C_J4 * rho_g * speed_f
    tt4 = _suave((reg - phicrit) / 0.05)
    mezcla = np.maximum(perm, 1.0e-30) ** (1.0 - tt4) * np.maximum(puro, 1.0e-30) ** tt4
    base = np.where(reg >= phicrit, mezcla, base)
    return np.where(phi < 1.0e-8, 0.0, base * fac)


def _suave(x):
    """Polinomio con derivada nula en 0 y en 1, para que C(φ) sea C1."""
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _gamma_xi(P, xi):
    den = (1.0 - XI0) * CO - (1.0 - xmax) * C1 * Patm ** beta
    den = max(float(den), 1.0e-30)
    num = (1.0 - XI0) * CO - (1.0 - xi) * C1 * np.maximum(P, 1.0e4) ** beta
    f2 = np.maximum(num / den, 0.0)
    xi_eq = XI0 + (xmax - XI0) * f2
    f3 = np.maximum(1.0 - xi / np.maximum(xi_eq, 1.0e-8), 0.0)
    return np.maximum((xmax - XI0) * f2 * f3 / tcar, 0.0)


def _gamma_N(P, phi, Nd, mu, phicrit):
    phi = np.asarray(phi, dtype=float)
    Nd = np.maximum(np.asarray(Nd, dtype=float), 1.0)
    mu = np.maximum(np.asarray(mu, dtype=float), 1.0)
    rho_g = np.maximum(P, 1.0e4) / (RV * T_GAS)
    rho_m = rho_de(P)
    rb = _radio_burbuja(phi, Nd)
    vivo = (phi > 1.0e-8) & (phi < phicrit) & (rb < 0.5 * R_COND)
    inner = (F1 + F2) * (
        (3.0 * np.maximum(phi, 0.0) * math.pi / (6.0 * phicrit)) / (4.0 * math.pi)
    ) ** (1.0 / 3.0)
    den = 1.0 - inner
    tasa = (
        -(Nd ** (2.0 / 3.0))
        * np.maximum(1.0 - phi, 1.0e-6) ** (-1.0 / 3.0)
        * ((rho_m - rho_g) * g / (9.0 * mu))
        * (3.0 * np.maximum(phi, 0.0) / (4.0 * math.pi)) ** (2.0 / 3.0)
        * (F1 ** 2 - F2 ** 2)
        / np.maximum(den, 1.0e-8)
        * Fc
        * (1.0 - phi / phicrit)
        * (R_COND - rb) / R_COND
    )
    return np.where(vivo & (den > 1.0e-8), tasa, 0.0)


def _nucleacion(st, prev):
    """N(r) de Toramaru donde ese radio cruza la saturación. En la pared no se evalúa."""
    h = max(st["z"] - prev["z"], 1.0e-6)
    dPdz = (st["P"] - prev["P"]) / h
    dPdr = _d_dr(st["r"], st["P"])
    ur = _ur_nodos(st["umr"])
    Pdot = -(ur * dPdr + st["umz"] * dPdz)
    N = np.full(st["P"].size, N0)
    vivo = (st["umz"] > 0.05) & (Pdot > 1.0)
    N[vivo] = np.power(10.0, 1.5 * np.log10(Pdot[vivo]) + 5.0)
    if not vivo[0]:
        N[0] = N[1] if N.size > 1 else N0
    for i in range(1, N.size):
        if not vivo[i]:
            N[i] = N[i - 1]
    return np.clip(N, 1.0e6, 1.0e16)


def _cortes(n):
    nf = n - 1
    i = 0
    s = {}
    s["P"] = slice(i, i + n)
    i += n
    s["phi"] = slice(i, i + n)
    i += n
    s["um"] = slice(i, i + nf)
    i += nf
    s["hz"] = slice(i, i + nf)
    i += nf
    s["ur"] = slice(i, i + nf)
    i += nf
    s["hr"] = slice(i, i + nf)
    i += nf
    s["xi"] = slice(i, i + n)
    i += n
    s["N"] = slice(i, i + n)
    i += n
    s["m"] = i
    return s


def _ug_flujo(ugz, fragmentado):
    """Velocidad axial del gas que entra en el flujo de masa.

    En la pared fragmentada el nodo vale 0, pero la media-celda no.
    """
    u = np.array(ugz, dtype=float, copy=True)
    if fragmentado and u.size >= 2:
        u[-1] = u[-2]
    return u


def _aplicar_pared(um_int, ug_int, fragmentado):
    n = um_int.size + 1
    umz = np.zeros(n)
    ugz = np.zeros(n)
    umz[:-1] = um_int
    ugz[:-1] = ug_int
    if fragmentado:
        umz[-1] = umz[-2]
        ugz[-1] = 0.0
    else:
        umz[-1] = 0.0
        ugz[-1] = ugz[-2]
    return umz, ugz


def _velocidad(y, plantilla, s0z, s0r, fragmentado):
    n = plantilla["r"].size
    sl = _cortes(n)
    P = np.maximum(y[sl["P"]], Patm)
    phi = np.clip(y[sl["phi"]], 0.0, 0.98)
    um_int = y[sl["um"]]
    hatz = y[sl["hz"]]
    ug_int = um_int + hatz * s0z[:-1]
    umz, ugz = _aplicar_pared(um_int, ug_int, fragmentado)
    umr = y[sl["ur"]].copy()
    ugr = umr + y[sl["hr"]] * s0r
    xi = np.clip(y[sl["xi"]], XI0 * 0.5, float(xmax))
    Nd = np.clip(y[sl["N"]], 1.0e6, 1.0e16)
    return P, phi, umz, ugz, umr, ugr, xi, Nd


def _empaquetar(st, s0z, s0r):
    hatz = (st["ugz"][:-1] - st["umz"][:-1]) / np.maximum(s0z[:-1], 1.0e-16)
    hatr = (st["ugr"] - st["umr"]) / np.maximum(s0r, 1.0e-16)
    return np.concatenate([
        st["P"], st["phi"], st["umz"][:-1], hatz, st["umr"], hatr, st["xi"], st["N"],
    ])


def _adveccion_radial(ur, campo, r):
    dr = r[1] - r[0]
    out = np.zeros_like(campo)
    for i in range(1, campo.size - 1):
        if ur[i] >= 0.0:
            out[i] = ur[i] * (campo[i] - campo[i - 1]) / dr
        else:
            out[i] = ur[i] * (campo[i + 1] - campo[i]) / dr
    return out


def _axial(lap, mu_eff, u, u_prev, mu_prev, u_prev2, h, h_prev):
    flujo = mu_eff * (u - u_prev) / h
    if u_prev2 is None:
        return lap + flujo / h
    flujo_prev = mu_prev * (u_prev - u_prev2) / max(h_prev, 1.0e-8)
    return lap + (flujo - flujo_prev) / (0.5 * (h + h_prev))


def _upwind_cara(ur, r):
    """Derivada radial centrada en las caras, con u_r = 0 en el eje y en la pared.

    El tramo líquido usa esta misma derivada y no alimenta el modo de damero.
    """
    x = np.concatenate([[0.0], 0.5 * (r[:-1] + r[1:]), [r[-1]]])
    return np.gradient(np.concatenate([[0.0], ur, [0.0]]), x)[1:-1]


def residual_bifasico(y, prev, h, mu, s0z, s0r, prev2, h_prev, fragmentado, umb, peso=1.0):
    r = prev["r"]
    dr = prev["dr"]
    n = r.size
    P, phi, umz, ugz, umr, ugr, xi, Nd = _velocidad(y, prev, s0z, s0r, fragmentado)
    rho_m = rho_de(P)
    rho_g = np.maximum(P, 1.0e4) / (RV * T_GAS)
    rho_m0 = rho_de(prev["P"])
    rho_g0 = np.maximum(prev["P"], 1.0e4) / (RV * T_GAS)
    umr_n = _ur_nodos(umr)
    ugr_n = _ur_nodos(ugr)
    phi_f = 0.5 * (phi[:-1] + phi[1:])
    rho_mf = 0.5 * (rho_m[:-1] + rho_m[1:])
    rho_gf = 0.5 * (rho_g[:-1] + rho_g[1:])
    mu_f = 0.5 * (mu[:-1] + mu[1:])

    n_new = n_de(P, xi, fragmentado)
    n_old = n_de(prev["P"], prev["xi"], prev["fragmentado"])
    # u_g(R)=0 es la condición puntual de la pared. La media-celda tiene un
    # espesor Δr y la capa límite del gas es mucho más fina: el flujo axial de
    # esa celda es el del nodo interior. El nivel anterior tiene que usar la
    # misma reconstrucción. Si se le resta el u_g(R)=0 guardado, cada paso
    # aparece un flujo de gas que no estaba y el conducto se frena.
    ugz_m = _ug_flujo(ugz, fragmentado)
    ugz_0 = _ug_flujo(prev["ugz"], prev["fragmentado"])
    jz = rho_m * (1.0 - phi) * umz + rho_g * phi * ugz_m
    jr = rho_m * (1.0 - phi) * umr_n + rho_g * phi * ugr_n
    Gamma = jr * _d_dr(r, n_new) + jz * (n_new - n_old) / h

    jz_m0 = rho_m0 * (1.0 - prev["phi"]) * prev["umz"]
    jz_g0 = rho_g0 * prev["phi"] * ugz_0
    masa_m = (rho_m * (1.0 - phi) * umz - jz_m0) / h + _div(r, dr, rho_mf * (1.0 - phi_f) * umr) + Gamma
    masa_g = (rho_g * phi * ugz_m - jz_g0) / h + _div(r, dr, rho_gf * phi_f * ugr) - Gamma

    # El resbalamiento real es 10^{-12} m/s y desaparece al sumarlo a u.
    # La fuerza se evalúa con el hat del sistema, sin pasar por u_g - u_m.
    sl = _cortes(n)
    hatz = np.zeros(n)
    hatz[:-1] = y[sl["hz"]]
    hatr = y[sl["hr"]]
    slip_z = hatz * s0z
    slip_r = hatr * s0r
    slip_r_n = _ur_nodos(slip_r)
    peso = float(np.clip(peso, 0.0, 1.0))
    reg = (1.0 - peso) * prev["phi"] + peso * phi
    reg_f = 0.5 * (reg[:-1] + reg[1:])
    speed_z = np.sqrt(slip_z ** 2 + slip_r_n ** 2)
    rb = _radio_burbuja(phi, Nd)
    Cz = coef_arrastre(
        phi, rb, mu, rho_g, speed_z, umb["phi1"], umb["phi2"], umb["phicrit"], reg,
    )
    Fz = Cz * slip_z
    slip_r_speed = np.sqrt((0.5 * (slip_z[:-1] + slip_z[1:])) ** 2 + slip_r ** 2)
    rb_f = _radio_burbuja(phi_f, 0.5 * (Nd[:-1] + Nd[1:]))
    Cr = coef_arrastre(
        phi_f, rb_f, mu_f, rho_gf, slip_r_speed,
        umb["phi1"], umb["phi2"], umb["phicrit"], reg_f,
    )
    Fr = Cr * slip_r

    dPdz = (P - prev["P"]) / h
    adv_mz = umz * (umz - prev["umz"]) / h + _adveccion_radial(umr_n, umz, r)
    adv_gz = ugz * (ugz - prev["ugz"]) / h + _adveccion_radial(ugr_n, ugz, r)
    # Después de fragmentar el fundido son piroclastos. μ del fundido a
    # escala del conducto (~10^5 Pa·s) reparte el núcleo rápido hacia la
    # pared y u_z baja. Se deja solo lo necesario para la capa de la pared;
    # el arrastre entre fases queda, y la viscosidad del gas sigue en su laplaciano.
    mu_g = MU_G * np.maximum(phi, 0.0)
    mu_g0 = MU_G * np.maximum(prev["phi"], 0.0)
    if fragmentado:
        mu_m = np.full_like(phi, MU_PIRO) * (1.0 - phi)
        mu_m0 = np.full_like(prev["phi"], MU_PIRO) * (1.0 - prev["phi"])
    else:
        mu_m = mu * (1.0 - phi)
        mu_m0 = prev["mu"] * (1.0 - prev["phi"])
    if prev2 is None:
        um2 = ug2 = None
        mu_m2 = mu_g2 = None
    else:
        um2 = prev2["umz"]
        ug2 = prev2["ugz"]
        mu_g2 = MU_G * np.maximum(prev2["phi"], 0.0)
        if fragmentado:
            mu_m2 = np.full_like(prev2["phi"], MU_PIRO) * (1.0 - prev2["phi"])
        else:
            mu_m2 = prev2["mu"] * (1.0 - prev2["phi"])
    lap_mz = _axial(_lap_uz(r, dr, mu_m, umz), mu_m, umz, prev["umz"], mu_m0, um2, h, h_prev)
    lap_gz = _axial(_lap_uz(r, dr, mu_g, ugz), mu_g, ugz, prev["ugz"], mu_g0, ug2, h, h_prev)
    mom_mz = (
        rho_m * (1.0 - phi) * adv_mz
        + (1.0 - phi) * dPdz
        + rho_m * (1.0 - phi) * g
        - Fz
        - lap_mz
    )
    mom_gz = rho_g * phi * adv_gz + phi * dPdz + rho_g * phi * g + Fz - lap_gz

    umz_f = 0.5 * (umz[:-1] + umz[1:])
    ugz_f = 0.5 * (ugz[:-1] + ugz[1:])
    dPdr = (P[1:] - P[:-1]) / dr
    if fragmentado:
        mu_mf = np.full_like(phi_f, MU_PIRO) * (1.0 - phi_f)
    else:
        mu_mf = mu_f * (1.0 - phi_f)
    mu_gf = MU_G * np.maximum(phi_f, 0.0)
    mu_mf0 = 0.5 * (mu_m0[:-1] + mu_m0[1:])
    mu_gf0 = 0.5 * (mu_g0[:-1] + mu_g0[1:])
    if prev2 is None:
        ur2 = ugr2 = None
        mu_mf2 = mu_gf2 = None
    else:
        ur2 = prev2["umr"]
        ugr2 = prev2["ugr"]
        mu_g2f = MU_G * np.maximum(prev2["phi"], 0.0)
        if fragmentado:
            mu_m2f = np.full_like(prev2["phi"], MU_PIRO) * (1.0 - prev2["phi"])
        else:
            mu_m2f = prev2["mu"] * (1.0 - prev2["phi"])
        mu_mf2 = 0.5 * (mu_m2f[:-1] + mu_m2f[1:])
        mu_gf2 = 0.5 * (mu_g2f[:-1] + mu_g2f[1:])
    lap_mr = _axial(
        _lap_ur(r, mu_mf, umr), mu_mf, umr, prev["umr"], mu_mf0, ur2, h, h_prev,
    )
    lap_gr = _axial(
        _lap_ur(r, mu_gf, ugr), mu_gf, ugr, prev["ugr"], mu_gf0, ugr2, h, h_prev,
    )
    dumr = _upwind_cara(umr, r)
    dugr = _upwind_cara(ugr, r)
    mom_mr = (
        rho_mf * (1.0 - phi_f) * (umz_f * (umr - prev["umr"]) / h + umr * dumr)
        + (1.0 - phi_f) * dPdr
        - Fr
        - lap_mr
    )
    mom_gr = (
        rho_gf * phi_f * (ugz_f * (ugr - prev["ugr"]) / h + ugr * dugr)
        + phi_f * dPdr
        + Fr
        - lap_gr
    )

    if fragmentado:
        gxi = np.zeros(n)
        gN = np.zeros(n)
    else:
        gxi = _gamma_xi(P, xi)
        gN = _gamma_N(P, phi, Nd, mu, umb["phicrit"])
    xi_eq = umz * (xi - prev["xi"]) / h + _adveccion_radial(umr_n, xi, r) - gxi
    N_eq = umz * (Nd - prev["N"]) / h + _adveccion_radial(umr_n, Nd, r) - gN

    esc_m = max(float(np.max(np.abs(jz_m0))) / h, 1.0)
    esc_g = max(
        float(np.max(np.abs(jz_g0))) / h,
        float(np.max(rho_g0)) * 1.0e-4 * max(float(np.max(prev["umz"])), 1.0) / h,
    )
    esc_z = max(float(np.max(rho_m0 * np.maximum(1.0 - prev["phi"], 0.05) * g)), 1.0)
    # Misma escala física para las dos fases: ρ φ g. Dividir por el arrastre
    # escondía un F de cientos de Pa/m detrás de un resbalamiento de 10^{-11} m/s.
    esc_gz = np.maximum(rho_g * np.maximum(phi, 1.0e-5) * g, 1.0e-2)
    esc_gr = np.maximum(0.5 * (esc_gz[:-1] + esc_gz[1:]), 1.0e-2)
    u_ref = max(float(np.max(np.abs(prev["umz"]))), 1.0)
    N_ref = max(float(np.max(prev["N"])), 1.0)
    masa_m = masa_m / esc_m
    masa_g = masa_g / esc_g
    mom_mz = mom_mz[:-1] / esc_z
    mom_gz = mom_gz[:-1] / esc_gz[:-1]
    mom_mr = mom_mr / esc_z
    mom_gr = mom_gr / esc_gr
    xi_eq = xi_eq / 1.0e-6
    N_eq = N_eq * h / (u_ref * N_ref)
    # En la pared la ecuación de transporte degenera: se impone derivada radial nula.
    xi_eq[-1] = (xi[-1] - xi[-2]) / 1.0e-4
    N_eq[-1] = (Nd[-1] - Nd[-2]) / N_ref
    # Sin agua exsuelta no hay fase gas: φ = 0 y el gas viaja con el fundido.
    seco = (n_new <= 1.0e-12) & (n_old <= 1.0e-12) & (prev["phi"] <= 1.0e-8)
    masa_g = np.where(seco, phi / 1.0e-4, masa_g)
    mom_gz = np.where(seco[:-1], (ugz[:-1] - umz[:-1]) / np.maximum(s0z[:-1], 1.0e-8), mom_gz)
    seco_f = seco[:-1] & seco[1:]
    mom_gr = np.where(seco_f, (ugr - umr) / np.maximum(s0r, 1.0e-8), mom_gr)
    return np.concatenate([masa_m, masa_g, mom_mz, mom_gz, mom_mr, mom_gr, xi_eq, N_eq])


def _escalas(st, mu, umb):
    n = st["r"].size
    rho_g = np.maximum(st["P"], 1.0e4) / (RV * T_GAS)
    umr_n = _ur_nodos(st["umr"])
    ugr_n = _ur_nodos(st["ugr"])
    speed = np.sqrt((st["ugz"] - st["umz"]) ** 2 + (ugr_n - umr_n) ** 2)
    rb = _radio_burbuja(st["phi"], st["N"])
    C = coef_arrastre(st["phi"], rb, mu, rho_g, speed, umb["phi1"], umb["phi2"], umb["phicrit"])
    s0z = PRES * np.maximum(st["phi"], 1.0e-6) / np.maximum(C, 1.0e-3)
    s0z = np.clip(s0z, 1.0e-12, 25.0)
    s0r = np.clip(0.5 * (s0z[:-1] + s0z[1:]), 1.0e-12, 25.0)
    return s0z, s0r


def _predecir(prev, prev2, h, fragmentado):
    st = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in prev.items()}
    h_prev = max(prev["z"] - prev2["z"], 1.0e-6) if prev2 is not None else h
    if prev2 is not None:
        dPdz = (prev["P"] - prev2["P"]) / h_prev
    else:
        dPdz = -rho_de(prev["P"]) * g
    st["P"] = np.maximum(prev["P"] + dPdz * h, Patm * 1.05)
    n1 = n_de(st["P"], prev["xi"], fragmentado)
    n0 = n_de(prev["P"], prev["xi"], prev["fragmentado"])
    rho_g = np.maximum(st["P"], 1.0e4) / (RV * T_GAS)
    jz = rho_de(prev["P"]) * (1.0 - prev["phi"]) * np.maximum(prev["umz"], 0.0)
    phi = prev["phi"] + jz * np.maximum(n1 - n0, 0.0) / np.maximum(
        rho_g * np.maximum(prev["umz"], 0.3), 1.0,
    )
    phi = np.clip(phi, 0.0, 0.97)
    phi[-1] = phi[-2]
    phi[0] = 0.5 * (phi[0] + phi[1])
    if fragmentado and not prev["fragmentado"]:
        # El salto de n no es dz·dn/dz. φ de equilibrio con el n nuevo, y el
        # caudal de masa se reparte. Sirve solo de semilla del Newton.
        vivo = n1 > 1.0e-8
        phi = np.array(prev["phi"], copy=True)
        rho_m = np.maximum(rho_de(st["P"]), 1.0)
        phi[vivo] = 1.0 / (1.0 + (rho_g[vivo] / rho_m[vivo]) * (1.0 - n1[vivo]) / n1[vivo])
        phi = np.clip(phi, 0.0, 0.95)
        phi[-1] = phi[-2]
        phi[0] = 0.5 * (phi[0] + phi[1])
    st["phi"] = phi
    rho_m = np.maximum(rho_de(st["P"]), 1.0)
    rho_new = np.maximum(rho_m * np.maximum(1.0 - phi, 0.02), 1.0)
    rho_old = np.maximum(rho_de(prev["P"]) * np.maximum(1.0 - prev["phi"], 0.02), 1.0)
    st["umz"] = prev["umz"] * rho_old / rho_new
    st["ugz"] = st["umz"] + (prev["ugz"] - prev["umz"])
    if fragmentado and not prev["fragmentado"]:
        q = jz + (np.maximum(prev["P"], 1.0e4) / (RV * T_GAS)) * prev["phi"] * prev["ugz"]
        st["umz"] = np.maximum((1.0 - n1) * q / rho_new, 0.0)
        st["ugz"] = n1 * q / np.maximum(rho_g * np.maximum(phi, 1.0e-3), 1.0e-6)
    if fragmentado:
        st["umz"][-1] = st["umz"][-2]
        st["ugz"][-1] = 0.0
    else:
        st["umz"][-1] = 0.0
        st["ugz"][-1] = st["ugz"][-2]
    st["umr"] = prev["umr"].copy()
    st["ugr"] = prev["ugr"].copy()
    st["fragmentado"] = fragmentado
    return st


def _cierre_axial(P, phi, i, prev, h, umb):
    """u_m, u_g y los dos momentos axiales, con la masa ya cerrada y u_r = 0."""
    P0 = float(prev["P"][i])
    phi0 = float(prev["phi"][i])
    um0 = float(prev["umz"][i])
    ug0 = float(prev["ugz"][i])
    xi0 = float(prev["xi"][i])
    Nd = float(prev["N"][i])
    rho_m0 = float(rho_de(P0))
    rho_g0 = max(P0, 1.0e4) / (RV * T_GAS)
    melt0 = rho_m0 * (1.0 - phi0) * max(um0, 0.0)
    gas0 = rho_g0 * phi0 * max(ug0, 0.0)
    jz0 = melt0 + gas0
    if jz0 < 30.0:
        return None
    n0 = float(n_de(P0, xi0, False))
    rho_m = float(rho_de(P))
    rho_g = max(P, 1.0e4) / (RV * T_GAS)
    dn = float(n_de(P, xi0, True)) - n0
    melt = melt0 - jz0 * dn
    gas = gas0 + jz0 * dn
    if melt <= 1.0 or gas <= 1.0 or not (0.02 < phi < 0.97):
        return None
    um = melt / (rho_m * (1.0 - phi))
    ug = gas / (rho_g * phi)
    if not (0.0 < um < 700.0 and 0.0 < ug < 700.0):
        return None
    slip = ug - um
    rb = float(_radio_burbuja(np.array([phi]), np.array([Nd]))[0])
    C = float(coef_arrastre(
        np.array([phi]), np.array([rb]), np.array([1.0e4]), np.array([rho_g]),
        np.array([abs(slip)]), umb["phi1"], umb["phi2"], umb["phicrit"],
    )[0])
    F = C * slip
    mom_m = (
        rho_m * (1.0 - phi) * um * (um - um0)
        + (1.0 - phi) * (P - P0)
        + rho_m * (1.0 - phi) * g * h
        - F * h
    )
    mom_g = rho_g * phi * ug * (ug - ug0) + phi * (P - P0) + rho_g * phi * g * h + F * h
    return um, ug, mom_m, mom_g, melt0


def _raiz_axial(prev, h, umb):
    """Raíz 1D del eje, la más cercana al estado anterior."""
    P0 = float(prev["P"][0])
    um0 = float(prev["umz"][0])
    mejor = None
    for P in np.linspace(max(P0 * 0.45, Patm * 3.0), P0 * 0.995, 28):
        for phi in np.linspace(0.55, 0.94, 22):
            st = _cierre_axial(P, phi, 0, prev, h, umb)
            if st is None:
                continue
            um, ug, mom_m, mom_g, melt0 = st
            c = (mom_m / 1.0e5) ** 2 + (mom_g / 1.0e4) ** 2
            cerca = abs(um - um0) / max(um0, 1.0) + abs(P - P0) / P0
            if mejor is None or c < mejor[0] or (c < 1.0 and mejor[0] > 1.0):
                mejor = (c, P, phi, cerca)
            elif c < 1.0e-2 and cerca < mejor[3] and mejor[0] < 1.0e-2:
                mejor = (c, P, phi, cerca)
    if mejor is None:
        return None

    def fun(x):
        st = _cierre_axial(x[0], x[1], 0, prev, h, umb)
        if st is None:
            return np.array([10.0, 10.0])
        return np.array([st[2] / 1.0e5, st[3] / 1.0e4])

    sol = least_squares(
        fun, np.array([mejor[1], mejor[2]]), method="trf",
        bounds=([Patm, 0.05], [P0 * 1.001, 0.96]),
        ftol=1.0e-14, xtol=1.0e-14, gtol=1.0e-14, max_nfev=60,
    )
    st = _cierre_axial(sol.x[0], sol.x[1], 0, prev, h, umb)
    if st is None or sol.cost > 1.0e-8:
        return None
    return float(sol.x[0]), float(sol.x[1]), float(st[0]), float(st[1])


def _semilla_fragmentacion(prev, h, umb):
    """Perfil plano con la raíz axial y el u_r que reparte la masa.

    El salto de n y el cambio de pared no caben en una extrapolación nodo a nodo:
    el eje se acelera y la pared, que estaba quieta, pasa a llevar fundido.
    """
    raiz = _raiz_axial(prev, h, umb)
    st = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in prev.items()}
    if raiz is None:
        return _predecir(prev, None, h, True)
    P, phi, um, ug = raiz
    n = prev["r"].size
    nf = n - 1
    r = prev["r"]
    dr = float(prev["dr"])
    st["P"] = np.full(n, P) + (prev["P"] - prev["P"][0])
    st["phi"] = np.full(n, phi)
    st["umz"] = np.full(n, um)
    st["ugz"] = np.full(n, ug)
    st["umz"][-1] = st["umz"][-2]
    st["ugz"][-1] = 0.0
    st["umr"] = np.zeros(nf)
    st["ugr"] = np.zeros(nf)
    rho_m0 = rho_de(prev["P"])
    rho_g0 = np.maximum(prev["P"], 1.0e4) / (RV * T_GAS)
    for _ in range(5):
        rho_m = rho_de(st["P"])
        rho_g = np.maximum(st["P"], 1.0e4) / (RV * T_GAS)
        n_new = n_de(st["P"], st["xi"], True)
        n_old = n_de(prev["P"], prev["xi"], False)
        ug_flujo = st["ugz"].copy()
        ug_flujo[-1] = st["ugz"][-2]
        umr_n = _ur_nodos(st["umr"])
        ugr_n = _ur_nodos(st["ugr"])
        jz = rho_m * (1.0 - st["phi"]) * st["umz"] + rho_g * st["phi"] * ug_flujo
        jr = rho_m * (1.0 - st["phi"]) * umr_n + rho_g * st["phi"] * ugr_n
        Gamma = jr * _d_dr(r, n_new) + jz * (n_new - n_old) / h
        melt = rho_m * (1.0 - st["phi"]) * st["umz"]
        gas = rho_g * st["phi"] * ug_flujo
        melt0 = rho_m0 * (1.0 - prev["phi"]) * prev["umz"]
        gas0 = rho_g0 * prev["phi"] * prev["ugz"]
        div_m = -(melt - melt0) / h - Gamma
        div_g = -(gas - gas0) / h + Gamma
        rf = 0.5 * (r[:-1] + r[1:])
        Fm = np.zeros(nf)
        Fg = np.zeros(nf)
        Fm[0] = div_m[0] * rf[0] / 2.0
        Fg[0] = div_g[0] * rf[0] / 2.0
        for i in range(1, n - 1):
            Fm[i] = (div_m[i] * r[i] * dr + rf[i - 1] * Fm[i - 1]) / max(rf[i], 1.0e-8)
            Fg[i] = (div_g[i] * r[i] * dr + rf[i - 1] * Fg[i - 1]) / max(rf[i], 1.0e-8)
        phi_f = 0.5 * (st["phi"][:-1] + st["phi"][1:])
        st["umr"] = Fm / np.maximum(0.5 * (rho_m[:-1] + rho_m[1:]) * (1.0 - phi_f), 1.0)
        st["ugr"] = Fg / np.maximum(0.5 * (rho_g[:-1] + rho_g[1:]) * phi_f, 1.0e-6)
    st["fragmentado"] = True
    return st


def _cotas(prev, n, fragmentado):
    """P, φ y u_z quedan libres. u_r solo puede apartarse poco del paso anterior,
    para no saltar a la raíz espuria del momento radial."""
    nf = n - 1
    margen = 15.0 if fragmentado else 2.0
    p_lo = np.full(n, Patm)
    p_hi = np.minimum(np.maximum(prev["P"] * 1.002, Patm * 2.0), P_BASE * 1.02)
    phi_hi = 0.98 if fragmentado else min(0.98, max(float(np.max(prev["phi"])) + 0.04, 0.02))
    lo = np.concatenate([
        p_lo,
        np.zeros(n),
        np.zeros(nf),
        np.full(nf, -40.0),
        np.maximum(prev["umr"] - margen, -40.0),
        np.full(nf, -40.0),
        np.full(n, 0.2),
        np.full(n, 1.0e6),
    ])
    hi = np.concatenate([
        p_hi,
        np.full(n, phi_hi),
        np.full(nf, 0.98 * CS),
        np.full(nf, 40.0),
        np.minimum(prev["umr"] + margen, 40.0),
        np.full(nf, 40.0),
        np.full(n, float(xmax)),
        np.full(n, 1.0e16),
    ])
    return lo, hi


def _paso_bifasico(prev, prev2, h, h_prev, fragmentado, umb):
    cruce = fragmentado and not prev["fragmentado"]
    if cruce:
        guess = _semilla_fragmentacion(prev, h, umb)
    else:
        guess = _predecir(prev, prev2, h, fragmentado)
    mu = viscosidad(
        guess["P"], guess["phi"], guess["xi"], guess["umz"], guess["N"],
        prev["dr"], umb["phicrit"], prev["umz"], h, fragmentado,
    )
    s0z, s0r = _escalas(guess, mu, umb)
    n = prev["r"].size
    nf = n - 1
    lo, hi = _cotas(prev, n, fragmentado)
    if cruce:
        # El reparto radial del salto usa unos m/s; la caja de continuación
        # de antes de fragmentar no alcanza.
        iur = 2 * n + 2 * nf
        lo = lo.copy()
        hi = hi.copy()
        lo[iur:iur + nf] = -20.0
        hi[iur:iur + nf] = 20.0
    y = np.minimum(np.maximum(_empaquetar(guess, s0z, s0r), lo + 1.0e-14), hi - 1.0e-14)
    sol = None
    if cruce:
        nfev = max(1200, 8 * y.size)
    elif fragmentado:
        nfev = max(800, 6 * y.size)
    else:
        nfev = max(80, 3 * y.size)

    def _resolver(y0, mu_f, s0z_f, s0r_f, peso, nfev=nfev):
        def fun(z, mu=mu_f, s0z=s0z_f, s0r=s0r_f, peso=peso):
            return residual_bifasico(
                z, prev, h, mu, s0z, s0r, prev2, h_prev, fragmentado, umb, peso,
            )

        return least_squares(
            fun, y0, bounds=(lo, hi), method="trf", x_scale="jac",
            ftol=1.0e-12, xtol=1.0e-12, gtol=1.0e-12, max_nfev=nfev,
        )

    picard = 3 if cruce else 3
    for _ in range(picard):
        y_aqui = y.copy()
        sol = _resolver(y, mu, s0z, s0r, 1.0)
        if sol.cost > 1.0e-4 and not cruce and not fragmentado:
            # El empalme de arrastre cambia C en órdenes de magnitud. Se cruza
            # con el régimen del paso anterior y recién al final queda el φ real.
            y = y_aqui
            for peso in (0.0, 0.5, 1.0):
                sol = _resolver(y, mu, s0z, s0r, peso, max(80, 3 * y.size))
                y = np.minimum(np.maximum(sol.x, lo + 1.0e-14), hi - 1.0e-14)
                print(f"  continuación peso={peso:.1f} costo={sol.cost:.3e}", flush=True)
        else:
            y = sol.x
        P, phi, umz, ugz, umr, ugr, xi, Nd = _velocidad(y, prev, s0z, s0r, fragmentado)
        guess = dict(prev)
        guess.update({
            "P": P, "phi": phi, "umz": umz, "ugz": ugz, "umr": umr, "ugr": ugr,
            "xi": xi, "N": Nd, "fragmentado": fragmentado,
        })
        mu = viscosidad(P, phi, xi, umz, Nd, prev["dr"], umb["phicrit"], prev["umz"], h, fragmentado)
        s0z, s0r = _escalas(guess, mu, umb)
        y = np.minimum(np.maximum(_empaquetar(guess, s0z, s0r), lo + 1.0e-14), hi - 1.0e-14)
        if sol.cost < 1.0e-4:
            break
    P, phi, umz, ugz, umr, ugr, xi, Nd = _velocidad(y, prev, s0z, s0r, fragmentado)
    nuevo = {
        "z": prev["z"] + h,
        "r": prev["r"],
        "dr": prev["dr"],
        "P": P,
        "phi": phi,
        "umz": umz,
        "ugz": ugz,
        "umr": umr,
        "ugr": ugr,
        "xi": xi,
        "N": Nd,
        "mu": mu,
        "fragmentado": fragmentado,
        "costo": float(sol.cost),
    }
    return nuevo, float(sol.cost)


def _desde_liquido(s):
    n = s["r"].size
    ur = np.asarray(s["ur"], dtype=float).copy()
    return {
        "z": float(s["z"]),
        "r": s["r"],
        "dr": s["dr"],
        "P": np.asarray(s["P"], dtype=float).copy(),
        "phi": np.zeros(n),
        "umz": np.asarray(s["uz"], dtype=float).copy(),
        "ugz": np.asarray(s["uz"], dtype=float).copy(),
        "umr": ur,
        "ugr": ur.copy(),
        "xi": np.full(n, XI0),
        "N": np.full(n, N0),
        "mu": np.asarray(s["mu"], dtype=float).copy(),
        "fragmentado": False,
        "costo": float(s.get("costo", 0.0)),
    }


def _caudal(st):
    r = st["r"]
    rho_m = rho_de(st["P"])
    rho_g = np.maximum(st["P"], 1.0e4) / (RV * T_GAS)
    ug = _ug_flujo(st["ugz"], st.get("fragmentado", False))
    jz = rho_m * (1.0 - st["phi"]) * st["umz"] + rho_g * st["phi"] * ug
    return float(2.0 * math.pi * np.trapezoid(jz * r, r))


def _h_max(phi, fragmentado, phicrit):
    # Por encima de φ_crit el paso de 40 m se salta el punto sónico y cae
    # en la raíz supersónica (P de unos 3 MPa y u de cientos de m/s), que
    # no se puede continuar. Con 16 m la raíz es la subsónica: la presión
    # baja poco y la velocidad sigue a la de aguas arriba.
    if fragmentado:
        return 16.0
    p = float(np.max(phi))
    if p < 0.05:
        return 12.0
    if p < 0.2:
        return 10.0
    return 8.0


def _h_piso(st, fragmentado):
    """Por debajo de este paso el marchar en z amplifica el modo radial de la malla.

    Sale de ρ u Δr² / (4 μ): con Δr ≈ 1,3 m y μ ≈ 5 kPa s el umbral es unos 8 m.
    """
    if fragmentado:
        return 4.0
    dr = float(st["dr"])
    uz = max(float(np.max(np.abs(st["umz"]))), 1.0)
    mu = max(float(np.min(st["mu"])), 1.0)
    rho = 2500.0
    return float(np.clip(1.15 * rho * uz * dr * dr / (4.0 * mu), 8.0, 16.0))


def _tablero(ur):
    """True si u_r alterna de signo: esa raíz no es el perfil radial suave."""
    if ur.size < 5:
        return False
    amp = float(np.max(np.abs(ur)))
    if amp < 1.0e-3:
        return False
    signo = np.sign(ur)
    signo[np.abs(ur) < 0.05 * amp] = 0.0
    nz = signo[signo != 0.0]
    if nz.size < 4:
        return False
    cambios = int(np.sum(nz[1:] * nz[:-1] < 0.0))
    return cambios >= max(3, nz.size // 2)


def _estado_de(ruta, pref):
    d = np.load(ruta)
    st = {
        k: np.array(d[f"{pref}{k}"], dtype=float)
        for k in ("z", "r", "dr", "P", "phi", "umz", "ugz", "umr", "ugr", "xi", "N", "mu")
    }
    st["z"] = float(st["z"])
    st["dr"] = float(st["dr"])
    st["fragmentado"] = False
    return st, float(np.asarray(d["z_sat"]).reshape(-1)[0]), float(np.asarray(d["z_f"]).reshape(-1)[0])


def _hist_eje(ruta):
    """Serie en el eje guardada a mitad de marcha, para no rehacer el tramo líquido."""
    d = np.load(ruta)
    hist = []
    for i in range(d["z"].size):
        hist.append({
            "z": float(d["z"][i]),
            "P": np.array([float(d["P"][i])]),
            "phi": np.array([float(d["phi"][i])]),
            "umz": np.array([float(d["um"][i])]),
            "ugz": np.array([float(d["ug"][i])]),
            "umr": np.array([float(d["ur"][i])]),
            "ugr": np.array([float(d["ugr"][i])]),
            "N": np.array([float(d["N"][i])]),
            "xi": np.array([float(d["xi"][i])]),
            "mu": np.array([float(d["mu"][i])]),
            "fragmentado": False,
        })
    return hist


def marchar_columna(vin=15.5, n_r=13, h_liq=40.0, umb=None, reanudar=None, parcial=None):
    if umb is None:
        umb = _umbrales(vin)
    if reanudar:
        st, z_sat, _z_f_archivo = _estado_de(reanudar, "s_")
        prev2, _, _ = _estado_de(reanudar, "p_")
        hist = _hist_eje(parcial) if parcial else []
        hist = [s for s in hist if s["z"] <= st["z"] + 1.0e-3]
        if not hist or abs(hist[-1]["z"] - st["z"]) > 1.0e-2:
            hist.append(st)
        else:
            hist[-1] = st
        h_prev = st["z"] - prev2["z"]
        h = 8.0
        marca = np.load(reanudar)
        fragmentado = bool(float(np.asarray(marca["fragmentado"]).reshape(-1)[0]))
        z_f_archivo = float(np.asarray(marca["z_f"]).reshape(-1)[0])
        z_f = None if z_f_archivo < -6000.0 else z_f_archivo
        st["fragmentado"] = fragmentado
        prev2["fragmentado"] = bool(
            fragmentado and z_f is not None and float(prev2["z"]) >= z_f - 1.0
        )
        q_ref = float(rho_de(P_BASE) * vin * math.pi * R_COND ** 2)
        n_liq = 0
    else:
        liquido = marchar(vin=vin, n_r=n_r, h=h_liq)
        hist = [_desde_liquido(s) for s in liquido]
        # N(r) queda fijado al cruzar la saturación, con la tasa local.
        hist[-1]["N"] = _nucleacion(hist[-1], hist[-2])
        z_sat = hist[-1]["z"]
        st = hist[-1]
        prev2 = hist[-2]
        h_prev = st["z"] - prev2["z"]
        h = 8.0
        fragmentado = False
        z_f = None
        q_ref = _caudal(hist[0])
        n_liq = len(liquido)
    recortes = 0
    while st["z"] < -0.2 and len(hist) < 4000:
        cruce = False
        if (not fragmentado) and float(np.max(st["phi"])) >= umb["phicrit"] - 0.035:
            fragmentado = True
            cruce = True
            z_f = st["z"]
            # 8 m no cierra el salto de n. 16 m da la raíz subsónica;
            # 40 m salta el punto sónico.
            h = 16.0
            print(f"z_f {z_f:.2f}  phi {float(np.max(st['phi'])):.4f}", flush=True)
            _guardar_estado(st, prev2, False, z_f, z_sat)
        piso = _h_piso(st, fragmentado)
        if cruce:
            h_uso = min(16.0, max(-st["z"], piso))
        else:
            h_uso = min(max(h, piso), _h_max(st["phi"], fragmentado, umb["phicrit"]), max(-st["z"], piso))
        nuevo, costo = _paso_bifasico(st, prev2, h_uso, h_prev, fragmentado, umb)
        dphi = float(np.max(nuevo["phi"]) - np.max(st["phi"]))
        cruza = (not fragmentado) and float(np.max(nuevo["phi"])) > umb["phicrit"] + 0.01
        ur_max = float(np.max(np.abs(nuevo["umr"])))
        margen = 15.0 if fragmentado else 2.0
        desv = float(np.max(np.abs(nuevo["umr"] - st["umr"])))
        # Si u_r queda pegado al borde de continuación, el paso no cerró la raíz suave.
        en_borde = desv > 0.92 * margen
        tablero = _tablero(nuevo["umr"]) or _tablero(nuevo["ugr"])
        # Con φ chico, |u_r| de varios cm/s es el modo de la malla, no el flujo.
        modo = (not fragmentado) and ur_max > 0.05 and float(np.max(nuevo["phi"])) < 0.05
        # Un salto de velocidad o una caída fuerte de P en un solo paso es la
        # raíz supersónica. No es la continuación del conducto.
        salto = fragmentado and (
            float(nuevo["umz"][0]) > float(st["umz"][0]) * 1.8
            or float(nuevo["P"][0]) < float(st["P"][0]) * 0.6
        )
        # Por encima de esto el momento radial se fue a la caja de búsqueda.
        # El primer paso fragmentado de la columna sana queda en unos 5 m/s.
        ur_falso = ur_max > 8.0
        feo = (
            (not np.isfinite(costo)) or costo > 1.0e-4 or cruza
            or en_borde or tablero or modo or salto or ur_falso
        )
        if feo and fragmentado and recortes < 4 and h_uso > 4.0 + 1.0e-9:
            recortes += 1
            print(
                f"reintento z={st['z']:.2f} h={h_uso:.3f} costo={costo:.2e} "
                f"salto={salto} |ur|={ur_max:.3e}",
                flush=True,
            )
            h = max(h_uso * 0.5, 4.0)
            continue
        if feo and (not fragmentado) and h_uso < _h_max(st["phi"], fragmentado, umb["phicrit"]) - 1.0e-9:
            print(
                f"reintento z={st['z']:.2f} h={h_uso:.3f} costo={costo:.2e} "
                f"|ur|={ur_max:.3e} d|ur|={desv:.3e} borde={en_borde} tablero={tablero}",
                flush=True,
            )
            h = min(_h_max(st["phi"], fragmentado, umb["phicrit"]), max(h_uso * 1.25, h_uso + 2.0))
            continue
        if feo:
            print(
                f"corte z={st['z']:.2f} h={h_uso:.3f} costo={costo:.2e} "
                f"phi={float(np.max(nuevo['phi'])):.4f} "
                f"um={nuevo['umz'][0]:.2f} ug={nuevo['ugz'][0]:.2f} "
                f"|ur|={ur_max:.3e} d|ur|={desv:.3e} borde={en_borde} tablero={tablero}",
                flush=True,
            )
            print("ur", np.array2string(nuevo["umr"], precision=3), flush=True)
            print("P MPa", np.array2string(nuevo["P"] / 1e6, precision=4), flush=True)
            print("phi", np.array2string(nuevo["phi"], precision=4), flush=True)
            print("umz", np.array2string(nuevo["umz"], precision=3), flush=True)
            print("mu", np.array2string(nuevo["mu"], precision=4), flush=True)
            st["corte_numerico"] = True
            break
        recortes = 0
        prev2 = st
        h_prev = h_uso
        st = nuevo
        hist.append(st)
        h_tope = _h_max(st["phi"], fragmentado, umb["phicrit"])
        h_piso_nuevo = _h_piso(st, fragmentado)
        if desv > 0.02 or costo > 1.0e-12 or ur_max > 0.02:
            h = min(max(h_uso, h_piso_nuevo), h_tope)
        elif dphi < 0.01:
            h = min(max(h_uso * 1.15, h_piso_nuevo), h_tope)
        else:
            h = min(max(h_uso, h_piso_nuevo), h_tope)
        n_bi = len(hist) - n_liq
        if n_bi <= 12 or n_bi % 10 == 0 or fragmentado:
            print(
                f"z={st['z']:.2f} h={h_uso:.2f} costo={costo:.2e} "
                f"P={st['P'][0]/1e6:.3f} phi={st['phi'][0]:.4f} "
                f"um={st['umz'][0]:.2f} ug={st['ugz'][0]:.2f} "
                f"slip={st['ugz'][0]-st['umz'][0]:.3e} "
                f"|ur|={np.max(np.abs(st['umr'])):.3e} |ugr|={np.max(np.abs(st['ugr'])):.3e} "
                f"dP={float(np.max(st['P'])-np.min(st['P']))/1e3:.2f}kPa "
                f"xi={st['xi'][0]:.6f} N={st['N'][0]:.3e} Q={_caudal(st)/q_ref:.4f}",
                flush=True,
            )
        Ma_m, _c_m = _mach_mezcla(
            st["P"][0], st["ugz"][0], fragmentado, st["xi"][0],
        )
        if Ma_m >= 0.98:
            print(
                f"choke Ma={Ma_m:.3f} c={_c_m:.1f} z={st['z']:.2f} "
                f"P={st['P'][0]/1e6:.3f}",
                flush=True,
            )
            break
        if st["P"][0] <= Patm * 1.05 or st["ugz"][0] >= 0.98 * CS:
            break
        if len(hist) % 20 == 0 or (fragmentado and len(hist) % 4 == 0):
            try:
                _guardar_parcial(hist, umb, z_sat, z_f)
                _guardar_estado(st, prev2, fragmentado, z_f, z_sat)
            except OSError as exc:
                print(f"no se pudo guardar el parcial ({exc})", flush=True)
    try:
        _guardar_parcial(hist, umb, z_sat, z_f)
    except OSError as exc:
        print(f"no se pudo guardar el parcial final ({exc})", flush=True)
    return hist, umb, z_f, z_sat


def _guardar_estado(st, prev2, fragmentado, z_f, z_sat):
    campos = ("z", "r", "dr", "P", "phi", "umz", "ugz", "umr", "ugr", "xi", "N", "mu")
    datos = {f"s_{k}": np.asarray(st[k]) for k in campos}
    datos.update({f"p_{k}": np.asarray(prev2[k]) for k in campos})
    datos["fragmentado"] = np.array([1.0 if fragmentado else 0.0])
    datos["z_f"] = np.array([-1.0 if z_f is None else z_f])
    datos["z_sat"] = np.array([z_sat])
    np.savez("/opt/cursor/artifacts/conducto_estado.npz", **datos)


def _guardar_parcial(hist, umb, z_sat, z_f):
    np.savez(
        "/opt/cursor/artifacts/conducto_fd_parcial.npz",
        z=np.array([s["z"] for s in hist]),
        P=np.array([s["P"][0] for s in hist]),
        phi=np.array([s["phi"][0] for s in hist]),
        um=np.array([s["umz"][0] for s in hist]),
        ug=np.array([s["ugz"][0] for s in hist]),
        ur=np.array([float(np.max(np.abs(s["umr"]))) for s in hist]),
        ugr=np.array([float(np.max(np.abs(s["ugr"]))) for s in hist]),
        N=np.array([s["N"][0] for s in hist]),
        xi=np.array([s["xi"][0] for s in hist]),
        mu=np.array([float(s["mu"][0]) for s in hist]),
        r=hist[-1]["r"],
        Nrad=hist[int(np.argmin([abs(s["z"] - z_sat) for s in hist]))]["N"],
        phicrit=umb["phicrit"],
        phi1=umb["phi1"],
        z_sat=z_sat,
        z_f=-1.0 if z_f is None else z_f,
    )
    graficar(hist, "/opt/cursor/artifacts/conducto_fd.png", umb, z_sat, z_f)


def graficar(hist, ruta, umb, z_sat, z_frag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.array([s["z"] for s in hist]) / 1000.0
    P = np.array([s["P"][0] for s in hist])
    phi = np.array([s["phi"][0] for s in hist])
    um = np.array([s["umz"][0] for s in hist])
    ug = np.array([s["ugz"][0] for s in hist])
    ur = np.array([float(np.max(np.abs(s["umr"]))) for s in hist])
    ugr = np.array([float(np.max(np.abs(s["ugr"]))) for s in hist])
    N = np.array([s["N"][0] for s in hist])
    xi = np.array([s["xi"][0] for s in hist])
    mu = np.array([float(s["mu"][0]) for s in hist])

    fig, ax = plt.subplots(2, 4, figsize=(13.8, 6.8), sharey=True)
    ax[0, 0].semilogx(np.maximum(P / 1e6, 1.0e-3), z, color="C0", lw=1.6)
    ax[0, 0].set_xlabel("P [MPa]")
    ax[0, 1].plot(phi, z, color="C0", lw=1.6)
    ax[0, 1].axvline(umb["phicrit"], color="k", lw=0.8, ls="--")
    ax[0, 1].axvline(umb["phi1"], color="0.45", lw=0.8, ls=":")
    ax[0, 1].set_xlabel(r"$\phi$")
    ax[0, 2].plot(um, z, color="C0", lw=1.8, label="fundido")
    ax[0, 2].plot(ug, z, color="C1", lw=1.2, ls="--", label="gas")
    ax[0, 2].set_xlabel(r"$u_z$ [m/s]")
    ax[0, 2].legend(frameon=False, fontsize=8)
    ax[0, 3].semilogx(np.maximum(ur, 1.0e-16), z, color="C0", lw=1.6, label="fundido")
    ax[0, 3].semilogx(np.maximum(ugr, 1.0e-16), z, color="C1", lw=1.2, ls="--", label="gas")
    ax[0, 3].set_xlabel(r"$|u_r|$ [m/s]")
    ax[0, 3].legend(frameon=False, fontsize=8)
    ax[1, 0].semilogx(np.maximum(N, 1.0), z, color="C0", lw=1.6)
    ax[1, 0].set_xlabel(r"$N$ [m$^{-3}$]")
    ax[1, 1].plot(xi, z, color="C0", lw=1.6)
    ax[1, 1].set_xlabel(r"$\xi$")
    ax[1, 2].semilogx(np.maximum(mu, 1.0), z, color="C0", lw=1.6)
    ax[1, 2].set_xlabel(r"$\mu$ [Pa s]")
    slip = np.abs(ug - um)
    ax[1, 3].semilogx(np.maximum(slip, 1.0e-16), z, color="C3", lw=1.5)
    ax[1, 3].set_xlabel(r"$|u_g-u_m|$ [m/s]")
    for a in ax.ravel():
        a.axhline(z_sat / 1000.0, color="0.5", lw=0.6, ls="--")
        if z_frag is not None:
            a.axhline(z_frag / 1000.0, color="C3", lw=0.7, ls=":")
        a.grid(True, alpha=0.3, which="both")
    ax[0, 0].set_ylabel("z [km]")
    ax[1, 0].set_ylabel("z [km]")
    fig.tight_layout()
    import time
    for intento in range(6):
        try:
            fig.savefig(ruta, dpi=140)
            break
        except OSError as exc:
            print(f"reintento figura {ruta} ({exc})", flush=True)
            time.sleep(0.4 * (intento + 1))
    else:
        print(f"no se pudo guardar {ruta}", flush=True)
    plt.close(fig)


def _mach_mezcla(P, u, fragmentado, xi):
    """Número de Mach de la mezcla en equilibrio, u / c(P)."""
    P = float(P)
    d = max(P * 1.0e-3, 50.0)

    def rho_eq(Pp):
        n = float(np.asarray(n_de(Pp, xi, fragmentado)).reshape(-1)[0])
        rm = float(rho_de(Pp))
        rg = max(Pp, 1.0e4) / (RV * T_GAS)
        if n <= 1.0e-8:
            return rm
        phi = n * rm / (rg * (1.0 - n) + n * rm)
        return rm * (1.0 - phi) + rg * phi

    p_lo = max(P - d, Patm)
    drdp = (rho_eq(P + d) - rho_eq(p_lo)) / (P + d - p_lo)
    if drdp <= 0.0:
        return 0.0, CS
    c = 1.0 / math.sqrt(drdp)
    return float(u) / c, c


def _clasificar_boca(sal):
    """'boca' si la salida ya cumple, 'sube' si vin es chica, 'baja' si es grande.

    Es el mismo criterio que el tiro 1D: o la boca está a presión atmosférica
    con el conducto fragmentado, o el flujo llega sónico a z = 0. Si la
    presión cae bajo la atmósfera antes, o el punto sónico queda bajo la
    boca, la velocidad de entrada es demasiado alta.
    """
    z, P, ug, phi = sal["z"], sal["P"], sal["ug"], sal["phi"]
    Ma = sal["Ma"]
    en_boca = z >= -8.0
    if en_boca and P >= 0.8 * Patm and (Ma >= 0.95 or abs(P - Patm) <= 1.5e5):
        return "boca"
    if P < Patm or (z < -15.0 and (Ma >= 0.98 or P <= 1.5 * Patm)):
        return "baja"
    # Un corte numérico ya pegado al Mach 1 es el punto sónico bajo la boca.
    if sal.get("fallo") and z < -15.0 and Ma >= 0.98:
        return "baja"
    if sal.get("fallo"):
        return "fallo"
    return "sube"


def evaluar_vin(vin, n_r=9, reanudar=None, parcial=None):
    """Una marcha completa. Devuelve la salida y la historia."""
    print(f"\n=== tiro  vin {vin:.4f}   n_r {n_r} ===", flush=True)
    umb = _umbrales(vin)
    hist, umb, z_f, z_sat = marchar_columna(
        vin=vin, n_r=n_r, h_liq=40.0, umb=umb, reanudar=reanudar, parcial=parcial,
    )
    ult = hist[-1]
    frag = bool(ult.get("fragmentado", z_f is not None))
    Ma, c = _mach_mezcla(ult["P"][0], ult["ugz"][0], frag or z_f is not None, ult["xi"][0])
    sal = {
        "vin": float(vin),
        "z": float(ult["z"]),
        "P": float(ult["P"][0]),
        "um": float(ult["umz"][0]),
        "ug": float(ult["ugz"][0]),
        "phi": float(ult["phi"][0]),
        "z_f": None if z_f is None else float(z_f),
        "z_sat": float(z_sat),
        "phicrit": float(umb["phicrit"]),
        "Ma": float(Ma),
        "c": float(c),
        "fallo": bool(ult.get("corte_numerico", False)),
    }
    sal["ajuste"] = _clasificar_boca(sal)
    print(
        f"resultado vin {vin:.3f}  {sal['ajuste']}  z {sal['z']:.1f}  "
        f"P {sal['P']/1e6:.3f} MPa  um {sal['um']:.2f}  ug {sal['ug']:.2f}  "
        f"Ma {sal['Ma']:.3f}  phi {sal['phi']:.3f}  z_f {sal['z_f']}",
        flush=True,
    )
    return sal, hist, umb, z_f, z_sat


def _guardar_tabla(tabla, ruta="/opt/cursor/artifacts/tiro_tabla.json"):
    import json
    import os
    os_dir = os.path.dirname(ruta)
    if os_dir:
        os.makedirs(os_dir, exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(tabla, f, indent=2)
        f.write("\n")


def tiro(n_r=13, tol=0.4, v_max=48.0, pasos=5, vin0=20.0, lo0=15.5, hi0=None):
    """Bisección de la velocidad en la base.

    15.5 m/s llega a la boca con unos 8.5 MPa y Mach de mezcla ~0.7, así que
    el intervalo parte de ahí y el primer disparo prueba 20 m/s. Se sube
    mientras la boca siga presurizada y se baja si el punto sónico queda
    bajo el cráter.
    """
    lo = float(lo0)
    hi = None if hi0 is None else float(hi0)
    vin = float(vin0)
    tabla = []
    mejor = None
    for k in range(pasos):
        sal, hist, umb, z_f, z_sat = evaluar_vin(vin, n_r=n_r)
        tabla.append(sal)
        _guardar_tabla(tabla)
        aj = sal["ajuste"]
        if aj == "boca" or (
            aj == "sube" and (mejor is None or sal["vin"] >= mejor[0]["vin"])
        ):
            mejor = (sal, hist, umb, z_f, z_sat)
            graficar(hist, "/opt/cursor/artifacts/conducto_tiro.png", umb, z_sat, z_f)
        if aj == "boca":
            break
        if aj == "fallo":
            vin = max(8.0, 0.8 * vin)
            print(f"reintento por corte numérico, vin {vin:.3f}", flush=True)
            continue
        if aj == "sube":
            lo = max(lo, vin)
            vin = 0.5 * (vin + hi) if hi is not None else min(vin * 1.4, v_max)
        else:
            hi = vin if hi is None else min(hi, vin)
            vin = 0.5 * (lo + hi)
        print(f"intervalo  {lo:.3f} .. {hi if hi is not None else v_max:.3f}  siguiente {vin:.3f}", flush=True)
        if hi is not None and (hi - lo) < tol:
            break
        if hi is None and vin >= v_max - 1.0e-6:
            break
    if mejor is None:
        print("ningún disparo llegó a la boca sin ahogarse", flush=True)
        sal = tabla[-1]
        hist = None
    else:
        sal, hist, umb, z_f, z_sat = mejor
    print("\nTabla del tiro", flush=True)
    print(f"{'vin':>8} {'ajuste':>6} {'z':>8} {'P_MPa':>8} {'ug':>8} {'Ma':>6} {'z_f':>8}", flush=True)
    for s in tabla:
        zf = s["z_f"] if s["z_f"] is not None else float("nan")
        print(
            f"{s['vin']:8.3f} {s['ajuste']:>6} {s['z']:8.1f} {s['P']/1e6:8.3f} "
            f"{s['ug']:8.2f} {s['Ma']:6.3f} {zf:8.1f}",
            flush=True,
        )
    if hist is not None:
        graficar(hist, "/opt/cursor/artifacts/conducto_tiro.png", umb, z_sat, z_f)
    print(
        f"vin elegido {sal['vin']:.3f}  P_boca {sal['P']/1e6:.3f} MPa  "
        f"um {sal['um']:.2f}  ug {sal['ug']:.2f}  Ma {sal['Ma']:.3f}  z_f {sal['z_f']}",
        flush=True,
    )
    return sal, tabla, hist


if __name__ == "__main__":
    import os
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "tiro":
        n_r = int(sys.argv[2]) if len(sys.argv) > 2 else 13
        vin0 = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0
        lo0 = float(sys.argv[4]) if len(sys.argv) > 4 else 15.5
        hi0 = float(sys.argv[5]) if len(sys.argv) > 5 else None
        tiro(
            n_r=n_r, vin0=vin0, lo0=lo0, hi0=hi0,
            pasos=2 if hi0 is not None else 5,
        )
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "seguir":
        vin = float(sys.argv[2])
        n_r = int(sys.argv[3])
        evaluar_vin(vin, n_r=n_r, reanudar=sys.argv[4], parcial=sys.argv[5])
        sys.exit(0)
    vin = 15.5
    umb = _umbrales(vin)
    print(
        f"vin {vin:.3f}  P* {umb['Pstar']/1e6:.2f} MPa  Ca {umb['Ca']:.4f}  "
        f"phi1 {umb['phi1']:.4f}  phi2 {umb['phi2']:.4f}  phi_crit {umb['phicrit']:.4f}",
        flush=True,
    )
    reanudar = os.environ.get("CONDUIT_REANUDAR")
    parcial = os.environ.get(
        "CONDUIT_PARCIAL", "/opt/cursor/artifacts/conducto_fd_parcial.npz",
    )
    hist, umb, z_f, z_sat = marchar_columna(
        vin=vin, n_r=13, h_liq=40.0, umb=umb,
        reanudar=reanudar, parcial=parcial if reanudar else None,
    )
    graficar(hist, "/opt/cursor/artifacts/conducto_fd.png", umb, z_sat, z_f)
    ult = hist[-1]
    q_ref = float(rho_de(P_BASE) * vin * math.pi * R_COND ** 2)
    print(
        f"z_sat {z_sat:.1f}  z_f {z_f}  z_final {ult['z']:.2f}  "
        f"P {ult['P'][0]/1e6:.3f}  phi {ult['phi'][0]:.4f}  "
        f"um {ult['umz'][0]:.2f}  ug {ult['ugz'][0]:.2f}",
        flush=True,
    )
    print(
        f"N eje {hist[0]['N'][0]:.3e} -> final {ult['N'][0]:.3e}  "
        f"xi {ult['xi'][0]:.6f}  Q { _caudal(ult)/q_ref:.5f}",
        flush=True,
    )
