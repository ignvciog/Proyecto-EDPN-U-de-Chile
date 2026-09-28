"""
Reducción 2D con las 5M+1 ecuaciones.

En cada altura, M radios libres (la pared no es incógnita):

* masa del fundido en cada radio
* masa del gas una sola vez, Q = const
* momento del fundido en cada radio
* momento del gas en cada radio
* N en cada radio
* ξ en cada radio

P = P(z). u_r = 0. ρ_m = ρ_m(P) (Bottinga y Weill, agua disuelta de Henry).

El momento del gas es de segundo orden en r y pide un borde en la pared.
Aquí se toma u_g(R) = 0, el mismo no-deslizamiento que u_m(R) = 0.
En el eje, ∂u/∂r = 0 para las dos fases.

La masa puntual del gas, ∂z(ρ_g φ u_g) = Γ, no se impone. Q es la
integral del flujo total (fundido + gas).
"""

from __future__ import annotations

import math

import numpy as np

from calbuco2015d import (
    C1,
    F1,
    F2,
    Fc,
    H,
    Patm,
    R,
    T1,
    beta,
    g,
    h2o1,
    overP1,
    pfinal,
    radius1,
    rcrust,
    tcar,
    xi1,
    xmax,
)
from RIconduit2D_FD import (
    _Fmg,
    _effective_visc,
    _fg,
    _rho_g,
    _rho_m,
    _thomas,
)
import RIconduit2D_FD as _fd

MU_G = 1.8e-5


def _configurar(radius, pressure, wt, temperature, content_crystal, n_r):
    gbl = _fd._G
    n_r = int(n_r)
    gbl["N_r"] = n_r
    gbl["wr"] = float(radius)
    gbl["r"] = np.linspace(1e-6, float(radius), n_r)
    gbl["dr"] = float(gbl["r"][1] - gbl["r"][0])
    gbl["T"] = float(temperature)
    gbl["Tc"] = float(temperature) - 273.15
    gbl["h2o"] = float(wt)
    gbl["co"] = float(wt) / 100.0
    gbl["xi"] = float(content_crystal)
    gbl["xmax_run"] = float(xmax)
    gbl["phicrit"] = 0.75
    gbl["limphi1"] = 0.15
    gbl["limphi2"] = 0.40
    gbl["Nd0"] = 1e8
    pi = rcrust * g * abs(H) + float(pressure)
    gbl["Pi"] = pi
    gbl["rho_m"] = _rho_m(pi)
    return gbl


def _matriz_radial(factor, mu):
    """(1/r) d/dr [ r a du/dr ] con a = factor*mu, u(R)=0, du/dr(0)=0."""
    r = _fd._G["r"]
    dr = _fd._G["dr"]
    n = len(r)
    nunk = n - 1
    lo = np.zeros(nunk)
    di = np.zeros(nunk)
    up = np.zeros(nunk)
    a = np.maximum(factor, 0.0) * np.maximum(mu, 1e-12)
    a_h = 0.5 * (a[0] + a[1])
    di[0] = -2.0 * a_h / dr ** 2
    up[0] = 2.0 * a_h / dr ** 2
    for i in range(1, nunk):
        a_p = 0.5 * (a[i] + a[i + 1])
        a_m = 0.5 * (a[i] + a[i - 1])
        ri = r[i]
        lo[i] = (ri - 0.5 * dr) * a_m / (ri * dr ** 2)
        di[i] = -((ri + 0.5 * dr) * a_p + (ri - 0.5 * dr) * a_m) / (ri * dr ** 2)
        if i < nunk - 1:
            up[i] = (ri + 0.5 * dr) * a_p / (ri * dr ** 2)
    return lo, di, up


def _k_arrastre(um, ug, phi, nd, rg, mu):
    """K tal que F_mg ≈ K (u_g - u_m), K ≥ 0."""
    um = np.asarray(um, dtype=float)
    ug = np.asarray(ug, dtype=float)
    slip = ug - um
    k = np.zeros(len(um))
    for i in range(len(um)):
        if float(phi[i]) < 1e-4:
            continue
        s = float(slip[i])
        if abs(s) < 1e-6:
            f1 = _Fmg(float(um[i]), float(um[i]) + 1.0, float(phi[i]), float(nd[i]), float(rg), float(mu[i]))
            k[i] = f1 / 1.0
        else:
            f = _Fmg(float(um[i]), float(ug[i]), float(phi[i]), float(nd[i]), float(rg), float(mu[i]))
            k[i] = f / s
    return np.clip(k, 0.0, 1e12)


def _resolver_fase(factor, mu, rho, dpdz, k_drag, u_otro, u_prev, sigma_prev, u_star, h):
    """Thomas del momento. El arrastre lineal K entra en la diagonal.

    En las dos fases el otro lado del arrastre queda como -K u_otro.
    """
    n = len(factor)
    nunk = n - 1
    lo, di, up = _matriz_radial(factor, mu)
    beta = rho * factor * np.maximum(u_star, 0.0) / h
    gamma = mu * factor / h ** 2
    coef = beta - gamma
    di = di - coef[:nunk] - k_drag[:nunk]
    # L u - coef u - K u = factor*dpdz + ρ factor g - F_explicito + σ_prev/h - coef u_prev
    # Fundido: F_mg = K(ug - um) y L u = ... - F_mg, con um implícito ya en -K u,
    #          queda -K*ug en el RHS.
    # Gas:     L u = ... + F_mg = ... + K(ug - um), ug implícito en -K u,
    #          queda -K*um en el RHS.
    f_lag = k_drag * u_otro
    rhs = (
        factor * dpdz
        + rho * factor * g
        - f_lag
        + sigma_prev / h
        - coef * u_prev
    )
    # Nodos sin fase: la ecuación se reemplaza por u = u_otro (no hay gas, o no hay fundido).
    piso = 1e-4
    for i in range(nunk):
        if factor[i] < piso and k_drag[i] < 1.0:
            lo[i] = 0.0
            up[i] = 0.0
            di[i] = 1.0
            rhs[i] = u_otro[i]
    u = np.zeros(n)
    try:
        u[:nunk] = _thomas(lo, di, up, rhs[:nunk])
    except Exception:
        u[:] = u_prev
    u[-1] = 0.0
    u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(u, 0.0)


def _phi_masa(p, um, ug, fm_prev, n_prev, xi):
    """φ desde la masa del fundido en cada radio. Q no entra aquí."""
    rm = _rho_m(p)
    rg = _rho_g(p)
    n = np.array([_fg(p, float(xi[i])) for i in range(len(xi))])
    dn = n - n_prev
    phi = np.zeros(len(um))
    for i in range(len(um) - 1):
        um_i = max(float(um[i]), 1e-6)
        num = float(fm_prev[i]) - rm * um_i * (1.0 + dn[i])
        den = rg * float(ug[i]) * dn[i] - rm * um_i * (1.0 + dn[i])
        if abs(den) < 1e-8 * rm * um_i:
            phi_i = 1.0 - float(fm_prev[i]) / (rm * um_i)
        else:
            phi_i = num / den
        phi[i] = float(np.clip(phi_i, 0.0, 0.95))
    phi[-1] = phi[-2]
    return phi, n


def _caudal(p, phi, um, ug):
    r = _fd._G["r"]
    rm = _rho_m(p)
    rg = _rho_g(p)
    jz = rm * (1.0 - phi) * um + rg * phi * ug
    return float(np.trapezoid(jz * 2.0 * math.pi * r, r))


def _gamma_n(p, phi, nd, um, xi):
    """Coalescencia (Slezin). dN/dt, negativa."""
    out = np.zeros(len(phi))
    rm = _rho_m(p)
    rg = _rho_g(p)
    wr = _fd._G["wr"]
    phicrit = _fd._G["phicrit"]
    for i in range(len(phi) - 1):
        phi_c = float(np.clip(phi[i], 1e-8, phicrit - 1e-4))
        nd_s = max(float(nd[i]), 1.0)
        rb = (phi_c / ((4.0 / 3.0) * math.pi * nd_s * (1.0 - phi_c))) ** (1.0 / 3.0)
        if rb >= 0.5 * wr or phi[i] >= 0.5:
            continue
        visc = _effective_visc(p, phi_c, float(xi[i]), max(float(um[i]), 1e-6), nd_s)
        den = 1.0 - (F1 + F2) * ((3.0 * phi_c * (math.pi / (6.0 * phicrit)) / (4.0 * math.pi)) ** (1.0 / 3.0))
        if abs(den) < 1e-8:
            continue
        out[i] = (
            -(nd_s ** (2.0 / 3.0))
            * ((1.0 / (1.0 - phi_c)) ** (1.0 / 3.0))
            * ((1.0 / 9.0) * (rm - rg) * g / max(visc, 1e-30))
            * ((3.0 * phi_c / (4.0 * math.pi)) ** (2.0 / 3.0))
            * (F1 ** 2 - F2 ** 2)
            / den
            * Fc
            * (1.0 - phi_c / phicrit)
            * (wr - rb)
            / wr
        )
    out[-1] = out[-2]
    return out


def _gamma_xi(p, xi, um):
    """dξ/dz = Γ_ξ / u_m. Γ_ξ es la tasa en el tiempo."""
    xi0 = _fd._G["xi"]
    xmax_run = _fd._G["xmax_run"]
    co = _fd._G["co"]
    out = np.zeros(len(xi))
    den0 = max(co * (1.0 - xi0) - (1.0 - xmax_run) * C1 * Patm ** beta, 1e-30)
    for i in range(len(xi) - 1):
        f2 = max(0.0, (co * (1.0 - xi0) - (1.0 - xi[i]) * C1 * max(p, 1e4) ** beta) / den0)
        xteo = xi0 + (xmax_run - xi0) * f2
        f3 = max(0.0, 1.0 - xi[i] / max(xteo, 1e-9))
        dxi_dt = max(0.0, (xmax_run - xi0) * f2 * f3 / tcar)
        out[i] = dxi_dt / max(float(um[i]), 1e-3)
    out[-1] = out[-2]
    return out


def _mu_nodos(p, phi, xi, um, nd):
    return np.array([
        _effective_visc(p, float(phi[i]), float(xi[i]), max(float(um[i]), 1e-6), max(float(nd[i]), 1.0))
        for i in range(len(phi))
    ])


def _paso(estado, h, q_obj, n_it=8):
    """Un paso hacia arriba. estado es un dict de campos en r, más P y z."""
    p_prev = float(estado["P"])
    phi = estado["phi"].copy()
    um = estado["um"].copy()
    ug = estado["ug"].copy()
    nd = estado["N"].copy()
    xi = estado["xi"].copy()
    n_prev = estado["n"].copy()
    fm_prev = _rho_m(p_prev) * (1.0 - estado["phi"]) * estado["um"]
    sigma_m = estado["sigma_m"].copy()
    sigma_g = estado["sigma_g"].copy()
    dpdz = float(estado["dpdz"])
    um_prev = estado["um"].copy()
    ug_prev = estado["ug"].copy()

    p = p_prev
    for _ in range(max(n_it, 14)):
        p_try = max(p_prev + dpdz * h, 1e4)
        rm = _rho_m(p_try)
        rg = _rho_g(p_try)
        mu = _mu_nodos(p_try, phi, xi, um, nd)
        k = _k_arrastre(um, ug, phi, nd, rg, mu)
        um_try = _resolver_fase(1.0 - phi, mu, rm, dpdz, k, ug, um_prev, sigma_m, np.maximum(um, 1e-3), h)
        ug_try = _resolver_fase(np.maximum(phi, 0.0), np.full(len(phi), MU_G), rg, dpdz, k, um_try, ug_prev, sigma_g, np.maximum(ug, 1e-3), h)
        phi_new, _n_tmp = _phi_masa(p_try, um_try, ug_try, fm_prev, n_prev, xi)
        phi_try = 0.65 * phi + 0.35 * phi_new
        phi_try[-1] = phi_try[-2]
        q_now = _caudal(p_try, phi_try, um_try, ug_try)
        p, um, ug, phi = p_try, um_try, ug_try, phi_try
        if q_now > 0.0 and abs(q_now - q_obj) / q_obj < 1e-4:
            break
        mezcla = rg * phi + rm * (1.0 - phi)
        rho_bar = float(np.average(mezcla))
        drive = max(-(dpdz + rho_bar * g), 1.0)
        drive *= float(np.clip(q_obj / max(q_now, 1e-6), 0.85, 1.15))
        dpdz_new = -(drive + rho_bar * g)
        if np.isfinite(dpdz_new):
            dpdz = dpdz_new

    q_now = _caudal(p, phi, um, ug)
    if q_now > 0.0:
        escala = float(np.clip(q_obj / q_now, 0.5, 1.5))
        um = um * escala
        ug = ug * escala
        um[-1] = 0.0
        ug[-1] = 0.0

    p = max(p, 1e4)
    gN = _gamma_n(p, phi, nd, um, xi)
    nd = np.maximum(nd + h * gN / np.maximum(um, 1e-3), 1.0)
    nd[-1] = nd[-2]
    xi = xi + h * _gamma_xi(p, xi, um)
    xi = np.clip(xi, _fd._G["xi"], _fd._G["xmax_run"])
    xi[-1] = xi[-2]
    mu = _mu_nodos(p, phi, xi, um, nd)
    sigma_m = mu * (1.0 - phi) * (um - um_prev) / h
    sigma_g = MU_G * phi * (ug - ug_prev) / h
    sigma_m[-1] = 0.0
    sigma_g[-1] = 0.0
    n = np.array([_fg(p, float(xi[i])) for i in range(len(xi))])
    return {
        "z": estado["z"] + h,
        "P": p,
        "phi": phi,
        "um": um,
        "ug": ug,
        "N": nd,
        "xi": xi,
        "n": n,
        "sigma_m": sigma_m,
        "sigma_g": sigma_g,
        "dpdz": dpdz,
        "Q": _caudal(p, phi, um, ug),
    }


def _estado_inicial(vinicial):
    """Base: P litostática, u_m = u_g = perfil con media vinicial, σ = 0."""
    gbl = _fd._G
    n = gbl["N_r"]
    r = gbl["r"]
    wr = gbl["wr"]
    pi = gbl["Pi"]
    xi = np.full(n, gbl["xi"])
    nd = np.full(n, gbl["Nd0"])
    n_gas = np.array([_fg(pi, gbl["xi"]) for _ in range(n)])
    rm = _rho_m(pi)
    if n_gas[0] <= 0.0:
        phi = np.zeros(n)
        rho_t = rm
    else:
        phi = np.full(n, 1.0 / (1.0 + (pi / (n_gas[0] * R * gbl["T"])) * (1.0 - n_gas[0]) / rm))
        phi = np.clip(phi, 0.0, 0.95)
        rho_t = rm * (1.0 - phi[0]) + _rho_g(pi) * phi[0]
    mu0 = _effective_visc(pi, float(phi[0]), gbl["xi"], max(vinicial, 1e-3), gbl["Nd0"])
    dpdz = -rho_t * g - 8.0 * mu0 * vinicial / wr ** 2
    # Perfil parabólico con u(R)=0 y media ≈ vinicial (Poiseuille: media = Umax/2).
    umax = 2.0 * float(vinicial)
    um = umax * (1.0 - (r / wr) ** 2)
    um[-1] = 0.0
    ug = um.copy()
    q0 = _caudal(pi, phi, um, ug)
    return {
        "z": float(H),
        "P": float(pi),
        "phi": phi,
        "um": um,
        "ug": ug,
        "N": nd,
        "xi": xi,
        "n": n_gas,
        "sigma_m": np.zeros(n),
        "sigma_g": np.zeros(n),
        "dpdz": float(dpdz),
        "Q": q0,
    }


def marchar(vinicial, radius=radius1, pressure=overP1, wt=h2o1, temperature=T1,
            content_crystal=xi1, n_r=10, dz=200.0, z_tope=0.0, phi_tope=0.7):
    """
    Marcha desde z=H hacia la boca para una velocidad de entrada.

    Se detiene al llegar a z_tope, si P baja de P_atm, o si φ llega a phi_tope
    (el régimen fragmentado es otro sistema).
    """
    _configurar(radius, pressure, wt, temperature, content_crystal, n_r)
    est = _estado_inicial(float(vinicial))
    q_obj = est["Q"]
    historia = [est]
    h = float(dz)
    h_min = 10.0
    mensaje = "boca"
    rechazos = 0
    for _ in range(800):
        if est["z"] >= z_tope - 1.0:
            mensaje = "boca"
            break
        if est["P"] <= pfinal * 1.05 and est["z"] > H + 50.0:
            mensaje = "presion atmosferica"
            break
        if float(np.max(est["phi"])) >= phi_tope:
            mensaje = "fraccion de gas alta"
            break
        h_uso = min(h, z_tope - est["z"])
        nxt = _paso(est, h_uso, q_obj)
        dphi = float(np.max(nxt["phi"] - est["phi"]))
        qrel = abs(nxt["Q"] - q_obj) / max(q_obj, 1.0)
        malo = (not np.isfinite(nxt["P"])) or nxt["P"] > est["P"] * 1.05 or dphi > 0.04 or qrel > 0.03
        if malo and h > h_min:
            h = max(h * 0.5, h_min)
            rechazos += 1
            if rechazos > 12:
                mensaje = "paso rechazado"
                break
            continue
        if malo:
            mensaje = "paso rechazado"
            break
        rechazos = 0
        est = nxt
        historia.append(est)
        if dphi < 0.01 and qrel < 0.01:
            h = min(h * 1.25, float(dz))
    return _empaquetar(historia, q_obj, float(vinicial), mensaje)


def _empaquetar(historia, q_obj, vinicial, mensaje):
    def col(k):
        return np.vstack([h[k] for h in historia]) if np.ndim(historia[0][k]) else np.array([h[k] for h in historia])

    return {
        "z": col("z"),
        "P": col("P"),
        "phi": col("phi"),
        "um": col("um"),
        "ug": col("ug"),
        "N": col("N"),
        "xi": col("xi"),
        "Q": col("Q"),
        "Q_objetivo": float(q_obj),
        "vinicial": float(vinicial),
        "r": np.array(_fd._G["r"], dtype=float),
        "mensaje": mensaje,
        "rho_m_base": float(_fd._G["rho_m"]),
        "n_pasos": len(historia) - 1,
    }


def tirar(radius=radius1, pressure=overP1, wt=h2o1, temperature=T1, content_crystal=xi1,
          n_r=10, dz=250.0, n_tiro=6, v_min=0.3, v_max=8.0):
    """
    Tiro corto sobre v_in. Busca la marcha cuyo P final queda más cerca de P_atm
    sin haberlo cruzado mucho antes de la boca.
    """
    mejor = None
    mejor_puntaje = 1e99
    lo, hi = float(v_min), float(v_max)
    v = 0.5 * (lo + hi)
    for _ in range(int(n_tiro)):
        sal = marchar(v, radius, pressure, wt, temperature, content_crystal, n_r, dz)
        # P más alto que la atmósfera al final: falta descarga, subir v no siempre.
        # Puntaje: distancia de P final a Patm, penalizando cortar por φ alto.
        puntaje = abs(sal["P"][-1] - pfinal) / pfinal
        if sal["mensaje"] == "fraccion de gas alta":
            puntaje += 2.0
        if puntaje < mejor_puntaje:
            mejor_puntaje = puntaje
            mejor = sal
        if sal["P"][-1] > 5.0 * pfinal:
            lo = v
        else:
            hi = v
        v = 0.5 * (lo + hi)
    return mejor
