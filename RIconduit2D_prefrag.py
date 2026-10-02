"""
Exsolución hasta antes de la fragmentación, en diferencias finitas.

Se parte del tramo sin burbujas. Antes de fragmentar se resuelven
la masa de la mezcla, los dos momentos y el transporte de N y de ξ.
φ = φ_Henry(P, ξ). u_r queda libre dentro del conducto. Se corta
cuando φ llega a φ_crit. Después, n va sin ξ y se integran las dos
masas, los dos momentos verticales y los dos momentos radiales, con
el laplaciano viscoso y con u_r de cada fase.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf

from calbuco2015d import (
    C1,
    Fc,
    Patm,
    R,
    tcar,
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
    tio2,
    xmax,
)
from fvrel import fvrel
from viscosity import viscosity

from RIconduit2D_sinbub import (
    CO,
    K_BULK,
    P_BASE,
    P_SAT,
    RHO_H,
    R_COND,
    XI0,
    _d_dr,
    _desempacar,
    _div,
    _empacar,
    _lap_ur,
    _lap_uz,
    _ur_nodos,
    marchar,
    rho_de,
)

N0 = 1.0e8
RV = float(R)
T_GAS = float(T1)


def n_henry(P, xi=None):
    """n(P, ξ) de Henry. Con ξ = ξ_0 recupera (1-ξ_0)(c_0 - C_1 P^β)/(1 - C_1 P^β)."""
    P = np.maximum(np.asarray(P, dtype=float), 1.0e4)
    if xi is None:
        xi = XI0
    xi = np.asarray(xi, dtype=float)
    num = (1.0 - XI0) * CO - (1.0 - xi) * C1 * P ** beta
    den = np.maximum(1.0 - C1 * P ** beta, 1.0e-12)
    return np.maximum(num / den, 0.0)


def phi_henry(P, xi=None):
    P = np.asarray(P, dtype=float)
    n = n_henry(P, xi)
    rho_m = rho_de(P)
    rho_g = np.maximum(P, 1.0e4) / (RV * T_GAS)
    phi = np.zeros_like(n, dtype=float)
    vivo = n > 1.0e-10
    phi[vivo] = 1.0 / (
        1.0 + (rho_g[vivo] / rho_m[vivo]) * (1.0 - n[vivo]) / n[vivo]
    )
    return np.clip(phi, 0.0, 0.95)


def rho_mix(P, xi=None):
    phi = phi_henry(P, xi)
    rho_g = np.maximum(np.asarray(P, dtype=float), 1.0e4) / (RV * T_GAS)
    return (1.0 - phi) * rho_de(P) + phi * rho_g


def _umbrales(vin=5.0):
    """φ1, φ2, φ_crit en el estado φ*=0.2, con la tasa de la parábola axial."""
    lo, hi = 1.0e5, P_SAT
    Pstar = 0.5 * (lo + hi)
    for _ in range(60):
        if abs(float(phi_henry(Pstar)) - 0.2) < 0.002:
            break
        if float(phi_henry(Pstar)) < 0.2:
            hi = Pstar
        else:
            lo = Pstar
        Pstar = 0.5 * (lo + hi)
    nstar = float(n_henry(Pstar))
    rho_m = float(rho_de(Pstar))
    rho_g = Pstar / (RV * T_GAS)
    dndp = (1.0 - XI0) * (CO - 1.0) * (C1 * beta * Pstar ** (beta - 1.0)) / max(
        (1.0 - C1 * Pstar ** beta) ** 2, 1.0e-30
    )
    # dφ/dP por diferenciación de φ = [1 + (ρ_g/ρ_m)(1-n)/n ]^{-1}
    a = (rho_g / rho_m) * (1.0 - nstar) / max(nstar, 1.0e-16)
    da_dp = (
        (1.0 / (rho_m * RV * T_GAS)) * (1.0 - nstar) / max(nstar, 1.0e-16)
        - (rho_g / rho_m) * dndp / max(nstar ** 2, 1.0e-30)
    )
    dphidp = -da_dp / max((1.0 + a) ** 2, 1.0e-30)
    phi = float(phi_henry(Pstar))
    mu = _mu_nodo(Pstar, phi, 1.0)
    rho_c = (1.0 - phi) * rho_m + phi * rho_g
    dpdz = -rho_c * g - 8.0 * mu * vin / R_COND ** 2
    dvdz = vin * (
        -dndp * (1.0 - phi) + (1.0 - nstar) * dphidp
    ) / max((1.0 - phi) ** 2, 1.0e-30) * dpdz
    gdot = 0.5 * (abs(dvdz) + vin / R_COND)
    rb = (
        phi / ((4.0 / 3.0) * math.pi * N0 * max(1.0 - phi, 1.0e-6))
    ) ** (1.0 / 3.0)
    Ca = abs(gdot * mu * rb / 0.3)
    phicrit = ((0.785 - 0.525) / 2.0) * erf(math.log10(max(Ca, 1.0e-30))) + (0.785 + 0.525) / 2.0
    phi1 = ((0.15 - 0.40) / 2.0) * erf(math.log10(max(Ca, 1.0e-30))) + (0.15 + 0.40) / 2.0
    return {
        "Pstar": Pstar,
        "Ca": Ca,
        "phicrit": float(phicrit),
        "phi1": float(phi1),
        "phi2": float(phi1 + 0.01),
    }


def _mu_nodo(P, phi, gdot):
    if float(n_henry(P)) <= 0.0:
        h2o = float(h2o1)
    else:
        h2o = float(C1 * max(P, 1.0e4) ** beta * 100.0)
    vis = float(viscosity(sio2, tio2, al2o3, feo, mno, mgo, cao, na2o, k2o, p2o5, h2o, f2o, Tc1))
    vis *= float(fvrel(model, XI0, XI0, ar1, ar2, xmax, max(gdot, 1.0e-3)))
    if phi < 1.0e-8:
        return max(vis, 1.0)
    phicritbub = 0.785 + 0.05
    phi_c = min(float(phi), 0.9 * phicritbub)
    rb = (
        max(phi_c, 1.0e-16)
        / ((4.0 / 3.0) * math.pi * N0 * max(1.0 - phi_c, 1.0e-6))
    ) ** (1.0 / 3.0)
    nca = max(rb * vis * max(gdot, 1.0e-6) / 3.0, 1.0e-30)
    AA = (1.0 - phi_c / phicritbub) ** (-phicritbub)
    BB = (1.0 - phi_c / phicritbub) ** (5.0 * phicritbub / 3.0)
    c1 = -0.2895 * phi_c + 0.8132
    factor = 0.5 * (AA - BB) * (1.0 - math.erf(c1 * math.log(nca) + phi_c)) + BB
    return max(float(factor) * vis, 1.0)


def _gdot(uz, dr):
    n = uz.size
    g = np.zeros(n)
    g[0] = abs(uz[1] - uz[0]) / dr
    g[-1] = abs(uz[-1] - uz[-2]) / dr
    if n > 2:
        g[1:-1] = np.abs(uz[2:] - uz[:-2]) / (2.0 * dr)
    return np.maximum(g, 1.0e-3)


def mu_mix(P, uz, dr):
    phi = phi_henry(P)
    g = _gdot(uz, dr)
    return np.array([_mu_nodo(float(P[i]), float(phi[i]), float(g[i])) for i in range(P.size)])


def residual(y, prev, h, mu):
    r, dr = prev["r"], prev["dr"]
    n = r.size
    P, uz, ur = _desempacar(y, n)
    xi = prev["xi"]
    rho = rho_mix(P, xi)
    rho0 = rho_mix(prev["P"], xi)
    rho_f = 0.5 * (rho[:-1] + rho[1:])
    masa = (rho * uz - rho0 * prev["uz"]) / h + _div(r, dr, rho_f * ur)
    ddz = (uz - prev["uz"]) / h
    urn = _ur_nodos(ur)
    lap = _lap_uz(r, dr, mu, uz)
    mom_z = rho * (uz * ddz + urn * _d_dr(r, uz)) + (P - prev["P"]) / h + rho * g - lap
    uz_f = 0.5 * (uz[:-1] + uz[1:])
    mu_f = 0.5 * (mu[:-1] + mu[1:])
    x = np.concatenate([[0.0], 0.5 * (r[:-1] + r[1:]), [r[-1]]])
    dur = np.gradient(np.concatenate([[0.0], ur, [0.0]]), x)[1:-1]
    mom_r = (
        rho_f * (uz_f * (ur - prev["ur"]) / h + ur * dur)
        + (P[1:] - P[:-1]) / dr
        - _lap_ur(r, mu_f, ur)
    )
    esc_m = max(float(np.max(rho0 * np.maximum(prev["uz"], 1.0))) / h, 1.0)
    esc_z = max(float(np.max(rho0)) * g, 1.0)
    return np.concatenate([masa / esc_m, mom_z[:-1] / esc_z, mom_r / esc_z])


def _gamma_xi(P, xi):
    num = (1.0 - XI0) * CO - (1.0 - xi) * C1 * np.maximum(P, 1.0e4) ** beta
    den = (1.0 - XI0) * CO - (1.0 - xmax) * C1 * Patm ** beta
    f2 = np.maximum(num / max(den, 1.0e-30), 0.0)
    xi_eq = XI0 + (xmax - XI0) * f2
    f3 = np.maximum(1.0 - xi / np.maximum(xi_eq, 1.0e-8), 0.0)
    return np.maximum((xmax - XI0) * f2 * f3 / tcar, 0.0)


def _gamma_N(P, phi, N, mu, phicrit):
    phi = np.asarray(phi, dtype=float)
    N = np.maximum(np.asarray(N, dtype=float), 1.0)
    mu = np.maximum(np.asarray(mu, dtype=float), 1.0)
    rho_g = np.maximum(P, 1.0e4) / (RV * T_GAS)
    rho_m = rho_de(P)
    rb = (
        np.maximum(phi, 1.0e-16)
        / ((4.0 / 3.0) * math.pi * N * np.maximum(1.0 - phi, 1.0e-6))
    ) ** (1.0 / 3.0)
    vivo = (phi > 1.0e-8) & (phi < phicrit) & (rb < 0.5 * R_COND)
    F1, F2 = 1.8, 0.2
    inner = (F1 + F2) * (
        (3.0 * np.maximum(phi, 0.0) * math.pi / (6.0 * phicrit)) / (4.0 * math.pi)
    ) ** (1.0 / 3.0)
    den = 1.0 - inner
    tasa = (
        -(N ** (2.0 / 3.0))
        * (1.0 / np.maximum(1.0 - phi, 1.0e-6)) ** (1.0 / 3.0)
        * ((rho_m - rho_g) * g / (9.0 * mu))
        * (3.0 * np.maximum(phi, 0.0) / (4.0 * math.pi)) ** (2.0 / 3.0)
        * (F1 ** 2 - F2 ** 2)
        / np.maximum(den, 1.0e-8)
        * Fc
        * (1.0 - phi / phicrit)
        * (R_COND - rb) / R_COND
    )
    return np.where(vivo & (den > 1.0e-8), tasa, 0.0)


def _transportar(prev, P, uz, ur, mu, phicrit, h):
    """Euler explícito de u_r ∂_r + u_z ∂_z = Γ, con ξ y N del nivel anterior."""
    xi = np.asarray(prev["xi"], dtype=float).copy()
    N = np.asarray(prev["N"], dtype=float).copy()
    r = prev["r"]
    urn = _ur_nodos(ur)
    uzs = np.maximum(uz, 1.0e-3)
    uzs[-1] = max(float(uzs[-2]), 1.0e-3)
    nsub = 8
    hs = h / nsub
    for _ in range(nsub):
        Gx = _gamma_xi(P, xi)
        GN = _gamma_N(P, phi_henry(P, xi), N, mu, phicrit)
        xi = xi + (hs / uzs) * (Gx - urn * _d_dr(r, xi))
        N = N + (hs / uzs) * (GN - urn * _d_dr(r, N))
        xi = np.clip(xi, 0.0, float(xmax))
        N = np.maximum(N, 1.0)
    return xi, N


def _paso(prev, h, mu, phicrit):
    n = prev["r"].size
    rho0 = rho_mix(prev["P"], prev["xi"])
    uz_c = float(prev["uz"][0])
    mu_c = float(np.mean(mu))
    Gvis = -float(np.mean(rho0)) * g - 8.0 * mu_c * (0.5 * uz_c) / R_COND ** 2
    P = np.maximum(prev["P"] + Gvis * h, 1.0e5)
    rho = np.maximum(rho_mix(P, prev["xi"]), 1.0)
    uz = prev["uz"] * rho0 / rho
    uz[-1] = 0.0
    ur = np.zeros_like(prev["ur"])
    y0 = _empacar(P, uz, ur)
    lo = np.concatenate([
        np.full(n, 1.0e5),
        np.zeros(n - 1),
        np.full(n - 1, -30.0),
    ])
    hi = np.concatenate([
        np.full(n, P_BASE * 1.02),
        np.full(n - 1, 800.0),
        np.full(n - 1, 30.0),
    ])
    y0 = np.minimum(np.maximum(y0, lo + 1.0e-8), hi - 1.0e-8)

    def fun(y, mu=mu):
        return residual(y, prev, h, mu)

    sol = least_squares(
        fun, y0, bounds=(lo, hi), method="trf",
        ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=60,
    )
    P, uz, ur = _desempacar(sol.x, n)
    mu_n = mu_mix(P, uz, prev["dr"])
    xi, Nd = _transportar(prev, P, uz, ur, mu_n, phicrit, h)
    nuevo = {
        "z": prev["z"] + h,
        "r": prev["r"],
        "dr": prev["dr"],
        "P": P,
        "uz": uz,
        "ur": ur,
        "mu": mu_n,
        "phi": phi_henry(P, xi),
        "ugz": uz.copy(),
        "N": Nd,
        "xi": xi,
    }
    return nuevo, float(sol.cost)


def marchar_exsol(vin=5.0, n_r=17, h_sb=40.0, h=20.0, phicrit=0.7):
    base = marchar(vin=vin, n_r=n_r, h=h_sb)
    hist = []
    for s in base:
        t = dict(s)
        t["phi"] = phi_henry(s["P"])
        t["ugz"] = s["uz"].copy()
        t["N"] = np.full(s["r"].size, N0)
        t["xi"] = np.full(s["r"].size, XI0)
        hist.append(t)
    st = hist[-1]
    h_uso = h
    while st["z"] < -0.5 and len(hist) < 800:
        if st["z"] + h_uso > 0.0:
            h_uso = max(0.5, -st["z"])
        nuevo, costo = _paso(st, h_uso, st["mu"], phicrit)
        cruza = float(np.max(nuevo["phi"])) >= phicrit
        malo = costo > 1.0e-8 or not np.isfinite(costo)
        if cruza or malo:
            if h_uso <= 1.0:
                break
            h_uso = max(1.0, 0.5 * h_uso)
            continue
        hist.append(nuevo)
        st = nuevo
        if st["z"] < -500.0:
            h_uso = min(h, 10.0)
        else:
            h_uso = h
        if (len(hist) - len(base)) % 15 == 0:
            print(
                f"z={st['z']:.1f} P={st['P'][0]/1e6:.2f} MPa "
                f"phi={st['phi'][0]:.4f} uz={st['uz'][0]:.2f} "
                f"ur={np.max(np.abs(st['ur'])):.3e} costo={costo:.2e}",
                flush=True,
            )
    return hist


C_DRAG = 300.0
MU_G = 1.0e-5
CS = math.sqrt(RV * T_GAS)


def n_sin_xi(P):
    """Fracción másica exsuelta después de fragmentar: Henry sin el factor (1-ξ)."""
    P = np.maximum(np.asarray(P, dtype=float), 1.0e4)
    return np.maximum((CO - C1 * P ** beta) / np.maximum(1.0 - C1 * P ** beta, 1.0e-12), 0.0)


def _dn_dP_sin_xi(P):
    P = np.maximum(np.asarray(P, dtype=float), 1.0e4)
    a = C1 * P ** beta
    den = np.maximum(1.0 - a, 1.0e-12)
    return (CO - 1.0) * a * beta / (P * den ** 2)


def _vel_frag(P, phi, q, rho_m):
    n = float(n_sin_xi(P))
    rho_g = max(float(P), 1.0e4) / (RV * T_GAS)
    phi = min(max(float(phi), 1.0e-3), 0.98)
    um = (1.0 - n) * q / (rho_m * (1.0 - phi))
    ug = n * q / (rho_g * phi)
    return um, ug


def _fmg(um, ug, phi):
    slip = ug - um
    return C_DRAG * phi * (1.0 - phi) * slip * abs(slip)


def _empacar_frag(P, phi, umz, ugz, umr, ugr):
    return np.concatenate([P, phi, umz[:-1], ugz, umr, ugr])


def _desempacar_frag(y, n):
    nf = n - 1
    i = 0
    P = y[i : i + n].copy()
    i += n
    phi = y[i : i + n].copy()
    i += n
    umz = np.concatenate([y[i : i + n - 1], [0.0]])
    i += n - 1
    ugz = y[i : i + n].copy()
    i += n
    umr = y[i : i + nf].copy()
    i += nf
    ugr = y[i : i + nf].copy()
    return P, phi, umz, ugz, umr, ugr


def _upwind_cara(ur, x):
    """Derivada radial de u_r en las caras, aguas arriba. En los extremos u_r=0."""
    u = np.concatenate([[0.0], ur, [0.0]])
    d = np.zeros(ur.size)
    for j in range(ur.size):
        if ur[j] >= 0.0:
            d[j] = (u[j + 1] - u[j]) / max(x[j + 1] - x[j], 1.0e-8)
        else:
            d[j] = (u[j + 2] - u[j + 1]) / max(x[j + 2] - x[j + 1], 1.0e-8)
    return d


def residual_frag(y, prev, h, mu):
    """Dos masas, dos momentos verticales y dos radiales. n(P) va sin ξ."""
    r, dr = prev["r"], prev["dr"]
    n = r.size
    P, phi, umz, ugz, umr, ugr = _desempacar_frag(y, n)
    phi = np.clip(phi, 1.0e-3, 0.98)
    rho_m = rho_de(P)
    rho_g = np.maximum(P, 1.0e4) / (RV * T_GAS)
    rho_m0 = rho_de(prev["P"])
    rho_g0 = np.maximum(prev["P"], 1.0e4) / (RV * T_GAS)
    phi0 = prev["phi"]
    umr_n = _ur_nodos(umr)
    ugr_n = _ur_nodos(ugr)
    phi_f = 0.5 * (phi[:-1] + phi[1:])
    rho_mf = 0.5 * (rho_m[:-1] + rho_m[1:])
    rho_gf = 0.5 * (rho_g[:-1] + rho_g[1:])
    mu_f = 0.5 * (mu[:-1] + mu[1:])
    umz_f = 0.5 * (umz[:-1] + umz[1:])
    ugz_f = 0.5 * (ugz[:-1] + ugz[1:])

    dndp = _dn_dP_sin_xi(P)
    jz = rho_m * (1.0 - phi) * umz + rho_g * phi * ugz
    jr = rho_m * (1.0 - phi) * umr_n + rho_g * phi * ugr_n
    Gamma = jr * dndp * _d_dr(r, P) + jz * dndp * (P - prev["P"]) / h

    jz_m = rho_m * (1.0 - phi) * umz
    jz_m0 = rho_m0 * (1.0 - phi0) * prev["uz"]
    jz_g = rho_g * phi * ugz
    jz_g0 = rho_g0 * phi0 * prev["ugz"]
    masa_m = (jz_m - jz_m0) / h + _div(r, dr, rho_mf * (1.0 - phi_f) * umr) + Gamma
    masa_g = (jz_g - jz_g0) / h + _div(r, dr, rho_gf * phi_f * ugr) - Gamma

    Fz = _fmg(umz, ugz, phi)
    lap_mz = _lap_uz(r, dr, mu * (1.0 - phi), umz)
    lap_gz = _lap_uz(r, dr, MU_G * phi, ugz)
    dPdz = (P - prev["P"]) / h
    adv_m = umz * (umz - prev["uz"]) / h + umr_n * _d_dr(r, umz)
    adv_g = ugz * (ugz - prev["ugz"]) / h + ugr_n * _d_dr(r, ugz)
    mom_mz = (
        rho_m * (1.0 - phi) * adv_m
        + (1.0 - phi) * dPdz
        + rho_m * (1.0 - phi) * g
        - Fz
        - lap_mz
    )
    mom_gz = (
        rho_g * phi * adv_g
        + phi * dPdz
        + rho_g * phi * g
        + Fz
        - lap_gz
    )

    Fr = _fmg(umr, ugr, phi_f)
    lap_mr = _lap_ur(r, mu_f * (1.0 - phi_f), umr)
    lap_gr = _lap_ur(r, MU_G * phi_f, ugr)
    x = np.concatenate([[0.0], 0.5 * (r[:-1] + r[1:]), [r[-1]]])
    dumr = _upwind_cara(umr, x)
    dugr = _upwind_cara(ugr, x)
    dPdr = (P[1:] - P[:-1]) / dr
    mom_mr = (
        rho_mf * (1.0 - phi_f) * (umz_f * (umr - prev["ur"]) / h + umr * dumr)
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

    esc_mm = max(float(np.max(np.abs(jz_m0))) / h, 1.0)
    esc_mg = max(float(np.max(np.abs(jz_g0))) / h, 1.0)
    esc_mz = max(float(np.max(rho_m0 * (1.0 - phi0) * g)), 1.0)
    esc_gz = max(float(np.max(rho_g0 * phi0 * g)), 1.0)
    # La pared no es un volumen: φ y u_g copian al radio interior. u_m(R)=0.
    masa_m = masa_m / esc_mm
    masa_g = masa_g / esc_mg
    masa_m[-1] = (phi[-1] - phi[-2]) / 1.0e-4
    masa_g[-1] = (ugz[-1] - ugz[-2]) / 1.0e-2
    return np.concatenate([
        masa_m,
        masa_g,
        mom_mz[:-1] / esc_mz,
        mom_gz / esc_gz,
        mom_mr / esc_mz,
        mom_gr / esc_gz,
    ])


def _paso_frag(prev, h, mu):
    n = prev["r"].size
    nf = n - 1
    P = np.maximum(prev["P"] - rho_de(prev["P"]) * (1.0 - prev["phi"]) * g * h, Patm * 1.2)
    phi = np.clip(prev["phi"] + 0.002, 0.2, 0.97)
    umz = prev["uz"].copy()
    ugz = prev["ugz"].copy()
    umr = prev["ur"].copy()
    ugr = prev["ugr"].copy()
    y0 = _empacar_frag(P, phi, umz, ugz, umr, ugr)
    lo = np.concatenate([
        np.full(n, Patm),
        np.full(n, 0.20),
        np.full(n - 1, 0.0),
        np.full(n, 0.0),
        np.full(nf, -80.0),
        np.full(nf, -80.0),
    ])
    hi = np.concatenate([
        np.maximum(prev["P"] * 1.02, Patm * 2.0),
        np.full(n, 0.98),
        np.full(n - 1, 0.98 * CS),
        np.full(n, 0.98 * CS),
        np.full(nf, 80.0),
        np.full(nf, 80.0),
    ])
    y0 = np.minimum(np.maximum(y0, lo + 1.0e-8), hi - 1.0e-8)

    def fun(y, mu=mu):
        return residual_frag(y, prev, h, mu)

    sol = least_squares(
        fun, y0, bounds=(lo, hi), method="trf",
        ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=180,
    )
    P, phi, umz, ugz, umr, ugr = _desempacar_frag(sol.x, n)
    return P, phi, umz, ugz, umr, ugr, float(sol.cost)


def marchar_frag(pre, h=0.5):
    """Desde z_f, con n(P) sin ξ. Siguen u_r, el momento radial y la viscosidad."""
    st = pre[-1]
    q = rho_mix(st["P"], st["xi"]) * st["uz"]
    rho_m = rho_de(st["P"])
    um = np.zeros_like(st["uz"])
    ug = np.zeros_like(st["uz"])
    for i in range(um.size - 1):
        if q[i] <= 0.0:
            continue
        um[i], ug[i] = _vel_frag(st["P"][i], st["phi"][i], q[i], rho_m[i])
    um[-1] = 0.0
    ug[-1] = ug[-2]
    salto = dict(st)
    salto["uz"] = um
    salto["ugz"] = ug
    salto["ur"] = np.asarray(st["ur"], dtype=float).copy()
    salto["ugr"] = np.asarray(st["ur"], dtype=float).copy()
    hist = [salto]
    z = st["z"]
    h_uso = min(h, 0.5)
    while z < -0.5 and len(hist) < 500:
        if z + h_uso > 0.0:
            h_uso = max(0.25, -z)
        P, phi, umz, ugz, umr, ugr, costo = _paso_frag(hist[-1], h_uso, hist[-1]["mu"])
        en_borde = max(float(np.max(np.abs(umr))), float(np.max(np.abs(ugr)))) > 70.0
        malo = (not np.isfinite(costo)) or costo > 5.0e-3 or en_borde
        if malo and h_uso > 0.25:
            h_uso = max(0.25, 0.5 * h_uso)
            continue
        if malo:
            print(
                f"frag se detiene en z={z:.2f} costo={costo:.2e} "
                f"|ur|={np.max(np.abs(umr)):.2f} |ugr|={np.max(np.abs(ugr)):.2f}",
                flush=True,
            )
            break
        mu_n = np.array([
            _mu_nodo(float(P[i]), float(min(phi[i], 0.9)), float(_gdot(umz, st["dr"])[i]))
            for i in range(P.size)
        ])
        z = z + h_uso
        hist.append({
            "z": z,
            "r": st["r"],
            "dr": st["dr"],
            "P": P.copy(),
            "uz": umz.copy(),
            "ugz": ugz.copy(),
            "ur": umr.copy(),
            "ugr": ugr.copy(),
            "mu": mu_n,
            "phi": phi.copy(),
            "N": np.asarray(st["N"], dtype=float).copy(),
            "xi": np.asarray(st["xi"], dtype=float).copy(),
        })
        if len(hist) == 2 or len(hist) % 8 == 0:
            Fz = _fmg(umz, ugz, np.clip(phi, 1.0e-3, 0.98))
            lap = _lap_uz(st["r"], st["dr"], hist[-2]["mu"] * (1.0 - phi), umz)
            dP = (P - hist[-2]["P"]) / max(z - hist[-2]["z"], 1.0e-6)
            print(
                f"frag z={z:.2f} P={P[0]/1e6:.3f} MPa phi={phi[0]:.3f} "
                f"um={umz[0]:.1f} ug={ugz[0]:.1f} "
                f"|ur|={np.max(np.abs(umr)):.3e} |ugr|={np.max(np.abs(ugr)):.3e} "
                f"costo={costo:.2e} h={z - hist[-2]['z']:.2f} "
                f"eje visc={-lap[0]:.3e} drag={-Fz[0]:.3e} dP={(1.0 - phi[0]) * dP[0]:.3e}",
                flush=True,
            )
        if ugz[0] >= 0.98 * CS or P[0] <= Patm * 1.05:
            break
        if costo < 1.0e-4 and h_uso < h:
            h_uso = min(h, h_uso * 1.25)
    return hist


def _caudal(st):
    r = st["r"]
    jz = rho_mix(st["P"]) * st["uz"]
    return float(2.0 * math.pi * np.trapezoid(jz * r, r))


def graficar(hist, ruta, phicrit, phi1, z_frag=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.array([s["z"] for s in hist])
    zk = z / 1000.0
    P = np.array([s["P"][0] for s in hist])
    phi = np.array([s["phi"][0] for s in hist])
    uz = np.array([s["uz"][0] for s in hist])
    ug = np.array([s["ugz"][0] for s in hist])
    ur = []
    for s in hist:
        comps = [np.asarray(s["ur"], dtype=float)]
        if "ugr" in s:
            comps.append(np.asarray(s["ugr"], dtype=float))
        vals = np.concatenate([np.abs(c).ravel() for c in comps])
        vals = vals[np.isfinite(vals)]
        ur.append(float(np.max(vals)) if vals.size else np.nan)
    ur = np.array(ur)
    mu = np.array([float(np.nanmean(s["mu"])) for s in hist])
    zsat = float(z[np.argmax(phi > 1.0e-6)]) if np.any(phi > 1.0e-6) else z[-1]

    fig, ax = plt.subplots(2, 4, figsize=(13.6, 6.6), sharey=True)
    ax[0, 0].semilogx(P / 1e6, zk, color="C0", lw=1.6)
    ax[0, 0].set_xlabel("P [MPa]")
    ax[0, 1].plot(phi, zk, color="C0", lw=1.6)
    ax[0, 1].axvline(phicrit, color="k", lw=0.8, ls="--")
    ax[0, 1].axvline(phi1, color="0.4", lw=0.8, ls=":")
    ax[0, 1].set_xlabel(r"$\phi$")
    ax[0, 2].plot(uz, zk, color="C0", lw=1.8, label="fundido")
    ax[0, 2].plot(ug, zk, color="C1", lw=1.2, ls="--", label="gas")
    ax[0, 2].set_xlabel(r"$u_z$ [m/s]")
    ax[0, 2].legend(frameon=False, fontsize=8)
    ax[0, 3].semilogx(np.maximum(ur, 1.0e-16), zk, color="C0", lw=1.6)
    ax[0, 3].set_xlabel(r"$|u_r|$ [m/s]")
    N = np.array([float(np.asarray(s["N"])[0]) for s in hist])
    xi = np.array([float(np.asarray(s["xi"])[0]) for s in hist])
    ax[1, 0].plot(N, zk, color="C0", lw=1.6)
    ax[1, 0].set_xlabel(r"$N$ [m$^{-3}$]")
    ax[1, 1].plot(xi, zk, color="C0", lw=1.6)
    ax[1, 1].set_xlabel(r"$\xi$")
    ax[1, 2].semilogx(np.maximum(mu, 1.0), zk, color="C0", lw=1.6)
    ax[1, 2].set_xlabel(r"$\mu$ [Pa s]")
    ax[1, 3].axis("off")
    for a in ax.ravel():
        a.axhline(zsat / 1000.0, color="0.5", lw=0.6, ls="--")
        if z_frag is not None:
            a.axhline(z_frag / 1000.0, color="C3", lw=0.7, ls=":")
        a.grid(True, alpha=0.3, which="both")
    ax[0, 0].set_ylabel("z [km]")
    ax[1, 0].set_ylabel("z [km]")
    fig.tight_layout()
    fig.savefig(ruta, dpi=140)
    plt.close(fig)
    return {"z": z, "P": P, "phi": phi, "uz": uz, "ug": ug, "ur": ur, "mu": mu}


if __name__ == "__main__":
    vin = 15.5
    umb = _umbrales(vin)
    print(
        f"vin {vin:.3f} m/s  P* {umb['Pstar']/1e6:.2f} MPa  Ca {umb['Ca']:.3e}  "
        f"phi1 {umb['phi1']:.3f}  phi_crit {umb['phicrit']:.3f}",
        flush=True,
    )
    pre = marchar_exsol(vin=vin, phicrit=umb["phicrit"])
    frag = marchar_frag(pre)
    zf = pre[-1]["z"]
    hist = pre + frag
    info = graficar(hist, "/tmp/prefrag_fd.png", umb["phicrit"], umb["phi1"], z_frag=zf)
    ultimo = hist[-1]
    salto = frag[0]
    print(f"z_f {zf:.2f}  P_f {pre[-1]['P'][0]/1e6:.3f} MPa  phi_f {pre[-1]['phi'][0]:.4f}")
    print(f"salto eje um {pre[-1]['uz'][0]:.2f} -> {salto['uz'][0]:.2f}  ug {salto['ugz'][0]:.2f}")
    print(f"z_final {ultimo['z']:.2f}  P {ultimo['P'][0]/1e6:.3f}  phi {ultimo['phi'][0]:.4f}")
    print(f"um {ultimo['uz'][0]:.2f}  ug {ultimo['ugz'][0]:.2f}  cs {CS:.1f}")
    print(f"xi eje {float(np.asarray(pre[0]['xi'])[0]):.5f} -> {float(np.asarray(pre[-1]['xi'])[0]):.5f}")
    print(f"N eje {float(np.asarray(pre[0]['N'])[0]):.4e} -> {float(np.asarray(pre[-1]['N'])[0]):.4e}")
