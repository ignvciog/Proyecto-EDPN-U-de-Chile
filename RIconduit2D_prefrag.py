"""
Exsolución hasta antes de la fragmentación, en diferencias finitas.

Se parte del tramo sin burbujas. Desde P_sat, Henry da n y
φ = φ_Henry(P, ξ_0). Antes de fragmentar las dos fases comparten la
velocidad, así que la masa de la mezcla y los dos momentos cierran
(P, u_z, u_r). u_r queda libre dentro del conducto. Se corta cuando
φ llega a φ_crit, sin entrar en la fragmentación.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf

from calbuco2015d import (
    C1,
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


def n_henry(P):
    P = np.maximum(np.asarray(P, dtype=float), 1.0e4)
    num = (1.0 - XI0) * (CO - C1 * P ** beta)
    den = np.maximum(1.0 - C1 * P ** beta, 1.0e-12)
    return np.maximum(num / den, 0.0)


def phi_henry(P):
    P = np.asarray(P, dtype=float)
    n = n_henry(P)
    rho_m = rho_de(P)
    rho_g = np.maximum(P, 1.0e4) / (RV * T_GAS)
    phi = np.zeros_like(n, dtype=float)
    vivo = n > 1.0e-10
    phi[vivo] = 1.0 / (
        1.0 + (rho_g[vivo] / rho_m[vivo]) * (1.0 - n[vivo]) / n[vivo]
    )
    return np.clip(phi, 0.0, 0.95)


def rho_mix(P):
    phi = phi_henry(P)
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
    rho = rho_mix(P)
    rho0 = rho_mix(prev["P"])
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


def _paso(prev, h, mu):
    n = prev["r"].size
    rho0 = rho_mix(prev["P"])
    uz_c = float(prev["uz"][0])
    mu_c = float(np.mean(mu))
    Gvis = -float(np.mean(rho0)) * g - 8.0 * mu_c * (0.5 * uz_c) / R_COND ** 2
    P = np.maximum(prev["P"] + Gvis * h, 1.0e5)
    rho = np.maximum(rho_mix(P), 1.0)
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
    nuevo = {
        "z": prev["z"] + h,
        "r": prev["r"],
        "dr": prev["dr"],
        "P": P,
        "uz": uz,
        "ur": ur,
        "mu": mu_mix(P, uz, prev["dr"]),
        "phi": phi_henry(P),
        "ugz": uz.copy(),
        "N": N0,
        "xi": XI0,
    }
    return nuevo, float(sol.cost)


def marchar_exsol(vin=5.0, n_r=17, h_sb=40.0, h=20.0, phicrit=0.7):
    base = marchar(vin=vin, n_r=n_r, h=h_sb)
    hist = []
    for s in base:
        t = dict(s)
        t["phi"] = phi_henry(s["P"])
        t["ugz"] = s["uz"].copy()
        t["N"] = N0
        t["xi"] = XI0
        hist.append(t)
    st = hist[-1]
    h_uso = h
    while st["z"] < -0.5 and len(hist) < 800:
        if st["z"] + h_uso > 0.0:
            h_uso = max(0.5, -st["z"])
        nuevo, costo = _paso(st, h_uso, st["mu"])
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
CS = math.sqrt(RV * T_GAS)


def n_sin_xi(P):
    """Fracción másica exsuelta después de fragmentar: Henry sin el factor (1-ξ)."""
    P = np.maximum(np.asarray(P, dtype=float), 1.0e4)
    return np.maximum((CO - C1 * P ** beta) / np.maximum(1.0 - C1 * P ** beta, 1.0e-12), 0.0)


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


def _paso_radio(P0, phi0, um0, ug0, q, rho_m, h):
    def fun(y):
        P, phi = float(y[0]), float(y[1])
        um, ug = _vel_frag(P, phi, q, rho_m)
        rho_g = max(P, 1.0e4) / (RV * T_GAS)
        F = _fmg(um, ug, phi)
        eq_m = rho_m * um * (um - um0) / h + (P - P0) / h + rho_m * g - F / (1.0 - phi)
        eq_g = rho_g * ug * (ug - ug0) / h + (P - P0) / h + rho_g * g + F / max(phi, 1.0e-3)
        esc = max(rho_m * g, 1.0)
        return np.array([eq_m / esc, eq_g / esc])

    y0 = np.array([max(P0 - rho_m * g * h, Patm * 1.2), phi0])
    lo = np.array([Patm, 0.05])
    hi = np.array([P0 * 1.01, 0.95])
    y0 = np.minimum(np.maximum(y0, lo + 1.0e-8), hi - 1.0e-8)
    sol = least_squares(fun, y0, bounds=(lo, hi), method="trf", ftol=1e-10, xtol=1e-10, max_nfev=40)
    P, phi = float(sol.x[0]), float(sol.x[1])
    um, ug = _vel_frag(P, phi, q, rho_m)
    return P, phi, um, ug, float(sol.cost)


def marchar_frag(pre, h=5.0):
    """Desde z_f, q(r) congelado y n(P) sin ξ. Cada radio integra los dos momentos."""
    st = pre[-1]
    q = rho_mix(st["P"]) * st["uz"]
    rho_m = rho_de(st["P"])
    um = np.zeros_like(st["uz"])
    ug = np.zeros_like(st["uz"])
    for i in range(um.size - 1):
        if q[i] <= 0.0:
            continue
        um[i], ug[i] = _vel_frag(st["P"][i], st["phi"][i], q[i], rho_m[i])
    salto = dict(st)
    salto["uz"] = um
    salto["ugz"] = ug
    salto["ur"] = np.full_like(st["ur"], np.nan)
    hist = [salto]
    vivo = np.array([q[i] > 1.0 and ug[i] < 0.98 * CS for i in range(um.size - 1)] + [False])
    z = st["z"]
    P = st["P"].copy()
    phi = st["phi"].copy()
    h_uso = h
    while z < -0.5 and vivo[:-1].any() and len(hist) < 400:
        P2, phi2, um2, ug2 = P.copy(), phi.copy(), um.copy(), ug.copy()
        costos = []
        for i in range(um.size - 1):
            if not vivo[i]:
                continue
            Pi, phii, umi, ugi, costo = _paso_radio(P[i], phi[i], um[i], ug[i], q[i], rho_m[i], h_uso)
            P2[i], phi2[i], um2[i], ug2[i] = Pi, phii, umi, ugi
            costos.append(costo)
            if ugi >= 0.98 * CS or Pi <= Patm * 1.05:
                vivo[i] = False
        if costos and max(costos) > 1.0e-6 and h_uso > 0.5:
            h_uso = max(0.5, 0.5 * h_uso)
            continue
        P2[-1], phi2[-1] = P2[-2], phi2[-2]
        z = z + h_uso
        um, ug, P, phi = um2, ug2, P2, phi2
        hist.append({
            "z": z,
            "r": st["r"],
            "dr": st["dr"],
            "P": P.copy(),
            "uz": um.copy(),
            "ugz": ug.copy(),
            "ur": np.full_like(st["ur"], np.nan),
            "mu": st["mu"].copy(),
            "phi": phi.copy(),
            "N": N0,
            "xi": XI0,
        })
        h_uso = h
        if len(hist) % 8 == 0:
            print(
                f"frag z={z:.1f} P={P[0]/1e6:.2f} MPa phi={phi[0]:.3f} "
                f"um={um[0]:.1f} ug={ug[0]:.1f}",
                flush=True,
            )
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
    ur = np.array([
        np.nanmax(np.abs(s["ur"])) if np.isfinite(s["ur"]).any() else np.nan
        for s in hist
    ])
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
    ax[1, 0].plot(np.full_like(zk, N0), zk, color="C0", lw=1.6)
    ax[1, 0].set_xlim(0.0, 2.0e8)
    ax[1, 0].set_xlabel(r"$N$ [m$^{-3}$]")
    ax[1, 1].plot(np.full_like(zk, XI0), zk, color="C0", lw=1.6)
    ax[1, 1].set_xlim(0.0, 1.0)
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
