"""
Tramo sin exsolución, en diferencias finitas.

Incógnitas: P(r,z), u_z(r,z), u_r(r,z). No hay gas.
ρ_m = ρ_m(P) con módulo K. En cada paso se impone el residuo de

    masa = 0,
    momento axial = 0,
    momento radial = 0,

con u_r libre dentro del conducto. u_r = 0 solo en el eje, en la
pared y en la base. La parábola es el dato de u_z en z = H, no un
perfil impuesto más arriba. Se corta cuando P llega a P_sat.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares

from calbuco2015d import (
    C1,
    H,
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
    tio2,
    xi1,
    xmax,
)
from density import density
from fvrel import fvrel
from viscosity import viscosity

R_COND = float(radius1)
CO = float(h2o1) / 100.0
K_BULK = 15.0e9
P_BASE = float(rcrust) * g * abs(H) + float(overP1)
P_SAT = (CO / C1) ** (1.0 / beta)
RHO_H = float(density(sio2, tio2, al2o3, feo, mgo, cao, na2o, k2o, h2o1, Tc1, P_BASE / 1e6))
XI0 = float(xi1)


def rho_de(P):
    P = np.asarray(P, dtype=float)
    return RHO_H * np.exp((P - P_BASE) / K_BULK)


def mu_de(umz, dr):
    """Giordano con el agua aún disuelta. El corte es |∂u_z/∂r|."""
    n = umz.size
    gdot = np.zeros(n)
    gdot[0] = abs(umz[1] - umz[0]) / dr
    gdot[-1] = abs(umz[-1] - umz[-2]) / dr
    if n > 2:
        gdot[1:-1] = np.abs(umz[2:] - umz[:-2]) / (2.0 * dr)
    gdot = np.maximum(gdot, 1.0e-3)
    out = np.empty(n)
    for i in range(n):
        vis = viscosity(sio2, tio2, al2o3, feo, mno, mgo, cao, na2o, k2o, p2o5, h2o1, f2o, Tc1)
        out[i] = max(float(fvrel(model, XI0, XI0, ar1, ar2, xmax, float(gdot[i])) * vis), 1.0)
    return out


def malla(n_r):
    r = np.linspace(0.0, R_COND, n_r)
    return r, float(r[1] - r[0])


def estado_inicial(vin, n_r):
    r, dr = malla(n_r)
    uz = 2.0 * vin * (1.0 - (r / R_COND) ** 2)
    uz[-1] = 0.0
    nf = r.size - 1
    return {
        "z": float(H),
        "r": r,
        "dr": dr,
        "P": np.full(r.size, P_BASE),
        "uz": uz,
        "ur": np.zeros(nf),
        "mu": mu_de(uz, dr),
    }


def _div(r, dr, F):
    n = r.size
    rf = 0.5 * (r[:-1] + r[1:])
    div = np.zeros(n)
    div[0] = 2.0 * F[0] / max(rf[0], 1.0e-8)
    for i in range(1, n - 1):
        div[i] = (rf[i] * F[i] - rf[i - 1] * F[i - 1]) / (r[i] * dr)
    div[-1] = (-rf[-1] * F[-1]) / (max(r[-1], dr) * 0.5 * dr)
    return div


def _lap_uz(r, dr, mu, u):
    n = r.size
    L = np.zeros(n)
    mu_h = 0.5 * (mu[:-1] + mu[1:])
    r_h = 0.5 * (r[:-1] + r[1:])
    flux = r_h * mu_h * np.diff(u) / dr
    L[0] = 4.0 * mu[0] * (u[1] - u[0]) / dr ** 2
    for i in range(1, n - 1):
        L[i] = (flux[i] - flux[i - 1]) / (r[i] * dr)
    L[-1] = (flux[-1] - flux[-2]) / (r[-1] * dr)
    return L


def _lap_ur(r, mu_f, ur):
    rf = 0.5 * (r[:-1] + r[1:])
    x = np.concatenate([[0.0], rf, [r[-1]]])
    u = np.concatenate([[0.0], ur, [0.0]])
    nf = ur.size
    lap = np.zeros(nf)
    for j in range(nf):
        h1 = x[j + 1] - x[j]
        h2 = x[j + 2] - x[j + 1]
        du_r = (u[j + 2] - u[j + 1]) / h2
        du_l = (u[j + 1] - u[j]) / h1
        mu_r = mu_f[min(j + 1, nf - 1)]
        mu_l = mu_f[max(j - 1, 0)]
        r_r = 0.5 * (x[j + 1] + x[j + 2])
        r_l = 0.5 * (x[j] + x[j + 1])
        ancho = 0.5 * (h1 + h2)
        centro = max(x[j + 1], 1.0e-8)
        lap[j] = (r_r * mu_r * du_r - r_l * mu_l * du_l) / (centro * ancho)
        lap[j] -= mu_f[j] * ur[j] / centro ** 2
    return lap


def _ur_nodos(ur):
    u = np.zeros(ur.size + 1)
    u[1:-1] = 0.5 * (ur[:-1] + ur[1:])
    return u


def _d_dr(r, y):
    dr = r[1] - r[0]
    d = np.zeros(y.size)
    d[0] = 0.0
    d[-1] = (y[-1] - y[-2]) / dr
    d[1:-1] = (y[2:] - y[:-2]) / (2.0 * dr)
    return d


def _empacar(P, uz, ur):
    return np.concatenate([P, uz[:-1], ur])


def _desempacar(y, n):
    nf = n - 1
    P = y[:n].copy()
    uz = np.concatenate([y[n : 2 * n - 1], [0.0]])
    ur = y[2 * n - 1 : 2 * n - 1 + nf].copy()
    return P, uz, ur


def residual(y, prev, h, mu):
    r, dr = prev["r"], prev["dr"]
    n = r.size
    P, uz, ur = _desempacar(y, n)
    rho = rho_de(P)
    rho0 = rho_de(prev["P"])
    rho_f = 0.5 * (rho[:-1] + rho[1:])
    # masa: ∂z(ρ u_z) + (1/r) ∂r(r ρ u_r) = 0
    masa = (rho * uz - rho0 * prev["uz"]) / h + _div(r, dr, rho_f * ur)
    # momento axial
    ddz = (uz - prev["uz"]) / h
    urn = _ur_nodos(ur)
    lap = _lap_uz(r, dr, mu, uz)
    mom_z = rho * (uz * ddz + urn * _d_dr(r, uz)) + (P - prev["P"]) / h + rho * g - lap
    # momento radial en las caras
    uz_f = 0.5 * (uz[:-1] + uz[1:])
    mu_f = 0.5 * (mu[:-1] + mu[1:])
    x = np.concatenate([[0.0], 0.5 * (r[:-1] + r[1:]), [r[-1]]])
    dur = np.gradient(np.concatenate([[0.0], ur, [0.0]]), x)[1:-1]
    mom_r = (
        rho_f * (uz_f * (ur - prev["ur"]) / h + ur * dur)
        + (P[1:] - P[:-1]) / dr
        - _lap_ur(r, mu_f, ur)
    )
    esc_m = max(RHO_H * max(float(np.max(prev["uz"])), 1.0) / h, 1.0)
    esc_z = max(RHO_H * g, 1.0)
    # u_z(R) no es incógnita: el momento axial se escribe en los nodos interiores
    return np.concatenate([masa / esc_m, mom_z[:-1] / esc_z, mom_r / esc_z])


def _paso(prev, h):
    n = prev["r"].size
    mu = prev["mu"]
    # semilla: Poiseuille solo para arrancar el solver; el paso lo corrige
    rho0 = rho_de(prev["P"])
    uz_c = float(prev["uz"][0])
    mu_c = float(np.mean(mu))
    Gvis = -float(np.mean(rho0)) * g - 8.0 * mu_c * (0.5 * uz_c) / R_COND ** 2
    P = np.maximum(prev["P"] + Gvis * h, 1.0e5)
    rho = rho_de(P)
    uz = prev["uz"] * rho0 / np.maximum(rho, 1.0)
    uz[-1] = 0.0
    ur = prev["ur"].copy()
    y0 = _empacar(P, uz, ur)
    lo = np.concatenate([
        np.full(n, P_SAT * 0.5),
        np.zeros(n - 1),
        np.full(n - 1, -5.0),
    ])
    hi = np.concatenate([
        np.full(n, P_BASE * 1.02),
        np.full(n - 1, 80.0),
        np.full(n - 1, 5.0),
    ])
    y0 = np.minimum(np.maximum(y0, lo + 1.0e-8), hi - 1.0e-8)

    def fun(y, mu=mu):
        return residual(y, prev, h, mu)

    sol = least_squares(fun, y0, bounds=(lo, hi), method="trf", ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=80)
    P, uz, ur = _desempacar(sol.x, n)
    nuevo = {
        "z": prev["z"] + h,
        "r": prev["r"],
        "dr": prev["dr"],
        "P": P,
        "uz": uz,
        "ur": ur,
        "mu": mu_de(uz, prev["dr"]),
    }
    return nuevo, float(sol.cost)


def _ubar(st):
    r = st["r"]
    return float(2.0 / R_COND ** 2 * np.trapezoid(st["uz"] * r, r))


def _caudal(st):
    r = st["r"]
    jz = rho_de(st["P"]) * st["uz"]
    return float(2.0 * math.pi * np.trapezoid(jz * r, r))


def marchar(vin=16.133, n_r=17, h=25.0):
    st = estado_inicial(vin, n_r)
    hist = [st]
    while st["P"].min() > P_SAT:
        h_uso = min(h, 10.0) if st["P"].min() < P_SAT + 5.0e5 else h
        # no pasar de largo la saturación en el eje: prueba y recorta
        nuevo, costo = _paso(st, h_uso)
        if nuevo["P"].min() < P_SAT and h_uso > 2.0:
            # un paso más corto para caer cerca de P_sat
            h_uso = max(2.0, h_uso * (st["P"].min() - P_SAT) / max(st["P"].min() - nuevo["P"].min(), 1.0))
            nuevo, costo = _paso(st, h_uso)
        nuevo["costo"] = costo
        hist.append(nuevo)
        st = nuevo
        if len(hist) % 20 == 0:
            print(
                f"z={st['z']:.1f} P={st['P'][0]/1e6:.2f} MPa "
                f"ubar={_ubar(st):.3f} ur={np.max(np.abs(st['ur'])):.3e} costo={costo:.2e}",
                flush=True,
            )
        if st["z"] >= -1.0:
            break
    return hist


def graficar(hist, ruta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.array([s["z"] for s in hist])
    P_eje = np.array([s["P"][0] for s in hist])
    P_pared = np.array([s["P"][-1] for s in hist])
    rho_eje = rho_de(P_eje)
    ubar = np.array([_ubar(s) for s in hist])
    Q = np.array([_caudal(s) for s in hist])
    urmax = np.array([np.max(np.abs(s["ur"])) for s in hist])
    # dP/dz numérico en el eje, y la fórmula local con ρ(P) y ū(z)
    dP = np.gradient(P_eje, z)
    mu_m = np.array([np.mean(s["mu"]) for s in hist])
    c2 = K_BULK / rho_eje
    formula = (-rho_eje * g - 8.0 * mu_m * ubar / R_COND ** 2) / (1.0 - (4.0 / 3.0) * ubar ** 2 / c2)

    fig, ax = plt.subplots(2, 3, figsize=(11.4, 6.6))
    ax[0, 0].plot(z, P_eje / 1e6, color="C0", label="eje")
    ax[0, 0].plot(z, P_pared / 1e6, color="C1", lw=1.0, label="pared")
    ax[0, 0].axhline(P_SAT / 1e6, color="k", lw=0.8, ls="--", label="P_sat")
    ax[0, 0].set_title("P [MPa]")
    ax[0, 0].legend(frameon=False, fontsize=8)
    ax[0, 1].plot(z, dP / 1e4, color="C0", label="eje, FD")
    ax[0, 1].plot(z, formula / 1e4, color="C3", ls="--", label="formula local")
    ax[0, 1].set_title(r"dP/dz [$10^{4}$ Pa/m]")
    ax[0, 1].legend(frameon=False, fontsize=8)
    ax[0, 2].plot(z, rho_eje, color="C0")
    ax[0, 2].set_title(r"$\rho_m$ en el eje [kg/m$^3$]")
    ax[1, 0].plot(z, ubar, color="C0")
    ax[1, 0].set_title(r"$\bar{u}$ [m/s]")
    ax[1, 0].set_xlabel("z [m]")
    ax[1, 1].plot(z, urmax, color="C0")
    ax[1, 1].set_title(r"max $|u_r|$ [m/s]")
    ax[1, 1].set_xlabel("z [m]")
    ax[1, 2].plot(z, Q / Q[0], color="C0")
    ax[1, 2].set_title(r"$Q/Q(H)$")
    ax[1, 2].set_xlabel("z [m]")
    for a in ax.ravel():
        a.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta, dpi=140)
    plt.close(fig)

    # perfiles radiales en tres alturas
    idxs = [0, len(hist) // 2, len(hist) - 1]
    fig, bx = plt.subplots(1, 3, figsize=(11.2, 3.6))
    r = hist[0]["r"]
    rf = 0.5 * (r[:-1] + r[1:])
    for k in idxs:
        s = hist[k]
        bx[0].plot(r, s["uz"], label=f"z={s['z']:.0f} m")
        bx[1].plot(rf, s["ur"], label=f"z={s['z']:.0f} m")
        bx[2].plot(r, s["P"] / 1e6, label=f"z={s['z']:.0f} m")
    bx[0].set_title(r"$u_z(r)$")
    bx[1].set_title(r"$u_r(r)$")
    bx[2].set_title(r"$P(r)$ [MPa]")
    for a in bx:
        a.set_xlabel("r [m]")
        a.legend(frameon=False, fontsize=8)
        a.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta.replace(".png", "_perfiles.png"), dpi=140)
    plt.close(fig)
    return {
        "z": z, "P_eje": P_eje, "dP": dP, "formula": formula,
        "ubar": ubar, "urmax": urmax, "Q": Q, "rho": rho_eje,
    }


if __name__ == "__main__":
    print(f"P_base {P_BASE/1e6:.3f} MPa  P_sat {P_SAT/1e6:.3f} MPa  rho_H {RHO_H:.2f}")
    hist = marchar(vin=16.133, n_r=17, h=40.0)
    info = graficar(hist, "/tmp/sinbub_fd.png")
    ultimo = hist[-1]
    print(f"z_final {ultimo['z']:.2f}  P_eje {ultimo['P'][0]/1e6:.3f}  P_min {ultimo['P'].min()/1e6:.3f}")
    print(f"ubar {info['ubar'][0]:.4f} -> {info['ubar'][-1]:.4f}")
    print(f"rho {info['rho'][0]:.2f} -> {info['rho'][-1]:.2f}")
    print(f"Q ratio {info['Q'][-1]/info['Q'][0]:.6f}")
    print(f"max|ur| final {info['urmax'][-1]:.4e}  pico {info['urmax'].max():.4e}")
    print(f"dP/dz eje final {info['dP'][-1]:.1f}  formula {info['formula'][-1]:.1f}")
    print(f"dP/dz eje a 200 m {info['dP'][min(5, len(info['dP'])-1)]:.1f}")
