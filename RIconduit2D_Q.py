"""
Conducto 2D cuya estructura en z es la del 1D.

P(z), φ(z), N(z), ξ(z) y las velocidades medias salen del mismo tiro que
RIconduitex5_5: litostático mientras no hay gas, y después de la exsolución
la inercia empina dP/dz hasta la boca (presión cercana a la atmosférica,
gas cerca del sónico). En r el Thomas da la forma de u_m y u_g; se escala
para que la media sea la del 1D. Q queda constante. En la pared u_m = u_g = 0.
"""

from __future__ import annotations

import math

import numpy as np

from calbuco2015d import (
    H,
    T1,
    g,
    h2o1,
    overP1,
    radius1,
    rcrust,
    xi1,
    xmax,
)
from RIconduit2D_FD import _effective_visc, _rho_g, _rho_m, _thomas
import RIconduit2D_FD as _fd

MU_G = 1.8e-5
PHI_FRAG = 0.70


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
    gbl["phicrit"] = PHI_FRAG
    gbl["limphi1"] = 0.15
    gbl["limphi2"] = 0.40
    gbl["Nd0"] = 1e8
    pi = rcrust * g * abs(H) + float(pressure)
    gbl["Pi"] = pi
    gbl["rho_m"] = _rho_m(pi)
    return gbl


def _matriz_radial(factor, mu):
    r = _fd._G["r"]
    dr = _fd._G["dr"]
    nunk = len(r) - 1
    lo = np.zeros(nunk)
    di = np.zeros(nunk)
    up = np.zeros(nunk)
    a = np.maximum(np.asarray(factor, dtype=float), 0.0) * np.maximum(np.asarray(mu, dtype=float), 1e-12)
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


def _perfil(u_medio, factor, mu, dpdz, rho, peso):
    """Forma viscosa en r, escalada para que la media sea u_medio. Pared en 0."""
    r = _fd._G["r"]
    wr = _fd._G["wr"]
    n = len(r)
    u_medio = max(float(u_medio), 0.0)
    if u_medio < 1e-8:
        return np.zeros(n)
    lo, di, up = _matriz_radial(factor, mu)
    rhs = np.full(n - 1, factor[:-1] * dpdz + rho * factor[:-1] * g)
    # peso es +ρ g en el RHS del operador, ya incluido. El término `peso` queda
    # por si el llamador quiere restar un arrastre uniforme.
    rhs = rhs - peso
    try:
        u = np.zeros(n)
        u[:-1] = _thomas(lo, di, up, rhs)
        u[-1] = 0.0
        u = np.maximum(np.nan_to_num(u), 0.0)
    except Exception:
        u = np.zeros(n)
    area = math.pi * wr ** 2
    media = float(np.trapezoid(u * 2.0 * math.pi * r, r) / area)
    if media < 1e-8:
        u = 2.0 * u_medio * (1.0 - (r / wr) ** 2)
    else:
        u *= u_medio / media
    u[-1] = 0.0
    return u


def _muestras(z, n_obj=420):
    """Menos nodos, conservando el tramo final donde dP/dz se empina."""
    n = len(z)
    if n <= n_obj:
        return np.arange(n)
    cuerpo = np.linspace(0, n - 1, n_obj).astype(int)
    cola = np.linspace(int(0.9 * (n - 1)), n - 1, 80).astype(int)
    return np.unique(np.concatenate([cuerpo, cola]))


_COLUMNA = {}


def _columna_1d(radius, pressure, wt, temperature, content_crystal):
    """Un tiro del solver 1D. Devuelve la columna ya limpia y el caudal."""
    import contextlib
    import io
    import os

    clave = (float(radius), float(pressure), float(wt), float(temperature), float(content_crystal))
    if clave in _COLUMNA:
        return _COLUMNA[clave]

    from RIconduitex5_5 import RIconduitex5_5_f

    buf = io.StringIO()
    guardado = os.dup(2)
    nulo = os.open(os.devnull, os.O_WRONLY)
    os.dup2(nulo, 2)
    try:
        with contextlib.redirect_stdout(buf):
            zsol, sol, _count, vshot, _rho, _rb, _visc, rho_ti, _frag, phicrit, *_resto = (
                RIconduitex5_5_f(radius, pressure, wt, temperature, content_crystal)
            )
    finally:
        os.dup2(guardado, 2)
        os.close(nulo)
        os.close(guardado)
    zsol = np.real(np.asarray(zsol, dtype=float))
    sol = np.real(np.asarray(sol, dtype=float))
    orden = np.argsort(zsol)
    zsol, sol = zsol[orden], sol[orden]
    quedo = np.ones(len(zsol), dtype=bool)
    quedo[1:] = np.diff(zsol) > 1e-4
    zsol, sol = zsol[quedo], sol[quedo]
    q = float(vshot) * float(rho_ti)
    salida = (zsol, sol, float(vshot), q, float(phicrit))
    _COLUMNA[clave] = salida
    return salida


def marchar(vinicial=None, radius=radius1, pressure=overP1, wt=h2o1, temperature=T1,
            content_crystal=xi1, n_r=10, dz=80.0):
    """
    Columna del tiro 1D (el caudal lo elige el tiro, no `vinicial`) y perfiles
    radiales con esa velocidad media. `dz` se conserva por compatibilidad.
    """
    z, sol, vshot, q, phicrit = _columna_1d(radius, pressure, wt, temperature, content_crystal)
    idx = _muestras(z)
    z, sol = z[idx], sol[idx]
    gbl = _configurar(radius, pressure, wt, temperature, content_crystal, n_r)
    gbl["phicrit"] = phicrit
    n = gbl["N_r"]
    area = math.pi * gbl["wr"] ** 2
    q_obj = q * area

    um = np.zeros((len(z), n))
    ug = np.zeros((len(z), n))
    phi = np.zeros((len(z), n))
    nd = np.zeros((len(z), n))
    xi = np.zeros((len(z), n))
    frag = np.zeros(len(z), dtype=bool)

    for j in range(len(z)):
        p = float(sol[j, 0])
        phi_m = float(np.clip(sol[j, 1], 0.0, 0.999))
        nd_m = float(max(sol[j, 2], 1.0))
        xi_m = float(sol[j, 3])
        u_m_med = float(max(sol[j, 4], 0.0))
        u_g_med = float(max(sol[j, 5], 0.0))
        if j == 0:
            dpdz = (float(sol[1, 0]) - p) / max(z[1] - z[0], 1e-6)
        else:
            dpdz = (p - float(sol[j - 1, 0])) / max(z[j] - z[j - 1], 1e-6)
        frag[j] = phi_m >= phicrit
        rho = _rho_g(p) * phi_m + _rho_m(p) * (1.0 - phi_m)
        fac_m = np.full(n, max(1.0 - phi_m, 0.02))
        mu_m = np.full(n, _effective_visc(p, phi_m, xi_m, max(u_m_med, 1e-3), nd_m))
        um[j] = _perfil(u_m_med, fac_m, mu_m, dpdz, rho, 0.0)
        fac_g = np.full(n, max(phi_m, 1e-3))
        mu_g = np.full(n, MU_G if frag[j] else mu_m[0])
        ug[j] = _perfil(u_g_med, fac_g, mu_g, dpdz, _rho_g(p), 0.0)
        phi[j] = phi_m
        nd[j] = nd_m
        xi[j] = xi_m

    en_boca = z[-1] > -15.0
    mensaje = "boca" if en_boca else "se detuvo antes de la boca"
    return {
        "z": z,
        "P": sol[:, 0].copy(),
        "phi": phi,
        "um": um,
        "ug": ug,
        "um_media": sol[:, 4].copy(),
        "ug_media": sol[:, 5].copy(),
        "N": nd,
        "xi": xi,
        "Q": np.full(len(z), q_obj),
        "Q_objetivo": float(q_obj),
        "vinicial": float(vshot),
        "r": np.array(gbl["r"], dtype=float),
        "mensaje": mensaje,
        "rho_m_base": float(gbl["rho_m"]),
        "n_pasos": len(z) - 1,
        "frag": frag,
        "phicrit": phicrit,
    }


def tirar(radius=radius1, pressure=overP1, wt=h2o1, temperature=T1, content_crystal=xi1,
          n_r=10, dz=80.0):
    """El tiro ya vive dentro de la columna 1D."""
    return marchar(None, radius, pressure, wt, temperature, content_crystal, n_r, dz)
