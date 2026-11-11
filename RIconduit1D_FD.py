"""
Modelo 1D estacionario de conducto (Kozono & Koyaguchi 2009 / Slezin 2003)
resuelto por diferencias finitas en z. Sin scikits.odes / IDA.

Incógnitas diferenciales: P(z), φ(z), Nd(z), x(z).
Incógnitas algebraicas: um(z), ug(z).

Pre-fragmentación: HEM + EOS (régimen impermeable, um = ug).
Post-fragmentación: las mismas ecuaciones 2×2 de Kozono que usa IDA
(n_eq = 4): φ es independiente de la EOS, um y ug salen de la masa, y
el gas se separa del fundido. Sin proyectar φ ni capar la velocidad.

Marcha en z (cámara → boca) con RK4 y, si un paso se rompe, Euler implícito
con Picard. El método de tiro sobre vinicial es el de RIconduitex5_5_f
(rama explosiva) o el de RIconduitef5_5_f (rama efusiva).

La firma de retorno coincide con RIconduitex5_5_f para que los notebooks
y el transiente no cambien.
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
    T1,
    al2o3,
    ar1,
    ar2,
    beta,
    cao,
    f2o,
    feo,
    g,
    geometry,
    h2o1,
    k2o,
    limperh,
    limperl,
    linf,
    lsup,
    mgo,
    mno,
    model,
    na2o,
    overP1,
    p2o5,
    pfinal,
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

warnings.filterwarnings("ignore", category=RuntimeWarning)

_G = {}

_VOLC_ATTRS = (
    "C1",
    "F1",
    "F2",
    "Fc",
    "H",
    "Patm",
    "R",
    "al2o3",
    "ar1",
    "ar2",
    "beta",
    "cao",
    "f2o",
    "feo",
    "g",
    "geometry",
    "h2o1",
    "k2o",
    "limperh",
    "limperl",
    "linf",
    "lsup",
    "mgo",
    "mno",
    "model",
    "na2o",
    "overP1",
    "p2o5",
    "pfinal",
    "radius1",
    "rcrust",
    "sio2",
    "tcar",
    "tio2",
    "xi1",
    "xmax",
    "T1",
    "Tc1",
)

_DEFAULTS = {name: globals()[name] for name in _VOLC_ATTRS if name in globals()}


def _looks_like_volc(obj):
    if obj is None or isinstance(obj, (int, float, np.generic)):
        return False
    return hasattr(obj, "sio2") or hasattr(obj, "radius1")


def _volc_get(p, *names, default=None):
    if p is None:
        return default
    for name in names:
        if hasattr(p, name):
            val = getattr(p, name)
            if val is not None:
                return val
    return default


def _apply_volc(params):
    """Copia un archivo de datos de volcán a los globales del solver."""
    if params is not None and hasattr(params, "LISTO") and params.LISTO is False:
        raise ValueError(
            "El archivo del volcán todavía tiene LISTO = False. "
            "Rellenen los datos buscados y pongan LISTO = True."
        )
    g = globals()
    for name in _VOLC_ATTRS:
        if params is not None and hasattr(params, name):
            val = getattr(params, name)
        else:
            val = _DEFAULTS.get(name, g.get(name))
        if name == "model" and isinstance(val, str):
            val = val.lower()
        if val is not None:
            g[name] = val

    if params is None:
        return

    h2o = _volc_get(params, "h2o1", "h2o")
    if h2o is not None:
        h2o = float(h2o)
        g["h2o1"] = h2o * 100.0 if h2o <= 1.0 else h2o

    Tc = _volc_get(params, "Tc1", "Tc")
    T = _volc_get(params, "T1", "T")
    if T is not None:
        T = float(T)
        if T < 200.0:
            g["Tc1"] = T
            g["T1"] = T + 273.15
        else:
            g["T1"] = T
            g["Tc1"] = T - 273.15
    elif Tc is not None:
        g["Tc1"] = float(Tc)
        g["T1"] = float(Tc) + 273.15

    xi = _volc_get(params, "xi1", "xi")
    if xi is not None:
        g["xi1"] = float(xi)

    if hasattr(params, "pfinal") and params.pfinal is not None:
        g["pfinal"] = float(params.pfinal)
        g["Patm"] = float(_volc_get(params, "Patm", default=params.pfinal))


def _extract_run_args(params, radius, Pressure, wt, Temperature, content_crystal):
    if radius is None:
        radius = radius1
    if Pressure is None:
        Pressure = overP1
    if wt is None:
        wt = h2o1
    if Temperature is None:
        Temperature = T1
    if content_crystal is None:
        content_crystal = xi1
    missing = [
        n
        for n, v in (
            ("radius1", radius),
            ("overP1", Pressure),
            ("h2o1", wt),
            ("T1", Temperature),
            ("xi1", content_crystal),
        )
        if v is None
    ]
    if missing:
        raise ValueError(
            "Faltan datos en el archivo del volcán: " + ", ".join(missing) + ". "
            "Rellenen la plantilla y pongan LISTO = True."
        )
    return float(radius), float(Pressure), float(wt), float(Temperature), float(content_crystal)


def _real(x, default=0.0):
    val = np.real(x)
    if not np.isfinite(val):
        return float(default)
    return float(val)


def _clip(x, lo, hi):
    return float(min(max(_real(x, lo), lo), hi))


def _rmax(a, b=0.0):
    return max(_real(a, b), _real(b, 0.0))


def _fg(P, x_cr):
    P = max(float(np.real(P)), 1e4)
    x_cr = _clip(x_cr, 0.0, 0.999)
    den = 1.0 - C1 * P ** beta
    if abs(den) < 1e-30:
        return 0.0
    test = (1.0 - _G["xi"]) * _G["co"] - (1.0 - x_cr) * C1 * P ** beta
    if test <= 0.0:
        return 0.0
    return float(((1.0 - _G["xi"]) * _G["co"] - C1 * (1.0 - x_cr) * P ** beta) / den)


def _rb(phi, Nd):
    phi = _clip(phi, 1e-16, 0.99)
    Nd = max(float(Nd), 1.0)
    return float((phi / ((4.0 / 3.0) * math.pi * Nd * (1.0 - phi))) ** (1.0 / 3.0))


def _visc(P, phi, Nd, x_cr, um):
    wr = _G["wr"]
    P = max(float(np.real(P)), 1e4)
    phi = _clip(phi, 0.0, 0.99)
    um = max(float(np.real(um)), 1e-12)
    sr = um / wr
    test = (1.0 - _G["xi"]) * _G["co"] - (1.0 - x_cr) * C1 * P ** beta
    h2o_d = (C1 * P ** beta) * 100.0 if test > 0.0 else _G["h2o"]
    viscl = fvrel(model, x_cr, _G["xi"], ar1, ar2, _G["xmax"], sr) * viscosity(
        sio2, tio2, al2o3, feo, mno, mgo, cao, na2o, k2o, p2o5, h2o_d, f2o, _G["Tc"]
    )
    viscl = max(float(np.real(viscl)), 1.0)
    if phi < 1e-12:
        return viscl

    rb = _rb(phi, Nd)
    nca = max(rb * viscl * sr / 3.0, 1e-300)
    phicritbub = min(_G["phicrit"] + 0.05, 0.999)
    phi_c = min(phi, 0.999 * phicritbub)
    AA = (1.0 - phi_c / phicritbub) ** (-phicritbub)
    BB = (1.0 - phi_c / phicritbub) ** (5.0 * phicritbub / 3.0)
    c1 = -0.2895 * phi_c + 0.8132
    c2 = phi_c
    viscrel = 0.5 * (AA - BB) * (1.0 - math.erf(c1 * math.log(nca) + c2)) + BB
    return max(_real(viscrel * viscl, 1.0), 1.0)


def _dfgdp(P, x_cr, um, fg):
    if fg <= 0.0:
        return 0.0
    P = max(float(np.real(P)), 1e4)
    um = max(float(um), 1e-9)
    den = (_G["co"] * (1.0 - _G["xi"]) - (1.0 - _G["xmax"]) * C1 * Patm ** beta) + 1e-30
    f2 = _rmax((_G["co"] * (1.0 - _G["xi"]) - (1.0 - x_cr) * C1 * P ** beta) / den, 0.0)
    xteo = _G["xi"] + (_G["xmax"] - _G["xi"]) * f2
    f3 = _rmax(1.0 - x_cr / max(xteo, 1e-10), 0.0)
    dxdp = _rmax((_G["xmax"] - _G["xi"]) * f2 * f3 / (tcar * um), 0.0)
    num = (
        -(-dxdp * C1 * P ** beta + (1.0 - x_cr) * C1 * beta * P ** (beta - 1.0))
        + (_G["co"] * (1.0 - _G["xi"]) - (1.0 - x_cr) * C1 * P ** beta)
        * C1
        * beta
        * P ** (beta - 1.0)
    )
    return float(num / (1.0 - C1 * P ** beta) ** 2)


def _dNdt(n_eq, phi, Nd, rb, visc, rho_g):
    lim = 0.5
    phi_lim = 0.5 if n_eq == 3 else _G["phicrit"]
    if not (rb < lim * _G["wr"] and phi < phi_lim):
        return 0.0
    phi_c = _clip(phi, 1e-12, 0.99)
    Nd_s = max(float(Nd), 1.0)
    visc = max(float(visc), 1.0)
    arg = (F1 + F2) * ((3.0 * phi_c * (math.pi / (6.0 * _G["phicrit"])) / (4.0 * math.pi)) ** (1.0 / 3.0))
    denom = 1.0 - arg
    if denom <= 0.01:
        return 0.0
    return float(
        -(Nd_s ** (2.0 / 3.0))
        * ((1.0 / (1.0 - phi_c)) ** (1.0 / 3.0))
        * ((1.0 / 9.0) * (_G["rho_m"] - rho_g) * 9.81 / visc)
        * ((3.0 * phi_c / (4.0 * math.pi)) ** (2.0 / 3.0))
        * (F1 * F1 - F2 * F2)
        / denom
        * Fc
        * (1.0 - phi_c / _G["phicrit"])
        * (_G["wr"] - rb)
        / _G["wr"]
    )


def _phi_eq(P, x_cr):
    """Fracción volumétrica de gas en equilibrio (EOS + solubilidad)."""
    fg = _fg(P, x_cr)
    if fg <= 1e-16:
        return 0.0, fg
    P = max(float(np.real(P)), 1e4)
    phi = 1.0 / (1.0 + (P / (fg * R * _G["T"])) * (1.0 - fg) / _G["rho_m"])
    return _clip(phi, 0.0, 0.99), fg


def _velocities(P, phi, x_cr, n_eq):
    rho_m = _G["rho_m"]
    q = _G["q"]
    P = max(float(np.real(P)), 1e4)
    phi = _clip(phi, 1e-16, 0.99)
    if n_eq == 4:
        x_use = _G["xfinal"]
    else:
        x_use = x_cr
    fg = _fg(P, x_use)
    rho_g = P / (R * _G["T"])
    um = q * (1.0 - fg) / max((1.0 - phi) * rho_m, 1e-12)
    if fg <= 0.0:
        ug = um
    else:
        ug = q * fg / max(phi * rho_g, 1e-30)
    um = float(np.real(um))
    ug = float(np.real(ug))
    return fg, rho_g, um, ug


def _rhs(y, n_eq):
    """Derivadas d(P,φ,Nd,x)/dz y velocidades algebraicas. None si el paso es inválido."""
    P = max(float(np.real(y[0])), 1e4)
    phi = _clip(y[1], 1e-16, 0.99)
    Nd = max(float(np.real(y[2])), 1.0)
    x_cr = _clip(y[3], _G["xi"], _G["xmax"])

    fg, rho_g, um, ug = _velocities(P, max(phi, 1e-12), x_cr, n_eq)
    if not np.isfinite(um) or not np.isfinite(ug):
        return None

    if n_eq == 4:
        # Kozono n_eq=4 (igual que momenteq1 en RIconduitex5_5).
        # φ NO sigue la EOS: um y ug se despejan de la masa y se separan.
        x_use = _G["xfinal"]
        phi = _clip(phi, 1e-4, 0.985)
        fg = _fg(P, x_use)
        rho_g = max(P / (R * _G["T"]), 1e-6)
        um = (1.0 - fg) * _G["q"] / (_G["rho_m"] * max(1.0 - phi, 1e-6))
        ug = (fg * _G["q"] / (rho_g * phi)) if fg > 0.0 else um
        um, ug = _real(um, 0.0), _real(ug, 0.0)
        if not np.isfinite(um) or not np.isfinite(ug):
            return None
        if um <= 0.0 or ug <= 0.0:
            return None

        Nd_use = max(_real(_G.get("Nd_final", _G.get("Nd", 1.0)), 1.0), 1.0)
        rb = _rb(phi, Nd_use)
        ra, cd = 1e-3, 0.8
        Fgw = 0.01 * rho_g * abs(ug) * ug / (4.0 * _G["wr"])
        slip = ug - um
        if phi < _G["phicrit"] + 0.05:
            tt = _clip((phi - _G["phicrit"]) / 0.05, 0.0, 1.0)
            Fmg = (
                ((0.33 / (4.0 * max(rb, 1e-9))) ** (1.0 - tt))
                * ((3.0 * cd / (8.0 * ra)) ** tt)
                * rho_g
                * abs(slip)
                * slip
                * phi
                * (1.0 - phi)
            )
        else:
            Fmg = 3.0 * cd * rho_g * abs(slip) * slip * phi * (1.0 - phi) / (8.0 * ra)
        Fmg = _real(Fmg, 0.0)

        test = (1.0 - _G["xi"]) * _G["co"] - (1.0 - x_use) * C1 * P ** beta
        if test <= 0.0:
            dfgdp = 0.0
        else:
            den = (_G["co"] * (1.0 - _G["xi"]) - (1.0 - _G["xmax"]) * C1 * Patm ** beta) + 1e-30
            f2 = _rmax((_G["co"] * (1.0 - _G["xi"]) - (1.0 - x_use) * C1 * P ** beta) / den, 0.0)
            xteo = _G["xi"] + (_G["xmax"] - _G["xi"]) * f2
            f3 = _rmax(1.0 - x_use / max(xteo, 1e-10), 0.0)
            dxdp = _rmax((_G["xmax"] - _G["xi"]) * f2 * f3 / tcar, 0.0)
            dfgdp = (
                -(-dxdp * C1 * P ** beta + (1.0 - x_use) * C1 * beta * P ** (beta - 1.0))
                + ((_G["co"] * (1.0 - _G["xi"]) - (1.0 - x_use) * C1 * P ** beta)
                   * C1 * beta * P ** (beta - 1.0))
            ) / (1.0 - C1 * P ** beta) ** 2
            dfgdp = _real(dfgdp, 0.0)

        aco = rho_g * ug ** 2
        bco = phi - (ug ** 2) * phi / (R * _G["T"]) + dfgdp * _G["q"] * ug
        cco = _G["rho_m"] * um ** 2
        dco = dfgdp * _G["q"] * um - (1.0 - phi)
        eco = -_G["rho_m"] * (1.0 - phi) * 9.8 + Fmg
        fco = rho_g * phi * 9.8 + Fmg + Fgw
        det = aco * dco - bco * cco
        if not np.isfinite(det) or abs(det) < 1e-18 * max(abs(aco * dco), 1.0):
            return None
        dphidz = (-eco * bco + dco * fco) / det
        dpdz = (-eco * aco + fco * cco) / det
        if not np.isfinite(dpdz) or not np.isfinite(dphidz):
            return None
        if abs(dphidz) > 2.0 or abs(dpdz) > 5e7:
            return None
        vsound = 0.99 * math.sqrt(R * _G["T"])
        if ug > 1.2 * vsound or um > 1.2 * vsound:
            return None
        return (
            np.array([_real(dpdz), _real(dphidz), 0.0, 0.0], dtype=float),
            um,
            ug,
        )

    # Pre-fragmentación: HEM + EOS. El DAE 2×2 de Kozono es rígido para
    # una marcha explícita (IDA lo integra implícito).
    phi_eos, fg = _phi_eq(P, x_cr)
    rho_g = P / (R * _G["T"])
    rho_mix = _G["rho_m"] * (1.0 - phi_eos) + rho_g * phi_eos
    rho_mix = max(rho_mix, 1.0)
    um = _G["q"] / rho_mix
    ug = um
    visc = _visc(P, phi_eos, Nd, x_cr, um)
    Fmw = _G["cg"] * visc * um / (_G["wr"] ** 2)
    dpdz = -rho_mix * g - Fmw
    epsP = max(abs(P) * 1e-6, 100.0)
    phi_eps, _ = _phi_eq(P + epsP, x_cr)
    dphidz = (phi_eps - phi_eos) / epsP * dpdz
    phi = phi_eos
    dfgdp = _dfgdp(P, x_cr, um, fg)
    rb = _rb(max(phi, 1e-16), Nd)
    if not np.isfinite(dpdz) or not np.isfinite(dphidz):
        return None

    if abs(ug) > 5.0e3 or abs(um) > 5.0e3:
        return None
    dNdt = _dNdt(n_eq, phi, Nd, rb, visc, rho_g)
    dNdz = dNdt / max(um, 1e-9)

    den = (_G["co"] * (1.0 - _G["xi"]) - (1.0 - _G["xmax"]) * C1 * Patm ** beta) + 1e-30
    f2 = _rmax((_G["co"] * (1.0 - _G["xi"]) - (1.0 - x_cr) * C1 * P ** beta) / den, 0.0)
    xteo = _G["xi"] + (_G["xmax"] - _G["xi"]) * f2
    f3 = _rmax(1.0 - x_cr / max(xteo, 1e-10), 0.0)
    if n_eq in (2, 3):
        den2 = (_G["co"] - C1 * Patm ** beta) + 1e-30
        f2 = _rmax((_G["co"] - C1 * P ** beta) / den2, 0.0)
        xteo = _G["xi"] + (_G["xmax"] - _G["xi"]) * f2
        f3 = _rmax(1.0 - x_cr / max(xteo, 1e-10), 0.0)
    if n_eq == 3:
        dxdz = _rmax((_G["xmax"] - _G["xi"]) * f2 * f3 / (tcar * max(um, 1e-9)), 0.0)
    else:
        dxdz = _rmax((_G["xmax"] - _G["xi"]) * f2 * f3 / tcar, 0.0)

    dvdz = _G["vinicial"] * (-dfgdp * dpdz * (1.0 - phi) + (1.0 - fg) * dphidz) / max((1.0 - phi) ** 2, 1e-12)
    _G["fragcrit"] = _real(dvdz * visc / (0.01 * 1e10), 0.0)
    return (
        np.array([_real(dpdz), _real(dphidz), _real(dNdz), _real(dxdz)], dtype=float),
        _real(um),
        _real(ug),
    )


def _sanitize(y, n_eq):
    out = np.array(y, dtype=float)
    out[0] = max(float(np.real(out[0])), 1e4)
    out[2] = max(float(np.real(out[2])), 1.0)
    out[3] = _clip(out[3], _G["xi"], _G["xmax"])
    if n_eq == 4:
        # φ es incógnita diferencial; no proyectar sobre la EOS
        out[1] = _clip(out[1], 1e-4, 0.985)
        out[2] = _G.get("Nd_final", out[2])
        out[3] = _G.get("xfinal", out[3])
    else:
        phi_eos, _ = _phi_eq(out[0], out[3])
        out[1] = phi_eos
    return out


def _rk4_step(y, dz, n_eq):
    r1 = _rhs(y, n_eq)
    if r1 is None:
        return None
    k1 = r1[0]
    r2 = _rhs(_sanitize(y + 0.5 * dz * k1, n_eq), n_eq)
    if r2 is None:
        return None
    k2 = r2[0]
    r3 = _rhs(_sanitize(y + 0.5 * dz * k2, n_eq), n_eq)
    if r3 is None:
        return None
    k3 = r3[0]
    r4 = _rhs(_sanitize(y + dz * k3, n_eq), n_eq)
    if r4 is None:
        return None
    k4 = r4[0]
    y_new = _sanitize(y + dz * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0, n_eq)
    r_new = _rhs(y_new, n_eq)
    if r_new is None:
        return None
    return y_new, r_new[1], r_new[2]


def _ieuler_step(y, dz, n_eq, n_pic=8):
    r0 = _rhs(y, n_eq)
    if r0 is None:
        return None
    y_new = _sanitize(y + dz * r0[0], n_eq)
    um, ug = r0[1], r0[2]
    for _ in range(n_pic):
        r = _rhs(y_new, n_eq)
        if r is None:
            return None
        y_new = _sanitize(y + dz * r[0], n_eq)
        um, ug = r[1], r[2]
    return y_new, um, ug


def _row(y, um, ug):
    return np.array([y[0], y[1], y[2], y[3], um, ug], dtype=float)


def _march_segment(z0, y0, n_eq, phi_stop, z_end=0.0, dz0=1.0):
    """Integra un tramo hasta z_end, P <= pfinal o φ >= phi_stop."""
    y = _sanitize(y0, n_eq)
    r = _rhs(y, n_eq)
    if r is None:
        return np.array([z0]), _row(y, y0[4] if len(y0) > 4 else 0.0, y0[5] if len(y0) > 5 else 0.0)[None, :]
    um, ug = r[1], r[2]
    z_list = [float(z0)]
    sol_list = [_row(y, um, ug)]

    dz_min = 1e-3 if n_eq == 4 else 5e-3
    dz_max = 2.0 if n_eq == 4 else 20.0
    dz = float(np.clip(abs(dz0), dz_min, dz_max))
    z = float(z0)
    n_steps = 0
    vsound = 0.99 * math.sqrt(R * _G["T"])

    while z < z_end - 1e-9 and n_steps < 80000:
        if y[0] <= pfinal:
            break
        if n_eq == 4 and ug >= 1.02 * vsound:
            break
        if phi_stop is not None and y[1] >= phi_stop:
            break
        if z + dz > z_end:
            dz = max(z_end - z, dz_min)

        step = _rk4_step(y, dz, n_eq)
        if step is None:
            step = _ieuler_step(y, dz, n_eq)
        if step is None:
            dz *= 0.5
            if dz < dz_min:
                break
            continue

        y_new, um_new, ug_new = step
        dphi = abs(y_new[1] - y[1])
        dP_rel = abs(y_new[0] - y[0]) / max(abs(y[0]), 1.0)
        if (dphi > 0.03 or dP_rel > 0.08) and dz > dz_min * 1.01:
            dz = max(dz * 0.5, dz_min)
            continue
        if phi_stop is not None and y[1] < phi_stop <= y_new[1] and dz > dz_min * 2.0:
            dz = max(dz * 0.5, dz_min)
            continue

        z = z + dz
        y = y_new
        z_list.append(z)
        sol_list.append(_row(y, um_new, ug_new))
        n_steps += 1

        um, ug = um_new, ug_new
        if dphi < 0.002 and dP_rel < 0.01 and not (n_eq == 4 and ug > 0.8 * vsound):
            dz = min(dz * 1.25, dz_max)
        if not np.isfinite(y[0]) or not np.isfinite(y[1]):
            break

    return np.asarray(z_list, dtype=float), np.vstack(sol_list)


def _update_thresholds(vinicial):
    """φ_crit, limphi1/2 y Nd como en RIconduitex5_5_f (número de Ca)."""
    wr = _G["wr"]
    rho_m = _G["rho_m"]
    rho_ti = _G["rho_ti"]
    Pi = _G["Pi"]
    dis = _G["dis"]
    velc = math.sqrt(15.0e9 / rho_m)
    visc = _real(
        fvrel(model, _G["xi"], _G["xi"], ar1, ar2, _G["xmax"], vinicial / wr)
        * viscosity(sio2, tio2, al2o3, feo, mno, mgo, cao, na2o, k2o, p2o5, dis * 100.0, f2o, _G["Tc"]),
        1.0,
    )
    dpdzcalc = (-rho_ti * (9.81 + _G["cg"] * visc * vinicial / ((wr * wr) * rho_ti))) / (
        1.0 - (vinicial ** 2) / (velc ** 2)
    )
    dpdt = -dpdzcalc * vinicial
    coefNd = math.log10(max(dpdt, 1e-30))
    Nd = 10.0 ** (1.5 * coefNd + 5.0)
    if C1 * Pi ** beta < _G["co"]:
        Nd = 1e8
    _G["Nd"] = float(Nd)
    _G["dpdz_ini"] = float(dpdzcalc)

    rhol = rho_m
    phisel = 0.2
    drhogdp = 1.0 / (R * _G["T"])
    Pmax = (_G["co"] / C1) ** (1.0 / beta)
    Pmin = pfinal
    Pcalc = 0.5 * (Pmax + Pmin)
    phicalc = phisel
    ncalc = _G["co"]
    for _ in range(50):
        ncalc = (1.0 - _G["xi"]) * (_G["co"] - C1 * Pcalc ** beta) / (1.0 - C1 * Pcalc ** beta)
        rhogcalc = Pcalc / (R * _G["T"])
        phicalc = 1.0 / ((1.0 / max(ncalc, 1e-30) - 1.0) * rhogcalc / rhol + 1.0)
        if abs(phicalc - phisel) < 0.002:
            break
        if phicalc < 0.198:
            Pmax = Pcalc
        else:
            Pmin = Pcalc
        Pcalc = 0.5 * (Pmax + Pmin)

    dfgdpcalc = (C1 * beta * Pcalc ** (beta - 1.0)) * (_G["co"] - 1.0) / ((1.0 - C1 * Pcalc ** beta) ** 2)
    dphidpcalc = -(
        -dfgdpcalc * (Pcalc / (R * _G["T"])) / (rhol * max(ncalc, 1e-30) ** 2)
        + (1.0 / max(ncalc, 1e-30) - 1.0) * drhogdp / rhol
    ) / (((1.0 / max(ncalc, 1e-30) - 1.0) * (Pcalc / (R * _G["T"])) / rhol + 1.0) ** 2)
    velcc = math.sqrt(15.0e9 / rhol)
    viscalc = _real(
        fvrel(model, _G["xi"], _G["xi"], ar1, ar2, _G["xmax"], vinicial / wr)
        * viscosity(
            sio2, tio2, al2o3, feo, mno, mgo, cao, na2o, k2o, p2o5, C1 * (Pcalc ** beta) * 100.0, f2o, _G["Tc"]
        ),
        1.0,
    )
    rho_ticalc = Pcalc * rhol / (rhol * R * _G["T"] * ncalc + Pcalc * (1.0 - ncalc))
    dpdzcalc2 = (-rho_ticalc * (9.81 + _G["cg"] * viscalc * vinicial / ((wr ** 2) * rho_ticalc))) / (
        1.0 - (vinicial ** 2) / (velcc ** 2)
    )
    dvdz = vinicial * (-dfgdpcalc * dpdzcalc2 * (1.0 - phicalc) + (1.0 - ncalc) * dphidpcalc * dpdzcalc2) / (
        (1.0 - phicalc) ** 2
    )
    dvdz = 0.5 * (dvdz + vinicial / wr)
    _G["fragcrit"] = dvdz * visc / (0.01 * 1e10)
    rbcalc = _rb(phicalc, Nd)
    Ca = abs(dvdz * viscalc * rbcalc / 0.3)
    Ca = max(Ca, 1e-30)
    _G["phicrit"] = ((lsup - linf) / 2.0) * math.erf(math.log10(Ca)) + (lsup + linf) / 2.0
    _G["limphi1"] = ((limperl - limperh) / 2.0) * math.erf(math.log10(Ca)) + (limperl + limperh) / 2.0
    _G["limphi2"] = _G["limphi1"] + 0.01
    return dpdzcalc


def _append(zsol, sol, z_seg, sol_seg):
    if z_seg.size == 0:
        return zsol, sol
    if zsol is None:
        return z_seg, sol_seg
    # evitar duplicar el nodo inicial del tramo
    if abs(z_seg[0] - zsol[-1]) < 1e-9:
        z_seg, sol_seg = z_seg[1:], sol_seg[1:]
        if z_seg.size == 0:
            return zsol, sol
    return np.concatenate([zsol, z_seg]), np.vstack([sol, sol_seg])


def _shot(vinicial, verbose=True):
    _G["vinicial"] = float(vinicial)
    _G["q"] = vinicial * _G["rho_ti"]
    dpdzcalc = _update_thresholds(vinicial)
    Pi = _G["Pi"]
    Nd = _G["Nd"]
    xi = _G["xi"]
    rho_m = _G["rho_m"]
    q = _G["q"]

    exi = (1.0 - xi) * (_G["co"] - C1 * Pi ** beta) / (1.0 - C1 * Pi ** beta)
    zsol, sol = None, None

    if exi <= 0.0:
        Pcrit = (_G["co"] / C1) ** (1.0 / beta)
        Pcrit = max(Pcrit, Patm)
        deltP = Pi - Pcrit
        deltH = -deltP / dpdzcalc
        Hi = H + deltH
        hspacing = 500
        zetash = np.zeros(hspacing)
        solaux = np.zeros((hspacing, 6))
        for p in range(hspacing):
            zetash[p] = H + p * deltH / hspacing
            solaux[p, 0] = Pi + p * deltH * dpdzcalc / hspacing
            solaux[p, 1] = 0.0
            solaux[p, 2] = Nd
            solaux[p, 3] = xi
            solaux[p, 4] = vinicial
            solaux[p, 5] = vinicial
        zsol = np.concatenate([zetash, [Hi]])
        sol = np.vstack([solaux, [Pcrit, 0.0, Nd, xi, vinicial, vinicial]])

        if Pcrit <= Patm:
            return zsol, sol, float(zsol[-1]), float(sol[-1, 0]), float(sol[-1, 5]), float(sol[-1, 1])

        if Hi >= 0.0:
            sol = np.vstack([sol, [Pcrit + dpdzcalc * deltH, 0.0, Nd, xi, vinicial, vinicial]])
            return zsol, sol, float(zsol[-1]), float(sol[-1, 0]), float(sol[-1, 5]), float(sol[-1, 1])

        epsd = 1.0
        if Hi > (-epsd):
            epsd = -Hi / 10.0
        had = Hi + epsd
        Pad = Pcrit + dpdzcalc * epsd
        fgad = (1.0 - xi) * (_G["co"] - C1 * Pad ** beta) / (1.0 - C1 * Pad ** beta)
        fgad = max(fgad, 1e-16)
        phiad = 1.0 / (1.0 + (Pad / (fgad * R * _G["T"])) * (1.0 - fgad) / rho_m)
        phiad = max(float(phiad), 1e-12)
        rho_gad = Pad / (R * _G["T"])
        viadm = q * (1.0 - fgad) / ((1.0 - phiad) * rho_m)
        viadg = q * fgad / (phiad * rho_gad)
        y0 = np.array([Pad, phiad, Nd, xi], dtype=float)
        z1, s1 = _march_segment(had, y0, n_eq=1, phi_stop=_G["limphi1"], dz0=max(abs(had) / 4000.0, 0.2))
        # el primer estado usa las velocidades del punto adicional
        s1[0, 4], s1[0, 5] = viadm, viadg
        zsol, sol = _append(zsol, sol, z1, s1)
    else:
        fgi = (1.0 - xi) * (_G["co"] - C1 * Pi ** beta) / (1.0 - C1 * Pi ** beta)
        phini = 1.0 / (1.0 + (Pi / (fgi * R * _G["T"])) * (1.0 - fgi) / rho_m)
        y0 = np.array([Pi, max(phini, 1e-12), Nd, xi], dtype=float)
        z1, s1 = _march_segment(H, y0, n_eq=1, phi_stop=_G["limphi1"], dz0=max(abs(H) / 4000.0, 0.2))
        zsol, sol = z1, s1

    Hi1, Pi1, phini1 = zsol[-1], sol[-1, 0], sol[-1, 1]
    if Hi1 < 0.0 and Pi1 > pfinal and phini1 >= _G["limphi1"]:
        y0 = sol[-1, :4].copy()
        z2, s2 = _march_segment(Hi1, y0, n_eq=2, phi_stop=_G["limphi2"], dz0=0.5)
        zsol, sol = _append(zsol, sol, z2, s2)

    Hi2, Pi2, phini2 = zsol[-1], sol[-1, 0], sol[-1, 1]
    if Hi2 < 0.0 and Pi2 > pfinal and phini2 >= _G["limphi2"]:
        y0 = sol[-1, :4].copy()
        z3, s3 = _march_segment(Hi2, y0, n_eq=3, phi_stop=_G["phicrit"], dz0=0.5)
        zsol, sol = _append(zsol, sol, z3, s3)

    Hi3, Pi3, phini3 = zsol[-1], sol[-1, 0], sol[-1, 1]
    _G["xfinal"] = float(sol[-1, 3])
    _G["Nd_final"] = float(sol[-1, 2])
    if Hi3 < 0.0 and Pi3 > pfinal and phini3 >= _G["phicrit"]:
        y0 = sol[-1, :4].copy()
        z4, s4 = _march_segment(Hi3, y0, n_eq=4, phi_stop=None, dz0=0.1)
        zsol, sol = _append(zsol, sol, z4, s4)

    # Igual que RIconduitex5_5 al empacar n_eq=4: um, ug se reconstruyen
    # con n = (co − C1 P^β)/(1 − C1 P^β), sin el factor (1−ξ). Eso es lo
    # que se ve en las figuras de IDA (ug se separa apenas fragmenta).
    sol = _pack_frag_velocities(sol)

    zexit = float(zsol[-1])
    pexit = float(sol[-1, 0])
    ugexit = float(sol[-1, 5])
    phiexit = float(sol[-1, 1])
    if verbose:
        print(
            f"Velocidad de salida: {ugexit} \n"
            f"Presión de salida: {pexit} \n"
            f"Profundidad de salida: {zexit} \n"
            f"Contenido de burbujas de salida: {phiexit}"
        )
    return zsol, sol, zexit, pexit, ugexit, phiexit


def _pack_frag_velocities(sol):
    """um, ug post-fragmentación como en RIconduitex5_5 (líneas 651–660)."""
    out = np.array(sol, dtype=float, copy=True)
    co = _G["co"]
    q = _G["q"]
    rho_m = _G["rho_m"]
    T = _G["T"]
    phicrit = _G["phicrit"]
    for j in range(out.shape[0]):
        if out[j, 1] < phicrit:
            continue
        P = max(float(out[j, 0]), 1e4)
        phi = _clip(out[j, 1], 1e-6, 0.99)
        test = co - C1 * P ** beta
        n = 0.0 if test <= 0.0 else test / (1.0 - C1 * P ** beta)
        rho_g = P / (R * T)
        out[j, 4] = (1.0 - n) * q / (rho_m * max(1.0 - phi, 1e-9))
        out[j, 5] = (n * q / (rho_g * max(phi, 1e-9))) if n > 0.0 else out[j, 4]
    return out


def _postprocess(zsol, sol):
    numb = sol.shape[0]
    visctot = np.zeros(numb)
    rbub = np.zeros(numb)
    wr = _G["wr"]
    for j in range(numb):
        P, phi, Nd, x_cr, um = sol[j, 0], sol[j, 1], sol[j, 2], sol[j, 3], sol[j, 4]
        test = (1.0 - _G["xi"]) * _G["co"] - (1.0 - x_cr) * C1 * P ** beta
        h2o_d = (C1 * P ** beta) * 100.0 if test > 0.0 else _G["h2o"]
        viscl = fvrel(model, x_cr, _G["xi"], ar1, ar2, _G["xmax"], max(um, 1e-12) / wr) * viscosity(
            sio2, tio2, al2o3, feo, mno, mgo, cao, na2o, k2o, p2o5, h2o_d, f2o, _G["Tc"]
        )
        viscl = max(float(np.real(viscl)), 1.0)
        phi_c = _clip(phi, 1e-16, 0.99)
        rb = _rb(phi_c, Nd)
        nca = max(rb * viscl * max(um, 1e-12) / wr / 3.0, 1e-300)
        phicritbub = min(_G["phicrit"] + 0.05, 0.999)
        phi_b = min(phi_c, 0.999 * phicritbub)
        AA = (1.0 - phi_b / phicritbub) ** (-phicritbub)
        BB = (1.0 - phi_b / phicritbub) ** (5.0 * phicritbub / 3.0)
        c1 = -0.2895 * phi_b + 0.8132
        viscrel = 0.5 * (AA - BB) * (1.0 - math.erf(c1 * math.log(nca) + phi_b)) + BB
        visc = viscrel * viscl
        visctot[j] = visc
        if phi > _G["phicrit"] and j > 0:
            visctot[j] = visctot[j - 1]
        rbub[j] = rb
    return rbub, visctot


def RIconduit1D_FD_f(
    radius,
    Pressure=None,
    wt=None,
    Temperature=None,
    content_crystal=None,
    verbose=True,
    max_shots=None,
    rama="explosiva",
    params=None,
):
    """
    Estacionario 1D por diferencias finitas.

    Se puede llamar con un archivo de datos:
        RIconduit1D_FD_f(datos.calbuco2015, rama='explosiva')
    o con los cinco números, como antes.

    rama='explosiva': tiro de RIconduitex5_5_f (boca sónica o P≈Patm con fragmentación).
    rama='efusiva': tiro de RIconduitef5_5_f (P≈Patm, z≈0, sin fragmentar).
    """
    rama = str(rama).strip().lower()
    if rama not in ("explosiva", "efusiva"):
        raise ValueError("rama debe ser 'explosiva' o 'efusiva'")
    if _looks_like_volc(radius):
        params = radius
        radius = Pressure = wt = Temperature = content_crystal = None
    _apply_volc(params)
    radius, Pressure, wt, Temperature, content_crystal = _extract_run_args(
        params, radius, Pressure, wt, Temperature, content_crystal
    )
    wr = float(radius)
    _G.clear()
    _G["wr"] = wr
    _G["T"] = float(Temperature)
    _G["Tc"] = float(Temperature) - 273.15
    _G["co"] = float(wt) / 100.0
    _G["xi"] = float(content_crystal)
    _G["h2o"] = float(wt)
    _G["xmax"] = float(xmax)
    _G["cg"] = 3 if geometry == "dyke" else 8
    _G["fragcrit"] = 0.0
    _G["xfinal"] = float(content_crystal)
    _G["Nd_final"] = 1e8
    _G["nombre"] = _volc_get(params, "nombre", default="")

    if model in ("em2s", "vona11"):
        phimax1 = 0.656 * np.exp(-(np.log10(ar1) ** 2) / (2.0 * (1.08 ** 2)))
        phimax2 = 0.656 * np.exp(-(np.log10(ar2) ** 2) / (2.0 * (1.08 ** 2)))
        if _G["xi"] > phimax1:
            _G["xi"] = 0.95 * phimax1
        if (_G["xmax"] - _G["xi"]) / (1.0 - _G["xi"]) > phimax2:
            _G["xmax"] = 0.95 * phimax2 * (1.0 - _G["xi"]) + _G["xi"]

    Pi = rcrust * g * abs(H) + float(Pressure)
    _G["Pi"] = Pi
    dis = C1 * Pi ** beta
    if dis > _G["co"]:
        dis = _G["co"]
    _G["dis"] = float(dis)
    _G["rho_m"] = density(sio2, tio2, al2o3, feo, mgo, cao, na2o, k2o, dis * 100.0, _G["Tc"], Pi / 1e6)

    exi = (1.0 - _G["xi"]) * (_G["co"] - C1 * Pi ** beta) / (1.0 - C1 * Pi ** beta)
    if exi <= 0.0:
        _G["rho_ti"] = _G["rho_m"]
    else:
        fgi = exi
        _G["rho_ti"] = Pi * _G["rho_m"] / (_G["rho_m"] * R * _G["T"] * fgi + Pi * (1.0 - fgi))

    if rama == "efusiva":
        vinicial = float(_volc_get(params, "vinicialef", default=0.1))
        vmin = float(_volc_get(params, "vminef", default=1e-5))
        vmax = float(_volc_get(params, "vmaxef", default=5.0))
        if max_shots is None:
            max_shots = 50
    else:
        vinicial = float(_volc_get(params, "vinicialex", default=10.0))
        vmin = float(_volc_get(params, "vminex", default=0.1))
        vmax = float(_volc_get(params, "vmaxex", default=50.0))
        if max_shots is None:
            max_shots = 60
    vsound = 0.99 * math.sqrt(R * _G["T"])
    count = 1
    converged = False
    U_prev, P_prev = None, None
    best = None
    best_dist = 1e30
    zsol = sol = None
    zexit = pexit = ugexit = phiexit = 0.0

    while count < max_shots:
        if verbose:
            print(f"\nCount = {count} \nInitial velocity = {vinicial} \n")
        try:
            zsol, sol, zexit, pexit, ugexit, phiexit = _shot(vinicial, verbose=verbose)
        except Exception as exc:
            if verbose:
                print(f"  ERROR en la marcha: {exc}")
            vmax = vinicial
            vinicial = vmin + (vmax - vmin) / 2.0
            count += 1
            continue

        dist = abs(zexit) + abs(pexit - pfinal) / 1e5
        if dist < best_dist:
            best_dist = dist
            best = (zsol.copy(), sol.copy(), vinicial, count)

        if rama == "efusiva":
            # RIconduitef5_5: P≈Patm, z≈0 y sin cruzar φ_crit.
            ok_ef = (
                (pfinal - 0.05e5) <= pexit <= (pfinal + 0.05e5)
                and -2.0 < zexit <= 0.0
                and phiexit <= _G["phicrit"]
            )
            if ok_ef:
                converged = True
                if verbose:
                    print(">>> Solucion efusiva de tiro convergida.")
                break
            v_shot = vinicial
            if (pexit < (pfinal - 0.01e5) or zexit < -2.0) or phiexit > _G["phicrit"]:
                vmax = v_shot
                vinicial = vmin + (vmax - vmin) / 2.0
            elif pexit > (pfinal + 0.05e5) or zexit > 0.0:
                vmin = v_shot
                vinicial = vmin + (vmax - vmin) / 2.0
            else:
                vinicial = vmin + (vmax - vmin) / 2.0
        else:
            ok1 = zexit >= -5 and ((pexit >= pfinal and 0.95 * vsound < ugexit <= 1.05 * vsound) or (
                abs(pexit - pfinal) < 0.05e5 and phiexit >= _G["phicrit"]
            ))
            ok2 = zexit >= -15 and abs(pexit - pfinal) < 1.5e5 and ugexit > 0.5
            if ok1:
                converged = True
                if verbose:
                    print(">>> Solucion de tiro convergida.")
                break
            if ok2:
                converged = True
                if verbose:
                    print(">>> Solucion de tiro convergida (tolerancia relajada en P/z).")
                break

            v_shot = vinicial
            if count > 20 and U_prev is not None:
                f0 = P_prev - pfinal
                f1 = pexit - pfinal
                if abs(f1 - f0) > 1e2:
                    v_sec = v_shot - f1 * (v_shot - U_prev) / (f1 - f0)
                    if np.isfinite(v_sec):
                        v_sec = float(np.clip(v_sec, vmin + 1e-4, vmax - 1e-4))
                        if vmin < v_sec < vmax:
                            vinicial = v_sec

            if pexit < pfinal:
                vmax = v_shot
                if vinicial == v_shot:
                    vinicial = vmin + (vmax - vmin) / 2.0
            else:
                if zexit < -5 or ugexit > 1.02 * vsound:
                    vmax = v_shot
                    if vinicial == v_shot:
                        vinicial = vmin + (vmax - vmin) / 2.0
                elif ugexit <= 0.98 * vsound:
                    vmin = v_shot
                    if vinicial == v_shot:
                        vinicial = vmin + (vmax - vmin) / 2.0

        U_prev, P_prev = v_shot, pexit
        if verbose:
            print(f"vmax: {vmax} \nvmin: {vmin}")
        if (vmax - vmin) < 1e-4:
            if verbose:
                print("Intervalo de bisección agotado.")
            break
        count += 1

    if not converged:
        if verbose:
            if rama == "efusiva":
                print(
                    "No hay solución efusiva: no se alcanza P_atm sin fragmentar. "
                    f"Ultima z={zexit:.1f} m, P={pexit:.3e} Pa, "
                    f"phi={phiexit:.4f}, ug={ugexit:.3f} m/s."
                )
            else:
                print(
                    "ADVERTENCIA: el metodo de tiro NO convergio. "
                    f"Ultima z={zexit:.1f} m, P={pexit:.3e} Pa, ug={ugexit:.3f} m/s."
                )
        if best is not None and (zsol is None or abs(zsol[-1]) > 20.0):
            zsol, sol, vinicial, count = best[0], best[1], best[2], best[3]

    rbub, visctot = _postprocess(zsol, sol)
    return [
        zsol,
        sol,
        count,
        vinicial,
        _G["rho_m"],
        rbub,
        visctot,
        _G["rho_ti"],
        _G.get("fragcrit", 0.0),
        _G["phicrit"],
        _G["limphi1"],
        _G["limphi2"],
        _G["co"],
        _G["xi"],
        _G.get("xfinal", _G["xi"]),
        _G["cg"],
    ]


if __name__ == "__main__":
    out = RIconduit1D_FD_f(radius1, overP1, h2o1, T1, xi1, verbose=True)
    zsol, sol, count, vinicial, rho_m, rbub, visctot, rho_ti, fragcrit, phicrit, limphi1, limphi2, co, xi, xfinal, cg = out
    print("\n── Resumen 1D diferencias finitas ──")
    print(f"  vinicial = {vinicial:.4f} m/s   count = {count}")
    print(f"  z_exit   = {zsol[-1]:.2f} m")
    print(f"  P_exit   = {sol[-1, 0]/1e5:.3f} bar")
    print(f"  phi_exit = {sol[-1, 1]:.4f}")
    print(f"  um_exit  = {sol[-1, 4]:.2f} m/s")
    print(f"  ug_exit  = {sol[-1, 5]:.2f} m/s")
    print(f"  φ_crit   = {phicrit:.3f}   limφ = [{limphi1:.3f}, {limphi2:.3f}]")
    try:
        import matplotlib.pyplot as plt

        z_km = zsol / 1e3
        fig, axes = plt.subplots(1, 3, figsize=(12, 6), sharey=True)
        axes[0].semilogx(sol[:, 0] / 1e6, z_km)
        axes[0].set_xlabel("P [MPa]")
        axes[0].set_ylabel("z [km]")
        axes[1].plot(sol[:, 1], z_km)
        axes[1].axvline(phicrit, color="k", ls="--", lw=0.8)
        axes[1].set_xlabel("φ")
        axes[2].plot(sol[:, 4], z_km, label="um")
        axes[2].plot(sol[:, 5], z_km, label="ug")
        axes[2].set_xlabel("u [m/s]")
        axes[2].legend()
        fig.suptitle(f"1D estacionario FD — Calbuco  v={vinicial:.2f} m/s")
        fig.tight_layout()
        fig.savefig("RIconduit1D_FD_calbuco.png", dpi=140)
        print("  figura: RIconduit1D_FD_calbuco.png")
    except Exception as exc:
        print(f"  (sin figura: {exc})")
