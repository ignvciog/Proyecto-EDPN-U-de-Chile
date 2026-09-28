"""
Diferencias finitas de la reducción II.F.

En cada altura, con Picard:

1. Cierres en el nivel nuevo: n(P, ξ), φ, μ, régimen.
2. Thomas en r. Pared en 0. Eje con 4 μ (1-φ) (u_1-u_0)/Δr².
   La inercia ρ u*/h va en la diagonal. El esfuerzo axial entra
   como fuerza del Picard (con u* congelada); si va implícito,
   μ/h² le gana a la viscosidad radial y la matriz cambia de signo.
3. Un solo dP/dz, el de la mezcla, para que el caudal integrado sea Q.
4. φ_i = φ_Henry(P, ξ_i). Con arrastre de Stokes las dos masas
   puntuales dan exactamente esa fracción si u_g = u_m. La otra
   forma, φ = 1 - J/(ρ u), manda φ → 1 donde la parábola se frena.
5. u_g = u_m. El Stokes no deja deslizamiento, y con eso las dos
   masas dan la fracción de Henry. La viscosidad del fundido sigue
   hasta la boca: apagarla al fragmentar pide más caída de presión
   que la que queda y el paso se va a P negativa.
6. N y ξ con diferencia hacia atrás. En la pared se copian.
7. Q se comprueba al cerrar el paso.

u_g(R) = 0. En el documento ese dato de la pared todavía no estaba.
El paso se acepta o se parte comparando h con dos pasos de h/2.
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
    limperh,
    limperl,
    overP1,
    pfinal,
    phicrit,
    radius1,
    rcrust,
    tcar,
    xi1,
    xmax,
)
from RIconduit2D_FD import _effective_visc, _fg, _rho_g, _rho_m, _thomas
from umbrales_reg import pesos_regimen

PHI_FRAG = float(phicrit)
# Caída de presión máxima que se acepta en un paso. Si es mayor, se parte h.
DP_PASO = 3.0e6
TAU_H = 0.02


def _malla(n_r, radius):
    r = np.linspace(0.0, float(radius), int(n_r))
    return r, float(r[1] - r[0])


def _lr(r, dr, coef):
    """L_r sobre los nodos libres. La pared no se despeja."""
    nunk = len(r) - 1
    lo = np.zeros(nunk)
    di = np.zeros(nunk)
    up = np.zeros(nunk)
    a = np.maximum(np.asarray(coef, dtype=float), 0.0)
    a_h = 0.5 * (a[0] + a[1])
    di[0] = -4.0 * a_h / dr ** 2
    up[0] = 4.0 * a_h / dr ** 2
    for i in range(1, nunk):
        a_p = 0.5 * (a[i] + a[i + 1])
        a_m = 0.5 * (a[i] + a[i - 1])
        ri = max(float(r[i]), 0.5 * dr)
        lo[i] = (ri - 0.5 * dr) * a_m / (ri * dr ** 2)
        di[i] = -((ri + 0.5 * dr) * a_p + (ri - 0.5 * dr) * a_m) / (ri * dr ** 2)
        if i < nunk - 1:
            up[i] = (ri + 0.5 * dr) * a_p / (ri * dr ** 2)
    return lo, di, up


def _resolver(lo, di, up, rhs):
    u = np.zeros(len(di) + 1)
    u[:-1] = _thomas(lo.copy(), di.copy(), up.copy(), np.asarray(rhs, dtype=float).copy())
    u[-1] = 0.0
    return np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)


def _sistema(r, dr, coef, alfa):
    """A = -L_r + diag(alfa)."""
    lo, di, up = _lr(r, dr, coef)
    di = -di + np.asarray(alfa, dtype=float)[:-1]
    lo = -lo
    up = -up
    di = np.where(np.abs(di) < 1e-14, 1e-14, di)
    return lo, di, up


def _phi_henry(p, xi, temperatura):
    """Fracción volumétrica de equilibrio en cada nodo."""
    phi = np.zeros(len(xi))
    rm = _rho_m(p)
    for i in range(len(xi)):
        n = _fg(p, float(xi[i]))
        if n < 1e-12:
            continue
        phi[i] = 1.0 / (1.0 + (p / (n * R * temperatura)) * (1.0 - n) / rm)
    return np.clip(phi, 0.0, 0.97)


def _media_caudal(r, flujo):
    return float(np.trapezoid(flujo * 2.0 * math.pi * r, r))


def _gamma_n(p, phi, nd, mu, w4, radius):
    out = np.zeros(len(phi))
    rm = _rho_m(p)
    rg = _rho_g(p)
    for i in range(len(phi) - 1):
        if w4[i] > 0.5 or phi[i] < 1e-5 or phi[i] > PHI_FRAG - 1e-3:
            continue
        phi_c = float(np.clip(phi[i], 1e-6, PHI_FRAG - 1e-3))
        nd_s = max(float(nd[i]), 1.0)
        rb = (phi_c / ((4.0 / 3.0) * math.pi * nd_s * max(1.0 - phi_c, 1e-6))) ** (1.0 / 3.0)
        if rb >= 0.5 * radius:
            continue
        den = 1.0 - (F1 + F2) * ((3.0 * phi_c * (math.pi / (6.0 * PHI_FRAG)) / (4.0 * math.pi)) ** (1.0 / 3.0))
        if abs(den) < 1e-8:
            continue
        out[i] = (
            -(nd_s ** (2.0 / 3.0))
            * ((1.0 / max(1.0 - phi_c, 1e-6)) ** (1.0 / 3.0))
            * ((1.0 / 9.0) * (rm - rg) * g / max(float(mu[i]), 1e-6))
            * ((3.0 * phi_c / (4.0 * math.pi)) ** (2.0 / 3.0))
            * (F1 ** 2 - F2 ** 2)
            / den
            * Fc
            * (1.0 - phi_c / PHI_FRAG)
            * (radius - rb)
            / radius
            * (1.0 - float(w4[i]))
        )
    out[-1] = out[-2]
    return out


def _tasa_xi(p, xi, xi0, co, w4):
    den0 = max(co * (1.0 - xi0) - (1.0 - xmax) * C1 * Patm ** beta, 1e-30)
    out = np.zeros(len(xi))
    for i in range(len(xi) - 1):
        f2 = max(0.0, (co * (1.0 - xi0) - (1.0 - xi[i]) * C1 * max(p, 1e4) ** beta) / den0)
        xteo = xi0 + (xmax - xi0) * f2
        f3 = max(0.0, 1.0 - float(xi[i]) / max(xteo, 1e-9))
        out[i] = (1.0 - float(w4[i])) * max(0.0, (xmax - xi0) * f2 * f3 / tcar)
    out[-1] = out[-2]
    return out


def _viscosidad(p, phi, xi, u, nd):
    return np.array([
        _effective_visc(p, float(phi[i]), float(xi[i]), max(float(u[i]), 1e-3), max(float(nd[i]), 1.0))
        for i in range(len(phi))
    ])


def _densidad_mezcla(p, phi):
    return _rho_m(p) * (1.0 - phi) + _rho_g(p) * phi


def _fallo(estado, texto):
    malo = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in estado.items()}
    malo["ok"] = False
    malo["mensaje"] = texto
    return malo


def _paso(estado, h, co, xi0, temperatura):
    """Un paso de altura h. La mezcla lleva el Thomas; el gas va pegado al fundido."""
    r = estado["r"]
    dr = estado["dr"]
    radius = float(r[-1])
    p_prev = float(estado["P"])
    u_prev = np.maximum(estado["u"], 0.0)
    ug_prev = np.maximum(estado["ug"], 0.0)
    nd_prev = estado["N"]
    xi_prev = estado["xi"]
    sig_prev = estado["sigma"]
    q_obj = float(estado["Q_obj"])
    frag = bool(estado.get("frag", False))

    u = u_prev.copy()
    p = p_prev
    dpdz = float(estado["dpdz"])
    phi = estado["phi"].copy()
    rho_c = _densidad_mezcla(p, phi)
    mu = _viscosidad(p, phi, xi_prev, u, nd_prev)
    w4 = np.zeros(len(r))

    for _ in range(6):
        phi = _phi_henry(p, xi_prev, temperatura)
        w1, w2, w3, w4 = pesos_regimen(phi, limperl, limperh, PHI_FRAG)
        # En la pared u=0 no es la tasa de corte: esa la da el nodo interior.
        u_mu = np.maximum(u, 1.0)
        u_mu[-1] = max(float(u[-2]), 1.0)
        mu = _viscosidad(p, np.minimum(phi, PHI_FRAG - 1e-3), xi_prev, u_mu, nd_prev)
        rho_c = _densidad_mezcla(p, phi)
        coef = mu * np.maximum(1.0 - phi, 1e-3)
        # Inercia en la diagonal. El esfuerzo axial va explícito, con u* = u del Picard.
        alfa = rho_c * np.maximum(u, 1e-3) / h
        lo, di, upb = _sistema(r, dr, coef, alfa)
        sig_star = coef * (u - u_prev) / h
        b = (
            -rho_c * g
            + (sig_star - sig_prev) / h
            + rho_c * np.maximum(u, 1e-3) / h * u_prev
        )
        u_p = _resolver(lo, di, upb, -np.ones(len(r) - 1))
        u_b = _resolver(lo, di, upb, b[:-1])
        ip = _media_caudal(r, rho_c * u_p)
        ib = _media_caudal(r, rho_c * u_b)
        if abs(ip) < 1e-8:
            return _fallo(estado, "sin respuesta a dP/dz")
        dpdz = (q_obj - ib) / ip
        u = dpdz * u_p + u_b
        u[-1] = 0.0
        u = np.maximum(u, 0.0)
        # P no vuelve a entrar al Picard: si φ(P) se actualiza en el mismo
        # paso, la densidad baja, el caudal pide más velocidad y dP/dz se dispara.
        if abs(float(u[0]) - float(u_prev[0])) < 1e-6 and _ > 0:
            break

    p_new = p_prev + dpdz * h
    if not np.isfinite(dpdz) or not np.isfinite(u).all() or not np.isfinite(p_new):
        return _fallo(estado, "paso no finito")
    if abs(p_new - p_prev) > 4.0 * DP_PASO or p_new < 0.5 * Patm:
        return _fallo(estado, "gradiente enorme")
    q_hecho = _media_caudal(r, rho_c * u)
    p = p_new
    # En la rama de Henry las dos masas piden u_g = u_m. El arrastre de
    # Stokes (y el de partículas si el deslizamiento arranca de cero) lo sostiene.
    ug = u.copy()

    phi = _phi_henry(p, xi_prev, temperatura)
    w1, w2, w3, w4 = pesos_regimen(phi, limperl, limperh, PHI_FRAG)
    u_mu = np.maximum(u, 1.0)
    u_mu[-1] = max(float(u[-2]), 1.0)
    mu = _viscosidad(p, np.minimum(phi, PHI_FRAG - 1e-3), xi_prev, u_mu, nd_prev)
    sig = mu * (1.0 - phi) * (u - u_prev) / h
    gN = _gamma_n(p, phi, nd_prev, mu, w4, radius)
    tasa = _tasa_xi(p, xi_prev, xi0, co, w4)
    nd = np.maximum(nd_prev + h * gN / np.maximum(u, 1e-3), 1.0)
    xi = np.clip(xi_prev + h * tasa / np.maximum(u, 1e-3), xi0, xmax)
    nd[-1] = nd[-2]
    xi[-1] = xi[-2]
    if not np.isfinite(p) or not np.isfinite(u).all() or not np.isfinite(ug).all():
        return _fallo(estado, "paso no finito")
    return {
        "r": r,
        "dr": dr,
        "P": float(p),
        "dpdz": float(dpdz),
        "u": u,
        "ug": ug,
        "phi": phi,
        "N": nd,
        "xi": xi,
        "sigma": sig,
        "sigmag": np.zeros(len(r)),
        "Q": q_hecho,
        "Q_obj": q_obj,
        "z": float(estado["z"] + h),
        "frag": frag,
        "ok": True,
        "mensaje": "",
    }


def _configurar_fd(radius, pressure, wt, temperature, content_crystal):
    import RIconduit2D_FD as fd

    co = float(wt) / 100.0
    fd._G["wr"] = float(radius)
    fd._G["T"] = float(temperature)
    fd._G["Tc"] = float(temperature) - 273.15
    fd._G["h2o"] = float(wt)
    fd._G["co"] = co
    fd._G["xi"] = float(content_crystal)
    fd._G["xmax_run"] = float(xmax)
    fd._G["phicrit"] = PHI_FRAG
    pi = rcrust * g * abs(H) + float(pressure)
    fd._G["Pi"] = pi
    fd._G["rho_m"] = _rho_m(pi)
    return co, pi


def _error_paso(a, b):
    ep = abs(float(a["P"]) - float(b["P"])) / (1.0 + abs(float(b["P"])))
    ephi = float(np.max(np.abs(a["phi"] - b["phi"]))) / (1.0 + float(np.max(np.abs(b["phi"]))))
    return max(ep, ephi)


def _paso_controlado(estado, h, h_min, h_max, co, xi0, temperatura):
    """Un paso h contra dos de h/2. Si el error pasa de TAU_H, no se acepta."""
    grande = _paso(estado, h, co, xi0, temperatura)
    if not grande["ok"] or grande["P"] > estado["P"] + 1e3:
        return None, None, h
    dphi = float(np.max(np.abs(grande["phi"] - estado["phi"])))
    if abs(grande["P"] - estado["P"]) > DP_PASO or dphi > 0.04:
        return None, None, h
    if h <= h_min * 1.5:
        return grande, 0.0, h
    medio = _paso(estado, 0.5 * h, co, xi0, temperatura)
    if not medio["ok"]:
        return None, None, h
    medio2 = _paso(medio, 0.5 * h, co, xi0, temperatura)
    if not medio2["ok"]:
        return None, None, h
    err = _error_paso(grande, medio2)
    if err > TAU_H:
        return None, err, h
    h_nuevo = float(np.clip(h * math.sqrt(TAU_H / max(err, 1e-8)), h_min, h_max))
    return medio2, err, h_nuevo


def marchar(vinicial=25.0, radius=radius1, pressure=overP1, wt=h2o1, temperature=T1,
            content_crystal=xi1, n_r=12, h0=40.0, h_min=1.0, h_max=80.0, z_tope=0.0):
    """Una marcha desde la base con velocidad de entrada v_in. Paso adaptativo."""
    r, dr = _malla(n_r, radius)
    co, pi = _configurar_fd(radius, pressure, wt, temperature, content_crystal)
    rm0 = _rho_m(pi)
    u0 = np.full(len(r), float(vinicial))
    u0[-1] = 0.0
    q0 = _media_caudal(r, rm0 * u0)
    cs = math.sqrt(R * float(temperature))
    estado = {
        "r": r,
        "dr": dr,
        "P": float(pi),
        "dpdz": -rm0 * g,
        "u": u0,
        "ug": u0.copy(),
        "phi": np.zeros(len(r)),
        "N": np.full(len(r), 1e8),
        "xi": np.full(len(r), float(content_crystal)),
        "sigma": np.zeros(len(r)),
        "sigmag": np.zeros(len(r)),
        "Q": q0,
        "Q_obj": q0,
        "z": float(H),
        "frag": False,
        "ok": True,
        "mensaje": "",
    }
    historia = [estado]
    h = min(float(h0), float(h_max))
    mensaje = "boca"
    for _ in range(8000):
        if estado["z"] >= z_tope - 0.5:
            mensaje = "boca"
            break
        if estado["P"] <= pfinal * 1.3:
            mensaje = "presion atmosferica"
            break
        ug_med = float(np.trapezoid(estado["ug"] * 2.0 * math.pi * r, r) / (math.pi * float(radius) ** 2))
        if ug_med > 0.92 * cs and float(np.mean(estado["phi"])) > 0.5:
            mensaje = "sonico"
            break
        h_uso = min(h, z_tope - estado["z"])
        nuevo, error_h, h_siguiente = _paso_controlado(
            estado, h_uso, float(h_min), float(h_max), co, float(content_crystal), float(temperature)
        )
        if nuevo is None:
            if h_uso <= float(h_min) * 1.01:
                sonda = _paso(estado, h_uso, co, float(content_crystal), float(temperature))
                if sonda["ok"] and sonda["P"] < estado["P"]:
                    estado = sonda
                    historia.append(estado)
                    continue
                # Queda menos presión de la que pide la aceleración: el tiro se ahoga.
                if (
                    sonda["mensaje"] == "gradiente enorme"
                    and estado["P"] < 2.0e7
                    and float(np.mean(estado["phi"])) > 0.4
                ):
                    mensaje = "ahogado"
                else:
                    mensaje = "paso no finito"
                break
            h = max(h_uso * 0.5, float(h_min))
            continue
        estado = nuevo
        historia.append(estado)
        h = h_siguiente

    def col(k):
        return np.vstack([e[k] for e in historia])

    area = math.pi * float(radius) ** 2
    u = col("u")
    ug = col("ug")

    def medias(vel):
        return np.array([
            float(np.trapezoid(vel[j] * 2.0 * math.pi * r, r) / area) for j in range(len(historia))
        ])

    return {
        "z": np.array([e["z"] for e in historia]),
        "P": np.array([e["P"] for e in historia]),
        "phi": col("phi"),
        "um": u,
        "ug": ug,
        "um_media": medias(u),
        "ug_media": medias(ug),
        "N": col("N"),
        "xi": col("xi"),
        "Q": np.array([e["Q"] for e in historia]),
        "Q_objetivo": float(q0),
        "vinicial": float(vinicial),
        "r": r,
        "mensaje": mensaje,
        "rho_m_base": float(rm0),
        "n_pasos": len(historia) - 1,
        "cs": float(cs),
    }


def tirar(v_min=12.0, v_max=30.0, n_tiro=7, **kwargs):
    """Caudal más alto que todavía sale por la boca. Ese deja la presión más baja."""
    lo, hi = float(v_min), float(v_max)
    mejor = None
    for _ in range(int(n_tiro)):
        v = 0.5 * (lo + hi)
        sal = marchar(v, **kwargs)
        if sal["mensaje"] == "boca" and sal["z"][-1] > -1.0:
            mejor = sal
            lo = v
        else:
            hi = v
    if mejor is None:
        return marchar(lo, **kwargs)
    return mejor
