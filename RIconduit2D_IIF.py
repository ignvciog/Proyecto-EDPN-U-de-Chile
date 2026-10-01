"""
Diferencias finitas de la reducción II.F.

Hasta la saturación la columna es algebraica: presión en recta, φ=0 y la
parábola de Poiseuille con la pared en 0. Un metro más arriba se evalúan
φ, u_m y u_g con Henry y con el caudal de cada radio. Desde ahí, en cada
altura, con Picard:

1. Cierres en el nivel nuevo: n(P, ξ), φ, μ, régimen.
2. Thomas en r. Pared en 0. Eje con 4 μ (1-φ) (u_1-u_0)/Δr².
   La inercia ρ u*/h va en la diagonal. El esfuerzo axial entra
   como fuerza del Picard (con u* congelada); si va implícito,
   μ/h² le gana a la viscosidad radial y la matriz cambia de signo.
3. Un solo dP/dz, el de la mezcla, para que el caudal integrado sea Q.
4. Mientras φ está bajo φ_crit(r), φ_i = φ_Henry(P, ξ_i) y u_g = u_m.
   Las dos masas se cumplen con esa única velocidad.
5. Cuando φ pasa φ_crit en toda la sección, el régimen es el 4 del 1D.
   φ deja de reponerse con Henry. u_m(r) sale del momento del líquido
   sin L_r y sin roce de pared; u_g(r) del momento del gas con L_r(μ_g)
   y con el arrastre de partícula. dP/dz es el que conserva el caudal
   integrado. q(r) de la base ya no rearma el perfil.
6. N y ξ con diferencia hacia atrás. En la pared se copian.
7. Q se comprueba al cerrar el paso.

u_g(R) = 0. En el documento ese dato de la pared todavía no estaba.
El paso se acepta o se parte comparando h con dos pasos de h/2.
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
    R,
    T1,
    ar1,
    ar2,
    beta,
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
    mno,
    mgo,
    model,
    na2o,
    overP1,
    p2o5,
    pfinal,
    phicrit,
    radius1,
    rcrust,
    sio2,
    tcar,
    tio2,
    xi1,
    xmax,
    al2o3,
    cao,
)
from fvrel import fvrel
from viscosity import viscosity
from RIconduit2D_FD import _effective_visc, _fg, _rho_g, _rho_m, _thomas
from umbrales_reg import pesos_regimen

PHI_FRAG = float(phicrit)
# Módulo del fundido en la recta sin burbujas. El mismo 15 GPa del 1D.
K_FUNDI = 15e9
# Metro por encima de la saturación donde arranca la marcha diferencial.
EPS_EXSOL = 1.0
# Caída de presión máxima que se acepta en un paso. Si es mayor, se parte h.
DP_PASO = 3.0e6
TAU_H = 0.02
# Viscosidad molecular del gas en L_r del tramo fragmentado.
MU_GAS = 1e-5


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


def _gamma_n(p, phi, nd, mu, w4, radius, phicrit_reg):
    out = np.zeros(len(phi))
    rm = _rho_m(p)
    rg = _rho_g(p)
    pc = np.asarray(phicrit_reg, dtype=float)
    for i in range(len(phi) - 1):
        tope = float(pc[i]) - 1e-3
        if w4[i] > 0.5 or phi[i] < 1e-5 or phi[i] > tope:
            continue
        phi_c = float(np.clip(phi[i], 1e-6, max(tope, 1e-6)))
        nd_s = max(float(nd[i]), 1.0)
        rb = (phi_c / ((4.0 / 3.0) * math.pi * nd_s * max(1.0 - phi_c, 1e-6))) ** (1.0 / 3.0)
        if rb >= 0.5 * radius:
            continue
        den = 1.0 - (F1 + F2) * ((3.0 * phi_c * (math.pi / (6.0 * pc[i])) / (4.0 * math.pi)) ** (1.0 / 3.0))
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
            * (1.0 - phi_c / pc[i])
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
        w1, w2, w3, w4 = pesos_regimen(phi, estado["limphi1"], estado["limphi2"], estado["phicrit_ca"])
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
    w1, w2, w3, w4 = pesos_regimen(phi, estado["limphi1"], estado["limphi2"], estado["phicrit_ca"])
    u_mu = np.maximum(u, 1.0)
    u_mu[-1] = max(float(u[-2]), 1.0)
    mu = _viscosidad(p, np.minimum(phi, PHI_FRAG - 1e-3), xi_prev, u_mu, nd_prev)
    sig = mu * (1.0 - phi) * (u - u_prev) / h
    gN = _gamma_n(p, phi, nd_prev, mu, w4, radius, estado["phicrit_ca"])
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
        "q_r": None if estado.get("q_r") is None else np.array(estado["q_r"], copy=True),
        "z": float(estado["z"] + h),
        "frag": frag,
        "limphi1": estado["limphi1"],
        "limphi2": estado["limphi2"],
        "phicrit_ca": estado["phicrit_ca"],
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


def _perfil(r, u_media):
    """Parábola de media u_media en la malla, con la pared en cero."""
    radio = float(r[-1])
    forma = 2.0 * (1.0 - (r / radio) ** 2)
    forma[-1] = 0.0
    area = math.pi * radio ** 2
    media = _media_caudal(r, forma) / area
    return forma * (float(u_media) / media)


def _mu_liquido(p, xi, corte, temperatura, co):
    """Viscosidad del fundido sin burbujas, al corte de referencia."""
    dis = min(float(co), C1 * float(p) ** beta) * 100.0
    tc = float(temperatura) - 273.15
    return float(
        fvrel(model, xi, xi, ar1, ar2, xmax, corte)
        * viscosity(sio2, tio2, al2o3, feo, mno, mgo, cao, na2o, k2o, p2o5, dis, f2o, tc)
    )


def _dpdz_liquido(rho, mu, u_media, radius):
    """Pendiente del tramo sin burbujas, promediada sobre la parábola."""
    c2 = K_FUNDI / float(rho)
    roce = 8.0 * float(mu) * float(u_media) / float(radius) ** 2
    return -(float(rho) * g + roce) / (1.0 - (4.0 / 3.0) * float(u_media) ** 2 / c2)


def _nd_entrada(dpdz, u_media, p, co):
    """Densidad de nucleación del 1D: 1e8 si el ingreso ya viene exsuelto."""
    if C1 * float(p) ** beta < float(co):
        return 1e8
    dpdt = abs(float(dpdz) * float(u_media))
    return 10.0 ** (1.5 * math.log10(max(dpdt, 1e-30)) + 5.0)


def limites_regiones(r, vinicial, radius, p_in, rho, mu_in, co, xi, temperatura):
    """Cortes del tiro, uno por radio, en el estado de referencia φ=0.2.

    La media ū = v_in (1-n)/(1-φ) se reparte en la parábola. En cada radio
    la extensión es ∂u/∂z y el corte es |∂u/∂r|. Ca usa el promedio de las
    dos, como el 1D promedia du/dz con v_in/R. φ_crit(r), φ_1(r) y φ_2(r)
    salen del mismo erf(log10 Ca).
    """
    pmax = (float(co) / C1) ** (1.0 / beta)
    pmin = float(pfinal)
    pcalc = 0.5 * (pmax + pmin)
    ncalc, rhog, phicalc = 0.0, 0.0, 0.0
    for _ in range(50):
        ncalc = (1.0 - xi) * (co - C1 * pcalc ** beta) / (1.0 - C1 * pcalc ** beta)
        rhog = pcalc / (R * float(temperatura))
        phicalc = 1.0 / ((1.0 / ncalc - 1.0) * rhog / rho + 1.0)
        if abs(phicalc - 0.2) < 0.002:
            break
        if phicalc < 0.198:
            pmax = pcalc
        else:
            pmin = pcalc
        pcalc = 0.5 * (pmax + pmin)
    dfgdp = (C1 * beta * pcalc ** (beta - 1.0)) * (co - 1.0) / (1.0 - C1 * pcalc ** beta) ** 2
    drhogdp = 1.0 / (R * float(temperatura))
    dphidp = -(
        -dfgdp * rhog / (rho * ncalc ** 2) + (1.0 / ncalc - 1.0) * drhogdp / rho
    ) / (((1.0 / ncalc - 1.0) * rhog / rho + 1.0) ** 2)
    mu_ref = _mu_liquido(pcalc, xi, float(vinicial) / float(radius), temperatura, co)
    cg = 3.0 if geometry == "dyke" else 8.0
    rho_t = pcalc * rho / (rho * R * float(temperatura) * ncalc + pcalc * (1.0 - ncalc))
    c2 = K_FUNDI / float(rho)
    dpdz2 = (-rho_t * (g + cg * mu_ref * float(vinicial) / (float(radius) ** 2 * rho_t))) / (
        1.0 - float(vinicial) ** 2 / c2
    )
    dudz = float(vinicial) * (
        -dfgdp * dpdz2 * (1.0 - phicalc) + (1.0 - ncalc) * dphidp * dpdz2
    ) / (1.0 - phicalc) ** 2
    nd = _nd_entrada(
        _dpdz_liquido(rho, mu_in, vinicial, radius), vinicial, p_in, co
    )
    rb = (phicalc / ((4.0 / 3.0) * math.pi * nd * (1.0 - phicalc))) ** (1.0 / 3.0)
    u_ref = float(vinicial) * (1.0 - ncalc) / (1.0 - phicalc)
    radio = float(radius)
    eta = np.asarray(r, dtype=float) / radio
    extension = np.abs(2.0 * dudz * (1.0 - eta ** 2))
    cizalle = 4.0 * u_ref * eta / radio
    edot = 0.5 * (extension + cizalle)
    mu_r = np.array([
        _mu_liquido(
            pcalc, xi, max(float(cizalle[i]), float(extension[i]), 1e-8), temperatura, co
        )
        for i in range(len(eta))
    ])
    ca = np.maximum(edot * mu_r * rb / 0.3, 1e-30)
    erf = np.array([math.erf(math.log10(float(c))) for c in ca])
    phicrit_ca = ((lsup - linf) / 2.0) * erf + (lsup + linf) / 2.0
    limphi1 = ((limperl - limperh) / 2.0) * erf + (limperl + limperh) / 2.0
    return {
        "Ca": ca,
        "edot": edot,
        "extension": extension,
        "cizalle": cizalle,
        "fragcrit": float(0.5 * (dudz + float(vinicial) / radio) * mu_in / (0.01 * 1e10)),
        "phicrit_ca": phicrit_ca,
        "limphi1": limphi1,
        "limphi2": limphi1 + 0.01,
        "Nd": float(nd),
        "P_ref": float(pcalc),
        "phi_ref": float(phicalc),
        "u_ref": float(u_ref),
        "rb": float(rb),
    }


def _estado(r, dr, z, p, dpdz, u, phi, nd, xi, q_obj, limites):
    flujo = _rho_m(p) * (1.0 - phi) * u + _rho_g(p) * phi * u
    return {
        "r": r,
        "dr": dr,
        "P": float(p),
        "dpdz": float(dpdz),
        "u": u,
        "ug": u.copy(),
        "phi": phi,
        "N": np.full(len(r), float(nd)),
        "xi": np.full(len(r), float(xi)),
        "sigma": np.zeros(len(r)),
        "sigmag": np.zeros(len(r)),
        "Q": _media_caudal(r, flujo),
        "Q_obj": float(q_obj),
        "z": float(z),
        "frag": False,
        "limphi1": limites["limphi1"],
        "limphi2": limites["limphi2"],
        "phicrit_ca": limites["phicrit_ca"],
        "ok": True,
        "mensaje": "",
    }


def _tramo_algebraico(r, dr, vinicial, radius, pi, rm0, co, xi, temperatura, limites, n_lin=40):
    """Recta sin burbujas y, un metro arriba de la saturación, el estado de Henry."""
    mu = _mu_liquido(pi, xi, float(vinicial) / float(radius), temperatura, co)
    dpdz = _dpdz_liquido(rm0, mu, vinicial, radius)
    nd = limites["Nd"]
    u_base = _perfil(r, vinicial)
    q_nodo = rm0 * u_base
    q_obj = _media_caudal(r, q_nodo)
    n_base = _fg(pi, xi)
    niveles = []
    if n_base > 0.0:
        phi = _phi_henry(pi, np.full(len(r), xi), temperatura)
        u = q_nodo * (1.0 - n_base) / ((1.0 - phi) * rm0)
        u[-1] = 0.0
        niveles.append(_estado(r, dr, H, pi, dpdz, u, phi, nd, xi, q_obj, limites))
        niveles[0]["Q"] = q_obj
        niveles[0]["q_r"] = np.array(q_nodo, copy=True)
        return niveles, dpdz, float(H), float(H)

    p_sat = (float(co) / C1) ** (1.0 / beta)
    p_sat = max(p_sat, float(Patm))
    z_sat = float(H) + (p_sat - float(pi)) / dpdz
    z_sat = min(z_sat, 0.0)
    for z in np.linspace(float(H), z_sat, int(n_lin)):
        p = float(pi) + dpdz * (float(z) - float(H))
        niveles.append(_estado(
            r, dr, z, p, dpdz, u_base, np.zeros(len(r)), nd, xi, q_obj, limites
        ))
    eps = EPS_EXSOL
    if z_sat > -eps:
        eps = max(-z_sat / 10.0, 0.0)
    z_ad = z_sat + eps
    if z_ad >= 0.0 or eps <= 0.0:
        for nivel in niveles:
            nivel["Q"] = q_obj
            nivel["q_r"] = np.array(q_nodo, copy=True)
        return niveles, dpdz, z_sat, z_sat
    p_ad = p_sat + dpdz * eps
    n_ad = (1.0 - xi) * (co - C1 * p_ad ** beta) / (1.0 - C1 * p_ad ** beta)
    rho_g = p_ad / (R * float(temperatura))
    phi_ad = 1.0 / (1.0 + (p_ad / (n_ad * R * float(temperatura))) * (1.0 - n_ad) / rm0)
    phi_ad = float(np.clip(phi_ad, 0.0, 0.97))
    u_ad = q_nodo * (1.0 - n_ad) / ((1.0 - phi_ad) * rm0)
    u_ad[-1] = 0.0
    niveles.append(_estado(
        r, dr, z_ad, p_ad, dpdz, u_ad, np.full(len(r), phi_ad), nd, xi, q_obj, limites
    ))
    for nivel in niveles:
        nivel["Q"] = q_obj
        nivel["q_r"] = np.array(q_nodo, copy=True)
    return niveles, dpdz, z_sat, z_ad


def _promedio_area(r, campo):
    radio = float(r[-1])
    area = math.pi * radio ** 2
    return float(np.trapezoid(np.asarray(campo, dtype=float) * 2.0 * math.pi * r, r) / area)


def _dfgdp_frag(p, x_cr, co, xi0):
    """∂n/∂P del régimen fragmentado: ξ está congelado, como en el 1D."""
    p = max(float(p), 1e4)
    fg = _fg(p, float(x_cr))
    if fg <= 0.0:
        return 0.0, 0.0
    den0 = co * (1.0 - xi0) - (1.0 - xmax) * C1 * Patm ** beta
    f2 = max(0.0, (co * (1.0 - xi0) - (1.0 - x_cr) * C1 * p ** beta) / den0)
    xteo = xi0 + (xmax - xi0) * f2
    f3 = max(0.0, 1.0 - float(x_cr) / max(xteo, 1e-9))
    dxdp = max(0.0, (xmax - xi0) * f2 * f3 / tcar)
    dfgdp = (
        -(-dxdp * C1 * p ** beta + (1.0 - x_cr) * C1 * beta * p ** (beta - 1.0))
        + (co * (1.0 - xi0) - (1.0 - x_cr) * C1 * p ** beta) * C1 * beta * p ** (beta - 1.0)
    ) / (1.0 - C1 * p ** beta) ** 2
    return fg, float(dfgdp)


def _seccion_fragmentada(estado):
    """φ ya pasó el φ_crit de su radio en toda la sección."""
    return bool(np.all(estado["phi"] >= estado["phicrit_ca"]))


def _cierre_fragmentado(p, phi, q, co, xi0, x_cr, radius, temperatura, nd, phicrit, um=None, ug=None):
    """Momento del 1D en régimen 4, sobre el promedio de la sección.

    Sin velocidades de entrada, las masas fijan u_m y u_g. Si el paso ya
    las trae del momento radial, dφ/dz se evalúa con esas.
    El roce del fundido con la pared no entra.
    """
    fg, dfgdp = _dfgdp_frag(p, x_cr, co, xi0)
    if fg <= 0.0 or not (0.02 < float(phi) < 0.995):
        return {"singular": True}
    rho_m = _rho_m(p)
    rho_g = max(float(p), 1e4) / (R * float(temperatura))
    if um is None or ug is None:
        um = (1.0 - fg) * q / (rho_m * (1.0 - phi))
        ug = fg * q / (rho_g * phi)
    else:
        um = float(um)
        ug = float(ug)
    rb = (phi / ((4.0 / 3.0) * math.pi * max(float(nd), 1.0) * (1.0 - phi))) ** (1.0 / 3.0)
    ra, cd = 1e-3, 0.8
    slip = ug - um
    if phi < float(phicrit) + 0.05:
        tt = min(max((phi - float(phicrit)) / 0.05, 0.0), 1.0)
        fmg = (
            ((0.33 / (4.0 * rb)) ** (1.0 - tt))
            * ((3.0 * cd / (8.0 * ra)) ** tt)
            * rho_g * abs(slip) * slip * phi * (1.0 - phi)
        )
    else:
        fmg = 3.0 * cd * rho_g * abs(slip) * slip * phi * (1.0 - phi) / (8.0 * ra)
    fgw = 0.01 * rho_g * abs(ug) * ug / (4.0 * float(radius))
    aco = rho_g * ug ** 2
    bco = phi - (ug ** 2) * phi / (R * float(temperatura)) + dfgdp * q * ug
    cco = rho_m * um ** 2
    dco = dfgdp * q * um - (1.0 - phi)
    eco = -rho_m * (1.0 - phi) * g + fmg
    fco = rho_g * phi * g + fmg + fgw
    den = aco * dco - bco * cco
    escala = abs(aco * dco) + abs(bco * cco)
    if (not np.isfinite(den)) or escala < 1e-30 or abs(den) < 1e-4 * escala:
        return {"singular": True, "um": um, "ug": ug}
    return {
        "singular": False,
        "dpdz": float((-eco * aco + fco * cco) / den),
        "dphidz": float((-eco * bco + dco * fco) / den),
        "um": float(um),
        "ug": float(ug),
    }


def _velocidades_masa(q_r, phi, p, xi, temperatura):
    """u_m y u_g desde las dos masas, con el caudal q(r) de cada radio."""
    rho_m = _rho_m(p)
    rho_g = max(float(p), 1e4) / (R * float(temperatura))
    um = np.zeros_like(q_r, dtype=float)
    ug = np.zeros_like(q_r, dtype=float)
    for i in range(len(q_r) - 1):
        n = _fg(p, float(xi[i]))
        ph = float(phi[i])
        if n <= 0.0 or ph <= 1e-6 or ph >= 0.999:
            um[i] = float(q_r[i]) / rho_m
            ug[i] = um[i]
            continue
        um[i] = float(q_r[i]) * (1.0 - n) / ((1.0 - ph) * rho_m)
        ug[i] = float(q_r[i]) * n / (ph * rho_g)
    return um, ug


def _arrastre_particula(phi, um, ug, nd, p, phicrit, temperatura):
    """Arrastre de una partícula líquida en gas. Positivo si el gas va más rápido."""
    rho_g = max(float(p), 1e4) / (R * float(temperatura))
    out = np.zeros(len(phi))
    ra, cd = 1e-3, 0.8
    libre = len(phi) - 1
    for i in range(libre):
        ph = float(np.clip(phi[i], 1e-4, 0.99))
        slip = float(ug[i] - um[i])
        rb = (ph / ((4.0 / 3.0) * math.pi * max(float(nd[i]), 1.0) * (1.0 - ph))) ** (1.0 / 3.0)
        if ph < float(phicrit[i]) + 0.05:
            tt = min(max((ph - float(phicrit[i])) / 0.05, 0.0), 1.0)
            coef = ((0.33 / (4.0 * max(rb, 1e-8))) ** (1.0 - tt)) * ((3.0 * cd / (8.0 * ra)) ** tt)
        else:
            coef = 3.0 * cd / (8.0 * ra)
        out[i] = coef * rho_g * abs(slip) * slip * ph * (1.0 - ph)
    return out


def _momento_fragmentado(r, dr, um_prev, ug_prev, phi, nd, phicrit, p_prev, h, q_obj, dpdz0, temperatura):
    """u_m sin L_r, u_g con L_r(μ_g), un dP/dz que deja el caudal integrado en Q.

    La pared de las dos fases queda en 0. El arrastre entra implícito.
    """
    n = len(r)
    m = n - 1
    um_prev = np.maximum(np.asarray(um_prev, dtype=float), 0.0)
    ug_prev = np.maximum(np.asarray(ug_prev, dtype=float), 0.0)
    phi = np.asarray(phi, dtype=float)
    nd = np.asarray(nd, dtype=float)
    phicrit = np.asarray(phicrit, dtype=float)

    def residual(x):
        um = np.zeros(n)
        ug = np.zeros(n)
        um[:-1] = x[:m]
        ug[:-1] = x[m:2 * m]
        dpdz = float(x[-1])
        p = float(p_prev) + dpdz * h
        res = np.zeros(2 * m + 1)
        if p < 0.5 * Patm:
            res[:] = 1e3
            return res
        rho_m = _rho_m(p)
        rho_g = max(p, 1e4) / (R * float(temperatura))
        fmg = _arrastre_particula(phi, um, ug, nd, p, phicrit, temperatura)
        for i in range(m):
            ph = float(phi[i])
            acel = (um[i] ** 2 - um_prev[i] ** 2) / (2.0 * h) + dpdz / rho_m + g - fmg[i] / (rho_m * (1.0 - ph))
            res[i] = acel / 50.0
        lo, di, up = _lr(r, dr, MU_GAS * phi)
        lr = np.zeros(m)
        lr[0] = di[0] * ug[0] + up[0] * ug[1]
        for i in range(1, m):
            lr[i] = lo[i] * ug[i - 1] + di[i] * ug[i] + up[i] * ug[i + 1]
        for i in range(m):
            ph = float(phi[i])
            acel = (
                (ug[i] ** 2 - ug_prev[i] ** 2) / (2.0 * h)
                + dpdz / rho_g + g + fmg[i] / (rho_g * ph) - lr[i] / (rho_g * ph)
            )
            res[m + i] = acel / 50.0
        flujo = rho_m * (1.0 - phi) * um + rho_g * phi * ug
        res[-1] = 100.0 * (_media_caudal(r, flujo) - q_obj) / max(q_obj, 1.0)
        return res

    bajo = np.concatenate([np.zeros(2 * m), [-5.0e6]])
    alto = np.concatenate([np.full(2 * m, 5.0e3), [1.0e5]])
    rho_ref = _rho_m(p_prev)
    semillas = [float(dpdz0), -rho_ref * g, -1.0e4, -2.0e4, -5.0e4, -1.0e5]
    sol = None
    mejor = None
    for semilla in semillas:
        x0 = np.concatenate([
            np.clip(um_prev[:-1], 0.0, 5.0e3),
            np.clip(ug_prev[:-1], 0.0, 5.0e3),
            [float(np.clip(semilla, -4.9e6, 9.0e4))],
        ])
        prueba = least_squares(
            residual, x0, bounds=(bajo, alto), ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=200,
        )
        if mejor is None or prueba.cost < mejor.cost:
            mejor = prueba
        if np.max(np.abs(residual(prueba.x))) <= 1e-4:
            sol = prueba
            break
    if sol is None and mejor is not None:
        sol = least_squares(
            residual, mejor.x, bounds=(bajo, alto), ftol=1e-14, xtol=1e-14, gtol=1e-14, max_nfev=200,
        )
    if sol is None or np.max(np.abs(residual(sol.x))) > 1e-4:
        return None
    um = np.zeros(n)
    ug = np.zeros(n)
    um[:-1] = sol.x[:m]
    ug[:-1] = sol.x[m:2 * m]
    dpdz = float(sol.x[-1])
    p = float(p_prev) + dpdz * h
    rho_m = _rho_m(p)
    rho_g = max(p, 1e4) / (R * float(temperatura))
    flujo = rho_m * (1.0 - phi) * um + rho_g * phi * ug
    return {"u": um, "ug": ug, "dpdz": dpdz, "P": p, "Q": _media_caudal(r, flujo), "q_r": flujo}



def _paso_frag(estado, h, h_min, co, xi0, temperatura, cortar_en_el_eje=True):
    """Un paso ya fragmentado: los dos momentos arman el perfil y Q fija dP/dz.

    φ sale de la masa del gas con esas velocidades: el flujo de gas nuevo
    es el anterior más lo que exsuelve n(P).
    """
    r = estado["r"]
    n = len(r)
    p_prev = float(estado["P"])
    phi_prev = np.asarray(estado["phi"], dtype=float)
    um_prev = np.asarray(estado["u"], dtype=float)
    ug_prev = np.asarray(estado["ug"], dtype=float)
    xi = np.asarray(estado["xi"], dtype=float)
    rho_g_prev = max(p_prev, 1e4) / (R * float(temperatura))
    gas_prev = rho_g_prev * phi_prev * ug_prev
    j_prev = _rho_m(p_prev) * (1.0 - phi_prev) * um_prev + gas_prev
    n_prev = np.array([_fg(p_prev, float(xi[i])) for i in range(n)])
    phi = phi_prev.copy()
    momento = None
    for _ in range(8):
        momento = _momento_fragmentado(
            r, estado["dr"], um_prev, ug_prev, phi, estado["N"], estado["phicrit_ca"],
            p_prev, h, float(estado["Q_obj"]), float(estado.get("dpdz", -5.0e3)), temperatura,
        )
        if momento is None:
            return _fallo(estado, "solver")
        p = float(momento["P"])
        rho_g = max(p, 1e4) / (R * float(temperatura))
        n_new = np.array([_fg(p, float(xi[i])) for i in range(n)])
        phi_new = (gas_prev + j_prev * (n_new - n_prev)) / (rho_g * np.maximum(momento["ug"], 1.0))
        phi_new[-1] = phi_new[-2]
        if (not np.isfinite(phi_new).all()) or np.any(phi_new[:-1] <= 0.02) or np.any(phi_new[:-1] >= 0.995):
            return _fallo(estado, "phi")
        if float(np.max(np.abs(phi_new - phi))) < 1e-3:
            phi = phi_new
            break
        phi = 0.5 * phi + 0.5 * phi_new
    p = float(momento["P"])
    dpdz = float(momento["dpdz"])
    if (not np.isfinite(p)) or p < 0.5 * Patm:
        return _fallo(estado, "presion")
    if abs(dpdz) * h > 0.2 * max(p_prev, Patm) and h > float(h_min) * 1.01:
        return _fallo(estado, "gradiente")
    um, ug = momento["u"], momento["ug"]
    if (not np.isfinite(um).all()) or (not np.isfinite(ug).all()):
        return _fallo(estado, "paso no finito")
    mensaje = ""
    cs = math.sqrt(R * float(temperatura))
    if cortar_en_el_eje and float(np.max(ug)) >= 0.98 * cs:
        mensaje = "sonico"
    elif p <= float(pfinal) * 1.05:
        mensaje = "presion atmosferica"
    return {
        "r": r,
        "dr": estado["dr"],
        "P": p,
        "dpdz": dpdz,
        "u": um,
        "ug": ug,
        "phi": phi,
        "N": np.array(estado["N"], copy=True),
        "xi": np.array(estado["xi"], copy=True),
        "sigma": np.zeros(len(r)),
        "sigmag": np.zeros(len(r)),
        "Q": float(momento["Q"]),
        "Q_obj": float(estado["Q_obj"]),
        "q_r": np.array(momento["q_r"], copy=True),
        "z": float(estado["z"] + h),
        "frag": True,
        "limphi1": estado["limphi1"],
        "limphi2": estado["limphi2"],
        "phicrit_ca": estado["phicrit_ca"],
        "ok": True,
        "mensaje": mensaje,
    }


def _frag_controlado(estado, h, h_min, h_max, co, xi0, temperatura, cortar_en_el_eje=True):
    """Un paso fragmentado contra dos de h/2. El sónico y la atmósfera se aceptan."""
    grande = _paso_frag(estado, h, h_min, co, xi0, temperatura, cortar_en_el_eje)
    if not grande["ok"]:
        hueco = max(-float(estado["z"]), 0.0)
        h2 = min(2.0 * h, float(h_max), hueco)
        if h2 > h * 1.2:
            doble = _paso_frag(estado, h2, h_min, co, xi0, temperatura, cortar_en_el_eje)
            if doble["ok"]:
                return doble, h2, ""
        return None, h, grande["mensaje"]
    if grande["mensaje"] in ("sonico", "presion atmosferica") or h <= float(h_min) * 1.5:
        return grande, float(np.clip(h, h_min, h_max)), ""
    medio = _paso_frag(estado, 0.5 * h, h_min, co, xi0, temperatura, cortar_en_el_eje)
    if not medio["ok"]:
        return grande, h, ""
    medio2 = _paso_frag(medio, 0.5 * h, h_min, co, xi0, temperatura, cortar_en_el_eje)
    if not medio2["ok"]:
        return grande, h, ""
    err = abs(float(grande["P"]) - float(medio2["P"])) / (1.0 + abs(float(medio2["P"])))
    if err > 0.05:
        return None, h, ""
    h_nuevo = float(np.clip(h * math.sqrt(TAU_H / max(err, 1e-8)), h_min, h_max))
    return medio2, h_nuevo, ""


def marchar(vinicial=25.0, radius=radius1, pressure=overP1, wt=h2o1, temperature=T1,
            content_crystal=xi1, n_r=12, h0=40.0, h_min=1.0, h_max=80.0, z_tope=0.0,
            corte="eje", n_lin=40):
    """Recta sin burbujas, metro de exsolución, y desde ahí la marcha en diferencias finitas."""
    r, dr = _malla(n_r, radius)
    co, pi = _configurar_fd(radius, pressure, wt, temperature, content_crystal)
    rm0 = _rho_m(pi)
    xi = float(content_crystal)
    mu_in = _mu_liquido(pi, xi, float(vinicial) / float(radius), temperature, co)
    limites = limites_regiones(r, vinicial, radius, pi, rm0, mu_in, co, xi, temperature)
    historia, dpdz_liq, z_sat, z_ad = _tramo_algebraico(
        r, dr, vinicial, radius, pi, rm0, co, xi, temperature, limites, n_lin
    )
    q0 = float(historia[0]["Q_obj"])
    cs = math.sqrt(R * float(temperature))
    vsound = 0.99 * cs
    cortar_en_el_eje = corte != "media"
    umbral_sonico = 0.92 * cs if cortar_en_el_eje else 0.95 * vsound
    estado = historia[-1]
    h = min(float(h0), float(h_max))
    mensaje = "boca"
    z_frag = None
    h_min_frag = min(float(h_min), 0.05)
    for _ in range(20000):
        if estado["z"] >= z_tope - 0.5:
            mensaje = "boca"
            break
        if estado["P"] <= pfinal * (1.3 if cortar_en_el_eje else 1.05):
            mensaje = "presion atmosferica"
            break
        ug_med = float(np.trapezoid(estado["ug"] * 2.0 * math.pi * r, r) / (math.pi * float(radius) ** 2))
        if ug_med > umbral_sonico and float(np.mean(estado["phi"])) > 0.5:
            mensaje = "sonico"
            break
        if estado.get("frag") or _seccion_fragmentada(estado):
            if z_frag is None:
                z_frag = float(estado["z"])
            h_uso = min(h, z_tope - estado["z"])
            nuevo, h_siguiente, corte_paso = _frag_controlado(
                estado, h_uso, h_min_frag, float(h_max), co, float(content_crystal), float(temperature),
                cortar_en_el_eje,
            )
            if corte_paso == "singular":
                mensaje = "sonico" if max(ug_med, float(np.max(estado["ug"]))) > 0.5 * cs else "ahogado"
                break
            if nuevo is None:
                if h_uso <= h_min_frag * 1.01:
                    mensaje = "ahogado"
                    break
                h = max(h_uso * 0.5, h_min_frag)
                continue
            estado = nuevo
            historia.append(estado)
            if estado["mensaje"] in ("sonico", "presion atmosferica"):
                mensaje = estado["mensaje"]
                break
            h = h_siguiente
            continue
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
        "dpdz_liquido": float(dpdz_liq),
        "z_sat": float(z_sat),
        "z_ad": float(z_ad),
        "z_frag": float("nan") if z_frag is None else float(z_frag),
        "Ca": limites["Ca"],
        "fragcrit": limites["fragcrit"],
        "phicrit_ca": limites["phicrit_ca"],
        "limphi1": limites["limphi1"],
        "limphi2": limites["limphi2"],
        "Nd": limites["Nd"],
    }


def _cierre_tiro_1d(sal, vsound):
    """Las dos bocas del tiro 1D, evaluadas con la velocidad media de la sección."""
    ug = float(sal["ug_media"][-1])
    pexit = float(sal["P"][-1])
    zexit = float(sal["z"][-1])
    phiexit = float(_promedio_area(sal["r"], sal["phi"][-1]))
    phicrit_sec = float(_promedio_area(sal["r"], sal["phicrit_ca"]))
    sonico = (
        zexit >= -5.0
        and pexit >= float(pfinal)
        and ug > 0.95 * vsound
        and ug <= 1.05 * vsound
    )
    atmosferica = (
        zexit >= -5.0
        and (float(pfinal) - 0.05e5) < pexit < (float(pfinal) + 0.05e5)
        and phiexit >= phicrit_sec
    )
    relajada = zexit >= -15.0 and abs(pexit - float(pfinal)) < 1.5e5 and ug > 0.5
    if sonico:
        criterio = "sonico"
    elif atmosferica:
        criterio = "atmosfera"
    elif relajada:
        criterio = "relajada"
    else:
        criterio = ""
    return ug, pexit, zexit, criterio


def tirar_como_1d(n_r=8, temperature=T1, **kwargs):
    """Tiro con la partida y el cierre del 1D.

    Parte en 10 m/s, con el intervalo [0.1, 50]. Acepta la marcha cuya
    velocidad media de gas cae en la ventana sónica del 1D cerca de la boca,
    o cuya presión cae en la ventana atmosférica. El corte del eje no entra
    en esa decisión: la velocidad que se compara es el promedio de la sección.
    """
    vsound = 0.99 * math.sqrt(R * float(temperature))
    vinicial = 10.0
    vmin, vmax = 0.1, 50.0
    count = 1
    pasos = []
    previo_v = previo_p = None
    sal = None
    criterio = ""
    while count < 60:
        v_shot = float(vinicial)
        sal = marchar(v_shot, n_r=n_r, temperature=temperature, corte="media", **kwargs)
        ug, pexit, zexit, criterio = _cierre_tiro_1d(sal, vsound)
        count += 1
        pasos.append({
            "paso": count - 1,
            "v": v_shot,
            "z": zexit,
            "P": pexit,
            "ug": ug,
            "um": float(sal["um_media"][-1]),
            "mensaje": sal["mensaje"],
            "criterio": criterio,
        })
        if criterio:
            break
        if count > 20 and previo_v is not None and previo_p is not None:
            f0 = previo_p - float(pfinal)
            f1 = pexit - float(pfinal)
            if abs(f1 - f0) > 1e2:
                v_sec = v_shot - f1 * (v_shot - previo_v) / (f1 - f0)
                if np.isfinite(v_sec):
                    v_sec = float(np.clip(v_sec, vmin + 1e-4, vmax - 1e-4))
                    if vmin < v_sec < vmax:
                        vinicial = v_sec
        if pexit < float(pfinal):
            vmax = v_shot
            if vinicial == v_shot:
                vinicial = vmin + (vmax - vmin) / 2.0
        elif zexit < -5.0 or ug > 1.02 * vsound:
            vmax = v_shot
            if vinicial == v_shot:
                vinicial = vmin + (vmax - vmin) / 2.0
        elif ug <= 0.98 * vsound:
            vmin = v_shot
            if vinicial == v_shot:
                vinicial = vmin + (vmax - vmin) / 2.0
        previo_v, previo_p = v_shot, pexit
        if vmax - vmin < 1e-4:
            break
    return {
        "solucion": sal,
        "pasos": pasos,
        "n_pasos": len(pasos),
        "convergido": bool(criterio),
        "criterio": criterio,
        "vsound": float(vsound),
        "vmin": float(vmin),
        "vmax": float(vmax),
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
