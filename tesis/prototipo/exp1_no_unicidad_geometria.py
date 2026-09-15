"""
exp1_no_unicidad_geometria.py
=============================
Experimento 1: NO-UNICIDAD DE LA SOLUCION RESPECTO DE LA GEOMETRIA.

Pregunta
--------
Si se mide la tasa de descarga masica (MER) de una erupcion, ?que restringe eso
sobre la geometria R(z) del conducto?

Resultados que produce
----------------------
(a) Teorema de degeneracion, verificado numericamente.
    En el limite dominado por friccion el problema estacionario da

        \\int_{P_frag}^{P_cam} rho(P)/mu(P) dP = (8 MER / pi) J_4[R],
        J_4[R] = \\int_H^{z_f} R(z)^{-4} dz .

    => MER depende de R(z) SOLO a traves del escalar J_4[R]. La preimagen de un
       dato es una hipersuperficie de codimension 1 en un espacio de funciones.

(b) Familia no-unica del modelo completo (con flotabilidad).
    Para cinco formas de conducto cualitativamente distintas se ajusta un unico
    factor de escala de modo que TODAS produzcan exactamente el mismo MER. Se
    reportan sus volumenes, que difieren en cientos de por ciento.

(c) Nucleo de sensibilidad.
    Se calcula la densidad de resistencia w(z) = mu(z)/R(z)^4 normalizada, que
    muestra donde "mira" el dato: practicamente todo el peso esta en el
    kilometro superior. El conducto profundo es invisible.

Salida: figuras/exp1_no_unicidad_geometria.png y exp1_kernel_resistencia.png
"""

from __future__ import annotations

import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from modelo_reducido import (
    Parametros, cilindro, ensanchamiento, constriccion, dos_tramos, campana,
    resolver_estacionario, escalar_a_MER, presion_estacionaria,
    rho_de_P, viscosidad,
)

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figuras")
os.makedirs(FIG, exist_ok=True)


# ---------------------------------------------------------------------------
def verificar_teorema_J4(p: Parametros) -> None:
    """(a) En el limite dominado por friccion, MER * J_4 es invariante."""
    print("\n(a) Verificacion del teorema de degeneracion J_4")
    print("    (flotabilidad apagada => limite dominado por friccion)")

    p_f = Parametros(H=p.H, z_f=p.z_f, R0=p.R0, dP=p.dP, T=p.T, c0=p.c0,
                     xi=p.xi)
    p_f.incluir_flotabilidad = False

    base = cilindro(p_f)
    ref = resolver_estacionario(base, p_f, n_eval=2)["MER"]
    J4_ref = base.J4()

    # integral analitica del lado izquierdo
    P = np.linspace(p_f.P_frag, p_f.P_camara, 20001)
    integral = float(np.trapezoid(rho_de_P(P, p_f) / viscosidad(P, p_f), P))
    predicho = np.pi * integral / (8.0 * J4_ref)
    print(f"    MER numerico  = {ref:.6e} kg/s")
    print(f"    MER analitico = {predicho:.6e} kg/s   "
          f"(error {100 * (predicho / ref - 1):+.3f} %)")

    print(f"    {'forma':<34}{'J_4':>12}{'MER':>14}{'MER*J_4':>14}{'desv':>9}")
    formas = [cilindro(p_f), ensanchamiento(p_f, 2.0), constriccion(p_f, 0.30),
              dos_tramos(p_f, 1.6), campana(p_f, 1.5, 0.2, 0.25)]
    for g in formas:
        g.escalar_a_J4(J4_ref)
        m = resolver_estacionario(g, p_f, n_eval=2, MER_guess=ref)["MER"]
        print(f"    {g.nombre:<34}{g.J4():>12.5e}{m:>14.6e}"
              f"{m * g.J4():>14.6e}{100 * (m / ref - 1):>+8.4f}%")


# ---------------------------------------------------------------------------
def familia_no_unica(p: Parametros):
    """(b) Geometrias muy distintas con MER estacionario identico."""
    print("\n(b) Familia no-unica del modelo completo (con flotabilidad)")
    base = cilindro(p)
    ref = resolver_estacionario(base, p, n_eval=2)
    MER_obj = ref["MER"]
    V_base = base.volumen()
    print(f"    MER de referencia = {MER_obj:.5e} kg/s "
          f"(cilindro R = {p.R0:g} m, V = {V_base:.3e} m3)")

    formas = [
        cilindro(p),
        campana(p, 2.0, 0.00, 0.35),
        campana(p, 3.0, 0.00, 0.50),
        ensanchamiento(p, 2.0),
        dos_tramos(p, 2.0, 0.50, 0.05),
        constriccion(p, 0.35, 0.20, 0.10),
    ]
    etiquetas = [
        "cilindro",
        "ensanche basal x3",
        "ensanche basal x4",
        "ensanchamiento superior x2",
        "dos tramos x2",
        "constriccion profunda -35%",
    ]

    resultados = []
    print(f"    {'geometria':<30}{'R(H)':>7}{'R(z_f)':>8}{'V [m3]':>12}"
          f"{'dV':>9}{'J_4':>11}{'err MER':>11}")
    for g, et in zip(formas, etiquetas):
        escalar_a_MER(g, p, MER_obj)
        r = resolver_estacionario(g, p, n_eval=400, MER_guess=MER_obj)
        V = g.volumen()
        resultados.append(dict(geo=g, etiqueta=et, res=r, V=V))
        print(f"    {et:<30}{g.R(p.H):>7.1f}{g.R(p.z_f):>8.1f}{V:>12.3e}"
              f"{100 * (V / V_base - 1):>+8.1f}%{g.J4():>11.4e}"
              f"{100 * (r['MER'] / MER_obj - 1):>+10.5f}%")
    return resultados, MER_obj, V_base


# ---------------------------------------------------------------------------
def kernel_resistencia(p: Parametros):
    """(c) Donde se concentra la informacion que aporta el MER."""
    print("\n(c) Nucleo de resistencia w(z) ~ mu(z)/R(z)^4")
    g = cilindro(p)
    r = resolver_estacionario(g, p, n_eval=3000)
    z, w = r["z"], r["mu"] / r["R"] ** 4
    w = w / np.trapezoid(w, z)
    acum = np.concatenate([[0.0],
                           np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(z))])
    for frac in (0.50, 0.90, 0.99):
        i = min(int(np.searchsorted(acum, frac)), z.size - 1)
        print(f"    {frac:.0%} de la resistencia total esta sobre "
              f"z = {z[i]:8.0f} m  (ultimos {abs(z[i] - p.z_f):.0f} m)")
    return z, w, acum


# ---------------------------------------------------------------------------
def figura_geometrias(resultados, MER_obj, V_base, p):
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 5.4))
    colores = plt.cm.viridis(np.linspace(0.0, 0.88, len(resultados)))

    for (d, c) in zip(resultados, colores):
        g, r = d["geo"], d["res"]
        z = np.linspace(p.H, p.z_f, 600)
        ax[0].plot(g.R(z), z / 1e3, color=c, lw=2,
                   label=f"{d['etiqueta']}  (V{100 * (d['V'] / V_base - 1):+.0f}%)")
        ax[1].plot(r["P"] / 1e6, r["z"] / 1e3, color=c, lw=2)
        ax[2].plot(r["u"], r["z"] / 1e3, color=c, lw=2)

    ax[0].set_xlabel("radio del conducto $R(z)$  [m]")
    ax[0].set_ylabel("profundidad $z$  [km]")
    ax[0].set_title("Seis geometrias distintas...")
    ax[0].legend(fontsize=7.5, loc="lower right", framealpha=0.95)

    ax[1].set_xlabel("presion $P$  [MPa]")
    ax[1].set_title("...con perfiles de presion distintos...")

    ax[2].set_xscale("log")
    ax[2].set_xlabel("velocidad de la mezcla $u$  [m/s]")
    ax[2].set_title(f"...y el MISMO caudal masico\n"
                    f"MER = {MER_obj:.3e} kg/s (identico a $10^{{-5}}$ %)")

    for a in ax:
        a.grid(alpha=0.3)
        a.set_ylim(p.H / 1e3, p.z_f / 1e3)
    fig.suptitle("No-unicidad respecto de la geometria: la tasa de descarga "
                 "estacionaria no determina $R(z)$", fontsize=12.5)
    fig.tight_layout()
    ruta = os.path.join(FIG, "exp1_no_unicidad_geometria.png")
    fig.savefig(ruta, dpi=160, bbox_inches="tight")
    print(f"\n    figura -> {ruta}")


def figura_kernel(z, w, acum, p):
    fig, ax = plt.subplots(1, 2, figsize=(10.0, 5.0))
    ax[0].plot(w * 1e3, z / 1e3, color="crimson", lw=2)
    ax[0].fill_betweenx(z / 1e3, 0, w * 1e3, color="crimson", alpha=0.22)
    ax[0].set_xlabel(r"densidad de resistencia  $w(z)\propto \mu/R^4$  [1/km]")
    ax[0].set_ylabel("profundidad $z$  [km]")
    ax[0].set_title("Donde se genera la caida de presion")

    ax[1].plot(100 * acum, z / 1e3, color="navy", lw=2)
    for frac, col in ((0.5, "0.5"), (0.9, "0.3")):
        ax[1].axvline(100 * frac, ls="--", lw=1, color=col)
        i = min(int(np.searchsorted(acum, frac)), z.size - 1)
        ax[1].annotate(f"{frac:.0%} sobre z = {z[i] / 1e3:.2f} km",
                       xy=(100 * frac, z[i] / 1e3), xytext=(6, -14),
                       textcoords="offset points", fontsize=8.5, color=col)
    ax[1].set_xlabel("resistencia acumulada desde la base  [%]")
    ax[1].set_title("El dato solo 've' el conducto somero")

    for a in ax:
        a.grid(alpha=0.3)
        a.set_ylim(p.H / 1e3, p.z_f / 1e3)
    fig.suptitle("Nucleo de sensibilidad del MER a la geometria del conducto",
                 fontsize=12.5)
    fig.tight_layout()
    ruta = os.path.join(FIG, "exp1_kernel_resistencia.png")
    fig.savefig(ruta, dpi=160, bbox_inches="tight")
    print(f"    figura -> {ruta}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    t0 = time.time()
    p = Parametros()
    print("=" * 74)
    print("EXPERIMENTO 1 - No-unicidad respecto de la geometria del conducto")
    print("=" * 74)
    print(f"    H = {p.H:g} m, z_f = {p.z_f:g} m, T = {p.T - 273.15:.0f} C, "
          f"c_0 = {100 * p.c0:.1f} wt%, xi = {p.xi:g}")
    print(f"    P_camara = {p.P_camara / 1e6:.1f} MPa, "
          f"P_frag = {p.P_frag / 1e6:.2f} MPa (phi = {p.phi_frag:g})")

    verificar_teorema_J4(p)
    resultados, MER_obj, V_base = familia_no_unica(p)
    z, w, acum = kernel_resistencia(p)

    figura_geometrias(resultados, MER_obj, V_base, p)
    figura_kernel(z, w, acum, p)
    print(f"\n    tiempo total: {time.time() - t0:.0f} s")
