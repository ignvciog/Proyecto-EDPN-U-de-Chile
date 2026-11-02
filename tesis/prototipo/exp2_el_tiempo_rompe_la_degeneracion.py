"""
exp2_el_tiempo_rompe_la_degeneracion.py
=======================================
Experimento 2: EL TIEMPO ROMPE LA DEGENERACION.

Pregunta
--------
Un unico dato estacionario (la tasa de descarga masica MER) no permite separar
el radio del conducto R de la sobrepresion de camara Delta P. ?Que agrega la
serie de tiempo de la erupcion?

Estructura del argumento
------------------------
1. CURVA DE INDETERMINACION. Se calcula el conjunto de nivel

       {(R, Delta P) : MER_ss(R, Delta P) = MER_obs}

   que es una curva unidimensional: el dato estacionario determina solo UNA
   combinacion de los dos parametros. En el limite de Poiseuille esa curva es
   aproximadamente Delta P_efectivo ~ R^{-4}.

2. CURVA CARACTERISTICA. Todos esos conductos comparten UN PUNTO de la curva
   MER_ss(P_cam), pero no su PENDIENTE

       G = dMER/dP_cam ~ pi R^4 / (8 mu L)    [kg/s/Pa],

   que depende de R de forma independiente del dato estacionario.

3. DINAMICA. Al drenar una camara elastica de capacitancia C, la erupcion
   evoluciona segun  C dP_cam/dt = -MER(P_cam), con tiempo caracteristico

       tau = C / G.

   Conductos indistinguibles en estado estacionario producen curvas MER(t) muy
   distintas. La serie de tiempo identifica R; el estacionario, no.

Salida: figuras/exp2_degeneracion_RdP.png
        figuras/exp2_curvas_MER_tiempo.png
"""

from __future__ import annotations

import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from modelo_reducido import (
    Parametros, cilindro, resolver_estacionario, drenaje_camara,
)

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figuras")
os.makedirs(FIG, exist_ok=True)

V_CAMARA = 5.0e10      # 50 km3, en el rango estimado para sistemas silicicos
K_CAMARA = 1.0e10      # Pa, rigidez efectiva de la roca encajante


def sobrepresion_para_MER(R0: float, MER_obj: float, base: Parametros,
                          lo: float = 1.0e5, hi: float = 3.0e8):
    """Delta P que reproduce MER_obj para un cilindro de radio R0."""
    def residuo(dP):
        p = Parametros(H=base.H, z_f=base.z_f, R0=R0, dP=dP, T=base.T,
                       c0=base.c0, xi=base.xi)
        try:
            m = resolver_estacionario(cilindro(p), p, n_eval=2)["MER"]
        except RuntimeError:
            return -np.inf
        return np.log(m / MER_obj)

    f_lo, f_hi = residuo(lo), residuo(hi)
    if not (np.isfinite(f_lo) and np.isfinite(f_hi)) or f_lo * f_hi > 0:
        return None
    return float(brentq(residuo, lo, hi, xtol=1e2, rtol=1e-12))


# ---------------------------------------------------------------------------
def curva_de_indeterminacion(base: Parametros, MER_obj: float,
                             radios: np.ndarray):
    """(1) Conjunto de nivel MER_ss(R, Delta P) = MER_obs."""
    print("\n(1) Curva de indeterminacion  {(R, dP) : MER = MER_obs}")
    print(f"    {'R [m]':>8}{'dP [MPa]':>12}{'MER [kg/s]':>15}{'error':>11}")
    puntos = []
    for R0 in radios:
        dP = sobrepresion_para_MER(R0, MER_obj, base)
        if dP is None:
            print(f"    {R0:>8.1f}     -- sin solucion en el rango --")
            continue
        p = Parametros(H=base.H, z_f=base.z_f, R0=R0, dP=dP, T=base.T,
                       c0=base.c0, xi=base.xi)
        m = resolver_estacionario(cilindro(p), p, n_eval=2)["MER"]
        puntos.append((R0, dP, m))
        print(f"    {R0:>8.1f}{dP / 1e6:>12.2f}{m:>15.5e}"
              f"{100 * (m / MER_obj - 1):>+10.4f}%")
    return puntos


def dinamica(puntos, base: Parametros, t_final: float):
    """(2)-(3) Curva caracteristica y drenaje de camara para cada punto."""
    print("\n(2)-(3) Curva caracteristica y drenaje de la camara")
    print(f"    V_camara = {V_CAMARA:.1e} m3, K_camara = {K_CAMARA:.1e} Pa")
    print(f"    {'R [m]':>7}{'dP [MPa]':>10}{'G [kg/s/Pa]':>13}{'tau [h]':>10}"
          f"{'MER(6h)':>12}{'masa 6h':>12}{'dm vs cil':>11}")
    salidas = []
    ref_masa = None
    for R0, dP, _ in puntos:
        p = Parametros(H=base.H, z_f=base.z_f, R0=R0, dP=dP, T=base.T,
                       c0=base.c0, xi=base.xi)
        g = cilindro(p)
        d = drenaje_camara(g, p, V_CAMARA, K_CAMARA, t_final=t_final)
        d["R0"], d["dP"] = R0, dP
        salidas.append(d)
        if abs(R0 - base.R0) < 1e-9:
            ref_masa = d["masa"][-1]
    for d in salidas:
        rel = (100 * (d["masa"][-1] / ref_masa - 1)
               if ref_masa else float("nan"))
        print(f"    {d['R0']:>7.1f}{d['dP'] / 1e6:>10.2f}{d['G']:>13.4f}"
              f"{d['tau'] / 3600:>10.2f}{d['MER'][-1]:>12.3e}"
              f"{d['masa'][-1]:>12.3e}{rel:>+10.1f}%")
    return salidas


# ---------------------------------------------------------------------------
def figura_degeneracion(puntos, base, MER_obj):
    R = np.array([q[0] for q in puntos])
    dP = np.array([q[1] for q in puntos]) / 1e6
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    ax.plot(R, dP, "o-", color="crimson", lw=2, ms=7,
            label=r"$\{(R,\Delta P):\ \mathrm{MER}=\mathrm{MER}_{obs}\}$")
    ax.set_yscale("log")
    ax.set_xlabel("radio del conducto  $R$  [m]")
    ax.set_ylabel(r"sobrepresion de camara  $\Delta P$  [MPa]")
    ax.set_title("Un unico dato estacionario deja una curva de soluciones\n"
                 rf"$\mathrm{{MER}}_{{obs}} = {MER_obj:.3e}$ kg/s")
    ax.grid(alpha=0.3, which="both")
    ax.annotate("todos estos conductos\nson indistinguibles\ncon el dato "
                "estacionario",
                xy=(R[len(R) // 2], dP[len(dP) // 2]), xytext=(0.42, 0.62),
                textcoords="axes fraction", fontsize=9,
                arrowprops=dict(arrowstyle="->", color="0.35"))
    ax.legend(fontsize=9)
    fig.tight_layout()
    ruta = os.path.join(FIG, "exp2_degeneracion_RdP.png")
    fig.savefig(ruta, dpi=160, bbox_inches="tight")
    print(f"\n    figura -> {ruta}")


def figura_tiempo(salidas, MER_obj, t_final):
    fig, ax = plt.subplots(1, 3, figsize=(14.0, 5.0))
    colores = plt.cm.plasma(np.linspace(0.05, 0.8, len(salidas)))

    for d, c in zip(salidas, colores):
        et = rf"$R={d['R0']:.0f}$ m, $\Delta P={d['dP'] / 1e6:.0f}$ MPa"
        ax[0].plot(d["P_grid"] / 1e6, d["MER_grid"] / 1e7, color=c, lw=2,
                   label=et)
        ax[1].plot(d["t"] / 3600, d["MER"] / 1e7, color=c, lw=2, label=et)
        ax[2].plot(d["t"] / 3600, d["masa"] / 1e11, color=c, lw=2)

    ax[0].plot([], [], " ", label=" ")
    ax[0].axhline(MER_obj / 1e7, color="0.4", ls="--", lw=1)
    ax[0].set_xlabel(r"presion en la base  $P_{cam}$  [MPa]")
    ax[0].set_ylabel(r"$\mathrm{MER}$  [$10^7$ kg/s]")
    ax[0].set_title("Curva caracteristica del conducto\n"
                    "mismo punto, distinta pendiente $G=dMER/dP$")
    ax[0].legend(fontsize=8)

    ax[1].set_xlabel("tiempo  [h]")
    ax[1].set_ylabel(r"$\mathrm{MER}(t)$  [$10^7$ kg/s]")
    ax[1].set_title("Drenaje de la camara:\nlas series de tiempo SI se separan")
    ax[1].legend(fontsize=8)

    ax[2].set_xlabel("tiempo  [h]")
    ax[2].set_ylabel(r"masa acumulada  [$10^{11}$ kg]")
    ax[2].set_title("Masa emitida\n(observable geologico directo)")

    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle("El tiempo rompe la degeneracion: conductos identicos en "
                 "estado estacionario, distinguibles en transiente",
                 fontsize=12.5)
    fig.tight_layout()
    ruta = os.path.join(FIG, "exp2_curvas_MER_tiempo.png")
    fig.savefig(ruta, dpi=160, bbox_inches="tight")
    print(f"    figura -> {ruta}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    t0 = time.time()
    base = Parametros()
    MER_obj = resolver_estacionario(cilindro(base), base, n_eval=2)["MER"]

    print("=" * 74)
    print("EXPERIMENTO 2 - El tiempo rompe la degeneracion (R, Delta P)")
    print("=" * 74)
    print(f"    caso de referencia: R = {base.R0:g} m, "
          f"dP = {base.dP / 1e6:g} MPa  ->  MER = {MER_obj:.4e} kg/s")

    radios = np.array([10.0, 12.0, 14.0, 16.0])
    puntos = curva_de_indeterminacion(base, MER_obj, radios)

    T_FINAL = 6.0 * 3600.0     # duracion de la segunda fase de Calbuco 2015
    salidas = dinamica(puntos, base, T_FINAL)

    figura_degeneracion(puntos, base, MER_obj)
    figura_tiempo(salidas, MER_obj, T_FINAL)
    print(f"\n    tiempo total: {time.time() - t0:.0f} s")
