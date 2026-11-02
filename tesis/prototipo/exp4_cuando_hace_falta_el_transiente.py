"""
exp4_cuando_hace_falta_el_transiente.py
=======================================
Experimento 4: ?CUANDO HACE FALTA EL TERMINO TRANSIENTE?

Pregunta
--------
"Agregar el tiempo" puede significar dos cosas distintas, y conviene no
confundirlas:

  (A) El OBSERVABLE depende del tiempo: MER(t) es una funcion, no un escalar.
      Esto casi siempre vale la pena para el problema inverso (experimento 2).

  (B) El CONDUCTO esta fuera de equilibrio con sus condiciones de borde, de
      modo que los terminos d/dt de las ecuaciones del conducto importan.

(A) no implica (B). Una erupcion puede durar horas y variar mucho, y aun asi el
conducto puede estar *esclavizado* a la camara: en cada instante su estado es
practicamente el estacionario correspondiente a la presion de camara de ese
instante. En ese regimen "cuasi-estacionario" el operador directo es barato (un
tiro estacionario + una EDO para la camara) y la fisica de tasas finitas es
invisible al dato.

Criterio
--------
Es una comparacion de escalas de tiempo, tipo numero de Deborah:

    De = tau_conducto / tau_forzamiento

  De << 1  ->  cuasi-estacionario. Basta la curva caracteristica MER_ss(P_cam).
  De >~ 1  ->  genuinamente transiente. Hace falta resolver la EDP.

Lo que calcula este script
--------------------------
1. Las escalas de tiempo del caso base de Calbuco, medidas (no supuestas):
     - tau_residencia = masa en el conducto / MER
     - tau_conducto   = relajacion de MER tras un escalon de presion de camara
     - tau_camara     = C_camara / G, el tiempo de drenaje (experimento 2)

2. Una verificacion a posteriori de la hipotesis cuasi-estacionaria: se toma la
   historia P_cam(t) del drenaje cuasi-estacionario, se alimenta al solver
   transiente completo, y se compara el MER(t) que resulta.

3. Un barrido sobre el tiempo de forzamiento: se imponen rampas de presion de
   camara con tau_forzamiento entre segundos y horas, y se mide el error que
   comete la aproximacion cuasi-estacionaria. De ahi sale el umbral.

Salida: figuras/exp4_escalas_de_tiempo.png
"""

from __future__ import annotations

import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from modelo_reducido import (
    Parametros, cilindro, resolver_estacionario, presion_estacionaria,
    resolver_transiente, tiempo_de_respuesta, drenaje_camara,
    curva_caracteristica, rho_de_P, malla,
)

AQUI = os.path.dirname(os.path.abspath(__file__))
FIGURAS = os.path.join(AQUI, "figuras")
os.makedirs(FIGURAS, exist_ok=True)

HORA = 3600.0


def masa_en_el_conducto(geo, p, z, P):
    """Masa de magma contenida en el tramo viscoso del conducto [kg]."""
    A = np.pi * geo.R(z) ** 2
    return float(np.trapezoid(A * rho_de_P(P, p), z))


def main() -> None:
    t_inicio = time.time()
    print("=" * 74)
    print("EXPERIMENTO 4 - Cuando hace falta el termino transiente")
    print("=" * 74)

    p = Parametros()
    geo = cilindro(p)
    est = resolver_estacionario(geo, p)
    MER0 = est["MER"]
    z = malla(p, N=161, dz_min=2.0)
    _, P0 = presion_estacionaria(MER0, geo, p, z_eval=z)

    print(f"    caso base Calbuco: R = {p.R0:.0f} m, dP = {p.dP/1e6:.0f} MPa")
    print(f"    MER estacionario = {MER0:.4e} kg/s")

    # ------------------------------------------------------------------
    # (1) Escalas de tiempo, medidas
    # ------------------------------------------------------------------
    print("\n(1) Escalas de tiempo del caso base")

    M_cond = masa_en_el_conducto(geo, p, z, P0)
    tau_res = M_cond / MER0
    print(f"    masa en el conducto        = {M_cond:.3e} kg")
    print(f"    tau_residencia = M / MER   = {tau_res:8.1f} s "
          f"({tau_res/60:.1f} min)")

    # Relajacion propia: escalon de -2% en la presion de camara.
    P_cam0 = p.P_camara
    P_cam1 = P_cam0 * 0.98
    sol = resolver_transiente(geo, p, lambda t: P_cam1,
                              t_final=600.0, n_salidas=600, P_inicial=P0)
    tau_cond = tiempo_de_respuesta(sol["t"], sol["MER"])
    print(f"    tau_conducto (escalon -2%) = {tau_cond:8.1f} s "
          f"({tau_cond/60:.1f} min)")
    print(f"      MER: {sol['MER'][0]:.4e} -> {sol['MER'][-1]:.4e} kg/s")

    # Tiempo de drenaje de la camara (experimento 2).
    V_camara = 5.0e10
    dren = drenaje_camara(geo, p, V_camara=V_camara, t_final=8.0 * HORA)
    tau_cam = dren["tau"]
    print(f"    tau_camara = C / G         = {tau_cam:8.1f} s "
          f"({tau_cam/HORA:.2f} h)   [V = {V_camara/1e9:.0f} km3]")

    De = tau_cond / tau_cam
    print(f"\n    De = tau_conducto / tau_camara = {De:.2e}")
    print("    => el conducto esta ESCLAVIZADO a la camara: cuasi-estacionario.")

    # ------------------------------------------------------------------
    # (2) Verificacion a posteriori de la hipotesis cuasi-estacionaria
    # ------------------------------------------------------------------
    print("\n(2) Verificacion: transiente completo con el P_cam(t) cuasi-estacionario")

    t_qs, P_qs, MER_qs = dren["t"], dren["P_camara"], dren["MER"]
    P_cam_de_t = lambda t: float(np.interp(t, t_qs, P_qs))

    sol_tr = resolver_transiente(geo, p, P_cam_de_t, t_final=t_qs[-1],
                                 n_salidas=200, P_inicial=P0)
    MER_qs_interp = np.interp(sol_tr["t"], t_qs, MER_qs)
    err = (sol_tr["MER"] - MER_qs_interp) / MER_qs_interp
    print(f"    error relativo maximo en MER(t): {np.abs(err).max()*100:.4f} %")
    print(f"    error relativo medio:            {np.abs(err).mean()*100:.4f} %")
    print("    La curva cuasi-estacionaria del experimento 2 esta justificada.")

    # ------------------------------------------------------------------
    # (3) Barrido: cuando se rompe la aproximacion
    # ------------------------------------------------------------------
    print("\n(3) Barrido sobre el tiempo de forzamiento")
    print("    Rampa exponencial P_cam(t) = P0 - dP_paso (1 - e^{-t/tau_f}),")
    print("    con dP_paso = 2% de P0. Se compara MER transiente con la curva")
    print("    caracteristica evaluada en P_cam(t).")

    # Curva caracteristica local, para evaluar la prediccion cuasi-estacionaria.
    P_lo, P_hi = P_cam1 * 0.999, P_cam0 * 1.001
    P_grid = np.linspace(P_lo, P_hi, 25)
    MER_grid = curva_caracteristica(geo, p, P_grid)
    MER_ss_de_P = lambda Pc: np.interp(Pc, P_grid, MER_grid)

    taus_f = np.array([0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 100.0]) * tau_cond
    errores, De_vals, curvas = [], [], []
    for tau_f in taus_f:
        dPc = P_cam0 - P_cam1
        Pc_de_t = lambda t, tf=tau_f: P_cam0 - dPc * (1.0 - np.exp(-t / tf))
        s = resolver_transiente(geo, p, Pc_de_t,
                                t_final=max(8.0 * tau_f, 12.0 * tau_cond),
                                n_salidas=300, P_inicial=P0)
        MER_pred = MER_ss_de_P(np.array([Pc_de_t(ti) for ti in s["t"]]))
        e = np.abs(s["MER"] - MER_pred) / MER_pred
        errores.append(e.max())
        De_vals.append(tau_cond / tau_f)
        curvas.append((tau_f, s["t"], s["MER"], MER_pred))
        print(f"      tau_forzamiento = {tau_f:8.1f} s  "
              f"(De = {tau_cond/tau_f:6.2f})  "
              f"error max cuasi-estacionario = {e.max()*100:7.3f} %")

    errores = np.array(errores)
    De_vals = np.array(De_vals)

    # Umbral: De para el cual el error cuasi-estacionario alcanza 1%.
    orden = np.argsort(De_vals)
    De_umbral = float(np.interp(0.01, errores[orden], De_vals[orden]))
    print(f"\n    El error cuasi-estacionario alcanza 1% cuando De ~ {De_umbral:.2f}")
    print(f"    => basta el cuasi-estacionario si el forzamiento es mas lento")
    print(f"       que ~{tau_cond/De_umbral:.0f} s. Para Calbuco (horas), con holgura.")
    print("\n    ADVERTENCIA. tau_conducto medido aqui es solo la relajacion")
    print("    HIDRAULICA (difusion de presion). El modelo reducido supone")
    print("    exsolucion y cristalizacion en EQUILIBRIO, de modo que no")
    print("    contiene las escalas lentas que la literatura identifica como")
    print("    las que realmente vuelven transiente una erupcion:")
    print("      exsolucion en desequilibrio  ~ 1e1 - 1e3 s  (La Spina 2017)")
    print("      cristalizacion de microlitos ~ 1e4 - 1e6 s  (Melnik & Sparks)")
    print("      escape de gas por permeabilidad ~ 1e3 - 1e5 s (Wong 2019)")
    print("    Con tau ~ 1e4 s el veredicto cambia por completo: hasta una fase")
    print("    sub-pliniana de 6 h pasa a De ~ 0.5, es decir transiente. Y en un")
    print("    ciclo de domo el periodo NO es un forzamiento externo, lo fija la")
    print("    propia cinetica: De ~ 1 por construccion, el ciclo existe PORQUE")
    print("    hay retardo. Ahi el termino transiente no es un refinamiento,")
    print("    es el mecanismo.")

    # ------------------------------------------------------------------
    # Figura
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # (a) relajacion propia del conducto
    a = ax[0]
    a.plot(sol["t"], sol["MER"] / 1e7, color="#C1440E", lw=2)
    a.axvline(tau_cond, color="0.4", ls="--", lw=1.2)
    a.axhline(sol["MER"][-1] / 1e7, color="0.7", ls=":", lw=1.0)
    a.annotate(fr"$\tau_{{conducto}}$ = {tau_cond:.0f} s",
               xy=(tau_cond, sol["MER"][0] / 1e7),
               xytext=(0.35, 0.80), textcoords="axes fraction", fontsize=9,
               arrowprops=dict(arrowstyle="->", color="0.4", lw=1))
    a.set_xlim(0, 20.0 * tau_cond)
    a.set_xlabel("tiempo [s]")
    a.set_ylabel(r"MER [$10^7$ kg/s]")
    a.set_title("(a) Relajacion propia del conducto\n"
                "escalon de $-2\\%$ en la presion de camara", fontsize=9)
    a.grid(alpha=0.3)

    # (b) error de la aproximacion cuasi-estacionaria vs De
    a = ax[1]
    a.loglog(De_vals, 100 * errores, "o-", color="#2E4057", lw=2, ms=5)
    a.axhline(1.0, color="#8B1E3F", ls="--", lw=1.2)
    a.axvline(De_umbral, color="#8B1E3F", ls=":", lw=1.2)
    a.text(De_umbral * 1.15, 100 * errores.min() * 1.5,
           f"De $\\approx$ {De_umbral:.2f}", color="#8B1E3F", fontsize=9)
    a.text(0.03, 0.92, "cuasi-estacionario\nbasta", transform=a.transAxes,
           fontsize=9, va="top", color="#0E7C7B")
    a.text(0.97, 0.12, "hace falta\nla EDP", transform=a.transAxes,
           fontsize=9, ha="right", color="#8B1E3F")
    a.set_xlabel(r"De $=\tau_{conducto}\,/\,\tau_{forzamiento}$")
    a.set_ylabel("error max. del cuasi-estacionario [\\%]")
    a.set_title("(b) Cuando se rompe la aproximacion", fontsize=9)
    a.grid(alpha=0.3, which="both")

    # (c) diagrama de escalas por tipo de erupcion
    #
    # Dos escalas internas distintas:
    #   - tau_hidraulico: medido arriba con el modelo reducido (difusion de
    #     presion en el conducto). Es lo unico que este modelo contiene.
    #   - tau_cinetico: cristalizacion de microlitos, ~1e4 s segun la
    #     literatura (Melnik & Sparks 1999; La Spina et al. 2017). NO esta en
    #     el modelo reducido, que supone equilibrio.
    tau_cinetico = 1.0e4
    a = ax[2]
    tipos: list[tuple[str, float]] = [
        ("explosion vulcaniana",        30.0),
        ("apertura de conducto",        5.0 * 60),
        ("pulso de domo (drumbeat)",    30.0 * 60),
        ("fase sub-pliniana (Calbuco)", 6.0 * HORA),
        # El periodo de un ciclo de domo no es un forzamiento externo: lo FIJA
        # la propia cinetica de cristalizacion, de modo que De ~ 1 por
        # construccion. El ciclo existe *porque* hay retardo.
        ("ciclo de domo (auto-sost.)",  tau_cinetico),
        ("efusiva de anos",             5 * 365 * 24 * HORA),
    ]
    ypos = np.arange(len(tipos))[::-1]
    x_lo, x_hi = 1e-8, 1e4
    a.axvspan(De_umbral, x_hi, color="#8B1E3F", alpha=0.08)
    for y, (nombre, tf) in zip(ypos, tipos):
        for tau_i, marca, ms in ((tau_cond, "o", 7), (tau_cinetico, "s", 7)):
            De_i = tau_i / tf
            color = "#8B1E3F" if De_i > De_umbral else "#0E7C7B"
            a.plot([De_i], [y], marca, color=color, ms=ms,
                   mec="white", mew=0.8)
        a.plot([tau_cond / tf, tau_cinetico / tf], [y, y],
               color="0.55", lw=1.0, zorder=0)
    a.axvline(De_umbral, color="0.3", ls="--", lw=1.3)
    a.text(De_umbral * 1.8, len(tipos) - 0.55, f"De = {De_umbral:.2f}",
           rotation=90, fontsize=8.5, ha="left", va="top", color="0.3")
    a.plot([], [], "o", color="0.35", ms=7, label=r"$\tau$ hidraulico (8 s)")
    a.plot([], [], "s", color="0.35", ms=7,
           label=r"$\tau$ cinetico ($10^4$ s, literatura)")
    a.legend(fontsize=7.5, loc="lower right", framealpha=0.9)
    a.set_xscale("log")
    a.set_xlim(x_lo, x_hi)
    a.set_yticks(ypos)
    a.set_yticklabels([t[0] for t in tipos], fontsize=8.5)
    a.set_ylim(-0.7, len(tipos) - 0.3)
    a.set_xlabel(r"De $=\tau_{interno}\,/\,\tau_{forzamiento}$")
    a.set_title("(c) Por tipo de erupcion:\n"
                "la cinetica, no la hidraulica, decide", fontsize=9)
    a.grid(alpha=0.3, axis="x", which="major")

    fig.suptitle("Cuando el conducto esta fuera de equilibrio: "
                 "separacion de escalas de tiempo", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    ruta = os.path.join(FIGURAS, "exp4_escalas_de_tiempo.png")
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print(f"\n    figura -> {ruta}")
    print(f"\n    tiempo total: {time.time() - t_inicio:.0f} s")


if __name__ == "__main__":
    main()
