"""
exp3_identificabilidad.py
=========================
Experimento 3: IDENTIFICABILIDAD CUANTITATIVA.

Formaliza lo que los experimentos 1 y 2 muestran cualitativamente. Se define el
operador directo restringido a dos parametros,

    theta = (log R, log Delta P)  ->  d(theta) = (log MER, log G),

donde MER es la tasa de descarga estacionaria y

    G = dMER/dP_cam

es la pendiente de la curva caracteristica del conducto, que controla el tiempo
de decaimiento tau = C/G de la erupcion (experimento 2). El primer observable
resume el dato estacionario; el segundo resume la serie de tiempo.

Se calcula:

1. Una tabla del operador directo sobre una malla (R, Delta P), de la cual se
   obtienen MER por tiro y G por diferenciacion en Delta P.

2. La matriz de sensibilidad J = d(log d)/d(log theta) en el punto nominal, su
   descomposicion en valores singulares, el numero de condicion y la direccion
   del (casi) nucleo.

3. La matriz de informacion de Fisher F = J^T Sigma^{-1} J con un modelo de
   ruido realista (el MER de una erupcion se conoce salvo un factor ~2). Es un
   diagnostico LOCAL: aqui resulta insuficiente porque el operador directo es
   fuertemente no lineal en Delta P.

4. Los mapas de desajuste chi^2(theta) y los PERFILES DE VEROSIMILITUD

       chi^2_perfil(theta_i) = min_{theta_j, j != i} chi^2(theta),

   que son el diagnostico global correcto de identificabilidad practica
   (Raue et al., 2009). Un perfil que no sube por encima del umbral indica un
   parametro no identificable con los datos disponibles.

Salida: figuras/exp3_identificabilidad.png
        cache/tabla_directa.npz  (la tabla es cara; se reutiliza)
"""

from __future__ import annotations

import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from modelo_reducido import Parametros, cilindro, resolver_estacionario

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figuras")
os.makedirs(FIG, exist_ok=True)

# Modelo de ruido (desviaciones estandar en escala logaritmica natural)
#   MER de una erupcion explosiva: conocido tipicamente salvo un factor ~2
#   G: se estima ajustando el decaimiento de MER(t), algo mejor determinado
SIGMA_MER = np.log(2.0) / 2.0     # ~0.35
SIGMA_G = 0.20


# ---------------------------------------------------------------------------
def tabla_directa(base: Parametros, radios: np.ndarray, dPs: np.ndarray):
    """
    MER(R, Delta P) por metodo de tiro, con arranque en caliente a lo largo de
    Delta P. Devuelve la tabla y la pendiente G = dMER/dP_cam obtenida por
    diferencias centradas (P_cam = rho_crust g |H| + Delta P, de modo que
    derivar en Delta P es derivar en P_cam).
    """
    MER = np.full((radios.size, dPs.size), np.nan)
    for i, R0 in enumerate(radios):
        guess = None
        for j, dP in enumerate(dPs):
            p = Parametros(H=base.H, z_f=base.z_f, R0=R0, dP=dP, T=base.T,
                           c0=base.c0, xi=base.xi)
            try:
                MER[i, j] = resolver_estacionario(cilindro(p), p, n_eval=2,
                                                  MER_guess=guess)["MER"]
                guess = MER[i, j]
            except RuntimeError:
                guess = None
        print(f"    R = {R0:5.1f} m  ->  "
              f"{np.count_nonzero(np.isfinite(MER[i]))}/{dPs.size} soluciones")

    G = np.full_like(MER, np.nan)
    G[:, 1:-1] = (MER[:, 2:] - MER[:, :-2]) / (dPs[2:] - dPs[:-2])
    return MER, G


def interpolar(tabla, radios, dPs, R0, dP):
    """Interpolacion bilineal en (log R, log dP)."""
    x = np.log(radios)
    y = np.log(dPs)
    xi = np.clip(np.log(R0), x[0], x[-1])
    yi = np.clip(np.log(dP), y[0], y[-1])
    i = np.clip(np.searchsorted(x, xi) - 1, 0, x.size - 2)
    j = np.clip(np.searchsorted(y, yi) - 1, 0, y.size - 2)
    tx = (xi - x[i]) / (x[i + 1] - x[i])
    ty = (yi - y[j]) / (y[j + 1] - y[j])
    return ((1 - tx) * (1 - ty) * tabla[i, j] + tx * (1 - ty) * tabla[i + 1, j]
            + (1 - tx) * ty * tabla[i, j + 1] + tx * ty * tabla[i + 1, j + 1])


# ---------------------------------------------------------------------------
def jacobiano(base: Parametros, h: float = 0.02):
    """
    J = d(log MER, log G) / d(log R, log Delta P) por diferencias centradas.
    """
    def observables(R0, dP):
        p = Parametros(H=base.H, z_f=base.z_f, R0=R0, dP=dP, T=base.T,
                       c0=base.c0, xi=base.xi)
        g = cilindro(p)
        m = resolver_estacionario(g, p, n_eval=2)["MER"]
        # G por diferencias centradas en P_cam
        eps = 0.02 * dP
        mp = resolver_estacionario(
            cilindro(Parametros(H=base.H, z_f=base.z_f, R0=R0, dP=dP + eps,
                                T=base.T, c0=base.c0, xi=base.xi)),
            Parametros(H=base.H, z_f=base.z_f, R0=R0, dP=dP + eps, T=base.T,
                       c0=base.c0, xi=base.xi), n_eval=2,
            MER_guess=m)["MER"]
        mm = resolver_estacionario(
            cilindro(Parametros(H=base.H, z_f=base.z_f, R0=R0, dP=dP - eps,
                                T=base.T, c0=base.c0, xi=base.xi)),
            Parametros(H=base.H, z_f=base.z_f, R0=R0, dP=dP - eps, T=base.T,
                       c0=base.c0, xi=base.xi), n_eval=2,
            MER_guess=m)["MER"]
        Gv = (mp - mm) / (2.0 * eps)
        return np.array([np.log(m), np.log(Gv)])

    R0, dP = base.R0, base.dP
    dR = np.exp(h)
    d_p = observables(R0 * dR, dP)
    d_m = observables(R0 / dR, dP)
    col_R = (d_p - d_m) / (2.0 * h)
    d_p = observables(R0, dP * dR)
    d_m = observables(R0, dP / dR)
    col_P = (d_p - d_m) / (2.0 * h)
    return np.column_stack([col_R, col_P]), observables(R0, dP)


def analizar(J: np.ndarray) -> None:
    """SVD, numero de condicion, nucleo e informacion de Fisher."""
    nombres = ("log R", "log dP")
    print("\n(2) Matriz de sensibilidad  J = d(log d)/d(log theta)")
    print(f"    {'':>12}{nombres[0]:>12}{nombres[1]:>12}")
    for fila, et in zip(J, ("log MER", "log G")):
        print(f"    {et:>12}{fila[0]:>12.4f}{fila[1]:>12.4f}")

    print(f"\n    El exponente d log MER / d log R = {J[0, 0]:.3f} reproduce la "
          f"ley de Poiseuille MER ~ R^4.")
    print(f"    En cambio d log MER / d log dP = {J[0, 1]:.3f} es pequeno: la "
          f"sobrepresion aporta")
    print(f"    solo una fraccion del empuje total (el resto lo aporta la "
          f"flotabilidad).")

    print("\n    Caso A: SOLO dato estacionario (primera fila)")
    Ja = J[:1, :]
    _, sa, vta = np.linalg.svd(Ja)
    print(f"      valores singulares: {sa}")
    print(f"      rango = {np.linalg.matrix_rank(Ja)} de 2  =>  nucleo de "
          f"dimension 1")
    nulo = vta[-1]
    if nulo[0] > 0:
        nulo = -nulo
    exponente = nulo[1] / nulo[0]
    print(f"      direccion no identificada: "
          f"(d log R, d log dP) = ({nulo[0]:+.4f}, {nulo[1]:+.4f})")
    print(f"      ley de compensacion local:  dP ~ R^({exponente:+.1f})")

    print("\n    Caso B: estacionario + temporal (ambas filas)")
    _, sb, _ = np.linalg.svd(J)
    print(f"      valores singulares: {sb}")
    print(f"      numero de condicion = {sb[0] / sb[-1]:.1f}")
    print("      El rango es completo, pero las dos filas son casi paralelas:")
    print("      el segundo valor singular es pequeno y la inversion sigue mal")
    print("      condicionada CERCA de este punto (ver perfiles globales).")

    Sigma_inv = np.diag([SIGMA_MER ** -2, SIGMA_G ** -2])
    C = np.linalg.inv(J.T @ Sigma_inv @ J)
    print(f"\n(3) Informacion de Fisher local (sigma_MER = {SIGMA_MER:.2f}, "
          f"sigma_G = {SIGMA_G:.2f} en log)")
    print(f"      sigma(log R)  = {np.sqrt(C[0, 0]):.3f}  =>  factor "
          f"{np.exp(np.sqrt(C[0, 0])):.2f} en R")
    print(f"      sigma(log dP) = {np.sqrt(C[1, 1]):.3f}  =>  factor "
          f"{np.exp(np.sqrt(C[1, 1])):.1f} en dP  (no identificable)")
    print(f"      correlacion   = {C[0, 1] / np.sqrt(C[0, 0] * C[1, 1]):+.3f}")
    print("      Advertencia: el operador directo es fuertemente no lineal en "
          "dP")
    print("      (la sensibilidad crece de 0.2 a ~0.8 al aumentar dP), de modo "
          "que")
    print("      la elipse de Fisher subestima la informacion real. El "
          "diagnostico")
    print("      valido es el perfil de verosimilitud global.")
    return C


# ---------------------------------------------------------------------------
def perfiles(radios, dPs, chi):
    """Perfiles de verosimilitud: minimo de chi^2 sobre el otro parametro."""
    return np.nanmin(chi, axis=1), np.nanmin(chi, axis=0)


def figura(radios, dPs, MER, G, base, MER_obs, G_obs):
    chi_est = ((np.log(MER) - np.log(MER_obs)) / SIGMA_MER) ** 2
    chi_tot = chi_est + ((np.log(G) - np.log(G_obs)) / SIGMA_G) ** 2

    fig, ax = plt.subplots(2, 2, figsize=(12.2, 9.6))
    niveles = [1.0, 2.3, 6.2, 11.8, 25.0, 60.0]

    for a, chi, tit in (
        (ax[0, 0], chi_est, "A. Solo dato estacionario (MER)\n"
                            r"$\chi^2$ tiene un valle abierto: no-unicidad"),
        (ax[0, 1], chi_tot, "B. Estacionario + serie de tiempo (MER, $G$)\n"
                            r"el valle se cierra por arriba"),
    ):
        cs = a.contourf(radios, dPs / 1e6, chi.T, levels=niveles,
                        cmap="magma_r", extend="max")
        a.contour(radios, dPs / 1e6, chi.T, levels=[2.3], colors="w",
                  linewidths=1.5)
        a.plot(base.R0, base.dP / 1e6, "*", color="cyan", ms=17,
               markeredgecolor="k", label="valor verdadero")
        a.set_yscale("log")
        a.set_xlabel("radio del conducto  $R$  [m]")
        a.set_ylabel(r"sobrepresion  $\Delta P$  [MPa]")
        a.set_title(tit, fontsize=11)
        a.legend(fontsize=9, loc="upper right")
        fig.colorbar(cs, ax=a, label=r"$\chi^2$")

    perf_R_est, perf_P_est = perfiles(radios, dPs, chi_est)
    perf_R_tot, perf_P_tot = perfiles(radios, dPs, chi_tot)
    umbral = 3.84        # chi^2 al 95% con 1 grado de libertad

    ax[1, 0].plot(radios, perf_R_est, lw=2, color="0.45",
                  label="solo estacionario")
    ax[1, 0].plot(radios, perf_R_tot, lw=2.4, color="crimson",
                  label="estacionario + tiempo")
    ax[1, 0].axvline(base.R0, color="k", ls=":", lw=1)
    ax[1, 0].set_xlabel("radio del conducto  $R$  [m]")
    ax[1, 0].set_title("Perfil de verosimilitud en $R$:\n"
                       "sube a ambos lados $\\Rightarrow$ IDENTIFICABLE",
                       fontsize=11)

    ax[1, 1].semilogx(dPs / 1e6, perf_P_est, lw=2, color="0.45",
                      label="solo estacionario")
    ax[1, 1].semilogx(dPs / 1e6, perf_P_tot, lw=2.4, color="crimson",
                      label="estacionario + tiempo")
    ax[1, 1].axvline(base.dP / 1e6, color="k", ls=":", lw=1)
    ax[1, 1].set_xlabel(r"sobrepresion  $\Delta P$  [MPa]")
    ax[1, 1].set_title(r"Perfil en $\Delta P$: plano hacia abajo"
                       "\n$\\Rightarrow$ solo se obtiene una COTA SUPERIOR",
                       fontsize=11)

    for a in (ax[1, 0], ax[1, 1]):
        a.axhline(umbral, color="steelblue", ls="--", lw=1.2)
        a.text(0.02, umbral, " umbral 95%", color="steelblue", fontsize=8.5,
               va="bottom", transform=a.get_yaxis_transform())
        a.set_ylabel(r"$\chi^2$ perfilado")
        a.set_ylim(0, 25)
        a.grid(alpha=0.3)
        a.legend(fontsize=9)

    fig.suptitle("Identificabilidad de $(R,\\Delta P)$: el radio se recupera; "
                 "la sobrepresion no", fontsize=13)
    fig.tight_layout()
    ruta = os.path.join(FIG, "exp3_identificabilidad.png")
    fig.savefig(ruta, dpi=160, bbox_inches="tight")
    print(f"\n    figura -> {ruta}")

    # Intervalos de confianza leidos del perfil
    def intervalo(vals, perf):
        ok = perf <= umbral
        return (vals[ok].min(), vals[ok].max()) if ok.any() else (np.nan,) * 2

    print("\n(4) Intervalos de confianza al 95% (perfil de verosimilitud)")
    for et, vR, vP in (("solo estacionario", perf_R_est, perf_P_est),
                       ("estacionario + tiempo", perf_R_tot, perf_P_tot)):
        a, b = intervalo(radios, vR)
        c, d = intervalo(dPs, vP)
        print(f"    {et:<24} R  in [{a:.1f}, {b:.1f}] m")
        print(f"    {'':<24} dP in [{c / 1e6:.2f}, {d / 1e6:.1f}] MPa")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    t0 = time.time()
    base = Parametros()
    print("=" * 74)
    print("EXPERIMENTO 3 - Identificabilidad de (R, Delta P)")
    print("=" * 74)

    print("\n(1) Tabla del operador directo")
    radios = np.linspace(8.0, 26.0, 19)
    dPs = np.logspace(np.log10(2.0e5), np.log10(2.5e8), 34)

    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, "tabla_directa.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        if (d["radios"].shape == radios.shape
                and np.allclose(d["radios"], radios)
                and np.allclose(d["dPs"], dPs)):
            MER, G, J, d_nom = d["MER"], d["G"], d["J"], d["d_nom"]
            print(f"    tabla leida de {cache}")
        else:
            os.remove(cache)
    if not os.path.exists(cache):
        MER, G = tabla_directa(base, radios, dPs)
        J, d_nom = jacobiano(base)
        np.savez(cache, radios=radios, dPs=dPs, MER=MER, G=G, J=J,
                 d_nom=d_nom)
        print(f"    tabla guardada en {cache}")

    MER_obs, G_obs = float(np.exp(d_nom[0])), float(np.exp(d_nom[1]))
    print(f"\n    punto nominal: R = {base.R0:g} m, "
          f"dP = {base.dP / 1e6:g} MPa")
    print(f"    MER = {MER_obs:.4e} kg/s,  G = {G_obs:.4f} kg/s/Pa")

    analizar(J)
    figura(radios, dPs, MER, G, base, MER_obs, G_obs)
    print(f"\n    tiempo total: {time.time() - t0:.0f} s")
