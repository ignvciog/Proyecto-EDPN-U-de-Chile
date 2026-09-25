"""Genera las figuras de regularizar_umbrales.ipynb (switch vs rampa)."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from umbrales_reg import (
    EPS_FRAG, EPS_HENRY, EPS_PHI, EPS_RE, EPS_RB, EPS_XI, RE_CRIT,
    g_phi, g_rb, pesos_regimen, residual_efusivo, residual_explosivo,
    s_frag, s_henry, s_phi1, s_phi2, s_re, softmin2, softplus, switch,
)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "umbrales_reg_figuras")
os.makedirs(OUT, exist_ok=True)

# valores típicos Calbuco / el código
PHICRIT = 0.70
LIM1 = 0.20
LIM2 = 0.21
WR = 16.0
PATM = 101325.0
CS = (461.11 * 1243.15) ** 0.5


def _guardar(fig, nombre):
    path = os.path.join(OUT, nombre)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def fig_fragmentacion():
    phi = np.linspace(0.40, 0.95, 400)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(phi, switch(phi, PHICRIT), "k", lw=2, label="switch")
    for eps, ls in [(0.01, "--"), (EPS_FRAG, "-"), (0.10, ":")]:
        ax.plot(phi, s_frag(phi, PHICRIT, eps), ls, lw=2, label=f"eps = {eps}")
    ax.axvline(PHICRIT, color="gray", lw=1)
    ax.set_xlabel(r"$\phi$", fontweight="bold")
    ax.set_ylabel(r"$s_f$", fontweight="bold")
    ax.set_title("fragmentación  (phicrit)")
    ax.legend()
    fig.tight_layout()
    _guardar(fig, "01_fragmentacion.png")


def fig_regimenes():
    phi = np.linspace(0.05, 0.95, 500)
    w1, w2, w3, w4 = pesos_regimen(phi, LIM1, LIM2, PHICRIT)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(phi, w1, lw=2, label="w1  Stokes")
    ax.plot(phi, w2, lw=2, label="w2  mezcla perm.")
    ax.plot(phi, w3, lw=2, label="w3  Darcy/inercial")
    ax.plot(phi, w4, lw=2, label="w4  fragmentado")
    ax.plot(phi, w1 + w2 + w3 + w4, "k--", lw=1, label="suma")
    ax.axvline(LIM1, color="gray", lw=1)
    ax.axvline(LIM2, color="gray", lw=1)
    ax.axvline(PHICRIT, color="gray", lw=1)
    ax.set_xlabel(r"$\phi$", fontweight="bold")
    ax.set_ylabel("peso", fontweight="bold")
    ax.set_title(rf"regímenes  (eps_phi={EPS_PHI}, eps_f={EPS_FRAG})")
    ax.legend(ncol=2)
    fig.tight_layout()
    _guardar(fig, "02_regimenes.png")


def fig_henry():
    test = np.linspace(-0.01, 0.01, 400)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(test, switch(test, 0.0), "k", lw=2, label="switch  test>0")
    for eps, ls in [(1e-4, "--"), (EPS_HENRY, "-"), (3e-3, ":")]:
        ax.plot(test, s_henry(test, eps), ls, lw=2, label=f"eps = {eps}")
    ax.axvline(0.0, color="gray", lw=1)
    ax.set_xlabel("test  (agua exsuelta)", fontweight="bold")
    ax.set_ylabel(r"$s_H$", fontweight="bold")
    ax.set_title("Henry  (exsolución on/off)")
    ax.legend()
    fig.tight_layout()
    _guardar(fig, "03_henry.png")


def fig_reynolds():
    Re = np.linspace(0, 5000, 400)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(Re, switch(Re, RE_CRIT), "k", lw=2, label="switch  Re>2200")
    for eps, ls in [(100, "--"), (EPS_RE, "-"), (600, ":")]:
        ax.plot(Re, s_re(Re, RE_CRIT, eps), ls, lw=2, label=f"eps = {eps}")
    ax.axvline(RE_CRIT, color="gray", lw=1)
    ax.set_xlabel("Re", fontweight="bold")
    ax.set_ylabel(r"$s_{Re}$", fontweight="bold")
    ax.set_title("Reynolds  (Stokes/Darcy vs inercial)")
    ax.legend()
    fig.tight_layout()
    _guardar(fig, "04_reynolds.png")


def fig_cristales():
    x = np.linspace(-0.02, 0.02, 400)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, np.maximum(0.0, x), "k", lw=2, label="max(0, x)")
    for eps, ls in [(3e-4, "--"), (EPS_XI, "-"), (4e-3, ":")]:
        ax.plot(x, softplus(x, eps), ls, lw=2, label=f"eps = {eps}")
    ax.set_xlabel("x  (f2, f3, dxi/dt)", fontweight="bold")
    ax.set_ylabel("softplus", fontweight="bold")
    ax.set_title("cristales  (max → softplus)")
    ax.legend()
    fig.tight_layout()
    _guardar(fig, "05_cristales.png")


def fig_coalescencia():
    rb = np.linspace(0, WR, 400)
    phi = np.linspace(0.2, 0.95, 400)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(rb / WR, (rb < 0.5 * WR).astype(float), "k", lw=2, label="switch")
    for eps, ls in [(0.02, "--"), (EPS_RB, "-"), (0.12, ":")]:
        axs[0].plot(rb / WR, g_rb(rb, WR, eps), ls, lw=2, label=f"eps/R = {eps}")
    axs[0].axvline(0.5, color="gray", lw=1)
    axs[0].set_xlabel(r"$r_b / R$", fontweight="bold")
    axs[0].set_ylabel(r"$g_r$", fontweight="bold")
    axs[0].set_title("coalescencia: radio de burbuja")
    axs[0].legend()

    axs[1].plot(phi, (phi < PHICRIT).astype(float), "k", lw=2, label="switch")
    for eps, ls in [(0.01, "--"), (EPS_FRAG, "-"), (0.10, ":")]:
        axs[1].plot(phi, g_phi(phi, PHICRIT, eps), ls, lw=2, label=f"eps = {eps}")
    axs[1].axvline(PHICRIT, color="gray", lw=1)
    axs[1].set_xlabel(r"$\phi$", fontweight="bold")
    axs[1].set_ylabel(r"$g_\phi$", fontweight="bold")
    axs[1].set_title("coalescencia: no fragmentó")
    axs[1].legend()
    fig.tight_layout()
    _guardar(fig, "06_coalescencia.png")


def fig_tiro_ex():
    P = np.linspace(0.2 * PATM, 2.2 * PATM, 300)
    ug = np.linspace(0.2 * CS, 1.8 * CS, 300)
    phi = np.linspace(0.4, 0.95, 300)
    z = np.linspace(-20, 2, 300)

    rex_P = residual_explosivo(P, PATM, CS, CS, 0.8, PHICRIT, 0.0)
    rex_u = residual_explosivo(PATM, PATM, ug, CS, 0.8, PHICRIT, 0.0)
    rex_f = residual_explosivo(PATM, PATM, CS, CS, phi, PHICRIT, 0.0)
    rex_z = residual_explosivo(PATM, PATM, CS, CS, 0.8, PHICRIT, z)

    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    axs[0, 0].plot(P / 1e5, rex_P["r_P"] / 1e5, lw=2)
    axs[0, 0].axhline(0, color="k", lw=1)
    axs[0, 0].set_xlabel("P (bar)")
    axs[0, 0].set_title(r"$r_P = P - P_{atm}$")

    axs[0, 1].plot(ug / CS, rex_u["r_c"] / CS, lw=2)
    axs[0, 1].axhline(0, color="k", lw=1)
    axs[0, 1].set_xlabel(r"$u_g / c_s$")
    axs[0, 1].set_title(r"$r_c = u_g - c_s$")

    axs[1, 0].plot(phi, rex_f["r_phi"], lw=2)
    axs[1, 0].axhline(0, color="k", lw=1)
    axs[1, 0].axvline(PHICRIT, color="gray", lw=1)
    axs[1, 0].set_xlabel(r"$\phi$")
    axs[1, 0].set_title(r"$r_\phi$  (0 si $\phi \geq \phi_{crit}$)")

    axs[1, 1].plot(z, rex_z["r_z"], lw=2)
    axs[1, 1].axhline(0, color="k", lw=1)
    axs[1, 1].axvline(-5, color="gray", lw=1)
    axs[1, 1].set_xlabel("z_exit (m)")
    axs[1, 1].set_title(r"$r_z$  (0 si $z \geq -5$)")
    fig.suptitle("residual explosivo (queremos todo en 0)")
    fig.tight_layout()
    _guardar(fig, "07_tiro_explosivo.png")


def fig_tiro_ef():
    P = np.linspace(0.2 * PATM, 2.2 * PATM, 300)
    phi = np.linspace(0.4, 0.95, 300)
    z = np.linspace(-8, 2, 300)
    ref_P = residual_efusivo(P, PATM, 0.4, PHICRIT, -1.0)
    ref_f = residual_efusivo(PATM, PATM, phi, PHICRIT, -1.0)
    ref_z = residual_efusivo(PATM, PATM, 0.4, PHICRIT, z)

    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8))
    axs[0].plot(P / 1e5, ref_P["r_P"] / 1e5, lw=2)
    axs[0].axhline(0, color="k", lw=1)
    axs[0].set_title(r"$r_P$")
    axs[0].set_xlabel("P (bar)")
    axs[1].plot(phi, ref_f["r_phi"], lw=2)
    axs[1].axvline(PHICRIT, color="gray", lw=1)
    axs[1].axhline(0, color="k", lw=1)
    axs[1].set_title(r"$r_\phi$  (0 si $\phi \leq \phi_{crit}$)")
    axs[1].set_xlabel(r"$\phi$")
    axs[2].plot(z, ref_z["r_z"], lw=2)
    axs[2].axvspan(-2, 0, color="0.85")
    axs[2].axhline(0, color="k", lw=1)
    axs[2].set_title(r"$r_z$  (0 si z en (-2, 0])")
    axs[2].set_xlabel("z_exit (m)")
    fig.suptitle("residual efusivo")
    fig.tight_layout()
    _guardar(fig, "08_tiro_efusivo.png")


def fig_softmin_or():
    """P_atm O choque: min(r_P^2, r_c^2) suave."""
    rP = np.linspace(-2, 2, 300)
    rc0 = 1.0
    duro = np.minimum(rP ** 2, rc0 ** 2)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(rP, duro, "k", lw=2, label="min(r_P², r_c²)")
    for eps, ls in [(0.05, "--"), (0.2, "-"), (0.6, ":")]:
        ax.plot(rP, softmin2(rP ** 2, rc0 ** 2, eps), ls, lw=2, label=f"eps = {eps}")
    ax.set_xlabel(r"$r_P$  (r_c fijo = 1)", fontweight="bold")
    ax.set_title(r"explosivo: $P_{atm}$ O choque")
    ax.legend()
    fig.tight_layout()
    _guardar(fig, "09_or_choque.png")


def fig_todas():
    """Tablero: switch vs epsilon escogido (y uno más flaco / más gordo)."""
    fig, axs = plt.subplots(3, 3, figsize=(13.5, 10.5))

    phi = np.linspace(0.40, 0.95, 400)
    axs[0, 0].plot(phi, switch(phi, PHICRIT), "k", lw=2, label="switch")
    axs[0, 0].plot(phi, s_frag(phi, PHICRIT, EPS_FRAG), lw=2.4, label=f"escogido {EPS_FRAG}")
    axs[0, 0].plot(phi, s_frag(phi, PHICRIT, 0.01), "--", lw=1.4, label="0.01")
    axs[0, 0].plot(phi, s_frag(phi, PHICRIT, 0.10), ":", lw=1.6, label="0.10")
    axs[0, 0].axvline(PHICRIT, color="gray", lw=1)
    axs[0, 0].set_title("fragmentación")
    axs[0, 0].legend(fontsize=8)

    w1, w2, w3, w4 = pesos_regimen(np.linspace(0.05, 0.95, 500), LIM1, LIM2, PHICRIT)
    phiw = np.linspace(0.05, 0.95, 500)
    axs[0, 1].plot(phiw, w1, lw=2, label="w1")
    axs[0, 1].plot(phiw, w2, lw=2, label="w2")
    axs[0, 1].plot(phiw, w3, lw=2, label="w3")
    axs[0, 1].plot(phiw, w4, lw=2, label="w4")
    axs[0, 1].plot(phiw, w1 + w2 + w3 + w4, "k--", lw=1, label="suma")
    axs[0, 1].set_title(f"regímenes  φ={EPS_PHI}, f={EPS_FRAG}")
    axs[0, 1].legend(ncol=3, fontsize=8)

    test = np.linspace(-0.01, 0.01, 400)
    axs[0, 2].plot(test, switch(test, 0.0), "k", lw=2, label="switch")
    axs[0, 2].plot(test, s_henry(test, EPS_HENRY), lw=2.4, label=f"escogido {EPS_HENRY}")
    axs[0, 2].plot(test, s_henry(test, 1e-4), "--", lw=1.4, label="1e-4")
    axs[0, 2].plot(test, s_henry(test, 3e-3), ":", lw=1.6, label="3e-3")
    axs[0, 2].axvline(0.0, color="gray", lw=1)
    axs[0, 2].set_title("Henry")
    axs[0, 2].legend(fontsize=8)

    Re = np.linspace(0, 5000, 400)
    axs[1, 0].plot(Re, switch(Re, RE_CRIT), "k", lw=2, label="switch")
    axs[1, 0].plot(Re, s_re(Re, RE_CRIT, EPS_RE), lw=2.4, label=f"escogido {EPS_RE}")
    axs[1, 0].plot(Re, s_re(Re, RE_CRIT, 100), "--", lw=1.4, label="100")
    axs[1, 0].plot(Re, s_re(Re, RE_CRIT, 600), ":", lw=1.6, label="600")
    axs[1, 0].axvline(RE_CRIT, color="gray", lw=1)
    axs[1, 0].set_title("Reynolds")
    axs[1, 0].legend(fontsize=8)

    x = np.linspace(-0.02, 0.02, 400)
    axs[1, 1].plot(x, np.maximum(0.0, x), "k", lw=2, label="max(0,x)")
    axs[1, 1].plot(x, softplus(x, EPS_XI), lw=2.4, label=f"escogido {EPS_XI}")
    axs[1, 1].plot(x, softplus(x, 3e-4), "--", lw=1.4, label="3e-4")
    axs[1, 1].plot(x, softplus(x, 4e-3), ":", lw=1.6, label="4e-3")
    axs[1, 1].set_title("cristales")
    axs[1, 1].legend(fontsize=8)

    rb = np.linspace(0, WR, 400)
    axs[1, 2].plot(rb / WR, (rb < 0.5 * WR).astype(float), "k", lw=2, label="switch")
    axs[1, 2].plot(rb / WR, g_rb(rb, WR, EPS_RB), lw=2.4, label=f"escogido {EPS_RB}")
    axs[1, 2].plot(rb / WR, g_rb(rb, WR, 0.02), "--", lw=1.4, label="0.02")
    axs[1, 2].plot(rb / WR, g_rb(rb, WR, 0.12), ":", lw=1.6, label="0.12")
    axs[1, 2].axvline(0.5, color="gray", lw=1)
    axs[1, 2].set_title("coalescencia  g_r")
    axs[1, 2].legend(fontsize=8)

    z = np.linspace(-20, 2, 300)
    rex_z = residual_explosivo(PATM, PATM, CS, CS, 0.8, PHICRIT, z)
    axs[2, 0].plot(z, rex_z["r_z"], lw=2)
    axs[2, 0].axhline(0, color="k", lw=1)
    axs[2, 0].axvline(-5, color="gray", lw=1)
    axs[2, 0].set_title("tiro ex  r_z")
    axs[2, 0].set_xlabel("z_exit")

    zef = np.linspace(-8, 2, 300)
    ref_z = residual_efusivo(PATM, PATM, 0.4, PHICRIT, zef)
    axs[2, 1].plot(zef, ref_z["r_z"], lw=2)
    axs[2, 1].axvspan(-2, 0, color="0.85")
    axs[2, 1].axhline(0, color="k", lw=1)
    axs[2, 1].set_title("tiro ef  r_z")
    axs[2, 1].set_xlabel("z_exit")

    rP = np.linspace(-2, 2, 300)
    axs[2, 2].plot(rP, np.minimum(rP ** 2, 1.0), "k", lw=2, label="min")
    axs[2, 2].plot(rP, softmin2(rP ** 2, 1.0, 0.2), lw=2.4, label="escogido 0.2")
    axs[2, 2].plot(rP, softmin2(rP ** 2, 1.0, 0.05), "--", lw=1.4, label="0.05")
    axs[2, 2].plot(rP, softmin2(rP ** 2, 1.0, 0.6), ":", lw=1.6, label="0.6")
    axs[2, 2].set_title(r"$P_{atm}$ O choque")
    axs[2, 2].legend(fontsize=8)

    fig.suptitle("todas las rampas  (línea gruesa = epsilon escogido)", fontweight="bold")
    fig.tight_layout()
    _guardar(fig, "10_todas.png")


if __name__ == "__main__":
    fig_fragmentacion()
    fig_regimenes()
    fig_henry()
    fig_reynolds()
    fig_cristales()
    fig_coalescencia()
    fig_tiro_ex()
    fig_tiro_ef()
    fig_softmin_or()
    fig_todas()
    from figuras_tasa_xi import main as fig_leak
    fig_leak()
    print("listo", OUT)
