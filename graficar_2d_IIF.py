"""Figuras de la marcha II.F: columna, perfiles y cortes radiales del capilar.

Uso, desde esta carpeta:
    python graficar_2d_IIF.py
Las imágenes quedan en ./figuras.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogFormatterMathtext

import RIconduit2D_IIF as IIF


def _log_eje(ax, ticks):
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(LogFormatterMathtext())


def columnas(sal, path, titulo):
    z = sal["z"] / 1000.0
    fig, ejes = plt.subplots(1, 4, figsize=(12.4, 6.2), sharey=True)
    ejes[0].plot(sal["P"] / 1e6, z, color="C0")
    ejes[0].set_xlabel("P [MPa]")
    ejes[0].set_xlim(1, 300)
    _log_eje(ejes[0], [1, 10, 100])

    ejes[1].plot(np.maximum(sal["um_media"], 1e-3), z, color="C1", label="u_m media")
    ejes[1].plot(np.maximum(sal["um"][:, 0], 1e-3), z, color="C3", ls="--", label="u_m eje")
    ejes[1].plot(np.maximum(sal["ug"][:, 0], 1e-3), z, color="C6", ls="-.", label="u_g eje")
    ejes[1].plot(np.maximum(sal["um"][:, -2], 1e-3), z, color="C5", ls=":", label="u_m junto al borde")
    ejes[1].set_xlabel("u [m/s]")
    ejes[1].set_xlim(1, 2000)
    _log_eje(ejes[1], [1, 10, 100, 1000])
    ejes[1].legend(frameon=False, fontsize=7)

    ejes[2].plot(sal["phi"][:, 0], z, color="C2", label="eje")
    ejes[2].plot(sal["phi"][:, -1], z, color="C5", ls=":", label="borde")
    ejes[2].set_xscale("linear")
    ejes[2].set_xlabel("φ")
    ejes[2].set_xlim(0.0, 1.0)
    ejes[2].set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ejes[2].legend(frameon=False, fontsize=7)

    ejes[3].plot(sal["xi"][:, 0], z, color="C4", label="eje")
    ejes[3].plot(sal["xi"][:, -1], z, color="C5", ls=":", label="borde")
    ejes[3].set_xlabel("ξ")
    ejes[3].legend(frameon=False, fontsize=7)

    for ax in ejes:
        ax.axhline(sal["z_sat"] / 1000.0, color="0.45", lw=0.7, label="saturación")
        ax.axhline(sal["z_ad"] / 1000.0, color="0.45", lw=0.7, ls="--", label="metro de exsolución")
        ax.grid(True, which="both", alpha=0.3)
    ejes[0].legend(frameon=False, fontsize=7)
    ejes[0].set_ylabel("z [km]")
    fig.suptitle(titulo)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def tramo_final(sal, path, metros=20.0):
    """Últimos metros, con P lineal, para ver el empinamiento y el borde."""
    z = sal["z"]
    corte = z[-1] - metros
    m = z >= corte
    zz = z[m]
    fig, ejes = plt.subplots(1, 3, figsize=(11.2, 4.6), sharey=True)
    ejes[0].plot(sal["P"][m] / 1e6, zz, color="C0")
    ejes[0].set_xlabel("P [MPa]")

    ejes[1].plot(sal["um"][m, 0], zz, color="C3", ls="--", label="u_m eje")
    ejes[1].plot(sal["ug"][m, 0], zz, color="C6", ls="-.", label="u_g eje")
    ejes[1].plot(sal["um"][m, -2], zz, color="C5", ls=":", label="u_m junto al borde")
    ejes[1].plot(sal["um"][m, -1], zz, color="0.3", label="borde, u = 0")
    ejes[1].set_xlabel("u [m/s]")
    ejes[1].set_xlim(left=0.0)
    ejes[1].legend(frameon=False, fontsize=7)

    ejes[2].plot(sal["phi"][m, 0], zz, color="C2", label="eje")
    ejes[2].plot(sal["phi"][m, -1], zz, color="C5", ls=":", label="borde")
    ejes[2].set_xlabel("φ")
    ejes[2].set_xlim(0.0, 1.0)
    ejes[2].legend(frameon=False, fontsize=7)

    for ax in ejes:
        ax.grid(True, alpha=0.3)
    ejes[0].set_ylabel("z [m]")
    fig.suptitle(
        f"Últimos {metros:.0f} m, v_in = {sal['vinicial']:.0f} m/s, {sal['mensaje']}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def perfiles(sal, path):
    r = sal["r"]
    z = sal["z"]
    indices = [
        0,
        int(np.argmin(np.abs(z - sal["z_ad"]))),
        int(np.argmin(np.abs(z - 0.5 * (sal["z_ad"] + z[-1])))),
        len(z) - 1,
    ]
    fig, ejes = plt.subplots(1, 2, figsize=(9.2, 4.4))
    for j in indices:
        etiqueta = f"z = {z[j]:.0f} m"
        ejes[0].plot(r, sal["um"][j], label=etiqueta)
        ejes[1].plot(r, sal["phi"][j], label=etiqueta)
    ejes[0].set_ylabel("u_m = u_g [m/s]")
    ejes[1].set_ylabel("φ")
    for ax in ejes:
        ax.set_xlabel("r [m]")
        ax.set_xlim(0, r[-1])
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Perfiles, v_in = {sal['vinicial']:.0f} m/s, {sal['mensaje']}")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def cortes(path):
    r, _ = IIF._malla(12, 16.0)
    co, pi = IIF._configurar_fd(16.0, 5.0e6, 4.0, 1243.15, 0.25)
    rho = IIF._rho_m(pi)
    fig, ejes = plt.subplots(1, 2, figsize=(9.4, 4.4))
    for v, color in ((10.0, "C0"), (23.0, "C3")):
        mu = IIF._mu_liquido(pi, 0.25, v / 16.0, 1243.15, co)
        lim = IIF.limites_regiones(r, v, 16.0, pi, rho, mu, co, 0.25, 1243.15)
        ejes[0].plot(r, lim["Ca"], color=color, label=f"v_in = {v:.0f} m/s")
        ejes[1].plot(r, lim["phicrit_ca"], color=color, label=f"φ_crit, {v:.0f} m/s")
        ejes[1].plot(r, lim["limphi1"], color=color, ls="--", label=f"φ_1, {v:.0f} m/s")
    ejes[0].set_ylabel("Ca")
    ejes[1].set_ylabel("corte en φ")
    ejes[1].set_ylim(0.1, 0.85)
    for ax in ejes:
        ax.set_xlabel("r [m]")
        ax.set_xlim(0, 16)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Cortes del capilar sobre la parábola, estado φ = 0.2")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(destino="figuras"):
    os.makedirs(destino, exist_ok=True)
    # Ocho radios: es la malla de las pruebas. Con doce, φ llega al tope 0.97
    # y la presión del último paso se va a unas décimas de MPa.
    alto = IIF.marchar(23.0, n_r=8)
    bajo = IIF.marchar(10.0, n_r=8)
    columnas(
        alto,
        os.path.join(destino, "columnas-v23.png"),
        f"v_in = 23 m/s, {alto['mensaje']}, z = {alto['z'][-1]:.0f} m",
    )
    columnas(
        bajo,
        os.path.join(destino, "columnas-v10.png"),
        f"v_in = 10 m/s, {bajo['mensaje']}, P = {bajo['P'][-1] / 1e6:.2f} MPa",
    )
    tramo_final(alto, os.path.join(destino, "cola-v23.png"))
    tramo_final(bajo, os.path.join(destino, "cola-v10.png"))
    perfiles(alto, os.path.join(destino, "perfiles-v23.png"))
    perfiles(bajo, os.path.join(destino, "perfiles-v10.png"))
    cortes(os.path.join(destino, "limites-r.png"))
    print(
        f"v=23 {alto['mensaje']} z={alto['z'][-1]:.1f} P={alto['P'][-1] / 1e6:.2f} MPa "
        f"φ={alto['phi'][-1, 0]:.3f}"
    )
    print(
        f"v=10 {bajo['mensaje']} z={bajo['z'][-1]:.1f} P={bajo['P'][-1] / 1e6:.2f} MPa "
        f"φ={bajo['phi'][-1, 0]:.3f}"
    )
    print("figuras en", os.path.abspath(destino))


if __name__ == "__main__":
    main()
