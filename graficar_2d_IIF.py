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
        if np.isfinite(sal["z_frag"]):
            ax.axhline(sal["z_frag"] / 1000.0, color="0.15", lw=0.8, ls=":", label="fragmentación")
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


def velocidades(sal, path):
    """u_m y u_g en escala logarítmica: la columna y el tramo ya fragmentado."""
    z = sal["z"] / 1000.0
    zf = float(sal["z_frag"])
    fig, ejes = plt.subplots(1, 2, figsize=(9.4, 6.0))
    mascara = (
        np.ones(len(z), dtype=bool),
        sal["z"] >= zf - 40.0,
    )
    titulos = ("columna completa", "desde la fragmentación")
    for ax, m, titulo in zip(ejes, mascara, titulos):
        ax.plot(np.maximum(sal["um"][m, 0], 1e-3), z[m], color="C3", ls="--", label="u_m eje")
        ax.plot(np.maximum(sal["ug"][m, 0], 1e-3), z[m], color="C6", ls="-.", label="u_g eje")
        if np.isfinite(zf):
            ax.axhline(zf / 1000.0, color="0.15", lw=0.8, ls=":", label="fragmentación")
        _log_eje(ax, [10, 100, 1000])
        ax.set_xlim(10, 2000)
        ax.set_xlabel("u [m/s]")
        ax.set_ylabel("z [km]")
        ax.set_title(titulo)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(
        f"v_in = {sal['vinicial']:.0f} m/s, {sal['mensaje']}, "
        f"fragmenta en z = {zf:.0f} m"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def altura_fragmentacion(sal):
    """Primera altura en que φ del nodo alcanza el φ_crit de su radio."""
    z = sal["z"]
    out = np.full(len(sal["r"]), np.nan)
    for i in range(len(sal["r"])):
        arriba = np.where(sal["phi"][:, i] >= sal["phicrit_ca"][i])[0]
        if len(arriba):
            out[i] = z[int(arriba[0])]
    return out


def fragmentacion(sal, path):
    """φ(z) contra los dos φ_crit, y la altura de cruce en cada radio."""
    r = sal["r"]
    zf = altura_fragmentacion(sal)
    m = (sal["z"] > zf[0] - 250.0) & (sal["z"] < zf[-1] + 80.0)
    fig, ejes = plt.subplots(1, 2, figsize=(9.8, 4.8))
    ejes[0].plot(sal["phi"][m, 0], sal["z"][m] / 1000.0, color="C2", label="φ, eje y borde")
    ejes[0].plot(sal["phi"][m, -1], sal["z"][m] / 1000.0, color="C5", ls=":")
    ejes[0].axvline(
        sal["phicrit_ca"][0], color="C2", lw=0.9, ls="--",
        label=f"φ_crit eje = {sal['phicrit_ca'][0]:.2f}",
    )
    ejes[0].axvline(
        sal["phicrit_ca"][-1], color="C5", lw=0.9, ls="--",
        label=f"φ_crit borde = {sal['phicrit_ca'][-1]:.2f}",
    )
    ejes[0].set_xlim(0.45, 0.80)
    ejes[0].set_xlabel("φ")
    ejes[0].set_ylabel("z [km]")
    ejes[0].legend(frameon=False, fontsize=8)

    ejes[1].plot(r, zf, color="C0", marker="o")
    ejes[1].axhline(sal["z_frag"], color="0.35", ls="--", label="la sección cambia de cierre")
    ejes[1].set_xlim(0.0, r[-1])
    ejes[1].set_xlabel("r [m]")
    ejes[1].set_ylabel("z en que φ alcanza φ_crit(r) [m]")
    ejes[1].legend(frameon=False, fontsize=8)
    for ax in ejes:
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        f"Fragmentación por radio, v_in = {sal['vinicial']:.0f} m/s"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def nucleacion(sal, path):
    """N(z) con el eje horizontal lineal."""
    z = sal["z"] / 1000.0
    fig, ax = plt.subplots(figsize=(5.6, 6.2))
    ax.plot(sal["N"][:, 0], z, color="C0", label="eje")
    ax.plot(sal["N"][:, -1], z, color="C5", ls=":", label="borde")
    ax.set_xscale("linear")
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.axhline(sal["z_frag"] / 1000.0, color="0.45", lw=0.7, ls="--")
    ax.set_xlabel("N [m$^{-3}$]")
    ax.set_ylabel("z [km]")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"N, v_in = {sal['vinicial']:.0f} m/s, {sal['mensaje']}")
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
    medio = IIF.marchar(18.0, n_r=8)
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
    velocidades(medio, os.path.join(destino, "velocidad-v18-log.png"))
    velocidades(bajo, os.path.join(destino, "velocidad-v10-log.png"))
    tramo_final(alto, os.path.join(destino, "cola-v23.png"))
    tramo_final(bajo, os.path.join(destino, "cola-v10.png"))
    fragmentacion(alto, os.path.join(destino, "fragmentacion-v23.png"))
    fragmentacion(bajo, os.path.join(destino, "fragmentacion-v10.png"))
    nucleacion(alto, os.path.join(destino, "nucleacion-v23.png"))
    nucleacion(bajo, os.path.join(destino, "nucleacion-v10.png"))
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
