"""
mainconduit_ambos.py
====================
Main del estacionario bifasico: tira las DOS ramas (explosiva y efusiva)
con los mismos datos y las dibuja juntas.

No reescribe RIconduitex5_5.py ni RIconduitef5_5.py. Solo los llama, como
main_transient.py llama al estacionario y despues al transiente.

Elegi el caso abajo (CASO) o deja "calbuco2015" para usar calbuco2015d.py.
Despues podes pisar radio / sobrepresion / agua / T / cristales.

Uso
---
    python3 mainconduit_ambos.py
    python3 mainconduit_ambos.py --caso merapi2010
    python3 mainconduit_ambos.py --caso villarrica2015
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from calbuco2015d import (
    radius1, overP1, h2o1, T1, xi1, geometry, dl,
)
from casos.catalogo import CASOS
from ambos_casos import (
    _aplicar, _area, _cargar_solvers, _ultimo,
    aceptada_ex, aceptada_ef,
)

# ── datos (se pueden pisar por CLI) ──────────────────────────────────────────
CASO = "calbuco2015"
radius = radius1
Pressure = overP1
h2o = h2o1
T = T1
xi = xi1


def preparar(caso: str):
    """Carga el catalogo sobre los dos solvers y sobre calbuco2015d."""
    if caso not in CASOS:
        raise KeyError(f"caso {caso!r} no esta. Claves: {list(CASOS)}")
    c = dict(CASOS[caso])
    ex, ef, cal = _cargar_solvers()
    _aplicar(cal, c)
    _aplicar(ex, c)
    _aplicar(ef, c)
    return c, ex, ef, cal


def caudal(v, wr, geom, dl_):
    if geom == "dyke":
        return v * wr * dl_
    return v * np.pi * wr * wr


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--caso", default=CASO)
    p.add_argument("--R", type=float, default=None)
    p.add_argument("--dP", type=float, default=None)
    p.add_argument("--h2o", type=float, default=None)
    p.add_argument("--T", type=float, default=None, help="temperatura [K]")
    p.add_argument("--xi", type=float, default=None)
    args = p.parse_args(argv)

    c, ex, ef, cal = preparar(args.caso)
    wr = args.R if args.R is not None else c["radius1"]
    dP = args.dP if args.dP is not None else c["overP1"]
    wt = args.h2o if args.h2o is not None else c["h2o"]
    Temp = args.T if args.T is not None else (c["Tc"] + 273.15)
    xic = args.xi if args.xi is not None else c["xi"]
    geom = c["geometry"]
    dl_ = c["dl"]
    pfinal = getattr(cal, "pfinal", 1.01325e5)
    Rv = getattr(cal, "R", 461.11)

    print("=" * 60)
    print(f"Caso: {c['nombre']}   (obs: {c['estilo_obs']})")
    print(f"  R={wr:g} m   dP={dP/1e6:g} MPa   H2O={wt} wt%   "
          f"T={Temp-273.15:.0f} C   xi={xic}   geom={geom}")
    print("=" * 60)

    print("\n--- rama EXPLOSIVA (RIconduitex5_5_f) ---")
    raw_ex = ex.RIconduitex5_5_f(wr, dP, wt, Temp, xic)
    zex, solex = np.asarray(raw_ex[0]), np.asarray(raw_ex[1])
    countex, vex, rho_tiex, viscex = raw_ex[2], raw_ex[3], raw_ex[7], raw_ex[6]
    phicritex = raw_ex[9] if len(raw_ex) > 9 else 0.8
    sex = _ultimo(raw_ex)
    sex["phicrit"] = float(phicritex)
    solex_ok = (countex < 49) and aceptada_ex(sex, pfinal, Rv, Temp)
    Qex = caudal(vex, wr, geom, dl_)
    MERex = Qex * rho_tiex

    print("\n--- rama EFUSIVA (RIconduitef5_5_f) ---")
    raw_ef = ef.RIconduitef5_5_f(wr, dP, wt, Temp, xic)
    zef, solef = np.asarray(raw_ef[0]), np.asarray(raw_ef[1])
    countef, vef, rho_tief, viscef = raw_ef[2], raw_ef[3], raw_ef[7], raw_ef[6]
    phicritef = raw_ef[9] if len(raw_ef) > 9 else 0.8
    sef = _ultimo(raw_ef)
    sef["phicrit"] = float(phicritef)
    solef_ok = (countef < 49) and aceptada_ef(sef, pfinal)
    Qef = caudal(vef, wr, geom, dl_)
    MERef = Qef * rho_tief

    print("\n" + "=" * 60)
    print(f"  explosiva : {'SI' if solex_ok else 'no'}   "
          f"count={countex}  vin={vex:.4g} m/s  MER={MERex:.3e} kg/s  "
          f"phi={sex['phi']:.3f}  z={sex['zexit']:.1f} m")
    print(f"  efusiva   : {'SI' if solef_ok else 'no'}   "
          f"count={countef}  vin={vef:.4g} m/s  MER={MERef:.3e} kg/s  "
          f"phi={sef['phi']:.3f}  z={sef['zexit']:.1f} m")
    if solex_ok and solef_ok:
        print("  => LAS DOS (no-unicidad de estilo)")
    elif solex_ok:
        print("  => solo explosiva")
    elif solef_ok:
        print("  => solo efusiva")
    else:
        print("  => ninguna (probo otros R / dP / agua)")
    print("=" * 60)

    fig, axs = plt.subplots(1, 6, figsize=(18, 6), sharey=True)
    ls_ex = "-" if solex_ok else ":"
    axs[0].semilogx(solex[:, 0], zex, color="#C1440E", ls=ls_ex, lw=2,
                    label="explosiva")
    axs[1].plot(solex[:, 1], zex, color="#C1440E", ls=ls_ex, lw=2)
    axs[2].semilogx(solex[:, 4], zex, color="#C1440E", ls=ls_ex, lw=2,
                    label="liq explosiva")
    axs[2].semilogx(solex[:, 5], zex, color="#C1440E", ls="--", lw=1.5,
                    label="gas explosiva")
    axs[3].plot(solex[:, 2], zex, color="#C1440E", ls=ls_ex, lw=2)
    axs[4].plot(solex[:, 3], zex, color="#C1440E", ls=ls_ex, lw=2)
    axs[5].semilogx(viscex, zex, color="#C1440E", ls=ls_ex, lw=2)
    if solef.size:
        ls_ef = "-" if solef_ok else ":"
        axs[0].semilogx(solef[:, 0], zef, color="#0E7C7B", ls=ls_ef, lw=2,
                        label="efusiva")
        axs[1].plot(solef[:, 1], zef, color="#0E7C7B", ls=ls_ef, lw=2)
        axs[2].semilogx(solef[:, 4], zef, color="#0E7C7B", ls=ls_ef, lw=2,
                        label="liq efusiva")
        axs[2].semilogx(solef[:, 5], zef, color="#0E7C7B", ls="--", lw=1.5,
                        label="gas efusiva")
        axs[3].plot(solef[:, 2], zef, color="#0E7C7B", ls=ls_ef, lw=2)
        axs[4].plot(solef[:, 3], zef, color="#0E7C7B", ls=ls_ef, lw=2)
        axs[5].semilogx(viscef, zef, color="#0E7C7B", ls=ls_ef, lw=2)

    axs[0].set_xlabel("Pressure (Pa)", fontweight="bold", fontsize=14)
    axs[1].set_xlabel("gas volume fraction", fontweight="bold", fontsize=14)
    axs[2].set_xlabel("velocity (m/s)", fontweight="bold", fontsize=14)
    axs[3].set_xlabel("Nd", fontweight="bold", fontsize=14)
    axs[4].set_xlabel("crystal content", fontweight="bold", fontsize=14)
    axs[5].set_xlabel("viscosity (Pa.s)", fontweight="bold", fontsize=14)
    axs[0].set_ylabel("Depth (m)", fontweight="bold", fontsize=14)
    axs[0].legend(fontsize=8)
    axs[2].legend(fontsize=7)
    for a in axs:
        a.grid(alpha=0.3)

    txt = []
    if solex_ok:
        txt.append(f"ex MER = {MERex:.3g} kg/s")
    if solef_ok:
        txt.append(f"ef MER = {MERef:.3g} kg/s")
    if txt:
        fig.text(0.62, 0.90, "   |   ".join(txt),
                 bbox=dict(facecolor="white", edgecolor="black"))
    fig.suptitle(f"{c['nombre']}   R={wr:g} m   dP={dP/1e6:g} MPa   "
                 f"H2O={wt} wt%   T={Temp-273.15:.0f} C", fontsize=12)
    fig.tight_layout()

    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "casos", "salida")
    os.makedirs(outdir, exist_ok=True)
    figpath = os.path.join(outdir, f"{args.caso}_mainconduit.png")
    fig.savefig(figpath, dpi=140)
    print(f"\nfigura: {figpath}")
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
