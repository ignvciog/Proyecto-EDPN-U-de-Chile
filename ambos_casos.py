#!/usr/bin/env python3
"""
Corre el estacionario explosivo y el efusivo con LOS MISMOS datos.

No reescribe RIconduitex5_5.py ni RIconduitef5_5.py: los llama. Un mismo
theta = (R, dP, c0, T, xi, ...) puede devolver:

  - solo explosiva
  - solo efusiva
  - las dos
  - ninguna

que es lo que hace el paper (Castruccio, Rebolledo & Gomez 2025).

Uso
---
    python3 ambos_casos.py --listar
    python3 ambos_casos.py --caso calbuco2015
    python3 ambos_casos.py --caso villarrica2015
    python3 ambos_casos.py --caso merapi2010 --quiet
    python3 ambos_casos.py --caso calbuco2015 --R 12 --dP 4e6 --h2o 4 --Tc 970 --xi 0.25

Salida: tabla en consola y, si hay al menos una solucion,
`casos/salida/<caso>_ambos.png`.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import traceback
from contextlib import redirect_stdout
from io import StringIO
from typing import Any

import numpy as np

_REPO = os.path.dirname(os.path.abspath(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
if os.path.join(_REPO, "casos") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO, "casos"))

from catalogo import CASOS, listar  # noqa: E402


CAMPOS_MODULO = (
    "sio2", "tio2", "al2o3", "feo", "mno", "mgo", "cao", "na2o", "k2o",
    "p2o5", "f2o", "C1", "beta", "H", "geometry", "dl", "rcrust",
    "model", "ar1", "ar2", "xmax", "tcar", "Fc",
)


def _area(R: float, geometry: str, dl: float) -> float:
    if geometry == "dyke":
        return 2.0 * R * dl
    return math.pi * R * R


def _aplicar(mod: Any, caso: dict) -> None:
    for k in CAMPOS_MODULO:
        if k in caso:
            setattr(mod, k, caso[k])
    # aliases que algunos bloques del solver leen del modulo
    if hasattr(mod, "h2o1"):
        setattr(mod, "h2o1", caso["h2o"])
    if hasattr(mod, "Tc1"):
        setattr(mod, "Tc1", caso["Tc"])
    if hasattr(mod, "T1"):
        setattr(mod, "T1", caso["Tc"] + 273.15)
    if hasattr(mod, "xi1"):
        setattr(mod, "xi1", caso["xi"])
    if hasattr(mod, "radius1"):
        setattr(mod, "radius1", caso["radius1"])
    if hasattr(mod, "overP1"):
        setattr(mod, "overP1", caso["overP1"])


def _cargar_solvers():
    try:
        import RIconduitex5_5 as ex
        import RIconduitef5_5 as ef
        import calbuco2015d as cal
    except ModuleNotFoundError as err:
        if "scikits" in str(err) or "odes" in str(err):
            raise SystemExit(
                "Falta scikits.odes (el paquete se llama scikit-odes).\n"
                "Los dos solvers originales lo necesitan; este script no los "
                "reemplaza. Instala SUNDIALS + `pip install scikit-odes` y "
                "vuelve a correr."
            ) from err
        raise
    return ex, ef, cal


def _ultimo(sol) -> dict:
    z, y = sol[0], sol[1]
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    return dict(
        z=z, y=y,
        zexit=float(z[-1]),
        P=float(y[-1, 0]),
        phi=float(y[-1, 1]),
        um=float(y[-1, 4]),
        ug=float(y[-1, 5]),
        vinicial=float(sol[3]),
        rho_ti=float(sol[7]),
        visc=sol[6],
    )


def aceptada_ex(s: dict, pfinal: float, Rv: float, T: float) -> bool:
    vs = math.sqrt(Rv * T)
    z, P, ug, phi = s["zexit"], s["P"], s["ug"], s["phi"]
    phic = s.get("phicrit", 0.8)
    choque = (P >= pfinal) and (0.95 * vs < ug <= 1.05 * vs)
    patm_frag = (abs(P - pfinal) < 0.05e5) and (phi >= phic)
    if z >= -5.0 and (choque or patm_frag):
        return True
    return z >= -15.0 and abs(P - pfinal) < 1.5e5 and ug > 0.5


def aceptada_ef(s: dict, pfinal: float) -> bool:
    phic = s.get("phicrit", 0.8)
    return (
        abs(s["P"] - pfinal) <= 0.05e5
        and -2.0 < s["zexit"] <= 0.0
        and s["phi"] <= phic
    )


def _correr(fn, args, quiet: bool):
    buf = StringIO()
    try:
        ctx = redirect_stdout(buf) if quiet else redirect_stdout(sys.stdout)
        with ctx:
            out = fn(*args)
        return out, None
    except Exception as exc:  # noqa: BLE001 — queremos reportar el fallo
        return None, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"


def correr_ambos(
    clave: str = "calbuco2015",
    R: float | None = None,
    dP: float | None = None,
    h2o: float | None = None,
    Tc: float | None = None,
    xi: float | None = None,
    quiet: bool = True,
    figura: str | None = None,
) -> dict:
    if clave not in CASOS:
        raise KeyError(f"caso desconocido {clave!r}. Prueba --listar.")
    caso = dict(CASOS[clave])
    if R is not None:
        caso["radius1"] = float(R)
    if dP is not None:
        caso["overP1"] = float(dP)
    if h2o is not None:
        caso["h2o"] = float(h2o)
    if Tc is not None:
        caso["Tc"] = float(Tc)
    if xi is not None:
        caso["xi"] = float(xi)

    T = caso["Tc"] + 273.15
    pfinal = 1.01325e5
    Rv = 461.11

    ex, ef, cal = _cargar_solvers()
    _aplicar(cal, caso)
    _aplicar(ex, caso)
    _aplicar(ef, caso)

    args = (caso["radius1"], caso["overP1"], caso["h2o"], T, caso["xi"])
    A = _area(caso["radius1"], caso["geometry"], caso["dl"])

    raw_ex, err_ex = _correr(ex.RIconduitex5_5_f, args, quiet)
    raw_ef, err_ef = _correr(ef.RIconduitef5_5_f, args, quiet)

    res: dict[str, Any] = {
        "caso": clave,
        "nombre": caso["nombre"],
        "estilo_obs": caso["estilo_obs"],
        "R": caso["radius1"],
        "dP": caso["overP1"],
        "h2o": caso["h2o"],
        "Tc": caso["Tc"],
        "xi": caso["xi"],
        "geometry": caso["geometry"],
        "ex": None,
        "ef": None,
        "err_ex": err_ex,
        "err_ef": err_ef,
    }

    if raw_ex is not None:
        s = _ultimo(raw_ex)
        s["phicrit"] = float(raw_ex[9]) if len(raw_ex) > 9 else 0.8
        s["q"] = s["vinicial"] * s["rho_ti"]
        s["MER"] = s["q"] * A
        s["ok"] = aceptada_ex(s, pfinal, Rv, T)
        res["ex"] = s

    if raw_ef is not None:
        s = _ultimo(raw_ef)
        s["phicrit"] = float(raw_ef[9]) if len(raw_ef) > 9 else 0.8
        s["q"] = s["vinicial"] * s["rho_ti"]
        s["MER"] = s["q"] * A
        s["ok"] = aceptada_ef(s, pfinal)
        res["ef"] = s

    if figura is None:
        figura = os.path.join(_REPO, "casos", "salida", f"{clave}_ambos.png")
    _dibujar(res, figura)
    res["figura"] = figura if os.path.isfile(figura) else None
    return res


def _dibujar(res: dict, path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    series = []
    if res["ex"] is not None:
        series.append(("explosiva", res["ex"], "#C1440E"))
    if res["ef"] is not None:
        series.append(("efusiva", res["ef"], "#0E7C7B"))
    if not series:
        return

    fig, ax = plt.subplots(1, 4, figsize=(12.2, 4.2), sharey=True)
    for nombre, s, color in series:
        z_km = s["z"] / 1000.0
        ls = "-" if s["ok"] else ":"
        lab = nombre + ("" if s["ok"] else " (no cerro)")
        ax[0].plot(s["y"][:, 0] / 1e6, z_km, color=color, ls=ls, lw=1.8, label=lab)
        ax[1].plot(s["y"][:, 1], z_km, color=color, ls=ls, lw=1.8)
        ax[2].plot(s["y"][:, 4], z_km, color=color, ls=ls, lw=1.6, label=r"$u_m$ "+nombre)
        ax[2].plot(s["y"][:, 5], z_km, color=color, ls="--", lw=1.2, label=r"$u_g$ "+nombre)
        visc = np.asarray(s["visc"], dtype=float)
        if visc.size == s["z"].size:
            ax[3].semilogx(np.clip(visc, 1.0, None), z_km, color=color, ls=ls, lw=1.8)

    ax[0].set_xlabel("P [MPa]")
    ax[1].set_xlabel(r"$\phi$")
    ax[2].set_xlabel("u [m/s]")
    ax[3].set_xlabel(r"$\mu$ [Pa s]")
    ax[0].set_ylabel("z [km]")
    ax[0].legend(fontsize=8, loc="lower right")
    ax[2].legend(fontsize=7, loc="lower right")
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle(
        f"{res['nombre']}  |  R={res['R']:.2f} m,  "
        f"dP={res['dP']/1e6:.2f} MPa,  "
        f"H2O={res['h2o']:.1f} wt%,  T={res['Tc']:.0f} C",
        fontsize=10,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _fmt_rama(s: dict | None, err: str | None, pfinal: float) -> str:
    if err:
        return f"FALLO ({err.splitlines()[0]})"
    if s is None:
        return "sin salida"
    marca = "SI" if s["ok"] else "no cerro el tiro"
    return (
        f"{marca:14s}  MER={s['MER']:.3e} kg/s   "
        f"vin={s['vinicial']:.4g} m/s   "
        f"z={s['zexit']:.1f} m   P={s['P']/1e5:.2f} bar   "
        f"phi={s['phi']:.3f}   ug={s['ug']:.2f} m/s"
    )


def imprimir(res: dict) -> None:
    print()
    print(f"Caso: {res['nombre']}  (obs: {res['estilo_obs']})")
    print(f"  R={res['R']:.3g} m   dP={res['dP']/1e6:.3g} MPa   "
          f"H2O={res['h2o']} wt%   T={res['Tc']} C   xi={res['xi']}   "
          f"geom={res['geometry']}")
    print(f"  explosiva : {_fmt_rama(res['ex'], res['err_ex'], 1.01325e5)}")
    print(f"  efusiva   : {_fmt_rama(res['ef'], res['err_ef'], 1.01325e5)}")
    ex_ok = res["ex"] is not None and res["ex"]["ok"]
    ef_ok = res["ef"] is not None and res["ef"]["ok"]
    if ex_ok and ef_ok:
        veredicto = "LAS DOS (no-unicidad de estilo)"
    elif ex_ok:
        veredicto = "solo explosiva"
    elif ef_ok:
        veredicto = "solo efusiva"
    else:
        veredicto = "ninguna (con estos R, dP)"
    print(f"  => {veredicto}")
    if res.get("figura"):
        print(f"  figura: {res['figura']}")
    print()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--caso", default="calbuco2015",
                   help="clave del catalogo (default: calbuco2015)")
    p.add_argument("--listar", action="store_true")
    p.add_argument("--R", type=float, default=None, help="radio o semi-ancho [m]")
    p.add_argument("--dP", type=float, default=None, help="sobrepresion [Pa]")
    p.add_argument("--h2o", type=float, default=None, help="agua total [wt%%]")
    p.add_argument("--Tc", type=float, default=None, help="temperatura [C]")
    p.add_argument("--xi", type=float, default=None, help="fraccion de cristales")
    p.add_argument("--quiet", action="store_true",
                   help="silencia el print de cada iteracion del tiro")
    args = p.parse_args(argv)
    if args.listar:
        listar()
        return 0
    res = correr_ambos(
        clave=args.caso, R=args.R, dP=args.dP, h2o=args.h2o,
        Tc=args.Tc, xi=args.xi, quiet=args.quiet,
    )
    imprimir(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
