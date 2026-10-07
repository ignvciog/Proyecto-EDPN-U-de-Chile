"""
Conducto volcanico para la auxiliar 8 (GL4212).

Solo usa la biblioteca estandar de Python. Si matplotlib esta instalado,
ademas guarda un grafico. No hace falta numpy, scipy ni scikits.odes.

Dos cuentas, a proposito distintas:

  analogico(...)   chorro inercial, tipo botella.  u = sqrt(2 dP / rho)
  correr(...)      conducto viscoso con exsolucion.  Poiseuille + Henry

Ejemplos
--------
  python3 conducto_actividad.py
  python3 conducto_actividad.py 16
  python3 -c "from conducto_actividad import correr; print(correr(radio=16))"
"""

from __future__ import annotations

import math


G = 9.81
PATM = 1.01325e5
RHO_CORTEZA = 2600.0
RHO_M = 2500.0
RHO_ANALOGO = 1000.0
MU = 3.0e3                 # Pa s, viscosidad efectiva del fundido con agua
C1 = 4.11e-6               # solubilidad de Henry, fraccion masica / Pa^beta
BETA = 0.5
RV = 461.11                # J/kg/K
T_MAGMA = 1200.0           # K
AGUA = 0.04                # fraccion masica
PHI_FRAG = 0.75
SOBREPRESION = 5.0e6       # Pa
PROFUNDIDAD = 7000.0       # m
RADIO = 16.0               # m
VOLUMEN = 0.3e9            # m3 de magma denso, ejemplo del orden de Calbuco 2015
# Mastin et al. (2009): H [km] = 2.00 * Vdot^0.241, con Vdot en m3/s de roca densa.
K_ALTURA = 2.00
B_ALTURA = 0.241


def agua_disuelta(P):
    """Fraccion masica que cabe disuelta a la presion P."""
    return C1 * max(P, 1.0) ** BETA


def mezcla(P, agua=AGUA, temperatura=T_MAGMA, rho_m=RHO_M):
    """Densidad de la mezcla y fraccion de gas, con Henry y gas ideal."""
    cd = min(agua, agua_disuelta(P))
    if agua <= cd:
        return rho_m, 0.0
    n = (agua - cd) / (1.0 - cd)
    rho_g = max(P, 1.0) / (RV * temperatura)
    rho = 1.0 / (n / rho_g + (1.0 - n) / rho_m)
    phi = n * rho / rho_g
    return rho, phi


def poiseuille(radio, profundidad=PROFUNDIDAD, sobrepresion=SOBREPRESION,
               viscosidad=MU, densidad=RHO_M):
    """Velocidad si el fundido no suelta gas y la viscosidad es constante.

    u = R^2 * dP / (8 mu L). La sobrepresion es la que empuja: el peso del
    magma y el de la corteza casi se cancelan.
    """
    u = radio ** 2 * sobrepresion / (8.0 * viscosidad * profundidad)
    area = math.pi * radio ** 2
    return {
        "velocidad_m_s": u,
        "caudal_m3_s": u * area,
        "tasa_kg_s": densidad * u * area,
    }


def altura_columna(caudal_m3_s):
    """Altura de columna en km. Mastin et al. (2009), caudal de roca densa."""
    if caudal_m3_s <= 0.0:
        return 0.0
    return K_ALTURA * caudal_m3_s ** B_ALTURA


def _integrar(radio, vin, profundidad, sobrepresion, agua, viscosidad,
              rho_m, temperatura, pasos):
    """Sube por el conducto. Devuelve el estado en la boca o donde se corta."""
    dz = profundidad / pasos
    P = RHO_CORTEZA * G * profundidad + sobrepresion
    q = rho_m * vin
    z = -profundidad
    phi = 0.0
    rho = rho_m
    z_frag = None
    for _ in range(pasos):
        rho, phi = mezcla(P, agua, temperatura, rho_m)
        mu = viscosidad if phi < PHI_FRAG else 1.0
        dPdz = -rho * G - 8.0 * mu * q / (max(rho, 1.0) * radio ** 2)
        P = P + dPdz * dz
        z = z + dz
        if z_frag is None and phi >= PHI_FRAG:
            z_frag = z
        if P <= PATM:
            break
    u = q / max(rho, 1.0)
    return {
        "z": z,
        "P": P,
        "phi": phi,
        "u": u,
        "q": q,
        "z_frag": z_frag,
    }


def correr(radio=RADIO, profundidad=PROFUNDIDAD, sobrepresion=SOBREPRESION,
           agua=AGUA, viscosidad=MU, volumen_m3=VOLUMEN, rho_m=RHO_M,
           temperatura=T_MAGMA, pasos=4000):
    """Conducto estacionario. Busca la velocidad de entrada.

    El criterio es que el magma fragmente junto a la boca (phi = 0.75).
    Mas lento y el gas no alcanza a fragmentar. Mas rapido y fragmenta
    hundido en el conducto. El caudal que sale es el de la entrada,
    rho * u_in * area. La altura de la columna se calcula con ese caudal.
    """
    lo, hi = 0.05, 400.0
    mejor = None
    for _ in range(46):
        vin = 0.5 * (lo + hi)
        sal = _integrar(
            radio, vin, profundidad, sobrepresion, agua, viscosidad,
            rho_m, temperatura, pasos,
        )
        mejor = sal
        mejor["vin"] = vin
        zf = sal["z_frag"]
        if zf is not None and zf < -30.0:
            hi = vin
        else:
            lo = vin
    area = math.pi * radio ** 2
    caudal = mejor["vin"] * area
    return {
        "radio_m": radio,
        "profundidad_m": profundidad,
        "vin_m_s": mejor["vin"],
        "velocidad_boca_m_s": mejor["u"],
        "presion_boca_Pa": mejor["P"],
        "phi_boca": mejor["phi"],
        "z_fragmentacion_m": mejor["z_frag"],
        "caudal_m3_s": caudal,
        "tasa_kg_s": rho_m * caudal,
        "altura_km": altura_columna(caudal),
        "duracion_s": volumen_m3 / caudal if caudal > 0.0 else math.inf,
        "duracion_h": (volumen_m3 / caudal) / 3600.0 if caudal > 0.0 else math.inf,
        "volumen_m3": volumen_m3,
    }


def torricelli(dP, densidad=RHO_ANALOGO):
    """Velocidad de un chorro inercial. No depende del radio."""
    return math.sqrt(2.0 * dP / densidad)


def analogico(radio_m, volumen_ml, dP=3.0e5, densidad=RHO_ANALOGO):
    """Botella: Torricelli para la velocidad y caudal = u * area del orificio."""
    u = torricelli(dP, densidad)
    area = math.pi * radio_m ** 2
    volumen = volumen_ml * 1.0e-6
    caudal = u * area
    return {
        "radio_m": radio_m,
        "volumen_ml": volumen_ml,
        "velocidad_m_s": u,
        "caudal_m3_s": caudal,
        "altura_chorro_m": u ** 2 / (2.0 * G),
        "duracion_s": volumen / caudal if caudal > 0.0 else math.inf,
    }


def _linea(texto=""):
    print(texto, flush=True)


def tabla_volcan(radios=(8.0, 16.0, 32.0), volumenes_km3=(0.05, 0.15, 0.30)):
    """Tres radios y tres volumenes, en la proporcion 1 : 2 : 4 y 1 : 3 : 6."""
    _linea("Conducto viscoso. Profundidad 7 km, dP 5 MPa, agua 4 %, mu 3000 Pa s.")
    _linea(f"{'R [m]':>8} {'u_in':>8} {'u_boca':>8} {'Q [m3/s]':>12} {'H [km]':>8}")
    base = None
    for radio in radios:
        s = correr(radio=radio)
        if base is None:
            base = s
        _linea(
            f"{radio:8.1f} {s['vin_m_s']:8.2f} {s['velocidad_boca_m_s']:8.1f} "
            f"{s['caudal_m3_s']:12.1f} {s['altura_km']:8.2f}"
        )
    _linea("")
    _linea("Duracion al vaciar un volumen, con R = 16 m.")
    _linea(f"{'V [km3]':>10} {'duracion [h]':>14}")
    for v in volumenes_km3:
        s = correr(radio=16.0, volumen_m3=v * 1.0e9)
        _linea(f"{v:10.2f} {s['duracion_h']:14.2f}")
    return base


def tabla_analogico():
    """Mismos factores que la auxiliar: volumen x1, x3, x6 y orificio x1, x2."""
    _linea("Analogo inercial. dP = 3 bar, densidad del agua.")
    _linea(f"{'V [ml]':>8} {'R [mm]':>8} {'u [m/s]':>8} {'H [m]':>8} {'t [s]':>8}")
    for volumen, radio_mm in ((500, 4), (500, 8), (1500, 4), (1500, 8), (3000, 4)):
        s = analogico(radio_mm / 1000.0, volumen)
        _linea(
            f"{volumen:8.0f} {radio_mm:8.0f} {s['velocidad_m_s']:8.1f} "
            f"{s['altura_chorro_m']:8.1f} {s['duracion_s']:8.2f}"
        )


def graficar(ruta="conducto_actividad.png"):
    """Guarda H(R) y la duracion contra el volumen. Si no hay matplotlib, no falla."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        _linea("matplotlib no esta instalado: la tabla igual quedo impresa.")
        return None
    radios = [4.0 + i * (40.0 - 4.0) / 24.0 for i in range(25)]
    alturas = [correr(radio=r)["altura_km"] for r in radios]
    volumenes = [0.02e9 * (1.5 ** i) for i in range(10)]
    horas = [correr(radio=16.0, volumen_m3=v)["duracion_h"] for v in volumenes]
    fig, ejes = plt.subplots(1, 2, figsize=(8.2, 3.6))
    ejes[0].plot(radios, alturas, color="C0", lw=1.8)
    ejes[0].set_xlabel("radio del conducto [m]")
    ejes[0].set_ylabel("altura de columna [km]")
    ejes[1].plot([v / 1.0e9 for v in volumenes], horas, color="C1", lw=1.8)
    ejes[1].set_xlabel("volumen de magma [km3]")
    ejes[1].set_ylabel("duracion [h]")
    fig.tight_layout()
    fig.savefig(ruta, dpi=140)
    _linea(f"figura {ruta}")
    return ruta


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        radio = float(sys.argv[1])
        profundidad = float(sys.argv[2]) if len(sys.argv) > 2 else PROFUNDIDAD
        s = correr(radio=radio, profundidad=profundidad)
        for clave, valor in s.items():
            _linea(f"{clave}: {valor}")
    else:
        tabla_volcan()
        _linea("")
        tabla_analogico()
        _linea("")
        simple = poiseuille(RADIO)
        _linea(
            "Sin gas, el mismo radio da "
            f"u = {simple['velocidad_m_s']:.3f} m/s. "
            "Con exsolucion la entrada queda en el valor de la tabla."
        )
        graficar()
