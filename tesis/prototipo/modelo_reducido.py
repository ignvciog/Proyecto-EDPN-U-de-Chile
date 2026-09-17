"""
modelo_reducido.py
==================
Modelo reducido del conducto volcanico con geometria variable R(z), pensado
como *laboratorio matematico* para el estudio del problema inverso.

Relacion con el codigo del repositorio
-------------------------------------
Este modulo NO reemplaza a `RIconduitex5_5.py` / `RIconduit1D_transient.py` /
`RIconduit2D_FD.py`. Es una version deliberadamente simplificada del mismo
sistema fisico, construida para que el operador directo sea:

  (i)   barato   (una solucion estacionaria en ~0.1 s),
  (ii)  suave    (sin switches de regimen ni bisecciones anidadas),
  (iii) abierto a geometria variable R(z),
  (iv)  consistente con el modelo completo en el regimen viscoso
        pre-fragmentacion.

Las constitutivas (densidad del fundido, viscosidad de Giordano et al. 2008,
viscosidad relativa por cristales) se importan del propio repositorio, de modo
que el modelo reducido y el completo comparten la misma fisica de materiales.

Dominio y cierre
----------------
Se resuelve el tramo VISCOSO del conducto: z en [H, z_f], con

    z = H    base del conducto (techo de la camara),   P(H) = P_lit(H) + dP
    z = z_f  nivel de fragmentacion,                   P(z_f) = P_frag

donde P_frag es la presion a la cual la fraccion volumetrica de gas en
equilibrio alcanza el umbral phi_frag (dato termodinamico, no ajustable una vez
fijado phi_frag). Por encima de z_f el flujo es una dispersion gas-piroclastos
que el modelo reducido no resuelve.

Ecuacion de estado (equilibrio de exsolucion, isotermo)

    c_d(P) = min(c_0, C_1 P^beta)                 agua disuelta
    n(P)   = (c_0 - c_d) / (1 - c_d)              gas exsuelto (frac. masica)
    rho_g  = P / (R_v T)                          gas ideal
    1/rho  = n/rho_g + (1-n)/rho_m                mezcla
    phi    = n rho / rho_g                        fraccion volumetrica de gas

Momentum (lubricacion / Poiseuille generalizado, inercia despreciable)

    w(z,t) = rho A u = - k(z) rho (dP/dz + rho g),     k(z) = pi R(z)^4 / (8 mu)

Conservacion de masa

    A drho/dt + dw/dz = 0

que combinadas dan la ecuacion parabolica no lineal para P(z,t)

    A(z) rho'(P) dP/dt = d/dz [ k(z) rho(P) ( dP/dz + rho(P) g ) ]

Estacionario: w = const = -MER, es decir

    dP/dz = - rho g - MER / (k(z) rho)

Resultado analitico clave (limite incompresible, mu constante)
--------------------------------------------------------------
Si rho = rho_m y mu son constantes la integracion es exacta:

    P_cam - P_frag - rho_m g L = (8 mu / (pi rho_m)) * MER * J_4[R],   L = z_f - H

con el UNICO funcional de la geometria

    J_4[R] = \\int_H^{z_f} R(z)^{-4} dz .

=> El caudal masico estacionario depende de R(z) solo a traves de J_4[R].
   Dos geometrias con el mismo J_4 son indistinguibles con datos estacionarios.
   La no-unicidad respecto de la geometria es EXACTA y de codimension 1 en un
   espacio de funciones (el nucleo de la derivada de J_4 es
   {dR : \\int R^{-5} dR dz = 0}, de dimension infinita).

El problema transiente, en cambio, involucra el termino de almacenamiento
A(z) rho'(P), es decir el VOLUMEN del conducto, y rompe esa degeneracion.

Autor: propuesta de tesis (geologia + ingenieria matematica)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# Constitutivas del repositorio (dos niveles arriba)
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from density import density      # noqa: E402  Bottinga & Weill (1970)
from viscosity import viscosity  # noqa: E402  Giordano et al. (2008)
from fvrel import fvrel          # noqa: E402  viscosidad relativa por cristales


# ---------------------------------------------------------------------------
# Composicion (Calbuco 2015, identica a calbuco2015d.py)
# ---------------------------------------------------------------------------
COMPOSICION = dict(
    sio2=62.72, tio2=1.0, al2o3=17.0, feo=5.44, mno=0.15, mgo=2.19,
    cao=5.22, na2o=4.44, k2o=1.2, p2o5=0.1, f2o=0.1,
)


@dataclass
class Parametros:
    """Parametros fisicos del modelo reducido."""

    # --- dominio ---
    H: float = -7000.0          # base del conducto [m]
    z_f: float = -1000.0        # nivel de fragmentacion [m]
    R0: float = 16.0            # radio de referencia [m]

    # --- condiciones del reservorio ---
    dP: float = 5.0e6           # sobrepresion en la base [Pa]
    T: float = 1243.15          # temperatura [K]
    c0: float = 0.040           # agua total (fraccion masica)
    xi: float = 0.25            # fraccion de cristales

    # --- fragmentacion ---
    phi_frag: float = 0.75      # umbral de fraccion de gas

    # --- solubilidad y gas ---
    C1: float = 4.11e-6
    beta: float = 0.5
    Rv: float = 461.11          # constante del vapor de agua [J/kg/K]

    # --- compresibilidad del fundido ---
    # Imprescindible para el problema transiente: sin ella el tramo profundo
    # (libre de gas) no tiene capacidad de almacenamiento y es invisible.
    K_m: float = 1.0e10         # modulo de compresibilidad del fundido [Pa]

    # --- entorno ---
    g: float = 9.81
    rho_crust: float = 2600.0
    Patm: float = 1.01325e5

    # --- reologia ---
    modelo_cristales: str = "er52"
    phimax_cristales: float = 0.6
    ar1: float = 4.0
    ar2: float = 8.0
    mu_max: float = 1.0e12      # tope numerico de viscosidad [Pa s]

    # --- regularizacion del min(.) de solubilidad ---
    eps_soft: float = 1.0e-3    # suavizado relativo del min (0 => min duro)

    # --- diagnostico: apagar la flotabilidad en el momentum ---
    # Solo para verificar numericamente el teorema de degeneracion J_4, que es
    # exacto en el limite dominado por friccion. No usar en simulaciones fisicas.
    incluir_flotabilidad: bool = True

    # --- tamano de la tabla de viscosidad del fundido ---
    n_tabla: int = 300

    # --- cache interno ---
    rho_m: float = field(default=np.nan, init=False)
    P_ref: float = field(default=np.nan, init=False)
    _logP_tabla: np.ndarray = field(default=None, init=False, repr=False)
    _logmu_tabla: np.ndarray = field(default=None, init=False, repr=False)
    _theta_x: float = field(default=np.nan, init=False, repr=False)
    _P_frag: float = field(default=np.nan, init=False, repr=False)

    def __post_init__(self) -> None:
        self.actualizar_constitutivas()

    def actualizar_constitutivas(self) -> None:
        """
        Recalcula densidad del fundido, factor reologico por cristales y la
        tabla log(mu_fundido) vs log(P).

        Debe llamarse tras modificar T, c_0, xi, H, dP o el modelo reologico.
        """
        Tc = self.T - 273.15
        c = COMPOSICION
        # Referencia de la EOS del fundido: se fija una sola vez, de modo que
        # variar la sobrepresion dP (p.ej. al drenar la camara) no desplace la
        # curva rho_m(P).
        self.P_ref = self.P_camara
        self.rho_m = density(
            c["sio2"], c["tio2"], c["al2o3"], c["feo"], c["mgo"], c["cao"],
            c["na2o"], c["k2o"], 100.0 * self.c0, Tc, self.P_ref * 1e-6,
        )
        self._theta_x = fvrel(self.modelo_cristales, self.xi, self.xi,
                              self.ar1, self.ar2, self.phimax_cristales, 1.0)

        # mu_fundido depende de P solo a traves del agua disuelta c_d(P)
        P = np.logspace(2.0, np.log10(3.0 * self.P_camara + 1.0e8), self.n_tabla)
        cd = np.atleast_1d(agua_disuelta(P, self))
        mu = np.array([
            viscosity(c["sio2"], c["tio2"], c["al2o3"], c["feo"], c["mno"],
                      c["mgo"], c["cao"], c["na2o"], c["k2o"], c["p2o5"],
                      100.0 * cdi, c["f2o"], Tc)
            for cdi in cd
        ])
        self._logP_tabla = np.log(P)
        self._logmu_tabla = np.log(mu)
        self._P_frag = np.nan  # invalida el cache

    @property
    def P_camara(self) -> float:
        """Presion en la base del conducto: litostatica + sobrepresion."""
        return self.rho_crust * self.g * abs(self.H) + self.dP

    @property
    def longitud(self) -> float:
        return self.z_f - self.H

    @property
    def P_frag(self) -> float:
        """Presion a la cual phi(P) = phi_frag (cierre superior del dominio)."""
        if not np.isfinite(self._P_frag):
            self._P_frag = presion_de_fragmentacion(self)
        return self._P_frag


# ---------------------------------------------------------------------------
# Ecuacion de estado y reologia
# ---------------------------------------------------------------------------
def _softmin(a: np.ndarray, b: float, eps: float) -> np.ndarray:
    """min(a, b) suavizado. eps <= 0 devuelve el min duro."""
    if eps <= 0.0:
        return np.minimum(a, b)
    return 0.5 * (a + b - np.sqrt((a - b) ** 2 + eps ** 2))


def agua_disuelta(P, p: Parametros):
    """c_d(P) = min(c_0, C_1 P^beta), suavizado."""
    P = np.maximum(np.asarray(P, dtype=float), 1.0)
    return _softmin(p.C1 * P ** p.beta, p.c0, p.eps_soft * p.c0)


def fraccion_gas_masica(P, p: Parametros):
    """n(P): fraccion masica de gas exsuelto respecto de la mezcla."""
    cd = agua_disuelta(P, p)
    return np.maximum((p.c0 - cd) / (1.0 - cd), 0.0)


def rho_fundido(P, p: Parametros):
    """
    Densidad del fundido con compresibilidad lineal:

        rho_m(P) = rho_m(P_ref) * (1 + (P - P_ref) / K_m)

    El valor de referencia rho_m(P_ref) proviene de Bottinga & Weill (1970).
    """
    P = np.asarray(P, dtype=float)
    return p.rho_m * (1.0 + (P - p.P_ref) / p.K_m)


def densidades(P, p: Parametros):
    """Devuelve (rho_mezcla, phi, rho_gas, n)."""
    P = np.maximum(np.asarray(P, dtype=float), 1.0)
    n = fraccion_gas_masica(P, p)
    rho_g = P / (p.Rv * p.T)
    v = n / rho_g + (1.0 - n) / rho_fundido(P, p)
    rho = 1.0 / v
    phi = n * rho / rho_g
    return rho, phi, rho_g, n


def rho_de_P(P, p: Parametros):
    return densidades(P, p)[0]


def drho_dP(P, p: Parametros):
    """Compresibilidad drho/dP por diferencias centradas relativas."""
    P = np.maximum(np.asarray(P, dtype=float), 1.0)
    h = np.maximum(1.0e-6 * P, 1.0)
    return (rho_de_P(P + h, p) - rho_de_P(P - h, p)) / (2.0 * h)


def presion_de_fragmentacion(p: Parametros) -> float:
    """Resuelve phi(P) = phi_frag. Unica por monotonia de phi en P."""
    def f(P):
        return densidades(np.array([P]), p)[1][0] - p.phi_frag
    return float(brentq(f, 1.0e3, 0.9 * p.P_camara, xtol=1.0, rtol=1e-12))


def viscosidad(P, p: Parametros):
    """
    Viscosidad efectiva mu(P) = mu_fundido(c_d) * theta_cristales * theta_burbujas.

    - mu_fundido: Giordano et al. (2008); depende del agua DISUELTA, de modo que
      la desgasificacion endurece el magma al ascender. Se evalua por
      interpolacion de la tabla log-log construida en `Parametros`.
    - theta_cristales: fvrel (Einstein-Roscoe por defecto).
    - theta_burbujas: Einstein-Taylor (1 - phi)^(-1) (limite de burbujas
      rigidas, capilaridad dominante).
    """
    P = np.atleast_1d(np.maximum(np.asarray(P, dtype=float), 1.0))
    phi = np.atleast_1d(densidades(P, p)[1])
    mu = np.exp(np.interp(np.log(P), p._logP_tabla, p._logmu_tabla))
    theta_b = 1.0 / np.maximum(1.0 - np.minimum(phi, 0.95), 0.05)
    return np.minimum(mu * p._theta_x * theta_b, p.mu_max)


# ---------------------------------------------------------------------------
# Geometria R(z)
# ---------------------------------------------------------------------------
class Geometria:
    """
    Radio del conducto R(z) sobre z en [H, z_f].

    Se construye a partir de una *forma* adimensional g(s) con
    s = (z - H)/(z_f - H) en [0, 1], y un factor de escala. El metodo
    `escalar_a_J4` ajusta ese factor para imponer un valor prescrito de

        J_4 = \\int_H^{z_f} R(z)^{-4} dz,

    el unico funcional de la geometria visible por el caudal estacionario en el
    limite incompresible.
    """

    def __init__(self, forma: Callable[[np.ndarray], np.ndarray],
                 H: float, z_f: float, escala: float = 1.0, nombre: str = ""):
        self.forma = forma
        self.H = float(H)
        self.z_f = float(z_f)
        self.escala = float(escala)
        self.nombre = nombre

    def R(self, z):
        s = (np.asarray(z, dtype=float) - self.H) / (self.z_f - self.H)
        return self.escala * np.asarray(self.forma(np.clip(s, 0.0, 1.0)),
                                        dtype=float)

    def J4(self, n: int = 8001) -> float:
        z = np.linspace(self.H, self.z_f, n)
        return float(np.trapezoid(self.R(z) ** -4, z))

    def volumen(self, n: int = 8001) -> float:
        z = np.linspace(self.H, self.z_f, n)
        return float(np.trapezoid(np.pi * self.R(z) ** 2, z))

    def escalar_a_J4(self, J4_objetivo: float, n: int = 8001) -> "Geometria":
        """Reescala R -> c R para que J_4 sea exactamente J4_objetivo."""
        z = np.linspace(self.H, self.z_f, n)
        s = (z - self.H) / (self.z_f - self.H)
        J4_unit = float(np.trapezoid(np.asarray(self.forma(s), dtype=float) ** -4, z))
        self.escala = (J4_unit / J4_objetivo) ** 0.25
        return self

    def copia(self) -> "Geometria":
        return Geometria(self.forma, self.H, self.z_f, self.escala, self.nombre)


def cilindro(p: Parametros, R0: float | None = None) -> Geometria:
    return Geometria(lambda s: np.ones_like(s), p.H, p.z_f,
                     p.R0 if R0 is None else R0, "cilindro")


def ensanchamiento(p: Parametros, razon: float = 2.0, s0: float = 0.65,
                   ancho: float = 0.12) -> Geometria:
    """Conducto que se ensancha hacia arriba (tipo NC3, Aravena et al. 2017)."""
    def f(s):
        return 1.0 + (razon - 1.0) * 0.5 * (1.0 + np.tanh((s - s0) / ancho))
    return Geometria(f, p.H, p.z_f, p.R0, f"ensanchamiento superior x{razon:g}")


def constriccion(p: Parametros, profundidad: float = 0.30, s0: float = 0.5,
                 ancho: float = 0.10) -> Geometria:
    """Conducto con una constriccion localizada (tapon)."""
    def f(s):
        return 1.0 - profundidad * np.exp(-((s - s0) / ancho) ** 2)
    return Geometria(f, p.H, p.z_f, p.R0, f"constriccion -{profundidad:.0%}")


def dos_tramos(p: Parametros, razon: float = 1.6, s0: float = 0.45,
               ancho: float = 0.05) -> Geometria:
    """Dos tramos cilindricos coaxiales unidos por una transicion (tipo NC2)."""
    def f(s):
        return 1.0 + (razon - 1.0) * 0.5 * (1.0 + np.tanh((s - s0) / ancho))
    return Geometria(f, p.H, p.z_f, p.R0, f"dos tramos x{razon:g}")


def campana(p: Parametros, amplitud: float = 0.45, s0: float = 0.30,
            ancho: float = 0.15) -> Geometria:
    """Ensanchamiento localizado en profundidad (sill / camara intermedia)."""
    def f(s):
        return 1.0 + amplitud * np.exp(-((s - s0) / ancho) ** 2)
    return Geometria(f, p.H, p.z_f, p.R0, f"ensanche profundo +{amplitud:.0%}")


# ---------------------------------------------------------------------------
# Problema directo ESTACIONARIO
# ---------------------------------------------------------------------------
def _dPdz(z, P, MER, geo: Geometria, p: Parametros):
    P = np.maximum(np.atleast_1d(P), 1.0e3)
    rho = rho_de_P(P, p)
    mu = viscosidad(P, p)
    R = np.atleast_1d(geo.R(z))
    k = np.pi * R ** 4 / (8.0 * mu)
    peso = rho * p.g if p.incluir_flotabilidad else 0.0
    return -peso - MER / (k * rho)


def presion_estacionaria(MER: float, geo: Geometria, p: Parametros,
                         n_eval: int | None = 400, z_eval=None,
                         rtol: float = 1e-10):
    """
    Integra dP/dz desde z = H (P = P_camara) hasta z = z_f, dado MER.

    `n_eval=None` y `z_eval=None` devuelve solo el estado final (modo rapido
    para el metodo de tiro). `z_eval` permite pedir la solucion exactamente
    sobre una malla dada, lo que evita interpolar a traves de la capa limite
    de presion que se forma bajo el nivel de fragmentacion.
    """
    def P_muy_baja(z, P, *args):
        return P[0] - 0.02 * p.P_frag
    P_muy_baja.terminal = True
    P_muy_baja.direction = -1

    if z_eval is None and n_eval is not None:
        z_eval = np.linspace(p.H, p.z_f, n_eval)

    # Sin sobrepresion suficiente la columna nunca alcanza P_frag: se devuelve
    # un tramo degenerado que el residuo del tiro interpreta como "no llega".
    fallo = (np.array([p.H]), np.array([p.P_camara]))
    if p.P_camara <= p.P_frag:
        return fallo
    try:
        sol = solve_ivp(
            _dPdz, (p.H, p.z_f), [p.P_camara], args=(MER, geo, p),
            t_eval=z_eval, events=P_muy_baja, method="LSODA",
            rtol=rtol, atol=1e-2, max_step=p.longitud / 50.0,
        )
    except Exception:
        return fallo
    if sol.status < 0 or sol.t.size == 0:
        return fallo
    return sol.t, sol.y[0]


def resolver_estacionario(geo: Geometria, p: Parametros,
                          MER_min: float = 1.0e-3, MER_max: float = 1.0e13,
                          n_eval: int = 600, MER_guess: float | None = None):
    """
    Problema directo: encuentra MER tal que P(z_f) = P_frag (metodo de tiro).

    El residuo P(z_f; MER) - P_frag es monotono decreciente en MER, de modo que
    basta expandir un intervalo alrededor de una estimacion inicial (la formula
    incompresible) en lugar de biseccionar sobre 16 ordenes de magnitud.

    Devuelve un dict con MER [kg/s], perfiles y diagnosticos.
    """
    P_frag = p.P_frag

    def residuo(log_MER):
        z, P = presion_estacionaria(float(np.exp(log_MER)), geo, p,
                                    n_eval=None, rtol=1e-8)
        if abs(z[-1] - p.z_f) > 1.0:          # la integracion aborto por P baja
            return -1.0e12
        return P[-1] - P_frag

    if MER_guess is None or not np.isfinite(MER_guess) or MER_guess <= 0:
        MER_guess = MER_analitico_incompresible(geo, p)
    centro = np.log(np.clip(MER_guess, MER_min, MER_max))

    a, b, paso = centro, centro, np.log(4.0)
    fa = fb = residuo(centro)
    for _ in range(30):                        # expansion del intervalo
        if fa > 0 and fb < 0:
            break
        if fb >= 0:
            b = min(b + paso, np.log(MER_max))
            fb = residuo(b)
        if fa <= 0:
            a = max(a - paso, np.log(MER_min))
            fa = residuo(a)
        if a <= np.log(MER_min) and b >= np.log(MER_max):
            break
    if fa < 0:
        raise RuntimeError("Sin solucion: P(z_f) < P_frag aun con MER minimo "
                           "(la columna no alcanza a fragmentar).")
    if fb > 0:
        raise RuntimeError("Sin solucion: P(z_f) > P_frag aun con MER maximo.")

    log_MER = brentq(residuo, a, b, xtol=1e-7, rtol=1e-12, maxiter=200)
    MER = float(np.exp(log_MER))

    z, P = presion_estacionaria(MER, geo, p, n_eval=n_eval)
    rho, phi, rho_g, n = densidades(P, p)
    R = geo.R(z)
    u = MER / (rho * np.pi * R ** 2)

    return dict(MER=MER, z=z, P=P, phi=phi, rho=rho, u=u, R=R,
                mu=viscosidad(P, p), n=n,
                J4=geo.J4(), volumen=geo.volumen(),
                u_frag=float(u[-1]), phi_frag=float(phi[-1]),
                tiempo_transito=float(np.trapezoid(1.0 / u, z)))


def escalar_a_MER(geo: Geometria, p: Parametros, MER_objetivo: float,
                  rango: tuple = (0.2, 5.0)) -> Geometria:
    """
    Reescala R -> c R para que el MER estacionario del modelo COMPLETO iguale
    a `MER_objetivo`.

    Genera, para cada forma g(s), un miembro de la familia de geometrias que
    son exactamente indistinguibles con un unico dato estacionario. Es el
    enunciado exacto de la no-unicidad para el modelo no lineal (no solo para
    el limite incompresible, donde el invariante es J_4[R]).
    """
    escala0 = geo.escala

    def residuo(log_c):
        geo.escala = escala0 * float(np.exp(log_c))
        return np.log(resolver_estacionario(geo, p, n_eval=2)["MER"] /
                      MER_objetivo)

    a, b = np.log(rango[0]), np.log(rango[1])
    log_c = brentq(residuo, a, b, xtol=1e-8, rtol=1e-10, maxiter=100)
    geo.escala = escala0 * float(np.exp(log_c))
    return geo


def MER_analitico_incompresible(geo: Geometria, p: Parametros,
                                mu: float | None = None) -> float:
    """
    MER en el limite incompresible con viscosidad constante:

        MER = pi rho_m (P_cam - P_frag - rho_m g L) / (8 mu J_4[R])

    Depende de la geometria UNICAMENTE via J_4[R]. Referencia analitica de la
    degeneracion; `mu` por defecto es la viscosidad en el nivel de fragmentacion
    (que domina la resistencia).
    """
    if mu is None:
        mu = float(viscosidad(np.array([p.P_frag]), p)[0])
    motor = p.P_camara - p.P_frag - p.rho_m * p.g * p.longitud
    return np.pi * p.rho_m * motor / (8.0 * mu * geo.J4())


# ---------------------------------------------------------------------------
# Problema directo TRANSIENTE
# ---------------------------------------------------------------------------
def malla(p: Parametros, N: int = 161, dz_min: float = 2.0) -> np.ndarray:
    """
    Malla geometrica en z, refinada hacia el nivel de fragmentacion.

    Bajo z_f la solucion estacionaria desarrolla una capa limite de presion de
    algunas decenas de metros (la viscosidad y la compresibilidad varian en
    ordenes de magnitud en el kilometro superior), de modo que una malla
    uniforme la resolveria mal. Se impone el tamano de la celda superior
    (`dz_min`) y se busca la razon geometrica que completa la longitud total.
    """
    L = p.longitud
    n = N - 1
    if dz_min * n >= L:                      # no hace falta graduar
        return np.linspace(p.H, p.z_f, N)
    r = brentq(lambda r: dz_min * (r ** n - 1.0) / (r - 1.0) - L,
               1.0 + 1e-12, 2.0, xtol=1e-14)
    dz = dz_min * r ** np.arange(n)[::-1]    # grande abajo, pequena arriba
    z = np.concatenate([[p.H], p.H + np.cumsum(dz)])
    # El extremo superior debe ser EXACTAMENTE z_f: ahi se impone P = P_frag, y
    # un residuo de redondeo deja el nodo fuera del intervalo de integracion de
    # `presion_estacionaria`, que entonces falla en silencio.
    z[0], z[-1] = p.H, p.z_f
    return z


def capacidad_de_almacenamiento(geo: Geometria, p: Parametros,
                                P: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Densidad de capacidad de almacenamiento c(z) = A(z) drho/dP [kg/Pa/m].

    Su integral es la masa que el conducto absorbe por Pascal de cambio de
    presion, y es la cantidad que el problema transiente "ve" de la geometria.
    """
    return np.pi * geo.R(z) ** 2 * drho_dP(P, p)


def resolver_transiente(geo: Geometria, p: Parametros,
                        P_camara_de_t: Callable[[float], float],
                        t_final: float, N: int = 161, n_salidas: int = 240,
                        dz_min: float = 2.0, P_inicial=None,
                        rtol: float = 1e-7, atol: float = 1e1):
    """
    Resuelve por metodo de lineas (volumenes finitos en malla graduada + BDF)

        A(z) rho'(P) dP/dt = d/dz [ k(z) rho(P) ( dP/dz + rho(P) g ) ]

    con P(H,t) = P_camara_de_t(t) y P(z_f,t) = P_frag.

    Observable: MER(t) = - k rho (dP/dz + rho g) |_{z=z_f}  [kg/s].
    """
    z = malla(p, N, dz_min)
    dz_cara = np.diff(z)                       # espaciamiento entre nodos
    dz_celda = np.empty(N)                     # medida de la celda de control
    dz_celda[1:-1] = 0.5 * (z[2:] - z[:-2])
    dz_celda[0] = 0.5 * (z[1] - z[0])
    dz_celda[-1] = 0.5 * (z[-1] - z[-2])
    A = np.pi * geo.R(z) ** 2
    R_cara = geo.R(0.5 * (z[:-1] + z[1:]))
    P_frag = p.P_frag

    if P_inicial is None:
        MER_ss = resolver_estacionario(geo, p, n_eval=2)["MER"]
        _, P_inicial = presion_estacionaria(MER_ss, geo, p, z_eval=z)
    P_inicial = np.asarray(P_inicial, dtype=float)

    def flujo_G(P_full):
        """G = k rho (dP/dz + rho g) en las caras. MER = -G."""
        P_cara = 0.5 * (P_full[:-1] + P_full[1:])
        rho_c = rho_de_P(P_cara, p)
        k_c = np.pi * R_cara ** 4 / (8.0 * viscosidad(P_cara, p))
        peso = rho_c * p.g if p.incluir_flotabilidad else 0.0
        return k_c * rho_c * (np.diff(P_full) / dz_cara + peso)

    def ensamblar(t, y):
        P_full = np.empty(N)
        P_full[0] = P_camara_de_t(t)
        P_full[-1] = P_frag
        P_full[1:-1] = y
        return P_full

    def rhs(t, y):
        P_full = ensamblar(t, y)
        G = flujo_G(P_full)
        return ((G[1:] - G[:-1]) /
                (dz_celda[1:-1] * A[1:-1] * drho_dP(P_full[1:-1], p)))

    n_int = N - 2
    idx = np.arange(n_int)
    sparsity = np.zeros((n_int, n_int))
    sparsity[idx, idx] = 1
    sparsity[idx[:-1], idx[:-1] + 1] = 1
    sparsity[idx[1:], idx[1:] - 1] = 1

    sol = solve_ivp(rhs, (0.0, t_final), P_inicial[1:-1],
                    t_eval=np.linspace(0.0, t_final, n_salidas),
                    method="BDF", jac_sparsity=sparsity, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"Transiente fallo: {sol.message}")

    MER = np.empty(sol.t.size)
    P_hist = np.empty((sol.t.size, N))
    for j, tj in enumerate(sol.t):
        P_full = ensamblar(tj, sol.y[:, j])
        P_hist[j] = P_full
        MER[j] = -flujo_G(P_full)[-1]

    almacen = capacidad_de_almacenamiento(geo, p, P_inicial, z)
    return dict(t=sol.t, MER=MER, z=z, P_hist=P_hist,
                J4=geo.J4(), volumen=geo.volumen(),
                capacidad=float(np.trapezoid(almacen, z)),
                capacidad_z=almacen)


# ---------------------------------------------------------------------------
# Acoplamiento camara - conducto
# ---------------------------------------------------------------------------
def curva_caracteristica(geo: Geometria, p: Parametros,
                         P_camara_vals: np.ndarray) -> np.ndarray:
    """
    Curva caracteristica del conducto MER_ss(P_cam): caudal masico estacionario
    en funcion de la presion en la base.

    Es el objeto que realmente controla la dinamica acoplada camara-conducto.
    Un unico dato estacionario fija solo UN PUNTO de esta curva; la serie de
    tiempo de una erupcion recorre un TRAMO de ella, y por lo tanto informa
    tambien sobre su PENDIENTE G = dMER/dP_cam, que es una funcion distinta de
    los parametros.
    """
    MER = np.empty(len(P_camara_vals))
    dP_orig = p.dP
    guess = None
    try:
        for i, Pc in enumerate(P_camara_vals):
            p.dP = Pc - p.rho_crust * p.g * abs(p.H)
            try:
                # arranque en caliente: la curva es suave en P_cam
                MER[i] = resolver_estacionario(geo, p, n_eval=2,
                                               MER_guess=guess)["MER"]
                guess = MER[i]
            except RuntimeError:
                MER[i] = 0.0          # el conducto no alcanza a fragmentar
    finally:
        p.dP = dP_orig
    return MER


def presion_minima_de_erupcion(geo: Geometria, p: Parametros) -> float:
    """
    Presion de camara por debajo de la cual el conducto deja de erupcionar
    (la columna ya no alcanza P_frag en z_f). Cota inferior para la curva
    caracteristica y punto final natural del drenaje.
    """
    dP_orig = p.dP

    def hay_erupcion(Pc):
        p.dP = Pc - p.rho_crust * p.g * abs(p.H)
        try:
            resolver_estacionario(geo, p, n_eval=2)
            return True
        except RuntimeError:
            return False

    try:
        lo = p.P_frag + p.rho_m * p.g * p.longitud
        hi = p.P_camara
        if not hay_erupcion(hi):
            return hi
        for _ in range(22):
            mid = 0.5 * (lo + hi)
            if hay_erupcion(mid):
                hi = mid
            else:
                lo = mid
        return hi
    finally:
        p.dP = dP_orig


def drenaje_camara(geo: Geometria, p: Parametros, V_camara: float,
                   K_camara: float = 1.0e10, Q_entrada: float = 0.0,
                   t_final: float = 6.0 * 3600.0, n_puntos: int = 400,
                   n_curva: int = 60):
    """
    Erupcion como drenaje de una camara elastica a traves del conducto.

    Balance de masa de la camara (magma y paredes compresibles):

        C_cam dP_cam/dt = Q_entrada - MER(P_cam),
        C_cam = rho_m V_camara (1/K_m + 1/K_camara)        [kg/Pa]

    El conducto se trata como cuasi-estatico, aproximacion valida porque su
    tiempo de relajacion propio (decenas de segundos) es varios ordenes de
    magnitud menor que C_cam / (dMER/dP_cam) (horas). La funcion
    `resolver_transiente` permite verificar esa separacion de escalas.

    Devuelve t, MER(t), P_cam(t), masa acumulada y el tiempo caracteristico.
    """
    C_cam = p.rho_m * V_camara * (1.0 / p.K_m + 1.0 / K_camara)
    P0 = p.P_camara

    # Tabula MER_ss(P_cam) una sola vez e interpola (el tiro es lo caro).
    P_lo = presion_minima_de_erupcion(geo, p)
    P_grid = np.linspace(P_lo, P0 * 1.02, n_curva)
    MER_grid = curva_caracteristica(geo, p, P_grid)

    def MER_de_P(Pc):
        if Pc <= P_lo:
            return 0.0
        return float(np.interp(Pc, P_grid, MER_grid))

    def rhs(t, y):
        return [(Q_entrada - MER_de_P(y[0])) / C_cam]

    sol = solve_ivp(rhs, (0.0, t_final), [P0],
                    t_eval=np.linspace(0.0, t_final, n_puntos),
                    method="LSODA", rtol=1e-8, atol=1.0)
    P_cam = sol.y[0]
    MER = np.array([MER_de_P(Pc) for Pc in P_cam])
    masa = np.concatenate([[0.0], np.cumsum(0.5 * (MER[1:] + MER[:-1]) *
                                            np.diff(sol.t))])

    # pendiente de la curva caracteristica en el punto de partida
    G = float(np.gradient(MER_grid, P_grid)[np.argmin(np.abs(P_grid - P0))])

    return dict(t=sol.t, MER=MER, P_camara=P_cam, masa=masa,
                C_camara=C_cam, G=G, tau=C_cam / G if G > 0 else np.nan,
                P_grid=P_grid, MER_grid=MER_grid)


def tiempo_de_respuesta(t, MER) -> float:
    """
    Tiempo caracteristico: instante en que MER(t) recorre el 63.2% (1 - 1/e)
    del camino entre su valor inicial y el final.
    """
    t, MER = np.asarray(t), np.asarray(MER)
    M0, Mf = MER[0], MER[-1]
    if abs(Mf - M0) < 1e-10 * max(abs(M0), 1.0):
        return np.nan
    objetivo = M0 + (1.0 - np.exp(-1.0)) * (Mf - M0)
    cruce = np.where((MER[:-1] - objetivo) * (MER[1:] - objetivo) <= 0)[0]
    if cruce.size == 0:
        return np.nan
    i = cruce[0]
    w = (objetivo - MER[i]) / (MER[i + 1] - MER[i])
    return float(t[i] + w * (t[i + 1] - t[i]))


__all__ = [
    "Parametros", "Geometria", "COMPOSICION",
    "cilindro", "ensanchamiento", "constriccion", "dos_tramos", "campana",
    "agua_disuelta", "fraccion_gas_masica", "densidades", "rho_de_P",
    "rho_fundido", "drho_dP", "viscosidad", "presion_de_fragmentacion",
    "resolver_estacionario", "presion_estacionaria", "escalar_a_MER",
    "MER_analitico_incompresible",
    "malla", "capacidad_de_almacenamiento",
    "resolver_transiente", "tiempo_de_respuesta",
    "curva_caracteristica", "drenaje_camara", "presion_minima_de_erupcion",
]
