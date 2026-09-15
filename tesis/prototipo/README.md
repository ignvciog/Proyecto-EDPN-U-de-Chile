# Prototipo numérico

Laboratorio matemático para la propuesta de tesis (`../PROPUESTA.md`). Contiene
un **modelo reducido** del conducto volcánico con geometría variable $R(z)$ y
tres experimentos que producen los resultados preliminares citados en la
propuesta y en la presentación.

El prototipo **no reemplaza** a los solvers del repositorio raíz
(`RIconduitex5_5.py`, `RIconduit1D_transient.py`, `RIconduit2D_FD.py`). Es una
simplificación deliberada del mismo sistema físico, hecha para que el operador
directo sea barato, suave (diferenciable) y abierto a geometría arbitraria —las
tres cosas que un estudio de problema inverso necesita y que los solvers
completos, tal como están hoy, no ofrecen.

Las constitutivas se importan del propio repositorio (`density.py`,
`viscosity.py`, `fvrel.py`), de modo que el modelo reducido y el completo
comparten la misma física de materiales.

---

## Cómo ejecutar

```bash
pip install numpy scipy matplotlib     # únicas dependencias
cd tesis/prototipo
python3 exp1_no_unicidad_geometria.py
python3 exp2_el_tiempo_rompe_la_degeneracion.py
python3 exp3_identificabilidad.py
```

Cada script imprime sus resultados en la consola y escribe figuras en
`figuras/`. El experimento 3 cachea la tabla del operador directo en
`cache/tabla_directa.npz`; para recomputarla desde cero, borrar ese archivo.

---

## `modelo_reducido.py` — el operador directo

Resuelve el tramo **viscoso** del conducto, $z\in[H,z_f]$, con

| Borde | Condición |
|---|---|
| $z=H$ (base, techo de la cámara) | $P(H)=P_{lit}(H)+\Delta P$ |
| $z=z_f$ (nivel de fragmentación) | $P(z_f)=P_{frag}$ |

$P_{frag}$ es la presión a la cual la fracción volumétrica de gas en equilibrio
alcanza el umbral $\phi_{frag}$: es un dato termodinámico, no un parámetro
ajustable una vez fijado $\phi_{frag}$. Sobre $z_f$ el flujo es una dispersión
gas–piroclastos que el modelo reducido no resuelve. Cortar el dominio ahí evita
las velocidades de salida no físicas que produce la teoría de lubricación
aplicada a la región fragmentada.

**Cierre termodinámico** (exsolución en equilibrio, isotermo):

$$
c_d(P)=\min(c_0,\,C_1P^\beta),\qquad
n(P)=\frac{c_0-c_d}{1-c_d},\qquad
\rho_g=\frac{P}{R_vT},\qquad
\frac1\rho=\frac{n}{\rho_g}+\frac{1-n}{\rho_m}.
$$

El `min` se reemplaza por una versión suavizada (`_softmin`) para que el
operador directo sea diferenciable, condición necesaria para métodos basados en
gradiente.

**Momentum** (lubricación compresible, flujo de Poiseuille con radio variable):

$$
\frac{dP}{dz}=-\rho(P)g-\frac{8\,\mu(P)\,\mathrm{MER}}{\pi\,\rho(P)\,R(z)^4}.
$$

**Estructura del módulo**

| Grupo | Funciones |
|---|---|
| Parámetros | `Parametros` (dataclass; incluye tabla de interpolación de $\mu_{fundido}$ para acelerar) |
| Termodinámica | `agua_disuelta`, `fraccion_gas_masica`, `rho_fundido`, `densidades`, `rho_de_P`, `drho_dP`, `presion_de_fragmentacion`, `viscosidad` |
| Geometría | `Geometria`, `cilindro`, `ensanchamiento`, `constriccion`, `dos_tramos`, `campana` |
| Estacionario | `presion_estacionaria`, `resolver_estacionario` (tiro sobre MER), `escalar_a_MER`, `MER_analitico_incompresible` |
| Transiente | `malla` (graduada hacia $z_f$), `capacidad_de_almacenamiento`, `resolver_transiente` (volúmenes finitos + BDF) |
| Cámara | `curva_caracteristica`, `presion_minima_de_erupcion`, `drenaje_camara`, `tiempo_de_respuesta` |

Caso base: parámetros de Calbuco 2015 (`../../calbuco2015d.py`). Con
$R=16$ m, $\Delta P=5$ MPa, 4 wt% H₂O, 970 °C y 25 % de cristales se obtiene
$\mathrm{MER}\approx3.0\times10^7$ kg/s, del mismo orden que los
$1.4$–$2.7\times10^7$ kg/s observados.

---

## `exp1_no_unicidad_geometria.py` — la geometría no es única

**(a) Teorema de degeneración, verificado numéricamente.** Apagando la
flotabilidad, el problema estacionario se integra exactamente:

$$
\int_{P_{frag}}^{P_{cam}}\frac{\rho(P)}{\mu(P)}\,dP=\frac{8\,\mathrm{MER}}{\pi}J_4[R],
\qquad J_4[R]=\int_H^{z_f}R(z)^{-4}dz .
$$

El lado izquierdo no depende de la geometría, luego $\mathrm{MER}\cdot J_4[R]$
es invariante: **el MER estacionario ve a $R(\cdot)$ sólo a través de un
escalar.** El script comprueba que ese producto es constante a $<10^{-6}$
relativo entre cinco formas distintas.

**(b) Familia no-única en el modelo completo.** Con flotabilidad el invariante
ya no es exactamente $J_4$, pero la no-unicidad persiste. Ajustando un único
factor de escala por forma, seis geometrías cualitativamente distintas —con
volúmenes que difieren en **+317 %**— producen el mismo MER con error relativo
$\sim10^{-7}$.

**(c) Núcleo de resistencia.** La densidad $w(z)\propto\mu(z)/R(z)^4$ muestra
dónde "mira" el dato: 50 % de la resistencia está en los últimos 830 m, 90 % en
los últimos 44 m, 99 % en los últimos 2 m. **El conducto profundo es invisible
para el MER de superficie.**

Figuras: `figuras/exp1_no_unicidad_geometria.png`, `figuras/exp1_kernel_resistencia.png`.

---

## `exp2_el_tiempo_rompe_la_degeneracion.py` — el tiempo aporta información

1. **Curva de indeterminación.** El conjunto de nivel
   $\{(R,\Delta P):\mathrm{MER}_{ss}=\mathrm{MER}_{obs}\}$ es una curva: el dato
   estacionario fija **una** combinación de los dos parámetros, no los dos.
2. **Curva característica.** Esos conductos comparten un punto de
   $\mathrm{MER}_{ss}(P_{cam})$ pero no su **pendiente**
   $G=d\,\mathrm{MER}/dP_{cam}\approx\pi R^4/(8\mu L)$, que depende de $R$ de
   forma independiente del dato estacionario.
3. **Dinámica.** Drenando una cámara elástica de capacitancia $C$ según
   $C\,\dot P_{cam}=-\mathrm{MER}(P_{cam})$, el tiempo característico es
   $\tau=C/G$. Cuatro conductos con MER inicial idéntico
   ($3.01\times10^7$ kg/s) dan $\tau$ entre 5.8 h y 19.5 h y masas emitidas en
   6 h que difieren en 29 %.

Figuras: `figuras/exp2_degeneracion_RdP.png`, `figuras/exp2_curvas_MER_tiempo.png`.

---

## `exp3_identificabilidad.py` — cuánto se gana, cuantificado

Operador directo restringido,
$\theta=(\log R,\log\Delta P)\mapsto d=(\log\mathrm{MER},\log G)$, donde el
primer observable resume el dato estacionario y el segundo, la serie de tiempo.
Se calculan:

1. Matriz de sensibilidad $J=\partial\log d/\partial\log\theta$ en el punto
   nominal, su SVD, número de condición y dirección del casi-núcleo.
2. Matriz de información de Fisher con ruido realista
   ($\sigma_{\log\mathrm{MER}}=\ln2/2$, es decir un factor 2). Diagnóstico
   **local**, aquí insuficiente por la fuerte no linealidad en $\Delta P$.
3. **Perfiles de verosimilitud**
   $\chi^2_{perfil}(\theta_i)=\min_{\theta_j,\,j\neq i}\chi^2(\theta)$
   (Raue et al., 2009), que son el diagnóstico global correcto.

Resultado principal:

| Datos | intervalo 95 % de $R$ | intervalo 95 % de $\Delta P$ |
|---|---|---|
| sólo estacionario | $[8,19]$ m (abierto) | $[0.2,250]$ MPa (todo el rango) |
| estacionario + serie de tiempo | $[13,18]$ m | $[0.25,44]$ MPa (sólo cota superior) |

La serie de tiempo identifica el radio y acota por arriba la sobrepresión, pero
no la determina: para eso hacen falta datos de otra naturaleza (deformación,
gas). Eso motiva la inversión conjunta del capítulo 4 de la propuesta.

Figura: `figuras/exp3_identificabilidad.png`.

---

## Limitaciones conocidas

Conviene tenerlas presentes al leer los resultados:

- **Una sola fase con velocidad única.** No hay deslizamiento gas–fundido ni
  escape de gas por permeabilidad; el modelo completo sí los tiene.
- **Exsolución y cristalización en equilibrio.** La Spina et al. (2017) muestran
  que las tasas finitas controlan la duración de la erupción.
- **Isotermo.** Sin liberación de calor latente ni enfriamiento en la pared.
- **La región fragmentada no se resuelve**, se usa como condición de borde.
- **Cámara elástica de un solo parámetro** (capacitancia $C$), sin recarga.

Ninguna de estas limitaciones afecta las conclusiones cualitativas sobre
no-unicidad e identificabilidad: la degeneración de §(a) es una propiedad
estructural del balance de momentum viscoso, presente en cualquier modelo que
contenga un término de fricción tipo Poiseuille. Levantarlas es el contenido de
la etapa 0 del plan de trabajo (punto E1 de la propuesta).
