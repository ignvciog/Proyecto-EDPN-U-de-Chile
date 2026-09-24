# Prototipo numérico

Laboratorio matemático para la propuesta de tesis (`../PROPUESTA.md`). Contiene
un **recorte** monofásico del conducto volcánico con geometría variable $R(z)$ y
experimentos que producen los resultados preliminares citados en la
propuesta y en la presentación.

El prototipo **no reemplaza** al modelo bifásico del repositorio raíz
(`RIconduitex5_5.py`, `RIconduit1D_transient.py`, `RIconduit2D_FD.py`). Es un
recorte deliberado del mismo sistema (una sola velocidad, exsolución en
equilibrio), hecho para que el operador sea barato, suave (diferenciable) y
abierto a geometría arbitraria. El modelo que se invierte en la tesis sigue
siendo el bifásico.

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
python3 exp4_cuando_hace_falta_el_transiente.py
```

Cada script imprime sus resultados en la consola y escribe figuras en
`figuras/`. Tiempos de referencia en una máquina modesta: experimento 1,
$\sim50$ s; experimento 2, $\sim7$ min; experimento 3, $\sim13$ min la primera
vez; experimento 4, $\sim1.5$ min. El experimento 3 cachea la tabla del operador
directo en `cache/tabla_directa.npz` y en corridas posteriores tarda segundos;
para recomputarla desde cero, borrar ese archivo.

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

## `exp4_cuando_hace_falta_el_transiente.py` — ¿cuándo importa el término $\partial_t$?

``Agregar el tiempo'' significa dos cosas distintas que conviene no confundir:

- **(A)** el **observable** depende del tiempo: $\mathrm{MER}(t)$ es una función
  y no un escalar. Esto casi siempre ayuda al problema inverso (experimento 2);
- **(B)** el **conducto** está fuera de equilibrio con sus condiciones de borde,
  de modo que los términos $\partial_t$ de sus ecuaciones importan.

(A) no implica (B). El criterio es una comparación de escalas, tipo número de
Deborah: $\mathrm{De}=\tau_{conducto}/\tau_{forzamiento}$.

**Escalas medidas** para el caso base de Calbuco (no supuestas: se obtienen del
propio modelo):

| Escala | Valor | Qué es |
|---|---|---|
| $\tau_{hidr}$ | 8 s | relajación de $\mathrm{MER}$ tras un escalón de $-2\%$ en $P_{cam}$ |
| $\tau_{resid}$ | 6.0 min | masa en el conducto dividida por $\mathrm{MER}$ |
| $\tau_{cam}$ | 5.83 h | drenaje de la cámara, $C/G$, con $V=50$ km³ |

De ahí $\mathrm{De}=5.7\times10^{-4}$: el conducto está **esclavizado** a la
cámara. Dos comprobaciones:

1. Alimentando el solver transiente completo con la historia $P_{cam}(t)$ del
   drenaje cuasi-estacionario, el error en $\mathrm{MER}(t)$ es de
   **0.08 % máximo** sobre toda la erupción. La curva cuasi-estacionaria del
   experimento 2 queda justificada a posteriori.
2. Barriendo el tiempo de forzamiento con rampas exponenciales, el error del
   cuasi-estacionario alcanza 1 % cuando $\mathrm{De}\approx0.09$, es decir
   para forzamientos más rápidos que $\sim90$ s.

**Advertencia importante.** El $\tau$ medido aquí es sólo la relajación
**hidráulica** (difusión de presión). El modelo reducido supone exsolución y
cristalización en equilibrio, de modo que **no contiene** las escalas lentas que
la literatura identifica como las que de verdad vuelven transiente una erupción:

| Proceso | Escala | Referencia |
|---|---|---|
| exsolución en desequilibrio | $10^1$–$10^3$ s | La Spina et al. (2017) |
| cristalización de microlitos | $10^4$–$10^6$ s | Melnik & Sparks (1999) |
| escape de gas por permeabilidad | $10^3$–$10^5$ s | Wong & Segall (2019) |

Con $\tau\sim10^4$ s el veredicto cambia por completo: hasta una fase
sub-pliniana de 6 h pasa a $\mathrm{De}\approx0.5$. Y en un ciclo de domo el
período **no es un forzamiento externo**: lo fija la propia cinética, de modo
que $\mathrm{De}\approx1$ por construcción — el ciclo existe *porque* hay
retardo. Ahí el término transiente no es un refinamiento, es el mecanismo.

Figura: `figuras/exp4_escalas_de_tiempo.png`.

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
