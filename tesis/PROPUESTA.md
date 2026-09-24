# Propuesta de tesis — Problemas inversos en el ascenso de magma por un conducto volcánico

**Doble titulación: Geología + Ingeniería Matemática**
Continuación del proyecto de conducto volcánico 1D/2D (Calbuco 2015) de este repositorio.

Nota corta para la reunión: [`nota_problema_inverso.pdf`](nota_problema_inverso.pdf).
Matemática del modelo bifásico (el que se invierte):
[`matematica_modelo_bifasico.pdf`](matematica_modelo_bifasico.pdf).

---

## 0. Resumen en una página

El modelo directo ya está construido: dado un conjunto de parámetros
$\theta$ (radio del conducto, sobrepresión de cámara, contenido de agua,
temperatura, cristales, umbral de fragmentación), el código de este repositorio
calcula la solución $(P,\phi,u_m,u_g)(z)$ —y ahora también $(z,t)$ y $(r,z)$— y,
con ella, observables de superficie como la tasa de descarga másica.

La tesis propone recorrer la flecha en sentido contrario: **dados observables de
una erupción, ¿qué se puede afirmar sobre el estado del magma en profundidad y
sobre la geometría del conducto, y con qué grado de certeza?**

La tesis **no** es "hacer una inversión y reportar números". La tesis es
caracterizar **qué es recuperable y qué no**, porque en este problema la
no-unicidad no es un accidente numérico: es estructural, y se puede demostrar.

Tres resultados preliminares, ya obtenidos con el prototipo de
`tesis/prototipo/`, definen el esqueleto de la propuesta:

1. **Teorema de degeneración geométrica.** En el régimen viscoso dominado por
   fricción, la tasa de descarga estacionaria depende del radio del conducto
   $R(z)$ **únicamente** a través del escalar
   $J_4[R]=\int_H^{z_f} R(z)^{-4}\,dz$. La preimagen de un dato es entonces una
   hipersuperficie de codimensión 1 en un espacio de funciones. Verificado
   numéricamente a precisión de máquina.

2. **La no-unicidad es enorme, no marginal.** En el *recorte* de lubricación
   (con flotabilidad, exsolución en equilibrio, viscosidad de Giordano y
   cristales), seis geometrías de conducto cualitativamente distintas, con
   volúmenes que difieren en **+317 %**, producen la misma tasa de descarga
   másica con un error relativo de $10^{-7}$. El 90 % de la caída de presión
   ocurre en el tramo más somero: **el conducto profundo es esencialmente
   invisible** para los observables de superficie. Esto está demostrado en el
   recorte; **la tesis pregunta si sobrevive en el modelo bifásico**.

3. **El tiempo aporta información nueva, pero no toda.** Cuatro conductos con
   tasa de descarga inicial idéntica ($3.01\times10^{7}$ kg/s) drenan una misma
   cámara con tiempos característicos entre 5.8 h y 19.5 h, y emiten masas que
   difieren en 29 % en seis horas. Un análisis de identificabilidad por perfil
   de verosimilitud muestra que agregar la serie de tiempo reduce el intervalo
   de confianza del radio de $[8,19]$ m a $[13,18]$ m, pero deja la
   sobrepresión con **sólo una cota superior**.

Ese último punto es la tesis en una frase: **agregar dimensiones al modelo
directo (tiempo, radio) es precisamente lo que permite recuperar parámetros que
el modelo estacionario 1D no puede distinguir** — y cuantificar cuánto se gana
es un resultado matemático y geológico publicable.

---

## 1. Qué hay construido hoy

| Módulo | Formulación | Cierre | Estado |
|---|---|---|---|
| `RIconduitex5_5.py` | DAE de 6 variables $(P,\phi,N_d,x,u_m,u_g)$, momentum de dos fluidos con arrastre, coalescencia, permeabilidad, cristalización cinética | Tiro sobre `vinicial`; salida estrangulada o $P=P_{atm}$ con fragmentación | Funciona; requiere `scikits.odes` (IDA) |
| `RIconduit1D_transient.py` | 3 EDP de transporte $(\phi,N_d,x)$ + presión cuasi-estática; IMEX (RK4 + relajación exacta) | Bisección sobre $q$ para $P_{vent}=P_{atm}$ | Funciona; sólo numpy |
| `RIconduit2D_FD.py` | Perfil radial $u_m(r)$ por diferencias finitas en cada nivel $z$ (lubricación), Thomas + Picard; marcha RK4 en $z$ | Tiro sobre `vinicial` con $P_{exit}\approx P_{atm}$ | Funciona; sólo numpy |

Tres observaciones técnicas que condicionan el problema inverso y que conviene
poner sobre la mesa con ambos profesores:

- **Los tres modelos están desacoplados.** El 2D no usa el transiente ni
  viceversa; comparten sólo `calbuco2015d.py` y las constitutivas. Un modelo
  inverso serio necesita **un** operador directo, no tres.
- **El operador directo actual no es diferenciable.** Hay `max(0,·)`, cambios de
  régimen `n_eq` con paradas por cruce de $\phi$, criterios de estrangulamiento
  y bisecciones anidadas. Eso descarta métodos basados en gradiente hasta que se
  regularice.
- **Hay inconsistencias entre versiones** (por ejemplo, $\phi_{crit}$ dinámico
  vía número de capilaridad en el estacionario frente a un valor fijo 0.75 en el
  transiente; el término `_Fmg` de permeabilidad está definido pero no acoplado
  en el transiente). Antes de invertir hay que fijar **una** física.

Ninguna de estas tres cosas es un defecto del trabajo previo: son exactamente
las tareas que convierten un código de simulación en un código de inversión, y
constituyen el primer capítulo natural de la tesis.

**El modelo de la tesis es el bifásico.** `RIconduitex5_5.py` (dos velocidades,
arrastre, coalescencia, cristalización) no se reemplaza. El prototipo de
`tesis/prototipo/` es un *recorte* monofásico del mismo sistema, hecho para
demostrar el teorema de degeneración y para explorar el inverso a bajo costo.
Comparte constitutivas y parámetros de Calbuco; no es un modelo nuevo. El
capítulo de identificabilidad del operador bifásico es el trabajo central.

---

## 2. Marco matemático

### 2.1 El operador directo

Sea $\mathcal{F}$ el operador que lleva parámetros a observables,

$$
\mathcal{F}:\ \Theta \longrightarrow \mathcal{D},\qquad
\theta \longmapsto d = \mathcal{F}(\theta),
$$

donde $\Theta$ puede ser de dimensión finita (radio, sobrepresión, $c_0$, $T$,
$\xi$, $\phi_{crit}$, $\tau_c$, constantes de permeabilidad) o de dimensión
infinita (la geometría $R(\cdot)$, un perfil de volátiles $c_0(\cdot)$, un
coeficiente de fricción de pared $\beta(\cdot)$, una historia de presión de
cámara $P_{cam}(\cdot)$).

El dato es $d^{obs} = \mathcal{F}(\theta^\dagger) + \varepsilon$ con ruido
$\varepsilon$. Las tres preguntas clásicas son:

1. **Unicidad / identificabilidad.** ¿Es $\mathcal{F}$ inyectivo? Si no,
   caracterizar $\mathcal{F}^{-1}(d)$.
2. **Estabilidad.** ¿Es $\mathcal{F}^{-1}$ continuo? ¿Con qué módulo?
3. **Reconstrucción.** Algoritmo, regularización, cuantificación de
   incertidumbre.

En este problema la respuesta a (1) es **no**, y se puede demostrar. Ése es el
núcleo de la propuesta.

### 2.2 Formulación determinista

$$
\theta_\alpha \in \arg\min_{\theta\in\Theta}\
\underbrace{\tfrac12\|\mathcal{F}(\theta)-d^{obs}\|^2_{\Sigma^{-1}}}_{\text{desajuste}}
\;+\;\underbrace{\alpha\,\mathcal{R}(\theta)}_{\text{regularización}} ,
$$

con $\mathcal{R}$ de Tikhonov ($\|\theta-\theta_{prior}\|^2$), de variación
total (para geometrías con saltos, tipo NC2 de Aravena) o de Sobolev
($\|R'\|^2_{L^2}$, para geometrías suaves). Elección de $\alpha$ por principio
de discrepancia de Morozov o por validación cruzada generalizada.

### 2.3 Formulación bayesiana

$$
\pi(\theta\mid d^{obs}) \;\propto\; \underbrace{\exp\!\left(-\tfrac12
\|\mathcal{F}(\theta)-d^{obs}\|^2_{\Sigma^{-1}}\right)}_{\text{verosimilitud}}
\;\cdot\;\underbrace{\pi_0(\theta)}_{\text{previa}} .
$$

Esta formulación es la adecuada aquí por dos razones geológicas concretas:

- La petrología entrega **previas informativas reales**: inclusiones vítreas
  acotan $c_0$, los equilibrios de fases acotan $T$ y la profundidad del
  reservorio, el conteo de cristales acota $\xi$.
- Los observables tienen **incertidumbres enormes y asimétricas**: la tasa de
  descarga másica de una erupción se conoce típicamente salvo un factor 2. Una
  estimación puntual sin barras de error no significa nada.

Es exactamente el camino de Anderson & Segall (2013), Wong et al. (2017) y
Wong & Segall (2019, 2020) para Mount St. Helens. La oportunidad es aplicarlo a
un caso chileno, explosivo (no efusivo), con un modelo de conducto que incluye
dos fases con deslizamiento y, en esta tesis, geometría variable.

### 2.4 Por qué "agregar dimensiones" es lo que hace posible invertir

Conviene formular esto explícitamente porque es el hilo que une el trabajo ya
hecho con el trabajo propuesto:

- El modelo **estacionario 1D** produce un puñado de escalares por simulación.
  Con pocos datos y muchos parámetros, el problema inverso está masivamente
  subdeterminado.
- El modelo **transiente** produce una *función del tiempo*. Cada instante es
  una restricción adicional, y el sistema recorre un tramo de la curva
  característica del conducto $\mathrm{MER}_{ss}(P_{cam})$, informando sobre su
  **pendiente**, que es una función distinta de los parámetros (experimento 2).
- El modelo **2D** produce un perfil radial. La reología dependiente de la tasa
  de corte deja una firma en la distribución radial de deformación, que se
  relaciona con la sismicidad de borde de conducto y con las texturas de los
  piroclastos del borde frente al centro.

En lenguaje de problemas inversos: cada dimensión añadida **aumenta el rango
efectivo del operador linealizado** y por tanto reduce la dimensión del espacio
nulo. Cuantificar esa reducción, con la matriz de información de Fisher y con
perfiles de verosimilitud, es un resultado concreto y demostrable.

---

## 3. Resultados preliminares

Todo lo de esta sección es reproducible con `tesis/prototipo/` (ver
`tesis/prototipo/README.md`). El prototipo es un **recorte** de lubricación
compresible del modelo bifásico: misma constitutivas
(`density.py`, `viscosity.py`, `fvrel.py`), una sola velocidad, geometría
$R(z)$ arbitraria. Sirve para demostrar teoremas; no sustituye a
`RIconduitex5_5.py`.

### 3.1 Teorema de degeneración geométrica

En el régimen viscoso dominado por fricción, el balance de momentum de
lubricación con conservación de masa da

$$
\int_{P_{frag}}^{P_{cam}} \frac{\rho(P)}{\mu(P)}\,dP
\;=\; \frac{8\,\mathrm{MER}}{\pi}\,J_4[R],
\qquad
J_4[R]=\int_H^{z_f} R(z)^{-4}\,dz .
$$

**El lado izquierdo no depende de la geometría.** Por lo tanto
$\mathrm{MER}\cdot J_4[R]$ es invariante: dos conductos con el mismo $J_4$ son
**exactamente** indistinguibles a partir de la tasa de descarga estacionaria.

La derivada de Fréchet es $DJ_4[R]\,\delta R = -4\int R^{-5}\delta R\,dz$, de
modo que el espacio nulo del operador linealizado es

$$
\mathcal{N} = \Big\{\delta R \ :\ \int_H^{z_f} R(z)^{-5}\,\delta R(z)\,dz = 0\Big\},
$$

un subespacio **cerrado de codimensión 1**: infinitas direcciones de perturbación
de la geometría son invisibles. La verificación numérica (experimento 1,
apagando la flotabilidad) da $\mathrm{MER}\cdot J_4$ constante con desviación
$<10^{-6}$ relativo entre cinco formas distintas, y la fórmula analítica
reproduce el $\mathrm{MER}$ numérico con error $+0.000\,\%$.

### 3.2 La no-unicidad persiste con flotabilidad (aún en el recorte)

Con flotabilidad el invariante ya no es exactamente $J_4$, pero la no-unicidad
persiste y es igual de grave. Ajustando un único factor de escala por forma:

| Geometría | $R(H)$ | $R(z_f)$ | Volumen | $\Delta V$ | $J_4$ | error en MER |
|---|---|---|---|---|---|---|
| cilindro | 16.0 | 16.0 | $4.83\times10^6$ | — | $9.16\times10^{-2}$ | $+0.00000\%$ |
| ensanche basal ×3 | 44.4 | 14.8 | $1.29\times10^7$ | $+166\%$ | $5.86\times10^{-2}$ | $-0.00001\%$ |
| ensanche basal ×4 | 51.4 | 13.6 | $2.01\times10^7$ | $+317\%$ | $4.24\times10^{-2}$ | $-0.00002\%$ |
| ensanchamiento superior ×2 | 11.2 | 22.4 | $4.74\times10^6$ | $-1.8\%$ | $2.18\times10^{-1}$ | $+0.00000\%$ |
| dos tramos ×2 (tipo NC2) | 10.7 | 21.5 | $5.38\times10^6$ | $+11.5\%$ | $2.22\times10^{-1}$ | $+0.00001\%$ |
| constricción profunda $-35\%$ | 16.8 | 16.9 | $4.80\times10^6$ | $-0.6\%$ | $1.16\times10^{-1}$ | $+0.00001\%$ |

Nótese que $J_4$ varía en un factor 5 entre estas geometrías y aun así el MER es
el mismo: con flotabilidad, el invariante es otro funcional, pero sigue siendo
**un solo funcional escalar**.

El núcleo de sensibilidad explica por qué. La densidad de resistencia
$w(z)\propto \mu(z)/R(z)^4$ se concentra brutalmente hacia arriba, porque la
desgasificación endurece el magma varios órdenes de magnitud:

- 50 % de la resistencia está en los últimos 830 m;
- 90 % en los últimos 44 m;
- 99 % en los últimos 2 m.

**Consecuencia geológica directa:** cualquier inferencia sobre el conducto
profundo hecha a partir de la tasa de descarga es, esencialmente, una inferencia
sobre la previa y no sobre el dato. Esto es una advertencia metodológica
publicable por sí sola.

### 3.3 El tiempo separa lo que el estacionario confunde

Para el par $(R,\Delta P)$, la matriz de sensibilidad en el punto nominal
($R=16$ m, $\Delta P=5$ MPa) es

$$
J=\frac{\partial(\log \mathrm{MER},\ \log G)}{\partial(\log R,\ \log \Delta P)}
=\begin{pmatrix} 4.000 & 0.195\\ 4.000 & 0.115\end{pmatrix},
\qquad G=\frac{d\,\mathrm{MER}}{d P_{cam}} .
$$

El $4.000$ es la ley de Poiseuille $\mathrm{MER}\propto R^4$ recuperada
numéricamente. El $0.195$ es pequeño porque la sobrepresión aporta sólo una
fracción del empuje total (el resto lo aporta la flotabilidad), y de ahí la ley
de compensación local $\Delta P \sim R^{-20.5}$: **variaciones enormes de
sobrepresión se compensan con variaciones minúsculas de radio.**

Drenando una cámara elástica de 50 km³, cuatro conductos con MER inicial
idéntico dan:

| $R$ [m] | $\Delta P$ [MPa] | $G$ [kg/s/Pa] | $\tau$ [h] | MER a 6 h | masa en 6 h |
|---|---|---|---|---|---|
| 10 | 92.5 | 0.353 | 19.5 | $2.23\times10^7$ | $5.62\times10^{11}$ |
| 12 | 45.3 | 0.617 | 11.1 | $1.84\times10^7$ | $5.10\times10^{11}$ |
| 14 | 20.6 | 0.905 | 7.6 | $1.55\times10^7$ | $4.66\times10^{11}$ |
| 16 | 5.0 | 1.174 | 5.8 | $1.36\times10^7$ | $4.34\times10^{11}$ |

Con un modelo de ruido realista ($\sigma_{\log \mathrm{MER}}=\ln 2/2$, es decir
un factor 2), los perfiles de verosimilitud dan:

| Datos | intervalo 95 % de $R$ | intervalo 95 % de $\Delta P$ |
|---|---|---|
| sólo estacionario | $[8, 19]$ m (abierto) | $[0.2, 250]$ MPa (todo el rango) |
| estacionario + serie de tiempo | $[13, 18]$ m | $[0.25, 44]$ MPa (sólo cota superior) |

**Conclusión honesta:** la serie de tiempo identifica el radio y acota por
arriba la sobrepresión, pero no la determina. Para determinarla hacen falta
datos de otra naturaleza (deformación, gas), lo que motiva la inversión
conjunta. Éste es un resultado, no un fracaso.

### 3.4 ¿Cuándo hace falta de verdad el término transiente?

Conviene separar dos cosas que la palabra "tiempo" confunde:

- **(A)** el **observable** depende del tiempo, $\mathrm{MER}(t)$ es una función
  y no un escalar;
- **(B)** el **conducto** está fuera de equilibrio con sus condiciones de borde,
  de modo que los términos $\partial_t$ de sus ecuaciones importan.

(A) no implica (B), y la distinción es decisiva para el problema inverso: si el
régimen es cuasi-estacionario, el operador directo es un tiro estacionario más
una EDO para la cámara —barato—, y **las tasas finitas son invisibles al dato**,
porque sólo entran a través de su límite de equilibrio.

El criterio es una comparación de escalas, tipo número de Deborah:
$\mathrm{De}=\tau_{conducto}/\tau_{forzamiento}$. El experimento 4 lo cuantifica
con escalas **medidas** del propio modelo, no supuestas:

| Escala | Valor | Qué es |
|---|---|---|
| $\tau_{hidr}$ | 8 s | relajación de $\mathrm{MER}$ tras un escalón de $-2\%$ en $P_{cam}$ |
| $\tau_{resid}$ | 6.0 min | masa en el conducto dividida por $\mathrm{MER}$ |
| $\tau_{cam}$ | 5.83 h | drenaje de la cámara, $C/G$, con $V=50$ km³ |

De ahí $\mathrm{De}=5.7\times10^{-4}$, y dos comprobaciones:

1. Alimentando el solver transiente completo con la historia $P_{cam}(t)$ del
   drenaje cuasi-estacionario, el error en $\mathrm{MER}(t)$ es de **0.08 %
   máximo** sobre toda la erupción: la aproximación del experimento 2 queda
   justificada a posteriori.
2. El error del cuasi-estacionario alcanza 1 % cuando $\mathrm{De}\approx0.09$,
   es decir para forzamientos más rápidos que $\sim90$ s.

**Pero el $\tau$ que domina no es el hidráulico.** El modelo reducido supone
exsolución y cristalización en equilibrio, de modo que no contiene las escalas
lentas que la literatura identifica como las que de verdad vuelven transiente
una erupción: exsolución en desequilibrio ($10^1$–$10^3$ s, La Spina et al.
2017), cristalización de microlitos ($10^4$–$10^6$ s, Melnik & Sparks 1999),
escape de gas por permeabilidad ($10^3$–$10^5$ s, Wong & Segall 2019). Con
$\tau\sim10^4$ s, hasta una fase sub-pliniana de 6 h pasa a
$\mathrm{De}\approx0.5$.

**Consecuencia directa para la tesis.** *Lo que se puede invertir depende del
régimen.* En régimen cuasi-estacionario los datos informan sobre geometría,
propiedades de la cámara y sobrepresión, y **nada** sobre cinética. Sólo cuando
$\mathrm{De}\gtrsim0.1$ los datos contienen información sobre las constantes de
tasa. Esto convierte la elección del caso de estudio (§9, pregunta 6) en una
decisión técnica y no de gusto, y justifica el punto E4 del catálogo: agregar
desequilibrio al modelo sólo tiene sentido si el caso elegido está en el régimen
en que ese desequilibrio es observable.

En los **ciclos de domo** el argumento se invierte de manera interesante: el
período no es un forzamiento externo, lo fija la propia cinética, de modo que
$\mathrm{De}\approx1$ por construcción. El ciclo existe *porque* hay retardo.
Ahí el término transiente no es un refinamiento: es el mecanismo, y por eso son
el caso donde la inversión de constantes de tasa está mejor planteada.

---

## 4. Catálogo de problemas inversos

Las cinco ideas conversadas con el profesor de geología aparecen marcadas con
(★). El resto son propuestas adicionales. La columna "dificultad" es relativa
al esfuerzo de implementación e infraestructura, no a tiempo de calendario.

### Familia A — Estimación de parámetros (dimensión finita)

| # | Qué se recupera | Datos | Herramienta | Dificultad |
|---|---|---|---|---|
| **A1** ★ | Sobrepresión de cámara $\Delta P$ y, con ella, $P(z)$ en profundidad | MER, duración, masa emitida, altura de columna | MCMC bayesiano con previas petrológicas | media |
| **A2** ★ | Contenido de volátiles $c_0$ y fracción de cristales $\xi$ | Igual que A1 + inclusiones vítreas | Igual + previa informativa fuerte | media |
| **A3** | Profundidad de fragmentación $z_f$ y umbral $\phi_{crit}$ | Densidad y porosidad de piroclastos, vesicularidad | Inversión conjunta con A1–A2 | media |
| **A4** | Constantes de la ley de permeabilidad $k(\phi)$ | Porosidad-permeabilidad medidas en laboratorio + MER | Estimación con datos de dos fuentes distintas | media-alta |

**Sobre A1 y "recuperar la presión en profundidad".** Conviene precisar el
enunciado con el profesor de geología. $P(z)$ **no** es un dato independiente:
una vez fijados los parámetros, el modelo directo la produce. Lo que de verdad
se invierte es la condición de borde $P(H)$ y, con ella, el perfil completo. La
sección 3.3 muestra que $\Delta P$ es justamente el parámetro peor determinado
por los datos de superficie, de modo que el enunciado defendible es:
*"¿qué cota superior sobre la sobrepresión de cámara imponen los observables de
la erupción, y qué dato adicional haría falta para acotarla por abajo?"*

### Familia B — Recuperación de funciones (dimensión infinita)

| # | Qué se recupera | Datos | Herramienta | Dificultad |
|---|---|---|---|---|
| **B1** ★★ | Geometría $R(z)$ | MER$(t)$, presión de salida, profundidad de fragmentación | Tikhonov / TV sobre $R$, con el teorema de §3.1 como resultado de no-unicidad | **alta — núcleo de la tesis** |
| **B2** | Coeficiente de fricción/deslizamiento en la pared $\beta(z)$ (condición de Navier $\mu\,\partial_r u = -\beta u$ en $r=R$) | MER$(t)$, sismicidad de borde de conducto, texturas de borde | Problema inverso de coeficiente de Robin: teoría de unicidad y estabilidad conocida | alta |
| **B3** | Perfil inicial de porosidad $\phi(z,0)$ | Serie temporal de porosidad/densidad de piroclastos en el vent | Observabilidad a lo largo de características de la ecuación de transporte | media |
| **B4** | Ley constitutiva $k(\phi)$ como función, no como parámetros | Igual que A4 | Identificación de coeficiente no lineal | alta |

**B2 merece un comentario**, porque es la extensión natural del código 2D que ya
existe. `RIconduit2D_FD.py` impone no-deslizamiento $u_m(R)=0$ y calcula
$\partial u_m/\partial r$ en la pared para obtener $F_{mw}$. Reemplazar esa
condición por una de Navier con coeficiente $\beta(z)$ desconocido convierte el
problema en la identificación de un coeficiente de Robin a partir de mediciones
de frontera, que tiene teoría de unicidad y estabilidad logarítmica bien
establecida. Geológicamente, $\beta(z)$ es la zona de cizalle del borde del
conducto, asociada a los sismos repetitivos ("drumbeats") de Mount St. Helens y
a la transición de flujo viscoso a deslizamiento friccional.

### Familia C — Recuperación de historias temporales

| # | Qué se recupera | Datos | Herramienta | Dificultad |
|---|---|---|---|---|
| **C1** | Historia de presión de cámara $P_{cam}(t)$ | MER$(t)$ de la columna eruptiva | Problema lateral (*sideways*) / deconvolución de Volterra, mal puesto, regularizado | media-alta |
| **C2** | Historia de recarga $Q_{in}(t)$ de la cámara | MER$(t)$ + deformación | Igual, con modelo cámara-conducto acoplado | alta |
| **C3** | Tasa de descompresión $dP/dt$ durante el ascenso | Densidad numérica de microlitos (MND) en piroclastos | Inversión de una ecuación de balance poblacional (nucleación-crecimiento) | media |

**C3 conecta directamente con la geología de terreno.** La densidad numérica de
microlitos es un "velocímetro" de descompresión ampliamente usado. Formularlo
como problema inverso explícito —con regularización y barras de error, en vez de
una calibración empírica— sería una contribución metodológica concreta a la
petrología, y es matemáticamente un problema inverso de coeficiente para una
ecuación de balance poblacional.

**C1 es atractivo por su limpieza matemática.** La ecuación parabólica no lineal
del prototipo,
$A(z)\rho'(P)\,\partial_t P = \partial_z\!\left[k(z)\rho(P)(\partial_z P+\rho g)\right]$,
con dato $\mathrm{MER}(t)$ en $z=z_f$ y la incógnita en $z=H$, es un problema de
frontera lateral para una ecuación parabólica: severamente mal puesto, con
teoría clásica de estabilidad condicional y métodos de regularización conocidos.
Es un tema de tesis de ingeniería matemática por derecho propio.

### Familia D — Teoría de unicidad y no-unicidad

| # | Resultado buscado | Herramienta | Dificultad |
|---|---|---|---|
| **D1** ★ | Teorema de degeneración $J_4$ y caracterización del espacio nulo (§3.1) | Cálculo variacional; ya demostrado y verificado | **hecho** |
| **D2** ★ | No-unicidad de la geometría desde datos sísmicos de resonancia | Analogía con la ecuación de Webster y teoría espectral inversa de Sturm–Liouville | alta, muy valiosa |
| **D3** | Multiplicidad de estados estacionarios (curva sigmoidal $p_{ch}$–$q$) como fuente adicional de no-unicidad | Teoría de bifurcaciones; Melnik & Sparks (1999, 2005), Kozono & Koyaguchi (2012) | media |

**D2 es la joya matemática de la propuesta.** Linealizando el flujo compresible
en un conducto de sección $A(z)$ se obtiene la ecuación de Webster,

$$
\frac{\partial^2 p}{\partial t^2}
= c^2\,\frac{1}{A}\frac{\partial}{\partial z}\!\left(A\,\frac{\partial p}{\partial z}\right),
$$

que con $p = A^{-1/2}\psi$ se transforma en un problema de Sturm–Liouville

$$
\psi'' + \left(\frac{\omega^2}{c^2} - V(z)\right)\psi = 0,
\qquad V = \frac{(A^{1/2})''}{A^{1/2}} .
$$

De aquí se siguen dos hechos clásicos con lectura volcanológica inmediata:

- **Una sola familia espectral no determina el potencial** (teorema de dos
  espectros de Borg). Es decir: **el espectro de resonancia de un conducto no
  determina su geometría.**
- Aun conociendo $V$ exactamente, la ecuación $(A^{1/2})'' = V\,A^{1/2}$ tiene un
  espacio de soluciones de dimensión 2. Existe entonces una **familia
  biparamétrica explícita de secciones $A(z)$ acústicamente indistinguibles**.

Como en volcanología se infieren rutinariamente dimensiones de conducto a partir
de eventos de largo período y tremor, exhibir esta familia con parámetros
realistas sería una advertencia metodológica fuerte y matemáticamente
irreprochable.

### Familia E — Metodología y realismo del modelo

| # | Qué se hace | Por qué importa | Dificultad |
|---|---|---|---|
| **E1** ★ | Unificar los tres solvers en **un** operador directo, con la fragmentación regularizada (transición suave en vez de switch) | Sin esto no hay gradientes ni inversión eficiente | media, **prerrequisito** |
| **E2** | Deducir e implementar el **método adjunto** para el sistema transiente | Gradiente a costo $O(1)$ solves en vez de $O(p)$; habilita $R(z)$ con cientos de incógnitas | alta |
| **E3** | **Selección de modelo** entre criterios de fragmentación ($\phi_{crit}$ fijo, $\phi_{crit}(\mathrm{Ca})$, criterio de tasa de deformación de Papale) | Convierte "¿cuál criterio es correcto?" en una pregunta con respuesta cuantitativa (factores de Bayes) | media |
| **E4** ★ | Modelo más realista: desequilibrio de exsolución y cristalización, desgasificación por permeabilidad con flujo de Darcy vertical y lateral, cámara acoplada | Wong et al. (2017) muestran que la permeabilidad controla el flujo másico erupcionado | alta |
| **E5** | **Emulador** (proceso gaussiano o red neuronal) del operador directo | Hace factible MCMC con $10^5$–$10^6$ evaluaciones | media |
| **E6** | **Diseño óptimo de experimentos**: qué observable reduce más la incertidumbre posterior | Respuesta directamente útil para el monitoreo volcánico en Chile | media |
| **E7** ★ | Geometría más compleja: $R(z)$ variable, dique en profundidad que pasa a cilindro somero, sección elíptica, cráter | Aravena et al. (2017, 2018) muestran efectos de primer orden en presión de salida y MER | media-alta |

---

## 5. Tesis recomendada

De todo lo anterior, la combinación con mejor relación entre riesgo, novedad y
coherencia entre ambas disciplinas es:

> **"¿Qué se puede recuperar de una erupción? Identificabilidad y no-unicidad en
> modelos de ascenso de magma por un conducto, con aplicación a Calbuco 2015."**

**Capítulo 1 — El operador directo (E1).** Unificación de los tres solvers;
regularización de la fragmentación; verificación de que la versión suave
reproduce la original; análisis de costo y de diferenciabilidad.

**Capítulo 2 — Teoría de no-unicidad (D1, D2).** El teorema $J_4$ y la
caracterización del espacio nulo. La analogía de Webster–Sturm–Liouville y la
familia biparamétrica de conductos acústicamente idénticos. Ilustración con
geometrías realistas.

**Capítulo 3 — Identificabilidad del modelo bifásico (§3.3 extendido).** Matriz
de información de Fisher y perfiles de verosimilitud sobre el operador de
`RIconduitex5_5.py` (y sus variantes transiente/2D). Pregunta central: ¿la
degeneración $J_4$ del recorte sobrevive al deslizamiento $u_g-u_m$?
Comparación sistemática: estacionario 1D bifásico, transiente 1D, estacionario
2D. **Éste es el resultado que conecta el trabajo previo con la tesis.**

**Capítulo 4 — Inversión bayesiana de Calbuco 2015 (A1–A3).** Previas de la
petrología de Castruccio et al. (2016); datos de MER por fases, masa emitida,
alturas de columna, texturas de piroclastos; posterior sobre $\Delta P$, $R$,
$c_0$, $z_f$; verificación predictiva posterior.

**Capítulo 5 — Geometría como incógnita funcional (B1, E7).** Inversión de
$R(z)$ con regularización; demostración numérica de que el mínimo no es único;
efecto del término de regularización sobre la geometría recuperada; conexión con
los mecanismos de ensanchamiento sineruptivo.

Si el tiempo alcanza, **B2** (coeficiente de Robin en la pared) es la extensión
natural del capítulo 5 hacia el modelo 2D y da material para un artículo
independiente.

---

## 6. Datos disponibles para Calbuco 2015

| Observable | Valor | Fuente |
|---|---|---|
| Volumen emitido total | 0.27–0.56 km³ (no DRE) | Romero et al. (2016); Castruccio et al. (2016); Van Eaton et al. (2016); Pardini et al. (2018) |
| Reparto entre fases | ~38 % fase 1, ~62 % fase 2 | Castruccio et al. (2016) |
| Masa fase 1 / fase 2 | $8\times10^{10}$ kg en 1.5 h / $3.2\times10^{11}$ kg en 6 h | Castruccio et al. (2016) |
| MER media | $1.4\times10^7$ y $1.5\times10^7$ kg/s | Castruccio et al. (2016) |
| MER máxima (por isopletas) | $2.4\times10^7$ y $2.7\times10^7$ kg/s | Castruccio et al. (2016) |
| Altura de columna | 15–17 km sobre el cráter | SERNAGEOMIN; Van Eaton et al. (2016); Romero et al. (2016) |
| Composición | andesita basáltica, 54–55 % SiO₂ | Romero et al. (2016) |
| Almacenamiento | 8–12 km, 900–950 °C, 5.5–6.5 wt% H₂O | Morgado et al. (2019) |
| Texturas de piroclastos | CSD, BSD, densidad, porosidad, permeabilidad | Castruccio et al. (AGU 2018) |
| Deformación | co-eruptiva detectada; pre-eruptiva bajo el nivel de ruido | Delgado et al. (2017) |
| Descargas eléctricas y pluma | serie temporal de la actividad | Van Eaton et al. (2016) |

**Nota importante sobre el ajuste del prototipo.** Con los parámetros de
`calbuco2015d.py` (R = 16 m, ΔP = 5 MPa, 4 wt% H₂O, 970 °C, 25 % cristales), el
modelo reducido da $\mathrm{MER}=3.0\times10^7$ kg/s, del mismo orden que los
$1.4$–$2.7\times10^7$ kg/s observados. El caso base es cuantitativamente
razonable, lo que hace creíble la inversión.

**Advertencia sobre la temperatura.** `calbuco2015d.py` usa 970 °C y 4 wt% H₂O,
mientras la literatura reciente sugiere 900–950 °C y 5.5–6.5 wt%. Vale la pena
revisar este punto con el profesor de geología antes de fijar las previas,
porque la viscosidad de Giordano es exponencialmente sensible a ambos.

---

## 7. Plan de trabajo por etapas

Sin estimaciones de calendario: las etapas están ordenadas por dependencia, y
cada una tiene un entregable verificable.

**Etapa 0 — Consolidación.**
Entregable: un solo operador directo, con tests de regresión contra los tres
solvers actuales, y la fragmentación regularizada. Decisión explícita sobre las
inconsistencias de §1.

**Etapa 1 — Teoría de no-unicidad.**
Entregable: demostración del teorema $J_4$ y de la caracterización del espacio
nulo; construcción explícita de la familia de Webster; figuras con geometrías
realistas. *(Parcialmente hecho: ver §3.1 y `exp1`.)*

**Etapa 2 — Identificabilidad sistemática.**
Entregable: tabla de identificabilidad de todos los parámetros frente a todos
los conjuntos de datos, con perfiles de verosimilitud. Respuesta cuantitativa a
"¿cuánto aporta el tiempo? ¿cuánto aporta la dimensión radial?".

**Etapa 3 — Inversión de Calbuco.**
Entregable: posteriores sobre $\Delta P$, $R$, $c_0$, $z_f$ con datos reales;
verificación predictiva posterior; comparación con las estimaciones
independientes de la petrología.

**Etapa 4 — Geometría como incógnita.**
Entregable: reconstrucción regularizada de $R(z)$ a partir de datos sintéticos y
reales; estudio del efecto del regularizador; demostración numérica de la
no-unicidad.

**Etapa 5 — Escritura y difusión.**
Entregable: manuscrito de tesis; un artículo candidato (los capítulos 2 y 3 son
el material más publicable: la no-unicidad es un resultado que la comunidad
volcanológica no ha formalizado).

---

## 8. Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| El operador directo es caro y MCMC no converge | Emulador (E5); o inferencia basada en estadísticos resumen y algoritmo de vecindad, como Wong et al. (2020) |
| `scikits.odes` no compila en todas las máquinas | El prototipo sólo usa numpy/scipy; portar el estacionario a un solver DAE puro de scipy |
| No hay suficientes datos de Calbuco para restringir el modelo | Ése **es** el resultado; se reporta como análisis de identificabilidad, y se complementa con datos sintéticos |
| Los switches de régimen impiden usar gradientes | Regularización de la fragmentación (E1); si falla, métodos sin derivadas o emulador |
| El alcance crece demasiado | El núcleo son los capítulos 2 y 3, que ya tienen resultados preliminares; 4 y 5 son incrementales |

---

## 9. Preguntas para la reunión con ambos profesores

**Para el profesor de matemáticas**

1. ¿Prefiere el enfoque determinista (Tikhonov, tasas de convergencia bajo
   condiciones de fuente) o bayesiano (posterior, MCMC, cuantificación de
   incertidumbre)? El capítulo 5 admite ambos; la elección cambia el sabor de la
   tesis.
2. ¿Vale la pena invertir en el método adjunto (E2), o basta con diferencias
   finitas y un número moderado de parámetros?
3. ¿Le parece suficiente el resultado de no-unicidad de §3.1 como teorema
   central, o conviene apuntar a un enunciado de unicidad condicional (bajo qué
   datos adicionales $R(z)$ **sí** queda determinada)?
4. La analogía de Webster–Sturm–Liouville (D2), ¿como capítulo propio o como
   sección?

**Para el profesor de geología**

1. Sobre "recuperar la presión en profundidad": ¿el objetivo es $\Delta P$ en la
   cámara, o el perfil $P(z)$ completo? (Ver la discusión en §4, familia A.)
2. Sobre el contenido de burbujas en profundidad: ¿hay serie temporal de
   porosidad/densidad de piroclastos por unidad estratigráfica para Calbuco? Si
   la hay, B3 se vuelve el problema inverso más directo y mejor condicionado de
   todo el catálogo.
3. ¿Existen mediciones de permeabilidad de los piroclastos? Habilitan A4/B4, y
   la literatura indica que la permeabilidad es el control de primer orden.
4. ¿Qué evidencia independiente hay sobre la geometría del conducto de Calbuco
   (componente lítico de los depósitos, geometría del cráter, diques
   aflorantes)? Serviría como previa y como verificación del capítulo 5.
5. ¿Confirma $T$ y $c_0$? (Ver la advertencia al final de §6.)

**Para ambos**

6. ¿El caso de estudio es Calbuco 2015, o conviene un caso con mejor cobertura
   de series de tiempo (por ejemplo un caso efusivo tipo domo, donde la tasa de
   extrusión se mide continuamente)? El análisis de identificabilidad sugiere
   que **los casos con series de tiempo largas son mucho más informativos**, lo
   que es un argumento técnico para la elección del caso.

7. Relacionado, y más fino (§3.4): ¿queremos un caso **cuasi-estacionario** o
   uno **genuinamente transiente**? Determina qué es invertible. En régimen
   cuasi-estacionario los datos informan sobre geometría y cámara pero **nada**
   sobre cinética; sólo con $\mathrm{De}\gtrsim0.1$ las constantes de tasa
   pasan a ser observables. Calbuco está cómodamente en el primer régimen si se
   mira la hidráulica, y en la frontera si se mira la cristalización. Un caso de
   domo cíclico está en el segundo por construcción.

---

## 10. Archivos de esta propuesta

```
tesis/
├── PROPUESTA.md                 ← este documento
├── ESTADO_DEL_ARTE.md           ← revisión bibliográfica anotada
├── prototipo/
│   ├── README.md                ← cómo reproducir los experimentos
│   ├── modelo_reducido.py       ← operador directo con geometría R(z)
│   ├── exp1_no_unicidad_geometria.py
│   ├── exp2_el_tiempo_rompe_la_degeneracion.py
│   ├── exp3_identificabilidad.py
│   ├── exp4_cuando_hace_falta_el_transiente.py
│   └── figuras/                 ← figuras generadas
└── presentacion/
    ├── propuesta_tesis.tex      ← presentación Beamer
    └── propuesta_tesis.pdf      ← 41 láminas + respaldo
```

Reproducir todo:

```bash
pip install numpy scipy matplotlib
cd tesis/prototipo
python3 exp1_no_unicidad_geometria.py
python3 exp2_el_tiempo_rompe_la_degeneracion.py
python3 exp3_identificabilidad.py
python3 exp4_cuando_hace_falta_el_transiente.py
cd ../presentacion && latexmk -pdf propuesta_tesis.tex
```
