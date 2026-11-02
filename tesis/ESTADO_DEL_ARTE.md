# Estado del arte

Revisión anotada orientada a tres preguntas:

1. ¿Qué modelos de ascenso de magma por un conducto han agregado **tiempo**, la
   **dimensión radial** o **geometría variable**?
2. ¿Qué se ha hecho en **problemas inversos** con modelos de conducto?
3. ¿Dónde queda un **nicho abierto** para esta tesis?

Todas las referencias llevan DOI verificado.

---

## 1. El linaje del modelo de este repositorio

**Kozono, T. & Koyaguchi, T. (2009, 2010, 2012)** — Es la base del solver
estacionario `RIconduitex5_5.py`: flujo bifásico 1D con velocidades separadas
de gas y fundido, arrastre viscoso, escape de gas por permeabilidad y
fragmentación por umbral de porosidad.

- Kozono & Koyaguchi (2012), *Effects of gas escape and crystallization on the
  complexity of conduit flow dynamics during lava dome eruptions*, JGR.
  [10.1029/2012JB009343](https://doi.org/10.1029/2012JB009343)
  Introduce la noción de **resistencia diferencial negativa**
  ($dp_{ch}/dq<0$), origen de la multiplicidad de estados estacionarios. Muy
  relevante para el capítulo de no-unicidad: existe una segunda fuente de
  no-unicidad, distinta de la geométrica.

**Castruccio et al. (2016)**, *Eruptive parameters and dynamics of the April 2015
sub-Plinian eruptions of Calbuco volcano (southern Chile)*, Bulletin of
Volcanology 78(9).
[10.1007/s00445-016-1058-8](https://doi.org/10.1007/s00445-016-1058-8)
Es la fuente de los parámetros de `calbuco2015d.py` y de los observables para la
inversión: volúmenes, masas por fase, tasas de descarga, alturas de columna.

**Giordano, Russell & Dingwell (2008)** — modelo de viscosidad implementado en
`viscosity.py`. **Bottinga & Weill (1970)** — densidad del fundido,
`density.py`. **Zuber & Findlay (1965)** — cierre de deriva (*drift-flux*) del
transiente.

---

## 2. Modelos con TIEMPO (transientes)

### 2.1 Cámara acoplada y comportamiento cíclico

**Melnik & Sparks (1999)**, *Nonlinear dynamics of lava dome extrusion*, Nature
402:37–41.
El resultado fundacional: para condiciones fijas pueden existir **hasta tres
tasas de descarga estacionarias**, que difieren en órdenes de magnitud. La
retroalimentación cristalización → viscosidad genera la curva sigmoidal. Cambios
pequeños en viscosidad, presión de cámara o **diámetro del conducto** producen
cambios enormes en la tasa de flujo.

**Barmin, Melnik & Sparks (2002)**, *Periodic behavior in lava dome eruptions*,
EPSL 199:173–184.
[10.1016/S0012-821X(02)00557-5](https://doi.org/10.1016/S0012-821X(02)00557-5)
Modelo reducido cámara + conducto que produce ciclos por transiciones entre
ramas estables. Matemáticamente limpio, ideal como modelo de juguete para probar
algoritmos de inversión.

**Melnik & Sparks (2005)**, *Controls on conduit magma flow dynamics during lava
dome building eruptions*, JGR.
[10.1029/2004JB003183](https://doi.org/10.1029/2004JB003183)
Transiente con cámara abierta y recarga continua; periodos controlados
principalmente por el **volumen de la cámara**. Confirma el mecanismo que el
experimento 2 del prototipo usa para romper la degeneración.

### 2.2 Transientes multifásicos modernos

**La Spina, de' Michieli Vitturi & Clarke (2017)**, *Transient numerical model of
magma ascent dynamics: application to the explosive eruptions at the Soufrière
Hills Volcano*, JVGR 336:118–139.
[10.1016/j.jvolgeores.2017.02.013](https://doi.org/10.1016/j.jvolgeores.2017.02.013)
**La referencia más cercana a lo que hace `RIconduit1D_transient.py`, pero mucho
más completa.** Sistema hiperbólico termodinámicamente compatible, con dos
presiones y dos velocidades, tasas finitas de exsolución y relajación de presión
y velocidad, y tratamiento unificado de flujo denso y diluido (arriba y abajo de
la fragmentación). Conclusión clave: **la duración de la erupción está
controlada por las escalas de tiempo de exsolución**, no por el equilibrio.
Lectura obligada antes de decidir el operador directo definitivo.

**La Spina et al. (2016)**, *Role of syn-eruptive plagioclase disequilibrium
crystallization in basaltic magma ascent dynamics*, Nature Communications.
[10.1038/ncomms13402](https://doi.org/10.1038/ncomms13402)
Cristalización en desequilibrio de tres fases minerales, con equilibrio
calculado por alphaMELTS y H₂O–CO₂ por VolatileCalc. Marca el estándar de
"realismo" al que apunta el punto E4 de la propuesta.

**Código abierto: MAMMA** (de' Michieli Vitturi, La Spina, Aravena),
Fortran 90, volúmenes finitos, esquema central semidiscreto.
<http://demichie.github.io/MAMMA/> — Vale la pena evaluarlo como operador
directo alternativo, en vez de reescribir todo.

**Wong & Segall (2019)**, *Numerical Analysis of Time-Dependent Conduit Magma
Flow in Dome-Forming Eruptions With Application to Mount St. Helens 2004–2008*,
JGR.
[10.1029/2019JB017585](https://doi.org/10.1029/2019JB017585)
Extiende el modelo estacionario de Wong et al. (2017) a dependencia temporal,
con escape de gas vertical y lateral por flujo de Darcy. Observación relevante
para la propuesta: **la mayoría de los modelos transientes a escala de erupción
siguen siendo 1D promediados radialmente**, no genuinamente 2D. Predice tres
series de tiempo simultáneas (deformación, flujo de extrusión, emisión de gas),
que es justo lo que se necesita para una inversión conjunta.

---

## 3. Modelos con GEOMETRÍA VARIABLE ("1.5D")

**de' Michieli Vitturi, Clarke, Neri & Voight (2008)**, *Effects of conduit
geometry on magma ascent dynamics in dome-forming eruptions*, EPSL
272:567–578.
[10.1016/j.epsl.2008.05.025](https://doi.org/10.1016/j.epsl.2008.05.025)
Código **DOMEFLOW**: modelo transiente "1.5D", isotermo, bifásico, con conducto
axisimétrico de **radio variable**. Barren la razón de radios
$0.4 \le R_t/R_b \le 2.5$ y la altura del cambio $0.1 \le H/L \le 0.7$. La
geometría afecta sobrepresión, fracción de gas, tasa de ascenso y espesor y
densidad del tapón. Reconocen explícitamente que no modelan la
retroalimentación flujo → geometría.

**de' Michieli Vitturi, Clarke, Neri & Voight (2010)**, *Transient effects of
magma ascent dynamics along a geometrically variable dome-feeding conduit*,
EPSL.
[10.1016/j.epsl.2010.04.029](https://doi.org/10.1016/j.epsl.2010.04.029)
Más de 300 corridas y análisis dimensional producen una relación universal: el
producto del cambio de tasa de extrusión por el tiempo de transición es
proporcional al cambio normalizado de presión de cámara, **al volumen del
conducto** y a la razón de radios. **Es exactamente la degeneración que el
experimento 2 del prototipo explota**, y da un punto de comparación directo.

**Costa, Melnik & Sparks (2007)**, *Controls of conduit geometry and wallrock
elasticity on lava dome eruptions*, EPSL 260:137–151.
[10.1016/j.epsl.2007.05.024](https://doi.org/10.1016/j.epsl.2007.05.024)
Acopla la geometría a la elasticidad de la roca encajante: el conducto se
deforma con la presión. Es el paso siguiente a "geometría prescrita" y el
antecedente natural del problema de frontera libre (idea O de la propuesta).

**Aravena, Cioni, de' Michieli Vitturi, Neri et al. (2017)**, *Stability of
volcanic conduits during explosive eruptions*, JVGR.
[10.1016/j.jvolgeores.2017.03.023](https://doi.org/10.1016/j.jvolgeores.2017.03.023)
Aplican criterios de colapso de Mohr–Coulomb y Mogi–Coulomb al perfil $P(z)$ del
modelo 1D. Resultado central: **los conductos cilíndricos son mecánicamente
estables sólo para radios grandes**, de modo que el ensanchamiento sineruptivo
conduce naturalmente a geometrías con radio dependiente de la profundidad.
Definen las configuraciones no cilíndricas NC2 (dos tramos coaxiales) y NC3
(ensanchamiento lineal somero), que el prototipo reproduce en el experimento 1.

**Aravena et al. (2018)**, *Conduit stability effects on intensity and steadiness
of explosive eruptions*, Scientific Reports 8.
[10.1038/s41598-018-22539-8](https://doi.org/10.1038/s41598-018-22539-8)
Extiende a magmas fonolíticos, traquíticos, dacíticos y riolíticos. Impedir el
colapso sobre el nivel de fragmentación puede aumentar la tasa de descarga hasta
un 15 %; un ensanchamiento sustancial baja la presión de salida.

> **Nota:** Alvaro Aravena es chileno y trabaja en esta línea exacta (geometría
> no cilíndrica + modelos de conducto). Es un contacto natural y un posible
> evaluador externo.

---

## 4. Modelos 2D

**Massol, Jaupart & Pepper (2001)**, *Ascent and decompression of viscous
vesicular magma in a volcanic conduit*, JGR 106(B8):16223–16240.
[10.1029/2001JB000385](https://doi.org/10.1029/2001JB000385)
**El 2D axisimétrico de referencia.** Elementos finitos, Navier–Stokes
compresible a bajo Reynolds, con componentes de velocidad radial *y* vertical.
Resultados que un modelo 1D no puede producir:

- La sobrepresión del gas en el vent puede superar 1 MPa.
- Hay **variaciones horizontales grandes** de presión de gas y de agua exsuelta.
- El resultado depende críticamente de la **condición de borde de salida**: con
  velocidad horizontal nula la sobrepresión es máxima en el centro; con esfuerzo
  de corte nulo, es máxima en las paredes. Diferencia del orden de 1 MPa.

Este último punto es directamente relevante para la propuesta: es un ejemplo
documentado de que **la condición de borde importa tanto como los parámetros**,
lo que justifica tratarla como incógnita (idea B2, coeficiente de Robin).

**Massol & Jaupart (2009)**, *Dynamics of magma flow near the vent: implications
for dome eruptions*, EPSL.
Agrega cristalización inducida por desgasificación; esfuerzos tensiles de ~2 MPa
en el vent podrían explicar el fracturamiento anular de los domos.

**Collier & Neuberg (2006)** y **Thomas & Neuberg (2014)** — modelos 2D
axisimétricos por elementos finitos (COMSOL) con tres fases (gas, cristales,
fundido), pérdida de gas, deslizamiento en la pared, enfriamiento y falla frágil
del fundido. Es la línea que conecta el flujo del conducto con las señales
sísmicas de largo período.

**Thomas & Neuberg (2019)**, *Combining Magma Flow and Deformation Modeling to
Explain Observed Changes in Tilt*, Frontiers in Earth Science.
[10.3389/feart.2019.00219](https://doi.org/10.3389/feart.2019.00219)
Acopla el flujo 2D con deformación elástica del edificio para predecir
inclinometría. Precedente directo de una inversión conjunta flujo + geodesia.

**Tsvetkova & Melnik (2018)** — modelos cuasi-2D que muestran que la dependencia
de la viscosidad con la tasa de corte afecta significativamente la dinámica de
ascenso. Justifica el bucle de Picard de `RIconduit2D_FD.py`.

**Dufek & Bergantz (2005)**, *Transient two-dimensional dynamics in the upper
conduit of a rhyolitic eruption*, JVGR — transiente **y** 2D simultáneamente,
todavía poco común.

---

## 5. Modelos 2D/3D multicomponente

**Longo et al. (código GALES)** — elementos finitos para fluidos multicomponente
compresibles e incompresibles con estabilización Galerkin mínimos cuadrados y
GMRES, resolviendo masa, momentum, energía y composición. Aplicado a dinámica
multidimensional transiente en cámaras y conductos antes, durante y después de
erupciones.

**Modelos acoplados fluido-estructura 2D axisimétricos en COMSOL** (por ejemplo
el estudio de Laguna del Maule,
[10.1002/2016JB013066](https://doi.org/10.1002/2016JB013066)) — resuelven
Navier–Stokes dependiente del tiempo acoplado a elastostática del encajante. Es
el marco natural si se quiere que la geometría del conducto responda al flujo.

> **Observación honesta para la reunión:** modelos genuinamente 3D de conducto
> son escasos, y por una buena razón física — el conducto es un tubo esbelto
> ($L/R \sim 400$ en el caso de Calbuco), de modo que la axisimetría captura casi
> toda la física relevante. El 3D se justifica sólo para diques, secciones
> elípticas, conductos inclinados o sistemas de conductos múltiples. **No
> recomiendo 3D como objetivo de la tesis**: el retorno científico es bajo frente
> al costo, y el problema inverso se vuelve intratable. La dimensión que sí paga
> es el **tiempo**, y después la **geometría variable**.

---

## 6. Problemas inversos en volcanología

Éste es el corazón bibliográfico de la propuesta. La línea está esencialmente
monopolizada por el grupo de Stanford/USGS.

**Anderson & Segall (2011)**, *Physics-based models of ground deformation and
extrusion rate at effusively erupting volcanoes*, JGR — el modelo directo.

**Anderson & Segall (2013)**, *Bayesian inversion of data from effusive volcanic
eruptions using physics-based models: Application to Mount St. Helens
2004–2008*, JGR 118:2017–2037.
[10.1002/jgrb.50169](https://doi.org/10.1002/jgrb.50169)
**La referencia fundacional.** Invierten GPS + tasa de extrusión con un modelo
físico de cámara elipsoidal + conducto cilíndrico, por MCMC. Estiman geometría,
presión, profundidad y contenido de volátiles de la cámara, y radio del
conducto, longitud del tapón y fricción. Resultado: cámara de >40 km³ a 11–18 km
de profundidad, 2.6–4.9 wt% de agua disuelta. Argumento central: a diferencia de
las inversiones geodésicas con modelos cinemáticos, esta técnica **sí** puede
restringir propiedades del magma y el volumen y presión absolutos de la cámara.

**Wong, Segall, Bradley & Anderson (2017)**, *Constraining the Magmatic System at
Mount St. Helens (2004–2008) Using Bayesian Inversion With Physics-Based Models
Including Gas Escape and Crystallization*, JGR 122:7789–7812.
[10.1002/2017JB014343](https://doi.org/10.1002/2017JB014343)
Agregan cristalización en equilibrio y transporte de gas por Darcy. Hallazgos
directamente relevantes:

- El flujo másico erupcionado **depende fuertemente de las permeabilidades** del
  magma y de la roca encajante.
- Incluir transporte de gas lateral *y* vertical produce comportamiento **no
  monótono** del flujo másico respecto de la permeabilidad del magma — es decir,
  no-unicidad adicional.
- La escala de permeabilidad queda bien restringida ($\sim10^{-11.4}$ m²),
  mientras que otros parámetros no.

**Wong & Segall et al. (2020)**, *Joint Inversions of Ground Deformation,
Extrusion Flux, and Gas Emissions Using Physics-Based Models for the Mount St.
Helens 2004–2008 Eruption*, G-Cubed.
[10.1029/2020GC009343](https://doi.org/10.1029/2020GC009343)
Inversión conjunta de tres series de tiempo (volumen extruido, deformación, CO₂)
con el algoritmo de vecindad de Sambridge (1999) y remuestreo bayesiano. Es el
estándar metodológico al que debería apuntar el capítulo 4.

### Lo que este cuerpo de trabajo NO hace

Y que define el nicho de la tesis:

1. **Ninguno trata la geometría del conducto como función incógnita.** Todos
   invierten un radio escalar (a lo sumo un radio y una longitud de tapón).
2. **Ninguno presenta un análisis de no-unicidad estructural.** Reportan
   posteriores anchas, pero no demuestran *por qué* son anchas ni caracterizan
   el espacio nulo.
3. **Todos son casos efusivos** (domos de Mount St. Helens, Soufrière Hills).
   No hay inversión bayesiana con modelo físico de una erupción **explosiva
   sub-Pliniana**.
4. **Ninguno usa un modelo con dimensión radial resuelta.**

---

## 7. No-unicidad y multiplicidad de estados

**Slezin (2003)**, *The mechanism of volcanic eruptions (a steady state
approach)*, JVGR — reconoce por primera vez la existencia de regímenes múltiples.

**Melnik (2000)**, *Dynamics of two-phase conduit flow of high-viscosity
gas-saturated magma: large variations of sustained explosive eruption
intensity*, Bulletin of Volcanology 62:153–170.
[10.1007/s004450000072](https://doi.org/10.1007/s004450000072)

**Kozono & Koyaguchi (2012)** (arriba) — diagnostican dos mecanismos de
retroalimentación que producen resistencia diferencial negativa: cristalización
retardada y cambio de porosidad por escape de gas.

**Barmin, Melnik & Sparks (2002)** (arriba) — la formulación adimensional más
limpia de la multiplicidad de estados estacionarios.

> Esta literatura documenta la no-unicidad **respecto de la tasa de flujo a
> parámetros fijos** (multiplicidad de soluciones del problema directo). La
> propuesta ataca una no-unicidad distinta y complementaria: **respecto de los
> parámetros y de la geometría a dato fijo** (no-inyectividad del operador
> directo). Que yo sepa, esta segunda no ha sido formalizada en la literatura
> volcanológica, y ése es el aporte original.

### Herramientas matemáticas importables

- **Perfiles de verosimilitud para identificabilidad práctica**: Raue et al.
  (2009), *Structural and practical identifiability analysis of partially
  observed dynamical models by exploiting the profile likelihood*,
  Bioinformatics 25:1923–1929.
  [10.1093/bioinformatics/btp358](https://doi.org/10.1093/bioinformatics/btp358)
  Estándar en biología de sistemas; se importa sin cambios. Es lo que usa el
  experimento 3 del prototipo.
- **Teoría espectral inversa de Sturm–Liouville**: teorema de dos espectros de
  Borg (1946); Gel'fand–Levitan; Pöschel & Trubowitz, *Inverse Spectral Theory*
  (1987). Base de la idea D2.
- **Identificación de coeficientes de Robin**: Chaabane & Jaoua (1999), Inglese
  (1997), Sincich (2007) — unicidad y estabilidad logarítmica. Base de la idea
  B2.
- **Problemas parabólicos laterales (*sideways*)**: Eldén, Berntsson & Regińska
  (2000) y la literatura de la ecuación del calor lateral. Base de la idea C1.
- **Regularización en espacios de Banach y métodos bayesianos en dimensión
  infinita**: Stuart (2010), *Inverse problems: a Bayesian perspective*, Acta
  Numerica 19:451–559.
  [10.1017/S0962492910000061](https://doi.org/10.1017/S0962492910000061)

---

## 8. Contexto de Calbuco 2015

**Castruccio et al. (2016)**, Bulletin of Volcanology 78(9).
[10.1007/s00445-016-1058-8](https://doi.org/10.1007/s00445-016-1058-8)
Volumen 0.38 km³ (no DRE); dos fases sub-Plinianas; andesita basáltica 54–55 %
SiO₂; al menos cuatro texturas de clastos juveniles; masa de $8\times10^{10}$ kg
en 1.5 h y $3.2\times10^{11}$ kg en 6 h.

**Romero et al. (2016)**, *Eruption dynamics of the 22–23 April 2015 Calbuco
Volcano (Southern Chile): Analyses of tephra fall deposits*, JVGR 317:15–29.
[10.1016/j.jvolgeores.2016.02.027](https://doi.org/10.1016/j.jvolgeores.2016.02.027)

**Van Eaton et al. (2016)**, *Volcanic lightning and plume behavior reveal
evolving hazards during the April 2015 eruption of Calbuco volcano, Chile*, GRL.
[10.1002/2016GL068076](https://doi.org/10.1002/2016GL068076)
Aporta **series de tiempo** de altura de pluma y actividad eléctrica, y una tasa
de descarga de $\sim1\times10^7$ kg/s. Es la fuente más prometedora de datos
temporales para la inversión.

**Arzilli et al. (2019)**, *The unexpected explosive sub-Plinian eruption of
Calbuco volcano (22–23 April 2015; southern Chile): Triggering mechanism
implications*, JVGR 378:35–50.
[10.1016/j.jvolgeores.2019.04.006](https://doi.org/10.1016/j.jvolgeores.2019.04.006)
Almacenamiento a 8–12 km (230–320 MPa), 900–950 °C, saturado en agua a
5.5–6.5 wt%. Proponen **gatillo interno**: cristalización prolongada → segunda
ebullición → sobrepresurización, sin intrusión de magma fresco.

**Morgado et al. (2019)** — hipótesis **competidora**: calentamiento localizado
del reservorio por inyección de magma caliente.

> **Oportunidad concreta.** Existen dos hipótesis publicadas y contrapuestas
> sobre el mecanismo gatillante de Calbuco 2015. Ambas predicen historias de
> sobrepresión distintas, y por tanto **curvas MER(t) distintas**. Discriminar
> entre ellas mediante selección bayesiana de modelos (factores de Bayes) sería
> un resultado con impacto en la comunidad volcanológica chilena, y es
> exactamente el tipo de pregunta que un problema inverso bien planteado puede
> responder. Vale la pena proponérselo al profesor de geología.

**Advertencia de consistencia.** `calbuco2015d.py` usa $T=970$ °C,
$c_0 = 4$ wt% y base del conducto a 7 km. Arzilli et al. (2019) reportan
900–950 °C, 5.5–6.5 wt% y almacenamiento a 8–12 km. La viscosidad de Giordano es
exponencialmente sensible a ambos parámetros. Conviene reconciliar esto antes de
fijar previas.

---

## 9. Síntesis: dónde encaja esta tesis

| Dimensión agregada | Estado del arte | Qué falta |
|---|---|---|
| **Tiempo** | Maduro (Melnik & Sparks; La Spina et al.; Wong & Segall) | Usarlo **para invertir**, no sólo para simular, y cuantificar cuánta información aporta |
| **Radial (2D)** | Maduro (Massol & Jaupart; Collier & Neuberg) | Nadie lo ha usado en un problema inverso; la condición de borde de pared como incógnita es territorio virgen |
| **3D** | Escaso y poco justificado físicamente | No recomendado para esta tesis |
| **Geometría variable** | Maduro como estudio paramétrico directo (de' Michieli Vitturi; Aravena; Costa) | Nadie la ha tratado como **incógnita de un problema inverso**, ni ha caracterizado su no-unicidad |
| **Inversión bayesiana** | Establecida para casos efusivos (Anderson & Segall; Wong et al.) | Ningún caso explosivo sub-Pliniano; ningún análisis estructural de no-unicidad; ningún caso chileno |

**El nicho es la intersección de las dos últimas filas:** tratar la geometría del
conducto como incógnita funcional en un problema inverso bayesiano, demostrar
que el problema es estructuralmente no único, caracterizar el espacio nulo y
determinar qué datos lo reducen — aplicado a una erupción explosiva chilena con
datos reales.
