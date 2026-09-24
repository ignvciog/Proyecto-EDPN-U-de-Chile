# Casos del paper (Castruccio, Rebolledo & Gómez, 2025)

Archivos **tal cual** se usaron en

> Castruccio, A., Rebolledo, A. & Gómez, I. (2025).
> *The influence of melt composition, temperature, crystallinity and water
> content on eruptive style and eruption rate: insights from a conduit model
> of magma ascent.* JGR Solid Earth 130, e2024JB030599.
> https://doi.org/10.1029/2024JB030599

No están reescritos ni “reducidos”. Cada `.m` es el script de parámetros de
entrada de RIConduit 5.2 (MATLAB). El solver Python equivalente está en la
raíz: `RIconduitex5_5.py` (explosivo) y `RIconduitef5_5.py` (efusivo).

Para ver **las dos ramas con los mismos datos** (sin tocar esos solvers):

```bash
# notebook (lo mas parecido a mainconduit5_5)
# abrir mainconduit_ambos.ipynb, cambiar CASO, correr todo

# o por terminal
python3 mainconduit_ambos.py --caso calbuco2015
python3 mainconduit_ambos.py --caso merapi2010
python3 mainconduit_ambos.py --caso villarrica2015
```

Eso llama a `RIconduitex5_5_f` y a `RIconduitef5_5_f`, imprime si cada tiro
cerró, y si hay perfiles escribe `casos/salida/<caso>_ambos.png`. Requiere
`scikits.odes`, igual que los solvers originales.

Los dos solvers resuelven **el mismo DAE**. Cambia solo la condición de
salida del tiro:

| | explosivo (`ex`) | efusivo (`ef`) |
|---|---|---|
| $P(0)$ | $P_{\mathrm{atm}}$ **o** choque $u_g\approx\sqrt{R_vT}$ | $P_{\mathrm{atm}}$ |
| $\phi(0)$ | $\phi\ge\phi_{\mathrm{crit}}$ (fragmentó) | $\phi\le\phi_{\mathrm{crit}}$ (no fragmentó) |
| $z_{\mathrm{exit}}$ | $\ge -5\,\mathrm{m}$ | $\in(-2,0]$ |

Para un mismo $\theta=(c_0,T,\xi_0,H,\ldots)$ el código puede devolver
solución explosiva, efusiva, **las dos**, o ninguna. Eso ya es una
no-unicidad de *estilo*, distinta de la degeneración geométrica $J_4$.

En el paper, $R$ y $\Delta P$ **no** son datos: los calcula el código
(estabilidad de conducto). En los `.m` aparecen los valores que ese
cálculo dejó para ese run.

## Tabla 2 del paper (inputs petrológicos)

Valores centrales; el paper barre $\pm5\,\mathrm{vol\%}$ en $\xi$,
$\pm25^\circ\mathrm{C}$ en $T$ y $\pm0.5\,\mathrm{wt\%}$ en H₂O
($3^3=27$ corridas por caso).

| Erupción | archivo | estilo (obs.) | SiO₂ | H₂O | $\xi$ | $T$ | $H$ |
|---|---|---|---:|---:|---:|---:|---:|
| Calbuco 2015 | `../calbuco2015d.py` | sub-Pliniana | 62.72 | 4 | 0.25 | 970 | 7 km |
| Vesuvius 79 | `vesuvius79.m` | Pliniana (pumita blanca) | 58.19 | 5 | 0.15 | 850 | 7.5 km |
| Huaynaputina 1600 | `huaynaputina1600b.m` | Pliniana | 73.3 | 6 | 0.17 | 850 | 8 km |
| Caulle 2011 | `caulle2011c.m` | sub-Pliniana → lava | 72.1 | 3.5 | 0.05 | 900 | 5 km |
| Merapi 2010 | `merapi2010m.m` | explosiva + domo | 62.3 | 4 | 0.4–0.5 | 970 | 3.5 km |
| St. Helens 2004 | `sthelens2004b.m` | domo (efusiva) | 73.5 | 3 | 0.40 | 950 | 5 km |
| Villarrica 2015 | `villarrica2015f.m` | hawaiana / estromboliana | 54.14 | 1.5 | 0.25 | 1130 | 6 km |

Villarrica usa Henry basáltico ($C_1=6.80\times10^{-8}$, $\beta=0.7$) y
geometría de **dique**. El resto usa Henry silíceo
($C_1=4.11\times10^{-6}$, $\beta=0.5$) y cilindro.

## Archivos extra (no están en la Tabla 2)

| Erupción | archivo | notas en el `.m` |
|---|---|---|
| Pinatubo 1991 | `pinatubo1991c.m` | $T=780^\circ\mathrm{C}$, 5.5 wt% H₂O, $H=8$ km, $R=90$ m, $\Delta P=16$ MPa |
| Quizapu 1932 | `quizapu1932c.m` | $T=865^\circ\mathrm{C}$, 5 wt% H₂O, $H=5.5$ km, $R=30$ m, $\Delta P=8$ MPa |

## Discrepancias archivo ↔ paper (no se “corrigieron”)

Los `.m` se copiaron sin editar. Donde no calzan con la Tabla 2:

- `caulle2011c.m` tiene `namevolc='St Helens 2004'` (copia) y H₂O = 4 wt%
  (el paper usa 3.5). $H=-5000$ m coincide con la tabla; el texto del
  paper menciona también 4 km.
- `vesuvius79.m` y `huaynaputina1600b.m` dejan $R$ y $\Delta P$ para que
  el código los calcule (van comentados).
- `merapi2010m.m` trae un run concreto: $\xi=0.5$, $R=5.75$ m,
  $\Delta P=9.76$ MPa.
- `villarrica2015f.m` trae dique $R=0.985$ m, $L=200$ m,
  $\Delta P=1.88$ MPa (un run del paper, no el valor central de $H$ del
  texto de 4 km; el archivo usa 6 km, igual que la Tabla 2).

## Qué implica para el problema inverso

El dato observado no es solo un número $\mathrm{MER}$. Es el par
$(\mathrm{MER},\,\mathrm{estilo})$. El operador directo del paper es

$$
\mathcal{F}(\theta)\;=\;
\bigl(q_{\mathrm{ex}}^\ast(\theta),\;q_{\mathrm{ef}}^\ast(\theta)\bigr)
\in
\bigl(\mathbb{R}_+\cup\{\varnothing\}\bigr)^2.
$$

Invertir solo $q$ sin el estilo deja las dos ramas confundidas
(Merapi y Caulle son el ejemplo). Invertir el par identifica la rama,
pero $R$ y $\Delta P$ siguen sin ser únicos: en el paper son *salidas*
de un cierre mecánico, no observables.
