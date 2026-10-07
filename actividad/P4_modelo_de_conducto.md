# P4 - Del orificio al conducto

Esta parte se agrega despues de la P3. El script `conducto_actividad.py` usa solo la biblioteca estandar de Python. No hay que instalar numpy ni el modelo completo del repositorio.

Desde la carpeta `actividad`:

```bash
python3 conducto_actividad.py
```

Para un radio distinto, en metros:

```bash
python3 conducto_actividad.py 16
```

Desde Python, las dos funciones que importan son estas:

```python
from conducto_actividad import correr, analogico

volcan = correr(radio=16, volumen_m3=0.3e9)
botella = analogico(radio_m=0.004, volumen_ml=500)
```

`correr` es el conducto. `analogico` es la botella.

## Que hace cada una

La botella es un chorro. La velocidad sale de Torricelli y no depende del radio del orificio:

\[
u = \sqrt{2\Delta P / \rho}.
\]

El caudal si depende del radio, porque \(Q = u \pi R^2\). La duracion es el volumen dividido por ese caudal. Con \(\Delta P = 3\,\mathrm{bar}\), la altura del chorro queda cerca de \(30\,\mathrm{m}\) para cualquier orificio.

El volcan es un conducto de \(7\,\mathrm{km}\), con sobrepresion de \(5\,\mathrm{MPa}\), \(4\,\%\) de agua y viscosidad efectiva \(3000\,\mathrm{Pa\,s}\). En cada paso se calcula la densidad con la ley de Henry y se integra

\[
\frac{dP}{dz} = -\rho g - \frac{8\mu u}{R^2}.
\]

La velocidad de entrada es la que deja la fragmentacion (\(\varphi = 0.75\)) junto a la boca. El caudal de roca densa es \(Q = u_{\mathrm{in}}\pi R^2\). La altura de la columna no se toma de \(u^2/2g\): se usa Mastin et al. (2009),

\[
H[\mathrm{km}] = 2{,}00\, Q^{0{,}241},
\]

con \(Q\) en \(\mathrm{m}^3/\mathrm{s}\). La duracion es el volumen de magma dividido por \(Q\).

Sin gas, la misma cuenta se reduce a Poiseuille, \(u = R^2\Delta P/(8\mu L)\). El script imprime ese valor al final para compararlo con el caso que exsuelve.

## Que tienen que entregar

1. Corran `analogico` con el radio del orificio y el volumen de su grupo. Comparen \(u\), la altura del chorro y la duracion con lo que midieron en la experiencia y con lo que estimaron en la P2.
2. Corran `correr` con \(R = 16\,\mathrm{m}\) y un volumen de \(0{,}3\,\mathrm{km}^3\). Anoten \(u_{\mathrm{in}}\), \(Q\), \(H\) y la duracion.
3. Repitan `correr` cambiando solo el radio, en la misma proporcion entre orificio chico y orificio grande que usaron en la experiencia. Si no alcanzaron a medir la razon, usen \(8\), \(16\) y \(32\,\mathrm{m}\).
4. Repitan `correr` cambiando solo el volumen, en la proporcion de las botellas: \(0{,}05\), \(0{,}15\) y \(0{,}30\,\mathrm{km}^3\) (es \(1:3:6\), como \(500\), \(1500\) y \(3000\,\mathrm{ml}\)).
5. Con esos numeros, respondan:
   - En la botella, al agrandar el orificio, que le pasa a la altura y que le pasa a la duracion.
   - En el conducto, al agrandar el radio, que le pasa a \(Q\), a la altura y a la duracion.
   - Que tendencia de la P1 se parece entre el analogo y el conducto, y cual no. Justifiquen con las dos formulas de arriba, no solo con el grafico.

Los numeros del conducto no tienen que coincidir con la columna de Calbuco. La viscosidad esta fija y la fragmentacion se impone en la boca. Lo que se compara es como escala cada modelo cuando cambian el radio y el volumen.
