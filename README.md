# Modelo de conducto volcánico 1D/2D

Proyecto final del curso **Análisis Numérico de Ecuaciones en Derivadas Parciales (MA-5307)** — DIM, Universidad de Chile.

Implementación en Python de un modelo de conducto volcánico con:

- **1D estacionario explosivo** (`RIconduitex5_5.py`) y **efusivo** (`RIconduitef5_5.py`): el mismo DAE bifásico; cambia la condición de salida del tiro
- **Fragmentación regularizada (laboratorio):** `fragmentacion_reg.py` + `RIconduitex5_5_reg.py` + `regularizar_fragmentacion.ipynb`. El original no se toca.
- **Todos los umbrales suavizados (laboratorio):** `umbrales_reg.py` + `RIconduitex5_5_suave.py` + `regularizar_umbrales.ipynb` (zip `umbrales_suave.zip`). `tasa_xi` solo suaviza `f2` y `f3`; no aplicar `softplus` otra vez a `dx/dz` (hincha ξ a 0.35). El original no se toca.
- **DAE unificado (laboratorio):** `RIconduitex5_5_unificado.py` — un solo IDA, \(F=\sum w_i F^{(i)}\) con `pesos_regimen`. Autónomo (no importa de `RIconduitex5_5_suave`). El suave de cuatro tramos no se reemplaza.
- Casos del paper JGR 2025 en `casos/` (Pinatubo, Quizapu, St. Helens, Vesuvius, Villarrica, Caulle, Huaynaputina, Merapi; Calbuco en `calbuco2015d.py`)
- **Ambas ramas a la vez:** `mainconduit_ambos.ipynb` (mismo estilo que `mainconduit5_5`) tira explosiva y efusiva con los mismos datos y las 6 figuras encima
- **1D transiente** (EDPs + esquema IMEX: MOL, FD upwind, RK4)
- **2D axisimétrico** (perfil radial \(u_m(r)\) con FD, Thomas y Picard)
- **2D con \(\varphi(r,z)\)** (`RIconduit2D_phi_r.py`): reducción \(u_r=0\), \(P=P(z)\), campos \(\varphi,N,\xi\) en cada nodo radial; paso adaptativo \(h\) vs \(h/2\); umbrales suaves de `umbrales_reg.py`. El `RIconduit2D_FD.py` no se toca (\(\varphi=\varphi(z)\) ahí).

Parámetros de prueba: composición y geometría del volcán **Calbuco** (`calbuco2015d.py`).

**Autores:** Ignacio Gómez G. & Matías Salinas

---

## Estructura del repositorio

```
Codigo Python/
├── README.md                          ← este archivo
├── requirements.txt                   ← dependencias Python
├── .gitignore
│
├── calbuco2015d.py                    ← parámetros del caso Calbuco
├── density.py                         ← densidad del fundido
├── viscosity.py                       ← viscosidad (Giordano et al.)
├── fvrel.py                           ← viscosidad relativa (cristales/burbujas)
│
├── RIconduitex5_5.py                  ← solver 1D estacionario explosivo (IDA)
├── RIconduitef5_5.py                  ← solver 1D estacionario efusivo (mismo DAE)
├── casos/                             ← inputs del paper JGR 2025 (MATLAB, sin editar)
├── RIconduit1D_transient.py           ← solver 1D transiente (MOL + IMEX)
├── RIconduit2D_FD.py                  ← solver 2D axisimétrico (FD radial)
├── RIconduit1D_chamber.py             ← extensión con cámara (si aplica)
│
├── main_transient.py                  ← script: estacionario → transiente + figuras
├── mainconduit_ambos.ipynb            ← notebook: explosiva + efusiva, mismos datos
├── mainconduit_transient_v3.ipynb     ← notebook transiente (recomendado)
├── mainconduit5_7.ipynb               ← notebook 2D
├── mainconduit5_5.ipynb               ← notebook estacionario
│
└── Template-Presentacion-dim/         ← presentación Beamer (LaTeX)
    ├── main.tex                       ← documento principal (compilar este)
    ├── example.tex                    ← contenido de las diapositivas
    ├── library.bib                    ← referencias APA
    ├── template.tex / template_config.tex
    ├── copiar_imagenes.py             ← utilidad opcional para figuras
    └── img/                           ← figuras de la presentación (subir a GitHub)
        ├── departamentos/dim.pdf      ← logo DIM (requerido por el template)
        └── *.png, *.jpg               ← resultados y esquemas
```

Carpetas **opcionales** (no necesarias para reproducir el núcleo del proyecto):

- `Versiones antiguas/` — código legacy
- `Codigo MatLab/` — versiones en MATLAB
- `Pruebas/` — notebooks de prueba
- `examples/` — otros volcanes de ejemplo

---

## Requisitos

- Python 3.9+
- [MiKTeX](https://miktex.org/) o [TeX Live](https://www.tug.org/texlive/) (para la presentación), o **Overleaf**

### Instalar dependencias Python

```bash
cd "Codigo Python"
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

pip install -r requirements.txt
```

> El modelo **estacionario** usa `scikits.odes` (solver IDA). El **transiente** y el **2D** solo requieren `numpy` y `matplotlib`.

---

## Cómo ejecutar el código

### 1. Modelo 1D estacionario

Desde Python o Jupyter:

```python
from calbuco2015d import radius1, overP1, h2o1, T1, xi1
from RIconduitex5_5 import RIconduitex5_5_f

result = RIconduitex5_5_f(radius1, overP1, h2o1, T1, xi1)
```

O abrir `mainconduit5_5.ipynb` (solo explosiva) o `mainconduit_ambos.ipynb` (explosiva y efusiva, mismas 6 figuras encima).

### 2. Modelo 1D transiente (estacionario → transiente)

**Opción A — script:**

```bash
python main_transient.py
```

Genera figuras en la carpeta actual.

**Opción B — notebook (recomendado):**

Abrir `mainconduit_transient_v3.ipynb` y ejecutar todas las celdas.

**Opción C — llamada directa:**

```python
from RIconduit1D_transient import RIconduit1D_transient_f
from calbuco2015d import radius1, overP1, h2o1, T1, xi1

out = RIconduit1D_transient_f(
    radius=radius1,
    Pressure=overP1,
    wt=h2o1,
    Temperature=T1,
    content_crystal=xi1,
    N_z=200,
    T_max=3600.0,
    CFL=0.4,
    tau_relax=50.0,
)
```

### 3. Modelo 2D axisimétrico

```python
from RIconduit2D_FD import RIconduit2D_FD_f
from calbuco2015d import radius1, overP1, h2o1, T1, xi1

res = RIconduit2D_FD_f(
    radius=radius1,
    Pressure=overP1,
    wt=h2o1,
    Temperature=T1,
    content_crystal=xi1,
    N_r=20,
    N_z=3000,
)
```

O abrir `mainconduit5_7.ipynb`.

### 4. Cambiar parámetros

Editar `calbuco2015d.py`: radio del conducto, sobrepresión, H₂O, temperatura, fracción de cristales, profundidad, etc.

---

## Cómo compilar la presentación

Carpeta: `Template-Presentacion-dim/`

### Overleaf (recomendado)

1. Subir **toda** la carpeta `Template-Presentacion-dim/` (con `img/` completa).
2. Compilar `main.tex` con pdfLaTeX + **BibTeX** (Recompile from scratch).
3. El paquete `apacite` debe estar disponible (Overleaf lo incluye).

### Local (MiKTeX / TeX Live)

```bash
cd Template-Presentacion-dim
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Salida: `main.pdf`

### Figuras necesarias en `img/`

| Archivo | Uso |
|---------|-----|
| `departamentos/dim.pdf` | Logo DIM (portada) |
| `Erupcion_Volcan_Calbuco_2015.jpg` | Motivación |
| `Modelo_Conducto_1D.png` | Esquema del conducto |
| `Sol_Caso_1D_Estacionario.png` | Resultados estacionarios |
| `Transiente_SinRecarga_CI1.png`, `Ev_flujoMasico_CI1.png` | Transiente CI 1 |
| `Transiente_SinRecarga_CI2.png`, `Ev_FlujoMasico_CI2.png` | Transiente CI 2 |
| `Sol_Conducto_2D.png`, `Sol_Conducto_2D_2.png` | Resultados 2D |

Genera las figuras ejecutando los notebooks/scripts y cópialas a `Template-Presentacion-dim/img/`.

---

## Qué subir a GitHub

### Sí subir (repositorio mínimo recomendado)

| Categoría | Archivos |
|-----------|----------|
| **Raíz** | `README.md`, `requirements.txt`, `.gitignore` |
| **Parámetros y propiedades** | `calbuco2015d.py`, `density.py`, `viscosity.py`, `fvrel.py` |
| **Solvers principales** | `RIconduitex5_5.py`, `RIconduitef5_5.py`, `RIconduit1D_transient.py`, `RIconduit2D_FD.py` |
| **Scripts / notebooks** | `main_transient.py`, `mainconduit_transient_v3.ipynb`, `mainconduit5_7.ipynb`, `mainconduit5_5.ipynb`, `mainconduit_ambos.ipynb` |
| **Presentación** | `Template-Presentacion-dim/main.tex`, `example.tex`, `library.bib`, `template.tex`, `template_config.tex`, `copiar_imagenes.py` |
| **Figuras** | `Template-Presentacion-dim/img/**` (todas las imágenes usadas en la presentación) |

### Opcional (subir si quieres mostrar historial o extras)

- `RIconduit1D_chamber.py`, `mainconduit_chamber.ipynb`
- Otros notebooks (`mainconduit_transient.ipynb`, `mainconduit_transient_v2.ipynb`)
- `examples/sthelens2004b.py`

### No subir (ya están en `.gitignore` o no aportan al repo)

| Tipo | Ejemplos |
|------|----------|
| Auxiliares LaTeX | `*.aux`, `*.log`, `*.nav`, `*.toc`, `*.bbl`, `*.blg`, … |
| Cache Python/Jupyter | `__pycache__/`, `.ipynb_checkpoints/` |
| Entornos virtuales | `.venv/`, `venv/` |
| PDF compilado | `main.pdf` (opcional: puedes subirlo en Releases) |
| Figuras sueltas en la raíz | PNG generados al correr scripts (mejor copiarlos a `img/`) |
| Código legacy / pruebas | `Versiones antiguas/`, `Pruebas/`, `Codigo MatLab/` (salvo que el curso lo pida) |
| Datos Excel | `*.xlsx` de barridos paramétricos |

---

## Crear el repositorio en GitHub

```bash
cd "Codigo Python"
git init
git add README.md requirements.txt .gitignore
git add calbuco2015d.py density.py viscosity.py fvrel.py
git add RIconduitex5_5.py RIconduit1D_transient.py RIconduit2D_FD.py
git add main_transient.py mainconduit_transient_v3.ipynb mainconduit5_7.ipynb mainconduit5_5.ipynb
git add Template-Presentacion-dim/main.tex Template-Presentacion-dim/example.tex
git add Template-Presentacion-dim/library.bib Template-Presentacion-dim/template.tex
git add Template-Presentacion-dim/template_config.tex Template-Presentacion-dim/copiar_imagenes.py
git add Template-Presentacion-dim/img/
git commit -m "Proyecto final: modelo de conducto volcánico 1D/2D y presentación"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/TU_REPO.git
git push -u origin main
```

> Asegúrate de que la carpeta `img/` tenga todas las figuras antes del `git add`. Si faltan, compila primero los notebooks y copia los PNG.

---

## Referencias principales

- Kozono & Koyaguchi (2009) — modelo 1D estacionario bifásico
- Castruccio et al. — parámetros Calbuco / modelo transiente
- Zuber & Findlay (1965) — Drift-Flux
- Giordano et al. (2008) — viscosidad magmática
- LeVeque (2002); Ascher et al. (1995) — métodos numéricos

Lista completa en `Template-Presentacion-dim/library.bib` (formato APA).

---

## Licencia del template LaTeX

La carpeta `Template-Presentacion-dim/` usa el template de [Pablo Pizarro R.](https://latex.ppizarror.com/presentacion) (licencia MIT). Conserva los créditos del template si redistribuyes la presentación.

---

## Problemas frecuentes

| Problema | Solución |
|----------|----------|
| `ModuleNotFoundError: scikits.odes` | `pip install scikit-odes` |
| Referencias vacías en el PDF | Compilar con BibTeX (`bibtex main`) |
| Figuras faltantes en Beamer | Copiar PNG/JPG a `Template-Presentacion-dim/img/` |
| `apacite` no encontrado | Instalar en MiKTeX: `mpm --install=apacite` |
| Logo DIM no aparece | Incluir `img/departamentos/dim.pdf` |
