# Datos de volcanes (actividad 1D)

Los alumnos **solo editan archivos de esta carpeta**. No toquen `RIconduit1D_FD.py` ni los solvers.

| archivo | qué es |
|---|---|
| `calbuco2015.py` | Erupción **explosiva** (sub-Pliniana). Datos dados. |
| `sthelens2004.py` | Erupción **efusiva** (domo 2004). Datos dados. |
| `plantilla_estromboliana.py` | Caso **estromboliano**: copiar a `mi_estromboliana.py` y rellenar. |
| `_constantes.py` | Constantes del modelo. No cambiar. |

Cómo cargar un volcán en el notebook:

```python
from datos import cargar
from RIconduit1D_FD import RIconduit1D_FD_f

volc = cargar("calbuco2015")
out = RIconduit1D_FD_f(volc, rama="explosiva", verbose=False)
```
