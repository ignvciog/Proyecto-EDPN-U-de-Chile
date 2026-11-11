"""
Plantilla para el caso ESTROMBOLIANO.

1. Copien este archivo a  datos/mi_estromboliana.py
2. Rellenen SOLO los campos marcados con  # BUSCAR
3. Pongan LISTO = True
4. En el notebook:  VOLCAN_ESTROMBOLIANA = "mi_estromboliana"

El modelo 1D es estacionario: no resuelve las explosiones discretas de
Stromboli. Con los datos que busquen, el código dice si el conducto admite
rama explosiva, efusiva, las dos, o ninguna. Eso es lo que hay que comentar.

Volcanes razonables: Stromboli, Villarrica, Etna, Pacaya, Parícutin.
Fuentes típicas: análisis de vidrio / inclusiones fundidas, papers de
petrología, Castruccio et al. (2025) JGR (Villarrica está ahí como
hawaiana/estromboliana).
"""
from datos._constantes import *  # noqa: F401,F403

LISTO = False

nombre = "… 20XX"          # BUSCAR  (ej. Stromboli 2007)
estilo_obs = "estromboliana"

# Composición del vidrio (wt%). Suman ~100 sin el agua.
sio2 = None                 # BUSCAR
tio2 = None
al2o3 = None
feo = None
mno = None
mgo = None
cao = None
na2o = None
k2o = None
p2o5 = None
f2o = 0.1

h2o1 = None                 # BUSCAR  wt% H2O (inclusiones fundidas; típ. 0.5–2.5)
Tc1 = None                  # BUSCAR  °C (basalto: ~1050–1150)
xi1 = None                  # BUSCAR  fracción de cristales (0–0.4)

H = None                    # BUSCAR  profundidad de la cámara, en m (negativa, ej. -3000)
geometry = "cylinder"       # BUSCAR  "cylinder" o "dyke"
radius1 = None              # BUSCAR  radio (m) o semi-ancho si es dique
dl = 100                    # solo si geometry == "dyke" (largo del dique, m)
overP1 = None               # BUSCAR  sobrepresión de cámara (Pa). Si no aparece,
                            #         prueben 1e6–5e6 y justifiquen el valor.
rcrust = 2600               # densidad de la corteza (kg/m3); 2500–2700 es típico

# Henry: silicico ya viene de _constantes (C1=4.11e-6, beta=0.5).
# Si SiO2 ≲ 57 wt% (basalto) descomenten las dos líneas:
# C1 = 6.80172e-8
# beta = 0.7
