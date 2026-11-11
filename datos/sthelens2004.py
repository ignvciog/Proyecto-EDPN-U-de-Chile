"""
Mount St. Helens 2004 — domo (efusiva).

Datos dados para la actividad. No hace falta buscarlos.
Fuente: Castruccio, Rebolledo & Gómez (2025), JGR Solid Earth
(caso sthelens2004b del modelo RIConduit).
"""
from datos._constantes import *  # noqa: F401,F403

nombre = "St. Helens 2004"
estilo_obs = "domo / efusiva"

sio2 = 73.5
tio2 = 0.2
al2o3 = 13.8
feo = 1.0
mno = 0.02
mgo = 0.08
cao = 1.25
na2o = 4.18
k2o = 3.38
p2o5 = 0.05
f2o = 0.0
h2o1 = 3.0
sio2tot = 64

Tc1 = 950
T1 = Tc1 + 273.15
xi1 = 0.40
xmax = 0.7
tcar = 3600

H = -5000
geometry = "cylinder"
radius1 = 12.11
dl = 100
overP1 = 16.157e6
rcrust = 2500

# Tiro efusivo del archivo original (más estrecho que el default).
vminef = 1e-5
vmaxef = 0.1
vinicialef = 0.01
