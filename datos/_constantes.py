"""Constantes del modelo que los alumnos NO tienen que buscar ni cambiar."""

import numpy as np

R = 461.11
Patm = 1.01325e5
pfinal = Patm
g = 9.81

# Henry silíceo (andesita–riolita). Si el magma es basáltico, el archivo
# de datos debe sobreescribir C1 y beta (ver plantilla_estromboliana.py).
C1 = 0.00000411
beta = 0.5

phicrit = 0.8
xmax = 0.5
phimax = 0.6
tcar = 2 * 3600
model = "em2s"
ar1 = 4
ar2 = 8

F1 = 1.8
F2 = 0.2
Fc = 1.0
lsup = 0.785
linf = 0.525
limperh = 0.4
limperl = 0.15
errtol = 1e-8
