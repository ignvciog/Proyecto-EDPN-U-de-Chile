"""
Regularizador de la fragmentación (laboratorio).

No toca RIconduitex5_5.py. Reemplaza el salto
    1_{phi >= phicrit}
por una rampa C^infty de ancho eps:

    s_eps(phi) = 1/2 * (1 + tanh((phi - phicrit) / eps))

s=0 es flujo viscoso (burbujas en líquido); s=1 es fragmentado
(partículas en gas). Las fuerzas se mezclan

    F = (1-s) F_visc + s F_frag

Cuando eps -> 0 se recupera el switch. El código original ya interpola
el arrastre en una capa de ancho 0.05; esto es lo mismo, pero con
tanh y también para F_mw / F_gw / dN / d xi.
"""

from __future__ import annotations

import numpy as np

# Ancho por defecto: del orden de la capa que ya usa el original (0.05).
EPS_FRAG = 0.04


def s_frag(phi, phicrit, eps=EPS_FRAG):
    """Peso de la rama fragmentada. Escalar o array."""
    eps = max(float(eps), 1e-12)
    return 0.5 * (1.0 + np.tanh((np.asarray(phi) - phicrit) / eps))


def mezclar(s, visc, frag):
    """(1-s)*visc + s*frag."""
    return (1.0 - s) * visc + s * frag


def fuerzas_mezcla(s, Fmw_v, Fgw_v, Fmg_v, Fmw_f, Fgw_f, Fmg_f):
    return (
        mezclar(s, Fmw_v, Fmw_f),
        mezclar(s, Fgw_v, Fgw_f),
        mezclar(s, Fmg_v, Fmg_f),
    )
