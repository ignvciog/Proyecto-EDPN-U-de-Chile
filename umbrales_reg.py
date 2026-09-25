"""
Rampas para todos los umbrales del DAE (laboratorio).

No toca RIconduitex5_5.py ni RIconduitef5_5.py.
s(x; x*, eps) = 1/2 (1 + tanh((x - x*) / eps))

Cuando eps -> 0 se recupera el if. Los EPS_* de abajo son de partida:
se miran en regularizar_umbrales.ipynb y se ajustan.
"""

from __future__ import annotations

import numpy as np

# ── anchos de partida (los mismos de la nota) ─────────────────────────────
EPS_PHI = 0.01          # limphi1, limphi2
EPS_FRAG = 0.04         # phicrit (fuerzas / n_eq=4)
EPS_HENRY = 1e-3        # test = agua exsuelta (fracción másica)
EPS_RE = 300.0          # Re = 2200
EPS_XI = 1e-3           # softplus de cristales
EPS_RB = 0.05           # fracción de R: guarda r_b < 0.5 R
RE_CRIT = 2200.0
RB_LIM = 0.5            # r_b / R


def s(x, xstar, eps):
    """Rampa 0→1 centrada en xstar. Escalar o array."""
    eps = max(float(eps), 1e-12)
    return 0.5 * (1.0 + np.tanh((np.asarray(x, dtype=float) - xstar) / eps))


def mezclar(w, a, b):
    """(1-w)*a + w*b."""
    return (1.0 - w) * a + w * b


def switch(x, xstar):
    """El if original: 1_{x >= xstar}."""
    return (np.asarray(x, dtype=float) >= xstar).astype(float)


# ── 1–2. regímenes n_eq y fragmentación ───────────────────────────────────

def s_phi1(phi, limphi1, eps=EPS_PHI):
    return s(phi, limphi1, eps)


def s_phi2(phi, limphi2, eps=EPS_PHI):
    return s(phi, limphi2, eps)


def s_frag(phi, phicrit, eps=EPS_FRAG):
    return s(phi, phicrit, eps)


def pesos_regimen(phi, limphi1, limphi2, phicrit,
                  eps_phi=EPS_PHI, eps_frag=EPS_FRAG):
    """Partición de la unidad: w1+w2+w3+w4 = 1. w4 es la rama fragmentada.

    Anidado para que sume 1 aunque limphi1 y limphi2 estén a un eps:
        w1 = 1-s1
        w2 = s1 (1-s2)
        w3 = s1 s2 (1-sf)
        w4 = s1 s2 sf
    """
    s1 = s_phi1(phi, limphi1, eps_phi)
    s2 = s_phi2(phi, limphi2, eps_phi)
    sf = s_frag(phi, phicrit, eps_frag)
    w1 = 1.0 - s1
    w2 = s1 * (1.0 - s2)
    w3 = s1 * s2 * (1.0 - sf)
    w4 = s1 * s2 * sf
    return w1, w2, w3, w4


def mezclar_regimenes(w1, w2, w3, w4, a1, a2, a3, a4):
    return w1 * a1 + w2 * a2 + w3 * a3 + w4 * a4


# ── 3. Henry ──────────────────────────────────────────────────────────────

def s_henry(test, eps=EPS_HENRY):
    """test > 0 → saturado (hay gas). test <= 0 era fg=0."""
    return s(test, 0.0, eps)


# ── 4. Reynolds ───────────────────────────────────────────────────────────

def s_re(Re, recrit=RE_CRIT, eps=EPS_RE):
    """Re > 2200 → arrastre inercial."""
    return s(Re, recrit, eps)


# ── 5. cristales: softplus en vez de max(0, ·) ────────────────────────────

def softplus(x, eps=EPS_XI):
    """max(0,x) suave. Estable para |x|/eps grande."""
    eps = max(float(eps), 1e-12)
    x = np.asarray(x, dtype=float)
    z = x / eps
    mid = eps * np.log1p(np.exp(np.clip(z, -40.0, 40.0)))
    return np.where(z > 40.0, x, np.where(z < -40.0, 0.0, mid))


# ── 6. coalescencia (rampas que APAGAN) ───────────────────────────────────

def g_rb(rb, wr, eps_frac=EPS_RB, rlim=RB_LIM):
    """1 si r_b << 0.5 R, 0 si r_b >> 0.5 R."""
    return 1.0 - s(rb, rlim * wr, eps_frac * wr)


def g_phi(phi, phicrit, eps=EPS_FRAG):
    """1 si phi << phicrit, 0 si ya fragmentó."""
    return 1.0 - s_frag(phi, phicrit, eps)


def guarda_coalescencia(rb, wr, phi, phicrit,
                        eps_frac=EPS_RB, eps_frag=EPS_FRAG):
    return g_rb(rb, wr, eps_frac) * g_phi(phi, phicrit, eps_frag)


# ── 8. residuales del tiro (no son tanh en phi) ───────────────────────────

def softmin2(a, b, eps=1.0):
    """min(a,b) suave: -eps log(e^{-a/eps} + e^{-b/eps})."""
    eps = max(float(eps), 1e-12)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.minimum(a, b)
    ea = np.exp(np.clip(-(a - m) / eps, -40.0, 0.0))
    eb = np.exp(np.clip(-(b - m) / eps, -40.0, 0.0))
    return m - eps * np.log(ea + eb)


def residual_explosivo(P, Patm, ug, cs, phi, phicrit, zexit,
                       eps_P=5e3, eps_c=0.05 * 370.0, eps_phi=EPS_FRAG, eps_z=1.0):
    """
    Componentes del cierre explosivo (ventanas del código original).
    r_phi y r_z son 0 si se cumple la desigualdad, >0 si se viola.
    r_or = softmin(r_P^2, r_c^2): P_atm O choque.
    """
    r_P = P - Patm
    r_c = ug - cs
    r_phi = np.maximum(0.0, phicrit - phi)          # quiere phi >= phicrit
    r_z = np.maximum(0.0, -5.0 - zexit)             # quiere z >= -5
    r_or = softmin2(r_P ** 2, r_c ** 2, eps=eps_P ** 2)
    return {"r_P": r_P, "r_c": r_c, "r_phi": r_phi, "r_z": r_z, "r_or": r_or}


def ancho_10_90(x, y):
    """Ancho de la rampa entre 0.1 y 0.9 (qué tan pegada queda al switch)."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y[-1] < y[0]:
        y = 1.0 - y
    i_lo = int(np.argmax(y >= 0.1))
    i_hi = int(np.argmax(y >= 0.9))
    return float(x[i_hi] - x[i_lo])


def residual_efusivo(P, Patm, phi, phicrit, zexit):
    r_P = P - Patm
    r_phi = np.maximum(0.0, phi - phicrit)          # quiere phi <= phicrit
    # dist a (-2, 0]: 0 adentro
    r_z = np.where(zexit > 0.0, zexit,
                   np.where(zexit < -2.0, -2.0 - zexit, 0.0))
    return {"r_P": r_P, "r_phi": r_phi, "r_z": r_z}
