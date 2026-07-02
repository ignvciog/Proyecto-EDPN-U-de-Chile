"""
RIconduit1D_transient.py
========================
Modelo de conducto volcánico 1D transiente — Método de Líneas (MOL)

Sistema de EDPs (variable independiente: z ∈ [H, 0], t ≥ 0)
─────────────────────────────────────────────────────────────
    ∂φ/∂t  + um ∂φ/∂z  = S_φ(P,φ,Nd,x)     [fracción de burbujas/gas]
    ∂Nd/∂t + um ∂Nd/∂z = Γ_N(φ,Nd,P,um)    [coalescencia de burbujas]
    ∂x/∂t  + um ∂x/∂z  = um·(dx/dz)_kin    [cristales]

Física del flujo — dos regímenes
─────────────────────────────────
  φ < φ_crit : HEM  → ug = um = q/ρ_mix,  S_φ = (φ_eq−φ)/τ_relax
  φ ≥ φ_crit : Drift-Flux (Zuber & Findlay 1965)
               u_mix = q/ρ_mix  (mezcla incompresible)
               u_g  = u_mix + (1−φ)·ρm/ρ_mix · v_slip   [gas más rápido]
               u_m  = u_mix −  φ   ·ρg/ρ_mix · v_slip   [piroclastos]
               v_slip = √[4·dp·(ρm−ρg)·g/(3·ρg·Cd)]    [terminal Newton]
               Propiedad clave: φ·ρg·ug + (1−φ)·ρm·um = q  [conserva masa]
               v_slip ≈ 5–50 m/s → NO supersónico (a diferencia de
               ug = fg·q/(ρg·φ) del modelo incompresible puro).
  dP/dz = −ρ_mix·g − cg·μ·u_mix/wr²  [mezcla, ambos regímenes]
  dNdt  = coalescencia   si φ < φ_crit;  = 0  si φ ≥ φ_crit

Discretización numérica
────────────────────────
    Espacio : diferencias finitas upwind (N_z nodos, flujo ascendente)
              ∂f/∂z|_j ≈ (f_j − f_{j−1}) / Δz
    Tiempo  : IMEX — RK4 advección + relajación exponencial exacta
              Δt ≤ CFL · Δz / um_max  (solo depende de advección, no de τ)

Basado en Castruccio et al. (2025) / Kozono & Koyaguchi (2009).
"""

import numpy as np
import math
import warnings
warnings.filterwarnings("ignore")

from density import density
from viscosity import viscosity
from fvrel import fvrel
from calbuco2015d import (
    sio2, tio2, al2o3, feo, mno, mgo, cao, na2o, k2o, p2o5, f2o,
    C1, beta, R, Patm, g, H, rcrust,
    F1, F2, Fc, limperh, limperl,
    tcar, xmax, model, ar1, ar2, geometry,
    overP1, radius1, h2o1, T1, xi1,
)


# ══════════════════════════════════════════════════════════════════════════════
# FÍSICA LOCAL (un solo nodo z)
# ══════════════════════════════════════════════════════════════════════════════

def _fg(P, x_cr, co, xi):
    """Fracción másica de gas exsuelto fg(P, x)."""
    test = (1 - xi)*co - (1 - x_cr)*C1*P**beta
    if test <= 0:
        return 0.0
    return ((1 - xi)*co - C1*(1 - x_cr)*P**beta) / (1 - C1*P**beta)


def _equil_phi(P, x_cr, co, xi, rho_m, T):
    """
    Fraccion volumetrica de burbujas en equilibrio termodinamico.
    Misma formula que RIconduitex5_5_f para phini.
    Evita la singularidad ug → ∞ cuando φ → 0 con fg > 0.
    """
    fg = _fg(P, x_cr, co, xi)
    if fg <= 1e-10:
        return 0.0
    return 1.0 / (1.0 + P*(1 - fg)/(fg*R*T*rho_m))


def _dfgdp(P, x_cr, fg, um, co, xi):
    """∂fg/∂P — coeficiente para los gradientes de presión y fracción de gas."""
    if fg <= 0:
        return 0.0
    den = (co*(1-xi) - (1-xmax)*C1*Patm**beta) + 1e-30
    f2  = max(0.0, (co*(1-xi) - (1-x_cr)*C1*P**beta) / den)
    xteo = xi + (xmax - xi)*f2
    f3   = max(0.0, 1 - x_cr / max(xteo, 1e-10))
    dxdp = max(0.0, (xmax - xi)*f2*f3 / (tcar * max(um, 1e-4)))
    num  = (- (-dxdp*C1*P**beta + (1-x_cr)*C1*beta*P**(beta-1))
            + (co*(1-xi) - (1-x_cr)*C1*P**beta)*C1*beta*P**(beta-1))
    return num / (1 - C1*P**beta)**2


def _visc(P, phi, Nd, x_cr, um, wr, co, xi, h2o_wt, Tc):
    """Viscosidad efectiva: Giordano + cristales (ER) + burbujas (Llewellin)."""
    h2o_d = ((C1*P**beta)*100
             if (1-xi)*co - (1-x_cr)*C1*P**beta > 0
             else h2o_wt)
    viscl = (fvrel(model, x_cr, xi, ar1, ar2, xmax, um/wr)
             * viscosity(sio2, tio2, al2o3, feo, mno, mgo, cao, na2o, k2o, p2o5,
                         h2o_d, f2o, Tc))
    rb_est = max(1e-8, ((phi / ((4/3)*np.pi*max(Nd, 1.0)*(1 - phi)))**(1/3)))
    nca    = rb_est * viscl * max(um/wr, 1e-10) / 3

    pcb = 0.75
    if phi >= pcb*0.999:
        return max(viscl, 1.0)
    AA  = (1 - phi/pcb)**(-pcb)
    BB  = (1 - phi/pcb)**(5*pcb/3)
    c1  = -0.2895*phi + 0.8132
    c2  =  phi
    viscrel = 0.5*(AA - BB)*(1 - math.erf(np.real(c1*np.log(max(nca, 1e-300))+c2))) + BB
    return max(viscrel*viscl, 1.0)


def _Fmg(phi, ug, um, rb, rho_g, visc, limphi1, limphi2, n_eq):
    """Fuerza de arrastre intrafase gas–fundido."""
    if n_eq == 1:
        return 3*visc*(ug - um)*phi*(1 - phi) / rb**2
    if n_eq == 2:
        tt   = (phi - limphi1)/(limphi2 - limphi1)
        Re   = 2*rb*rho_g*abs(ug - um)/1e-5
        if Re > 2200:
            return ((0.33/(4*rb))*rho_g*abs(ug-um)**tt
                    * (3*visc/rb**2)**(1-tt) * (ug-um)*phi*(1-phi))
        kper = 0.131*rb**2*((phi - limphi1 + 0.05)**2.1)
        return (1e-5/kper)**tt * (3*visc/rb**2)**(1-tt) * (ug-um)*phi*(1-phi)
    # n_eq == 3
    Re = 2*rb*rho_g*abs(ug - um)/1e-5
    if Re > 2200:
        return (0.33/(4*rb))*rho_g*(ug-um)**2*phi*(1-phi)
    kper = 0.131*rb**2*((phi - limphi1 + 0.05)**2.1)
    return (1e-5/kper)*(ug - um)*phi*(1-phi)


def _local_phys(P, phi, Nd, x_cr, q, rho_m, wr, T, Tc, co, xi, h2o_wt,
                phicrit, cg, tau_relax):
    """
    Física local en un nodo z.

    PRE-FRAGMENTACIÓN  (φ < φ_crit)  — HEM puro
    ────────────────────────────────────────────
        ug = um = q / ρ_mix
        dP/dz = −ρ_mix·g − cg·μ·um/wr²
        S_φ   = (φ_eq − φ) / τ_relax
        dNdt  = coalescencia Stokes

    POST-FRAGMENTACIÓN (φ ≥ φ_crit) — Drift-Flux (Zuber & Findlay 1965)
    ────────────────────────────────────────────────────────────────────
    La mezcla sigue incompresible (u_mix = q/ρ_mix), pero el gas es más
    rápido que los piroclastos por flotabilidad (drift).  El modelo de
    flujo con deriva da velocidades FÍSICAS sin resolver PDEs de momentum:

        u_g  = u_mix + (1−φ)·ρm/ρ_mix · v_slip   [gas sube más rápido]
        u_m  = u_mix − φ   ·ρg/ρ_mix · v_slip   [piroclastos más lentos]

    donde la velocidad de deslizamiento viene del equilibrio arrastre ↔
    flotabilidad (régimen Newton, piroclasto en gas):

        v_slip = √[ 4·d_p·(ρm − ρg)·g / (3·ρg·Cd) ]   [m/s]

    Esta formulación conserva el flujo másico total automáticamente:
        φ·ρg·ug + (1−φ)·ρm·um = ρ_mix·u_mix = q

    v_slip es de decenas de m/s (razonable), no supersónica como
    ug = fg·q/(ρg·φ) del modelo incompresible puro.

    Cap de velocidad: evita ρ_mix→0 si φ→0.99 (u_mix ≤ 350 m/s).

    Retorna dict con: um, ug, dpdz, src_phi, dNdt, dxdz.
    """
    _UM_MAX  = 350.0    # cap sónico de la mezcla [m/s]
    _DP      = 2e-3     # diámetro piroclasto [m] (2 mm)
    _CD      = 0.8      # coeficiente arrastre (esferas, régimen Newton)
    _BLEND   = 0.05     # ancho de zona de transición en φ

    P    = max(float(P),    Patm)
    phi  = float(np.clip(phi, 0.0, 0.99))
    Nd   = max(float(Nd),   1.0)
    x_cr = float(np.clip(x_cr, xi, xmax))

    rho_g = P / (R*T)

    # ── Cinética de cristalización ─────────────────────────────────────────
    den2 = (co*(1-xi) - (1-xmax)*C1*Patm**beta) + 1e-30
    f2_x = max(0.0, (co*(1-xi) - (1-x_cr)*C1*P**beta) / den2)
    xteo = xi + (xmax - xi)*f2_x
    f3_x = max(0.0, 1.0 - x_cr / max(xteo, 1e-10))
    dxdz = max(0.0, (xmax - xi)*f2_x*f3_x / tcar)

    # ── Velocidad de la mezcla (HEM incompresible, cap sónico) ────────────
    rho_mix = rho_m*(1.0 - phi) + rho_g*phi
    rho_mix = max(rho_mix, q / _UM_MAX)
    u_mix   = q / rho_mix

    # ── Drift-Flux post-fragmentación ─────────────────────────────────────
    if phi >= phicrit:
        # Velocidad terminal del piroclasto en el gas (régimen Newton)
        delta_rho = max(rho_m - rho_g, 0.0)
        v_slip_full = math.sqrt(4.0 * _DP * delta_rho * g
                                / max(3.0 * rho_g * _CD, 1e-6))
        v_slip_full = min(v_slip_full, 200.0)   # cap físico

        # Transición suave desde HEM en φ_crit (evita discontinuidad)
        tt     = min((phi - phicrit) / _BLEND, 1.0)
        v_slip = tt * v_slip_full

        # Velocidades de cada fase (conservan φ·ρg·ug + (1-φ)·ρm·um = q)
        um = u_mix - phi   * (rho_g / rho_mix) * v_slip
        ug = u_mix + (1.0 - phi) * (rho_m / rho_mix) * v_slip
    else:
        um = u_mix
        ug = u_mix

    # ── Gradiente de presión (HEM para la mezcla) ─────────────────────────
    # Pre-frag:  fricción viscosa  Hagen-Poiseuille  cg·μ·u/wr²
    # Post-frag: fricción turbulenta gas-piroclastos  0.01·ρ·u²/(4·wr)
    #   Reproduciría el Fgw del modelo estacionario n_eq=4 usando u_mix.
    visc    = _visc(P, phi, Nd, x_cr, u_mix, wr, co, xi, h2o_wt, Tc)
    if phi < phicrit:
        Fmw = cg * visc * u_mix / wr**2
    else:
        Fmw = 0.01 * rho_mix * u_mix**2 / (4.0 * wr)
    dpdz    = -rho_mix*g - Fmw

    # ── Fuente φ: relajación termodinámica ────────────────────────────────
    phi_eq  = _equil_phi(P, x_cr, co, xi, rho_m, T)
    src_phi = (phi_eq - phi) / tau_relax

    # ── Dinámica de burbujas ───────────────────────────────────────────────
    if phi >= phicrit:
        dNdt = 0.0   # sin coalescencia post-fragmentación
    else:
        phi_pos    = max(phi, 1e-8)
        rb         = max(1e-8, (phi_pos / ((4/3)*np.pi*Nd*(1.0 - phi_pos)))**(1/3))
        arg_coal   = (F1+F2) * ((3*phi_pos*np.pi / (6*phicrit*4*np.pi))**(1/3))
        denom_coal = 1.0 - arg_coal
        if rb < 0.5*wr and phi_pos < phicrit and denom_coal > 0.01:
            dNdt = (-(Nd**(2/3)) * ((1.0/(1.0 - phi_pos))**(1/3))
                    * ((1.0/9.0)*(rho_m - rho_g)*9.81 / visc)
                    * ((3.0*phi_pos/(4*np.pi))**(2/3))
                    * (F1**2 - F2**2) / denom_coal
                    * Fc*(1.0 - phi_pos/phicrit)*(wr - rb)/wr)
        else:
            dNdt = 0.0

    return dict(um=um, ug=ug, dpdz=dpdz, src_phi=src_phi, dNdt=dNdt, dxdz=dxdz)


# ══════════════════════════════════════════════════════════════════════════════
# INTEGRACIÓN CUASIESTÁTICA DE P(z) — marcha ascendente j = 0 → N-1
# ══════════════════════════════════════════════════════════════════════════════

def _pressure_march(phi_arr, Nd_arr, x_arr, z_arr, Pi,
                    q, rho_m, wr, T, Tc, co, xi, h2o_wt,
                    phicrit, cg, tau_relax):
    """
    Integra dP/dz ascendente (j=0 cámara → j=N-1 vent).

    HEM puro: dP/dz = -ρ_mix·g - Fmw,  ug = um en todos los nodos.
    La fragmentación es un diagnóstico (φ ≥ φ_crit), no cambia la física.

    Retorna (P_arr, um_arr, ug_arr, lp_list).
    """
    N   = len(z_arr)
    dz  = z_arr[1] - z_arr[0]
    P   = np.empty(N);  P[0] = Pi
    um  = np.empty(N)
    ug  = np.empty(N)
    lps = []

    for j in range(N):
        lp = _local_phys(P[j], phi_arr[j], Nd_arr[j], x_arr[j],
                         q, rho_m, wr, T, Tc, co, xi, h2o_wt,
                         phicrit, cg, tau_relax)
        um[j] = lp['um']
        ug[j] = lp['ug']
        lps.append(lp)
        if j < N - 1:
            P[j+1] = max(P[j] + lp['dpdz']*dz, Patm)

    return P, um, ug, lps


# ══════════════════════════════════════════════════════════════════════════════
# BÚSQUEDA DE q QUE SATISFACE P_vent = P_atm  (método de bisección)
# ══════════════════════════════════════════════════════════════════════════════

def _find_q(phi_arr, Nd_arr, x_arr, z_arr, Pi,
            q0, rho_m, wr, T, Tc, co, xi, h2o_wt,
            phicrit, cg, tau_relax):
    """
    Encuentra el flujo másico q tal que P_vent = P_atm.

    Físicamente: el flujo se ajusta para balancear la presión de cámara
    (Pi) con la presión atmosférica en el vent (P_atm). Esta es la
    condición de borde correcta para un conducto abierto.

    Usa bisección en escala logarítmica con tolerancia 1e4 Pa (0.1 bar).
    Retorna q (float).
    """
    N  = len(z_arr)
    dz = z_arr[1] - z_arr[0]

    def P_vent_of_q(q):
        """P_vent sin clamping para poder hacer bisección."""
        P = Pi
        for j in range(N - 1):
            lp = _local_phys(P, phi_arr[j], Nd_arr[j], x_arr[j],
                             q, rho_m, wr, T, Tc, co, xi, h2o_wt,
                             phicrit, cg, tau_relax)
            P = P + lp['dpdz'] * dz
            if P < 0:
                return 0.0   # q demasiado alto: presión se volvió negativa
        return P

    # residual(q) = P_vent(q) - Patm
    # Mayor q → mayor fricción → mayor caída de presión → P_vent más bajo
    # Buscamos el q donde residual = 0

    # Bracket: desde q muy bajo (poco drop, P_vent > Patm)
    #          hasta q muy alto (mucho drop, P_vent < Patm)
    q_lo = max(q0 * 1e-3, 10.0)
    q_hi = q0 * 100.0

    r_lo = P_vent_of_q(q_lo) - Patm
    r_hi = P_vent_of_q(q_hi) - Patm

    # Expandir bracket si es necesario
    for _ in range(10):
        if r_lo * r_hi <= 0:
            break
        if r_lo > 0:          # ambos dan P_vent > Patm → subir q_hi
            q_hi *= 10.0
            r_hi = P_vent_of_q(q_hi) - Patm
        else:                  # ambos dan P_vent < Patm → bajar q_lo
            q_lo /= 10.0
            r_lo = P_vent_of_q(q_lo) - Patm

    if r_lo * r_hi > 0:
        # No se pudo establecer bracket; retornar q0 como fallback
        return q0

    # Bisección en escala log para convergencia uniforme
    for _ in range(60):
        q_mid = math.sqrt(q_lo * q_hi)   # midpoint en escala log
        r_mid = P_vent_of_q(q_mid) - Patm
        if abs(r_mid) < 1e4:             # tolerancia 0.1 bar
            return q_mid
        if r_lo * r_mid < 0:
            q_hi, r_hi = q_mid, r_mid
        else:
            q_lo, r_lo = q_mid, r_mid

    return math.sqrt(q_lo * q_hi)


# ══════════════════════════════════════════════════════════════════════════════
# RHS DEL SISTEMA MOL (3 ecuaciones × N_z nodos)
# ══════════════════════════════════════════════════════════════════════════════

def _mol_rhs(phi, Nd, x, z_arr, Pi,
             q, rho_m, wr, T, Tc, co, xi, h2o_wt,
             phicrit, cg, tau_relax,
             phi_bc, Nd_bc, x_bc,
             advection_only=False):
    """
    Calcula ∂φ/∂t, ∂Nd/∂t, ∂x/∂t en cada nodo mediante FD upwind.

    Esquema upwind (flujo ascendente, um > 0):
        ∂f/∂z|_j ≈ (f_j − f_{j−1}) / Δz

        ∂φ/∂t  = S_φ(j)         − um·(φ_j − φ_{j−1})/Δz
        ∂Nd/∂t = Γ_N(j)         − um·(Nd_j − Nd_{j−1})/Δz
        ∂x/∂t  = um·(dx/dz)_kin − um·(x_j  − x_{j−1})/Δz

    advection_only=True : S_φ = 0 (sólo advección). Usado en el esquema
        IMEX: la fuente rígida (φ_eq−φ)/τ se aplica después con la
        solución exponencial exacta, evitando la condición Δt < 2τ.
    """
    N  = len(z_arr)
    dz = z_arr[1] - z_arr[0]

    _, um_arr, ug_arr, lps = _pressure_march(phi, Nd, x, z_arr, Pi,
                                             q, rho_m, wr, T, Tc, co, xi, h2o_wt,
                                             phicrit, cg, tau_relax)

    dphi = np.zeros(N)
    dNd  = np.zeros(N)
    dx   = np.zeros(N)

    for j in range(1, N):
        um_j = um_arr[j]
        lp   = lps[j]

        src_phi = 0.0 if advection_only else lp['src_phi']
        dphi[j] = src_phi           - um_j*(phi[j] - phi[j-1])/dz
        dNd[j]  = lp['dNdt']        - um_j*(Nd[j]  - Nd[j-1])/dz
        dx[j]   = um_j*lp['dxdz']  - um_j*(x[j]   - x[j-1])/dz

    # Nodo 0: condicion de borde fija (camara)
    dphi[0] = 0.0
    dNd[0]  = 0.0
    dx[0]   = 0.0

    return dphi, dNd, dx


# ══════════════════════════════════════════════════════════════════════════════
# UN PASO RK4
# ══════════════════════════════════════════════════════════════════════════════

def _rk4(phi, Nd, x, dt, z_arr, Pi,
         q, rho_m, wr, T, Tc, co, xi, h2o_wt,
         phicrit, cg, tau_relax,
         phi_bc, Nd_bc, x_bc,
         allow_frag_healing=False):
    """
    Avanza [φ, Nd, x] un paso IMEX (splitting de operadores):

      Paso 1 — RK4 para advección pura  (sin fuente rígida S_φ)
      Paso 2 — Relajación exponencial exacta para S_φ = (φ_eq−φ)/τ

    La relajación exacta:
        φ^{n+1} = φ_eq(P) + (φ^* − φ_eq(P)) · exp(−Δt/τ)

    es incondicionalmente estable para cualquier τ y Δt:
      · τ grande  → exp ≈ 1  → φ apenas cambia (relajación lenta)
      · τ → 0     → exp → 0  → φ → φ_eq instántaneamente (estacionario)

    allow_frag_healing=True (modelo cámara, ΔP≈0):
      Sin relajación ni irreversibilidad — sólo advección.  El magma
      pobre en gas (φ_bc≈0) desde la cámara desplaza y re-sella la zona
      fragmentada.  Necesario porque φ_eq(P) es alto en el vent aunque
      la cámara ya no tenga sobrepresión.
    """
    _args = (z_arr, Pi, q, rho_m, wr, T, Tc, co, xi, h2o_wt,
             phicrit, cg, tau_relax, phi_bc, Nd_bc, x_bc)

    def F(ph_, Nd_, x_):
        # Sólo advección: la fuente rígida se aplica en el Paso 2
        return _mol_rhs(ph_, Nd_, x_, *_args, advection_only=True)

    # ── Paso 1: RK4 advección pura ────────────────────────────────────────
    k1p, k1N, k1x = F(phi, Nd, x)
    k2p, k2N, k2x = F(phi+0.5*dt*k1p, Nd+0.5*dt*k1N, x+0.5*dt*k1x)
    k3p, k3N, k3x = F(phi+0.5*dt*k2p, Nd+0.5*dt*k2N, x+0.5*dt*k2x)
    k4p, k4N, k4x = F(phi+dt*k3p,     Nd+dt*k3N,     x+dt*k3x)

    phi_adv = np.clip(phi + (dt/6)*(k1p+2*k2p+2*k3p+k4p), 0.0, 0.99)
    Nd_new  = np.maximum(Nd + (dt/6)*(k1N+2*k2N+2*k3N+k4N), 1e4)
    x_new   = np.clip(x  + (dt/6)*(k1x+2*k2x+2*k3x+k4x), xi, xmax)

    # ── Paso 2: relajación (o sólo advección si cámara agotada) ─────────
    if allow_frag_healing:
        phi_new = phi_adv.copy()
    else:
        P_arr, _, _, _ = _pressure_march(phi_adv, Nd_new, x_new, z_arr, Pi,
                                         q, rho_m, wr, T, Tc, co, xi, h2o_wt,
                                         phicrit, cg, tau_relax)
        decay   = math.exp(-dt / max(tau_relax, 1e-12))
        phi_new = np.empty_like(phi_adv)
        for j in range(len(z_arr)):
            phi_eq_j   = _equil_phi(P_arr[j], x_new[j], co, xi, rho_m, T)
            phi_new[j] = phi_eq_j + (phi_adv[j] - phi_eq_j) * decay
            if phi_adv[j] >= phicrit and phi_eq_j >= phicrit:
                phi_new[j] = max(phi_new[j], phi_adv[j])
    phi_new = np.clip(phi_new, 0.0, 0.99)

    # ── Condiciones de borde en j=0 ───────────────────────────────────────
    phi_new[0] = phi_bc
    Nd_new[0]  = Nd_bc
    x_new[0]   = x_bc

    return phi_new, Nd_new, x_new


# ══════════════════════════════════════════════════════════════════════════════
# SOLVER PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def RIconduit1D_transient_f(radius, Pressure, wt, Temperature, content_crystal,
                             N_z=200, T_max=3600.0, save_every=50,
                             CFL=0.4, tau_relax=200.0, vinicial_override=None,
                             phicrit_override=None, stationary_ic=None):
    """
    Modelo de conducto volcánico 1D transiente con fragmentación.

    Resuelve el sistema de EDPs por el Método de Líneas (MOL):

        ∂φ/∂t  + um ∂φ/∂z  = S_φ(P,φ,Nd,x)        [fracción gas/burbujas]
        ∂Nd/∂t + um ∂Nd/∂z = Γ_N(φ,Nd,P)           [coalescencia de burbujas]
        ∂x/∂t  + um ∂x/∂z  = um·(dx/dz)_kin(P,x)   [cristalización]

    Dos regímenes físicos automáticos:
        φ < φ_crit : HEM   — dP/dz = -ρ_mix·g - Fmw,  S_φ = (φ_eq−φ)/τ_relax
        φ ≥ φ_crit : n_eq=4 — um/ug separados, Fmg Newton, dP/dz 2-fluidos

    Discretización:
        Espacio z : FD upwind (∂f/∂z|_j ≈ (f_j−f_{j-1})/Δz)
        Tiempo  t : RK4 clásico
        Paso    Δt: adaptativo, CFL·Δz/max(um, ug)

    Condición inicial  : φ = 0 (pre-erupción, sin burbujas)
    Condición de borde : j=0 fijo (cámara); j=N-1 outflow libre

    Parámetros
    ----------
    radius, Pressure, wt, Temperature, content_crystal
        Mismos que RIconduitex5_5_f.
    N_z        : número de nodos espaciales.
    T_max      : tiempo total de simulación [s].
    save_every : guardar snapshot cada N pasos.
    CFL        : número de Courant (recomendado 0.3–0.5).
    tau_relax  : tiempo de relajación para la exsolución de gas [s].
                 El esquema IMEX hace esto incondicionalmente estable:
                   τ → 0  : φ = φ_eq(P) instantáneo → replica estacionario
                   τ = 1  : exsolución casi instantánea (recomendado)
                   τ grande: exsolución lenta (frente de frag cerca del vent)
    phicrit_override : umbral de fragmentación (default 0.75). Pasar el
                       mismo valor que usa RIconduitex5_5_f (calbuco: 0.8).
    vinicial_override: velocidad inicial [m/s] en z=H.
                       Si None, usa 10 m/s (rama efusiva / guess inicial).
                       Para explorar la rama EXPLOSIVA, pasar la vinicial
                       convergida por RIconduitex5_5_f (rama explosiva).
                       IMPORTANTE: q = vinicial_override * rho_m determina
                       en qué rama del sistema bistable converge la solución.

    Retorna
    -------
    dict con claves: t_save, z, phi_hist, P_hist, um_hist, ug_hist, Nd_hist, x_hist,
                     q, vinicial.
    """
    # ── Parámetros del run ────────────────────────────────────────────────
    wr     = radius
    T      = Temperature
    Tc     = Temperature - 273.15
    co     = wt / 100.0
    xi     = content_crystal
    h2o_wt = wt
    cg     = 3 if geometry == 'dyke' else 8

    # Presión en la base del conducto (cámara)
    Pi  = rcrust*g*abs(H) + Pressure
    dis = min(C1*Pi**beta, co)
    rho_m = density(sio2, tio2, al2o3, feo, mgo, cao, na2o, k2o,
                    dis*100, Tc, Pi/1e6)

    # Condiciones iniciales en la cámara
    exi = (1 - xi)*(co - C1*Pi**beta) / (1 - C1*Pi**beta)
    if exi <= 0:
        fgi, phini, rho_ti = 0.0, 1e-8, rho_m
    else:
        fgi    = (1 - xi)*(co - C1*Pi**beta) / (1 - C1*Pi**beta)
        phini  = 1.0 / (1.0 + (Pi/(fgi*R*T))*(1 - fgi)/rho_m)
        rho_ti = Pi*rho_m / (rho_m*R*T*fgi + Pi*(1 - fgi))

    # Flujo másico total q [kg/m²/s]
    # -------------------------------------------------------------------
    # BISTABILIDAD: el sistema tiene dos ramas estables para los mismos
    # parámetros (efusiva y explosiva). El valor de q determina en cuál
    # converge la solución transiente.
    #
    # Rama EFUSIVA  : vinicial ≈ 10 m/s (guess inicial) → φ_max < φ_crit
    # Rama EXPLOSIVA: vinicial = valor convergido de RIconduitex5_5_f
    #                 → φ_max → φ_crit,  P_vent → P_atm
    # -------------------------------------------------------------------
    vinicial = float(vinicial_override) if vinicial_override is not None else 10.0
    q = vinicial * rho_ti   # flujo másico: usa densidad real de la mezcla inlet

    # Densidad de nucleacion inicial (estimacion conservadora)
    Nd_init = 1e10

    phicrit = float(phicrit_override) if phicrit_override is not None else 0.75

    # ── Grilla espacial: j=0 en z=H (cámara), j=N-1 en z=0 (superficie) ─
    z_arr = np.linspace(H, 0.0, N_z)
    dz    = z_arr[1] - z_arr[0]    # Δz > 0

    # ── Condición inicial ─────────────────────────────────────────────────
    if stationary_ic is not None:
        # CI = solución estacionaria interpolada a la grilla uniforme.
        # Permite: (1) test de consistencia, (2) análisis de estabilidad,
        # (3) evitar el spin-up desde φ=0 cuando se conoce la rama explosiva.
        phi_arr = np.clip(np.asarray(stationary_ic['phi'], dtype=float), 0.0, 0.99)
        Nd_arr  = np.maximum(np.asarray(stationary_ic['Nd'],  dtype=float), 1e4)
        x_arr   = np.clip(np.asarray(stationary_ic['x'],   dtype=float), float(xi), xmax)
        if len(phi_arr) != N_z:
            raise ValueError(f"stationary_ic debe tener N_z={N_z} elementos, "
                             f"pero tiene {len(phi_arr)}. "
                             "Interpola el perfil estacionario a la grilla uniforme antes de pasarlo.")
        ci_label = "solución estacionaria interpolada"
    else:
        # CI por defecto: φ = 0 (pre-erupción, sin burbujas).
        # A t=0+, src_phi = phi_eq/tau_relax > 0 y las burbujas empiezan a crecer.
        phi_arr = np.zeros(N_z)
        Nd_arr  = np.full(N_z, Nd_init)
        x_arr   = np.full(N_z, float(xi))
        ci_label = "phi=0 (pre-erupcion, sin burbujas)"

    # ── Condiciones de borde (cámara, j=0, Dirichlet) ────────────────────
    phi_bc = phi_arr[0]   # mantener la CB consistente con la CI
    Nd_bc  = Nd_arr[0]
    x_bc   = x_arr[0]

    # ── Calcular q inicial satisfaciendo P_vent = P_atm ─────────────────
    # Con phi=0 (CI), q_inicial es el flujo mínimo necesario para llevar
    # la presión desde Pi hasta P_atm en un conducto sin burbujas.
    q = _find_q(phi_arr, Nd_arr, x_arr, z_arr, Pi,
                q, rho_m, wr, T, Tc, co, xi, h2o_wt,
                phicrit, cg, tau_relax)

    # Perfil de P, um, ug inicial (cuasiestático con phi=0, q ajustado)
    P_arr, um_arr, ug_arr, _ = _pressure_march(
        phi_arr, Nd_arr, x_arr, z_arr, Pi,
        q, rho_m, wr, T, Tc, co, xi, h2o_wt,
        phicrit, cg, tau_relax)

    # ── Almacenamiento ────────────────────────────────────────────────────
    t_save   = [0.0]
    phi_hist = [phi_arr.copy()]
    P_hist   = [P_arr.copy()]
    um_hist  = [um_arr.copy()]
    ug_hist  = [ug_arr.copy()]
    Nd_hist  = [Nd_arr.copy()]
    x_hist   = [x_arr.copy()]
    q_hist   = [q]

    print("═"*62)
    print("  RIconduit1D_transient (HEM + q dinámico) — Parámetros")
    print(f"  Pi      = {Pi:.3e} Pa     ρm  = {rho_m:.0f} kg/m³")
    print(f"  q(t=0)  = {q:.1f} kg/m²/s   τ_relax = {tau_relax:.0f} s")
    print(f"  N_z     = {N_z}     Δz  = {dz:.1f} m")
    print(f"  T_max   = {T_max:.0f} s     CFL = {CFL}")
    print(f"  φ_crit  = {phicrit:.2f}  (fragmentación)")
    print(f"  CI      : {ci_label}")
    print(f"  P_vent(t=0) = {P_arr[-1]/1e5:.2f} bar  (debe ser ~1 bar)")
    print("═"*62)

    # ── Integración temporal ──────────────────────────────────────────────
    t    = 0.0
    step = 0

    while t < T_max - 1e-10:
        # Recalcular q para que P_vent = P_atm con el perfil φ actual
        q = _find_q(phi_arr, Nd_arr, x_arr, z_arr, Pi,
                    q, rho_m, wr, T, Tc, co, xi, h2o_wt,
                    phicrit, cg, tau_relax)

        # CFL con velocidad máxima de AMBAS fases (um piroclastos, ug gas)
        _, um_cfl, ug_cfl, _ = _pressure_march(
            phi_arr, Nd_arr, x_arr, z_arr, Pi,
            q, rho_m, wr, T, Tc, co, xi, h2o_wt,
            phicrit, cg, tau_relax)
        u_max = max(np.max(np.abs(um_cfl)), np.max(np.abs(ug_cfl)), 0.01)
        dt    = min(CFL * dz / u_max, T_max - t)

        # Avance RK4 IMEX
        phi_arr, Nd_arr, x_arr = _rk4(
            phi_arr, Nd_arr, x_arr, dt, z_arr, Pi,
            q, rho_m, wr, T, Tc, co, xi, h2o_wt,
            phicrit, cg, tau_relax,
            phi_bc, Nd_bc, x_bc)

        t    += dt
        step += 1

        if step % save_every == 0 or t >= T_max - 1e-10:
            P_arr, um_arr, ug_arr, _ = _pressure_march(
                phi_arr, Nd_arr, x_arr, z_arr, Pi,
                q, rho_m, wr, T, Tc, co, xi, h2o_wt,
                phicrit, cg, tau_relax)
            n_frag     = int(np.sum(phi_arr >= phicrit))
            frag_idx   = np.where(phi_arr >= phicrit)[0]
            # Profundidad del frente de fragmentación: nodo fragmentado más profundo
            z_frag_m   = z_arr[frag_idx[0]]  if n_frag > 0 else float('nan')
            z_frag_str = f"{z_frag_m/1e3:.2f} km" if n_frag > 0 else "  ---  "
            t_save.append(t)
            phi_hist.append(phi_arr.copy())
            P_hist.append(P_arr.copy())
            um_hist.append(um_arr.copy())
            ug_hist.append(ug_arr.copy())
            Nd_hist.append(Nd_arr.copy())
            x_hist.append(x_arr.copy())
            q_hist.append(q)
            print(f"  t = {t:8.1f} s   φ_max = {phi_arr.max():.4f}   "
                  f"z_frag = {z_frag_str}   "
                  f"P_vent = {P_arr[-1]/1e5:.2f} bar   "
                  f"um_vent = {um_arr[-1]:.1f} m/s   "
                  f"q = {q:.0f} kg/m²/s   "
                  f"nfrag = {n_frag}   Δt = {dt:.2f} s")

    frag_final = np.where(phi_arr >= phicrit)[0]
    z_frag_final = z_arr[frag_final[0]]/1e3 if len(frag_final) > 0 else float('nan')
    print("═"*62)
    print("  Integración transiente completada.")
    print(f"  Nodos fragmentados al final: {len(frag_final)}/{N_z}")
    if len(frag_final) > 0:
        print(f"  Profundidad de fragmentación: {z_frag_final:.2f} km")
    print(f"  q final = {q:.1f} kg/m²/s")
    print("═"*62)

    return dict(
        t_save   = np.array(t_save),
        z        = z_arr,
        phi_hist = np.array(phi_hist),
        P_hist   = np.array(P_hist),
        um_hist  = np.array(um_hist),
        ug_hist  = np.array(ug_hist),
        Nd_hist  = np.array(Nd_hist),
        x_hist   = np.array(x_hist),
        q_hist   = np.array(q_hist),
        q        = q,
        vinicial = vinicial,
    )


# ══════════════════════════════════════════════════════════════════════════════
# DEMO (ejecutar como script principal)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # ── Opción A: rama efusiva (vinicial = 10 m/s, default) ──────────────────
    # out = RIconduit1D_transient_f(
    #     radius=radius1, Pressure=overP1, wt=h2o1,
    #     Temperature=T1, content_crystal=xi1,
    #     N_z=150, T_max=1800.0, save_every=100, CFL=0.4,
    # )

    # ── Opción B: rama explosiva — usar vinicial del modelo estacionario ──────
    # El sistema es BISTABLE. Para obtener fragmentación, usar el q de la
    # solución convergida de RIconduitex5_5_f (rama explosiva).
    #
    #   from RIconduitex5_5 import RIconduitex5_5_f
    #   _, _, _, vinicial_conv, *_ = RIconduitex5_5_f(radius1, overP1, h2o1, T1, xi1)
    #   out = RIconduit1D_transient_f(..., vinicial_override=vinicial_conv)
    #
    # ── Opción C: exploración rápida — probar vinicial más alto ──────────────
    out = RIconduit1D_transient_f(
        radius            = radius1,
        Pressure          = overP1,
        wt                = h2o1,
        Temperature       = T1,
        content_crystal   = xi1,
        N_z               = 150,
        T_max             = 1800.0,
        save_every        = 100,
        CFL               = 0.4,
        tau_relax         = 50.0,
        vinicial_override = None,   # ← cambiar a valor convergido para rama explosiva
    )
    print(f"\nSnapshots guardados : {len(out['t_save'])}")
    print(f"q = {out['q']:.1f} kg/m²/s   vinicial = {out['vinicial']:.2f} m/s")
    print(f"φ final (superficie): {out['phi_hist'][-1, -1]:.4f}")
    print(f"um final (superficie): {out['um_hist'][-1, -1]:.2f} m/s")
    print(f"ug final (superficie): {out['ug_hist'][-1, -1]:.2f} m/s")
    n_frag_final = int(np.sum(out['phi_hist'][-1] >= 0.75))
    print(f"Nodos fragmentados al final: {n_frag_final}/{out['z'].size}")
