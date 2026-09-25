# Laboratorio: mismo DAE y el mismo tiro que RIconduitex5_5.py,
# con TODOS los umbrales suavizados (umbrales_reg.py).
# NO reemplaza al original.
#
#1D conduit model to calculate eruptive parameters based in equations of Slezin (2003) and Kozono and Koyaguchi (2009)
#numerical procedure: differential system of equations solved numerically with Python solver DBF
#shooting method using bisection technique with border conditions cited
#by Yoshida and Koyaguchi (1999) and Kozono and Koyaguchi (2009).
#Viscosity model: Melt: Giordano et al. (2008). 
#Crystal content: Einstein-Roscoe
#Melt density is taken as constant and calculated at the beginning with chemical composition, water content, temperature and pressure

from density import *
from viscosity import *
import numpy as np
import math
from calbuco2015d import *
from fvrel import *
from umbrales_reg import (
    EPS_FRAG, EPS_HENRY, EPS_PHI, EPS_RE, EPS_RB, EPS_XI,
    s_frag, s_henry, s_re, guarda_coalescencia, mezclar, tasa_xi,
    pesos_regimen,
)
from scikits.odes import dae
import warnings

eps_frag = EPS_FRAG
eps_phi = EPS_PHI
eps_henry = EPS_HENRY
eps_re = EPS_RE
eps_xi = EPS_XI
eps_rb = EPS_RB


def _mezclar_fragmentacion(phi, rho_g, um, ug, Fmw_v, Fgw_v, Fmg_v):
    """Mezcla fuerzas viscosas y fragmentadas con s_eps(phi)."""
    sf = float(s_frag(phi, phicrit, eps_frag))
    ra = 1e-3
    cd = 0.8
    Fmw_f = 0.0
    Fgw_f = 0.01 * rho_g * np.abs(ug) * ug / (4.0 * wr)
    Fmg_f = 3 * cd * rho_g * np.abs(ug - um) * (ug - um) * phi * (1.0 - phi) / (8.0 * ra)
    Fmw = mezclar(sf, Fmw_v, Fmw_f)
    Fgw = mezclar(sf, Fgw_v, Fgw_f)
    Fmg = mezclar(sf, Fmg_v, Fmg_f)
    return Fmw, Fgw, Fmg, sf
warnings.simplefilter("ignore", category=RuntimeWarning)
# Suprimir advertencias y errores
warnings.filterwarnings("ignore", category=UserWarning)  # Ignora UserWarnings generados por el solver

# Optional 2D radial coupling: override wall friction from FD profile (RIconduitex5_9).
suppress_Fmw = False
radial_Fmw_override = None


def _fin(x, fill=0.0):
    """NaN/inf → número finito, para que IDA no muera en silencio."""
    x = np.asarray(x, dtype=float)
    out = np.nan_to_num(x, nan=fill, posinf=1e30, neginf=-1e30)
    return float(out) if out.ndim == 0 else out


def _vel_algebraicas(P, phi, xcrys, qmass, rhom):
    """um, ug del DAE con la misma rampa de Henry (no recalcula phi)."""
    test = (1.0 - xi) * co - (1.0 - xcrys) * C1 * P ** beta
    fg_sat = (co * (1.0 - xi) - C1 * (1.0 - xcrys) * P ** beta) / ((1.0 - C1 * P ** beta))
    fg = float(s_henry(test, eps_henry)) * fg_sat
    if phi <= 1e-16 or fg <= 1e-16:
        um = qmass / rhom
        return um, um
    rho_g = P / (R * T)
    um = qmass * (1.0 - fg) / ((1.0 - phi) * rhom)
    ug = qmass * fg / (phi * rho_g)
    return um, ug


def _como_filas(y, ncomp):
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return None
    if y.ndim == 1:
        return y.reshape(1, -1) if y.size == ncomp else None
    return y if y.shape[1] == ncomp else None


def _pegar(sol, y):
    y2 = _como_filas(y, sol.shape[1])
    return sol if y2 is None else np.vstack((sol, y2))


def _pegar_t(zsol, t):
    t = np.asarray(t, dtype=float).ravel()
    return zsol if t.size == 0 else np.append(zsol, t)


def _interpolar_cruce(y_lo, t_lo, y_hi, t_hi, cond, idx=1):
    """Punto sobre phi=cond entre dos pasos (evita el salto 0→1 en una celda)."""
    y_lo = np.asarray(y_lo, dtype=float).ravel()
    y_hi = np.asarray(y_hi, dtype=float).ravel()
    a, b = float(y_lo[idx]), float(y_hi[idx])
    if not np.isfinite(a) or not np.isfinite(b) or b == a:
        return y_hi, float(t_hi)
    w = float(np.clip((cond - a) / (b - a), 0.0, 1.0))
    return (1.0 - w) * y_lo + w * y_hi, (1.0 - w) * float(t_lo) + w * float(t_hi)


def _cortar_cruce(y_all, t_all, y0, t0, cond):
    """Recorta solve() al primer cruce de phi=cond.

    Se deja el primer punto del DAE que ya cruzó (como el original).
    Interpolar al umbral exacto parte n_eq siguiente con un y0 que
    no cumple el residual y el tiro no cierra.
    """
    y_all = np.atleast_2d(np.asarray(y_all, dtype=float))
    t_all = np.asarray(t_all, dtype=float).ravel()
    if y_all.size == 0 or t_all.size == 0:
        return np.atleast_2d(y0), np.atleast_1d(float(t0))
    phi0 = float(y0[1])
    for i in range(y_all.shape[0]):
        phi = float(y_all[i, 1])
        if (phi0 < cond and phi >= cond) or (phi0 > cond and phi <= cond):
            return y_all[: i + 1], t_all[: i + 1]
        phi0 = phi
    return y_all, t_all


def _P_sH(sH_min=0.95, xcrys=None):
    """Presión (más baja que Pcrit) donde s_henry >= sH_min.

    No partir el DAE en test=0: ahí sH=1/2 y φ es minúscula.
    """
    if xcrys is None:
        xcrys = xi
    a = min(0.999999, max(-0.999999, 2.0 * sH_min - 1.0))
    test_need = np.arctanh(a) * max(float(eps_henry), 1e-12)
    rhs = (1.0 - xi) * co - test_need
    den = (1.0 - xcrys) * C1
    if rhs <= 0.0 or den <= 0.0:
        return None
    return float((rhs / den) ** (1.0 / beta))


def _clamp_y(y):
    y = np.asarray(y, dtype=float).copy()
    y[0] = float(np.clip(y[0], 1.0e4, 2.0e9))
    y[1] = float(np.clip(y[1], 0.0, 0.99))
    y[2] = float(np.clip(y[2], 1.0, 1.0e20))
    y[3] = float(np.clip(y[3], 0.0, 0.95))
    return y


def _sanear_perfil(zsol, sol, zmin, zmax):
    zsol = np.asarray(zsol, dtype=float).ravel()
    sol = np.asarray(sol, dtype=float)
    n = min(zsol.size, sol.shape[0])
    zsol, sol = zsol[:n], sol[:n]
    ok = np.isfinite(zsol) & np.all(np.isfinite(sol), axis=1)
    ok &= (zsol >= zmin) & (zsol <= zmax)
    zsol, sol = zsol[ok], sol[ok]
    if zsol.size < 2:
        return zsol, sol
    keep = np.ones(zsol.size, dtype=bool)
    last = zsol[0]
    for i in range(1, zsol.size):
        if zsol[i] <= last + 1e-12:
            keep[i] = False
        else:
            last = zsol[i]
    return zsol[keep], sol[keep]


def _filtrar_tramo(y, t, t0, tf):
    y = np.atleast_2d(np.asarray(y, dtype=float))
    t = np.asarray(t, dtype=float).ravel()
    n = min(y.shape[0], t.size)
    y, t = y[:n], t[:n]
    lo, hi = min(float(t0), float(tf)) - 1.0, max(float(t0), float(tf)) + 1.0
    ok = np.isfinite(t) & np.all(np.isfinite(y), axis=1) & (t >= lo) & (t <= hi)
    return y[ok], t[ok]


def _fg_phi_vel(P, xcrys, qmass, rhom):
    """fg, phi, um, ug con la misma rampa de Henry que momenteq.

    El tiro original arma y0 con el if duro. Acá el primer punto está
    justo en test≈0 (Pcrit), sH≈1/2, y IDA no inicializa si um/ug
    no coinciden con fg = sH * fg_sat.
    """
    test = (1.0 - xi) * co - (1.0 - xcrys) * C1 * P ** beta
    fg_sat = (co * (1.0 - xi) - C1 * (1.0 - xcrys) * P ** beta) / ((1.0 - C1 * P ** beta))
    sH = float(s_henry(test, eps_henry))
    fg = sH * fg_sat
    if fg <= 1e-16:
        um = qmass / rhom
        return 0.0, 0.0, um, um
    phi = 1.0 / (1.0 + (P / (fg * R * T)) * (1.0 - fg) / rhom)
    rho_g = P / (R * T)
    um = qmass * (1.0 - fg) / ((1.0 - phi) * rhom)
    ug = qmass * fg / (phi * rho_g)
    return fg, phi, um, ug


def _ida_inicializo(solver, ret=None):
    if hasattr(solver, "initialized"):
        return bool(solver.initialized)
    if ret is not None and hasattr(ret, "flag"):
        try:
            return int(ret.flag) == 0
        except (TypeError, ValueError):
            return "SUCCESS" in str(ret.flag).upper()
    return True


def _melt_wall_friction(visc, um):
    if suppress_Fmw:
        return float(radial_Fmw_override) if radial_Fmw_override is not None else 0.0
    if radial_Fmw_override is not None:
        return float(radial_Fmw_override)
    return cg * visc * um / (wr ** 2)


def _rama_o_cero(w, valor):
    """Si el peso es ~0 no evaluamos un F divergente (1/rb cuando φ→0)."""
    if abs(float(w)) < 1e-8:
        return 0.0
    return _fin(valor)


def _fuerzas_unificadas(phi, rb, rho_g, visc, um, ug):
    """Fmw, Fgw, Fmg = Σ w_i F^{(i)}. Mismo estado 6D en todo el conducto."""
    w1, w2, w3, w4 = pesos_regimen(
        phi, limphi1, limphi2, phicrit, eps_phi, eps_frag)
    rb = max(abs(_fin(rb, 1e-8)), 1e-8)
    phi_c = float(np.clip(phi, 0.0, 0.999))
    slip = ug - um
    Fmw1 = _melt_wall_friction(visc, um)
    Fgw1 = 0.0
    Fmg1 = 3.0 * visc * slip * phi_c * (1.0 - phi_c) / (rb ** 2)

    span = max(limphi2 - limphi1, 1e-8)
    tt = float(np.clip((phi_c - limphi1) / span, 0.0, 1.0))
    Re = 2.0 * rb * rho_g * slip / 1e-5
    sRe = float(s_re(Re, eps=eps_re))
    kper = 0.131 * (rb ** 2) * ((max(phi_c - limphi1, 0.0) + 0.05) ** 2.1)
    kper = max(kper, 1e-30)
    Fmg2_in = ((0.33 / (4.0 * rb)) * rho_g * (abs(slip) ** tt)) * (
        (3.0 * visc / (rb ** 2)) ** (1.0 - tt)) * slip * phi_c * (1.0 - phi_c)
    Fmg2_st = ((1e-5 / kper) ** tt) * (
        (3.0 * visc / (rb ** 2)) ** (1.0 - tt)) * slip * phi_c * (1.0 - phi_c)
    Fmg2 = mezclar(sRe, _fin(Fmg2_st), _fin(Fmg2_in))

    Fmg3_in = (0.33 / (4.0 * rb)) * rho_g * slip * slip * phi_c * (1.0 - phi_c)
    Fmg3_st = (1e-5 / kper) * slip * phi_c * (1.0 - phi_c)
    Fmg3 = mezclar(sRe, _fin(Fmg3_st), _fin(Fmg3_in))

    ra, cd = 1e-3, 0.8
    Fmw4 = 0.0
    Fgw4 = 0.01 * rho_g * np.abs(ug) * ug / (4.0 * wr)
    Fmg4 = 3.0 * cd * rho_g * np.abs(slip) * slip * phi_c * (1.0 - phi_c) / (8.0 * ra)

    Fmw = (w1 + w2 + w3) * Fmw1 + w4 * Fmw4
    Fgw = (w1 + w2 + w3) * Fgw1 + _rama_o_cero(w4, Fgw4)
    Fmg = (
        _rama_o_cero(w1, Fmg1) + _rama_o_cero(w2, Fmg2)
        + _rama_o_cero(w3, Fmg3) + _rama_o_cero(w4, Fmg4)
    )
    return _fin(Fmw), _fin(Fgw), _fin(Fmg), (w1, w2, w3, w4)


def momenteq(t, y, yprime, result):

    global fragcrit
    rho_g=y[0]/(R*T) #Densidad del gas

    if n_eq in [1, 2, 3]:
        test=(1-xi)*co - (1-y[3])*C1*y[0]**beta #Agua exsuelta con una presión y[0]
        fg_sat=(co*(1-xi) - C1*(1-(y[3]))*y[0]**beta)/((1-C1*y[0]**beta))
        sH=float(s_henry(test, eps_henry))
        fg=sH*fg_sat
        
        ## calculation of volume fraction of bubbles
        rb=((y[1]/((4/3)*np.pi*y[2]*(1-y[1]))))**(1/3)
        
        viscl_sat=fvrel(model,y[3],xi,ar1,ar2,xmax,(y[4]/wr))*viscosity(sio2,tio2,al2o3,feo,mno,mgo,cao,na2o,k2o,p2o5,(C1*y[0]**beta)*100,f2o,Tc)
        viscl_un=fvrel(model,y[3],xi,ar1,ar2,xmax,(y[4]/wr))*viscosity(sio2,tio2,al2o3,feo,mno,mgo,cao,na2o,k2o,p2o5,h2o,f2o,Tc)
        viscl=mezclar(sH, viscl_un, viscl_sat)
        
        nca=rb*viscl*(y[4]/wr)/3
        phicritbub=phicrit+0.05
        AA=(1-(y[1]/phicritbub))**(-phicritbub)
        BB=(1-(y[1]/phicritbub))**(5*phicritbub/3)
        
        c1=-0.2895*y[1] + 0.8132
        c2=y[1]
    
        viscrel=0.5*(AA-BB)*(1-math.erf(np.real(c1*np.log(nca)+c2)))+BB
        visc=viscrel*viscl

        f2, xteo, f3, dxdp = tasa_xi(
            (co*(1-xi)-((1-y[3])*C1*y[0]**beta))/(co*(1-xi) - (1-xmax)*C1*(Patm)**beta),
            y[3], xi, xmax, tcar*y[4], eps_xi)
        dfgdp=(-(-dxdp*C1*y[0]**beta + (1-y[3])*C1*beta*y[0]**(beta-1))+(co*(1-xi)-(1-y[3])*C1*y[0]**beta)*C1*beta*y[0]**(beta-1))/((1-C1*y[0]**beta)**2)
        
        
        Fgw=0 #Fuerza int. gas-wall
        Fmw=_melt_wall_friction(visc, y[4])#Fuerza int. melt-wall
        if n_eq==1:
            Fmg=3*visc*(y[5]-y[4])*y[1]*(1-y[1])/(rb**2)

        if n_eq==2:
            tt=float(np.clip((y[1]-limphi1)/(limphi2-limphi1), 0.0, 1.0))
            Re=2*rb*rho_g*(y[5]-y[4])/(1e-5)
            kper=0.131*(rb**2)*((y[1]-limphi1 + 0.05)**2.1)
            slip=np.abs(y[5]-y[4])
            Fmg_in=((0.33/(4*rb))*rho_g*(slip**tt))*(((3)*visc*(1/(rb**2)))**(1-tt))*(y[5]-y[4])*y[1]*(1-y[1])
            Fmg_st=((1e-5/kper)**tt)*(((3)*visc*(1/(rb**2)))**(1-tt))*(y[5]-y[4])*y[1]*(1-y[1])
            Fmg=mezclar(float(s_re(Re, eps=eps_re)), _fin(Fmg_st), _fin(Fmg_in))

        if n_eq==3:
            Re=2*rb*rho_g*(y[5]-y[4])/(1e-5)
            kper=0.131*(rb**2)*((y[1]-limphi1 + 0.05)**2.1)
            Fmg_in=(0.33/(4*rb))*rho_g*(y[5]-y[4])*(y[5]-y[4])*y[1]*(1-y[1])
            Fmg_st=(1e-5/kper)*(y[5]- y[4])*y[1]*(1-y[1])
            Fmg=mezclar(float(s_re(Re, eps=eps_re)), _fin(Fmg_st), _fin(Fmg_in))

        # n_eq 1–3: mismas fuerzas que el original. Mezclar fragmentación
        # acá mueve φ_crit y el tiro no encuentra la ventana.
        sfrag = 0.0
        
        aco=rho_g*(y[5]**2)
        bco=y[1] - (y[5]**2)*y[1]/(R*T) + dfgdp*q*y[5]
        cco=rho_m*(y[4]**2)
        dco=dfgdp*q*y[4] - (1-y[1])
        eco=-rho_m*(1-y[1])*g + Fmg - Fmw
        fco=rho_g*y[1]*g + Fmg + Fgw
        
        dphidz= (-eco*bco + dco*fco)/(aco*dco - bco*cco)
        dpdz= (-eco*aco + fco*cco)/(aco*dco - bco*cco)
        dvdz=vinicial*(-dfgdp*dpdz*(1-y[1]) + (1-fg)*dphidz)/((1-y[1])**2)
        fragcrit=dvdz*visc/(0.01*1e10)

        dNdt_on=-(y[2]**(2/3))*((1/(1-y[1]))**(1/3))*((1/9)*(rho_m-rho_g)*9.81/visc)*((3*y[1]/(4*np.pi))**(2/3))*(F1*F1 - F2*F2)*(1/(1-((F1+F2)*((3*y[1]*(np.pi/(6*phicrit))/(4*np.pi))**(1/3)))))*Fc*(1-y[1]/phicrit)*(wr-rb)/wr
        phi_off = 0.5 if n_eq == 3 else phicrit
        dNdt=float(guarda_coalescencia(rb, wr, y[1], phi_off, eps_frac=eps_rb, eps_frag=eps_frag))*dNdt_on

        if n_eq in [2,3]:
            f2, xteo, f3, _ = tasa_xi(
                (co-(C1*y[0]**beta))/(co - C1*(Patm)**beta),
                y[3], xi, xmax, tcar, eps_xi)

         ########### En momenteq aca va multp. pr y[4] en matlab (Preguntar)

        if n_eq==3:
            dxdz = (xmax - xi) * f2 * f3 / max(tcar * y[4], 1e-30)
        else:
            dxdz = (xmax - xi) * f2 * f3 / max(tcar, 1e-30)

        dNdt = (1.0 - sfrag) * dNdt
        dxdz = (1.0 - sfrag) * dxdz

        result[0] = -yprime[0] + _fin(dpdz)
        result[1] = -yprime[1] + _fin(dphidz)
        result[2] = -yprime[2] + _fin(dNdt/y[4])
        result[3] = -yprime[3] + _fin(dxdz)
        result[4] =  y[4] - q*(1-fg)/((1-y[1])*rho_m)
        result[5] = y[5] - q*fg/(y[1]*rho_g)

    else:
        test=(1-xi)*co - (1-xfinal)*C1*y[0]**beta
        fg_sat=(co*(1-xi) - C1*(1-xfinal)*y[0]**beta)/((1-C1*y[0]**beta))
        sH=float(s_henry(test, eps_henry))
        fg=sH*fg_sat

        ra=1e-3 
        cd=0.8
        f2, xteo, f3, dxdp = tasa_xi(
            (co*(1-xi)-((1-xfinal)*C1*y[0]**beta))/(co*(1-xi) - (1-xmax)*C1*(Patm)**beta),
            xfinal, xi, xmax, tcar, eps_xi)
        dfgdp_sat=(-(-dxdp*C1*y[0]**beta + (1-xfinal)*C1*beta*y[0]**(beta-1))+(co*(1-xi)-(1-xfinal)*C1*y[0]**beta)*C1*beta*y[0]**(beta-1))/((1-C1*y[0]**beta)**2)
        dfgdp=sH*dfgdp_sat

        um=(1-fg)*q/(rho_m*(1-y[1]))
        ug=fg*q/(rho_g*y[1])

        #calculation of density and viscosity
        rb=((y[1]/((4/3)*np.pi*Nd*(1-y[1]))))**(1/3)

        Fmg_v = (0.33/(4*rb))*rho_g*np.abs(ug-um)*(ug-um)*y[1]*(1-y[1])
        Fmw, Fgw, Fmg, sfrag = _mezclar_fragmentacion(
            y[1], rho_g, um, ug, 0.0, 0.0, Fmg_v)
        
        aco=rho_g*(ug**2)
        bco=y[1] - (ug**2)*y[1]/(R*T) + dfgdp*q*ug
        cco=rho_m*(um**2)
        dco=dfgdp*q*um - (1-y[1])
        eco=-rho_m*(1-y[1])*9.8 + Fmg - Fmw
        fco=rho_g*y[1]*9.8 + Fmg + Fgw
        
        dphidz= (-eco*bco + dco*fco)/(aco*dco - bco*cco)
        dpdz= (-eco*aco + fco*cco)/(aco*dco - bco*cco)

        result[0] = -yprime[0] + dpdz 
        result[1] = -yprime[1] + dphidz

def momenteq1(t, y):
    rho_g=y[0]/(R*T) #Densidad del gas

    if n_eq in [1, 2, 3]:
        test=(1-xi)*co - (1-y[3])*C1*y[0]**beta #Agua exsuelta con una presión y[0]
        fg_sat=(co*(1-xi) - C1*(1-(y[3]))*y[0]**beta)/((1-C1*y[0]**beta))
        sH=float(s_henry(test, eps_henry))
        fg=sH*fg_sat
        
        ## calculation of volume fraction of bubbles
        rb=((y[1]/((4/3)*np.pi*y[2]*(1-y[1]))))**(1/3)
        
        viscl_sat=fvrel(model,y[3],xi,ar1,ar2,xmax,(y[4]/wr))*viscosity(sio2,tio2,al2o3,feo,mno,mgo,cao,na2o,k2o,p2o5,(C1*y[0]**beta)*100,f2o,Tc)
        viscl_un=fvrel(model,y[3],xi,ar1,ar2,xmax,(y[4]/wr))*viscosity(sio2,tio2,al2o3,feo,mno,mgo,cao,na2o,k2o,p2o5,h2o,f2o,Tc)
        viscl=mezclar(sH, viscl_un, viscl_sat)
        
        nca=rb*viscl*(y[4]/wr)/3
        phicritbub=phicrit+0.05
        AA=(1-(y[1]/phicritbub))**(-phicritbub)
        BB=(1-(y[1]/phicritbub))**(5*phicritbub/3)
        
        c1=-0.2895*y[1] + 0.8132
        c2=y[1]
    
        viscrel=0.5*(AA-BB)*(1-math.erf(np.real(c1*np.log(nca)+c2)))+BB
        visc=viscrel*viscl

        f2, xteo, f3, dxdp = tasa_xi(
            (co*(1-xi)-((1-y[3])*C1*y[0]**beta))/(co*(1-xi) - (1-xmax)*C1*(Patm)**beta),
            y[3], xi, xmax, tcar*y[4], eps_xi)
        dfgdp=(-(-dxdp*C1*y[0]**beta + (1-y[3])*C1*beta*y[0]**(beta-1))+(co*(1-xi)-(1-y[3])*C1*y[0]**beta)*C1*beta*y[0]**(beta-1))/((1-C1*y[0]**beta)**2)
        
        
        Fgw=0 #Fuerza int. gas-wall
        Fmw=_melt_wall_friction(visc, y[4])#Fuerza int. melt-wall
        if n_eq==1:
            Fmg=3*visc*(y[5]-y[4])*y[1]*(1-y[1])/(rb**2)

        if n_eq==2:
            tt=float(np.clip((y[1]-limphi1)/(limphi2-limphi1), 0.0, 1.0))
            Re=2*rb*rho_g*(y[5]-y[4])/(1e-5)
            kper=0.131*(rb**2)*((y[1]-limphi1 + 0.05)**2.1)
            slip=np.abs(y[5]-y[4])
            Fmg_in=((0.33/(4*rb))*rho_g*(slip**tt))*(((3)*visc*(1/(rb**2)))**(1-tt))*(y[5]-y[4])*y[1]*(1-y[1])
            Fmg_st=((1e-5/kper)**tt)*(((3)*visc*(1/(rb**2)))**(1-tt))*(y[5]-y[4])*y[1]*(1-y[1])
            Fmg=mezclar(float(s_re(Re, eps=eps_re)), _fin(Fmg_st), _fin(Fmg_in))

        if n_eq==3:
            Re=2*rb*rho_g*(y[5]-y[4])/(1e-5)
            kper=0.131*(rb**2)*((y[1]-limphi1 + 0.05)**2.1)
            Fmg_in=(0.33/(4*rb))*rho_g*(y[5]-y[4])*(y[5]-y[4])*y[1]*(1-y[1])
            Fmg_st=(1e-5/kper)*(y[5]- y[4])*y[1]*(1-y[1])
            Fmg=mezclar(float(s_re(Re, eps=eps_re)), _fin(Fmg_st), _fin(Fmg_in))

        # n_eq 1–3: mismas fuerzas que el original. Mezclar fragmentación
        # acá mueve φ_crit y el tiro no encuentra la ventana.
        sfrag = 0.0
        
        aco=rho_g*(y[5]**2)
        bco=y[1] - (y[5]**2)*y[1]/(R*T) + dfgdp*q*y[5]
        cco=rho_m*(y[4]**2)
        dco=dfgdp*q*y[4] - (1-y[1])
        eco=-rho_m*(1-y[1])*g + Fmg - Fmw
        fco=rho_g*y[1]*g + Fmg + Fgw
        
        dphidz= (-eco*bco + dco*fco)/(aco*dco - bco*cco)
        dpdz= (-eco*aco + fco*cco)/(aco*dco - bco*cco)
        dvdz=vinicial*(-dfgdp*dpdz*(1-y[1]) + (1-fg)*dphidz)/((1-y[1])**2)

        dNdt_on=-(y[2]**(2/3))*((1/(1-y[1]))**(1/3))*((1/9)*(rho_m-rho_g)*9.81/visc)*((3*y[1]/(4*np.pi))**(2/3))*(F1*F1 - F2*F2)*(1/(1-((F1+F2)*((3*y[1]*(np.pi/(6*phicrit))/(4*np.pi))**(1/3)))))*Fc*(1-y[1]/phicrit)*(wr-rb)/wr
        phi_off = 0.5 if n_eq == 3 else phicrit
        dNdt=float(guarda_coalescencia(rb, wr, y[1], phi_off, eps_frac=eps_rb, eps_frag=eps_frag))*dNdt_on

        if n_eq in [2,3]:
            f2, xteo, f3, _ = tasa_xi(
                (co-(C1*y[0]**beta))/(co - C1*(Patm)**beta),
                y[3], xi, xmax, tcar, eps_xi)

        if n_eq==3:
            dxdz = (xmax - xi) * f2 * f3 / max(tcar * y[4], 1e-30)
        else:
            dxdz = (xmax - xi) * f2 * f3 / max(tcar, 1e-30)

        dNdt = (1.0 - sfrag) * dNdt
        dxdz = (1.0 - sfrag) * dxdz
        
        return np.array([dpdz ,dphidz,dNdt/y[4],dxdz, y[4] - q*(1-fg)/((1-y[1])*rho_m), y[5] - q*fg/(y[1]*rho_g)])

    else:
        test=(1-xi)*co - (1-xfinal)*C1*y[0]**beta
        fg_sat=(co*(1-xi) - C1*(1-xfinal)*y[0]**beta)/((1-C1*y[0]**beta))
        sH=float(s_henry(test, eps_henry))
        fg=sH*fg_sat

        ra=1e-3 
        cd=0.8
        f2, xteo, f3, dxdp = tasa_xi(
            (co*(1-xi)-((1-xfinal)*C1*y[0]**beta))/(co*(1-xi) - (1-xmax)*C1*(Patm)**beta),
            xfinal, xi, xmax, tcar, eps_xi)
        dfgdp_sat=(-(-dxdp*C1*y[0]**beta + (1-xfinal)*C1*beta*y[0]**(beta-1))+(co*(1-xi)-(1-xfinal)*C1*y[0]**beta)*C1*beta*y[0]**(beta-1))/((1-C1*y[0]**beta)**2)
        dfgdp=sH*dfgdp_sat

        um=(1-fg)*q/(rho_m*(1-y[1]))
        ug=fg*q/(rho_g*y[1])

        #calculation of density and viscosity
        rb=((y[1]/((4/3)*np.pi*Nd*(1-y[1]))))**(1/3)

        Fmg_v = (0.33/(4*rb))*rho_g*np.abs(ug-um)*(ug-um)*y[1]*(1-y[1])
        Fmw, Fgw, Fmg, sfrag = _mezclar_fragmentacion(
            y[1], rho_g, um, ug, 0.0, 0.0, Fmg_v)
        
        aco=rho_g*(ug**2)
        bco=y[1] - (ug**2)*y[1]/(R*T) + dfgdp*q*ug
        cco=rho_m*(um**2)
        dco=dfgdp*q*um - (1-y[1])
        eco=-rho_m*(1-y[1])*9.8 + Fmg - Fmw
        fco=rho_g*y[1]*9.8 + Fmg + Fgw
        
        dphidz= (-eco*bco + dco*fco)/(aco*dco - bco*cco)
        dpdz= (-eco*aco + fco*cco)/(aco*dco - bco*cco)

        return np.array([dpdz,dphidz])  
        
def solv(t0, tf, y0, yp0, atol, rtol, n, param, cond=None):
    tspan = np.linspace(t0, tf, int(n))
    y0 = np.asarray(y0, dtype=float).copy()
    if param == 1 and y0.size >= 6:
        um, ug = _vel_algebraicas(y0[0], y0[1], y0[3], q, rho_m)
        y0[4], y0[5] = um, ug
        yp0 = np.asarray(momenteq1(t0, y0), dtype=float).copy()
        if yp0.size >= 6:
            yp0[4] = 0.0
            yp0[5] = 0.0
    elif param in [2, 3] and y0.size >= 6:
        yp0 = np.asarray(momenteq1(t0, y0), dtype=float).copy()
        if yp0.size >= 6:
            yp0[4] = 0.0
            yp0[5] = 0.0
    else:
        yp0 = np.asarray(yp0, dtype=float).copy()

    def _armar(compute_ic, first_step=1e-18):
        kw = dict(atol=atol, rtol=rtol, old_api=False)
        if first_step is not None:
            kw['first_step_size'] = first_step
        if compute_ic:
            kw['compute_initcond'] = 'yp0'
        if param in [1, 2, 3]:
            kw['algebraic_vars_idx'] = [4, 5]
        return dae('ida', momenteq, **kw)

    vacio = (np.atleast_2d(y0), np.atleast_1d(float(t0)))

    if param in [1,2,3]:
        solver = _armar(True)
        ret = solver.init_step(t0, y0, yp0)
        if not _ida_inicializo(solver, ret):
            solver = _armar(False, first_step=None)
            ret = solver.init_step(t0, y0, yp0)
        y_values = []
        t_values = []
        if not (hasattr(solver, "initialized") and not solver.initialized):
            y_anterior = float(y0[1])
            for time in tspan[1:]:
                solution = solver.step(time)
                if solution.values.y is None:
                    break
                y_now = np.asarray(solution.values.y, dtype=float).ravel()
                y_actual = float(y_now[1])
                y_values.append(y_now)
                t_values.append(solution.values.t)
                if cond is not None and (
                    (y_anterior < cond and y_actual >= cond)
                    or (y_anterior > cond and y_actual <= cond)
                ):
                    break
                y_anterior = y_actual

        if len(y_values) == 0:
            # step() no avanzó: solve() hacia z=0 y se corta en cond
            tspan2 = np.linspace(t0, tf, max(400, int(n) // 5))
            solver = _armar(True, first_step=None)
            solution = solver.solve(tspan2, y0, yp0)
            y_all, t_all = solution.values.y, solution.values.t
            if y_all is None or np.asarray(y_all).size == 0:
                return vacio
            if cond is None:
                return np.atleast_2d(y_all), np.asarray(t_all, dtype=float).ravel()
            return _cortar_cruce(y_all, t_all, y0, t0, cond)

        return np.vstack(y_values), np.asarray(t_values, dtype=float)
    else:
        solver = dae('ida', momenteq,
                compute_initcond='yp0',
                atol=atol,
                rtol=rtol,
                old_api=False)
        solution = solver.solve(tspan, y0, yp0)
        y_values, t_values = solution.values.y, solution.values.t
        if y_values is None or np.asarray(y_values).size == 0:
            return vacio
        return np.atleast_2d(y_values), np.asarray(t_values, dtype=float).ravel()


def _estado_unificado(y):
    """fg, visc, rb, F*, dfgdp, dxdz, dNdz — um,ug algebraicas (P,φ,fg,q)."""
    global fragcrit
    y = _clamp_y(y)
    P, phi, Nd_y, xcrys = float(y[0]), float(y[1]), float(y[2]), float(y[3])
    rho_g = max(P / (R * T), 1e-12)
    test = (1.0 - xi) * co - (1.0 - xcrys) * C1 * P ** beta
    den_h = max(1.0 - C1 * P ** beta, 1e-12)
    fg_sat = (co * (1.0 - xi) - C1 * (1.0 - xcrys) * P ** beta) / den_h
    sH = float(s_henry(test, eps_henry))
    fg = max(0.0, sH * fg_sat)
    if fg <= 1e-16:
        um = q / rho_m
        ug = um
    else:
        phi_eff = max(phi, 1e-10)
        um = q * (1.0 - fg) / ((1.0 - min(phi_eff, 0.999)) * rho_m)
        ug = q * fg / (phi_eff * rho_g)
    um = _fin(um, q / max(rho_m, 1.0))
    ug = _fin(ug, um)
    rb = ((max(phi, 0.0) / ((4.0 / 3.0) * np.pi * max(Nd_y, 1e-30) * max(1.0 - phi, 1e-12)))) ** (1.0 / 3.0)
    viscl_sat = fvrel(model, xcrys, xi, ar1, ar2, xmax, (um / wr)) * viscosity(
        sio2, tio2, al2o3, feo, mno, mgo, cao, na2o, k2o, p2o5, (C1 * P ** beta) * 100, f2o, Tc)
    viscl_un = fvrel(model, xcrys, xi, ar1, ar2, xmax, (um / wr)) * viscosity(
        sio2, tio2, al2o3, feo, mno, mgo, cao, na2o, k2o, p2o5, h2o, f2o, Tc)
    viscl = mezclar(sH, viscl_un, viscl_sat)
    nca = rb * viscl * (um / wr) / 3.0
    phicritbub = phicrit + 0.05
    phi_v = min(max(phi, 0.0), 0.999 * phicritbub)
    AA = (1.0 - (phi_v / phicritbub)) ** (-phicritbub)
    BB = (1.0 - (phi_v / phicritbub)) ** (5.0 * phicritbub / 3.0)
    c1 = -0.2895 * phi + 0.8132
    c2 = phi
    viscrel = 0.5 * (AA - BB) * (1.0 - math.erf(np.real(c1 * np.log(max(nca, 1e-30)) + c2))) + BB
    visc = viscrel * viscl
    w1, w2, w3, w4 = pesos_regimen(phi, limphi1, limphi2, phicrit, eps_phi, eps_frag)
    if fg <= 1e-16:
        Fmw = _melt_wall_friction(viscl, um)
        dpdz = -rho_m * g - Fmw
        return {
            "fg": 0.0, "dpdz": _fin(dpdz), "dphidz": 0.0, "dNdz": 0.0,
            "dxdz": 0.0, "um_alg": um, "ug_alg": um, "w": (w1, w2, w3, w4),
        }
    Fmw, Fgw, Fmg, (w1, w2, w3, w4) = _fuerzas_unificadas(phi, rb, rho_g, visc, um, ug)

    f2, xteo, f3, dx1 = tasa_xi(
        (co * (1.0 - xi) - ((1.0 - xcrys) * C1 * P ** beta))
        / (co * (1.0 - xi) - (1.0 - xmax) * C1 * (Patm) ** beta),
        xcrys, xi, xmax, tcar, eps_xi)
    dx3 = (xmax - xi) * f2 * f3 / max(tcar * max(abs(um), 1e-6), 1e-30)
    dxdz = (1.0 - w4) * ((w1 + w2) * dx1 + w3 * dx3)
    dxdp = (1.0 - w4) * dx1 / max(abs(um), 1e-6)
    dfgdp = ((-(-dxdp * C1 * P ** beta + (1.0 - xcrys) * C1 * beta * P ** (beta - 1))
              + (co * (1.0 - xi) - (1.0 - xcrys) * C1 * P ** beta) * C1 * beta * P ** (beta - 1))
             / ((1.0 - C1 * P ** beta) ** 2))

    aco = rho_g * (ug ** 2)
    bco = phi - (ug ** 2) * phi / (R * T) + dfgdp * q * ug
    cco = rho_m * (um ** 2)
    dco = dfgdp * q * um - (1.0 - phi)
    eco = -rho_m * (1.0 - phi) * g + Fmg - Fmw
    fco = rho_g * phi * g + Fmg + Fgw
    den = aco * dco - bco * cco
    if abs(den) < 1e-18 or not np.isfinite(den):
        dphidz, dpdz = 0.0, -rho_m * (1.0 - phi) * g - Fmw
    else:
        dphidz = (-eco * bco + dco * fco) / den
        dpdz = (-eco * aco + fco * cco) / den
    dpdz = float(np.clip(_fin(dpdz), -1.0e9, 1.0e8))
    dphidz = float(np.clip(_fin(dphidz), -20.0, 20.0))
    dvdz = vinicial * (-dfgdp * dpdz * (1.0 - phi) + (1.0 - fg) * dphidz) / max((1.0 - phi) ** 2, 1e-30)
    fragcrit = dvdz * visc / (0.01 * 1e10)

    inner = (F1 + F2) * ((3.0 * max(phi, 0.0) * (np.pi / (6.0 * phicrit)) / (4.0 * np.pi)) ** (1.0 / 3.0))
    den_n = 1.0 - inner
    if abs(den_n) < 1e-8:
        dNdt_on = 0.0
    else:
        dNdt_on = -(Nd_y ** (2.0 / 3.0)) * ((1.0 / max(1.0 - phi, 1e-12)) ** (1.0 / 3.0)) * (
            (1.0 / 9.0) * (rho_m - rho_g) * 9.81 / max(visc, 1e-30)) * (
            (3.0 * max(phi, 0.0) / (4.0 * np.pi)) ** (2.0 / 3.0)) * (F1 * F1 - F2 * F2) * (
            1.0 / den_n) * Fc * (1.0 - phi / phicrit) * (wr - rb) / wr
    dNdt = float(guarda_coalescencia(rb, wr, phi, phicrit, eps_frac=eps_rb, eps_frag=eps_frag)) * _fin(dNdt_on)
    dNdz = (1.0 - w4) * dNdt / max(abs(um), 1e-6)
    return {
        "fg": fg, "dpdz": _fin(dpdz), "dphidz": _fin(dphidz), "dNdz": _fin(dNdz),
        "dxdz": _fin(dxdz), "um_alg": um, "ug_alg": ug, "w": (w1, w2, w3, w4),
    }


def momenteq_u(t, y, yprime, result):
    st = _estado_unificado(y)
    if st["fg"] <= 1e-16:
        um_a = q / rho_m
        result[0] = -yprime[0] + st["dpdz"]
        result[1] = -yprime[1] + st["dphidz"]
        result[2] = -yprime[2]
        result[3] = -yprime[3]
        result[4] = y[4] - um_a
        result[5] = y[5] - um_a
        return
    result[0] = -yprime[0] + st["dpdz"]
    result[1] = -yprime[1] + st["dphidz"]
    result[2] = -yprime[2] + st["dNdz"]
    result[3] = -yprime[3] + st["dxdz"]
    result[4] = y[4] - st["um_alg"]
    result[5] = y[5] - st["ug_alg"]


def momenteq1_u(t, y):
    st = _estado_unificado(y)
    if st["fg"] <= 1e-16:
        um_a = q / rho_m
        return np.array([st["dpdz"], st["dphidz"], 0.0, 0.0, y[4] - um_a, y[5] - um_a])
    return np.array([
        st["dpdz"], st["dphidz"], st["dNdz"], st["dxdz"],
        y[4] - st["um_alg"], y[5] - st["ug_alg"],
    ])


def _preparar_y0(t0, y0):
    y0 = np.asarray(y0, dtype=float).copy()
    um, ug = _vel_algebraicas(y0[0], y0[1], y0[3], q, rho_m)
    y0[4], y0[5] = um, ug
    yp0 = np.asarray(momenteq1_u(t0, y0), dtype=float).copy()
    yp0[4] = 0.0
    yp0[5] = 0.0
    return y0, yp0


def _ida_hasta(t0, tf, y0, atol, rtol, n, cond=None):
    y0, yp0 = _preparar_y0(t0, y0)
    tspan = np.linspace(t0, tf, max(int(n), 200))
    vacio = (np.atleast_2d(y0), np.atleast_1d(float(t0)))

    def _armar(compute_ic, first_step=1e-18):
        kw = dict(atol=atol, rtol=rtol, old_api=False, algebraic_vars_idx=[4, 5])
        if first_step is not None:
            kw["first_step_size"] = first_step
        if compute_ic:
            kw["compute_initcond"] = "yp0"
        return dae("ida", momenteq_u, **kw)

    solver = _armar(True)
    ret = solver.init_step(t0, y0, yp0)
    if not _ida_inicializo(solver, ret):
        solver = _armar(False, first_step=None)
        ret = solver.init_step(t0, y0, yp0)
    y_values, t_values = [], []
    if not (hasattr(solver, "initialized") and not solver.initialized):
        y_anterior = float(y0[1])
        for time in tspan[1:]:
            solution = solver.step(time)
            if solution.values.y is None:
                break
            y_now = np.asarray(solution.values.y, dtype=float).ravel()
            y_actual = float(y_now[1])
            y_values.append(y_now)
            t_values.append(solution.values.t)
            if cond is not None and (
                (y_anterior < cond and y_actual >= cond)
                or (y_anterior > cond and y_actual <= cond)
            ):
                break
            y_anterior = y_actual
    if len(y_values) == 0:
        solver = _armar(True, first_step=None)
        solution = solver.solve(tspan, y0, yp0)
        y_all, t_all = solution.values.y, solution.values.t
        if y_all is None or np.asarray(y_all).size == 0:
            return vacio
        if cond is None:
            return _filtrar_tramo(y_all, t_all, t0, tf)
        y_c, t_c = _cortar_cruce(y_all, t_all, y0, t0, cond)
        return _filtrar_tramo(y_c, t_c, t0, tf)
    return _filtrar_tramo(np.vstack(y_values), t_values, t0, tf)


def solv_u(t0, tf, y0, yp0, atol, rtol, n):
    """Un solo campo. Se reinicia en limphi1, limphi2, phicrit (como el original)."""
    y0, yp0 = _preparar_y0(t0, y0)
    vacio = (np.atleast_2d(y0), np.atleast_1d(float(t0)))
    y_acc, t_acc = [], []
    y = y0
    t = float(t0)
    for cond in (limphi1, limphi2, phicrit, None):
        if t >= float(tf) - 1e-9:
            break
        yi, ti = _ida_hasta(t, tf, y, atol, rtol, n, cond=cond)
        if yi is None or np.asarray(yi).size == 0:
            break
        yi, ti = np.atleast_2d(yi), np.asarray(ti, dtype=float).ravel()
        y_acc.append(yi)
        t_acc.append(ti)
        y = yi[-1]
        t = float(ti[-1])
        if t >= float(tf) - 1e-6:
            break
    if not y_acc:
        return vacio
    return np.vstack(y_acc), np.concatenate(t_acc)


def _aplicar_eps(ef, ep, eh, er, ex, erb):
    global eps_frag, eps_phi, eps_henry, eps_re, eps_xi, eps_rb
    eps_frag = float(ef)
    eps_phi = float(ep)
    eps_henry = float(eh)
    eps_re = float(er)
    eps_xi = float(ex)
    eps_rb = float(erb)


#Función principal del laboratorio: mismos kwargs de umbrales_reg / el notebook
def RIconduitex5_5_suave_f(radius,Pressure, wt, Temperature, content_crystal,
                           eps_frag=EPS_FRAG, eps_phi=EPS_PHI, eps_henry=EPS_HENRY,
                           eps_re=EPS_RE, eps_xi=EPS_XI, eps_rb=EPS_RB,
                           unificado=False):
    global Nd, rho_m, vinicial, q, count, results, vinicial, Pi, limphi1, limphi2, phicrit, xfinal, n_eq, rho_ti
    global overP, wr, T, Tc, co, xi, h2o, cg, xmax, phimax1, phimax2, cA
    _aplicar_eps(eps_frag, eps_phi, eps_henry, eps_re, eps_xi, eps_rb)

    wr = radius
    overP = Pressure
    T=Temperature
    Tc= Temperature-273.15
    co=wt/100
    xi=content_crystal
    h2o = wt

    # DO NOT CHANGE THE FOLLOWING LINES #
    if model=='em2s' or model=='vona11':
        phimax1=0.656*np.exp(-(np.log10(ar1)**2)/(2*(1.08**2))) 
        phimax2=0.656*np.exp(-(np.log10(ar2)**2)/(2*(1.08**2))) 

        if xi>phimax1:
            xi=0.95*phimax1 

        if (xmax-xi)/(1-xi)>phimax2:
            xmax=0.95*phimax2*(1-xi) + xi

    if geometry=='dyke':
        cg=3 
    else:
        cg=8 

    cA=(limperh-limperl)/(linf-lsup)

    Pi=rcrust*g*H*(-1) + overP #pressure at the inlet of conduit (Pa)
    exi=(1-xi)*(co-C1*Pi**beta)/(1-C1*Pi**beta)   #initial exsolved water
    dis=C1*Pi**beta   #initial dissolved water

    #Si tenemos que el agua inicial disuelta es menor que 0, entonces solo colocamos que el agua disuelta inicial es 0
    if dis>co:
        dis=co
    
    rho_m=density(sio2,tio2,al2o3,feo,mgo,cao,na2o,k2o,dis*100,Tc,Pi/1e6) #density of melt. Constant through the conduit

    #Aquí tenemos diferentes condiciones en caso de que el agua exsuelta inicial sea menor o igual a 0, o si tenemos que es mayor a 0
    if exi<=0:
        #Si es menor o igual a 0 (o sea que no hay agua exsuelta), entonces la densidad total inicial, es la densidad del líquido, fracción de gas es 0, y cont. de burbujas es 0 
        rho_ti=rho_m
        fgi=0
        phini=0 
    else:
        #Si es mayor a 0 (o sea que hay agua exsuelta en el líquido), entonces la fracción de gas sigue la ecuación mostrada en Kozono et al (2009), la densidad total inicial debe ser la densidad del líquido más la densidad de la fracción de gas exsuelta que hay
        fgi=(1-xi)*(co - C1*Pi**beta)/(1-C1*Pi**beta)
        rho_ti=Pi*rho_m/(rho_m*R*T*(fgi)+Pi*(1 - fgi))
        phini=1/(1 + (Pi/(fgi*R*T))*(1-fgi)/rho_m) 
    
    vinicial=10 #initial guess of velocity at the conduit inlet (m/s)
    
    vmin=0.1 #lower limit for velocity in the shooting method (m/s)
    vmax=50 #upper limit

    vmin1=vmin
    vmax1=vmax
    
    vsound=0.99*np.sqrt(R*T)
    count=1
    converged = False
    U_prev, P_prev, z_prev = None, None, None

    ## Here shooting method starts
    while count<60:
        print(f"\nCount = {count} \nInitial velocity = {vinicial} \n")

        xfinal = xi
        q=vinicial*rho_ti 
        velc=np.sqrt(15e9/rho_m) 
        visc=fvrel(model,xi,xi,ar1,ar2,xmax,(vinicial/wr))*viscosity(sio2,tio2,al2o3,feo,mno,mgo,cao,na2o,k2o,p2o5,dis*100,f2o,Tc);
        dpdzcalc = (-rho_ti*(9.81 + cg*visc*vinicial/((wr*wr)*rho_ti)))/(1-(vinicial**2)/(velc**2))
        
        dpdt=-dpdzcalc*vinicial 
        coefNd=np.log10(dpdt) 
        Nd=10**(1.5*coefNd + 5) 

        if C1*Pi**beta<co:
            Nd=1e8

        ############################################
        rhol=rho_m #Densidad del líquido
        phisel=0.2 #Contenido de gas en el líquido
        
        drhogdp=1/(R*T) 

        Pmax=(co/C1)**(1/beta)
        Pmin=pfinal
        Pcalc=(Pmax+Pmin)/2
        
        count2=1
        while count2<50:
            ncalc=(1-xi)*(co - C1*Pcalc**beta)/(1-C1*Pcalc**beta)
            rhogcalc=Pcalc/(R*T)
            phicalc=1/((1/ncalc-1)*rhogcalc/rhol + 1)
        
            if phicalc>(phisel - 0.002) and phicalc<(phisel + 0.002):            
                break               
            else:
                if phicalc<0.198:
                    Pmax=Pcalc
                    Pcalc=(Pmax+Pmin)/2
                else:
                    Pmin=Pcalc
                    Pcalc=(Pmax+Pmin)/2
            count2=count2+1
        
        dfgdpcalc=(C1*beta*Pcalc**(beta-1))*(co-1)/((1-C1*Pcalc**beta)**2)
        dphidpcalc=-(-dfgdpcalc*rhogcalc/(rhol*ncalc**2) + (1/ncalc - 1)*drhogdp/rhol)/(((1/ncalc - 1)*rhogcalc/rhol + 1)**2)
        
        velcc=np.sqrt(15e9/rhol)
        viscalc=fvrel(model,xi,xi,ar1,ar2,xmax,(vinicial/wr))*viscosity(sio2,tio2,al2o3,feo,mno,mgo,cao,na2o,k2o,p2o5,C1*(Pcalc**beta)*100,f2o,Tc)
        rho_ticalc=Pcalc*rhol/(rhol*R*T*(ncalc)+Pcalc*(1 - ncalc))
        
        dpdzcalc2 = (-rho_ticalc*(9.81 + cg*viscalc*vinicial/((wr**2)*rho_ticalc)))/(1-(vinicial**2)/(velcc**2))
        dvdz=vinicial*(-dfgdpcalc*dpdzcalc2*(1-phicalc) + (1-ncalc)*dphidpcalc*dpdzcalc2)/((1-phicalc)**2)
        dvdz2=vinicial/wr
        dvdz3=(dvdz + dvdz2)/2
        dvdz=dvdz3
        fragcrit=dvdz*visc/(0.01*1e10)
        
        rbcalc=((phicalc/((4/3)*np.pi*Nd*(1-phicalc))))**(1/3)
        
        Ca=np.abs(dvdz*viscalc*rbcalc/0.3)
        
        phicrit=((lsup - linf)/2)*math.erf(np.log10(Ca)) + (lsup + linf)/2
        limphi1=((limperl - limperh)/2)*math.erf(np.log10(Ca)) + (limperl + limperh)/2
        limphi2=limphi1 + 0.01
        #print(phicrit, limphi1, limphi2)
        ###############################################################################

        ## ¿Qué sucede si el contenido de agua exsuelta calculado es menor a 0?
        if exi<=0:
            Pcrit=(co/C1)**(1/beta) 
            
            if Pcrit<Patm:
                Pcrit=Patm

            
            deltP=Pi-Pcrit 
            deltH=-deltP/dpdzcalc
            if not np.isfinite(deltH) or deltH <= 0.0:
                deltH = min(abs(deltP) / max(abs(dpdzcalc), 1.0), abs(H) * 0.8)
            Hi=float(np.clip(H+deltH, H, -1e-3))
            deltH = Hi - H
            hspacing = 500

            zetash = np.zeros(hspacing)
            solaux = np.zeros((hspacing, 6))

            for p in range(1, hspacing+1):
                zetash[p-1] = H + (p-1)*deltH/hspacing
                solaux[p-1, 0] = Pi + (p-1)*deltH*dpdzcalc/hspacing
                solaux[p-1, 1] = 0
                solaux[p-1, 2] = Nd
                solaux[p-1, 3] = xi
                solaux[p-1, 4] = vinicial
                solaux[p-1, 5] = vinicial

            zsoladi = zetash
            sol1 = solaux

            zsol = np.concatenate((zsoladi, [Hi]))
            sol = np.vstack((sol1, [Pcrit, 0, Nd, xi, vinicial, vinicial]))

            if Pcrit>Patm:

                if Hi<0:
                    zsoladi = zsol
                    sol1 = sol

                    ##Cálculo de punto adicional por problemas númericos
                    epsd=1 #Delta de distancia vertical
                    if Hi>(-epsd):
                        epsd=-Hi/10

                    had=Hi+epsd
                    Pad=Pcrit + dpdzcalc*(had-Hi)
                    # Mismo y0 que el original (Henry duro). Solo si sH sigue
                    # a medias se baja un poco P para que IDA inicialice.
                    test_ad = (1.0 - xi) * co - (1.0 - xi) * C1 * Pad ** beta
                    if float(s_henry(test_ad, eps_henry)) < 0.9:
                        P_ok = _P_sH(0.95, xi)
                        if P_ok is not None and P_ok < Pad and dpdzcalc < 0.0:
                            had = float(np.clip(Hi + (P_ok - Pcrit) / dpdzcalc, Hi + epsd, -1e-3))
                            Pad = Pcrit + dpdzcalc*(had-Hi)
                        fgad, phiad, viadm, viadg = _fg_phi_vel(Pad, xi, q, rho_m)
                    else:
                        fgad=(1-xi)*(co - C1*Pad**beta)/(1-C1*Pad**beta)
                        phiad=1/(1 + (Pad/(fgad*R*T))*(1-fgad)/rho_m)
                        rho_gad=Pad/(R*T)
                        viadm= q*(1-fgad)/((1-phiad)*rho_m)
                        viadg= q*fgad/(phiad*rho_gad)
                    ## Fin calculo
                    
                    n_eq=1
                    # consistent initial conditions
                    y0 = np.array([Pad, phiad, Nd, xi, viadm, viadg])
                    yp0 = np.zeros_like(y0)
                    if unificado:
                        yp0 = momenteq1_u(had, y0)
                        y, t = solv_u(t0=had, tf=0, y0=y0, yp0=yp0, atol=errtol, rtol=errtol, n=1e4)
                        zsol = _pegar_t(zsoladi, t)
                        sol = _pegar(sol1, y)
                        zsol=np.real(zsol)
                        sol=np.real(sol)
                    else:
                        yp0 = momenteq1(had, y0)
                        y, t = solv(t0=had, tf=0, y0=y0, yp0=yp0, atol=errtol, rtol=errtol, n=1e4, param=n_eq, cond=limphi1)

                        zsol = _pegar_t(zsoladi, t)
                        sol = _pegar(sol1, y)
                        zsol=np.real(zsol)
                        sol=np.real(sol)
                else:
                    sol = np.vstack((sol1, [Pcrit+dpdzcalc*deltH, 0, Nd, xi, vinicial, vinicial]))
                    zsol=np.real(zsol)
                    sol=np.real(sol)

                if unificado:
                    xfinal = float(sol[sol[:,0].size-1, 3])
                else:
                    Hi1,Pi1,phini1 = zsol[sol[:,0].size-1],sol[sol[:,0].size-1,0],sol[sol[:,0].size-1,1]
                    if (Hi1<0 and Pi1>pfinal) and phini1>=limphi1:
                        um1, ug1, nd1, x1 = sol[sol[:,0].size-1,4],sol[sol[:,0].size-1,5],sol[sol[:,0].size-1,2],sol[sol[:,0].size-1,3]
                        n_eq=2
                        y0 = np.array([Pi1, phini1, nd1, x1, um1, ug1])
                        yp0 = np.zeros_like(y0)
                        yp0 = momenteq1(Hi1, y0)
                        y, t = solv(t0=Hi1, tf=0, y0=y0, yp0=yp0, atol=errtol, rtol=errtol, n=1e4, param=n_eq, cond=limphi2)
                        zsol = _pegar_t(zsol, t)
                        sol = _pegar(sol, y)

                    Hi2,Pi2,phini2 = zsol[sol[:,0].size-1],sol[sol[:,0].size-1,0],sol[sol[:,0].size-1,1]
                    if (Hi2<0 and Pi2>pfinal) and phini2>=limphi2:
                        um2, ug2, nd2, x2 = sol[sol[:,0].size-1,4],sol[sol[:,0].size-1,5],sol[sol[:,0].size-1,2],sol[sol[:,0].size-1,3]
                        n_eq=3
                        y0 = np.array([Pi2,phini2,nd2,x2,um2,ug2])
                        yp0 = np.zeros_like(y0)
                        yp0 = momenteq1(Hi2, y0)
                        y, t = solv(t0=Hi2, tf=0, y0=y0, yp0=yp0, atol=errtol, rtol=errtol, n=1e4, param=n_eq, cond=phicrit)
                        zsol = _pegar_t(zsol, t)
                        sol = _pegar(sol, y)

                    Hi3, Pi3, phini3, xfinal, ndfinal = zsol[sol[:,0].size-1],sol[sol[:,0].size-1,0],sol[sol[:,0].size-1,1],sol[sol[:,0].size-1,3],sol[sol[:,0].size-1,2]
                    if (Hi3<0 and Pi3>pfinal) and phini3>=phicrit:
                        n_eq=4
                        y0 = np.array([Pi3, phini3])
                        yp0 = np.zeros_like(y0)
                        yp0 = momenteq1(Hi3, y0)
                        y, t = solv(t0=Hi3, tf=0, y0=y0, yp0=yp0, atol=errtol, rtol=errtol, n=1e4, param=n_eq)
                        zsol = _pegar_t(zsol, t)
                        y = np.atleast_2d(y)
                        if y.size > 0 and y.shape[1] >= 2:
                            numb2 = y[:,0].size
                            um = np.zeros(numb2)
                            ug = np.zeros(numb2)
                            nd3 = np.zeros(numb2)
                            xf3 = np.zeros(numb2)
                            n = np.zeros(numb2)
                            rho_g = np.zeros(numb2)
                            for j in range(numb2):
                                test = co - C1*y[j,0]**beta
                                if test<=0:
                                    n[j] =0
                                else:
                                    n[j] = (co - C1*y[j,0]**beta)/(1-C1*y[j,0]**beta)
                                rho_g[j]=y[j,0]/(R*T);
                                um[j]=(1-n[j])*q/(rho_m*(1-y[j,1]))
                                ug[j]=n[j]*q/(rho_g[j]*y[j,1])
                                nd3[j]=ndfinal
                                xf3[j]=xfinal
                            sol4=np.column_stack((y[:,0],y[:,1], nd3,xf3, um, ug))
                            sol = _pegar(sol, sol4)
            else:
                n_eq=1
                # consistent initial conditions
                y0 = np.array([Pi, phini, Nd, xi, vinicial, vinicial])
                yp0 = np.zeros_like(y0)
                if unificado:
                    yp0 = momenteq1_u(H, y0)
                    y, t = solv_u(t0=H, tf=0, y0=y0, yp0=yp0, atol=errtol, rtol=errtol, n=1e4)
                    zsol=np.real(t)
                    sol=np.real(y)
                    xfinal = float(sol[sol[:,0].size-1, 3])
                else:
                    yp0 = momenteq1(had, y0)
                    y, t = solv(t0=H, tf=0, y0=y0, yp0=yp0, atol=errtol, rtol=errtol, n=1e4, param=n_eq, cond=limphi1)

                    zsol=np.real(t)
                    sol=np.real(y)
                
                    Hi1,Pi1,phini1 = zsol[sol[:,0].size-1],sol[sol[:,0].size-1,0],sol[sol[:,0].size-1,1]
                    if (Hi1<0 and Pi1>pfinal) and phini1>=limphi1:
                        um1, ug1, nd1, x1 = sol[sol[:,0].size-1,4],sol[sol[:,0].size-1,5],sol[sol[:,0].size-1,2],sol[sol[:,0].size-1,3]
                        n_eq=2
                        y0 = np.array([Pi1, phini1, nd1, x1, um1, ug1])
                        yp0 = np.zeros_like(y0)
                        yp0 = momenteq1(Hi1, y0)
                        y, t = solv(t0=Hi1, tf=0, y0=y0, yp0=yp0, atol=errtol, rtol=errtol, n=1e4, param=n_eq, cond=limphi2)
                    
                        zsol = _pegar_t(zsol, t)
                        sol = _pegar(sol, y)  
                
                    Hi2,Pi2,phini2 = zsol[sol[:,0].size-1],sol[sol[:,0].size-1,0],sol[sol[:,0].size-1,1]
                    if (Hi2<0 and Pi2>pfinal) and phini2>=limphi2:
                        um2, ug2, nd2, x2 = sol[sol[:,0].size-1,4],sol[sol[:,0].size-1,5],sol[sol[:,0].size-1,2],sol[sol[:,0].size-1,3]
                        n_eq=3
                        y0 = np.array([Pi2,phini2,nd2,x2,um2,ug2])
                        yp0 = np.zeros_like(y0)
                        yp0 = momenteq1(Hi2, y0)
                        y, t = solv(t0=Hi2, tf=0, y0=y0, yp0=yp0, atol=errtol, rtol=errtol, n=1e4, param=n_eq, cond=phicrit)

                        zsol = _pegar_t(zsol, t)
                        sol = _pegar(sol, y) 
                
                    Hi3,Pi3,phini3, xfinal, ndfinal = zsol[sol[:,0].size-1],sol[sol[:,0].size-1,0],sol[sol[:,0].size-1,1],sol[sol[:,0].size-1,3],sol[sol[:,0].size-1,2]
                    if (Hi3<0 and Pi3>pfinal) and phini3>=phicrit:
                        n_eq=4
                        y0 = np.array([Pi3, phini3])
                        yp0 = np.zeros_like(y0)
                        yp0 = momenteq1(Hi3, y0)
                        y, t = solv(t0=Hi3, tf=0, y0=y0, yp0=yp0, atol=errtol, rtol=errtol, n=1e4, param=n_eq)
                    
                        zsol = _pegar_t(zsol, t)
                        y = np.atleast_2d(y)
                        if y.size > 0 and y.shape[1] >= 2:
                            numb2 = y[:,0].size
                            um = np.zeros(numb2)
                            ug = np.zeros(numb2)
                            nd3 = np.zeros(numb2)
                            xf3 = np.zeros(numb2)
                            n = np.zeros(numb2)
                            rho_g = np.zeros(numb2)
                            for j in range(numb2):
                                test = co - C1*y[j,0]**beta
                                if test<=0:
                                    n[j] =0
                                else:
                                    n[j] = (co - C1*y[j,0]**beta)/(1-C1*y[j,0]**beta)
                                rho_g[j]=y[j,0]/(R*T);
                                um[j]=(1-n[j])*q/(rho_m*(1-y[j,1]))
                                ug[j]=n[j]*q/(rho_g[j]*y[j,1])
                                nd3[j]=ndfinal
                                xf3[j]=xfinal
                            sol4=np.column_stack((y[:,0],y[:,1], nd3,xf3, um, ug))
                            sol = _pegar(sol, sol4)
        #Aquí termina la condición si el agua exsuelta es mayor a 0
        numb=sol[:,0].size
        ugexit,pexit,zexit,phiexit = sol[numb-1,5], sol[numb-1,0], zsol[numb-1], sol[numb-1,1]
        print(f"Velocidad de salida: {ugexit} \nPresión de salida: {pexit} \nProfundidad de salida: {zexit} \nContenido de burbujas de salida: {phiexit}")
    
    #Awuí definimos las nuevas variables para el ciclo while
        count+=1
        condition1 = zexit >= -5
        condition2 = pexit >= pfinal and (ugexit > 0.95 * vsound and ugexit <= 1.05 * vsound)
        condition3 = pexit > (pfinal - 0.05e5) and pexit < (pfinal + 0.05e5)
        condition4 = phiexit >= phicrit

        #print(f"condition1 (zexit >= -5): {condition1}")
        #print(f"condition2 (pexit >= pfinal and (ugexit > 0.95 * vsound and ugexit <= 1.05 * vsound)): {condition2}")
        #print(f"condition3 (pexit > (pfinal - 0.05e5) and pexit < (pfinal + 0.05e5)): {condition3}")
        #print(f"condition4 (phiexit >= phicrit): {condition4}")
        if zexit>=-5 and ((pexit>=pfinal and (ugexit>0.95*vsound and ugexit<=1.05*vsound)) or ((pexit>(pfinal - 0.05e5) and pexit<(pfinal + 0.05e5)) and phiexit>=phicrit)):
            converged = True
            print(">>> Solucion de tiro convergida.")
            break

        if zexit >= -15 and abs(pexit - pfinal) < 1.5e5 and ugexit > 0.5:
            converged = True
            print(">>> Solucion de tiro convergida (tolerancia relajada en P/z).")
            break
        
        v_shot = vinicial
        if count > 20 and U_prev is not None and P_prev is not None:
            f0 = P_prev - pfinal
            f1 = pexit - pfinal
            if abs(f1 - f0) > 1e2:
                v_sec = v_shot - f1 * (v_shot - U_prev) / (f1 - f0)
                if np.isfinite(v_sec):
                    v_sec = float(np.clip(v_sec, vmin + 1e-4, vmax - 1e-4))
                    if vmin < v_sec < vmax:
                        vinicial = v_sec

        ida_corto = unificado and zexit < -50 and phiexit < max(0.05, 0.25 * limphi1)
        if ida_corto:
            print("IDA no llegó a la boca (φ chica): subo v, no la bajo.")
            vmin = v_shot
            if vinicial == v_shot:
                vinicial = vmin + (vmax - vmin) / 2
        elif pexit<pfinal:
            vmax=v_shot
            if vinicial == v_shot:
                vinicial=vmin + (vmax-vmin)/2
        else:
            if zexit<-5 or ugexit>1.02*vsound:
                vmax=v_shot
                if vinicial == v_shot:
                    vinicial = vmin + (vmax - vmin)/2
            else:
                if ugexit<=0.98*vsound:
                    vmin=v_shot
                    if vinicial == v_shot:
                        vinicial= vmin + (vmax-vmin)/2

        U_prev, P_prev, z_prev = v_shot, pexit, zexit
        vmin1, vmax1 = vmin, vmax

        print(f"vmax: {vmax} \nvmin: {vmin}")    
        ##End shooting method
        if (vmax1-vmin1)<0.0001:
            if zexit >= -15 and phiexit >= 0.5 * phicrit:
                converged = True
                print(">>> Solucion de tiro convergida (ventana vmin/vmax cerrada).")
            else:
                count=60
                print(f"vmax1: {vmax1} \nvmin1: {vmin1}")

    if not converged:
        print(
            "ADVERTENCIA: el metodo de tiro NO convergio. "
            f"Ultima z={zexit:.1f} m, P={pexit:.3e} Pa, ug={ugexit:.3f} m/s."
        )

    #results = np.concatenate((zsol[:,None],sol),axis=1)#Los resultados son en la primera fila la prof. y en las demás P, cont. burbujas, vm y ug, respectivamente

    zsol, sol = _sanear_perfil(zsol, sol, H - 50.0, 20.0)
    if zsol.size == 0:
        zsol = np.array([H, 0.0])
        sol = np.zeros((2, 6))
        count = 60

    numb=sol[:,0].size

    n = np.zeros(numb)
    visctot = np.zeros(numb)
    captot = np.zeros(numb)
    rbtot = np.zeros(numb)
    rbub = np.zeros(numb)
    dvdzf = np.zeros(numb)
    drdrbub = np.zeros(numb)
    dvdr = np.zeros(numb)

    for j in range(numb):
        test =(1-xi)*co - (1-sol[j,3])*C1*sol[j,0]**beta
 
        if test<=0:
            n[j] = 0
        else:
            n[j] = ((1-xi)*co - (1-sol[j,3])*C1*sol[j,0]**beta)/(1-C1*sol[j,0]**beta)

        if test>0:
            viscl = (fvrel(model, sol[j,3], xi, ar1, ar2, xmax, sol[j,4]/wr)
                *viscosity(sio2, tio2, al2o3, feo, mno, mgo, cao, na2o, k2o, p2o5, (C1*sol[j,0]**beta)*100, f2o, Tc))
        else:
            viscl = (fvrel(model, sol[j,3], xi, ar1, ar2, xmax, sol[j,4]/wr)
                *viscosity(sio2, tio2, al2o3, feo, mno, mgo, cao, na2o, k2o, p2o5, h2o, f2o, Tc))

        rb = ((sol[j,1]/((4/3)*np.pi*sol[j,2]*(1-sol[j,1]))))**(1/3)

        c1 = -0.2895*sol[j,1] + 0.8132
        c2 = sol[j,1]
        nca = rb*viscl*(sol[j,4]/wr)/3

        phicritbub = phicrit + 0.05
        AA = (1-(sol[j,1]/phicritbub))**(-phicritbub)
        BB = (1-(sol[j,1]/phicritbub))**(5*phicritbub/3)

        viscrel = 0.5*(AA-BB)*(1-math.erf(np.real(c1*np.log(nca)+c2))) + BB

        visc = viscrel*viscl

        rbtot[j] = viscrel
        captot[j] = nca
        visctot[j] = visc
        if sol[j,1]>phicrit:
            visctot[j] = visctot[j-1]
        rbub[j] = rb

        if j==0:
            dvdzf[j]=0
            drdrbub[j]=0
            dvdr[j]=0
        else:
            dvdzf[j] = (sol[j, 3]-sol[j-1,3])/(zsol[j]-zsol[j-1])
            drdrbub[j] = (rbub[j]-rbub[j-1])*sol[j,3]/((zsol[j]-zsol[j-1])*rbub[j])
            dvdr[j] = sol[j,3]/wr

    return [zsol, sol, count, vinicial, rho_m, rbub, visctot, rho_ti, fragcrit, phicrit, limphi1, limphi2, co, xi, xfinal, cg]


def RIconduitex5_5_unificado_f(radius, Pressure, wt, Temperature, content_crystal,
                               eps_frag=EPS_FRAG, eps_phi=EPS_PHI, eps_henry=EPS_HENRY,
                               eps_re=EPS_RE, eps_xi=EPS_XI, eps_rb=EPS_RB):
    """Alias. El import público es RIconduitex5_5_unificado (no este archivo)."""
    return RIconduitex5_5_suave_f(
        radius, Pressure, wt, Temperature, content_crystal,
        eps_frag=eps_frag, eps_phi=eps_phi, eps_henry=eps_henry,
        eps_re=eps_re, eps_xi=eps_xi, eps_rb=eps_rb, unificado=True)


# alias por si alguien copia el unpack de mainconduit5_5
RIconduitex5_5_f = RIconduitex5_5_suave_f
RIconduitex5_5_reg_f = RIconduitex5_5_suave_f