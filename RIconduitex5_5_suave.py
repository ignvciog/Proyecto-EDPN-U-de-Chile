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
    s_frag, s_henry, s_re, softplus, guarda_coalescencia, mezclar,
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


def _melt_wall_friction(visc, um):
    if suppress_Fmw:
        return float(radial_Fmw_override) if radial_Fmw_override is not None else 0.0
    if radial_Fmw_override is not None:
        return float(radial_Fmw_override)
    return cg * visc * um / (wr ** 2)


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

        f2=float(softplus((co*(1-xi)-((1-y[3])*C1*y[0]**beta))/(co*(1-xi) - (1-xmax)*C1*(Patm)**beta), eps_xi))
        xteo=xi + (xmax-xi)*f2
        f3=float(softplus(1-y[3]/xteo, eps_xi))
        
        dxdp=float(softplus((xmax-xi)*f2*f3/(tcar*y[4]), eps_xi))
        dfgdp=(-(-dxdp*C1*y[0]**beta + (1-y[3])*C1*beta*y[0]**(beta-1))+(co*(1-xi)-(1-y[3])*C1*y[0]**beta)*C1*beta*y[0]**(beta-1))/((1-C1*y[0]**beta)**2)
        
        
        Fgw=0 #Fuerza int. gas-wall
        Fmw=_melt_wall_friction(visc, y[4])#Fuerza int. melt-wall
        if n_eq==1:
            Fmg=3*visc*(y[5]-y[4])*y[1]*(1-y[1])/(rb**2)

        if n_eq==2:
            tt=(y[1]-limphi1)/(limphi2-limphi1)
            Re=2*rb*rho_g*(y[5]-y[4])/(1e-5)
            kper=0.131*(rb**2)*((y[1]-limphi1 + 0.05)**2.1)
            Fmg_in=((0.33/(4*rb))*rho_g*np.abs(y[5]-y[4])**tt)*(((3)*visc*(1/(rb**2)))**(1-tt))*(y[5]-y[4])*y[1]*(1-y[1])
            Fmg_st=((1e-5/kper)**tt)*(((3)*visc*(1/(rb**2)))**(1-tt))*(y[5]-y[4])*y[1]*(1-y[1])
            Fmg=mezclar(float(s_re(Re, eps=eps_re)), Fmg_st, Fmg_in)

        if n_eq==3:
            Re=2*rb*rho_g*(y[5]-y[4])/(1e-5)
            kper=0.131*(rb**2)*((y[1]-limphi1 + 0.05)**2.1)
            Fmg_in=(0.33/(4*rb))*rho_g*(y[5]-y[4])*(y[5]-y[4])*y[1]*(1-y[1])
            Fmg_st=(1e-5/kper)*(y[5]- y[4])*y[1]*(1-y[1])
            Fmg=mezclar(float(s_re(Re, eps=eps_re)), Fmg_st, Fmg_in)

        Fmw, Fgw, Fmg, sfrag = _mezclar_fragmentacion(
            y[1], rho_g, y[4], y[5], Fmw, Fgw, Fmg)
        
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
            f2 = float(softplus((co-(C1*y[0]**beta))/(co - C1*(Patm)**beta), eps_xi))
            xteo=xi + (xmax-xi)*f2
            f3 = float(softplus(1-y[3]/xteo, eps_xi))

         ########### En momenteq aca va multp. pr y[4] en matlab (Preguntar)

        if n_eq==3:
            dxdz=float(softplus((xmax-xi)*f2*f3/(tcar*y[4]), eps_xi))
        else:
            dxdz=float(softplus((xmax-xi)*f2*f3/(tcar), eps_xi))

        dNdt = (1.0 - sfrag) * dNdt
        dxdz = (1.0 - sfrag) * dxdz

        result[0] = -yprime[0] + dpdz 
        result[1] = -yprime[1] + dphidz
        result[2] = -yprime[2] + dNdt/y[4]
        result[3] = -yprime[3] + dxdz
        result[4] =  y[4] - q*(1-fg)/((1-y[1])*rho_m)
        result[5] = y[5] - q*fg/(y[1]*rho_g)

    else:
        test=(1-xi)*co - (1-xfinal)*C1*y[0]**beta
        fg_sat=(co*(1-xi) - C1*(1-xfinal)*y[0]**beta)/((1-C1*y[0]**beta))
        sH=float(s_henry(test, eps_henry))
        fg=sH*fg_sat

        ra=1e-3 
        cd=0.8
        f2=float(softplus((co*(1-xi)-((1-xfinal)*C1*y[0]**beta))/(co*(1-xi) - (1-xmax)*C1*(Patm)**beta), eps_xi))
        xteo=xi + (xmax-xi)*f2
        f3=float(softplus(1-xfinal/xteo, eps_xi))
        dxdp=float(softplus((xmax-xi)*f2*f3/tcar, eps_xi)) #######Aqui no va multiplicado
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

        f2=float(softplus((co*(1-xi)-((1-y[3])*C1*y[0]**beta))/(co*(1-xi) - (1-xmax)*C1*(Patm)**beta), eps_xi))
        xteo=xi + (xmax-xi)*f2
        f3=float(softplus(1-y[3]/xteo, eps_xi))
        
        dxdp=float(softplus((xmax-xi)*f2*f3/(tcar*y[4]), eps_xi))
        dfgdp=(-(-dxdp*C1*y[0]**beta + (1-y[3])*C1*beta*y[0]**(beta-1))+(co*(1-xi)-(1-y[3])*C1*y[0]**beta)*C1*beta*y[0]**(beta-1))/((1-C1*y[0]**beta)**2)
        
        
        Fgw=0 #Fuerza int. gas-wall
        Fmw=_melt_wall_friction(visc, y[4])#Fuerza int. melt-wall
        if n_eq==1:
            Fmg=3*visc*(y[5]-y[4])*y[1]*(1-y[1])/(rb**2)

        if n_eq==2:
            tt=(y[1]-limphi1)/(limphi2-limphi1)
            Re=2*rb*rho_g*(y[5]-y[4])/(1e-5)
            kper=0.131*(rb**2)*((y[1]-limphi1 + 0.05)**2.1)
            Fmg_in=((0.33/(4*rb))*rho_g*np.abs(y[5]-y[4])**tt)*(((3)*visc*(1/(rb**2)))**(1-tt))*(y[5]-y[4])*y[1]*(1-y[1])
            Fmg_st=((1e-5/kper)**tt)*(((3)*visc*(1/(rb**2)))**(1-tt))*(y[5]-y[4])*y[1]*(1-y[1])
            Fmg=mezclar(float(s_re(Re, eps=eps_re)), Fmg_st, Fmg_in)

        if n_eq==3:
            Re=2*rb*rho_g*(y[5]-y[4])/(1e-5)
            kper=0.131*(rb**2)*((y[1]-limphi1 + 0.05)**2.1)
            Fmg_in=(0.33/(4*rb))*rho_g*(y[5]-y[4])*(y[5]-y[4])*y[1]*(1-y[1])
            Fmg_st=(1e-5/kper)*(y[5]- y[4])*y[1]*(1-y[1])
            Fmg=mezclar(float(s_re(Re, eps=eps_re)), Fmg_st, Fmg_in)

        Fmw, Fgw, Fmg, sfrag = _mezclar_fragmentacion(
            y[1], rho_g, y[4], y[5], Fmw, Fgw, Fmg)
        
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
            f2 = float(softplus((co-(C1*y[0]**beta))/(co - C1*(Patm)**beta), eps_xi))
            xteo=xi + (xmax-xi)*f2
            f3 = float(softplus(1-y[3]/xteo, eps_xi))

        if n_eq==3:
            dxdz=float(softplus((xmax-xi)*f2*f3/(tcar*y[4]), eps_xi))
        else:
            dxdz=float(softplus((xmax-xi)*f2*f3/(tcar), eps_xi))

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
        f2=float(softplus((co*(1-xi)-((1-xfinal)*C1*y[0]**beta))/(co*(1-xi) - (1-xmax)*C1*(Patm)**beta), eps_xi))
        xteo=xi + (xmax-xi)*f2
        f3=float(softplus(1-xfinal/xteo, eps_xi))
        dxdp=float(softplus((xmax-xi)*f2*f3/tcar, eps_xi)) #######Aqui no va multiplicado
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

    if param in [1,2,3]:
        solver = dae('ida', momenteq, 
                compute_initcond='yp0',
                first_step_size=1e-18,
                atol=atol,
                rtol=rtol,
                algebraic_vars_idx=[4, 5],
                old_api=False)
        #print(t0, y0, yp0)
        solver.init_step(t0, y0, yp0)
        #print(solver)
        y_anterior = y0[1]  # Valor inicial de y[1]
        # Lista para almacenar las soluciones
        y_values = []
        t_values = []
        for time in tspan[1:]:  
            solution = solver.step(time)
            if solution.values.y is None:
                break
            y_actual = solution.values.y[1]  # Valor actual de y[1]
            # Guardar tiempo y valor de y[1] en la lista de soluciones
            y_values.append(solution.values.y)
            t_values.append(solution.values.t)

            # Detectar cruce con limphi1 FIJO
            if (y_anterior < cond and y_actual >= cond):  
                #print(f"Cruzó de ABAJO hacia ARRIBA en t = {solution.values.t}, limphi1 = {cond}")
                break  
            elif (y_anterior > cond and y_actual <= cond):  
                #print(f"Cruzó de ARRIBA hacia ABAJO en t = {solution.values.t}, limphi1 = {cond}")
                break  

            # Actualizar valores para laprint siguiente iteración
            y_anterior = y_actual
    else:
        solver = dae('ida', momenteq, 
                compute_initcond='yp0',
                first_step_size=1e-18,
                atol=atol,
                rtol=rtol,
                old_api=False)
        solution = solver.solve(tspan, y0, yp0)
        y_values, t_values = solution.values.y, solution.values.t
    

    return y_values, t_values

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
                           eps_re=EPS_RE, eps_xi=EPS_XI, eps_rb=EPS_RB):
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
            
            Hi=H+deltH
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

                    had=Hi+epsd  #altura (o prof) adicional
                    Pad=Pcrit + dpdzcalc*epsd  #presión a esa prof. adicional
                    fgad=(1-xi)*(co - C1*Pad**beta)/(1-C1*Pad**beta)  #caudal másico para el punto adicional
                    phiad=1/(1 + (Pad/(fgad*R*T))*(1-fgad)/rho_m)  #contenido de burbujas de este punto adicional
                    rho_gad=Pad/(R*T)  #Densidad del gas en este punto adicional
                    viadm= q*(1-fgad)/((1-phiad)*rho_m)  #velocidad inicial en el punto adicional del fundido
                    viadg= q*fgad/(phiad*rho_gad)  #Velocidad inicial en el punto adicional del gas
                    ## Fin calculo
                    
                    n_eq=1
                    # consistent initial conditions
                    y0 = np.array([Pad, phiad, Nd, xi, viadm, viadg])
                    yp0 = np.zeros_like(y0)
                    yp0 = momenteq1(had, y0)
                    y, t = solv(t0=had, tf=0, y0=y0, yp0=yp0, atol=errtol, rtol=errtol, n=1e4, param=n_eq, cond=limphi1)

                    zsol = np.append(zsoladi, t)
                    sol = np.vstack((sol1, y))
                    zsol=np.real(zsol)
                    sol=np.real(sol)
                else:
                    sol = np.vstack((sol1, [Pcrit+dpdzcalc*deltH, 0, Nd, xi, vinicial, vinicial]))
                    zsol=np.real(zsol)
                    sol=np.real(sol)

                Hi1,Pi1,phini1 = zsol[sol[:,0].size-1],sol[sol[:,0].size-1,0],sol[sol[:,0].size-1,1]
                #print(Hi1, Pi1)
                if (Hi1<0 and Pi1>pfinal) and phini1>=limphi1:
                    um1, ug1, nd1, x1 = sol[sol[:,0].size-1,4],sol[sol[:,0].size-1,5],sol[sol[:,0].size-1,2],sol[sol[:,0].size-1,3]
                    #Segunda condición (momentumeq2)
                    n_eq=2
                    y0 = np.array([Pi1, phini1, nd1, x1, um1, ug1])
                    yp0 = np.zeros_like(y0)
                    yp0 = momenteq1(Hi1, y0)
                    y, t = solv(t0=Hi1, tf=0, y0=y0, yp0=yp0, atol=errtol, rtol=errtol, n=1e4, param=n_eq, cond=limphi2)
                    
                    zsol = np.append(zsol, t)
                    sol = np.vstack((sol, y))  
                    
                Hi2,Pi2,phini2 = zsol[sol[:,0].size-1],sol[sol[:,0].size-1,0],sol[sol[:,0].size-1,1]
                #print(Hi2, Pi2)
                if (Hi2<0 and Pi2>pfinal) and phini2>=limphi2:
                    um2, ug2, nd2, x2 = sol[sol[:,0].size-1,4],sol[sol[:,0].size-1,5],sol[sol[:,0].size-1,2],sol[sol[:,0].size-1,3]
                    n_eq=3
                    y0 = np.array([Pi2,phini2,nd2,x2,um2,ug2])
                    yp0 = np.zeros_like(y0)
                    yp0 = momenteq1(Hi2, y0)
                    y, t = solv(t0=Hi2, tf=0, y0=y0, yp0=yp0, atol=errtol, rtol=errtol, n=1e4, param=n_eq, cond=phicrit)

                    zsol = np.append(zsol, t)
                    sol = np.vstack((sol, y))  
                
                Hi3, Pi3, phini3, xfinal, ndfinal = zsol[sol[:,0].size-1],sol[sol[:,0].size-1,0],sol[sol[:,0].size-1,1],sol[sol[:,0].size-1,3],sol[sol[:,0].size-1,2]
                #print(Hi3, Pi3)
                if (Hi3<0 and Pi3>pfinal) and phini3>=phicrit:
                    n_eq=4
                    y0 = np.array([Pi3, phini3])
                    yp0 = np.zeros_like(y0)
                    yp0 = momenteq1(Hi3, y0)
                    y, t = solv(t0=Hi3, tf=0, y0=y0, yp0=yp0, atol=errtol, rtol=errtol, n=1e4, param=n_eq)
                    #sol4 = solve_ivp(momenteq1,[Hi3,0],np.array([Pi3,phini3]),atol=errtol,rtol=errtol,method='BDF')
                    #zsol,sol = get_solution(zsol,sol,sol3) 

                    zsol = np.append(zsol,t)
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
                    sol = np.vstack((sol,sol4))
                    #print(zsol[sol[:,0].size-1], sol[sol[:,0].size-1,0])
            else:
                n_eq=1
                # consistent initial conditions
                y0 = np.array([Pi, phini, Nd, xi, vinicial, vinicial])
                yp0 = np.zeros_like(y0)
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
                    
                    zsol = np.append(zsol, t)
                    sol = np.vstack((sol, y))  
                
                Hi2,Pi2,phini2 = zsol[sol[:,0].size-1],sol[sol[:,0].size-1,0],sol[sol[:,0].size-1,1]
                if (Hi2<0 and Pi2>pfinal) and phini2>=limphi2:
                    um2, ug2, nd2, x2 = sol[sol[:,0].size-1,4],sol[sol[:,0].size-1,5],sol[sol[:,0].size-1,2],sol[sol[:,0].size-1,3]
                    n_eq=3
                    y0 = np.array([Pi2,phini2,nd2,x2,um2,ug2])
                    yp0 = np.zeros_like(y0)
                    yp0 = momenteq1(Hi2, y0)
                    y, t = solv(t0=Hi2, tf=0, y0=y0, yp0=yp0, atol=errtol, rtol=errtol, n=1e4, param=n_eq, cond=phicrit)

                    zsol = np.append(zsol, t)
                    sol = np.vstack((sol, y)) 
                
                Hi3,Pi3,phini3, xfinal, ndfinal = zsol[sol[:,0].size-1],sol[sol[:,0].size-1,0],sol[sol[:,0].size-1,1],sol[sol[:,0].size-1,3],sol[sol[:,0].size-1,2]
                if (Hi3<0 and Pi3>pfinal) and phini3>=phicrit:
                    n_eq=4
                    y0 = np.array([Pi3, phini3])
                    yp0 = np.zeros_like(y0)
                    yp0 = momenteq1(Hi3, y0)
                    y, t = solv(t0=Hi3, tf=0, y0=y0, yp0=yp0, atol=errtol, rtol=errtol, n=1e4, param=n_eq)
                    
                    zsol = np.append(zsol,t)
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
                    sol = np.vstack([sol,sol4])
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

        if pexit<pfinal:
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

        print(f"vmax: {vmax} \nvmin: {vmin}")    
        ##End shooting method
        if (vmax1-vmin1)<0.0001:
            count=60
            print(f"vmax1: {vmax1} \nvmin1: {vmin1}")

    if not converged:
        print(
            "ADVERTENCIA: el metodo de tiro NO convergio. "
            f"Ultima z={zexit:.1f} m, P={pexit:.3e} Pa, ug={ugexit:.3f} m/s."
        )

    #results = np.concatenate((zsol[:,None],sol),axis=1)#Los resultados son en la primera fila la prof. y en las demás P, cont. burbujas, vm y ug, respectivamente

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


# alias por si alguien copia el unpack de mainconduit5_5
RIconduitex5_5_f = RIconduitex5_5_suave_f
RIconduitex5_5_reg_f = RIconduitex5_5_suave_f