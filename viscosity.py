 #Viscosity function based in the work of Giordano et al. (2008)

import numpy as np

def viscosity(Wsio2,Wtio2,Wal2o3,Wfeo,Wmno,Wmgo,Wcao,Wna2o,Wk2o,Wp2o5,Wh2o,Wf2o,mTc):

    #1 wt # of composition
    
    totalElements=(Wsio2+Wtio2+Wal2o3+Wfeo+Wmgo+Wcao+Wna2o+Wk2o+Wf2o) 

    sio2n=(100-Wh2o)*Wsio2/totalElements 
    tio2n=(100-Wh2o)*Wtio2/totalElements 
    al2o3n=(100-Wh2o)*Wal2o3/totalElements 
    feon=(100-Wh2o)*Wfeo/totalElements 
    mnon=(100-Wh2o)*Wmno/totalElements 
    mgon=(100-Wh2o)*Wmgo/totalElements 
    caon=(100-Wh2o)*Wcao/totalElements 
    na2on=(100-Wh2o)*Wna2o/totalElements 
    k2on=(100-Wh2o)*Wk2o/totalElements 
    p2o5n=(100-Wh2o)*Wp2o5/totalElements 
    f2on=(100-Wh2o)*Wf2o/totalElements 
    h2on=Wh2o 

    #Constants
        #1 Molecular weights 
    Msio2=60.085  
    Mtio2=79.899  
    Mal2o3=101.961  
    Mfeo=71.846 
    Mmno=70.9374 
    Mmgo=40.304  
    Mcao=56.079  
    Mna2o=61.979  
    Mk2o=94.203  
    Mp2o5=141.9446 
    Mh2o=18.01528  
    Mf2o=37.9968 

    AT= -4.55 
    b1=159.56   
    b2=-173.34  
    b3=72.13  
    b4=75.69  
    b5=-38.98  
    b6=-84.08  
    b7=141.54  
    b11=-2.43 
    b12=-0.91 
    b13=17.62 

    c1=2.75 
    c2=15.72 
    c3=8.32 
    c4=10.2 
    c5=-12.29 
    c6=-99.54 
    c11=0.3 

    #Calculations
        #Step 1 Mol Prop
    a=sio2n/Msio2  
    b=tio2n/Mtio2  
    c=al2o3n/Mal2o3  
    d=feon/Mfeo  
    e=mgon/Mmgo  
    f=caon/Mcao 
    g=na2on/Mna2o  
    h=k2on/Mk2o  
    j=h2on/Mh2o  
    k=f2on/Mf2o 
    m=mnon/Mmno 
    n=p2o5n/Mp2o5 

    GFW=100/(a+b+c+d+e+f+g+h+j+k+m+n) 

    molsio2=GFW*sio2n/Msio2 
    moltio2=GFW*tio2n/Mtio2 
    molal2o3=GFW*al2o3n/Mal2o3 
    molfeo=GFW*feon/Mfeo 
    molmgo=GFW*mgon/Mmgo 
    molcao=GFW*caon/Mcao 
    molna2o=GFW*na2on/Mna2o 
    molk2o=GFW*k2on/Mk2o 
    molh2o=GFW*h2on/Mh2o 
    molf2o=GFW*f2on/Mf2o 
    molp2o5=GFW*p2o5n/Mp2o5 
    molmno=GFW*mnon/Mmno 


    bb1=b1*(molsio2+moltio2) 
    bb2=b2*molal2o3 
    bb3=b3*(molfeo+molp2o5+molmno) 
    bb4=b4*molmgo 
    bb5=b5*molcao 
    bb6=b6*(molna2o+molh2o+molf2o) 
    bb7=b7*(molh2o+molf2o + np.log(1+molh2o)) 
    bb11=b11*(molsio2+moltio2)*(molfeo+molmno+molmgo) 
    bb12=b12*(molsio2+moltio2+molal2o3+molp2o5)*(molna2o+molk2o+molh2o) 
    bb13=b13*(molal2o3)*(molna2o+molk2o) 

    cc1=c1*molsio2 
    cc2=c2*(moltio2+molal2o3) 
    cc3=c3*(molfeo+molmno+molmgo) 
    cc4=c4*molcao 
    cc5=c5*(molna2o+molk2o) 
    cc6=c6*(np.log(1+molh2o+molf2o)) 
    cc11=c11*(molal2o3+molfeo+molmno+molmgo+molcao-molp2o5)*(molna2o+molk2o+molh2o+molf2o) 


    B=bb1+bb2+bb3+bb4+bb5+bb6+bb7+bb11+bb12+bb13 
    C=cc1+cc2+cc3+cc4+cc5+cc6+cc11 

    lnvisc= AT + B/(mTc+273.15-C) 

    viscalc=10**lnvisc 

    return viscalc

    