#Magma Density Calculation 
#Author: Tom D Pering - University of Sheffield
#You are free to use and alter with acknowledgement
#The calculations are based on the method of Bottinga & Weill - ref below

#Bottinga & Weill (1970).
#Densities of liquid silicate systems calculated from partial molar volumes
#of oxide components.
#American Journal of Science 269, pp. 169-182

#I have also used the following Excel programs as a guide
#John D. Winter - Available at: http://www.whitman.edu/geology/winter/
#"Magma Density spreadsheet"
#GabbroSoft - http://www.gabbrosoft.org/spreadsheets.html
#"MAGMA-DENSITY"

#Inputs
#Input the wt # of each element listed and the wt # of water (Wh2o) and the magma
#temperature in celsius (mTc)

def density(Wsio2,Wtio2,Wal2o3,Wfeo,Wmgo,Wcao,Wna2o,Wk2o,Wh2o,mTc,mPp):
        #1 wt # of composition

    #totalElements=(Wsio2+Wtio2+Wal2o3+Wfeo+Wmgo+Wcao+Wna2o+Wk2o)

        #2 wt # of gases
    #totalGas=Wh2o 

        #3 Totals (can't exceed 100 #)
    #totalweight=(totalElements+totalGas) 

    #Constants
        #1 Molecular weights 
    totalElements=(Wsio2+Wtio2+Wal2o3+Wfeo+Wmgo+Wcao+Wna2o+Wk2o)
    
     #wt Normalizado sin considerar agua
    Wsio2=(100-Wh2o)*Wsio2/totalElements 
    Wtio2=(100-Wh2o)*Wtio2/totalElements 
    Wal2o3=(100-Wh2o)*Wal2o3/totalElements 
    Wfeo=(100-Wh2o)*Wfeo/totalElements 
    Wmgo=(100-Wh2o)*Wmgo/totalElements 
    Wcao=(100-Wh2o)*Wcao/totalElements 
    Wna2o=(100-Wh2o)*Wna2o/totalElements 
    Wk2o=(100-Wh2o)*Wk2o/totalElements 
    
    #Peso molecular (gr/mol)
    Msio2=60.085 
    Mtio2=79.899 
    Mal2o3=101.961 
    Mfeo=71.846 
    Mmgo=40.304 
    Mcao=56.079 
    Mna2o=61.979 
    Mk2o=94.203 
    Mh2o=18.01528 

    #Calculations
        #Step 1 Mol Prop
    a=Wsio2/Msio2 
    b=Wtio2/Mtio2 
    c=Wal2o3/Mal2o3 
    d=Wfeo/Mfeo 
    e=Wmgo/Mmgo 
    f=Wcao/Mcao
    g=Wna2o/Mna2o 
    h=Wk2o/Mk2o 
    j=Wh2o/Mh2o 

     # Molecular volume at 1400C (cc/mol)
    MVsio2=26.9   
    MVtio2=23.16   
    MVal2o3=37.11  
    MVfeo=13.65  
    MVmgo=11.45  
    MVcao=16.57  
    MVna2o=28.78  
    MVk2o=45.84 
    MVh2o=17  

    #DV/DT

    dvdtsio2=0   
    dvdttio2=7.24   
    dvdtal2o3=2.62  
    dvdtfeo=2.92  
    dvdtmgo=2.62  
    dvdtcao=2.92  
    dvdtna2o=7.41  
    dvdtk2o=11.91 
    dvdth2o=9.46 

    #DV/DP

    dvdpsio2=-1.89   
    dvdptio2=-2.31   
    dvdpal2o3=-2.26  
    dvdpfeo=-0.45  
    dvdpmgo=-0.27  
    dvdpcao=0.34  
    dvdpna2o=-2.4  
    dvdpk2o=-6.75 
    dvdph2o=-3.15 

     #Volumen molar a temperatura T y P
    a4=MVsio2+dvdtsio2*0.001*(mTc-1400)+dvdpsio2*0.001*(mPp-0.1)  
    b4=MVtio2+dvdttio2*0.001*(mTc-1400)+dvdptio2*0.001*(mPp-0.1)  
    c4=MVal2o3+dvdtal2o3*0.001*(mTc-1400)+dvdpal2o3*0.001*(mPp-0.1)  
    d4=MVfeo+dvdtfeo*0.001*(mTc-1400)+dvdpfeo*0.001*(mPp-0.1)  
    e4=MVmgo+dvdtmgo*0.001*(mTc-1400)+dvdpmgo*0.001*(mPp-0.1)  
    f4=MVcao+dvdtcao*0.001*(mTc-1400)+dvdpcao*0.001*(mPp-0.1)  
    g4=MVna2o+dvdtna2o*0.001*(mTc-1400)+dvdpna2o*0.001*(mPp-0.1)  
    h4=MVk2o+dvdtk2o*0.001*(mTc-1400)+dvdpk2o*0.001*(mPp-0.1)  
    j4=MVh2o+dvdth2o*0.001*(mTc-1400)+dvdph2o*0.001*(mPp-0.1)  

    # volumen molecular (cc/mol) * Mol Prop (mol algo/gr) = 1/densidad
    a5=a*a4  
    b5=b*b4  
    c5=c*c4  
    d5=d*d4  
    e5=e*e4  
    f5=f*f4  
    g5=g*g4  
    h5=h*h4  
    j5=j*j4  

    suma=a5+b5+c5+d5+e5+f5+g5+h5+j5    #Suma de 1/densidades de cada molecula (gr/cc)

    density_gas=100000/suma   
    #density_no_gas=100000/(suma-j5)       #without so2 and co2

    return density_gas