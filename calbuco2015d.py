###########################################################################
# INPUT PARAMETERS FOR RICONDUIT 5.2 MODEL                                #
###########################################################################
import numpy as np

# Glass chemical composition
sio2 =62.72 
tio2 =1 
al2o3=17 
feo  =5.44 
mno  =0.15 
mgo  =2.19 
cao  =5.22 
na2o =4.44 
k2o  =1.2 
p2o5 =0.1 
f2o  =0.1 
h2o1  =4 

sio2tot=62 

co1=h2o1/100  #initial water content in fraction (from 0 to 1)
C1=0.00000411  #constant solubility
beta=0.5  #constant solubility
R=461.11  #constant of water
Tc1=970 #temperature in celsius
T1=Tc1+273.15  #temperature in kelvin
Patm=1.01325e5 #atmospheric pressure

xi1=0.25  #initial fraction of crystals (not yet included to estimate density)
xmax=0.5  #maximum crystal content
phimax=0.6  # max packing fraction
tcar=2*3600  # characteristic time of kinetic of crystallization
model='em2s'  # rheological model: ER52=Einstein Roscoe 1952, EM2s=Effective medium with 2 solids, costa09=Costa et al., 2009, costa09v11=Costa et al., 2009 with paremeters of Vona et al., 2011, vona11=Vona et al., 2011
ar1=4  # aspect ratio of large phenocrysts (only used in EM2s and vona11 models)
ar2=8  # aspect ratio of microlites (only used in EM2s and vona11 models)


phicrit=0.8  #volume fraction criteria for fragmentation
H=-7000  #depth of top of magma chamber
geometry='cylinder'  #conduit geometry: cylinder or dyke
radius1=16  #cylinder radius or dyke half-width. Not used if user chooses that the code calculate this parameter
dl=100  #dyke length (used only if geometry == dyke)

overP1=5e6  # overpressure at the inlet of the conduit. Not used if user chooses that the code calculate this parameter
pfinal=Patm  #final pressure (atmospheric)
rcrust=2600  #crustal density (used to calculate lithostatic pressure)
g=9.81  #gravity acceleration

###########################################################################
# Do not change the following lines unless you are sure of what are you doing!!

F1=1.8  # relative size of large bubbles 
F2=0.2  # relative size of small bubbles (F1 + F2 should equal 2)
Fc=1  # efficiency of bubble coalescense (0 - 1) 

lsup=0.785  # upper limit of bubble content for fragmetation
linf=0.525  # lower limit of bubble content for fragmentation

limperh=0.4  # upper limit of bubble content for beginning of permeability 
limperl=0.15  # lower limit of bubble content for beginning of permeability

errtol=1e-8  # error tolerance (absolute and relative) 
###########################################################################

