%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INPUT PARAMETERS FOR RICONDUIT 5.2 MODEL                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

global R T Tc co C1 beta phicrit H pfinal xi xmax tcar phimax Fc F1 F2 cg radius1 dl overP1;
global sio2 tio2 al2o3 feo mno mgo cao na2o k2o p2o5 f2o h2o;
global errtol lsup linf limperl limperh cA model ar1 ar2 geometry;
global vminef vmaxef vinicialef

% Glass chemical composition
namevolc='St Helens 2004';
sio2 =72.1;
tio2 =0.49;
al2o3=14.00;
feo  =3.83;
mno  =0.13;
mgo  =0.49;
cao  =1.56;
na2o =4.17;
k2o  =2.98;
p2o5 =0.07;
f2o  =0;
h2o  =4;

sio2tot=70;
composition='silicic'; % 'silicic' or 'basaltic' for radrub3 to chose the correct parameters

co=h2o/100; %initial water content in fraction (from 0 to 1)
C1=0.00000411; %constant solubility
beta=0.5; %constant solubility
R=461.11; %constant of water
Tc=900;%temperature in celsius
T=Tc+273.15; %temperature in kelvin


xi=0.05; %initial fraction of crystals (not yet included to estimate density)
xmax=0.10; %maximum crystal content
phimax=0.6; % max packing fraction
tcar=2*3600; % characteristic time of kinetic of crystallization
model='EM2s'; % rheological model: ER52=Einstein Roscoe 1952, EM2s=Effective medium with 2 solids, costa09=Costa et al., 2009, costa09v11=Costa et al., 2009 with paremeters of Vona et al., 2011, vona11=Vona et al., 2011
ar1=4; % aspect ratio of large phenocrysts (only used in EM2s and vona11 models)
ar2=8; % aspect ratio of microlites (only used in EM2s and vona11 models)


phicrit=0.8; %volume fraction criteria for fragmentation
H=-5000; %depth of top of magma chamber
geometry='cylinder'; %conduit geometry: cylinder or dyke
radius1=7; %cylinder radius or dyke half-width
dl=100; %dyke length (used only if geometry == dyke)

overP1=7e6; % overpressure at the inlet of the conduit
pfinal=1.013e5; %final pressure (atmospheric)
rcrust=2500; %crustal density (used to calculate lithostatic pressure)
g=9.81; %gravity acceleration

vminef=1e-5;
vmaxef=0.3;
vinicialef=0.5;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Do not change the following lines unless you are sure of what are you doing!!

F1=1.8; % relative size of large bubbles 
F2=0.2; % relative size of small bubbles (F1 + F2 should equal 2)
Fc=1; % efficiency of bubble coalescense (0 - 1) 

lsup=0.785; % upper limit of bubble content for fragmetation
linf=0.525; % lower limit of bubble content for fragmentation

limperh=0.3; % upper limit of bubble content for beginning of permeability 
limperl=0.05; % lower limit of bubble content for beginning of permeability


odesolver='ode15s'; % solver used by MatLab to solve system of diferential equations
errtol=1e-8; % error tolerance (absolute and relative) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% DO NOT CHANGE THE FOLLOWING LINES %
if strcmpi(model,'EM2s') || strcmpi(model,'vona11')

phimax1=0.656*exp(-(log10(ar1)^2)/(2*(1.08^2)));
phimax2=0.656*exp(-(log10(ar2)^2)/(2*(1.08^2)));

if xi>phimax1
    xi=0.95*phimax1;
end

if (xmax-xi)/(1-xi)>phimax2
    xmax=0.95*phimax2*(1-xi) + xi;
end

end

if strcmpi(geometry,'dyke')
    cg=3;
else
    cg=8;
end


cA=(limperh-limperl)/(linf-lsup);

% END OF SCRIPT %
