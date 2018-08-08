# Easier version. 2017. JA
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from L96_model import lorenz96
from L96_misc import gen_obs, rmse_spread, createH, getBsimple
from L96_kfs import kfs_lor96
from L96_plots import plotL96, plotL96obs, plotL96DA_kf, \
                      plotRMSP, plotRH, tileplotB


###############################################################################
### 1.a The Nature Run
# Let us perform a 'free' run of the model, which we will consider the truth
# The initial conditions
x0 = None # let it spin from rest (x_n(t=0) = F, forall n )
tmax = 4
Nx = 12
t,xt = lorenz96(tmax,x0,Nx) # Nx>=12
plotL96(t,xt,Nx)

# imperfect initial guess for our DA experiments
forc = 8.0; aux1 = forc*np.ones(Nx); aux2 = range(Nx); 
x0guess = aux1 + ((-1)*np.ones(Nx))**aux2
del aux1, aux2


###############################################################################
### 2. The observations
# Decide what variables to observe
obsgrid = '1010'
H, observed_vars = createH(obsgrid,Nx)
period_obs = 4
var_obs = 2
# Generating the observations
tobs,y,R = gen_obs(t,xt,period_obs,H,var_obs)
plotL96obs(t,xt,Nx,tobs,y,observed_vars)

    
##############################################################################
### 3. Data assimilation using KFa (SEnKF, LSEnKF and ETKF)
# No LETKF since R-localisation is extremely slow without parallel implementation    
rho = 0.05
M = 6
lam = 2
loctype = 'GC'
met = 'SEnKF' 
Xb,xb,Xa,xa,locmatrix = kfs_lor96(x0guess,t,tobs,y,H,R,rho,M,met,lam,loctype)
plotL96DA_kf(t,xt,tobs,y,Nx,observed_vars,Xb,xb,Xa,xa)

if np.any(locmatrix) != None:
 tileplotB(locmatrix)

rmse_step=1
rmseb,spreadb = rmse_spread(xt,xb,Xb,rmse_step)
rmsea,spreada = rmse_spread(xt,xa,Xa,rmse_step)
plotRMSP(t,rmseb,rmsea,spreadb,spreada)


