# Easier version. 2017. JA
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from L96_model import lorenz96
from L96_misc import gen_obs, rmse_spread, createH, getBsimple
from L96_var import var3d, var4d
from L96_plots import plotL96, plotL96obs, plotL96DA_var, plotRMSP, tileplotB


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
period_obs = 2
var_obs = 2
# Generating the observations
tobs,y,R = gen_obs(t,xt,period_obs,H,var_obs)
plotL96obs(t,xt,Nx,tobs,y,observed_vars)

    
##############################################################################
## 3 Data Assimilation Using Variational Methods
# creating a climatological matrix (very simple)
model = 'lor96'
Bpre,Bcorr = getBsimple(model,Nx)
tune = 1 # depends on the observational frequency
B = tune*Bpre
tileplotB(B)


###########################################################################
# 3a. 3DVar
xb,xa = var3d(x0guess,t,tobs,y,H,B,R,model,Nx)
plotL96DA_var(t,xt,Nx,tobs,y,observed_vars,xb,xa)
rmse_step=1
rmseb = rmse_spread(xt,xb,None,rmse_step)
rmsea = rmse_spread(xt,xa,None,rmse_step)
plotRMSP(t,rmseb,rmsea)


###########################################################################
# 3b. 4DVar
anawin = 4
xb,xa = var4d(x0guess,t,tobs,anawin,y,H,B,R,model,Nx)
plotL96DA_var(t,xt,Nx,tobs,y,observed_vars,xb,xa)
rmse_step=1
rmseb = rmse_spread(xt,xb,None,rmse_step)
rmsea = rmse_spread(xt,xa,None,rmse_step)
plotRMSP(t,rmseb,rmsea)



 
