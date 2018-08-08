# Easier version. 2017. JA
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from L63_model import lorenz63
from L63_misc import gen_obs, rmse_spread, createH, getBsimple
from L63_var import var3d, var4d
from L63_plots import plotL63, plotL63obs, plotL63DA_var, plotRMSP, tileplotB


###############################################################################
### 1.The Nature Run
# Let us perform a 'free' run of the model, which we will consider the truth
# The initial conditions
x0 = [-10,-10,20]
Nx = np.size(x0)
# The final time
tmax = 10
# Computing the nature run
paramtrue = [10.0,8/3.0,28.0]
t,xt = lorenz63(x0,tmax,paramtrue)
plotL63(t,xt)
# A guess to start from in our assimilation experiments
x0guess = [-11,-12,10]


###############################################################################
### 2. The observations
# Decide what variables to observe
obsgrid = 'xyz'
H,observed_vars = createH(obsgrid)
period_obs = 10
var_obs = 1
# Generating the observations
tobs,y,R = gen_obs(t,xt,period_obs,H,var_obs)
plotL63obs(t,xt,tobs,y,observed_vars)

    
#############################################################################
## 3 Data Assimilation Using Variational Methods
# creating a climatological matrix (very simple)
model = 'lor63'
Bpre,Bcorr = getBsimple(model,Nx)
tune = 1 # depends on the observational frequency
B = tune*Bpre
tileplotB(B)


############################################################################
## 3.a. 3DVar
xb,xa = var3d(x0guess,t,tobs,y,H,B,R,model,Nx)
plotL63DA_var(t,xt,tobs,y,observed_vars,xb,xa)
rmse_step=1
rmseb = rmse_spread(xt,xb,None,rmse_step)
rmsea = rmse_spread(xt,xa,None,rmse_step)
plotRMSP(t,rmseb,rmsea)

###########################################################################
# 3.b. 4DVar
anawin=2
xb,xa = var4d(x0guess,t,tobs,anawin,y,H,B,R,model,Nx)
plotL63DA_var(t,xt,tobs,y,observed_vars,xb,xa)
rmse_step=1
rmseb = rmse_spread(xt,xb,None,rmse_step)
rmsea = rmse_spread(xt,xa,None,rmse_step)
plotRMSP(t,rmseb,rmsea)

