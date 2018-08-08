# Easier version. 2017. JA
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from L63_model import lorenz63
from L63_misc import gen_obs, rmse_spread, createH
from L63_kfs import kfs_lor63, kfs_lor63_pe
from L63_plots import plotL63, plotL63obs, plotL63DA_kf, \
                       plotRMSP, tileplotB, plotpar


###############################################################################
### 1 The Nature Run
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
period_obs = 20
var_obs = 2
# Generating the observations
tobs,y,R = gen_obs(t,xt,period_obs,H,var_obs)
plotL63obs(t,xt,tobs,y,observed_vars)

    
##############################################################################
### 3. Data assimilation using KFs    
rho=0.10
M = 3
met = 'ETKF'
Xb,xb,Xa,xa = kfs_lor63(x0guess,t,tobs,y,H,R,rho,M,met)
plotL63DA_kf(t,xt,tobs,y,observed_vars,Xb,xb,Xa,xa)

rmse_step=1
rmseb,spreadb = rmse_spread(xt,xb,Xb,rmse_step)
rmsea,spreada = rmse_spread(xt,xa,Xa,rmse_step)
plotRMSP(t,rmseb,rmsea,spreadb,spreada)


"""
##############################################################################
## 4 Parameter estimation using KF
parambad = [6.0,3.0,25.0]; Nparam = np.size(parambad)
t,xbad = lorenz63(x0,tmax,parambad)
plotL63(t,xbad)

alpha=0.5
Xb,xb,Xa,xa,Para,para = kfs_lor63_pe(x0guess,parambad,t,tobs,y,H,R,rho,alpha,M,met)
plotL63DA_kf(t,xt,tobs,y,observed_vars,Xb,xb,Xa,xa)
paramt_time = np.ones((len(tobs),1))*paramtrue
plotpar(Nparam,tobs,paramt_time,Para,para)

rmse_step=1
rmseb,spreadb = rmse_spread(xt,xb,Xb,rmse_step)
rmsea,spreada = rmse_spread(xt,xa,Xa,rmse_step)
plotRMSP(t,rmseb,rmsea,spreadb,spreada)

"""


