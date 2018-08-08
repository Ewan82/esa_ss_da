# 2017 JA
import numpy as np
from scipy.linalg import pinv, sqrtm
from L96_model import lorenz96

def kfs_lor96(x0_t,t,tobs,y,H,R,rho,M,met,lam,loctype):
 """Data assimilation for Lorenz 1996 using Ensemble Kalman Filters.
 Inputs:  - x0_t, the real initial position
          - t, time array of the model (should be evenly spaced)
          - tobs, time array of the observations (should be evenly spaced 
            with a timestep that is a multiple of the model timestep)
          - y, the observations
          - H, observation matrix
          - R, the observational error covariance matrix
          - rho, inflation for P.  Notice we multiply (1+rho)*Xpert
            or P*(1+rho)^2.
          - M, the ensemble size
          - met, a string containing the method: 'SEnKF', 'ETKF'
          - lam, the localization radius in gridpoint units.  If None,
            it means no localization.
          - loctype, a string indicating the type of localization: 'GC'
            to use the Gaspari-Cohn function, 'cutoff' for a sharp cutoff

 Outputs: - Xb, the background ensemble 3D array [time,vars,members]
          - xb, background mean
          - Xa, the analysis ensemble 3D array [time,vars,members]
          - xa, analysis mean
          - locmatrix, localization matrix (or None if lam is None)"""

 # General settings
 # Number of observations and variables
 Nsteps = np.size(t)
 L,N = np.shape(H)
 # For the true time
 tstep_truth = t[1]-t[0]
 # For the analysis (we assimilate everytime we get observations)
 tstep_obs = tobs[1]-tobs[0]
 # The ratio
 o2t = int(tstep_obs/tstep_truth+0.5)
 # Precreate the arrays for background and analysis
 Xb = np.empty((Nsteps,N,M)); Xb.fill(np.nan)
 Xa = np.empty((Nsteps,N,M)); Xa.fill(np.nan)

 # For the original background ensemble
 # Two options: fixed and random
 back0 = 'fixed'
 #back0 = 'random'
 desv = 1.0

 # Fixed initial conditions for our ensemble (created ad hoc)
 if back0=='fixed':
  for j in range(N):
   Xb[0,j,:] = np.linspace(x0_t[j]-np.sqrt(desv), x0_t[j]+np.sqrt(desv), M)
  del j  
 # Random initial conditions for our ensemble
 elif back0=='random':
  for j in range(M):
   Xb[0,:,j] = x0_t + np.sqrt(desv)*np.random.randn(N)
  del j
  
 # Since we don't have obs at t=0 the first analysis is the same as
 # background
 Xa[0,:,:] = Xb[0,:,:]

 # Getting the R-localization weights
 if np.any(lam) != None:
  locmatrix = getlocmat(N,L,H,lam,loctype)
 else:
  locmatrix = None

 # The following cycle contains evolution and assimilation for all time steps
 for j in range(len(tobs)-1):
  # Evolve from analysis!
  xold = Xa[j*o2t,:,:] # [N,M]
  # Time goes forward
  xnew = evolvemembers(xold,tstep_truth,o2t) # needs [N,M] arrays,
  # The new background
  Xb[j*o2t+1:(j+1)*o2t+1,:,:] = xnew[1:,:,:] # [o2t,N,M]
  Xa[j*o2t+1:(j+1)*o2t+1,:,:] = xnew[1:,:,:] # [o2t,N,M]
  # The assimilation
  Xa_aux = enkfs(Xb[(j+1)*o2t,:,:],y[j+1,:],H,R,rho,met,lam,locmatrix)
  Xa[(j+1)*o2t,:,:] = Xa_aux # introduce the auxiliary variable
  print 't=',t[j*o2t]
 del j
 # The background and analysis mean
 x_b = np.mean(Xb,axis=2) # [t,N,M] -> [t,N]
 x_a = np.mean(Xa,axis=2) # [t,N,M] -> [t,N]

 return Xb, x_b, Xa, x_a, locmatrix
 

############################################################################
def evolvemembers(xold,tstep_truth,o2t):
 """Evolving the members.
 Inputs:  - xold, a [N,M] array of initial conditions for the
            M members and N variables
          - tstep_truth, the time step used in the nature run
          - o2t, frequency of observations in time steps
 Outputs: - xnew, a [o2t+1,N,M] array with the evolved members"""

 t_anal = o2t*tstep_truth
 N,M = np.shape(xold)
 xnew = np.empty((o2t+1,N,M)); xnew.fill(np.nan)
 
 for j in range(M):
  taux,xaux = lorenz96(t_anal,xold[:,j],N) # [o2t+1,N]
  xnew[:,:,j] = xaux
 del j
 
 return xnew


##############################################################################
## The EnKF algorithms
def enkfs(Xb,y,H,R,rho,met,lam,locmatrix):
 """Performs the analysis using different EnKF methods.
 Inputs: - Xb, the ensemble background [N,M]
         - y, the observations [L]
         - H, the observation matrix [L,N]
         - R, the obs error covariance matrix [L,L]
         - rho, inflation for P.  Notice we multiply (1+rho)*Xpert
           or P*(1+rho)^2.
         - met, a string that indicated what method to use
         - lam, the localization radius
         - locmatrix, localization matrix
 Output: - Xa, the full analysis ensemble [N,M]"""

 # General settings
 # The background information
 Xb = np.mat(Xb) # array -> matrix
 y = np.mat(y).T # array -> column vector
 sqR = np.real_if_close(sqrtm(R))
 
 # Number of state variables, ensemble members and observations
 N,M = np.shape(Xb)
 L,N = np.shape(H)

 # Auxiliary matrices that will ease the computation of averages and
 # covariances
 U = np.mat(np.ones((M,M))/M)
 I = np.mat(np.eye(M))

 # The ensemble is inflated (rho can be zero)
 Xb_pert = (1+rho)*Xb*(I-U)
 Xb = Xb_pert + Xb*U

 # Create the ensemble in Y-space
 Yb = np.mat(np.empty((L,M))); Yb.fill(np.nan)

 # Map every ensemble member into observation space
 for jm in range(M):
  Yb[:,jm] = H*Xb[:,jm]
 del jm
        
 # The matrix of perturbations
 Xb_pert = Xb*(I-U)
 Yb_pert = Yb*(I-U)

 # Now, we choose from one of three methods
 # Stochastic Ensemble Kalman Filter
 if met=='SEnKF':
  if np.any(locmatrix) == None:
   # The Kalman gain matrix without localization
   Khat = 1.0/(M-1)*Xb_pert*Yb_pert.T * pinv(1.0/(M-1)*Yb_pert*Yb_pert.T+R)
  else:
   # The Kalman gain with localization
   Caux = np.mat(locmatrix.A * (Xb_pert*Yb_pert.T).A)
   Khat = 1.0/(M-1)*Caux * pinv(1.0/(M-1)*H*Caux+R)

  # Fill Xa (the analysis matrix) member by member using perturbed observations
  Xa = np.mat(np.empty((N,M))); Xa.fill(np.nan)
  for jm in range(M):
   yaux = y + sqR*np.mat(np.random.randn(L,1))
   Xa[:,jm] = Xb[:,jm] + Khat*(yaux-Yb[:,jm])
  del jm
        
 # Ensemble Transform Kalman Filter
 elif met=='ETKF':
  # Means
  xb_bar = Xb*np.ones((M,1))/M
  yb_bar = Yb*np.ones((M,1))/M
 
  if np.any(locmatrix) == None:
   # The method without localization (ETKF)
   Pa_ens = pinv((M-1)*np.eye(M)+Yb_pert.T*pinv(R)*Yb_pert)
   Wa = sqrtm((M-1)*Pa_ens) # matrix square root (symmetric)
   Wa = np.real_if_close(Wa)
   wa = Pa_ens*Yb_pert.T*pinv(R)*(y-yb_bar)
   Xa_pert = Xb_pert*Wa
   xa_bar = xb_bar + Xb_pert*wa
   Xa = Xa_pert + xa_bar*np.ones((1,M))
  else:   
   Xa = letkf(Xb_pert,xb_bar,Yb_pert,yb_bar,y,H,lam,locmatrix,R)
  
 return Xa


##############################################################################
## Localization functions
def getlocmat(N,L,H,lam,loctype):
    #To get the localization weights.

    indx = np.mat(range(N)).T
    indy = H*indx
    dist = np.mat(np.empty((N,L)))
    dist.fill(np.nan)

    # First obtain a matrix that indicates the distance (in grid points)
    # between state variables and observations
    for jrow in range(N):
        for jcol in range(L):
            dist[jrow,jcol] = np.amin([abs(indx[jrow]-indy[jcol]),\
                                       N-abs(indx[jrow]-indy[jcol])])

    # Now we create the localization matrix
    # If we want a sharp cuttof
    if loctype=='cutoff':
        locmatrix = 1.0*(dist<=lam)
    # If we want something smooth, we use the Gaspari-Cohn function
    elif loctype=='GC':
        locmatrix = np.empty_like(dist)
        locmatrix.fill(np.nan)
        for j in range(L):
            locmatrix[:,j] = gasparicohn(dist[:,j],lam)

    return locmatrix


def gasparicohn(z,lam):
    "The Gaspari-Cohn function."
    c = lam/np.sqrt(3.0/10)
    zn = abs(z)/c
    C0 = np.zeros_like(zn)
    for j in range(len(C0)):
        if zn[j]<=1:
            C0[j] = - 1.0/4*zn[j]**5 + 1.0/2*zn[j]**4 \
                    + 5.0/8*zn[j]**3 - 5.0/3*zn[j]**2 + 1
        if zn[j]>1 and zn[j]<=2:
            C0[j] = 1.0/12*zn[j]**5 - 1.0/2*zn[j]**4 \
                    + 5.0/8*zn[j]**3 + 5.0/3*zn[j]**2 \
                    - 5*zn[j] + 4 - 2.0/3*zn[j]**(-1)
    return C0
    
    
###############################################################################    
    
    
#    def etkf_main(Xt,opcini,t,x,M,Nx_obs,H,R,y,obsnum,period_obs,adap,rho0,lam,\
#              locmatrix,gridobs,B0,B0sq,noiseswitch,Qsq,smooth,loc_nobs):
# Nx = np.size(x)
# dt = t[1]-t[0]
# Nsteps = np.size(t)
#    
# # Precreate arrays for background and analysis
# Xb = np.empty(shape=(Nsteps,Nx,M))
# Xa = np.empty(shape=(Nsteps,Nx,M))
#
# rhoa = np.empty(shape=(obsnum+1,Nx))
# rhoa[0,:] = rho0 
#
#    # Generate initial conditions  
# for j in range(M):
#  if opcini==0:
#   Xb[0,:,j] = Xt[0,:] + np.dot(B0sq,(np.random.randn(Nx)).T) 
#  if opcini==1:
#   Xb[0,:,j] = Xt[0,:] + np.dot(B0sq,(np.random.randn(Nx)).T) 
#  if opcini==2:
#   Xb[0,:,j] = Xt[:,j]  
# del j  
#    
# Xa[0,:,:] = Xb[0,:,:] # Background = Analysis at t=0 as no observations        
#     
# ## Evolution and Assimilation cycles
# #First step is to evolve from the analysis from the prior window
# taux = dt*np.arange(0.0,period_obs+1,1)        
#    
# for i in range(obsnum):   
#  #print ("cycle", i)
#  xold = Xa[i*period_obs,:,:]        
#  xb_aux = np.empty((np.size(taux),Nx,M))
#  
#  ## First Guess
#  for m in range(M):
#   xevol,seed = l96num(x,taux,xold[:,m],noiseswitch,Qsq)
#   xb_aux[:,:,m] = xevol
#   Xb[period_obs*i+1:period_obs*(i+1)+1,:,m] = xevol[1:,:]  
#   del xevol
#  del m          
#  # The new background
#  rho_old = rhoa[i,:]
#  xa_aux,rho_new = etkf(xb_aux,period_obs,M,R,Nx,Nx_obs,H,y[i,:],adap,lam,\
#                    locmatrix,rho_old,gridobs,smooth,loc_nobs)
#  
#  Xa[period_obs*i:period_obs*(i+1)+1,:,:] = xa_aux
#  rhoa[i+1,:] = rho_new.T
# del i
#       
# # The background and analysis mean
# x_b = np.mean(Xb,axis=2) #
# x_a = np.mean(Xa,axis=2) #
#
# return Xb,x_b,Xa,x_a
#
#
#### ----------------------------------------------------------------------------------------------------------------------------------
#def etkf(Xbtraj,period_obs,M,R,Nx,Ny,H,y,adap,lam,locmatrix,rhob,gridobs,\
#         smooth,loc_nobs):
# # nx and ny are number of gridpoints in state and obs space respectively    
# Xb = Xbtraj[-1,:,:]    
# Yb = np.empty(shape=(Ny,M))
# Xatraj = np.empty((period_obs+1,Nx,M))
#
# for m in range(M):
#  Yb[:,m]=np.dot(H,Xb[:,m])
# del m
#
# U = np.mat(np.ones((M,M))/M)
# I = np.mat(np.eye(M))
#
# # Means and perts
# xb_bar = np.mean(Xb,axis=1) 
# xb_bar = np.reshape(xb_bar,(Nx,1))
# Xb_pert = np.dot(Xb,(I-U))
# yb_bar = np.mean(Yb,axis=1) 
# Yb_pert = np.dot(Yb,(I-U))
#
# rhoa = np.empty((Nx,1))
# mineig = np.empty((Nx,M))
#    
# indX = np.arange(Nx)
# influence_dist = infdislist(lam)
# rhob = np.reshape(rhob,(Nx,1))
#   
# for jgrid in range(Nx):
#  mineig_aux, rhoag_aux, Xtrajag_aux = etkfpergp(Xbtraj[:,jgrid,:],period_obs,\
#   jgrid,Nx,Ny,y,adap,lam,influence_dist,indX,H,R,Xb[jgrid,:],xb_bar[jgrid],\
#   Xb_pert[jgrid,:],Yb,yb_bar,Yb_pert,locmatrix,rhob[jgrid,0],M,smooth,loc_nobs)
#  mineig[jgrid,:] = mineig_aux    
#  rhoa[jgrid] = rhoag_aux
#  Xatraj[:,jgrid,:] = np.real(Xtrajag_aux)
# del jgrid
# 
# #print mineig
# return Xatraj, rhoa
#
#
#def etkfpergp(Xbtraj,period_obs,jgrid,Nx,Ny,y,adap,lam,influence_dist,indX,H,\
#    R,Xb,xb_bar,Xb_pert,Yb,yb_bar,Yb_pert,locmatrix,rhob,M,smooth,loc_nobs):
# # select the obs  
# if lam == []:
#  useobsX = indX        
# else:     
#  lim1 = mod2(jgrid-influence_dist,Nx) 
#  lim2 = mod2(jgrid+influence_dist,Nx)
#  if lim1==lim2:  
#   useobsX = indX
#  if lim1>lim2:
#   useobsX = np.append([np.arange(lim1,Nx)],[np.arange(0,lim2)]).T
#  if lim1<lim2:   
#   useobsX = np.arange(lim1,lim2,1).T
#  useobsX.sort()
#
# NuseobsX = len(useobsX)           
# H_aux = np.zeros(shape=(Ny,NuseobsX))
#    
# for j in range(NuseobsX):
#  H_aux[:,j] = H[:,useobsX[j]]
# del j
# 
# indobs_pre = np.dot(H_aux,useobsX+1)
# indobs_pre = np.reshape(indobs_pre,(Ny,1))
# indobs = np.where(indobs_pre!=0)
# indobs = indobs[0]
# H_aux = H_aux[indobs,:]
#
# NuseobsY = len(indobs)
# Xatraj = Xbtraj # by default
# 
# # if localization killed all observations, nothing can be done
# if NuseobsY==0: 
#  rhoag = rhob
#  eigmin = 1
#  Xatraj = Xbtraj 
# 
# # if there are some observations left, then we can assimilate
# if NuseobsY!=0:
#  # Trim loc and R
#  locmatrix_aux = locmatrix[jgrid,indobs]
#  invR_aux = R[indobs,indobs]**(-1)
#  loc_invR = locmatrix_aux * invR_aux       
#  loc_invR = np.diag(loc_invR)
#  
#  # trim Yb_pert
#  Yb_pert_aux = Yb_pert[indobs,:]
#  yb_bar_aux = yb_bar[indobs]
#  # trim y
#  y_aux = y[indobs]
#  d_aux = y_aux - yb_bar_aux
#           
#  # adaptive inflation
#  if adap == 0:  
#   rhoa_aux = rhob
#  elif adap == 1:
#   loc_tr = np.sum(locmatrix_aux)          
#   vb = 0.05**2.0 # This is something prescribed and tuned!
#   den = np.trace(np.dot(Yb_pert_aux,Yb_pert_aux.T)/(M-1.0)* loc_invR)
#   alphab = (1+rhob)**2
#   alphao = (np.trace( np.dot(d_aux,d_aux.T)* np.mat(loc_invR)) - loc_tr)/den
#   vo = 2/loc_tr*((alphab*den + loc_tr)/den)**2
#   alphaa = (alphab*vo + alphao*vb)/(vo+vb)
#   rhoa_aux = np.sqrt(alphaa)-1
#   if rhoa_aux<0.01 or rhoa_aux>0.4: # patch
#    rhoa_aux = rhob
#  
#  if np.isfinite(rhoa_aux):
#   rhoag = rhoa_aux
#  else:
#   rhoag = rhob
#  
#  # the actual ETKF for each gridpoint
#  Yb_pert_aux = (1.0 + rhoa_aux) * Yb_pert_aux
#  beta = np.dot(loc_invR, Yb_pert_aux)
#  beta = np.dot(Yb_pert_aux.T, beta)/(M-1)
#
#  if np.isfinite(beta).all():
#   Gamma,C = np.linalg.eig(beta)
#   Gamma = np.real(Gamma)
#  else: 
#   Gamma = np.ones((M))
#   C = np.eye(M)
#  
#  # to ensure the neighbouring grdipoints have the same order  
#  ind = np.flipud(np.argsort(Gamma))
#  Gamma = Gamma[ind]
#  C = C[:,ind]
#  del ind
#
#  Gamma_new = np.zeros((M))
#  ind = Gamma >= 10**(-4)
#  Gamma_new[ind] = Gamma[ind]
#  Gamma = Gamma_new; del Gamma_new
#
#  Gamma_moh = (1 + Gamma)**(-1.0/2.0)
#  eigmin = np.min(Gamma_moh)
#       
#  Wag = np.dot(C,np.diag(Gamma_moh))
#  Wag = np.real(np.dot(Wag,C.T))
#            
#  aux = np.dot(loc_invR,d_aux)
#  aux = np.dot(Yb_pert_aux.T,aux)
#  aux = np.dot(Wag.T,aux.T)        
#  wag = 1.0/(M-1) * np.dot(Wag,aux)
#  del aux  
#                  
#  Xa_pert_j = (1+rhoa_aux)*np.dot(Xb_pert,Wag)        
#  xa_bar_j = xb_bar + (1.0 + rhoa_aux) * np.dot(Xb_pert, wag)        
#  Xag = np.zeros(shape=(1,M))        
#        
#  for m in range(M):
#   Xag_aux = Xa_pert_j[:,m] + xa_bar_j
#   Xag[:,m] =  np.squeeze(Xag_aux)
#  del m
#
#  # by default
#  toljump = gettoljump(period_obs)
#
#  if np.isfinite(Xag).all() and np.max(np.abs(Xag-Xb))<toljump:   
#    smooth_posttest = smooth
#    Xag = Xag
#  else:
#   smooth_posttest = 0
#   Xag = Xb
#
#  if smooth_posttest==0:
#   Xatraj[-1,:] = Xag        
#  if smooth_posttest==1:
#   for j in range(period_obs):
#    Xatraj[j,:] = Smoother(Xbtraj[j,:],M,Wag,wag) 
#   del j
#   Xatraj[-1,:] = Xag            
#   
# return eigmin, rhoag, Xatraj

    
    
    