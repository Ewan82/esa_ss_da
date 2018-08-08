# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 12:10:17 2017
@author: jamezcua
"""
import numpy as np
import matplotlib.pyplot as plt


##############################################################################
def plotL96(t,xt,N):
 plt.figure().suptitle('Truth')
 for i in range(N):
  plt.subplot(np.ceil(N/4.0),4,i+1)
  plt.plot(t,xt[:,i],'k')
  plt.ylabel('x['+str(i)+']')
  plt.xlabel('time')
  plt.ylim([-20,20])
 del i    
 plt.subplots_adjust(wspace=0.7,hspace=0.3)

 fig = plt.figure()
 fig.suptitle('Truth')
 ax = fig.add_subplot(111, projection='3d')
 jj,tt = np.meshgrid(np.arange(N),t)
 ax.plot_wireframe(jj,tt,xt,rstride=len(t))
 ax.set_xlabel('variable number')
 ax.set_ylabel('time')

 plt.figure().suptitle('Truth')
 C = plt.contourf(np.arange(N),t,xt,10)
 plt.contour(np.arange(N),t,xt,10,colors='k',linestyles='solid')
 plt.xlabel('variable number')
 plt.ylabel('time')
 plt.title('Hovmoller diagram')
 plt.colorbar(C)    
   

##############################################################################
def plotL96obs(t,xt,Nx,tobs,y,observed_vars):
 plt.figure().suptitle('Truth and Observations')
 for i in range(Nx):
  plt.subplot(np.ceil(Nx/4.0),4,i+1)
  plt.plot(t,xt[:,i],'k')
  plt.hold(True)
  if i in observed_vars:
   plt.scatter(tobs,y[:,observed_vars.index(i)],20,'r')
  plt.ylabel('x['+str(i)+']')
  plt.xlabel('time')
  plt.ylim([-20,20])
  plt.xlim([0,t[-1]])
 del i    
 plt.subplots_adjust(wspace=0.7,hspace=0.3)

        
#############################################################################        
def plotL96DA_kf(t,xt,tobs,y,Nx,observed_vars,Xb,xb,Xa,xa):
 plt.figure().suptitle('Truth, Observations, Background Ensemble, and Analysis Ensemble')
 for i in range(Nx):
  plt.subplot(np.ceil(Nx/4.0),4,i+1)
  plt.plot(t,xt[:,i],'k')
  plt.hold(True)
  plt.plot(t,Xb[:,i,:],'--b')
  plt.plot(t,Xa[:,i,:],'--m')
  if i in observed_vars:
   plt.autoscale(False) # prevent scatter() from rescaling axes
   plt.scatter(tobs,y[:,observed_vars.index(i)],20,'r')
  plt.ylabel('x['+str(i)+']')
  plt.xlabel('time')
  plt.grid(True)
 del i    
 plt.subplots_adjust(wspace=0.7,hspace=0.3)

 plt.figure().suptitle('Truth, Observations, and Analysis Mean')
 for i in range(Nx):
  plt.subplot(np.ceil(Nx/4.0),4,i+1)
  plt.plot(t,xt[:,i],'k')
  plt.hold(True)
  if i in observed_vars:
   plt.autoscale(False) # prevent scatter() from rescaling axes
   plt.scatter(tobs,y[:,observed_vars.index(i)],20,'r')
  plt.plot(t,xa[:,i],'m')
  plt.ylabel('x['+str(i)+']')
  plt.xlabel('time')
  plt.grid(True)
 del i    
 plt.subplots_adjust(wspace=0.7,hspace=0.3)


#############################################################################
def plotL96DA_var(t,xt,N,tobs,y,observed_vars,xb,xa):
 plt.figure().suptitle('Truth, Observations, and Analysis')
 for i in range(N):
  plt.subplot(np.ceil(N/4.0),4,i+1)
  plt.plot(t,xt[:,i],'k')
  plt.hold(True)
  if i in observed_vars:
   plt.autoscale(False) # prevent scatter() from rescaling axes
   plt.scatter(tobs,y[:,observed_vars.index(i)],20,'r')
  plt.plot(t,xa[:,i],'m')
  plt.ylabel('x['+str(i)+']')
  plt.xlabel('time')
  plt.grid(True)
 del i    
 plt.subplots_adjust(wspace=0.7,hspace=0.3)


#####################################################
def plotL96DA_pf(t,xt,Nx,tobs,y,observed_vars,xpf,x_m):
 plt.figure().suptitle('Truth, Observations and Ensemble')
 for i in range(Nx):
  plt.subplot(np.ceil(Nx/4.0),4,i+1)
  plt.hold(True)
  plt.plot(t,xpf[:,i,:],'--m')
  plt.plot(t,xt[:,i],'-k',linewidth=2.0)
  plt.plot(t,x_m[:,i],'-m',linewidth=2)
  if i in observed_vars:
   plt.autoscale(False) # prevent scatter() from rescaling axes
   plt.scatter(tobs,y[:,observed_vars.index(i)],20,'r')
  plt.ylabel('x['+str(i)+']')
  plt.xlabel('time')
  plt.grid(True)
 del i 
 plt.subplots_adjust(wspace=0.7,hspace=0.3)


############################################################################
def plotRMSP(t,rmseb=None,rmsea=None,spreadb=None,spreada=None):
 plt.figure()
 plt.subplot(2,1,1)
 if rmseb is not None:
  plt.plot(t,rmseb,'b',label='bgd')
  plt.hold(True)
 plt.plot(t,rmsea,'m',label='ana')
 plt.legend()
 plt.title('root mean squared error')
 plt.xlabel('time')
 plt.grid(True)

 if spreadb is not None:
  plt.subplot(2,2,3)
  if rmseb is not None:
   plt.plot(t,rmseb,'b',label='RMSE')
   plt.hold(True)
  plt.plot(t,spreadb,'--k',label='spread')
  plt.legend()
  plt.title('background')
  plt.xlabel('time')
  plt.grid(True)

 if spreada is not None:
  plt.subplot(2,2,4)
  plt.plot(t,rmsea,'m',label='RMSE')
  plt.hold(True)
  plt.plot(t,spreada,'--k',label='spread')
  plt.legend()
  plt.title('analysis')
  plt.xlabel('time')
  plt.grid(True)

 plt.subplots_adjust(hspace=0.25)


#############################################################################
def tileplotB(mat):
    N1,N2 = np.shape(mat)
    plt.figure()
    plt.pcolor(mat.A.T,edgecolors='k')
    ymin,ymax = plt.ylim()
    plt.ylim(ymax,ymin)
    plt.clim(-3,3)
    plt.colorbar()
    plt.title('B')
    plt.xlabel('variable number')
    plt.ylabel('variable number')
    plt.xticks(np.arange(0.5,N1+0.5),np.arange(N1))
    plt.yticks(np.arange(0.5,N2+0.5),np.arange(N2))


#############################################################################
def plotpar(Nparam,tobs,paramt_time,Parama,parama):
 plt.figure().suptitle('True Parameters and Estimated Parameters')
 for i in range(Nparam):
  plt.subplot(Nparam,1,i+1)
  plt.plot(tobs,paramt_time[:,i],'k')
  plt.hold(True)
  plt.plot(tobs,Parama[:,i,:],'--m')
  plt.plot(tobs,parama[:,i],'-m',linewidth=2)
  plt.ylabel('parameter['+str(i)+']')
  plt.xlabel('time')
  plt.grid(True)
 del i 
 plt.subplots_adjust(hspace=0.3)
        

#############################################################################
def plotRH(M,tobs,xt,xpf,rank):
 nbins = M+1
 plt.figure().suptitle('Rank histogram')
 for i in range(3):
  plt.subplot(1,3,i+1)
  plt.hist(rank[:,i],bins=nbins)
  plt.xlabel('x['+str(i)+']')
  plt.axis('tight')
 plt.subplots_adjust(hspace=0.3)
        
        

        