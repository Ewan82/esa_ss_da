# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 22:18:56 2017
@author: jamezcua
"""

import numpy as np

##################################
def bayes_gm(x,alphas_prior,mus_prior,sigmas_prior,y,alphas_like,mus_like,sigmas_like):   
 N_x = np.size(x)   

 # the prior
 N_prior = np.size(alphas_prior)
 p_prior = np.empty((N_x,N_prior)); p_prior.fill(np.NaN)
 for n in range(N_prior):
  p_prior[:,n] = alphas_prior[n]*Gaussian(x,mus_prior[n],sigmas_prior[n])
 del n     
 p_prior = np.sum(p_prior,axis=1)

 # the likelihood
 N_like = np.size(alphas_like)
 p_like = np.empty((N_x,N_like)); p_like.fill(np.NaN)
 for n in range(N_like): #note: kernel is y-(x+mu)
  p_like[:,n] = alphas_like[n]*Gaussian(x+mus_like[n],y,sigmas_like[n])
 del n     
 p_like = np.sum(p_like,axis=1)

 # the marginal is a scalar, since y is a scalar
 p_ymarg = np.empty((N_prior,N_like)); p_ymarg.fill(np.NaN)
 for npr in range(N_prior):
  for nli in range(N_like):
   p_ymarg[npr,nli] = alphas_prior[npr]*alphas_like[nli]*\
                      Gaussian(y,mus_prior[npr]+mus_like[nli],\
                      np.sqrt(sigmas_prior[npr]**2+sigmas_like[nli]**2))
  del nli
 del npr    
 p_ymarg = np.sum(np.sum(p_ymarg,axis=1))
 # the posterior is the product
 p_post = p_prior * p_like /p_ymarg

 return p_prior, p_like, p_post


###############################################
def Gaussian(x,mu,sigma):
 coeff = (np.sqrt(2*np.pi)*sigma)**(-1)
 expo =  1.0/2.0* (x-mu)**2 /sigma**2
 p = coeff*np.exp(-expo) 
 return p



