import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats

def Salm(nchain, init, propsd, X, Y):
  chain=np.zeros((nchain+1, 4))
  alpha=init[0]
  beta=init[1]
  gamma=init[2]
  lambd=init[3]
  tau=init[4]
  chain[0]=[alpha, beta, gamma, tau]
  for k in range (nchain):

  #alpha
    logmu=np.zeros((3,6))#on définit mu pour simplifier le code
    for i in range(3):
      for j in range(6):
        logmu[i][j]=alpha+beta*np.log(X[j]+10)+gamma*X[j]+lambd[i][j]
    alphaprop=alpha+propsd[0]*sp.stats.norm.rvs()#on fait une proposition pour alpha avec une marche gaussienne
    logmuprop=np.zeros((3,6)) #on définit mu de proposition pour simplifier le code   
    for i in range(3):
      for j in range(6):
        logmuprop[i][j]=alphaprop+beta*np.log(X[j]+10)+gamma*X[j]+lambd[i][j]
    #on définit top avec la proposition
    top=(Y*logmuprop-np.exp(logmuprop)).sum()
    #on définit bottom
    bottom=(Y*logmu-np.exp(logmu)).sum()
    if np.exp(top-bottom)>np.random.uniform():#on regarde la probabilité d'acceptation
      alpha=alphaprop
    
    #beta
