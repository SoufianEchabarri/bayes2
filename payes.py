import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats

def Salm(nchain, init, propsd, X, Y):
  chain=np.zeros((nchain+1, 4))
  lambdchain=np.zeros((nchain, 18))
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
    logmu=np.zeros((3,6))#on définit mu pour simplifier le code
    for i in range(3):
      for j in range(6):
        logmu[i][j]=alpha+beta*np.log(X[j]+10)+gamma*X[j]+lambd[i][j] #OK
    betaprop=beta+propsd[1]*sp.stats.norm.rvs()#on fait une proposition pour alpha avec une marche gaussienne
    logmuprop=np.zeros((3,6)) #on définit mu de proposition pour simplifier le code
    for i in range(3):
      for j in range(6):
        logmuprop[i][j]=alpha+betaprop*np.log(X[j]+10)+gamma*X[j]+lambd[i][j]
    #on définit top avec la proposition
    top=(Y*logmuprop-np.exp(logmuprop)).sum()
    #on définit bottom
    bottom=(Y*logmu-np.exp(logmu)).sum()
    if np.exp(top-bottom)>np.random.uniform():#on regarde la probabilité d'acceptation
      beta=betaprop

    #gamma
    logmu=np.zeros((3,6))#on définit mu pour simplifier le code
    for i in range(3):
      for j in range(6):
        logmu[i][j]=alpha+beta*np.log(X[j]+10)+gamma*X[j]+lambd[i][j]
    gammaprop=gamma+propsd[2]*sp.stats.norm.rvs()#on fait une proposition pour alpha avec une marche gaussienne
    logmuprop=np.zeros((3,6)) #on définit mu de proposition pour simplifier le code
    for i in range(3):
      for j in range(6):
        logmuprop[i][j]=alpha+beta*np.log(X[j]+10)+gammaprop*X[j]+lambd[i][j]
    #on définit top avec la proposition
    top=(Y*logmuprop-np.exp(logmuprop)).sum()
    #on définit bottom
    bottom=(Y*logmu-np.exp(logmu)).sum()
    if np.exp(top-bottom)>np.random.uniform():#on regarde la probabilité d'acceptation
      gamma=gammaprop
    
    #tau, avec une loi conjuguée
    param1=0.01
    param2=0.01#on prend pour tau une loi a priori gamma(param1,param2), on a donc une loi conjuguée pour lambda
    param1conjug=param1+3*6/2
    param2conjug=(param2+(lambd**2).sum()/2)
    tau=sp.stats.gamma.rvs(param1conjug, param2conjug)
    for i in range(3):
      lambd[i]=sp.stats.norm.rvs(6)/tau**0.5

    chain[k+1]=[alpha, beta, gamma, tau]
    c=0
    for i in range(3):
      for j in range(6):
        lambdchain[k][c]=lambd[i][j]
        c+=1
  return(chain, lambdchain)

propsd=[5,0.7,0.007,1]
X=np.array([0, 10, 33, 100, 333, 1000])
Y=np.array([[15, 16, 16, 27, 33, 20],
  [21, 18, 26, 41, 38, 27],
  [29, 21, 33, 69, 41, 42]])
init=np.array([2, 0, 0, np.zeros((3, 6)), 0])
print(X.shape)

[res, lambdchain]=Salm(10000, init, propsd, X, Y)


print(res[:100,0])
plt.plot(res[:,2])

fig, axs=plt.subplots(2,2,figsize=(10,10))

axs[0,0].plot(np.arange(9901),res[100:,0])
axs[0,0].set_title("alpha")
axs[0,1].plot(np.arange(9901),res[100:,1])
axs[0,1].set_title("beta")
axs[1,0].plot(np.arange(9901),res[100:,2])
axs[1,0].set_title("gamma")
axs[1,1].plot(np.arange(9901),res[100:,3])
axs[1,1].set_title("tau")

alpha=np.mean(res[100:,0])
beta=np.mean(res[100:,1])
gamma=np.mean(res[100:,2])
tau=np.mean(res[100:,3])
lambd=np.array([[np.mean(lambdchain[100:,0]), np.mean(lambdchain[100:,1]), np.mean(lambdchain[100:,2]), np.mean(lambdchain[100:,3]), np.mean(lambdchain[100:,4]), np.mean(lambdchain[100:,5])],
       [np.mean(lambdchain[100:,6]), np.mean(lambdchain[100:,7]), np.mean(lambdchain[100:,8]), np.mean(lambdchain[100:,9]), np.mean(lambdchain[100:,10]), np.mean(lambdchain[100:,11])],
       [np.mean(lambdchain[100:,12]), np.mean(lambdchain[100:,13]), np.mean(lambdchain[100:,14]), np.mean(lambdchain[100:,15]), np.mean(lambdchain[100:,16]), np.mean(lambdchain[100:,17])]])
print([alpha, beta, gamma, tau])

def xtest(n):
  res=np.zeros((n, 6))
  for i in range(n):
    res[i]=np.random.randint(0, 1000, size=6)
  return(res)

def pred(x, alpha, beta, gamma, tau, lambd):
  mu=np.exp(alpha+beta*np.log(x+10)+gamma*x+lambd)
  Y=sp.stats.poisson.rvs(mu)
  return(Y)

c=pred(xtest(1), alpha, beta, gamma, tau, lambd)
for i in range(9):
  c+=pred(xtest(1), alpha, beta, gamma, tau, lambd)
print(c/10)

