# Financial Econometrics - Take Home Assignment


from scipy.special import gamma
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats 
import seaborn as sns 
sns.set(color_codes=True)
from pandas import read_csv
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.optimize import minimize
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from arch import arch_model
from statsmodels.tsa.arima_model import ARIMA#, ARMAResults
import pandas as pd





# Reading Data
csv_data = read_csv('Data9.csv')

rt = np.array(csv_data.log_return)

rt_insample = rt[:252]
rt_outsample = rt[252:]

index0 = rt_insample==0 # replacing 0 values by smaller value
rt_insample[index0]=0.0001/100

rt_insample_square = rt[:252]**2


def dstats(x):
    T = np.size(x,0)
    mu = np.sum(x,0)/T # column wise
    
    sigma2 = np.sum((x-mu)**2,axis=0)/T # axis 0 matlab column wise
    sigma = np.sqrt(sigma2)
    skew = np.sum((x-mu)**3/(sigma**3),axis=0)/T
    kurt = np.sum((x-mu)**4/(sigma**4),axis=0)/T
    d = np.array([mu, sigma, skew, kurt])
    return d

ds = dstats(rt_insample)

# EGARCH(1,1)      N(0,1) distributed
####################################


def ll_Egarch11(par0,x, out=None):
    T = np.size(x,axis = 0)
    omega = par0[0]
    alpha = par0[1]
    beta = par0[2]
    Gaama = par0[3]
    
    s2 =np.ones((T+1,1))*omega/(1-alpha-beta)
    x = np.append(0,x).reshape(T+1,1)
    
    for t in range(1,T+1):
        s2[t] =    np.exp( omega + (alpha * x[t-1]/np.sqrt(s2[t-1])) + (Gaama*(np.abs((x[t-1]/np.sqrt(s2[t-1]))) - np.sqrt((2/np.pi)) )) + beta*np.log(s2[t-1]))
        
    s2 = s2[1:]
    ll = T*np.log(2*np.pi)/2 + np.sum(np.log(s2))/2 + np.sum(np.divide(x[1:]**2,2*s2))
    ll = ll
    if out is None:
        return ll
    else:
        return ll,s2




par0 = np.array([0.1,0.05,0.5,0.2]) # omega, alpha, beta, Gaama
# x = eps_insample*100

'''
No Parameter Restriction for positivity of sigma_square in case of EGARCH
'''
loglik = minimize(ll_Egarch11, par0, method='SLSQP',args = (eps_insample*100),
                  bounds = [(-1000,1000.0),(-1000,1000.0),(-1000,1000.0),(-1000,1000.0)],options={'ftol':1e-10}) #options={'disp': True,'ftol':1e-10})
estimates_EGARCH11_Normal = loglik.x #parameter estimates
print(pd.DataFrame(estimates_EGARCH11_Normal, index = ['omega','alpha','beta','Gaama'],columns = ['EGARCH11 - Normal Dist']))

##############################################################################


# EGARCH(1,1)     t - distributed
####################################

def ll_Egarch11_t_distribution(par0,x, out=None):
    T = np.size(x,axis = 0)
    omega = par0[0]
    alpha = par0[1]
    beta = par0[2]
    Gaama = par0[3]
    dof = par0[4]
    
    s2 =np.ones((T+1,1))*omega/(1-alpha-beta)
    x = np.append(0,x).reshape(T+1,1)
    
    for t in range(1,T+1):
        s2[t] = np.exp( omega + (alpha * x[t-1]/np.sqrt(s2[t-1])) + (Gaama*(np.abs((x[t-1]/np.sqrt(s2[t-1]))) - np.sqrt((2/np.pi)) )) + beta*np.log(s2[t-1]))
    s2 = s2[1:]
    
    # Using slide 57/73 Fin Eco
    first_term = T * np.log( gamma((dof+1)/2) * (np.pi**(-0.5)) * (1/gamma((dof/2))) * ((dof-2)**(-0.5)))
    second_term = -0.5 * np.sum(np.log(s2))
    third_term = -0.5 * (dof + 1) * np.sum( np.log( 1 + np.divide(x[1:]**2,s2*(dof-2)) ) )
    
    ll = -first_term - second_term - third_term  # taking negative of log likelihood
    #ll =  - second_term - third_term  # taking negative of log likelihood
    
    if out is None:
        return ll
    else:
        return ll,s2

par0 = np.array([0.1,0.05,0.5,0.2,5]) # omega, alpha, beta, gamma, DOF

loglik = minimize(ll_Egarch11_t_distribution, par0, method='SLSQP',args = (eps_insample*100),
                  bounds = [(-1000,1000.0),(-1000,1000.0),(-1000,1000.0),(-1000,1000.0),(5,1000)],options={'ftol':1e-10})  #options={'disp': True,'ftol':1e-10})
estimates_EGARCH11_t_distribution = loglik.x #parameter estimates
print(pd.DataFrame(estimates_EGARCH11_t_distribution, index = ['omega','alpha','beta','Gaama','DOF'],columns = ['EGARCH11 - t Dist']))

##############################################################################
