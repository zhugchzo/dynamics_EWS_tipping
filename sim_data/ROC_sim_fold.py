# import python libraries
import numpy as np
import pandas as pd
import os

if not os.path.exists('../data/fold/ROC'):
    os.makedirs('../data/fold/ROC')

# Simulation parameters
dt = 0.01
t0 = 0
tburn = 500 # burn-in period
sigma = 0.01 # noise intensity

#----------------------------------
# Simulate model
#----------------------------------

# Model

def de_fun(x,g,k,h,s):
    return g*x*(1-x/k) - h*x**2/(s**2 + x**2)

def recov_fun(x,g,k,h,s):
    rrate = g - 2*g*x/k - 2*s**2*h*x/(s**2 + x**2)**2
    return rrate

# Model parameters
g = 1 # growth rate
k = 9 # carrying capacity
s = 1 # half-saturation constant of harvesting function

bcrit = 2.3 # bifurcation point (computed in Mathematica)
bl = 1
bh = 2.5 # bifurcation parameter high
c_rate = 1.5e-5 # bifurcation parameter change rate

sim_len = int((bh-bl)*dt/c_rate)
series_len = 1000

x0 = 0.8197 # intial condition (equilibrium value computed in Mathematica)
equi0 = 0.8197

for random_seed in range(100):

    np.random.seed(random_seed)

    t = np.arange(t0,sim_len,dt)

    x = np.zeros(len(t))
    x_null = np.zeros(len(t))
    equi = np.zeros(len(t))
    rrate = np.zeros(len(t))

    b = pd.Series(np.linspace(bl,bh,len(t)),index=t)

    # Create brownian increments (s.d. sqrt(dt))
    dW_burn = np.random.normal(loc=0, scale=sigma*np.sqrt(dt), size = int(tburn/dt))
    dW = np.random.normal(loc=0, scale=sigma*np.sqrt(dt), size = len(t))

    # Run burn-in period on x0
    for i in range(int(tburn/dt)):
        x0 = x0 + de_fun(x0,g,k,b[0],s)*dt + dW_burn[i]
        equi0 = equi0 + de_fun(equi0,g,k,b[0],s)*dt

    rrate0 = recov_fun(equi0,g,k,b[0],s)

    # Initial condition post burn-in period
    x[0]=x0
    x_null[0]=x0
    equi[0]=equi0
    rrate[0]=rrate0

    rrate_record = []

    # Run simulation
    for i in range(len(t)-1):
        x[i+1] = x[i] + de_fun(x[i],g,k,b.iloc[i],s)*dt + dW[i]
        x_null[i+1] = x_null[i] + de_fun(x_null[i],g,k,b.iloc[0],s)*dt + dW[i]
        equi[i+1] = equi[i] + de_fun(equi[i],g,k,b.iloc[i],s)*dt
        rrate[i+1] = recov_fun(equi[i+1],g,k,b.iloc[i+1],s)
        # make sure that state variable remains >= 0
        if x[i+1] < 0:
            x[i+1] = 0
        # Determine the tipping point by the recovery rate changing from negative to positive
        if rrate[i] < 0 and rrate[i+1] > 0:
            rrate_record.append(i+1)

    if len(rrate_record) != 0:

        trans_time = rrate_record[0] # the time of tipping point

        c = np.linspace(0,trans_time,series_len)
        c = list(map(int,c))

    else:
        print('Cant find rrate = 0')

    # Store series data in a temporary DataFrame
    data = {'t':t,'traj_x': x,'equix':equi,'b':b,'rrate': rrate}
    data_null = {'t':t,'traj_x': x_null}
    df_temp = pd.DataFrame(data)
    df_temp_null = pd.DataFrame(data_null)

    df_temp_c = df_temp.iloc[c].copy()
    df_temp_null_c = df_temp_null.iloc[c].copy()

    df_temp_c['Time'] = range(len(df_temp_c))
    df_temp_null_c['Time'] = range(len(df_temp_null_c))

    df_temp_c['t_diff'] = df_temp_c['t'].diff()
    df_temp_null_c['t_diff'] = df_temp_null_c['t'].diff()

    cols = ['Time'] + [col for col in df_temp_c.columns if col != 'Time']
    cols_null = ['Time'] + [col for col in df_temp_null_c.columns if col != 'Time']

    df_temp_c = df_temp_c[cols]
    df_temp_null_c = df_temp_null_c[cols_null]

    print('Simulation complete {}'.format(random_seed))

    #------------------------------------
    # Export data 
    #-----------------------------------

    filepath = '../data/fold/ROC/fold_{}.csv'.format(random_seed)
    filepath_null = '../data/fold/ROC/fold_null_{}.csv'.format(random_seed)

    df_temp_c.to_csv(filepath,index=False)
    df_temp_null_c.to_csv(filepath_null,index=False)