# import python libraries
import numpy as np
import pandas as pd
import os
import math

# Simulation parameters
dt = 0.01
t0 = 0
tburn = 500 # burn-in period

#----------------------------------
# Simulate model
#----------------------------------

# Model

def de_fun_x(x,y,k,r,a,b):
    return r*x*(1-x/k) - (a*x*y)/(b+x)

def de_fun_y(x,y,a,b,e,d):
    return e*a*x*y/(b+x) - d*y

def recov_fun(x,y,k,r,a,b,e,d):

    j11 = r - 2*r*x/k - a*b*y/(b+x)**2
    j12 = -a*x/(b+x)
    j21 = e*a*b*y/(b+x)**2
    j22 = e*a*x/(b+x) - d

    jac = np.array([[j11,j12],[j21,j22]])

    try:
        evals = np.linalg.eigvals(jac)
    except:
        return 1997.0209, 1997.0209, 1997.0209, 1997.0209  # if error; This particular number has no other significance, it's just the author's birthday

    evals = [lam for lam in evals]
    dom_eval = max(evals, key=lambda x: x.real)

    evals.remove(dom_eval)
    oth_eval = evals[0]

    rrate = dom_eval.real
    imag_dom_eval = dom_eval.imag

    real_oth_eval = oth_eval.real
    imag_oth_eval = oth_eval.imag
    
    return rrate, imag_dom_eval, real_oth_eval, imag_oth_eval

# Model parameters
r = 0.5 # maximum growth rate of resource
a = 0.4 # maximum grazing rate
b = 0.6 # half saturation constant
e = 0.6 # assimilation efficiency of grazer
d = 0.15 # mortality rate of grazer

kbif = 0 # bifurcation point (computed in Mathematica)
kl = 1.6
kh = 4 # 2.9 control parameter final value
c_rate = 1e-5

sim_len = int((kh-kl)*dt/c_rate)
series_len = 1000

x0 = 1 # intial condition (equilibrium value computed in Mathematica)
y0 = 1
equix0 = 1
equiy0 = 1

for sigma in [0.0025, 0.01, 0.02, 0.03, 0.04, 0.05]:

    if not os.path.exists('../../data/sup_hopf/noise/{}'.format(sigma)):
        os.makedirs('../../data/sup_hopf/noise/{}'.format(sigma))

    ii = 0
    random_seed = 0

    while ii < 50:

        np.random.seed(random_seed)

        t = np.arange(t0,sim_len,dt)

        x = np.zeros(len(t))
        y = np.zeros(len(t))
        x_null = np.zeros(len(t))
        y_null = np.zeros(len(t))
        equix = np.zeros(len(t))
        equiy = np.zeros(len(t))

        rrate = np.zeros(len(t))

        k = pd.Series(np.linspace(kl,kh,len(t)),index=t)

        # Create brownian increments (s.d. sqrt(dt))
        dW_x_burn = np.random.normal(loc=0, scale=sigma*np.sqrt(dt), size = int(tburn/dt))
        dW_x = np.random.normal(loc=0, scale=sigma*np.sqrt(dt), size = len(t))

        dW_y_burn = np.random.normal(loc=0, scale=sigma*np.sqrt(dt), size = int(tburn/dt))
        dW_y = np.random.normal(loc=0, scale=sigma*np.sqrt(dt), size = len(t))

        # Run burn-in period on x0
        for i in range(int(tburn/dt)):
            x0 = x0 + de_fun_x(x0,y0,k[0],r,a,b)*dt + dW_x_burn[i]
            y0 = y0 + de_fun_y(x0,y0,a,b,e,d)*dt + dW_y_burn[i]
            equix0 = equix0 + de_fun_x(equix0,equiy0,k[0],r,a,b)*dt
            equiy0 = equiy0 + de_fun_y(equix0,equiy0,a,b,e,d)*dt

        rrate0 = recov_fun(equix0,equiy0,k[0],r,a,b,e,d)[0]

        if rrate0 == 1997.0209 or math.isnan(x0) or math.isnan(y0):
            print('Nan appears')
            x0 = 1 # intial condition (equilibrium value computed in Mathematica)
            y0 = 1
            equix0 = 1
            equiy0 = 1

        # Initial condition post burn-in period
        x[0] = x0
        y[0] = y0
        x_null[0] = x0
        y_null[0] = y0
        equix[0] = equix0
        equiy[0] = equiy0

        rrate[0] = rrate0

        rrate_record = []

        # Run simulation
        for i in range(len(t)-1):
            x[i+1] = x[i] + de_fun_x(x[i],y[i],k.iloc[i],r,a,b)*dt + dW_x[i]
            y[i+1] = y[i] + de_fun_y(x[i],y[i],a,b,e,d)*dt + dW_y[i]
            x_null[i+1] = x_null[i] + de_fun_x(x_null[i],y_null[i],k.iloc[0],r,a,b)*dt + dW_x[i]
            y_null[i+1] = y_null[i] + de_fun_y(x_null[i],y_null[i],a,b,e,d)*dt + dW_y[i]
            equix[i+1] = equix[i] + de_fun_x(equix[i],equiy[i],k.iloc[i],r,a,b)*dt
            equiy[i+1] = equiy[i] + de_fun_y(equix[i],equiy[i],a,b,e,d)*dt

            rrate[i+1] = recov_fun(equix[i+1],equiy[i+1],k.iloc[i+1],r,a,b,e,d)[0]

            if rrate[i+1] == 1997.0209:
                print('rrate Nan appears')
                break

            if math.isnan(x[i+1]) or math.isnan(y[i+1]):
                print('sim Nan appears')
                break

            # Determine the tipping point by the recovery rate changing from negative to positive
            if rrate[i] < 0 and rrate[i+1] > 0:
                rrate_record.append(i+1)
                break

            if x[i] > 10:
                length = i
                break

            if x_null[i] > 10:
                length = i
                break

        if len(rrate_record) != 0:

            trans_time = rrate_record[0] # the time of tipping point

            c = np.linspace(0,trans_time,series_len)
            c = list(map(int,c))

            # Store series data in a temporary DataFrame
            data = {'t':t,'traj_x': x, 'traj_y': y,'equix': equix, 'equiy': equiy, 'b': k, 'real_eval1': rrate}
            data_null = {'t':t,'traj_x': x_null, 'traj_y': y_null}
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

            #------------------------------------
            # Export data 
            #-----------------------------------

            filepath = '../../data/sup_hopf/noise/{}/sup_hopf_{}.csv'.format(sigma,ii)
            filepath_null = '../../data/sup_hopf/noise/{}/sup_hopf_null_{}.csv'.format(sigma,ii)

            df_temp_c.to_csv(filepath,index=False)
            df_temp_null_c.to_csv(filepath_null,index=False)

            print('Simulation complete, ii = {}, seed = {}'.format(ii,random_seed))

            ii += 1
            random_seed += 1

        else:
            print('Cant find rrate = 0, ii = {}, seed = {}'.format(ii,random_seed))

            random_seed += 1