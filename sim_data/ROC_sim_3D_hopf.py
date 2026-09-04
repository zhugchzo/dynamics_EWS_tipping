# import python libraries
import numpy as np
import pandas as pd
import os
import math

if not os.path.exists('../data/3D_hopf/ROC'):
    os.makedirs('../data/3D_hopf/ROC')

# Simulation parameters
dt = 0.01
t0 = 0
tburn = 500 # burn-in period
sigma_x = 0.01 # noise intensity
sigma_y = 0.01
sigma_z = 0.01

#----------------------------------
# Simulate model
#----------------------------------

# Model

def de_fun_x(x,y):
    return -x-y

def de_fun_y(y,z,u,p,s):
    return -p*z+u*y+s*z**2-y*z**2

def de_fun_z(x,z,q):
    return -q*(x+z)

def recov_fun(y,z,u,p,q,s):

    j11 = -1
    j12 = -1
    j13 = 0
    j21 = 0
    j22 = u-z**2
    j23 = -p+2*z*(s-y)
    j31 = -q
    j32 = 0
    j33 = -q
    
    jac = np.array([[j11,j12,j13],[j21,j22,j23],[j31,j32,j33]])

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
p = 1
q = 1.2
s = 0.8

ul = 0
ubif = 0.35 # bifurcation point (computed in Mathematica)
uh = 0.4 # bifurcation parameter high

c_rate = 1e-4

sim_len = int((uh-ul)*dt/c_rate)
series_len = 1000

x0 = 0 # intial condition (equilibrium value computed in Mathematica)
y0 = 0
z0 = 0
equix0 = 0
equiy0 = 0
equiz0 = 0

for random_seed in range(200):

    np.random.seed(random_seed)

    t = np.arange(t0,sim_len,dt)

    x = np.zeros(len(t))
    y = np.zeros(len(t))
    z = np.zeros(len(t))
    x_null = np.zeros(len(t))
    y_null = np.zeros(len(t))
    z_null = np.zeros(len(t))
    equix = np.zeros(len(t))
    equiy = np.zeros(len(t))
    equiz  = np.zeros(len(t))

    rrate = np.zeros(len(t))

    u = pd.Series(np.linspace(ul,uh,len(t)),index=t)

    # Create brownian increments (s.d. sqrt(dt))
    dW_x_burn = np.random.normal(loc=0, scale=sigma_x*np.sqrt(dt), size = int(tburn/dt))
    dW_x = np.random.normal(loc=0, scale=sigma_x*np.sqrt(dt), size = len(t))

    dW_y_burn = np.random.normal(loc=0, scale=sigma_y*np.sqrt(dt), size = int(tburn/dt))
    dW_y = np.random.normal(loc=0, scale=sigma_y*np.sqrt(dt), size = len(t))

    dW_z_burn = np.random.normal(loc=0, scale=sigma_z*np.sqrt(dt), size = int(tburn/dt))
    dW_z = np.random.normal(loc=0, scale=sigma_z*np.sqrt(dt), size = len(t))
            
    # Run burn-in period on x0
    for i in range(int(tburn/dt)):
        x0 = x0 + de_fun_x(x0,y0)*dt + dW_x_burn[i]
        y0 = y0 + de_fun_y(y0,z0,u[0],p,s)*dt + dW_y_burn[i]
        z0 = z0 + de_fun_z(x0,z0,q)*dt + dW_z_burn[i]
        equix0 = equix0 + de_fun_x(equix0,equiy0)*dt
        equiy0 = equiy0 + de_fun_y(equiy0,equiz0,u[0],p,s)*dt
        equiz0 = equiz0 + de_fun_z(equix0,equiz0,q)*dt

    rrate0 = recov_fun(equiy0,equiz0,u[0],p,q,s)[0]
    
    if rrate0 == 1997.0209:
        print('rrate Nan appears')
        x0 = 0 # intial condition (equilibrium value computed in Mathematica)
        y0 = 0
        z0 = 0
        equix0 = 0
        equiy0 = 0
        equiz0 = 0

    if  math.isnan(x0) or math.isnan(y0) or math.isnan(z0):
        print('sim Nan appears')
        x0 = 0 # intial condition (equilibrium value computed in Mathematica)
        y0 = 0
        z0 = 0
        equix0 = 0
        equiy0 = 0
        equiz0 = 0
    
    # Initial condition post burn-in period
    x[0]=x0
    y[0]=y0
    z[0]=z0
    x_null[0] = x0
    y_null[0] = y0
    z_null[0] = z0
    equix[0]=equix0
    equiy[0]=equiy0
    equiz[0]=equiz0

    rrate[0]=rrate0

    rrate_record = []

    # Run simulation
    for i in range(len(t)-1):
        x[i+1] = x[i] + de_fun_x(x[i],y[i])*dt + dW_x[i]
        y[i+1] = y[i] + de_fun_y(y[i],z[i],u.iloc[i],p,s)*dt + dW_y[i]
        z[i+1] = z[i] + de_fun_z(x[i],z[i],q)*dt + dW_z[i]
        x_null[i+1] = x_null[i] + de_fun_x(x_null[i],y_null[i])*dt + dW_x[i]
        y_null[i+1] = y_null[i] + de_fun_y(y_null[i],z_null[i],u.iloc[0],p,s)*dt + dW_y[i]
        z_null[i+1] = z_null[i] + de_fun_z(x_null[i],z_null[i],q)*dt + dW_z[i]
        equix[i+1] = equix[i] + de_fun_x(equix[i],equiy[i])*dt
        equiy[i+1] = equiy[i] + de_fun_y(equiy[i],equiz[i],u.iloc[i],p,s)*dt
        equiz[i+1] = equiz[i] + de_fun_z(equix[i],equiz[i],q)*dt

        rrate[i+1] = recov_fun(equiy[i+1],equiz[i+1],u.iloc[i+1],p,q,s)[0]

        if rrate[i+1] == 1997.0209:
            print('rrate Nan appears')
            break

        if math.isnan(x[i+1]) or math.isnan(y[i+1]) or math.isnan(z[i+1]):
            print('sim Nan appears')
            break
               
        # Determine the tipping point by the recovery rate changing from negative to positive
        if rrate[i] < 0 and rrate[i+1] > 0:
            rrate_record.append(i+1)

    if len(rrate_record) != 0:

        trans_time = rrate_record[0] # the time of tipping point

        c = np.linspace(0,trans_time,series_len)
        c = list(map(int,c))

        print('Simulation complete {}'.format(random_seed))

    else:
        print('Cant find rrate = 0')
            
    # Store series data in a temporary DataFrame
    data = {'t':t,'traj_x': x, 'traj_y': y, 'traj_z': z,
             'equix': equix, 'equiy': equiy, 'equiz': equiz,
               'u': u, 'real_eval1': rrate}
    data_null = {'t':t,'traj_x': x_null, 'traj_y': y_null, 'traj_z': z_null}
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

    filepath = '../data/3D_hopf/ROC/3D_hopf_{}.csv'.format(random_seed)
    filepath_null = '../data/3D_hopf/ROC/3D_hopf_null_{}.csv'.format(random_seed)

    df_temp_c.to_csv(filepath,index=False)
    df_temp_null_c.to_csv(filepath_null,index=False)