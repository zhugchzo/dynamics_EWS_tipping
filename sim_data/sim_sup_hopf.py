# import python libraries
import numpy as np
import pandas as pd
import os
import math
from sympy import symbols, diff

if not os.path.exists('../data/sup_hopf'):
    os.makedirs('../data/sup_hopf')

# Simulation parameters
dt = 0.01
t0 = 0
tburn = 500 # burn-in period
sigma_x = 0.0025 # noise intensity
sigma_y = 0.0025

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

def stable_exponent(u_t, v_t, b_t, imag_t,r = 0.5,a = 0.4,b = 0.6,e = 0.6,d = 0.15):

    u,v, x,y  = symbols('u v x y')

    F = de_fun_x(u,v,b_t,r,a,b)

    dF_du = diff(F, u)
    dF_dv = diff(F, v)

    dF_du_value = dF_du.subs({u: u_t, v: v_t})
    dF_dv_value = dF_dv.subs({u: u_t, v: v_t})

    f = de_fun_x((x+u_t),(-(dF_du_value*x+imag_t*y)/dF_dv_value+v_t),b_t,r,a,b) + imag_t*y
    g = -(dF_du_value*de_fun_x((x+u_t),(-(dF_du_value*x+imag_t*y)/dF_dv_value+v_t),b_t,r,a,b) + dF_dv_value*de_fun_y((x+u_t),(-(dF_du_value*x+imag_t*y)/dF_dv_value+v_t),a,b,e,d))/imag_t - imag_t*x

    df_dx = diff(f, x)
    df_dy = diff(f, y)

    df_dxdx = diff(df_dx, x)
    df_dxdy = diff(df_dx, y)
    df_dydy = diff(df_dy, y)

    df_dxdxdx = diff(df_dxdx, x)
    df_dxdydy = diff(df_dxdy, y)

    df_dxdx_value = df_dxdx.subs({x:0,y:0})
    df_dxdy_value = df_dxdy.subs({x:0,y:0})
    df_dydy_value = df_dydy.subs({x:0,y:0})

    df_dxdxdx_value = df_dxdxdx.subs({x:0,y:0})
    df_dxdydy_value = df_dxdydy.subs({x:0,y:0})

    dg_dx = diff(g, x)
    dg_dy = diff(g, y)

    dg_dxdx = diff(dg_dx, x)
    dg_dxdy = diff(dg_dx, y)
    dg_dydy = diff(dg_dy, y)

    dg_dxdxdy = diff(dg_dxdx, y)
    dg_dydydy = diff(dg_dydy, y)

    dg_dxdx_value = dg_dxdx.subs({x:0,y:0})
    dg_dxdy_value = dg_dxdy.subs({x:0,y:0})
    dg_dydy_value = dg_dydy.subs({x:0,y:0})

    dg_dxdxdy_value = dg_dxdxdy.subs({x:0,y:0})
    dg_dydydy_value = dg_dydydy.subs({x:0,y:0})

    stable = 1/16*(df_dxdxdx_value + df_dxdydy_value + dg_dxdxdy_value + dg_dydydy_value) + 1/(16*imag_t)*(df_dxdy_value*(df_dxdx_value + df_dydy_value) - dg_dxdy_value*(dg_dxdx_value + dg_dydy_value) - df_dxdx_value*dg_dxdx_value + df_dydy_value*dg_dydy_value)

    return stable

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

np.random.seed(40)

sim_len = int((kh-kl)*dt/c_rate)
series_len = 1000

x0 = 1 # intial condition (equilibrium value computed in Mathematica)
y0 = 1
equix0 = 1
equiy0 = 1

t = np.arange(t0,sim_len,dt)

x = np.zeros(len(t))
y = np.zeros(len(t))
equix = np.zeros(len(t))
equiy = np.zeros(len(t))

rrate = np.zeros(len(t))
imag_dom_eval = np.zeros(len(t))
real_oth_eval = np.zeros(len(t))
imag_oth_eval = np.zeros(len(t))

k = pd.Series(np.linspace(kl,kh,len(t)),index=t)

# Create brownian increments (s.d. sqrt(dt))
dW_x_burn = np.random.normal(loc=0, scale=sigma_x*np.sqrt(dt), size = int(tburn/dt))
dW_x = np.random.normal(loc=0, scale=sigma_x*np.sqrt(dt), size = len(t))

dW_y_burn = np.random.normal(loc=0, scale=sigma_y*np.sqrt(dt), size = int(tburn/dt))
dW_y = np.random.normal(loc=0, scale=sigma_y*np.sqrt(dt), size = len(t))

# Run burn-in period on x0
for i in range(int(tburn/dt)):
    x0 = x0 + de_fun_x(x0,y0,k[0],r,a,b)*dt + dW_x_burn[i]
    y0 = y0 + de_fun_y(x0,y0,a,b,e,d)*dt + dW_y_burn[i]
    equix0 = equix0 + de_fun_x(equix0,equiy0,k[0],r,a,b)*dt
    equiy0 = equiy0 + de_fun_y(equix0,equiy0,a,b,e,d)*dt

rrate0 = recov_fun(equix0,equiy0,k[0],r,a,b,e,d)[0]
imag_dom_eval0 = recov_fun(equix0,equiy0,k[0],r,a,b,e,d)[1]
real_oth_eval0 = recov_fun(equix0,equiy0,k[0],r,a,b,e,d)[2]
imag_oth_eval0 = recov_fun(equix0,equiy0,k[0],r,a,b,e,d)[3]

if rrate0 == 1997.0209 or math.isnan(x0) or math.isnan(y0):
    print('Nan appears')
    x0 = 1 # intial condition (equilibrium value computed in Mathematica)
    y0 = 1
    equix0 = 1
    equiy0 = 1

# Initial condition post burn-in period
x[0] = x0
y[0] = y0
equix[0] = equix0
equiy[0] = equiy0

rrate[0] = rrate0
imag_dom_eval[0] = imag_dom_eval0
real_oth_eval[0] = real_oth_eval0
imag_oth_eval[0] = imag_oth_eval0

rrate_record = []

# Run simulation
for i in range(len(t)-1):
    x[i+1] = x[i] + de_fun_x(x[i],y[i],k.iloc[i],r,a,b)*dt + dW_x[i]
    y[i+1] = y[i] + de_fun_y(x[i],y[i],a,b,e,d)*dt + dW_y[i]
    equix[i+1] = equix[i] + de_fun_x(equix[i],equiy[i],k.iloc[i],r,a,b)*dt
    equiy[i+1] = equiy[i] + de_fun_y(equix[i],equiy[i],a,b,e,d)*dt

    rrate[i+1] = recov_fun(equix[i+1],equiy[i+1],k.iloc[i+1],r,a,b,e,d)[0]
    imag_dom_eval[i+1] = recov_fun(equix[i+1],equiy[i+1],k.iloc[i+1],r,a,b,e,d)[1]
    real_oth_eval[i+1] = recov_fun(equix[i+1],equiy[i+1],k.iloc[i+1],r,a,b,e,d)[2]
    imag_oth_eval[i+1] = recov_fun(equix[i+1],equiy[i+1],k.iloc[i+1],r,a,b,e,d)[3]

    if rrate[i+1] == 1997.0209:
        print('rrate Nan appears')
        break

    if math.isnan(x[i+1]) or math.isnan(y[i+1]):
        print('sim Nan appears')
        break

    # Determine the tipping point by the recovery rate changing from negative to positive
    if rrate[i] < 0 and rrate[i+1] > 0:
        rrate_record.append(i+1)

    # if k.iloc[i] > 2.9:
    #     length = i
    #     break

if len(rrate_record) != 0:

    trans_time = rrate_record[0] # the time of tipping point

    c1 = np.linspace(0,trans_time,series_len)
    c1 = list(map(int,c1))

    c2 = c1.copy()

    #c0 = np.linspace(trans_time,length,int((length-trans_time)*series_len/trans_time))
    c0 = np.linspace(trans_time,len(t)-1,int((len(t)-1-trans_time)*series_len/trans_time))
    c0 = c0[1:]
    c0 = list(map(int,c0))
    c2.extend(c0)

    print('Simulation complete')

else:
    print('Cant find rrate = 0')

# Store series data in a temporary DataFrame
data = {'t':t,'traj_x': x, 'traj_y': y,'equix': equix, 'equiy': equiy, 'b': k, 'real_eval1': rrate, 'imag_eval1': imag_dom_eval, 'real_eval2': real_oth_eval, 'imag_eval2': imag_oth_eval}
df_temp = pd.DataFrame(data)

df_temp_c1 = df_temp.iloc[c1].copy()
df_temp_c2 = df_temp.iloc[c2].copy()

df_temp_c1['Time'] = range(len(df_temp_c1))
df_temp_c2['Time'] = range(len(df_temp_c2))

df_temp_c1['t_diff'] = df_temp_c1['t'].diff()
df_temp_c2['t_diff'] = df_temp_c2['t'].diff()

cols = ['Time'] + [col for col in df_temp_c1.columns if col != 'Time']

df_temp_c1 = df_temp_c1[cols]
df_temp_c2 = df_temp_c2[cols]

stable_list = []

df_temp_c1_stable = df_temp_c1[['Time','equix','equiy','b','imag_eval1']].iloc[-50:]

T = df_temp_c1_stable['Time'].values + 1
u_t = df_temp_c1_stable['equix'].values
v_t = df_temp_c1_stable['equiy'].values
b_t = df_temp_c1_stable['b'].values
imag_t = df_temp_c1_stable['imag_eval1'].values

for i in range(len(u_t)):

    stable = stable_exponent(u_t[i],v_t[i],b_t[i],imag_t[i])

    stable_list.append(stable)

dic_stable = pd.DataFrame({'Time':T, 'stable_expnt':stable_list})

#------------------------------------
# Export data 
#-----------------------------------

filepath_train = '../data/sup_hopf/sup_hopf_train.csv'
filepath = '../data/sup_hopf/sup_hopf.csv'
filepath_stable = '../data/sup_hopf/sup_hopf_stable.csv'

# df_temp_c1.to_csv(filepath_train,index=False)
df_temp_c2.to_csv(filepath,index=False)
dic_stable.to_csv(filepath_stable,index=False)