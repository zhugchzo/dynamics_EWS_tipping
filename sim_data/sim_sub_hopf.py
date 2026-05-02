# import python libraries
import numpy as np
import pandas as pd
import os
import math
from sympy import symbols, diff

if not os.path.exists('../data/sub_hopf'):
    os.makedirs('../data/sub_hopf')

# Simulation parameters
dt = 0.01
t0 = 0
tburn = 500 # burn-in period
sigma_x = 0.01 # noise intensity
sigma_y = 0.01

#----------------------------------
# Simulate model
#----------------------------------

# Model

def de_fun_x(x,y,a,b,c,d):
    return 1 - d*x - (b+a*y/(y+c))*x*y

def de_fun_y(x,y,a,b,c):
    return (b+a*y/(y+c))*x*y - y

def recov_fun(x,y,a,b,c,d):

    j11 = -d - (b+a*y/(y+c))*y
    j12 = -a*c*x*y/(y+c)**2 - (b+a*y/(y+c))*x
    j21 = (b+a*y/(y+c))*y
    j22 = a*c*x*y/(y+c)**2 + (b+a*y/(y+c))*x - 1

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

def stable_exponent(u_t, v_t, a_t, c_t, imag_t, b = 0.06, d = 0.057):

    u,v, x,y  = symbols('u v x y')

    F = de_fun_x(u,v,a_t,b,c_t,d)

    dF_du = diff(F, u)
    dF_dv = diff(F, v)

    dF_du_value = dF_du.subs({u: u_t, v: v_t})
    dF_dv_value = dF_dv.subs({u: u_t, v: v_t})

    f = de_fun_x((x+u_t),(-(dF_du_value*x+imag_t*y)/dF_dv_value+v_t),a_t,b,c_t,d) + imag_t*y
    g = -(dF_du_value*de_fun_x((x+u_t),(-(dF_du_value*x+imag_t*y)/dF_dv_value+v_t),a_t,b,c_t,d) + dF_dv_value*de_fun_y((x+u_t),(-(dF_du_value*x+imag_t*y)/dF_dv_value+v_t),a_t,b,c_t))/imag_t - imag_t*x

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
b = 0.06
d = 0.057

al = 1.2
cl = 0
ah = 0.3
ch = 0.9

c_rate = 1.5e-5

np.random.seed(17)

sim_len = int((ch-cl)*dt/c_rate)
series_len = 1000

x0 = 2.9509  # intial condition (equilibrium value computed in Mathematica)
y0 = 0.8318
equix0 = 2.9509
equiy0 = 0.8318

t = np.arange(t0,sim_len,dt)

x = np.zeros(len(t))
y = np.zeros(len(t))
equix = np.zeros(len(t))
equiy = np.zeros(len(t))

rrate = np.zeros(len(t))
imag_dom_eval = np.zeros(len(t))
real_oth_eval = np.zeros(len(t))
imag_oth_eval = np.zeros(len(t))

a = pd.Series(np.linspace(al,ah,len(t)),index=t)
c = pd.Series(np.linspace(cl,ch,len(t)),index=t)

# Create brownian increments (s.d. sqrt(dt))
dW_x_burn = np.random.normal(loc=0, scale=sigma_x*np.sqrt(dt), size = int(tburn/dt))
dW_x = np.random.normal(loc=0, scale=sigma_x*np.sqrt(dt), size = len(t))

dW_y_burn = np.random.normal(loc=0, scale=sigma_y*np.sqrt(dt), size = int(tburn/dt))
dW_y = np.random.normal(loc=0, scale=sigma_y*np.sqrt(dt), size = len(t))

# Run burn-in period on x0
for i in range(int(tburn/dt)):
    x0 = x0 + de_fun_x(x0,y0,a[0],b,c[0],d)*dt + dW_x_burn[i]
    y0 = y0 + de_fun_y(x0,y0,a[0],b,c[0])*dt + dW_y_burn[i]
    equix0 = equix0 + de_fun_x(equix0,equiy0,a[0],b,c[0],d)*dt
    equiy0 = equiy0 + de_fun_y(equix0,equiy0,a[0],b,c[0])*dt

rrate0 = recov_fun(equix0,equiy0,a[0],b,c[0],d)[0]
imag_dom_eval0 = recov_fun(equix0,equiy0,a[0],b,c[0],d)[1]
real_oth_eval0 = recov_fun(equix0,equiy0,a[0],b,c[0],d)[2]
imag_oth_eval0 = recov_fun(equix0,equiy0,a[0],b,c[0],d)[3]

if rrate0 == 1997.0209 or math.isnan(x0) or math.isnan(y0):
    print('Nan appears')
    x0 = 2.9509 # intial condition (equilibrium value computed in Mathematica)
    y0 = 0.8318
    equix0 = 2.9509
    equiy0 = 0.8318

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
    x[i+1] = x[i] + de_fun_x(x[i],y[i],a.iloc[i],b,c.iloc[i],d)*dt + dW_x[i]
    y[i+1] = y[i] + de_fun_y(x[i],y[i],a.iloc[i],b,c.iloc[i])*dt + dW_y[i]
    equix[i+1] = equix[i] + de_fun_x(equix[i],equiy[i],a.iloc[i],b,c.iloc[i],d)*dt
    equiy[i+1] = equiy[i] + de_fun_y(equix[i],equiy[i],a.iloc[i],b,c.iloc[i])*dt

    rrate[i+1] = recov_fun(equix[i+1],equiy[i+1],a.iloc[i+1],b,c.iloc[i+1],d)[0]
    imag_dom_eval[i+1] = recov_fun(equix[i+1],equiy[i+1],a.iloc[i+1],b,c.iloc[i+1],d)[1]
    real_oth_eval[i+1] = recov_fun(equix[i+1],equiy[i+1],a.iloc[i+1],b,c.iloc[i+1],d)[2]
    imag_oth_eval[i+1] = recov_fun(equix[i+1],equiy[i+1],a.iloc[i+1],b,c.iloc[i+1],d)[3]

    if rrate[i+1] == 1997.0209:
        print('rrate Nan appears')
        break

    if math.isnan(x[i+1]) or math.isnan(y[i+1]):
        print('sim Nan appears')
        break

    # Determine the tipping point by the recovery rate changing from negative to positive
    if rrate[i] < 0 and rrate[i+1] > 0:
        rrate_record.append(i+1)

    # if a.iloc[i] < 0.39:
    #     length = i
    #     break

if len(rrate_record) != 0:

    trans_time = rrate_record[0] # the time of tipping point

    c1 = np.linspace(0,trans_time,series_len)
    c1 = list(map(int,c1))

    c2 = c1.copy()

    # c0 = np.linspace(trans_time,length,int((length-trans_time)*series_len/trans_time))
    c0 = np.linspace(trans_time,len(t)-1,int((len(t)-1-trans_time)*series_len/trans_time))
    c0 = c0[1:]
    c0 = list(map(int,c0))
    c2.extend(c0)

    print('Simulation complete')

else:
    print('Cant find rrate = 0')

# Store series data in a temporary DataFrame
data = {'t':t,'traj_x': x, 'traj_y': y,'equix': equix, 'equiy': equiy, 'a': a, 'c': c, 'real_eval1': rrate, 'imag_eval1': imag_dom_eval, 'real_eval2': real_oth_eval, 'imag_eval2': imag_oth_eval}
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

df_temp_c1_stable = df_temp_c1[['Time','equix','equiy','a','c','imag_eval1']].iloc[-50:]

T = df_temp_c1_stable['Time'].values + 1
u_t = df_temp_c1_stable['equix'].values
v_t = df_temp_c1_stable['equiy'].values
a_t = df_temp_c1_stable['a'].values
c_t = df_temp_c1_stable['c'].values
imag_t = df_temp_c1_stable['imag_eval1'].values

for i in range(len(u_t)):

    stable = stable_exponent(u_t[i],v_t[i],a_t[i],c_t[i],imag_t[i])

    stable_list.append(stable)

dic_stable = pd.DataFrame({'Time':T, 'stable_expnt':stable_list})

#------------------------------------
# Export data 
#-----------------------------------

filepath_train = '../data/sub_hopf/sub_hopf_train.csv'
filepath = '../data/sub_hopf/sub_hopf.csv'
filepath_stable = '../data/sub_hopf/sub_hopf_stable.csv'

#df_temp_c1.to_csv(filepath_train,index=False)
df_temp_c2.to_csv(filepath,index=False)
dic_stable.to_csv(filepath_stable,index=False)