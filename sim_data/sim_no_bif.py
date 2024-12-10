# import python libraries
import numpy as np
import pandas as pd
import os

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)

# Get the directory containing the current file
current_dir = os.path.dirname(current_file_path)

# Change the working directory to the directory of the current file
os.chdir(current_dir)

if not os.path.exists('../data/no_bif'):
    os.makedirs('../data/no_bif')

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
k = 2 # carrying capacity
s = 1 # half-saturation constant of harvesting function

bl = 1
bh = 3 # bifurcation parameter high
c_rate = 1.5e-5 # bifurcation parameter change rate

np.random.seed(17)

sim_len = int((bh-bl)*dt/c_rate)
series_len = 1000

x0 = 0.8197 # intial condition (equilibrium value computed in Mathematica)
equi0 = 0.8197

t = np.arange(t0,sim_len,dt)

x = np.zeros(len(t))
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
equi[0]=equi0
rrate[0]=rrate0

rrate_record = []

# Run simulation
for i in range(len(t)-1):
    x[i+1] = x[i] + de_fun(x[i],g,k,b.iloc[i],s)*dt + dW[i]
    equi[i+1] = equi[i] + de_fun(equi[i],g,k,b.iloc[i],s)*dt
    rrate[i+1] = recov_fun(equi[i+1],g,k,b.iloc[i+1],s)
    # make sure that state variable remains >= 0
    if x[i+1] < 0:
        x[i+1] = 0
    # Determine the tipping point by the recovery rate changing from negative to positive
    if rrate[i] < 0 and rrate[i+1] > 0:
        rrate_record.append(i+1)

c1 = np.linspace(0,len(t)-1,series_len)
c1 = list(map(int,c1))

# Store series data in a temporary DataFrame
data = {'traj_x': x,'equix':equi,'b':b,'rrate': rrate}
df_temp = pd.DataFrame(data)

df_temp_c1 = df_temp.iloc[c1].copy()

df_temp_c1['Time'] = range(len(df_temp_c1))

cols = ['Time'] + [col for col in df_temp_c1.columns if col != 'Time']

df_temp_c1 = df_temp_c1[cols]

print('Simulation complete')

#------------------------------------
# Export data 
#-----------------------------------

filepath_train = '../data/no_bif/no_bif_train.csv'

df_temp_c1.to_csv(filepath_train,index=False)