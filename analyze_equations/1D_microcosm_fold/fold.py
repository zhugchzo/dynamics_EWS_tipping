import numpy as np
import pandas as pd
import scipy.integrate as spi
import os

# Make folders for this directory
if not os.path.exists('output_fold'):
    os.makedirs('output_fold')

df = pd.read_csv('../../results/microcosm/microcosm.csv')

bif_pars_col = ['Time']
states_col = ['traj_x']
f_elements = ['cte', 'x', 'b', 'x^2', 'bx', 'x^3', 'bx^2']

bif_pars = df[bif_pars_col].values
states = df[states_col].values
f_coef = df[f_elements].values

initial_theta = df['initial_theta'].values
delta_theta = df['delta_theta'].values

end_theta = initial_theta + 499*delta_theta

def f(s,t,b,f0):

    x = s[0]
    b = b[0]

    polys = np.array([1, x, b, x**2, b*x, x**3, b*x**2])

    dydt = np.dot(f0,polys)
    
    return dydt

for i in range(len(bif_pars)):

    s0 = states[i]
    b0 = np.array([end_theta[i]])
    f0 = f_coef[i]

    ## Simulate the system
    t = np.arange(0, 500, 0.01)
    s, info_dict = spi.odeint(f, s0, t, args=(b0,f0),
                                full_output=True,
                                hmin=1e-14,
                                mxhnil=0) # solve the equation with 50000 steps

    # Put into pandas
    df_traj = pd.DataFrame(s, index=t, columns=['x'])

    # Does the sysetm blow up?
    if df_traj.abs().max().max() > 1e3:
        print('System blew up - run new model')
        continue

    # Does the system contain Nan?
    if df_traj.isna().values.any():
        print('System contains Nan value - run new model')
        continue
        
    # Does the system contain inf?
    if np.isinf(df_traj.values).any():
        print('System contains Inf value - run new model')
        continue

    # Does the system converge?
    # Difference between max and min of last 10 data points
    diff = df_traj.iloc[-10:-1].max() - df_traj.iloc[-10:-1].min()
    # L2 norm
    norm = np.sqrt(np.square(diff).sum())
    # Define convergence threshold
    conv_thresh = 1e-8
    if norm > conv_thresh:
        print('System does not converge - run new model')
        continue

    # Export equilibrim data
    equi = df_traj.iloc[-1].values
    np.savetxt("output_fold/equi_{}.csv".format(i), equi, delimiter=",")

    # Export parameter data
    pars = np.concatenate([b0,f0])
    np.savetxt("output_fold/pars_{}.csv".format(i), pars, delimiter=",")
