import numpy as np
import pandas as pd
import scipy.integrate as spi
import os

# Make folders for this directory
if not os.path.exists('output_sup_hopf'):
    os.makedirs('output_sup_hopf')

df = pd.read_csv('../../results/sup_hopf/sup_hopf.csv')

bif_pars_col = ['b']
states_col = ['equi_x','equi_y']
f1_elements = ['cte_f1', 'x_f1', 'y_f1', 'b_f1', 'x^2_f1', 'xy_f1', 'bx_f1', 'y^2_f1', 'by_f1', 'x^3_f1', 'x^2y_f1', 'bx^2_f1', 'xy^2_f1', 'bxy_f1', 'y^3_f1', 'by^2_f1']
f2_elements = ['cte_f2', 'x_f2', 'y_f2', 'b_f2', 'x^2_f2', 'xy_f2', 'bx_f2', 'y^2_f2', 'by_f2', 'x^3_f2', 'x^2y_f2', 'bx^2_f2', 'xy^2_f2', 'bxy_f2', 'y^3_f2', 'by^2_f2']

bif_pars = df[bif_pars_col].values
states = df[states_col].values
f1_coef = df[f1_elements].values
f2_coef = df[f2_elements].values

initial_theta = df['initial_theta'].values
delta_theta = df['delta_theta'].values

end_theta = initial_theta + 499*delta_theta

def f(s,t,b,f1,f2):

    x = s[0]
    y = s[1]
    b = b[0]

    polys = np.array([1, x, y, b, x**2, x*y, b*x, y**2, b*y, x**3, x**2*y, b*x**2, x*y**2, b*x*y, y**3, b*y**2])

    dydt = np.array([np.dot(f1,polys), np.dot(f2,polys)])
    
    return dydt

for i in range(len(bif_pars)):

    s0 = states[i]
    b0 = np.array([end_theta[i]])
    f1 = f1_coef[i]
    f2 = f2_coef[i]

    ## Simulate the system
    t = np.arange(0, 500, 0.01)
    s, info_dict = spi.odeint(f, s0, t, args=(b0,f1,f2),
                                full_output=True,
                                hmin=1e-14,
                                mxhnil=0) # solve the equation with 50000 steps

    # Put into pandas
    df_traj = pd.DataFrame(s, index=t, columns=['x','y'])

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
    np.savetxt("output_sup_hopf/equi_{}.csv".format(i), equi, delimiter=",")

    # Export parameter data
    pars = np.concatenate([b0,f1,f2])
    np.savetxt("output_sup_hopf/pars_{}.csv".format(i), pars, delimiter=",")
