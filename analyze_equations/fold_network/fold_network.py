import numpy as np
import pandas as pd
import os

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)

# Get the directory containing the current file
current_dir = os.path.dirname(current_file_path)

# Change the working directory to the directory of the current file
os.chdir(current_dir)

# Make folders for this directory
if not os.path.exists('output_fold_network'):
    os.makedirs('output_fold_network')

N = 10

df = pd.read_csv('../../data/fold_network/fold_network_train.csv')

states_col = list()

for node in range(N):
    states_col.append('equix{}'.format(node))

bif_pars_col = ['b']

bif_pars = df[bif_pars_col].values[499:999]
states = df[states_col].values[499:999]

f_coef = np.zeros((len(bif_pars),N,7+(N-1)))

for node in range(N):

    W_out = pd.read_csv('../../results/fold_network/W_out_{}.csv'.format(node))

    f_elements = ['cte', 'x_{}'.format(node), 'b', 'x_{}^2'.format(node), 'bx_{}'.format(node), 'x_{}^3'.format(node), 'bx_{}^2'.format(node)]
    interaction = [i for i in range(N)]
    interaction.remove(node)
    
    for inter in interaction:
        f_elements.append('x_{}'.format(inter))

    f_coef[:,node,:] = W_out[f_elements].values

for i in range(len(bif_pars)):

    b = np.array([4.99])
    W = f_coef[i]

    ## Simulate the system
    dt = 0.01
    t = np.arange(0, 500, dt)

    s = np.zeros((len(t),N))
    s[0] = states[i]

    for j in range(len(t)-1):

        delta_s = np.zeros(len(s[j]))

        for node in range(N):

            s_inter = np.delete(s[j], node)
            delta_s[node] = W[node] @ np.concatenate([np.array([1, s[j,node], b[0], s[j,node]**2, b[0]*s[j,node], s[j,node]**3, b[0]*s[j,node]**2]) , s_inter])

        s[j+1] = s[j] + delta_s*dt

    node_columns = list()

    for node in range(N):
        node_columns.append('x_{}'.format(node))

    # Put into pandas
    df_traj = pd.DataFrame(s, index=t, columns=node_columns)

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
    np.savetxt("output_fold_network/equi_{}.csv".format(i), equi, delimiter=",")

    # Export parameter data
    flattened_W = W.flatten()
    pars = np.concatenate([b,flattened_W])
    np.savetxt("output_fold_network/pars_{}.csv".format(i), pars, delimiter=",")
