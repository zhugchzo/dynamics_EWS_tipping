import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression

if not os.path.exists('../../results/ROC/fold_network/ews'):
    os.makedirs('../../results/ROC/fold_network/ews')

def smap(X, Y, theta=0.5):

    X = np.asarray(X)
    Y = np.asarray(Y)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    assert len(X) == len(Y), 'X and Y must have the same length'
    n_samples, n_features = X.shape

    i = n_samples - 1

    x_i = X[i]

    dists = np.linalg.norm(X - x_i, axis=1)
    d_mean = np.mean(dists) if np.mean(dists) != 0 else 1.0

    weights = np.ones_like(dists) if theta == 0 else np.exp(-theta * dists / d_mean)

    model = LinearRegression()
    model.fit(X, Y, sample_weight=weights)

    return model.coef_

length = 1000

train_length = 500 # Length of the rolling window
test_length = length - train_length

for random_seed in range(100):

    df_tseries = pd.read_csv('../../data/fold_network/ROC/fold_network_{}.csv'.format(random_seed))
    df_network = pd.read_csv('../../data/fold_network/ROC/network_{}.csv'.format(random_seed), header=None)

    N = len(df_network)

    keep_col_tseries = ['Time']

    for node in range(N):
        keep_col_tseries.append('traj_x{}'.format(node))

    new_f_tseries = df_tseries[keep_col_tseries]
    values_tseries = new_f_tseries.values

    tseries = np.array(values_tseries[:,0])
    x_tseries = np.array(values_tseries[:,-N:])

    DEV_list = list()

    for winstart in range(0,test_length,20):

        Y_t = x_tseries[winstart : winstart + train_length]
        Y_tp = x_tseries[winstart + 1 : winstart + 1 + train_length]

        Jaco_matrix = np.zeros((N,N))

        for node in range(N):

            J_i = smap(Y_t,Y_tp[:,node])

            Jaco_matrix[node,:] = J_i

        eigenvalues = np.linalg.eigvals(Jaco_matrix)

        DEV = np.max(np.abs(eigenvalues))

        DEV_list.append(DEV)

    dic_DEV = {'Time':tseries[train_length - 1 : -1][::20],'DEV':DEV_list}

    csv_out = pd.DataFrame(dic_DEV)
    csv_out.to_csv('../../results/ROC/fold_network/ews/DEV_fold_network_{}.csv'.format(random_seed),header = True)

# ################################################################################################

for random_seed in range(100):

    df_tseries = pd.read_csv('../../data/fold_network/ROC/fold_network_null_{}.csv'.format(random_seed))
    df_network = pd.read_csv('../../data/fold_network/ROC/network_{}.csv'.format(random_seed), header=None)

    N = len(df_network)

    keep_col_tseries = ['Time']

    for node in range(N):
        keep_col_tseries.append('traj_x{}'.format(node))

    new_f_tseries = df_tseries[keep_col_tseries]
    values_tseries = new_f_tseries.values

    tseries = np.array(values_tseries[:,0])
    x_tseries = np.array(values_tseries[:,-N:])

    DEV_list = list()

    for winstart in range(0,test_length,20):

        Y_t = x_tseries[winstart : winstart + train_length]
        Y_tp = x_tseries[winstart + 1 : winstart + 1 + train_length]

        Jaco_matrix = np.zeros((N,N))

        for node in range(N):

            J_i = smap(Y_t,Y_tp[:,node])

            Jaco_matrix[node,:] = J_i

        eigenvalues = np.linalg.eigvals(Jaco_matrix)

        DEV = np.max(np.abs(eigenvalues))

        DEV_list.append(DEV)

    dic_DEV = {'Time':tseries[train_length - 1 : -1][::20],'DEV':DEV_list}

    csv_out = pd.DataFrame(dic_DEV)
    csv_out.to_csv('../../results/ROC/fold_network/ews/DEV_fold_network_null_{}.csv'.format(random_seed),header = True)