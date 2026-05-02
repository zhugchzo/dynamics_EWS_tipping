import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression

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

for sigma in [0.0025, 0.01, 0.02, 0.03, 0.04, 0.05]:

    if sigma in [0.0025, 0.01]:
        train_length = 500
    else:
        train_length = 900
    test_length = length - train_length

    if not os.path.exists('../../../results/noise/sup_hopf/ews/{}'.format(sigma)):
        os.makedirs('../../../results/noise/sup_hopf/ews/{}'.format(sigma))

    for random_seed in range(50):

        df_tseries = pd.read_csv('../../../data/sup_hopf/noise/{}/sup_hopf_{}.csv'.format(sigma,random_seed))

        keep_col_tseries = ['Time','traj_x','traj_y']

        new_f_tseries = df_tseries[keep_col_tseries]
        values_tseries = new_f_tseries.values

        tseries = np.array(values_tseries[:,0])
        x_tseries = np.array(values_tseries[:,-2:])

        DEV_list = list()
        imag_list = list()

        for winstart in range(0,test_length,10):

            Y_t = x_tseries[winstart : winstart + train_length]
            Y_tp = x_tseries[winstart + 1 : winstart + 1 + train_length]

            Jaco_matrix = np.zeros((2,2))

            for dim in range(2):

                J_i = smap(Y_t,Y_tp[:,dim])

                Jaco_matrix[dim,:] = J_i

            eigenvalues = np.linalg.eigvals(Jaco_matrix)

            idx = np.argmax(np.abs(eigenvalues))
            lam = eigenvalues[idx]

            DEV = np.abs(lam)
            Im = lam.imag  

            DEV_list.append(DEV)
            imag_list.append(Im)

        dic_DEV = {'Time':tseries[train_length - 1 : -1][::10],'DEV':DEV_list, 'imag':imag_list}

        csv_out = pd.DataFrame(dic_DEV)
        csv_out.to_csv('../../../results/noise/sup_hopf/ews/{}/DEV_sup_hopf_{}.csv'.format(sigma,random_seed),header = True)

################################################################################################

for sigma in [0.0025, 0.01, 0.02, 0.03, 0.04, 0.05]:

    if sigma in [0.0025, 0.01]:
        train_length = 500
    else:
        train_length = 900
    test_length = length - train_length
    
    for random_seed in range(50):

        df_tseries = pd.read_csv('../../../data/sup_hopf/noise/{}/sup_hopf_null_{}.csv'.format(sigma,random_seed))

        keep_col_tseries = ['Time','traj_x','traj_y']

        new_f_tseries = df_tseries[keep_col_tseries]
        values_tseries = new_f_tseries.values

        tseries = np.array(values_tseries[:,0])
        x_tseries = np.array(values_tseries[:,-2:])

        DEV_list = list()


        for winstart in range(0,test_length,10):

            Y_t = x_tseries[winstart : winstart + train_length]
            Y_tp = x_tseries[winstart + 1 : winstart + 1 + train_length]

            Jaco_matrix = np.zeros((2,2))

            for dim in range(2):

                J_i = smap(Y_t,Y_tp[:,dim])

                Jaco_matrix[dim,:] = J_i

            eigenvalues = np.linalg.eigvals(Jaco_matrix)

            DEV = np.max(np.abs(eigenvalues))

            DEV_list.append(DEV)

        dic_DEV = {'Time':tseries[train_length - 1 : -1][::10],'DEV':DEV_list}

        csv_out = pd.DataFrame(dic_DEV)
        csv_out.to_csv('../../../results/noise/sup_hopf/ews/{}/DEV_sup_hopf_null_{}.csv'.format(sigma,random_seed),header = True)