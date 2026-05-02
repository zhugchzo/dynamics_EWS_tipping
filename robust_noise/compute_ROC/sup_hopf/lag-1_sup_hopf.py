import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA


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

        pca = PCA(n_components=1)
        pc_tseries = pca.fit_transform(x_tseries).reshape(-1)

        ac_list = list()

        for winstart in range(test_length):

            Y_t = pc_tseries[winstart : winstart + train_length].reshape(-1, 1)
            Y_tp = pc_tseries[winstart + 1 : winstart + 1 + train_length]

            model = LinearRegression()
            model.fit(Y_t, Y_tp)

            ac = model.coef_[0]

            ac_list.append(ac)

        dic_ac = {'Time':tseries[train_length - 1 : -1],'Lag-1 AC':ac_list}

        csv_out = pd.DataFrame(dic_ac)
        csv_out.to_csv('../../../results/noise/sup_hopf/ews/{}/lag-1_sup_hopf_{}.csv'.format(sigma,random_seed),header = True)

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

        pca = PCA(n_components=1)
        pc_tseries = pca.fit_transform(x_tseries).reshape(-1)

        ac_list = list()

        for winstart in range(test_length):

            Y_t = pc_tseries[winstart : winstart + train_length].reshape(-1, 1)
            Y_tp = pc_tseries[winstart + 1 : winstart + 1 + train_length]

            model = LinearRegression()
            model.fit(Y_t, Y_tp)

            ac = model.coef_[0]

            ac_list.append(ac)

        dic_ac = {'Time':tseries[train_length - 1 : -1],'Lag-1 AC':ac_list}

        csv_out = pd.DataFrame(dic_ac)
        csv_out.to_csv('../../../results/noise/sup_hopf/ews/{}/lag-1_sup_hopf_null_{}.csv'.format(sigma,random_seed),header = True)