import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

if not os.path.exists('../../results/ROC/trans/ews'):
    os.makedirs('../../results/ROC/trans/ews')

length = 1000

train_length = 950 # Length of the rolling window
test_length = length - train_length

for random_seed in range(200):

    df_tseries = pd.read_csv('../../data/trans/ROC/trans_{}.csv'.format(random_seed))

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
    csv_out.to_csv('../../results/ROC/trans/ews/lag-1_trans_{}.csv'.format(random_seed),header = True)

################################################################################################

for random_seed in range(200):

    df_tseries = pd.read_csv('../../data/trans/ROC/trans_null_{}.csv'.format(random_seed))

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
    csv_out.to_csv('../../results/ROC/trans/ews/lag-1_trans_null_{}.csv'.format(random_seed),header = True)