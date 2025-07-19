import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression

if not os.path.exists('../../results/ROC/fold/ews'):
    os.makedirs('../../results/ROC/fold/ews')

length = 1000

train_length = 500 # Length of the rolling window
test_length = length - train_length

for random_seed in range(100):

    df_tseries = pd.read_csv('../../data/fold/ROC/fold_{}.csv'.format(random_seed))
    keep_col_tseries = ['traj_x','Time']
    new_f_tseries = df_tseries[keep_col_tseries]
    values_tseries = new_f_tseries.values

    x_tseries = np.array(values_tseries[:,0])
    tseries = np.array(values_tseries[:,1])

    ac_list = list()

    for winstart in range(test_length):

        Y_t = x_tseries[winstart : winstart + train_length].reshape(-1, 1)
        Y_tp = x_tseries[winstart + 1 : winstart + 1 + train_length]

        model = LinearRegression()
        model.fit(Y_t, Y_tp)

        ac = model.coef_[0]

        ac_list.append(ac)

    dic_ac = {'Time':tseries[train_length - 1 : -1],'Lag-1 AC':ac_list}

    csv_out = pd.DataFrame(dic_ac)
    csv_out.to_csv('../../results/ROC/fold/ews/lag-1_fold_{}.csv'.format(random_seed),header = True)

################################################################################################

for random_seed in range(100):

    df_tseries = pd.read_csv('../../data/fold/ROC/fold_null_{}.csv'.format(random_seed))
    keep_col_tseries = ['traj_x','Time']
    new_f_tseries = df_tseries[keep_col_tseries]
    values_tseries = new_f_tseries.values

    x_tseries = np.array(values_tseries[:,0])
    tseries = np.array(values_tseries[:,1])

    ac_list = list()

    for winstart in range(test_length):

        Y_t = x_tseries[winstart : winstart + train_length].reshape(-1, 1)
        Y_tp = x_tseries[winstart + 1 : winstart + 1 + train_length]

        model = LinearRegression()
        model.fit(Y_t, Y_tp)

        ac = model.coef_[0]

        ac_list.append(ac)

    dic_ac = {'Time':tseries[train_length - 1 : -1],'Lag-1 AC':ac_list}

    csv_out = pd.DataFrame(dic_ac)
    csv_out.to_csv('../../results/ROC/fold/ews/lag-1_fold_null_{}.csv'.format(random_seed),header = True)