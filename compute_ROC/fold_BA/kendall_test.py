import pandas as pd
import numpy as np
from scipy.stats import kendalltau

count_list = list()
tau_list = list()
tau_null_list = list()

count = 0

for network_size in [15, 20, 25, 30, 35, 40, 45, 50, 55, 60]:

    for random_seed in range(10):

        df = pd.read_csv('../../results/ROC/fold_BA/ews/{}/dynamics_fold_BA_{}.csv'.format(network_size,random_seed))
        df_null = pd.read_csv('../../results/ROC/fold_BA/ews/{}/dynamics_fold_BA_null_{}.csv'.format(network_size,random_seed))

        t = df['Time'].to_numpy().reshape(-1)
        rrate = df['esti_rrate'].to_numpy().reshape(-1)
        rrate_null = df_null['esti_rrate'].to_numpy().reshape(-1)

        tau, p_value = kendalltau(t, rrate)
        tau_null, p_value_null = kendalltau(t, rrate_null)

        count_list.append(count)
        tau_list.append(tau)
        tau_null_list.append(tau_null)

        count += 1

dic_tau = {'Count':count_list, 'tau':tau_list}
dic_tau_null = {'Count':count_list, 'tau':tau_null_list}

csv_out = pd.DataFrame(dic_tau)
csv_out_null = pd.DataFrame(dic_tau_null)

csv_out.to_csv('../../results/ROC/fold_BA/dynamics_tau.csv',header = True)
csv_out_null.to_csv('../../results/ROC/fold_BA/dynamics_tau_null.csv',header = True)

################################################################################################

count_list = list()
tau_list = list()
tau_null_list = list()

count = 0

for network_size in [15, 20, 25, 30, 35, 40, 45, 50, 55, 60]:

    for random_seed in range(10):

        df = pd.read_csv('../../results/ROC/fold_BA/ews/{}/lag-1_fold_BA_{}.csv'.format(network_size,random_seed))
        df_null = pd.read_csv('../../results/ROC/fold_BA/ews/{}/lag-1_fold_BA_null_{}.csv'.format(network_size,random_seed))

        t = df['Time'].to_numpy().reshape(-1)
        ac = df['Lag-1 AC'].to_numpy().reshape(-1)
        ac_null = df_null['Lag-1 AC'].to_numpy().reshape(-1)

        tau, p_value = kendalltau(t, ac)
        tau_null, p_value_null = kendalltau(t, ac_null)

        count_list.append(count)
        tau_list.append(tau)
        tau_null_list.append(tau_null)

        count += 1

dic_tau = {'Count':count_list, 'tau':tau_list}
dic_tau_null = {'Count':count_list, 'tau':tau_null_list}

csv_out = pd.DataFrame(dic_tau)
csv_out_null = pd.DataFrame(dic_tau_null)

csv_out.to_csv('../../results/ROC/fold_BA/lag-1_tau.csv',header = True)
csv_out_null.to_csv('../../results/ROC/fold_BA/lag-1_tau_null.csv',header = True)

################################################################################################

count_list = list()
tau_list = list()
tau_null_list = list()

count = 0

for network_size in [15, 20, 25, 30, 35, 40, 45, 50, 55, 60]:

    for random_seed in range(10):

        df = pd.read_csv('../../results/ROC/fold_BA/ews/{}/DEV_fold_BA_{}.csv'.format(network_size,random_seed))
        df_null = pd.read_csv('../../results/ROC/fold_BA/ews/{}/DEV_fold_BA_null_{}.csv'.format(network_size,random_seed))

        t = df['Time'].to_numpy().reshape(-1)
        DEV = df['DEV'].to_numpy().reshape(-1)
        DEV_null = df_null['DEV'].to_numpy().reshape(-1)

        tau, p_value = kendalltau(t, DEV)
        tau_null, p_value_null = kendalltau(t, DEV_null)

        count_list.append(count)
        tau_list.append(tau)
        tau_null_list.append(tau_null)

        count += 1

dic_tau = {'Count':count_list, 'tau':tau_list}
dic_tau_null = {'Count':count_list, 'tau':tau_null_list}

csv_out = pd.DataFrame(dic_tau)
csv_out_null = pd.DataFrame(dic_tau_null)

csv_out.to_csv('../../results/ROC/fold_BA/DEV_tau.csv',header = True)
csv_out_null.to_csv('../../results/ROC/fold_BA/DEV_tau_null.csv',header = True)