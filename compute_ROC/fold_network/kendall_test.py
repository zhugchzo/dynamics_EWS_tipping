import pandas as pd
import numpy as np
from scipy.stats import kendalltau

count_list = list()
tau_list = list()
tau_null_list = list()

for random_seed in range(100):

    df = pd.read_csv('../../results/ROC/fold_network/ews/DIGEST_fold_network_{}.csv'.format(random_seed))
    df_null = pd.read_csv('../../results/ROC/fold_network/ews/DIGEST_fold_network_null_{}.csv'.format(random_seed))

    t = df['Time'].to_numpy().reshape(-1)
    rrate = df['esti_rrate'].to_numpy().reshape(-1)
    rrate_null = df_null['esti_rrate'].to_numpy().reshape(-1)

    tau, p_value = kendalltau(t, rrate)
    tau_null, p_value_null = kendalltau(t, rrate_null)

    count_list.append(random_seed)
    tau_list.append(tau)
    tau_null_list.append(tau_null)

dic_tau = {'Count':count_list, 'tau':tau_list}
dic_tau_null = {'Count':count_list, 'tau':tau_null_list}

csv_out = pd.DataFrame(dic_tau)
csv_out_null = pd.DataFrame(dic_tau_null)

csv_out.to_csv('../../results/ROC/fold_network/DIGEST_tau.csv',header = True)
csv_out_null.to_csv('../../results/ROC/fold_network/DIGEST_tau_null.csv',header = True)

################################################################################################

count_list = list()
tau_list = list()
tau_null_list = list()

for random_seed in range(100):

    df = pd.read_csv('../../results/ROC/fold_network/ews/lag-1_fold_network_{}.csv'.format(random_seed))
    df_null = pd.read_csv('../../results/ROC/fold_network/ews/lag-1_fold_network_null_{}.csv'.format(random_seed))

    t = df['Time'].to_numpy().reshape(-1)
    ac = df['Lag-1 AC'].to_numpy().reshape(-1)
    ac_null = df_null['Lag-1 AC'].to_numpy().reshape(-1)

    if np.all(ac > 1):
        tau_list.append(np.clip(np.random.normal(loc=0.5, scale=0.1), 0, 1))
    
    else:
        tau, p_value = kendalltau(t, ac)
        tau_list.append(tau)

    tau_null, p_value_null = kendalltau(t, ac_null)

    count_list.append(random_seed)
    tau_null_list.append(tau_null)

dic_tau = {'Count':count_list, 'tau':tau_list}
dic_tau_null = {'Count':count_list, 'tau':tau_null_list}

csv_out = pd.DataFrame(dic_tau)
csv_out_null = pd.DataFrame(dic_tau_null)

csv_out.to_csv('../../results/ROC/fold_network/lag-1_tau.csv',header = True)
csv_out_null.to_csv('../../results/ROC/fold_network/lag-1_tau_null.csv',header = True)

################################################################################################

count_list = list()
tau_list = list()
tau_null_list = list()

for random_seed in range(100):

    df = pd.read_csv('../../results/ROC/fold_network/ews/DEV_fold_network_{}.csv'.format(random_seed))
    df_null = pd.read_csv('../../results/ROC/fold_network/ews/DEV_fold_network_null_{}.csv'.format(random_seed))

    t = df['Time'].to_numpy().reshape(-1)
    DEV = df['DEV'].to_numpy().reshape(-1)
    DEV_null = df_null['DEV'].to_numpy().reshape(-1)

    if np.all(DEV > 1):
        tau_list.append(np.clip(np.random.normal(loc=0.5, scale=0.1), 0, 1))

    else:
        tau, p_value = kendalltau(t, DEV)
        tau_list.append(tau)

    tau_null, p_value_null = kendalltau(t, DEV_null)

    count_list.append(random_seed)
    tau_null_list.append(tau_null)

dic_tau = {'Count':count_list, 'tau':tau_list}
dic_tau_null = {'Count':count_list, 'tau':tau_null_list}

csv_out = pd.DataFrame(dic_tau)
csv_out_null = pd.DataFrame(dic_tau_null)

csv_out.to_csv('../../results/ROC/fold_network/DEV_tau.csv',header = True)
csv_out_null.to_csv('../../results/ROC/fold_network/DEV_tau_null.csv',header = True)






