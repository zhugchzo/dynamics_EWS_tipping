import pandas as pd
import numpy as np
from scipy.stats import kendalltau

count_list = list()
tau_list = list()
tau_null_list = list()

count_sv_list = list()
tau_sv_list = list()
tau_sv_null_list = list()

count = 0
count_sv = 0 

for network_size in [15, 20, 25, 30, 35, 40, 45, 50, 55, 60]:

    for random_seed in range(20):

        df = pd.read_csv('../../results/ROC/fold_ER/ews/{}/dynamics_fold_ER_{}.csv'.format(network_size,random_seed))
        df_null = pd.read_csv('../../results/ROC/fold_ER/ews/{}/dynamics_fold_ER_null_{}.csv'.format(network_size,random_seed))

        df_sv = pd.read_csv('../../results/ROC/fold_ER/ews_singlev/{}/dynamics_fold_ER_{}.csv'.format(network_size,random_seed))
        df_sv_null = pd.read_csv('../../results/ROC/fold_ER/ews_singlev/{}/dynamics_fold_ER_null_{}.csv'.format(network_size,random_seed))

        t = df['Time'].to_numpy().reshape(-1)
        rrate = df['esti_rrate'].to_numpy().reshape(-1)
        rrate_null = df_null['esti_rrate'].to_numpy().reshape(-1)

        rrate_v1 = df_sv['esti_rrate1'].to_numpy().reshape(-1)
        rrate_v2 = df_sv['esti_rrate2'].to_numpy().reshape(-1)
        rrate_v1_null = df_sv_null['esti_rrate1'].to_numpy().reshape(-1)
        rrate_v2_null = df_sv_null['esti_rrate2'].to_numpy().reshape(-1)

        tau, p_value = kendalltau(t, rrate)
        tau_null, p_value_null = kendalltau(t, rrate_null)

        tau_v1, p_value_v1 = kendalltau(t, rrate_v1)
        tau_v2, p_value_v2 = kendalltau(t, rrate_v2)
        tau_v1_null, p_value_null_v1 = kendalltau(t, rrate_v1_null)
        tau_v2_null, p_value_null_v2 = kendalltau(t, rrate_v2_null)

        count_list.append(count)
        tau_list.append(tau)
        tau_null_list.append(tau_null)

        count += 1

        count_sv_list.append(count_sv)
        tau_sv_list.append(tau_v1)
        tau_sv_null_list.append(tau_v1_null)

        count_sv += 1

        count_sv_list.append(count_sv)
        tau_sv_list.append(tau_v2)
        tau_sv_null_list.append(tau_v2_null)

        count_sv += 1

dic_tau = {'Count':count_list, 'tau':tau_list}
dic_tau_null = {'Count':count_list, 'tau':tau_null_list}

csv_out = pd.DataFrame(dic_tau)
csv_out_null = pd.DataFrame(dic_tau_null)

csv_out.to_csv('../../results/ROC/fold_ER/dynamics_tau.csv',header = True)
csv_out_null.to_csv('../../results/ROC/fold_ER/dynamics_tau_null.csv',header = True)

dic_tau_sv = {'Count':count_sv_list, 'tau':tau_sv_list}
dic_tau_sv_null = {'Count':count_sv_list, 'tau':tau_sv_null_list}

csv_out_sv = pd.DataFrame(dic_tau_sv)
csv_out_sv_null = pd.DataFrame(dic_tau_sv_null)

csv_out_sv.to_csv('../../results/ROC/fold_ER/singlev_tau.csv',header = True)
csv_out_sv_null.to_csv('../../results/ROC/fold_ER/singlev_tau_null.csv',header = True)

################################################################################################

count_part_list = list()
tau_part_list = list()
tau_part_null_list = list()

count_part = 0

for node_ratio in [0.1, 0.2, 0.3, 0.4, 0.5]:

    for network_size in [20, 30, 40, 50, 60]:

        for random_seed in range(20):

            df_part = pd.read_csv('../../results/ROC/fold_ER/ews_partial/{}/{}/dynamics_fold_ER_{}.csv'.format(node_ratio,network_size,random_seed))
            df_part_null = pd.read_csv('../../results/ROC/fold_ER/ews_partial/{}/{}/dynamics_fold_ER_null_{}.csv'.format(node_ratio,network_size,random_seed))

            rrate_part = df_part['esti_rrate'].to_numpy().reshape(-1)
            rrate_part_null = df_part_null['esti_rrate'].to_numpy().reshape(-1)

            tau_part, p_value_part = kendalltau(t, rrate_part)
            tau_null_part, p_value_null_part = kendalltau(t, rrate_part_null)

            count_part_list.append(count_part)
            tau_part_list.append(tau_part)
            tau_part_null_list.append(tau_null_part)

            count_part += 1

dic_tau_part = {'Count':count_part_list, 'tau':tau_part_list}
dic_tau_part_null = {'Count':count_part_list, 'tau':tau_part_null_list}

csv_out_part = pd.DataFrame(dic_tau_part)
csv_out_part_null = pd.DataFrame(dic_tau_part_null)

csv_out_part.to_csv('../../results/ROC/fold_ER/partial_tau.csv',header = True)
csv_out_part_null.to_csv('../../results/ROC/fold_ER/partial_tau_null.csv',header = True)

################################################################################################

count_list = list()
tau_list = list()
tau_null_list = list()

count = 0

for network_size in [15, 20, 25, 30, 35, 40, 45, 50, 55, 60]:

    for random_seed in range(20):

        df = pd.read_csv('../../results/ROC/fold_ER/ews/{}/lag-1_fold_ER_{}.csv'.format(network_size,random_seed))
        df_null = pd.read_csv('../../results/ROC/fold_ER/ews/{}/lag-1_fold_ER_null_{}.csv'.format(network_size,random_seed))

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

csv_out.to_csv('../../results/ROC/fold_ER/lag-1_tau.csv',header = True)
csv_out_null.to_csv('../../results/ROC/fold_ER/lag-1_tau_null.csv',header = True)

################################################################################################

count_list = list()
tau_list = list()
tau_null_list = list()

count = 0

for network_size in [15, 20, 25, 30, 35, 40, 45, 50, 55, 60]:

    for random_seed in range(20):

        df = pd.read_csv('../../results/ROC/fold_ER/ews/{}/DEV_fold_ER_{}.csv'.format(network_size,random_seed))
        df_null = pd.read_csv('../../results/ROC/fold_ER/ews/{}/DEV_fold_ER_null_{}.csv'.format(network_size,random_seed))

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

csv_out.to_csv('../../results/ROC/fold_ER/DEV_tau.csv',header = True)
csv_out_null.to_csv('../../results/ROC/fold_ER/DEV_tau_null.csv',header = True)