import pandas as pd
from scipy.stats import kendalltau
import os

if not os.path.exists('tau'):
    os.makedirs('tau')

for theta in [0, 0.5, 1, 1.5, 2, 2.5]:

    count_list = list()
    tau_list = list()
    tau_null_list = list()

    for random_seed in range(200):

        df = pd.read_csv('ROC/fold/{}/DEV_fold_{}.csv'.format(theta,random_seed))
        df_null = pd.read_csv('ROC/fold/{}/DEV_fold_null_{}.csv'.format(theta,random_seed))

        t = df['Time'].to_numpy().reshape(-1)
        DEV = df['DEV'].to_numpy().reshape(-1)
        DEV_null = df_null['DEV'].to_numpy().reshape(-1)

        tau, p_value = kendalltau(t, DEV)
        tau_null, p_value_null = kendalltau(t, DEV_null)

        count_list.append(random_seed)
        tau_list.append(tau)
        tau_null_list.append(tau_null)

    dic_tau = {'Count':count_list, 'tau':tau_list}
    dic_tau_null = {'Count':count_list, 'tau':tau_null_list}

    csv_out = pd.DataFrame(dic_tau)
    csv_out_null = pd.DataFrame(dic_tau_null)

    csv_out.to_csv('tau/fold_{}.csv'.format(theta),header = True)
    csv_out_null.to_csv('tau/fold_null_{}.csv'.format(theta),header = True)

for theta in [0, 0.5, 1, 1.5, 2, 2.5]:

    count_list = list()
    tau_list = list()
    tau_null_list = list()

    for random_seed in range(200):

        df = pd.read_csv('ROC/trans/{}/DEV_trans_{}.csv'.format(theta,random_seed))
        df_null = pd.read_csv('ROC/trans/{}/DEV_trans_null_{}.csv'.format(theta,random_seed))

        t = df['Time'].to_numpy().reshape(-1)
        DEV = df['DEV'].to_numpy().reshape(-1)
        DEV_null = df_null['DEV'].to_numpy().reshape(-1)

        tau, p_value = kendalltau(t, DEV)
        tau_null, p_value_null = kendalltau(t, DEV_null)

        count_list.append(random_seed)
        tau_list.append(tau)
        tau_null_list.append(tau_null)

    dic_tau = {'Count':count_list, 'tau':tau_list}
    dic_tau_null = {'Count':count_list, 'tau':tau_null_list}

    csv_out = pd.DataFrame(dic_tau)
    csv_out_null = pd.DataFrame(dic_tau_null)

    csv_out.to_csv('tau/trans_{}.csv'.format(theta),header = True)
    csv_out_null.to_csv('tau/trans_null_{}.csv'.format(theta),header = True)

for theta in [0, 0.5, 1, 1.5, 2, 2.5]:

    count_list = list()
    tau_list = list()
    tau_null_list = list()

    for random_seed in range(200):

        df = pd.read_csv('ROC/sup_hopf/{}/DEV_sup_hopf_{}.csv'.format(theta,random_seed))
        df_null = pd.read_csv('ROC/sup_hopf/{}/DEV_sup_hopf_null_{}.csv'.format(theta,random_seed))

        t = df['Time'].to_numpy().reshape(-1)
        DEV = df['DEV'].to_numpy().reshape(-1)
        DEV_null = df_null['DEV'].to_numpy().reshape(-1)

        tau, p_value = kendalltau(t, DEV)
        tau_null, p_value_null = kendalltau(t, DEV_null)

        count_list.append(random_seed)
        tau_list.append(tau)
        tau_null_list.append(tau_null)

    dic_tau = {'Count':count_list, 'tau':tau_list}
    dic_tau_null = {'Count':count_list, 'tau':tau_null_list}

    csv_out = pd.DataFrame(dic_tau)
    csv_out_null = pd.DataFrame(dic_tau_null)

    csv_out.to_csv('tau/sup_hopf_{}.csv'.format(theta),header = True)
    csv_out_null.to_csv('tau/sup_hopf_null_{}.csv'.format(theta),header = True)

for theta in [0, 0.5, 1, 1.5, 2, 2.5]:

    count_list = list()
    tau_list = list()
    tau_null_list = list()

    for random_seed in range(200):

        df = pd.read_csv('ROC/sub_hopf/{}/DEV_sub_hopf_{}.csv'.format(theta,random_seed))
        df_null = pd.read_csv('ROC/sub_hopf/{}/DEV_sub_hopf_null_{}.csv'.format(theta,random_seed))

        t = df['Time'].to_numpy().reshape(-1)
        DEV = df['DEV'].to_numpy().reshape(-1)
        DEV_null = df_null['DEV'].to_numpy().reshape(-1)

        tau, p_value = kendalltau(t, DEV)
        tau_null, p_value_null = kendalltau(t, DEV_null)

        count_list.append(random_seed)
        tau_list.append(tau)
        tau_null_list.append(tau_null)

    dic_tau = {'Count':count_list, 'tau':tau_list}
    dic_tau_null = {'Count':count_list, 'tau':tau_null_list}

    csv_out = pd.DataFrame(dic_tau)
    csv_out_null = pd.DataFrame(dic_tau_null)

    csv_out.to_csv('tau/sub_hopf_{}.csv'.format(theta),header = True)
    csv_out_null.to_csv('tau/sub_hopf_null_{}.csv'.format(theta),header = True)

for theta in [0, 0.5, 1, 1.5, 2, 2.5]:

    count_list = list()
    tau_list = list()
    tau_null_list = list()

    count = 0

    for network_size in [15, 20, 25, 30, 35, 40, 45, 50, 55, 60]:

        for random_seed in range(20):

            df = pd.read_csv('ROC/fold_ER/{}/{}/DEV_fold_ER_{}.csv'.format(theta,network_size,random_seed))
            df_null = pd.read_csv('ROC/fold_ER/{}/{}/DEV_fold_ER_null_{}.csv'.format(theta,network_size,random_seed))

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

    csv_out.to_csv('tau/fold_ER_{}.csv'.format(theta),header = True)
    csv_out_null.to_csv('tau/fold_ER_null_{}.csv'.format(theta),header = True)

for theta in [0, 0.5, 1, 1.5, 2, 2.5]:

    count_list = list()
    tau_list = list()
    tau_null_list = list()

    count = 0

    for network_size in [15, 20, 25, 30, 35, 40, 45, 50, 55, 60]:

        for random_seed in range(20):

            df = pd.read_csv('ROC/fold_BA/{}/{}/DEV_fold_BA_{}.csv'.format(theta,network_size,random_seed))
            df_null = pd.read_csv('ROC/fold_BA/{}/{}/DEV_fold_BA_null_{}.csv'.format(theta,network_size,random_seed))

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

    csv_out.to_csv('tau/fold_BA_{}.csv'.format(theta),header = True)
    csv_out_null.to_csv('tau/fold_BA_null_{}.csv'.format(theta),header = True)