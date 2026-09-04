import pandas as pd
from scipy.stats import kendalltau

count_list = list()
tau_list = list()
tau_null_list = list()

count = 0

for i in range(3):

    for random_seed in range(200):

        df = pd.read_csv('../../results/ROC/3D_hopf/ews/{}/dynamics_3D_hopf_{}.csv'.format(i,random_seed))
        df_null = pd.read_csv('../../results/ROC/3D_hopf/ews/{}/dynamics_3D_hopf_null_{}.csv'.format(i,random_seed))

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

csv_out.to_csv('../../results/ROC/3D_hopf/dynamics_tau.csv',header = True)
csv_out_null.to_csv('../../results/ROC/3D_hopf/dynamics_tau_null.csv',header = True)