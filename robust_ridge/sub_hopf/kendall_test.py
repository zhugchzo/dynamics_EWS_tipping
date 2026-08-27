import pandas as pd
import os
from scipy.stats import kendalltau

if not os.path.exists('../../results/ridge/sub_hopf/tau'):
    os.makedirs('../../results/ridge/sub_hopf/tau')

for sigma in [0.01, 0.02, 0.03, 0.04, 0.05]:

    for ridge_param in [1e-3, 5e-3, 1e-2, 5e-2, 1e-1]:

        count_list = list()
        tau_list = list()
        tau_null_list = list()

        count = 0

        for random_seed in range(50):

            df = pd.read_csv('../../results/ridge/sub_hopf/ews/{}/{}/dynamics_sub_hopf_{}.csv'.format(sigma,ridge_param,random_seed))
            df_null = pd.read_csv('../../results/ridge/sub_hopf/ews/{}/{}/dynamics_sub_hopf_null_{}.csv'.format(sigma,ridge_param,random_seed))

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

        csv_out.to_csv('../../results/ridge/sub_hopf/tau/dynamics_tau_{}_{}.csv'.format(sigma,ridge_param),header = True)
        csv_out_null.to_csv('../../results/ridge/sub_hopf/tau/dynamics_tau_null_{}_{}.csv'.format(sigma,ridge_param),header = True)