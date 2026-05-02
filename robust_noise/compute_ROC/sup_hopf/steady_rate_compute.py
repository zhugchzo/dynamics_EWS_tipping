import pandas as pd
import numpy as np

count_list = list()
ratio_zero_list = list()

count = 0

for sigma in [0.0025, 0.01, 0.02, 0.03, 0.04, 0.05]:

    for random_seed in range(50):

        df = pd.read_csv('../../../results/noise/sup_hopf/ews/{}/dynamics_sup_hopf_{}.csv'.format(sigma,random_seed))

        imag = df['esti_imag_eval1'].to_numpy()
        ratio_zero = (imag == 0).mean()

        count_list.append(count)
        ratio_zero_list.append(ratio_zero)

        count += 1

dic_tau = {'Count':count_list, 'steady_rate':ratio_zero_list}

csv_out = pd.DataFrame(dic_tau)

csv_out.to_csv('../../../results/noise/sup_hopf/dynamics_rate.csv',header = True)

################################################################################################

count_list = list()
ratio_zero_list = list()

count = 0

for sigma in [0.0025, 0.01, 0.02, 0.03, 0.04, 0.05]:

    for random_seed in range(50):

        df = pd.read_csv('../../../results/noise/sup_hopf/ews/{}/DEV_sup_hopf_{}.csv'.format(sigma,random_seed))

        imag = df['imag'].to_numpy()
        ratio_zero = (imag == 0).mean()

        count_list.append(count)
        ratio_zero_list.append(ratio_zero)

        count += 1

dic_tau = {'Count':count_list, 'steady_rate':ratio_zero_list}

csv_out = pd.DataFrame(dic_tau)

csv_out.to_csv('../../../results/noise/sup_hopf/DEV_rate.csv',header = True)