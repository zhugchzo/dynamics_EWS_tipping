import pandas as pd
import numpy as np

count_list = list()
ratio_zero_list = list()

for random_seed in range(100):

    df = pd.read_csv('../../results/ROC/fold_ER/ews/dynamics_fold_ER_{}.csv'.format(random_seed))

    imag = df['esti_imag_dom_eval'].to_numpy()
    ratio_zero = (imag == 0).mean()

    count_list.append(random_seed)
    ratio_zero_list.append(ratio_zero)

dic_tau = {'Count':count_list, 'steady_rate':ratio_zero_list}

csv_out = pd.DataFrame(dic_tau)

csv_out.to_csv('../../results/ROC/fold_ER/dynamics_rate.csv',header = True)

################################################################################################

count_list = list()
ratio_zero_list = list()

for random_seed in range(100):

    df = pd.read_csv('../../results/ROC/fold_ER/ews/DEV_fold_ER_{}.csv'.format(random_seed))

    imag = df['imag'].to_numpy()
    ratio_zero = (imag == 0).mean()

    count_list.append(random_seed)
    ratio_zero_list.append(ratio_zero)

dic_tau = {'Count':count_list, 'steady_rate':ratio_zero_list}

csv_out = pd.DataFrame(dic_tau)

csv_out.to_csv('../../results/ROC/fold_ER/DEV_rate.csv',header = True)






