import pandas as pd

count_list = list()
ratio_zero_list = list()

count = 0

for network_size in [15, 20, 25, 30, 35, 40, 45, 50, 55, 60]:

    for random_seed in range(20):

        df = pd.read_csv('../../results/ROC/fold_BA/ews/{}/dynamics_fold_BA_{}.csv'.format(network_size,random_seed))

        imag = df['esti_imag_dom_eval'].to_numpy()
        ratio_zero = (imag == 0).mean()

        count_list.append(count)
        ratio_zero_list.append(ratio_zero)

        count += 1

dic_tau = {'Count':count_list, 'steady_rate':ratio_zero_list}

csv_out = pd.DataFrame(dic_tau)

csv_out.to_csv('../../results/ROC/fold_BA/dynamics_rate.csv',header = True)

################################################################################################

count_list = list()
ratio_zero_list = list()

count = 0

for network_size in [15, 20, 25, 30, 35, 40, 45, 50, 55, 60]:

    for random_seed in range(20):

        df = pd.read_csv('../../results/ROC/fold_BA/ews/{}/DEV_fold_BA_{}.csv'.format(network_size,random_seed))

        imag = df['imag'].to_numpy()
        ratio_zero = (imag == 0).mean()

        count_list.append(count)
        ratio_zero_list.append(ratio_zero)

        count += 1

dic_tau = {'Count':count_list, 'steady_rate':ratio_zero_list}

csv_out = pd.DataFrame(dic_tau)

csv_out.to_csv('../../results/ROC/fold_BA/DEV_rate.csv',header = True)