import pandas as pd
import numpy as np

count_list = list()
ratio_sup_list = list()

for random_seed in range(200):

    df = pd.read_csv('../../results/ROC/sup_hopf/stable/sup_hopf_stable_{}.csv'.format(random_seed))

    stable = df['stable_expnt'].to_numpy()
    ratio_sup = (stable < 0).mean()

    count_list.append(random_seed)
    ratio_sup_list.append(ratio_sup)

dic_tau = {'Count':count_list, 'sup_rate':ratio_sup_list}

csv_out = pd.DataFrame(dic_tau)

csv_out.to_csv('../../results/ROC/sup_hopf/sup_rate.csv',header = True)