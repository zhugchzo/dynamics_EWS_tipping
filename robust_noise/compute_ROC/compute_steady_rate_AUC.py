import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

fold_ER_dynamics_rate = pd.read_csv('../../results/noise/fold_ER/dynamics_rate.csv')
fold_BA_dynamics_rate = pd.read_csv('../../results/noise/fold_BA/dynamics_rate.csv')
sup_hopf_dynamics_rate = pd.read_csv('../../results/noise/sup_hopf/dynamics_rate.csv')
trans_dynamics_rate = pd.read_csv('../../results/noise/trans/dynamics_rate.csv')
sub_hopf_dynamics_rate = pd.read_csv('../../results/noise/sub_hopf/dynamics_rate.csv')

fold_ER_dev_rate = pd.read_csv('../../results/noise/fold_ER/DEV_rate.csv')
fold_BA_dev_rate = pd.read_csv('../../results/noise/fold_BA/DEV_rate.csv')
sup_hopf_dev_rate = pd.read_csv('../../results/noise/sup_hopf/DEV_rate.csv')
trans_dev_rate = pd.read_csv('../../results/noise/trans/DEV_rate.csv')
sub_hopf_dev_rate = pd.read_csv('../../results/noise/sub_hopf/DEV_rate.csv')

################################################################################################

dynamics_rate_ER = fold_ER_dynamics_rate['steady_rate'].to_numpy()
dynamics_rate_BA = fold_BA_dynamics_rate['steady_rate'].to_numpy()
dynamics_rate_sup_hopf = sup_hopf_dynamics_rate['steady_rate'].to_numpy()
dynamics_rate_trans = trans_dynamics_rate['steady_rate'].to_numpy()
dynamics_rate_sub_hopf = sub_hopf_dynamics_rate['steady_rate'].to_numpy()

dev_rate_ER = fold_ER_dev_rate['steady_rate'].to_numpy()
dev_rate_BA = fold_BA_dev_rate['steady_rate'].to_numpy()
dev_rate_sup_hopf = sup_hopf_dev_rate['steady_rate'].to_numpy()
dev_rate_trans = trans_dev_rate['steady_rate'].to_numpy()
dev_rate_sub_hopf = sub_hopf_dev_rate['steady_rate'].to_numpy()

################################################################################################

n = 50
label = np.concatenate((np.ones(n), np.ones(n), np.ones(n), np.zeros(n), np.zeros(n)), axis=0)

dynamics_auc_list = []
dev_auc_list = []

################################################################################################

sigma = [0.0025, 0.01, 0.02, 0.03, 0.04, 0.05]

for i in range(len(sigma)):

    dynamics_rate_ER_i = dynamics_rate_ER[n*i:n*(i+1)]
    dynamics_rate_BA_i = dynamics_rate_BA[n*i:n*(i+1)]
    dynamics_rate_trans_i = dynamics_rate_trans[n*i:n*(i+1)]
    dynamics_rate_sup_hopf_i = dynamics_rate_sup_hopf[n*i:n*(i+1)]
    dynamics_rate_sub_hopf_i = dynamics_rate_sub_hopf[n*i:n*(i+1)]

    dev_rate_ER_i = dev_rate_ER[n*i:n*(i+1)]
    dev_rate_BA_i = dev_rate_BA[n*i:n*(i+1)]
    dev_rate_trans_i = dev_rate_trans[n*i:n*(i+1)]
    dev_rate_sup_hopf_i = dev_rate_sup_hopf[n*i:n*(i+1)]
    dev_rate_sub_hopf_i = dev_rate_sub_hopf[n*i:n*(i+1)]

    dynamics_i = np.concatenate((dynamics_rate_ER_i, dynamics_rate_BA_i, dynamics_rate_trans_i, dynamics_rate_sup_hopf_i, dynamics_rate_sub_hopf_i), axis=0)
    dev_i = np.concatenate((dev_rate_ER_i, dev_rate_BA_i, dev_rate_trans_i, dev_rate_sup_hopf_i, dev_rate_sub_hopf_i), axis=0)

    fpr_dynamics_i, tpr_dynamics_i, thresholds_dynamics_i = roc_curve(label, dynamics_i)
    auc_dynamics_i = auc(fpr_dynamics_i, tpr_dynamics_i)

    fpr_dev_i, tpr_dev_i, thresholds_dev_i = roc_curve(label, dev_i)
    auc_dev_i = auc(fpr_dev_i, tpr_dev_i)

    dynamics_auc_list.append(auc_dynamics_i)
    dev_auc_list.append(auc_dev_i)

dic = {'sigma':sigma, 'dynamics_auc':dynamics_auc_list, 'dev_auc':dev_auc_list}
csv_out = pd.DataFrame(dic)

csv_out.to_csv('../../results/noise/auc_steady.csv',header = True)