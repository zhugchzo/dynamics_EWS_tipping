import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

sup_hopf_dynamics = pd.read_csv('../../../results/noise/sup_hopf/dynamics_tau.csv')
sup_hopf_null_dynamics = pd.read_csv('../../../results/noise/sup_hopf/dynamics_tau_null.csv')

sup_hopf_ac = pd.read_csv('../../../results/noise/sup_hopf/lag-1_tau.csv')
sup_hopf_null_ac = pd.read_csv('../../../results/noise/sup_hopf/lag-1_tau_null.csv')

sup_hopf_dev = pd.read_csv('../../../results/noise/sup_hopf/DEV_tau.csv')
sup_hopf_null_dev = pd.read_csv('../../../results/noise/sup_hopf/DEV_tau_null.csv')

################################################################################################

dynamics_tau = sup_hopf_dynamics['tau'].to_numpy()
null_dynamics_tau = sup_hopf_null_dynamics['tau'].to_numpy()

ac_tau = sup_hopf_ac['tau'].to_numpy()
null_ac_tau = sup_hopf_null_ac['tau'].to_numpy()

dev_tau = sup_hopf_dev['tau'].to_numpy()
null_dev_tau = sup_hopf_null_dev['tau'].to_numpy()

################################################################################################

n = 50
label = np.concatenate((np.ones(n), np.zeros(n)), axis=0)

dynamics_auc_list = []
ac_auc_list = []
dev_auc_list = []

################################################################################################

sigma = [0.0025, 0.01, 0.02, 0.03, 0.04, 0.05]

for i in range(len(sigma)):

    dynamics_tau_i = dynamics_tau[n*i:n*(i+1)]
    null_dynamics_tau_i = null_dynamics_tau[n*i:n*(i+1)]

    ac_tau_i = ac_tau[n*i:n*(i+1)]
    null_ac_tau_i = null_ac_tau[n*i:n*(i+1)]

    dev_tau_i = dev_tau[n*i:n*(i+1)]
    null_dev_tau_i = null_dev_tau[n*i:n*(i+1)]

    dynamics_i = np.concatenate((dynamics_tau_i, null_dynamics_tau_i), axis=0)
    ac_i = np.concatenate((ac_tau_i, null_ac_tau_i), axis=0)
    dev_i = np.concatenate((dev_tau_i, null_dev_tau_i), axis=0)

    fpr_dynamics_i, tpr_dynamics_i, thresholds_dynamics_i = roc_curve(label, dynamics_i)
    auc_dynamics_i = auc(fpr_dynamics_i, tpr_dynamics_i)

    fpr_ac_i, tpr_ac_i, thresholds_ac_i = roc_curve(label, ac_i)
    auc_ac_i = auc(fpr_ac_i, tpr_ac_i)

    fpr_dev_i, tpr_dev_i, thresholds_dev_i = roc_curve(label, dev_i)
    auc_dev_i = auc(fpr_dev_i, tpr_dev_i)

    dynamics_auc_list.append(auc_dynamics_i)
    ac_auc_list.append(auc_ac_i)
    dev_auc_list.append(auc_dev_i)

dic = {'sigma':sigma, 'dynamics_auc':dynamics_auc_list, 'ac_auc':ac_auc_list, 'dev_auc':dev_auc_list}
csv_out = pd.DataFrame(dic)

csv_out.to_csv('../../../results/noise/sup_hopf/auc.csv',header = True)