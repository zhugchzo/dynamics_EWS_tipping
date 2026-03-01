import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

fold_BA_dynamics = pd.read_csv('../../results/ROC/fold_BA/dynamics_tau.csv')
fold_BA_null_dynamics = pd.read_csv('../../results/ROC/fold_BA/dynamics_tau_null.csv')

fold_BA_ac = pd.read_csv('../../results/ROC/fold_BA/lag-1_tau.csv')
fold_BA_null_ac = pd.read_csv('../../results/ROC/fold_BA/lag-1_tau_null.csv')

fold_BA_dev = pd.read_csv('../../results/ROC/fold_BA/DEV_tau.csv')
fold_BA_null_dev = pd.read_csv('../../results/ROC/fold_BA/DEV_tau_null.csv')

################################################################################################

dynamics_tau_BA = fold_BA_dynamics['tau'].to_numpy()
null_dynamics_tau_BA = fold_BA_null_dynamics['tau'].to_numpy()

ac_tau_BA = fold_BA_ac['tau'].to_numpy()
null_ac_tau_BA = fold_BA_null_ac['tau'].to_numpy()

dev_tau_BA = fold_BA_dev['tau'].to_numpy()
null_dev_tau_BA = fold_BA_null_dev['tau'].to_numpy()

################################################################################################

n = 10
label = np.concatenate((np.ones(n), np.zeros(n)), axis=0)

dynamics_auc_list = []
ac_auc_list = []
dev_auc_list = []

################################################################################################

network_size = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60]

for i in range(len(network_size)):

    dynamics_tau_BA_i = dynamics_tau_BA[n*i:n*(i+1)]
    null_dynamics_tau_BA_i = null_dynamics_tau_BA[n*i:n*(i+1)]

    ac_tau_BA_i = ac_tau_BA[n*i:n*(i+1)]
    null_ac_tau_BA_i = null_ac_tau_BA[n*i:n*(i+1)]

    dev_tau_BA_i = dev_tau_BA[n*i:n*(i+1)]
    null_dev_tau_BA_i = null_dev_tau_BA[n*i:n*(i+1)]

    dynamics_BA_i = np.concatenate((dynamics_tau_BA_i, null_dynamics_tau_BA_i), axis=0)
    ac_BA_i = np.concatenate((ac_tau_BA_i, null_ac_tau_BA_i), axis=0)
    dev_BA_i = np.concatenate((dev_tau_BA_i, null_dev_tau_BA_i), axis=0)

    fpr_dynamics_BA_i, tpr_dynamics_BA_i, thresholds_dynamics_BA_i = roc_curve(label, dynamics_BA_i)
    auc_dynamics_BA_i = auc(fpr_dynamics_BA_i, tpr_dynamics_BA_i)

    fpr_ac_BA_i, tpr_ac_BA_i, thresholds_ac_BA_i = roc_curve(label, ac_BA_i)
    auc_ac_BA_i = auc(fpr_ac_BA_i, tpr_ac_BA_i)

    fpr_dev_BA_i, tpr_dev_BA_i, thresholds_dev_BA_i = roc_curve(label, dev_BA_i)
    auc_dev_BA_i = auc(fpr_dev_BA_i, tpr_dev_BA_i)

    dynamics_auc_list.append(auc_dynamics_BA_i)
    ac_auc_list.append(auc_ac_BA_i)
    dev_auc_list.append(auc_dev_BA_i)

dic = {'network_size':network_size, 'dynamics_auc':dynamics_auc_list, 'ac_auc':ac_auc_list, 'dev_auc':dev_auc_list}
csv_out = pd.DataFrame(dic)

csv_out.to_csv('../../results/ROC/fold_BA/auc.csv',header = True)
