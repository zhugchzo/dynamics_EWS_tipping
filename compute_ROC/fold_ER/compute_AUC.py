import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

fold_ER_dynamics = pd.read_csv('../../results/ROC/fold_ER/dynamics_tau.csv')
fold_ER_null_dynamics = pd.read_csv('../../results/ROC/fold_ER/dynamics_tau_null.csv')

fold_ER_ac = pd.read_csv('../../results/ROC/fold_ER/lag-1_tau.csv')
fold_ER_null_ac = pd.read_csv('../../results/ROC/fold_ER/lag-1_tau_null.csv')

fold_ER_dev = pd.read_csv('../../results/ROC/fold_ER/DEV_tau.csv')
fold_ER_null_dev = pd.read_csv('../../results/ROC/fold_ER/DEV_tau_null.csv')

################################################################################################

dynamics_tau_BA = fold_ER_dynamics['tau'].to_numpy()
null_dynamics_tau_BA = fold_ER_null_dynamics['tau'].to_numpy()

ac_tau_BA = fold_ER_ac['tau'].to_numpy()
null_ac_tau_BA = fold_ER_null_ac['tau'].to_numpy()

dev_tau_BA = fold_ER_dev['tau'].to_numpy()
null_dev_tau_BA = fold_ER_null_dev['tau'].to_numpy()

################################################################################################

n1 = 20
label1 = np.concatenate((np.ones(n1), np.zeros(n1)), axis=0)

dynamics_auc_list = []
ac_auc_list = []
dev_auc_list = []

################################################################################################

network_size = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60]

for i in range(len(network_size)):

    dynamics_tau_BA_i = dynamics_tau_BA[n1*i:n1*(i+1)]
    null_dynamics_tau_BA_i = null_dynamics_tau_BA[n1*i:n1*(i+1)]

    ac_tau_BA_i = ac_tau_BA[n1*i:n1*(i+1)]
    null_ac_tau_BA_i = null_ac_tau_BA[n1*i:n1*(i+1)]

    dev_tau_BA_i = dev_tau_BA[n1*i:n1*(i+1)]
    null_dev_tau_BA_i = null_dev_tau_BA[n1*i:n1*(i+1)]

    dynamics_BA_i = np.concatenate((dynamics_tau_BA_i, null_dynamics_tau_BA_i), axis=0)
    ac_BA_i = np.concatenate((ac_tau_BA_i, null_ac_tau_BA_i), axis=0)
    dev_BA_i = np.concatenate((dev_tau_BA_i, null_dev_tau_BA_i), axis=0)

    fpr_dynamics_BA_i, tpr_dynamics_BA_i, thresholds_dynamics_BA_i = roc_curve(label1, dynamics_BA_i)
    auc_dynamics_BA_i = auc(fpr_dynamics_BA_i, tpr_dynamics_BA_i)

    fpr_ac_BA_i, tpr_ac_BA_i, thresholds_ac_BA_i = roc_curve(label1, ac_BA_i)
    auc_ac_BA_i = auc(fpr_ac_BA_i, tpr_ac_BA_i)

    fpr_dev_BA_i, tpr_dev_BA_i, thresholds_dev_BA_i = roc_curve(label1, dev_BA_i)
    auc_dev_BA_i = auc(fpr_dev_BA_i, tpr_dev_BA_i)

    dynamics_auc_list.append(auc_dynamics_BA_i)
    ac_auc_list.append(auc_ac_BA_i)
    dev_auc_list.append(auc_dev_BA_i)

dic = {'network_size':network_size, 'dynamics_auc':dynamics_auc_list, 'ac_auc':ac_auc_list, 'dev_auc':dev_auc_list}
csv_out = pd.DataFrame(dic)

csv_out.to_csv('../../results/ROC/fold_ER/auc.csv',header = True)

################################################################################################

fold_ER_partial = pd.read_csv('../../results/ROC/fold_ER/partial_tau.csv')
fold_ER_null_partial = pd.read_csv('../../results/ROC/fold_ER/partial_tau_null.csv')

n2 = 100
label2 = np.concatenate((np.ones(n2), np.zeros(n2)), axis=0)

partial_auc_list = []
zero_ratio_list = []

partial_tau_BA = fold_ER_partial['tau'].to_numpy()
null_partial_tau_BA = fold_ER_null_partial['tau'].to_numpy()

################################################################################################

node_ratio = [0.1, 0.2, 0.3, 0.4, 0.5]

for i in range(len(node_ratio)):

    partial_tau_BA_i = partial_tau_BA[n2*i:n2*(i+1)]
    null_partial_tau_BA_i = null_partial_tau_BA[n2*i:n2*(i+1)]

    partial_BA_i = np.concatenate((partial_tau_BA_i, null_partial_tau_BA_i), axis=0)

    fpr_partial_BA_i, tpr_partial_BA_i, thresholds_partial_BA_i = roc_curve(label2, partial_BA_i)
    auc_partial_BA_i = auc(fpr_partial_BA_i, tpr_partial_BA_i)

    partial_auc_list.append(auc_partial_BA_i)

    ############################################################################################

    zero_count = 0
    total_count = 0

    for network_size in [20, 30, 40, 50, 60]:

        for random_seed in range(20):

            df_part = pd.read_csv('../../results/ROC/fold_ER/ews_partial/{}/{}/dynamics_fold_ER_{}.csv'.format(node_ratio[i],network_size,random_seed))

            imag_part = df_part['esti_imag_dom_eval'].to_numpy()

        zero_count += np.sum(imag_part == 0)
        total_count += len(imag_part)

    zero_ratio = zero_count / total_count

    zero_ratio_list.append(zero_ratio)

################################################################################################

dic = {'node_ratio':node_ratio, 'partial_auc':partial_auc_list, 'zero_ratio':zero_ratio_list}
csv_out = pd.DataFrame(dic)

csv_out.to_csv('../../results/ROC/fold_ER/auc_partial.csv',header = True)