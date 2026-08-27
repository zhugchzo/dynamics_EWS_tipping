import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

n1 = 200
label1 = np.concatenate((np.ones(n1), np.zeros(n1)), axis=0)

theta_list = [0, 0.5, 1, 1.5, 2, 2.5]
fold_list = []
trans_list = []
sup_hopf_list = []
sub_hopf_list = []
fold_ER_list = []
fold_BA_list = []

for theta in theta_list:

    fold = pd.read_csv('tau/fold_{}.csv'.format(theta))['tau'].to_numpy()
    fold_null = pd.read_csv('tau/fold_null_{}.csv'.format(theta))['tau'].to_numpy()

    trans = pd.read_csv('tau/trans_{}.csv'.format(theta))['tau'].to_numpy()
    trans_null = pd.read_csv('tau/trans_null_{}.csv'.format(theta))['tau'].to_numpy()

    sup_hopf = pd.read_csv('tau/sup_hopf_{}.csv'.format(theta))['tau'].to_numpy()
    sup_hopf_null = pd.read_csv('tau/sup_hopf_null_{}.csv'.format(theta))['tau'].to_numpy()

    sub_hopf = pd.read_csv('tau/sub_hopf_{}.csv'.format(theta))['tau'].to_numpy()
    sub_hopf_null = pd.read_csv('tau/sub_hopf_null_{}.csv'.format(theta))['tau'].to_numpy()

    fold_ER = pd.read_csv('tau/fold_ER_{}.csv'.format(theta))['tau'].to_numpy()
    fold_ER_null = pd.read_csv('tau/fold_ER_null_{}.csv'.format(theta))['tau'].to_numpy()

    fold_BA = pd.read_csv('tau/fold_BA_{}.csv'.format(theta))['tau'].to_numpy()
    fold_BA_null = pd.read_csv('tau/fold_BA_null_{}.csv'.format(theta))['tau'].to_numpy()

    ################################################################################################

    fold_tau = np.concatenate((fold, fold_null), axis=0)
    trans_tau = np.concatenate((trans, trans_null), axis=0)
    sup_hopf_tau = np.concatenate((sup_hopf, sup_hopf_null), axis=0)
    sub_hopf_tau = np.concatenate((sub_hopf, sub_hopf_null), axis=0)
    fold_ER_tau = np.concatenate((fold_ER, fold_ER_null), axis=0)
    fold_BA_tau = np.concatenate((fold_BA, fold_BA_null), axis=0)

    fpr_fold, tpr_fold, thresholds_fold = roc_curve(label1, fold_tau)
    auc_fold = auc(fpr_fold, tpr_fold)

    fpr_trans, tpr_trans, thresholds_trans = roc_curve(label1, trans_tau)
    auc_trans = auc(fpr_trans, tpr_trans)

    fpr_sup_hopf, tpr_sup_hopf, thresholds_sup_hopf = roc_curve(label1, sup_hopf_tau)
    auc_sup_hopf = auc(fpr_sup_hopf, tpr_sup_hopf)

    fpr_sub_hopf, tpr_sub_hopf, thresholds_sub_hopf = roc_curve(label1, sub_hopf_tau)
    auc_sub_hopf = auc(fpr_sub_hopf, tpr_sub_hopf)

    fpr_fold_ER, tpr_fold_ER, thresholds_fold_ER = roc_curve(label1, fold_ER_tau)
    auc_fold_ER = auc(fpr_fold_ER, tpr_fold_ER)

    fpr_fold_BA, tpr_fold_BA, thresholds_fold_BA = roc_curve(label1, fold_BA_tau)
    auc_fold_BA = auc(fpr_fold_BA, tpr_fold_BA)

    fold_list.append(auc_fold)
    trans_list.append(auc_trans)
    sup_hopf_list.append(auc_sup_hopf)
    sub_hopf_list.append(auc_sub_hopf)
    fold_ER_list.append(auc_fold_ER)
    fold_BA_list.append(auc_fold_BA)

dic = {'theta':theta_list,
    'fold_auc':fold_list, 'trans_auc':trans_list, 'sup_hopf_auc':sup_hopf_list,
    'sub_hopf_auc':sub_hopf_list, 'fold_ER_auc':fold_ER_list, 'fold_BA_auc':fold_BA_list}

csv_out = pd.DataFrame(dic)

csv_out.to_csv('auc.csv',header = True)