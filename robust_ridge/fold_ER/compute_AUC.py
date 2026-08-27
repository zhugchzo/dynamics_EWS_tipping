import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

n = 50
label = np.concatenate((np.ones(n), np.zeros(n)), axis=0)

sigma_values = [0.01, 0.02, 0.03, 0.04, 0.05]
ridge_values = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1]

# sigma 按降序排列，使其在表格中从下往上递增
auc_matrix = pd.DataFrame(
    index=sigma_values[::-1],
    columns=ridge_values,
    dtype=float
)

for sigma in sigma_values:
    for ridge_param in ridge_values:

        fold_ER_dynamics = pd.read_csv(
            '../../results/ridge/fold_ER/tau/dynamics_tau_{}_{}.csv'.format(
                sigma, ridge_param
            )
        )

        fold_ER_null_dynamics = pd.read_csv(
            '../../results/ridge/fold_ER/tau/dynamics_tau_null_{}_{}.csv'.format(
                sigma, ridge_param
            )
        )

        dynamics_tau_i = fold_ER_dynamics['tau'].to_numpy()
        null_dynamics_tau_i = fold_ER_null_dynamics['tau'].to_numpy()

        dynamics_i = np.concatenate(
            (dynamics_tau_i, null_dynamics_tau_i),
            axis=0
        )

        fpr_dynamics_i, tpr_dynamics_i, _ = roc_curve(label, dynamics_i)
        auc_dynamics_i = auc(fpr_dynamics_i, tpr_dynamics_i)

        auc_matrix.loc[sigma, ridge_param] = auc_dynamics_i

auc_matrix = auc_matrix.sort_index(ascending=False)

# 去掉行轴名和列轴名，但保留行列数值
auc_matrix.index.name = None
auc_matrix.columns.name = None

auc_matrix.to_csv('../../results/ridge/fold_ER/auc.csv',header=True,index=True)