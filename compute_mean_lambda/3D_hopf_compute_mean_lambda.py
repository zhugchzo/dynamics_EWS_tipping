import pandas as pd
import numpy as np
import os

if not os.path.exists('../results/mean_lambda/partial'):
    os.makedirs('../results/mean_lambda/partial')

for traj_col in range(3):

    df_esti_re_part_sum = pd.DataFrame()
    df_esti_im1_part_sum = pd.DataFrame()
    df_esti_im2_part_sum = pd.DataFrame()

    for random_seed in range(200):

        df_esti_part_re = pd.read_csv('../results/ROC/3D_hopf/ews/{}/dynamics_3D_hopf_{}.csv'.format(traj_col,random_seed))[['esti_rrate']]
        df_esti_part_im1 = pd.read_csv('../results/ROC/3D_hopf/ews/{}/dynamics_3D_hopf_{}.csv'.format(traj_col,random_seed))[['esti_imag_eval1']]
        df_esti_part_im2 = pd.read_csv('../results/ROC/3D_hopf/ews/{}/dynamics_3D_hopf_{}.csv'.format(traj_col,random_seed))[['esti_imag_eval2']]

        df_esti_re_part_sum = pd.concat([df_esti_re_part_sum.reset_index(drop=True),df_esti_part_re.reset_index(drop=True)], axis=1)
        df_esti_im1_part_sum = pd.concat([df_esti_im1_part_sum.reset_index(drop=True),df_esti_part_im1.reset_index(drop=True)], axis=1)
        df_esti_im2_part_sum = pd.concat([df_esti_im2_part_sum.reset_index(drop=True),df_esti_part_im2.reset_index(drop=True)], axis=1)

    esti_re_part_mean = df_esti_re_part_sum.mean(axis=1)
    esti_im1_part_mean = df_esti_im1_part_sum.mean(axis=1)
    esti_im2_part_mean = df_esti_im2_part_sum.mean(axis=1)

    esti_re_part_std = df_esti_re_part_sum.std(axis=1, ddof=1)
    esti_im1_part_std = df_esti_im1_part_sum.std(axis=1, ddof=1)
    esti_im2_part_std = df_esti_im2_part_sum.std(axis=1, ddof=1)

    n_re_part = df_esti_re_part_sum.count(axis=1)
    n_im1_part = df_esti_im1_part_sum.count(axis=1)
    n_im2_part = df_esti_im2_part_sum.count(axis=1)

    esti_re_part_ci = 1.96 * esti_re_part_std / np.sqrt(n_re_part)
    esti_im1_part_ci = 1.96 * esti_im1_part_std / np.sqrt(n_im1_part)
    esti_im2_part_ci = 1.96 * esti_im2_part_std / np.sqrt(n_im2_part)

    df_esti_part_stats = pd.DataFrame({
        'esti_re_mean': esti_re_part_mean,
        'esti_re_ci_lower': esti_re_part_mean - esti_re_part_ci,
        'esti_re_ci_upper': esti_re_part_mean + esti_re_part_ci,
        'esti_im1_mean': esti_im1_part_mean,
        'esti_im1_ci_lower': esti_im1_part_mean - esti_im1_part_ci,
        'esti_im1_ci_upper': esti_im1_part_mean + esti_im1_part_ci,
        'esti_im2_mean': esti_im2_part_mean,
        'esti_im2_ci_lower': esti_im2_part_mean - esti_im2_part_ci,
        'esti_im2_ci_upper': esti_im2_part_mean + esti_im2_part_ci
    })

    df_esti_part_stats.insert(0, 'Time', range(len(df_esti_part_stats)))

    df_esti_part_stats.to_csv('../results/mean_lambda/partial/3D_hopf_mean_lambda_{}.csv'.format(traj_col), index=False)