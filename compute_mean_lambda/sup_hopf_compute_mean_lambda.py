import pandas as pd
import numpy as np

def robust_mean_iqr(row):
    q1 = row.quantile(0.1)
    q3 = row.quantile(0.9)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    row_filtered = row[(row >= lower) & (row <= upper)]
    return row_filtered.mean()

def robust_std_iqr(row):
    q1 = row.quantile(0.1)
    q3 = row.quantile(0.9)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    row_filtered = row[(row >= lower) & (row <= upper)]
    return row_filtered.std(ddof=1)

def robust_count_iqr(row):
    q1 = row.quantile(0.1)
    q3 = row.quantile(0.9)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    row_filtered = row[(row >= lower) & (row <= upper)]
    return row_filtered.count()

df_esti_re1_sum = pd.DataFrame()
df_esti_re2_sum = pd.DataFrame()
df_esti_im1_sum = pd.DataFrame()
df_esti_im2_sum = pd.DataFrame()
df_esti_stable_sum = pd.DataFrame()

df_esti_re_v1_sum = pd.DataFrame()
df_esti_re_v2_sum = pd.DataFrame()

for random_seed in range(200):

    df_esti_re1 = pd.read_csv('../results/ROC/sup_hopf/ews/dynamics_sup_hopf_{}.csv'.format(random_seed))[['esti_rrate']]
    df_esti_re2 = pd.read_csv('../results/ROC/sup_hopf/ews/dynamics_sup_hopf_{}.csv'.format(random_seed))[['esti_real_eval2']]
    df_esti_im1 = pd.read_csv('../results/ROC/sup_hopf/ews/dynamics_sup_hopf_{}.csv'.format(random_seed))[['esti_imag_eval1']]
    df_esti_im2 = pd.read_csv('../results/ROC/sup_hopf/ews/dynamics_sup_hopf_{}.csv'.format(random_seed))[['esti_imag_eval2']]

    df_esti_stable = pd.read_csv('../results/ROC/sup_hopf/stable/sup_hopf_stable_{}.csv'.format(random_seed))[['stable_expnt']]

    df_esti_re_v1 = pd.read_csv('../results/ROC/sup_hopf/ews_singlev/dynamics_sup_hopf_{}.csv'.format(random_seed))[['esti_rrate1']]
    df_esti_re_v2 = pd.read_csv('../results/ROC/sup_hopf/ews_singlev/dynamics_sup_hopf_{}.csv'.format(random_seed))[['esti_rrate2']]

    df_esti_re1_sum = pd.concat([df_esti_re1_sum.reset_index(drop=True),df_esti_re1.reset_index(drop=True)], axis=1)
    df_esti_re2_sum = pd.concat([df_esti_re2_sum.reset_index(drop=True),df_esti_re2.reset_index(drop=True)], axis=1)
    df_esti_im1_sum = pd.concat([df_esti_im1_sum.reset_index(drop=True),df_esti_im1.reset_index(drop=True)], axis=1)
    df_esti_im2_sum = pd.concat([df_esti_im2_sum.reset_index(drop=True),df_esti_im2.reset_index(drop=True)], axis=1)

    df_esti_stable_sum = pd.concat([df_esti_stable_sum.reset_index(drop=True),df_esti_stable.reset_index(drop=True)], axis=1)

    df_esti_re_v1_sum = pd.concat([df_esti_re_v1_sum.reset_index(drop=True),df_esti_re_v1.reset_index(drop=True)], axis=1)
    df_esti_re_v2_sum = pd.concat([df_esti_re_v2_sum.reset_index(drop=True),df_esti_re_v2.reset_index(drop=True)], axis=1)

# 按行均值
esti_re1_mean = df_esti_re1_sum.mean(axis=1)
esti_re2_mean = df_esti_re2_sum.mean(axis=1)
esti_im1_mean = df_esti_im1_sum.mean(axis=1)
esti_im2_mean = df_esti_im2_sum.mean(axis=1)

esti_stable_mean = df_esti_stable_sum.apply(robust_mean_iqr, axis=1)

esti_re_v1_mean = df_esti_re_v1_sum.mean(axis=1)
esti_re_v2_mean = df_esti_re_v2_sum.mean(axis=1)

# 按行标准差
esti_re1_std = df_esti_re1_sum.std(axis=1, ddof=1)
esti_re2_std = df_esti_re2_sum.std(axis=1, ddof=1)
esti_im1_std = df_esti_im1_sum.std(axis=1, ddof=1)
esti_im2_std = df_esti_im2_sum.std(axis=1, ddof=1)

esti_stable_std = df_esti_stable_sum.apply(robust_std_iqr, axis=1)

esti_re_v1_std = df_esti_re_v1_sum.std(axis=1, ddof=1)
esti_re_v2_std = df_esti_re_v2_sum.std(axis=1, ddof=1)

# 每行有效样本数
n_re1 = df_esti_re1_sum.count(axis=1)
n_re2 = df_esti_re2_sum.count(axis=1)
n_im1 = df_esti_im1_sum.count(axis=1)
n_im2 = df_esti_im2_sum.count(axis=1)

n_stable = df_esti_stable_sum.apply(robust_count_iqr, axis=1)

n_re_v1 = df_esti_re_v1_sum.count(axis=1)
n_re_v2 = df_esti_re_v2_sum.count(axis=1)

# 95% 置信区间半宽
esti_re1_ci = 1.96 * esti_re1_std / np.sqrt(n_re1)
esti_re2_ci = 1.96 * esti_re2_std / np.sqrt(n_re2)
esti_im1_ci = 1.96 * esti_im1_std / np.sqrt(n_im1)
esti_im2_ci = 1.96 * esti_im2_std / np.sqrt(n_im2)

esti_stable_ci = 1.96 * esti_stable_std / np.sqrt(n_stable)

esti_re_v1_ci = 1.96 * esti_re_v1_std / np.sqrt(n_re_v1)
esti_re_v2_ci = 1.96 * esti_re_v2_std / np.sqrt(n_re_v2)

# 整理成一个 DataFrame
df_esti_stats = pd.DataFrame({
    'esti_re1_mean': esti_re1_mean,
    'esti_re1_ci_lower': esti_re1_mean - esti_re1_ci,
    'esti_re1_ci_upper': esti_re1_mean + esti_re1_ci,
    'esti_re2_mean': esti_re2_mean,
    'esti_re2_ci_lower': esti_re2_mean - esti_re2_ci,
    'esti_re2_ci_upper': esti_re2_mean + esti_re2_ci,
    'esti_im1_mean': esti_im1_mean,
    'esti_im1_ci_lower': esti_im1_mean - esti_im1_ci,
    'esti_im1_ci_upper': esti_im1_mean + esti_im1_ci,
    'esti_im2_mean': esti_im2_mean,
    'esti_im2_ci_lower': esti_im2_mean - esti_im2_ci,
    'esti_im2_ci_upper': esti_im2_mean + esti_im2_ci
})

df_esti_stats.insert(0, 'Time', range(len(df_esti_stats)))

df_esti_stats.to_csv('../results/mean_lambda/sup_hopf_mean_lambda.csv', index=False)

df_esti_stab = pd.DataFrame({
    'esti_stable_mean': esti_stable_mean,
    'esti_stable_ci_lower': esti_stable_mean - esti_stable_ci,
    'esti_stable_ci_upper': esti_stable_mean + esti_stable_ci
})

df_esti_stab.insert(0, 'Time', range(len(df_esti_stab)))

df_esti_stab.to_csv('../results/mean_lambda/sup_hopf_mean_stable.csv', index=False)

df_esti_sv_stats = pd.DataFrame({
    'esti_re_v1_mean': esti_re_v1_mean,
    'esti_re_v1_ci_lower': esti_re_v1_mean - esti_re_v1_ci,
    'esti_re_v1_ci_upper': esti_re_v1_mean + esti_re_v1_ci,
    'esti_re_v2_mean': esti_re_v2_mean,
    'esti_re_v2_ci_lower': esti_re_v2_mean - esti_re_v2_ci,
    'esti_re_v2_ci_upper': esti_re_v2_mean + esti_re_v2_ci
})

df_esti_sv_stats.insert(0, 'Time', range(len(df_esti_sv_stats)))

df_esti_sv_stats.to_csv('../results/mean_lambda/singlev/sup_hopf_mean_lambda.csv', index=False)

###########################################################################################

df_null_esti_sum = pd.DataFrame()

for random_seed in range(200):

    df_null_esti = pd.read_csv('../results/ROC/sup_hopf/ews/dynamics_sup_hopf_null_{}.csv'.format(random_seed))[['esti_rrate']]
    df_null_esti_sum = pd.concat([df_null_esti_sum.reset_index(drop=True),df_null_esti.reset_index(drop=True)], axis=1)

# 按行均值
esti_mean = df_null_esti_sum.mean(axis=1)

# 按行标准差
esti_std = df_null_esti_sum.std(axis=1, ddof=1)

# 每行有效样本数
n = df_null_esti_sum.count(axis=1)

# 95% 置信区间半宽
esti_ci = 1.96 * esti_std / np.sqrt(n)

# 整理成一个 DataFrame
df_null_esti_stats = pd.DataFrame({
    'esti_mean': esti_mean,
    'esti_ci_lower': esti_mean - esti_ci,
    'esti_ci_upper': esti_mean + esti_ci,
})

df_null_esti_stats.insert(0, 'Time', range(len(df_null_esti_stats)))

df_null_esti_stats.to_csv('../results/mean_lambda/sup_hopf_null_mean_lambda.csv', index=False)