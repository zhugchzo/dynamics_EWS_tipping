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

df_esti_re_sum = pd.DataFrame()
df_esti_im_sum = pd.DataFrame()

for random_seed in range(200):

    df_esti_re = pd.read_csv('../results/ROC/trans/ews/dynamics_trans_{}.csv'.format(random_seed))[['esti_rrate']]
    df_esti_im = pd.read_csv('../results/ROC/trans/ews/dynamics_trans_{}.csv'.format(random_seed))[['esti_imag_eval1']]

    df_esti_re_sum = pd.concat([df_esti_re_sum.reset_index(drop=True),df_esti_re.reset_index(drop=True)], axis=1)
    df_esti_im_sum = pd.concat([df_esti_im_sum.reset_index(drop=True),df_esti_im.reset_index(drop=True)], axis=1)

# 按行均值
esti_re_mean = df_esti_re_sum.apply(robust_mean_iqr, axis=1)
esti_im_mean = df_esti_im_sum.apply(robust_mean_iqr, axis=1)

# 按行标准差
esti_re_std = df_esti_re_sum.apply(robust_std_iqr, axis=1)
esti_im_std = df_esti_im_sum.apply(robust_std_iqr, axis=1)

# 每行有效样本数
n_re = df_esti_re_sum.apply(robust_count_iqr, axis=1)
n_im = df_esti_im_sum.apply(robust_count_iqr, axis=1)

# 95% 置信区间半宽
esti_re_ci = 1.96 * esti_re_std / np.sqrt(n_re)
esti_im_ci = 1.96 * esti_im_std / np.sqrt(n_im)

# 整理成一个 DataFrame
df_esti_stats = pd.DataFrame({
    'esti_re_mean': esti_re_mean,
    'esti_re_ci_lower': esti_re_mean - esti_re_ci,
    'esti_re_ci_upper': esti_re_mean + esti_re_ci,
    'esti_im_mean': esti_im_mean,
    'esti_im_ci_lower': esti_im_mean - esti_im_ci,
    'esti_im_ci_upper': esti_im_mean + esti_im_ci
})

df_esti_stats.insert(0, 'Time', range(len(df_esti_stats)))

df_esti_stats.to_csv('../results/mean_lambda/trans_mean_lambda.csv', index=False)

###########################################################################################

df_null_esti_sum = pd.DataFrame()

for random_seed in range(200):

    df_null_esti = pd.read_csv('../results/ROC/trans/ews/dynamics_trans_null_{}.csv'.format(random_seed))[['esti_rrate']]
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

df_null_esti_stats.to_csv('../results/mean_lambda/trans_null_mean_lambda.csv', index=False)