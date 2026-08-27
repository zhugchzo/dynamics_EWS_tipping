import pandas as pd
import numpy as np
import os

df_esti_sum = pd.DataFrame()

for random_seed in range(200):

    df_esti = pd.read_csv('../results/ROC/fold/ews/dynamics_fold_{}.csv'.format(random_seed))[['esti_rrate']]

    df_esti_sum = pd.concat([df_esti_sum.reset_index(drop=True),df_esti.reset_index(drop=True)], axis=1)

# 按行均值
esti_mean = df_esti_sum.mean(axis=1)

# 按行标准差
esti_std = df_esti_sum.std(axis=1, ddof=1)

# 每行有效样本数
n = df_esti_sum.count(axis=1)

# 95% 置信区间半宽
esti_ci = 1.96 * esti_std / np.sqrt(n)

# 整理成一个 DataFrame
df_esti_stats = pd.DataFrame({
    'esti_mean': esti_mean,
    'esti_ci_lower': esti_mean - esti_ci,
    'esti_ci_upper': esti_mean + esti_ci,
})

df_esti_stats.insert(0, 'Time', range(len(df_esti_stats)))

df_esti_stats.to_csv('../results/mean_lambda/fold_mean_lambda.csv', index=False)

###########################################################################################

true_re_dict = {}
esti_re_dict = {}

for random_seed in range(200):

    df = pd.read_csv('../results/transient/fold/dynamics_fold_{}.csv'.format(random_seed))

    true_re_dict[random_seed] = df['rrate'].to_numpy()
    esti_re_dict[random_seed] = df['esti_rrate'].to_numpy()

df_transient_re_sum = pd.DataFrame(true_re_dict)
df_transient_esti_re_sum = pd.DataFrame(esti_re_dict)

true_transient_re_mean = df_transient_re_sum.mean(axis=1).to_frame(name='true_re_mean')

# 按行均值
esti_transient_re_mean = df_transient_esti_re_sum.mean(axis=1)

# 按行标准差
esti_transient_re_std = (df_transient_esti_re_sum.std(axis=1, ddof=1).fillna(0))

# 每行有效样本数
n_transient_re = df_transient_esti_re_sum.count(axis=1)

# 95% 置信区间半宽
esti_transient_re_ci = 1.96 * esti_transient_re_std / np.sqrt(n_transient_re)

# 整理成一个 DataFrame
df_transient_esti_stats = pd.DataFrame({
    'esti_re_mean': esti_transient_re_mean,
    'esti_re_ci_lower': esti_transient_re_mean - esti_transient_re_ci,
    'esti_re_ci_upper': esti_transient_re_mean + esti_transient_re_ci
})

df_transient_mean = pd.concat([true_transient_re_mean, df_transient_esti_stats], axis=1)

df_transient_mean.insert(0, 'Time', range(len(df_transient_esti_stats)))

df_transient_mean.to_csv('../results/mean_lambda/transient/fold_mean_lambda.csv', index=False)

###########################################################################################

df_null_esti_sum = pd.DataFrame()

for random_seed in range(200):

    df_null_esti = pd.read_csv('../results/ROC/fold/ews/dynamics_fold_null_{}.csv'.format(random_seed))[['esti_rrate']]
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

df_null_mean = pd.concat([df_null_esti_stats], axis=1)
df_null_mean.insert(0, 'Time', range(len(df_null_mean)))

df_null_mean.to_csv('../results/mean_lambda/fold_null_mean_lambda.csv', index=False)

###########################################################################################

df_esti_sindy_sum = pd.DataFrame()

for random_seed in range(200):

    df_sindy_esti = pd.read_csv('../results/ROC/fold/SINDy/fold_{}.csv'.format(random_seed))[['esti_rrate']]

    df_esti_sindy_sum = pd.concat([df_esti_sindy_sum.reset_index(drop=True),df_sindy_esti.reset_index(drop=True)], axis=1)

# 按行均值
esti_sindy_mean = df_esti_sindy_sum.mean(axis=1)

# 按行标准差
esti_sindy_std = df_esti_sindy_sum.std(axis=1, ddof=1)

# 每行有效样本数
n = df_esti_sindy_sum.count(axis=1)

# 95% 置信区间半宽
esti_sindy_ci = 1.96 * esti_sindy_std / np.sqrt(n)

# 整理成一个 DataFrame
df_esti_sindy_stats = pd.DataFrame({
    'esti_mean': esti_sindy_mean,
    'esti_ci_lower': esti_sindy_mean - esti_sindy_ci,
    'esti_ci_upper': esti_sindy_mean + esti_sindy_ci,
})

df_esti_sindy_stats.insert(0, 'Time', range(len(df_esti_sindy_stats)))

df_esti_sindy_stats.to_csv('../results/mean_lambda/fold_SINDy_mean_lambda.csv', index=False)