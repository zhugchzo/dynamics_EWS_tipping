import pandas as pd
import numpy as np
import os

models = ('gaussian', 'rational')
output_dir = '../results/mean_lambda/transient'
os.makedirs(output_dir, exist_ok=True)

for model in models:

    input_dir = os.path.join('../results/transient/multi_zero', model)

    true_re_dict = {}
    cubic_re_dict = {}
    esti_re_dict = {}

    for random_seed in range(200):

        df = pd.read_csv(os.path.join(input_dir, 'dynamics_multi_zero_{}.csv'.format(random_seed)))

        true_re_dict[random_seed] = df['rrate']
        cubic_re_dict[random_seed] = df['rrate_cubic']
        esti_re_dict[random_seed] = df['esti_rrate']

    df_transient_re_sum = pd.DataFrame(true_re_dict)
    df_transient_cubic_re_sum = pd.DataFrame(cubic_re_dict)
    df_transient_esti_re_sum = pd.DataFrame(esti_re_dict)

    true_transient_re_mean = df_transient_re_sum.mean(axis=1).to_frame(name='true_re_mean')
    true_transient_cubic_re_mean = df_transient_cubic_re_sum.mean(axis=1).to_frame(name='cubic_re_mean')

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

    df_transient_mean = pd.concat([true_transient_re_mean, true_transient_cubic_re_mean, df_transient_esti_stats], axis=1)

    df_transient_mean.insert(0, 'Time', range(len(df_transient_esti_stats)))

    df_transient_mean.to_csv(os.path.join(output_dir, 'multi_zero_mean_lambda_{}.csv'.format(model)), index=False)
