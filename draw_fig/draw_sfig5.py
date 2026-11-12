import pandas
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import font_manager

font_x = {'family': 'Arial', 'weight': 'normal', 'size': 40}
font_y = {'family': 'Arial', 'weight': 'normal', 'size': 40}
font_panel = {'family': 'Arial', 'weight': 'normal', 'size': 60}

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['xtick.labelsize'] = 36
plt.rcParams['ytick.labelsize'] = 36

fig, axs = plt.subplots(2, 2, figsize=(20, 18))
(ax1, ax3, ax2, ax4) = axs.flatten()

fold_time = 1000
gaussian_end = 1047
rational_end = 1038
blue_after = 5

df_gaussian = pandas.read_csv(
    '../data/transient/multi_zero/gaussian/multi_zero_0.csv'
)
df_gaussian_results = pandas.read_csv(
    '../results/mean_lambda/transient/multi_zero_mean_lambda_gaussian.csv'
)
df_rational = pandas.read_csv(
    '../data/transient/multi_zero/rational/multi_zero_0.csv'
)
df_rational_results = pandas.read_csv(
    '../results/mean_lambda/transient/multi_zero_mean_lambda_rational.csv'
)

# Gaussian 数据。轨迹表 Time=999 和结果表 Time=0 都画在 1000。
t1 = df_gaussian['Time'] + 1
x_gaussian = pandas.Series(df_gaussian['traj_x'].to_numpy(), index=t1)
x_gaussian_show = x_gaussian.loc[fold_time - 5:gaussian_end + blue_after]

df_gaussian_results = df_gaussian_results.loc[
    df_gaussian_results['Time'].between(0, gaussian_end - fold_time)
]
t2 = df_gaussian_results['Time'].to_numpy() + fold_time
x_gaussian_transient = x_gaussian.loc[t2]
gaussian_xlim = (fold_time - 7, gaussian_end + blue_after + 2)

# Rational 数据
t3 = df_rational['Time'] + 1
x_rational = pandas.Series(df_rational['traj_x'].to_numpy(), index=t3)
x_rational_show = x_rational.loc[fold_time - 5:rational_end + blue_after]

df_rational_results = df_rational_results.loc[
    df_rational_results['Time'].between(0, rational_end - fold_time)
]
t4 = df_rational_results['Time'].to_numpy() + fold_time
x_rational_transient = x_rational.loc[t4]
rational_xlim = (fold_time - 7, rational_end + blue_after + 2)

# (a) Gaussian 状态
ax1.plot(x_gaussian_show.index, x_gaussian_show,
         c='darkslateblue', linewidth=6)
ax1.scatter(x_gaussian_show.index, x_gaussian_show,
            s=150, c='darkslateblue', marker='o')
ax1.plot(t2, x_gaussian_transient, c='darkgrey', linewidth=6, zorder=3)
ax1.scatter(t2, x_gaussian_transient, s=150, c='darkgrey', marker='o', zorder=4)
ax1.scatter(fold_time, x_gaussian.loc[fold_time], s=1000, marker='o',
            facecolors='white', edgecolors='black', linewidths=3, zorder=5)
ax1.set_ylabel('System state', font_y)
ax1.set_xlim(*gaussian_xlim)
ax1.set_xticks([fold_time, gaussian_end])
ax1.set_ylim(-0.85, 1.35)
ax1.set_yticks([-0.5, 0, 0.5, 1], labels=['-0.5', '0', '0.5', '1'])
ax1.set_title('Gaussian composite', fontsize=40, fontweight='normal', fontfamily='DejaVu Sans', fontstyle= 'italic', y=1.01)
ax1.text(0.05, 0.85, '(a)', ha='left', transform=ax1.transAxes, fontdict=font_panel)

# (b) Gaussian Re
ax2.plot(t2, df_gaussian_results['true_re_mean'].to_numpy(),
         c='dimgrey', linewidth=6, alpha=0.9, zorder=1)
ax2.plot(t2, df_gaussian_results['cubic_re_mean'].to_numpy(),
         c='royalblue', linewidth=6, alpha=0.9, zorder=1)
ax2.fill_between(t2, df_gaussian_results['esti_re_ci_lower'].to_numpy(),
                 df_gaussian_results['esti_re_ci_upper'].to_numpy(),
                 color='crimson', alpha=0.3, linewidth=0, zorder=2)
ax2.plot(t2, df_gaussian_results['esti_re_mean'].to_numpy(),
         c='crimson', linewidth=6, alpha=0.9, zorder=3)
ax2.axhline(y=0, color='grey', linestyle='--', linewidth=4, alpha=0.5, zorder=0)
ax2.set_xlabel('Time', font_x, labelpad=5)
ax2.set_ylabel('Re', font_y)
ax2.set_xlim(*gaussian_xlim)
ax2.set_xticks([fold_time, gaussian_end])
ax2.set_yscale('symlog', linthresh=0.1)
ax2.set_yticks([-1, -0.1, 0, 0.1, 1])
ax2.text(0.05, 0.85, '(c)', ha='left', transform=ax2.transAxes, fontdict=font_panel)

# (c) Rational 状态
ax3.plot(x_rational_show.index, x_rational_show,
         c='darkslateblue', linewidth=6)
ax3.scatter(x_rational_show.index, x_rational_show,
            s=150, c='darkslateblue', marker='o')
ax3.plot(t4, x_rational_transient, c='darkgrey', linewidth=6, zorder=3)
ax3.scatter(t4, x_rational_transient, s=150, c='darkgrey', marker='o', zorder=4)
ax3.scatter(fold_time, x_rational.loc[fold_time], s=1000, marker='o',
            facecolors='white', edgecolors='black', linewidths=3, zorder=5)
ax3.set_xlim(*rational_xlim)
ax3.set_xticks([fold_time, rational_end])
ax3.set_ylim(-0.85, 1.35)
ax3.set_yticks([-0.5, 0, 0.5, 1], labels=['-0.5', '0', '0.5', '1'])
ax3.set_title('Rational', fontsize=40, fontweight='normal', fontfamily='DejaVu Sans', fontstyle= 'italic', y=1.01)
ax3.text(0.05, 0.85, '(b)', ha='left', transform=ax3.transAxes, fontdict=font_panel)

# (d) Rational Re
ax4.plot(t4, df_rational_results['true_re_mean'].to_numpy(),
         c='dimgrey', linewidth=6, alpha=0.9, zorder=1)
ax4.plot(t4, df_rational_results['cubic_re_mean'].to_numpy(),
         c='royalblue', linewidth=6, alpha=0.9, zorder=1)
ax4.fill_between(t4, df_rational_results['esti_re_ci_lower'].to_numpy(),
                 df_rational_results['esti_re_ci_upper'].to_numpy(),
                 color='crimson', alpha=0.3, linewidth=0, zorder=2)
ax4.plot(t4, df_rational_results['esti_re_mean'].to_numpy(),
         c='crimson', linewidth=6, alpha=0.9, zorder=3)
ax4.axhline(y=0, color='grey', linestyle='--', linewidth=4, alpha=0.5, zorder=0)
ax4.set_xlabel('Time', font_x, labelpad=5)
ax4.set_xlim(*rational_xlim)
ax4.set_xticks([fold_time, rational_end])
ax4.set_yscale('symlog', linthresh=0.1)
ax4.set_yticks([-1, -0.1, 0, 0.1, 1])
ax4.text(0.05, 0.85, '(d)', ha='left', transform=ax4.transAxes, fontdict=font_panel)

for ax in (ax1, ax2, ax3, ax4):
    ax.yaxis.set_label_coords(-0.15, 0.5)
    ax.tick_params(axis='x', pad=10)
    ax.tick_params(direction='in', length=10)
    ax.minorticks_on()
    ax.tick_params(axis='x', which='minor', bottom=False)
    ax.tick_params(axis='y', which='minor', direction='in', length=5)

legend_state = mlines.Line2D(
    [], [], color='darkslateblue', linestyle='-', marker='o',
    linewidth=3, markersize=15
)
legend_pstate = mlines.Line2D(
    [], [], color='darkgrey', linestyle='-', marker='o',
    linewidth=3, markersize=15
)
legend_bif = mlines.Line2D(
    [], [], linestyle='None', marker='o', markerfacecolor='white',
    markeredgecolor='black', markersize=20, markeredgewidth=3
)
legend_true_re = mlines.Line2D(
    [], [], color='dimgrey', linestyle='-', linewidth=6, alpha=0.9
)
legend_cubic_re = mlines.Line2D(
    [], [], color='royalblue', linestyle='-', linewidth=6, alpha=0.9
)
legend_esti_re = mlines.Line2D(
    [], [], color='crimson', linestyle='-', linewidth=6, alpha=0.9
)

fig.legend(
    handles=[legend_state, legend_pstate, legend_true_re, legend_cubic_re, legend_esti_re, legend_bif],
    labels=['State', 'Forecast regime\n(transient regime)', 'True Re', 'Re calculated\nfrom $P_{\\phi}(x)$', 'Estimated Re', 'Fold bifurcation'],
    loc='upper center', bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False,
    prop=font_manager.FontProperties(family='Arial', size=30, weight='normal'),
    handletextpad=0.4
)

plt.subplots_adjust(top=0.85, bottom=0.07, left=0.09, right=0.98,
                    hspace=0.2, wspace=0.2)
plt.savefig('../figures/SFIG5.pdf', format='pdf')