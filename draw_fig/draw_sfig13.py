import pandas
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import font_manager

font_x = {'family':'Arial','weight':'normal','size': 60}
font_y = {'family':'Arial','weight':'normal','size': 60}

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['xtick.labelsize'] = 45
plt.rcParams['ytick.labelsize'] = 45

fig, axs = plt.subplots(2, 5, figsize=(48,20))

train_length = 500
length = 1000

node_ratio = [0.1, 0.2, 0.3, 0.4, 0.5]
fig_label_ER = ['(a)', '(b)', '(c)', '(d)', '(e)']
fig_label_BA = ['(f)', '(g)', '(h)', '(i)', '(j)']

for ii in range(len(node_ratio)):

    ax = axs[0,ii]

    df_fold_ER_results = pandas.read_csv('../results/mean_lambda/partial/fold_ER_mean_lambda_{}.csv'.format(node_ratio[ii]))

    t_fold_ER = df_fold_ER_results['Time'] + 501

    fold_ER_esti_re = df_fold_ER_results['esti_re_mean']
    fold_ER_esti_re_lower = df_fold_ER_results['esti_re_ci_lower']
    fold_ER_esti_re_upper = df_fold_ER_results['esti_re_ci_upper']

    fold_ER_esti_im1 = df_fold_ER_results['esti_im_mean']
    fold_ER_esti_im1_lower = df_fold_ER_results['esti_im_ci_lower']
    fold_ER_esti_im1_upper = df_fold_ER_results['esti_im_ci_upper']

################################################################################################

    ax.plot(t_fold_ER,fold_ER_esti_re,c='crimson',linewidth=6,alpha=0.9,zorder=2)
    ax.fill_between(t_fold_ER,fold_ER_esti_re_lower,fold_ER_esti_re_upper,color='crimson',alpha=0.3,linewidth=0,zorder=2)
    ax.plot(t_fold_ER,fold_ER_esti_im1,c='royalblue',linewidth=6,alpha=0.9,zorder=2)
    ax.fill_between(t_fold_ER,fold_ER_esti_im1_lower,fold_ER_esti_im1_upper,color='royalblue',alpha=0.3,linewidth=0,zorder=2)
    ax.set_xlim(480,1020)
    ax.set_xticks([500,1000])
    ax.set_xticklabels(['500','1000'])
    ax.set_ylim(-0.24,0.24)
    ax.set_yticks([-0.2,0,0.2])
    ax.set_yticklabels(['-0.2','0','0.2'])
    ax.tick_params(direction='in', length=10)
    ax.minorticks_on()
    ax.tick_params(axis='x', which='minor', bottom=False)
    ax.tick_params(axis='y', which='minor', direction='in', length=5)
    ax.text(0.06,0.88,fig_label_ER[ii],ha='left', transform=ax.transAxes,fontdict={'family':'Arial','size':60,'weight':'normal'})
    ax.hlines(y=0, xmin=480, xmax=1020, color='grey', linestyle='--', linewidth=4, alpha=0.5, zorder=0)

    ER_str = 'ER network\n' + r'$\rho_{{\text{{obs}}}}={}$'.format(node_ratio[ii])
    ax.text(0.3,0.85, ER_str, transform=ax.transAxes, fontsize=45, verticalalignment='center')

    # if ii == 0:
    #     ax.set_ylabel('Re',font_y)
    #     ax.yaxis.set_label_coords(-0.2, 0.5)

################################################################################################

for ii in range(len(node_ratio)):

    ax = axs[1,ii]

    df_fold_ER_results = pandas.read_csv('../results/mean_lambda/partial/fold_BA_mean_lambda_{}.csv'.format(node_ratio[ii]))

    t_fold_ER = df_fold_ER_results['Time'] + 501

    fold_ER_esti_re = df_fold_ER_results['esti_re_mean']
    fold_ER_esti_re_lower = df_fold_ER_results['esti_re_ci_lower']
    fold_ER_esti_re_upper = df_fold_ER_results['esti_re_ci_upper']

    fold_ER_esti_im1 = df_fold_ER_results['esti_im_mean']
    fold_ER_esti_im1_lower = df_fold_ER_results['esti_im_ci_lower']
    fold_ER_esti_im1_upper = df_fold_ER_results['esti_im_ci_upper']

################################################################################################

    ax.plot(t_fold_ER,fold_ER_esti_re,c='crimson',linewidth=6,alpha=0.9,zorder=2)
    ax.fill_between(t_fold_ER,fold_ER_esti_re_lower,fold_ER_esti_re_upper,color='crimson',alpha=0.3,linewidth=0,zorder=2)
    ax.plot(t_fold_ER,fold_ER_esti_im1,c='royalblue',linewidth=6,alpha=0.9,zorder=2)
    ax.fill_between(t_fold_ER,fold_ER_esti_im1_lower,fold_ER_esti_im1_upper,color='royalblue',alpha=0.3,linewidth=0,zorder=2)
    ax.set_xlim(480,1020)
    ax.set_xticks([500,1000])
    ax.set_xticklabels(['500','1000'])
    ax.set_ylim(-0.72,0.72)
    ax.set_yticks([-0.6,-0,0.6])
    ax.set_yticklabels(['-0.6','0','0.6'])
    ax.tick_params(direction='in', length=10)
    ax.minorticks_on()
    ax.tick_params(axis='x', which='minor', bottom=False)
    ax.tick_params(axis='y', which='minor', direction='in', length=5)
    ax.text(0.06,0.88,fig_label_BA[ii],ha='left', transform=ax.transAxes,fontdict={'family':'Arial','size':60,'weight':'normal'})
    ax.hlines(y=0, xmin=480, xmax=1020, color='grey', linestyle='--', linewidth=4, alpha=0.5, zorder=0)

    BA_str = 'BA network\n' + r'$\rho_{{\text{{obs}}}}={}$'.format(node_ratio[ii])
    ax.text(0.3,0.85, BA_str, transform=ax.transAxes, fontsize=45, verticalalignment='center')

    # if ii == 0:
    #     ax.set_ylabel('Re',font_y)
    #     ax.yaxis.set_label_coords(-0.2, 0.5)

    ax.set_xlabel('Time',font_x,labelpad=5)

legend_esti_re = mlines.Line2D([], [], color='crimson', linestyle='-',linewidth=8,alpha=0.9)
legend_esti_im = mlines.Line2D([], [], color='royalblue', linestyle='-',linewidth=8,alpha=0.9)

fig.legend(handles = [legend_esti_re, legend_esti_im],
            labels = ['Estimated Re', 'Estimated Im'],
              loc='upper center', bbox_to_anchor=(0.5,1.03), ncol=2, frameon=False, 
              prop=font_manager.FontProperties(family='Arial', size=60, weight='bold'))

plt.subplots_adjust(top=0.9, bottom=0.07, left=0.03, right=0.99, hspace=0.2, wspace=0.25)
plt.savefig('../figures/SFIG13.pdf',format='pdf')