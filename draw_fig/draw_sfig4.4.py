import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

font_x = {'family':'Arial','weight':'normal','size': 60}
font_y = {'family':'Arial','weight':'normal','size': 60}

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['xtick.labelsize'] = 45
plt.rcParams['ytick.labelsize'] = 40

fig, axs = plt.subplots(2, 5, figsize=(48,16))

network_size = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60]

for ii in range(10):

    df_fold_BA_results = pd.read_csv('../results/mean_lambda/transient/fold_BA/fold_BA_mean_lambda_{}.csv'.format(network_size[ii]))

    t_fold_BA = df_fold_BA_results['Normalized time'].to_numpy() + 1
    fold_BA_re = df_fold_BA_results['true_re_mean'].to_numpy()
    fold_BA_esti_re = df_fold_BA_results['esti_re_mean'].to_numpy()
    fold_BA_esti_re_lower = df_fold_BA_results['esti_re_ci_lower'].to_numpy()
    fold_BA_esti_re_upper = df_fold_BA_results['esti_re_ci_upper'].to_numpy()

    if ii <= 4:

        ax = axs[0,ii]

        ax.plot(t_fold_BA,fold_BA_re,c='dimgrey',linewidth=6,alpha=0.9,zorder=1)
        ax.plot(t_fold_BA,fold_BA_esti_re,c='crimson',linewidth=6,alpha=0.9,zorder=2)
        ax.fill_between(t_fold_BA,fold_BA_esti_re_lower,fold_BA_esti_re_upper,color='crimson',alpha=0.3,linewidth=0,zorder=2)
        ax.set_xticks([t_fold_BA[0],t_fold_BA[-1]])
        ax.tick_params(direction='in', length=10)
        ax.minorticks_on()
        ax.tick_params(axis='x', which='minor', bottom=False)
        ax.tick_params(axis='y', which='minor', direction='in', length=5)

        ax.text(0.4, 0.1,'$N$={} (BA)'.format(network_size[ii]),transform=ax.transAxes,fontsize=60,ha='center',va='center',color='black')

        ax.set_yscale('symlog', linthresh=0.1)
        ax.set_yticks([-1, -0.1, 0, 0.1, 1])

        ax.hlines(y=0, xmin=t_fold_BA[0], xmax=t_fold_BA[-1], color='grey', linestyle='--', linewidth=4, alpha=0.5, zorder=0)

        if ii == 0:

            ax.set_ylabel('Re',font_y)
            ax.yaxis.set_label_coords(-0.2, 0.5)
            ax.text(-0.3,0.85,'(d)',ha='left', transform=ax.transAxes,fontdict={'family':'Arial','size':80,'weight':'normal'})

    else:
        ax = axs[1,ii-5]

        ax.plot(t_fold_BA,fold_BA_re,c='dimgrey',linewidth=6,alpha=0.9,zorder=1)
        ax.plot(t_fold_BA,fold_BA_esti_re,c='crimson',linewidth=6,alpha=0.9,zorder=2)
        ax.fill_between(t_fold_BA,fold_BA_esti_re_lower,fold_BA_esti_re_upper,color='crimson',alpha=0.3,linewidth=0,zorder=2)
        ax.set_xticks([t_fold_BA[0],t_fold_BA[-1]])
        ax.tick_params(direction='in', length=10)
        ax.minorticks_on()
        ax.tick_params(axis='x', which='minor', bottom=False)
        ax.tick_params(axis='y', which='minor', direction='in', length=5)

        ax.hlines(y=0, xmin=t_fold_BA[0], xmax=t_fold_BA[-1], color='grey', linestyle='--', linewidth=4, alpha=0.5, zorder=0)

        ax.text(0.4, 0.1,'$N$={} (BA)'.format(network_size[ii]),transform=ax.transAxes,fontsize=60,ha='center',va='center',color='black')

        ax.set_yscale('symlog', linthresh=0.1)
        ax.set_yticks([-1, -0.1, 0, 0.1, 1])

        ax.set_xlabel('Normalized time',font_x,labelpad=5)

        if ii == 5:
            
            ax.set_ylabel('Re',font_y)
            ax.yaxis.set_label_coords(-0.2, 0.5)

plt.subplots_adjust(top=0.99, bottom=0.1, left=0.05, right=0.99, hspace=0.15, wspace=0.2)
plt.savefig('../figures/SFIG4.4.pdf',format='pdf')