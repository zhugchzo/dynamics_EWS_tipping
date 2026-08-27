import pandas
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import font_manager

font_x = {'family':'Arial','weight':'normal','size': 35}
font_y = {'family':'Arial','weight':'normal','size': 35}

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['xtick.labelsize'] = 30
plt.rcParams['ytick.labelsize'] = 30

fig, axs = plt.subplots(2, 3, figsize=(22,15))

ax1, ax2, ax3 = axs[0,0], axs[0,1], axs[0,2]
ax4, ax5, ax6 = axs[1,0], axs[1,1], axs[1,2]

df_fold = pandas.read_csv('../results/noise/fold/auc.csv')
df_trans = pandas.read_csv('../results/noise/trans/auc.csv')
df_sup_hopf = pandas.read_csv('../results/noise/sup_hopf/auc.csv')
df_sub_hopf = pandas.read_csv('../results/noise/sub_hopf/auc.csv')
df_fold_ER = pandas.read_csv('../results/noise/fold_ER/auc.csv')
df_fold_BA = pandas.read_csv('../results/noise/fold_BA/auc.csv')

###################################################

sigma = [0.0025, 0.01, 0.02, 0.03, 0.04, 0.05]

###################################################

dynamics_fold = df_fold['dynamics_auc'].to_numpy()
dynamics_trans = df_trans['dynamics_auc'].to_numpy()
dynamics_sup_hopf = df_sup_hopf['dynamics_auc'].to_numpy()
dynamics_sub_hopf = df_sub_hopf['dynamics_auc'].to_numpy()
dynamics_fold_ER = df_fold_ER['dynamics_auc'].to_numpy()
dynamics_fold_BA = df_fold_BA['dynamics_auc'].to_numpy()

ac_fold = df_fold['ac_auc'].to_numpy()
ac_trans = df_trans['ac_auc'].to_numpy()
ac_sup_hopf = df_sup_hopf['ac_auc'].to_numpy()
ac_sub_hopf = df_sub_hopf['ac_auc'].to_numpy()
ac_fold_ER = df_fold_ER['ac_auc'].to_numpy()
ac_fold_BA = df_fold_BA['ac_auc'].to_numpy()

dev_fold = df_fold['dev_auc'].to_numpy()
dev_trans = df_trans['dev_auc'].to_numpy()
dev_sup_hopf = df_sup_hopf['dev_auc'].to_numpy()
dev_sub_hopf = df_sub_hopf['dev_auc'].to_numpy()
dev_fold_ER = df_fold_ER['dev_auc'].to_numpy()
dev_fold_BA = df_fold_BA['dev_auc'].to_numpy()

###################################################

ax1.plot(sigma,dynamics_fold,c='crimson',linewidth=5,zorder=3)
ax1.scatter(sigma,dynamics_fold,c='crimson',marker='o',s=150,zorder=3)

ax1.plot(sigma,ac_fold,c='darkslateblue',linewidth=5,zorder=2)
ax1.scatter(sigma,ac_fold,c='darkslateblue',marker='o',s=150,zorder=2)

ax1.plot(sigma,dev_fold,c='darkorange',linewidth=5,zorder=1)
ax1.scatter(sigma,dev_fold,c='darkorange',marker='o',s=150,zorder=1)

ax1.set_xlim(-0.0025,0.055)
ax1.set_xticks([0.0025, 0.01, 0.02, 0.03, 0.04, 0.05])
ax1.set_xticklabels(['0.25', '1', '2', '3', '4', '5'])
ax1.annotate(
    r'$\times 10^{-2}$',
    xy=(1, 0), xycoords='axes fraction',
    xytext=(8, -2), textcoords='offset points',
    ha='left', va='top',fontsize=18
)

ax1.set_ylim(0.15,1.05)
ax1.set_yticks([0.2,0.6,1])
ax1.set_yticklabels(['0.2','0.6','1'])

ax1.tick_params(direction='in', length=10)
ax1.minorticks_on()
ax1.tick_params(axis='x', which='minor', bottom=False)
ax1.tick_params(axis='y', which='minor', direction='in', length=5)

ax1.set_ylabel('AUC', font_y)
ax1.yaxis.set_label_coords(-0.15, 0.5)

ax1.text(-0.2,1.05,'(a)',ha='left', transform=ax1.transAxes,fontdict={'family':'Arial','size':40,'weight':'normal'})
ax1.set_title('Intraspecific competition', fontsize=30, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.02)

###################################################

ax2.plot(sigma,dynamics_trans,c='crimson',linewidth=5,zorder=3)
ax2.scatter(sigma,dynamics_trans,c='crimson',marker='o',s=150,zorder=3)

ax2.plot(sigma,ac_trans,c='darkslateblue',linewidth=5,zorder=2)
ax2.scatter(sigma,ac_trans,c='darkslateblue',marker='o',s=150,zorder=2)

ax2.plot(sigma,dev_trans,c='darkorange',linewidth=5,zorder=1)
ax2.scatter(sigma,dev_trans,c='darkorange',marker='o',s=150,zorder=1)

ax2.set_xlim(-0.0025,0.055)
ax2.set_xticks([0.0025, 0.01, 0.02, 0.03, 0.04, 0.05])
ax2.set_xticklabels(['0.25', '1', '2', '3', '4', '5'])
ax2.annotate(
    r'$\times 10^{-2}$',
    xy=(1, 0), xycoords='axes fraction',
    xytext=(8, -2), textcoords='offset points',
    ha='left', va='top',fontsize=18
)

ax2.set_ylim(0.15,1.05)
ax2.set_yticks([0.2,0.6,1])
ax2.set_yticklabels(['0.2','0.6','1'])

ax2.tick_params(direction='in', length=10)
ax2.minorticks_on()
ax2.tick_params(axis='x', which='minor', bottom=False)
ax2.tick_params(axis='y', which='minor', direction='in', length=5)

ax2.text(-0.2,1.05,'(b)',ha='left', transform=ax2.transAxes,fontdict={'family':'Arial','size':40,'weight':'normal'})
ax2.set_title('Predator-prey (transcritical)', fontsize=30, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.02)

###################################################

ax3.plot(sigma,dynamics_sup_hopf,c='crimson',linewidth=5,zorder=3)
ax3.scatter(sigma,dynamics_sup_hopf,c='crimson',marker='o',s=150,zorder=3)

ax3.plot(sigma,ac_sup_hopf,c='darkslateblue',linewidth=5,zorder=2)
ax3.scatter(sigma,ac_sup_hopf,c='darkslateblue',marker='o',s=150,zorder=2)

ax3.plot(sigma,dev_sup_hopf,c='darkorange',linewidth=5,zorder=1)
ax3.scatter(sigma,dev_sup_hopf,c='darkorange',marker='o',s=150,zorder=1)

ax3.set_xlim(-0.0025,0.055)
ax3.set_xticks([0.0025, 0.01, 0.02, 0.03, 0.04, 0.05])
ax3.set_xticklabels(['0.25', '1', '2', '3', '4', '5'])
ax3.annotate(
    r'$\times 10^{-2}$',
    xy=(1, 0), xycoords='axes fraction',
    xytext=(8, -2), textcoords='offset points',
    ha='left', va='top',fontsize=18
)

ax3.set_ylim(0.15,1.05)
ax3.set_yticks([0.2,0.6,1])
ax3.set_yticklabels(['0.2','0.6','1'])

ax3.tick_params(direction='in', length=10)
ax3.minorticks_on()
ax3.tick_params(axis='x', which='minor', bottom=False)
ax3.tick_params(axis='y', which='minor', direction='in', length=5)

ax3.text(-0.2,1.05,'(c)',ha='left', transform=ax3.transAxes,fontdict={'family':'Arial','size':40,'weight':'normal'})
ax3.set_title('Predator-prey (Hopf)', fontsize=30, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.02)

###################################################

ax4.plot(sigma,dynamics_sub_hopf,c='crimson',linewidth=5,zorder=3)
ax4.scatter(sigma,dynamics_sub_hopf,c='crimson',marker='o',s=150,zorder=3)

ax4.plot(sigma,ac_sub_hopf,c='darkslateblue',linewidth=5,zorder=2)
ax4.scatter(sigma,ac_sub_hopf,c='darkslateblue',marker='o',s=150,zorder=2)

ax4.plot(sigma,dev_sub_hopf,c='darkorange',linewidth=5,zorder=1)
ax4.scatter(sigma,dev_sub_hopf,c='darkorange',marker='o',s=150,zorder=1)

ax4.set_xlim(-0.0025,0.055)
ax4.set_xticks([0.0025, 0.01, 0.02, 0.03, 0.04, 0.05])
ax4.set_xticklabels(['0.25', '1', '2', '3', '4', '5'])
ax4.annotate(
    r'$\times 10^{-2}$',
    xy=(1, 0), xycoords='axes fraction',
    xytext=(8, -2), textcoords='offset points',
    ha='left', va='top',fontsize=18
)

ax4.set_ylim(0.15,1.05)
ax4.set_yticks([0.2,0.6,1])
ax4.set_yticklabels(['0.2','0.6','1'])

ax4.tick_params(direction='in', length=10)
ax4.minorticks_on()
ax4.tick_params(axis='x', which='minor', bottom=False)
ax4.tick_params(axis='y', which='minor', direction='in', length=5)

ax4.set_xlabel('Noise amplitude',font_x,labelpad=20)
ax4.set_ylabel('AUC', font_y)
ax4.yaxis.set_label_coords(-0.15, 0.5)

ax4.text(-0.2,1.05,'(d)',ha='left', transform=ax4.transAxes,fontdict={'family':'Arial','size':40,'weight':'normal'})
ax4.set_title('Epidemic', fontsize=30, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.02)

###################################################

ax5.plot(sigma,dynamics_fold_ER,c='crimson',linewidth=5,zorder=3)
ax5.scatter(sigma,dynamics_fold_ER,c='crimson',marker='o',s=150,zorder=3)

ax5.plot(sigma,ac_fold_ER,c='darkslateblue',linewidth=5,zorder=2)
ax5.scatter(sigma,ac_fold_ER,c='darkslateblue',marker='o',s=150,zorder=2)

ax5.plot(sigma,dev_fold_ER,c='darkorange',linewidth=5,zorder=1)
ax5.scatter(sigma,dev_fold_ER,c='darkorange',marker='o',s=150,zorder=1)

ax5.set_xlim(-0.0025,0.055)
ax5.set_xticks([0.0025, 0.01, 0.02, 0.03, 0.04, 0.05])
ax5.set_xticklabels(['0.25', '1', '2', '3', '4', '5'])
ax5.annotate(
    r'$\times 10^{-2}$',
    xy=(1, 0), xycoords='axes fraction',
    xytext=(8, -2), textcoords='offset points',
    ha='left', va='top',fontsize=18
)

ax5.set_ylim(0.15,1.05)
ax5.set_yticks([0.2,0.6,1])
ax5.set_yticklabels(['0.2','0.6','1'])

ax5.tick_params(direction='in', length=10)
ax5.minorticks_on()
ax5.tick_params(axis='x', which='minor', bottom=False)
ax5.tick_params(axis='y', which='minor', direction='in', length=5)

ax5.set_xlabel('Noise amplitude',font_x,labelpad=20)

ax5.text(-0.2,1.05,'(e)',ha='left', transform=ax5.transAxes,fontdict={'family':'Arial','size':40,'weight':'normal'})
ax5.set_title('Mutualistic network (ER)', fontsize=30, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.02)

###################################################

ax6.plot(sigma,dynamics_fold_BA,c='crimson',linewidth=5,zorder=3)
ax6.scatter(sigma,dynamics_fold_BA,c='crimson',marker='o',s=150,zorder=3)

ax6.plot(sigma,ac_fold_BA,c='darkslateblue',linewidth=5,zorder=2)
ax6.scatter(sigma,ac_fold_BA,c='darkslateblue',marker='o',s=150,zorder=2)

ax6.plot(sigma,dev_fold_BA,c='darkorange',linewidth=5,zorder=1)
ax6.scatter(sigma,dev_fold_BA,c='darkorange',marker='o',s=150,zorder=1)

ax6.set_xlim(-0.0025,0.055)
ax6.set_xticks([0.0025, 0.01, 0.02, 0.03, 0.04, 0.05])
ax6.set_xticklabels(['0.25', '1', '2', '3', '4', '5'])
ax6.annotate(
    r'$\times 10^{-2}$',
    xy=(1, 0), xycoords='axes fraction',
    xytext=(8, -2), textcoords='offset points',
    ha='left', va='top',fontsize=18
)

ax6.set_ylim(0.15,1.05)
ax6.set_yticks([0.2,0.6,1])
ax6.set_yticklabels(['0.2','0.6','1'])

ax6.tick_params(direction='in', length=10)
ax6.minorticks_on()
ax6.tick_params(axis='x', which='minor', bottom=False)
ax6.tick_params(axis='y', which='minor', direction='in', length=5)

ax6.set_xlabel('Noise amplitude',font_x,labelpad=20)

ax6.text(-0.2,1.05,'(f)',ha='left', transform=ax6.transAxes,fontdict={'family':'Arial','size':40,'weight':'normal'})
ax6.set_title('Mutualistic network (BA)', fontsize=30, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.02)

################################################################################################

legend_dynamics = mlines.Line2D([], [], color='crimson',linestyle='-',linewidth=5,marker='o',markersize=15)
legend_ac = mlines.Line2D([], [], color='darkslateblue',linestyle='-',linewidth=5,marker='o',markersize=15)
legend_dev = mlines.Line2D([], [], color='darkorange',linestyle='-',linewidth=5,marker='o',markersize=15)

fig.legend(handles = [legend_dynamics, legend_ac, legend_dev],
            labels = ['Dynamics-based EWS','Lag-1 AC','Dynamical eigenvalue'],
              loc='upper center', bbox_to_anchor=(0.5,1.02), ncol=3, frameon=False, 
              prop=font_manager.FontProperties(family='Arial', size=35))

plt.subplots_adjust(top=0.85, bottom=0.08, left=0.06, right=0.95, hspace=0.3, wspace=0.3)
plt.savefig('../figures/SFIG8.pdf',format='pdf')