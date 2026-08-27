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

df_auc = pandas.read_csv('auc.csv')

###################################################

theta = df_auc['theta'].to_numpy()
dev_fold = df_auc['fold_auc'].to_numpy()
dev_trans = df_auc['trans_auc'].to_numpy()
dev_sup_hopf = df_auc['sup_hopf_auc'].to_numpy()
dev_sub_hopf = df_auc['sub_hopf_auc'].to_numpy()
dev_fold_ER = df_auc['fold_ER_auc'].to_numpy()
dev_fold_BA = df_auc['fold_BA_auc'].to_numpy()

dynamics_auc = [1, 1, 0.9, 0.95, 0.99, 1]

for i in range(len(dynamics_auc)):

    if i < 3:

        axs[0,i].hlines(y=dynamics_auc[i], xmin=0, xmax=2.5, color='crimson', linestyle='--', linewidth=5, zorder=2)

    else:

        axs[1,i-3].hlines(y=dynamics_auc[i], xmin=0, xmax=2.5, color='crimson', linestyle='--', linewidth=5, zorder=2)

###################################################

ax1.plot(theta,dev_fold,c='darkorange',linewidth=5,zorder=1)
ax1.scatter(theta,dev_fold,c='darkorange',marker='o',s=150,zorder=1)

ax1.set_xlim(-0.25,2.75)
ax1.set_xticks([0, 0.5, 1, 1.5, 2, 2.5])
ax1.set_xticklabels(['0', '0.5', '1', '1.5', '2', '2.5'])

ax1.set_ylim(0.978,1.002)
ax1.set_yticks([0.98,0.99,1])
ax1.set_yticklabels(['0.98','0.99','1'])

ax1.tick_params(direction='in', length=10)
ax1.minorticks_on()
ax1.tick_params(axis='x', which='minor', bottom=False)
ax1.tick_params(axis='y', which='minor', direction='in', length=5)

ax1.set_ylabel('AUC', font_y)
ax1.yaxis.set_label_coords(-0.22, 0.5)

ax1.text(-0.2,1.05,'(a)',ha='left', transform=ax1.transAxes,fontdict={'family':'Arial','size':40,'weight':'normal'})
ax1.set_title('Intraspecific competition', fontsize=30, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.02)

###################################################

ax2.plot(theta,dev_trans,c='darkorange',linewidth=5,zorder=1)
ax2.scatter(theta,dev_trans,c='darkorange',marker='o',s=150,zorder=1)

ax2.set_xlim(-0.25,2.75)
ax2.set_xticks([0, 0.5, 1, 1.5, 2, 2.5])
ax2.set_xticklabels(['0', '0.5', '1', '1.5', '2', '2.5'])

ax2.set_ylim(0.936,1.006)
ax2.set_yticks([0.94,0.97,1])
ax2.set_yticklabels(['0.94','0.97','1'])

ax2.tick_params(direction='in', length=10)
ax2.minorticks_on()
ax2.tick_params(axis='x', which='minor', bottom=False)
ax2.tick_params(axis='y', which='minor', direction='in', length=5)

ax2.text(-0.2,1.05,'(b)',ha='left', transform=ax2.transAxes,fontdict={'family':'Arial','size':40,'weight':'normal'})
ax2.set_title('Predator-prey (transcritical)', fontsize=30, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.02)

###################################################

ax3.plot(theta,dev_sup_hopf,c='darkorange',linewidth=5,zorder=1)
ax3.scatter(theta,dev_sup_hopf,c='darkorange',marker='o',s=150,zorder=1)

ax3.set_xlim(-0.25,2.75)
ax3.set_xticks([0, 0.5, 1, 1.5, 2, 2.5])
ax3.set_xticklabels(['0', '0.5', '1', '1.5', '2', '2.5'])

ax3.set_ylim(0.736,1.024)
ax3.set_yticks([0.76,0.88,1])
ax3.set_yticklabels(['0.76','0.88','1'])

ax3.tick_params(direction='in', length=10)
ax3.minorticks_on()
ax3.tick_params(axis='x', which='minor', bottom=False)
ax3.tick_params(axis='y', which='minor', direction='in', length=5)

ax3.text(-0.2,1.05,'(c)',ha='left', transform=ax3.transAxes,fontdict={'family':'Arial','size':40,'weight':'normal'})
ax3.set_title('Predator-prey (Hopf)', fontsize=30, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.02)

###################################################

ax4.plot(theta,dev_sub_hopf,c='darkorange',linewidth=5,zorder=1)
ax4.scatter(theta,dev_sub_hopf,c='darkorange',marker='o',s=150,zorder=1)

ax4.set_xlim(-0.25,2.75)
ax4.set_xticks([0, 0.5, 1, 1.5, 2, 2.5])
ax4.set_xticklabels(['0', '0.5', '1', '1.5', '2', '2.5'])

ax4.set_ylim(0.12,1.08)
ax4.set_yticks([0.2,0.6,1])
ax4.set_yticklabels(['0.2','0.6','1'])

ax4.tick_params(direction='in', length=10)
ax4.minorticks_on()
ax4.tick_params(axis='x', which='minor', bottom=False)
ax4.tick_params(axis='y', which='minor', direction='in', length=5)

ax4.set_xlabel(r'Nonlinearity parameter ($\theta$)',font_x,labelpad=15)
ax4.set_ylabel('AUC', font_y)
ax4.yaxis.set_label_coords(-0.22, 0.5)

ax4.text(-0.2,1.05,'(d)',ha='left', transform=ax4.transAxes,fontdict={'family':'Arial','size':40,'weight':'normal'})
ax4.set_title('Epidemic', fontsize=30, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.02)

###################################################

ax5.plot(theta,dev_fold_ER,c='darkorange',linewidth=5,zorder=1)
ax5.scatter(theta,dev_fold_ER,c='darkorange',marker='o',s=150,zorder=1)

ax5.set_xlim(-0.25,2.75)
ax5.set_xticks([0, 0.5, 1, 1.5, 2, 2.5])
ax5.set_xticklabels(['0', '0.5', '1', '1.5', '2', '2.5'])

ax5.set_ylim(0.956,1.004)
ax5.set_yticks([0.96,0.98,1])
ax5.set_yticklabels(['0.96','0.98','1'])

ax5.tick_params(direction='in', length=10)
ax5.minorticks_on()
ax5.tick_params(axis='x', which='minor', bottom=False)
ax5.tick_params(axis='y', which='minor', direction='in', length=5)

ax5.set_xlabel(r'Nonlinearity parameter ($\theta$)',font_x,labelpad=15)

ax5.text(-0.2,1.05,'(e)',ha='left', transform=ax5.transAxes,fontdict={'family':'Arial','size':40,'weight':'normal'})
ax5.set_title('Mutualistic network (ER)', fontsize=30, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.02)

###################################################

ax6.plot(theta,dev_fold_BA,c='darkorange',linewidth=5,zorder=1)
ax6.scatter(theta,dev_fold_BA,c='darkorange',marker='o',s=150,zorder=1)

ax6.set_xlim(-0.25,2.75)
ax6.set_xticks([0, 0.5, 1, 1.5, 2, 2.5])
ax6.set_xticklabels(['0', '0.5', '1', '1.5', '2', '2.5'])

ax6.set_ylim(0.824,1.016)
ax6.set_yticks([0.84,0.92,1])
ax6.set_yticklabels(['0.84','0.92','1'])

ax6.tick_params(direction='in', length=10)
ax6.minorticks_on()
ax6.tick_params(axis='x', which='minor', bottom=False)
ax6.tick_params(axis='y', which='minor', direction='in', length=5)

ax6.set_xlabel(r'Nonlinearity parameter ($\theta$)',font_x,labelpad=15)

ax6.text(-0.2,1.05,'(f)',ha='left', transform=ax6.transAxes,fontdict={'family':'Arial','size':40,'weight':'normal'})
ax6.set_title('Mutualistic network (BA)', fontsize=30, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.02)

################################################################################################

legend_dynamics = mlines.Line2D([], [], color='crimson',linestyle='--',linewidth=5,marker=None)
legend_dev = mlines.Line2D([], [], color='darkorange',linestyle='-',linewidth=5,marker='o',markersize=15)

fig.legend(handles = [legend_dynamics, legend_dev],
            labels = [r'Dynamics-based EWS ($\alpha=0.01$)','Dynamical eigenvalue'],
              loc='upper center', bbox_to_anchor=(0.5,1.02), ncol=2, frameon=False, 
              prop=font_manager.FontProperties(family='Arial', size=35))

plt.subplots_adjust(top=0.85, bottom=0.08, left=0.08, right=0.98, hspace=0.3, wspace=0.3)
plt.savefig('RFIG2.pdf',format='pdf')