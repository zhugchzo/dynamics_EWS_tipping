import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import font_manager

font_x = {'family':'Arial','weight':'medium','size': 25}
font_y = {'family':'Arial','weight':'medium','size': 25}

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['xtick.labelsize'] = 25
plt.rcParams['ytick.labelsize'] = 25

# df for ax1
df_ER = pd.read_csv('../results/ROC/fold_ER/auc.csv')
# df for ax2
df_BA = pd.read_csv('../results/ROC/fold_BA/auc.csv')

network_size = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60]

auc_ER = df_ER[['dynamics_auc', 'ac_auc', 'dev_auc']].to_numpy()
auc_BA = df_BA[['dynamics_auc', 'ac_auc', 'dev_auc']].to_numpy()

################################################################################################

fig, axs = plt.subplots(1, 2, figsize=(12,6))

ax1, ax2 = axs[0], axs[1]

# ax1

dynamics_auc_ax1 = auc_ER[:,0]
ac_auc_ax1 = auc_ER[:,1]
dev_auc_ax1 = auc_ER[:,2]

ax1.plot(network_size,dynamics_auc_ax1,c='crimson',linewidth=3,zorder=3)
ax1.scatter(network_size,dynamics_auc_ax1,c='crimson',marker='o',s=50,zorder=3)

ax1.plot(network_size,ac_auc_ax1,c='darkslateblue',linewidth=3,zorder=2)
ax1.scatter(network_size,ac_auc_ax1,c='darkslateblue',marker='o',s=50,zorder=2)

ax1.plot(network_size,dev_auc_ax1,c='darkorange',linewidth=3,zorder=1)
ax1.scatter(network_size,dev_auc_ax1,c='darkorange',marker='o',s=50,zorder=1)

ax1.set_xlim(12,63)
ax1.set_xticks([15,25,35,45,55])
ax1.set_ylim(0.55,1.05)
ax1.set_yticks([0.6,1])
ax1.tick_params(direction='in', length=10)

ax1.set_xlabel('Network size',font_x)
ax1.set_ylabel('AUC',font_y)
ax1.yaxis.set_label_coords(-0.05, 0.5)

ax1.set_title('ER network', fontsize=24, fontweight='normal', fontfamily='DejaVu Sans', fontstyle= 'italic', y=1.02)
ax1.text(0.05, 0.25,'(a)',ha='left', transform=ax1.transAxes,fontdict={'family':'Arial','size':30,'weight':'normal'})

# ax2

dynamics_auc_ax2 = auc_BA[:,0]
ac_auc_ax2 = auc_BA[:,1]
dev_auc_ax2 = auc_BA[:,2]

ax2.plot(network_size,dynamics_auc_ax2,c='crimson',linewidth=3,zorder=3)
ax2.scatter(network_size,dynamics_auc_ax2,c='crimson',marker='o',s=50,zorder=3)

ax2.plot(network_size,ac_auc_ax2,c='darkslateblue',linewidth=3,zorder=2)
ax2.scatter(network_size,ac_auc_ax2,c='darkslateblue',marker='o',s=50,zorder=2)

ax2.plot(network_size,dev_auc_ax2,c='darkorange',linewidth=3,zorder=1)
ax2.scatter(network_size,dev_auc_ax2,c='darkorange',marker='o',s=50,zorder=1)

ax2.set_xlim(12,63)
ax2.set_xticks([15,25,35,45,55])
ax2.set_ylim(0.55,1.05)
ax2.set_yticks([0.6,1])
ax2.tick_params(direction='in', length=10)

ax2.set_xlabel('Network size',font_x)

ax2.set_title('BA network', fontsize=24, fontweight='normal', fontfamily='DejaVu Sans', fontstyle= 'italic', y=1.02)
ax2.text(0.05, 0.25,'(b)',ha='left', transform=ax2.transAxes,fontdict={'family':'Arial','size':30,'weight':'normal'})

################################################################################################

legend_dynamics = mlines.Line2D([], [], color='crimson',linestyle='-',linewidth=3,marker='o',markersize=10)
legend_ac = mlines.Line2D([], [], color='darkslateblue',linestyle='-',linewidth=3,marker='o',markersize=10)
legend_dev = mlines.Line2D([], [], color='darkorange',linestyle='-',linewidth=3,marker='o',markersize=10)

fig.legend(handles = [legend_dynamics, legend_ac, legend_dev],
            labels = ['Universal EWS','Lag-1 AC','Dynamical eigenvalue'],
              loc='upper center', bbox_to_anchor=(0.5,1.04), ncol=3, frameon=False, 
              prop=font_manager.FontProperties(family='Arial', size=24))

plt.subplots_adjust(top=0.81, bottom=0.13, left=0.06, right=0.98, wspace=0.2)
plt.savefig('../figures/FIG3.pdf',format='pdf')