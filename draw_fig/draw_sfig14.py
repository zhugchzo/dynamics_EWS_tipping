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
df_ER = pd.read_csv('../results/ROC/fold_ER/auc_partial.csv')
# df for ax2
df_BA = pd.read_csv('../results/ROC/fold_BA/auc_partial.csv')

node_ratio = [0.1, 0.2, 0.3, 0.4, 0.5]

data_ER = df_ER[['partial_auc', 'zero_ratio']].to_numpy()
data_BA = df_BA[['partial_auc', 'zero_ratio']].to_numpy()

################################################################################################

fig, axs = plt.subplots(1, 2, figsize=(12,7))

ax1, ax2 = axs[0], axs[1]

# ax1

auc_ax1 = data_ER[:,0]
ratio_ax1 = data_ER[:,1]

ax1.plot(node_ratio,auc_ax1,c='crimson',linewidth=4,zorder=3)
ax1.scatter(node_ratio,auc_ax1,c='crimson',marker='o',s=80,zorder=3)
ax1.plot(node_ratio, ratio_ax1, c='royalblue', linewidth=4, zorder=2)
ax1.scatter(node_ratio, ratio_ax1, c='royalblue', marker='o', s=80, zorder=2)

ax1.set_xlim(0.05,0.55)
ax1.set_xticks([0.1,0.2,0.3,0.4,0.5])
ax1.set_ylim(0.78,1.02)
ax1.set_yticks([0.8,0.9,1])
ax1.set_yticklabels(['0.8','0.9','1'])
ax1.tick_params(direction='in', length=10)
ax1.minorticks_on()
ax1.tick_params(axis='x', which='minor', bottom=False)
ax1.tick_params(axis='y', which='minor', direction='in', length=5)

ax1.set_xlabel(r'Observed fraction ($\rho_{\text{obs}}$)',font_x,labelpad=10)

# ax1.set_ylabel('AUC',font_y)
# ax1.yaxis.set_label_coords(-0.2, 0.5)
# ax1_right = ax1.twinx()
# ax1_right.set_ylim(0.75, 1.05)
# ax1_right.set_yticks([0.8, 0.9, 1])
# ax1_right.set_yticklabels(['0.8', '0.9', '1'])
# ax1_right.tick_params(direction='in', length=10)
# ax1_right.minorticks_on()
# ax1_right.tick_params(axis='y', which='minor', direction='in', length=5)
# ax1_right.set_ylabel('$\rho_{\text{Im=0}}$', font_y)

# ax1_right.yaxis.set_label_coords(1.2, 0.5)

ax1.set_title('ER network', fontsize=24, fontweight='normal', fontfamily='DejaVu Sans', fontstyle= 'italic', y=1.02)
ax1.text(0.1, 0.2,'(a)',ha='left', transform=ax1.transAxes,fontdict={'family':'Arial','size':40,'weight':'normal'})

# ax2

auc_ax2 = data_BA[:,0]
ratio_ax2 = data_BA[:,1]

ax2.plot(node_ratio,auc_ax2,c='crimson',linewidth=4,zorder=3)
ax2.scatter(node_ratio,auc_ax2,c='crimson',marker='o',s=80,zorder=3)
ax2.plot(node_ratio, ratio_ax2, c='royalblue', linewidth=4, zorder=2)
ax2.scatter(node_ratio, ratio_ax2, c='royalblue', marker='o', s=80, zorder=2)

ax2.set_xlim(0.05,0.55)
ax2.set_xticks([0.1,0.2,0.3,0.4,0.5])
ax2.set_ylim(0.78,1.02)
ax2.set_yticks([0.8,0.9,1])
ax2.set_yticklabels(['0.8','0.9','1'])
ax2.tick_params(direction='in', length=10)
ax2.minorticks_on()
ax2.tick_params(axis='x', which='minor', bottom=False)
ax2.tick_params(axis='y', which='minor', direction='in', length=5)

ax2.set_xlabel(r'Observed fraction ($\rho_{\text{obs}}$)',font_x,labelpad=10)

ax2.set_title('BA network', fontsize=24, fontweight='normal', fontfamily='DejaVu Sans', fontstyle= 'italic', y=1.02)
ax2.text(0.1, 0.2,'(b)',ha='left', transform=ax2.transAxes,fontdict={'family':'Arial','size':40,'weight':'normal'})

################################################################################################

legend_auc = mlines.Line2D([], [], color='crimson',linestyle='-',linewidth=4,marker='o',markersize=12)
legend_rho = mlines.Line2D([], [], color='royalblue',linestyle='-',linewidth=4,marker='o',markersize=12)

fig.legend(handles = [legend_auc, legend_rho],
            labels = ['AUC', r'$\rho_{\text{Im=0}}$'],
              loc='upper center', bbox_to_anchor=(0.5,1.04), ncol=2, frameon=False, 
              prop=font_manager.FontProperties(family='Arial', size=30))

plt.subplots_adjust(top=0.81, bottom=0.13, left=0.06, right=0.98, wspace=0.2)
plt.savefig('../figures/SFIG14.pdf',format='pdf')