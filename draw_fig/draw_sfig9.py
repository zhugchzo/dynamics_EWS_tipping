import pandas
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

fig, axs = plt.subplots(1, 1, figsize=(7,7))

ax = axs

df_steady_rate = pandas.read_csv('../results/noise/auc_steady.csv')

###################################################

sigma = [0.0025, 0.01, 0.02, 0.03, 0.04, 0.05]

###################################################

dynamics_rate = df_steady_rate['dynamics_auc'].to_numpy()
dev_rate = df_steady_rate['dev_auc'].to_numpy()

###################################################

ax.plot(sigma,dynamics_rate,c='crimson',linewidth=4,zorder=3)
ax.scatter(sigma,dynamics_rate,c='crimson',marker='o',s=80,zorder=3)

ax.plot(sigma,dev_rate,c='darkorange',linewidth=4,zorder=1)
ax.scatter(sigma,dev_rate,c='darkorange',marker='o',s=80,zorder=1)

ax.set_xlim(-0.0025,0.055)
ax.set_xticks([0.0025, 0.01, 0.02, 0.03, 0.04, 0.05])
ax.set_xticklabels(['0.25', '1', '2', '3', '4', '5'])
ax.annotate(
    r'$\times 10^{-2}$',
    xy=(1, 0), xycoords='axes fraction',
    xytext=(8, -2), textcoords='offset points',
    ha='left', va='top',fontsize=18
)

ax.set_ylim(0.73,1.02)
ax.set_yticks([0.75,1])
ax.set_yticklabels(['0.75','1'])

ax.tick_params(direction='in', length=10)
ax.minorticks_on()
ax.tick_params(axis='x', which='minor', bottom=False)
ax.tick_params(axis='y', which='minor', direction='in', length=5)

ax.set_xlabel('Noise amplitude', font_x, labelpad=5)
ax.set_ylabel('AUC', font_y)
ax.yaxis.set_label_coords(-0.1, 0.5)

ax.set_title('Classification', fontsize=24, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.02)

################################################################################################

legend_dynamics = mlines.Line2D([], [], color='crimson',linestyle='-',linewidth=4,marker='o',markersize=12)
legend_dev = mlines.Line2D([], [], color='darkorange',linestyle='-',linewidth=4,marker='o',markersize=12)

# fig.legend(handles = [legend_dynamics, legend_dev],
#             labels = ['Dynamics-based EWS','Dynamical eigenvalue'],
#               loc='upper center', bbox_to_anchor=(0.5,1.02), ncol=1, frameon=False, 
#               prop=font_manager.FontProperties(family='Arial', size=20))

ax.legend(handles = [legend_dynamics, legend_dev],
            labels = ['Dynamics-based EWS','Dynamical eigenvalue'],
              loc='upper center', bbox_to_anchor=(0.57,0.25), ncol=1, frameon=False, 
              prop=font_manager.FontProperties(family='Arial', size=22))

plt.subplots_adjust(top=0.98, bottom=0.12, left=0.15, right=0.85)
plt.savefig('../figures/SFIG9.pdf',format='pdf')