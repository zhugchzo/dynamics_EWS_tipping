import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import font_manager

font_x = {'family':'Arial','weight':'normal','size': 24}
font_y = {'family':'Arial','weight':'normal','size': 24}

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['ytick.labelsize'] = 24

################################################################################################

n = 200

label = np.concatenate((np.ones(n), np.zeros(n)), axis=0)

################################################################################################

sup_hopf_sup_rate = pd.read_csv('../results/ROC/sup_hopf/sup_rate.csv')
sub_hopf_sup_rate = pd.read_csv('../results/ROC/sub_hopf/sup_rate.csv')

################################################################################################

fig, axs = plt.subplots(1, 1, figsize=(5,5))

ax = axs
ax.set_aspect('equal', adjustable='box')

################################################################################################

sup_rate_sup_hopf = sup_hopf_sup_rate['sup_rate'].to_numpy()
sup_rate_sub_hopf = sub_hopf_sup_rate['sup_rate'].to_numpy()

sup_rate = np.concatenate((sup_rate_sup_hopf, sup_rate_sub_hopf), axis=0)

fpr_sup_rate, tpr_sup_rate, thresholds_sup_rate = roc_curve(label, sup_rate)
auc_sup_rate = auc(fpr_sup_rate, tpr_sup_rate)

textstr = '\n'.join([
    r'$\text{A}_{\text{UEWS}} = %.2f$' % auc_sup_rate
])

ax.plot(fpr_sup_rate, tpr_sup_rate, c='crimson',linewidth=4)
ax.plot([0, 1], [0, 1], linestyle='--', linewidth=2, color='silver', zorder=0)

ax.text(0.45, 0.35, textstr, transform=ax.transAxes, fontsize=24,
         verticalalignment='top')

ax.set_xlabel('False positive rate',font_x)
ax.set_ylabel('True positive rate',font_x)

ax.set_xticks([0,0.5,1])
ax.set_xticklabels(['0','0.5','1'])
ax.set_yticks([0,0.5,1])
ax.set_yticklabels(['0','0.5','1'])

# ax.set_title('Classification', fontsize=12, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.02)

################################################################################################

# legend_dynamics = mlines.Line2D([], [], color='crimson',linestyle='-',linewidth=3)
# legend_ac = mlines.Line2D([], [], color='darkslateblue',linestyle='-',linewidth=3)
# legend_dev = mlines.Line2D([], [], color='darkorange',linestyle='-',linewidth=3)

# fig.legend(handles = [legend_dynamics, legend_ac, legend_dev],
#             labels = ['Universal EWS','Lag-1 AC','Dynamical eigenvalue'],
#               loc='upper center', bbox_to_anchor=(0.5,1.01), ncol=3, frameon=False, 
#               prop=font_manager.FontProperties(family='Arial', size=15))

plt.subplots_adjust(top=0.96, bottom=0.16, left=0.2, right=0.99)
plt.savefig('../figures/SFIG1.pdf',format='pdf')