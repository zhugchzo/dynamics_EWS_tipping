import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import font_manager

font_x = {'family':'Arial','weight':'normal','size': 12}
font_y = {'family':'Arial','weight':'normal','size': 12}

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

################################################################################################

n = 100

label = np.concatenate((np.ones(n), np.zeros(n)), axis=0)

################################################################################################

fold_digest = pd.read_csv('../results/ROC/fold/DIGEST_tau.csv')
fold_null_digest = pd.read_csv('../results/ROC/fold/DIGEST_tau_null.csv')

fold_ac = pd.read_csv('../results/ROC/fold/lag-1_tau.csv')
fold_null_ac = pd.read_csv('../results/ROC/fold/lag-1_tau_null.csv')

fold_dev = pd.read_csv('../results/ROC/fold/DEV_tau.csv')
fold_null_dev = pd.read_csv('../results/ROC/fold/DEV_tau_null.csv')

########################

fold_network_digest = pd.read_csv('../results/ROC/fold_network/DIGEST_tau.csv')
fold_network_null_digest = pd.read_csv('../results/ROC/fold_network/DIGEST_tau_null.csv')

fold_network_ac = pd.read_csv('../results/ROC/fold_network/lag-1_tau.csv')
fold_network_null_ac = pd.read_csv('../results/ROC/fold_network/lag-1_tau_null.csv')

fold_network_dev = pd.read_csv('../results/ROC/fold_network/DEV_tau.csv')
fold_network_null_dev = pd.read_csv('../results/ROC/fold_network/DEV_tau_null.csv')

########################

sup_hopf_digest = pd.read_csv('../results/ROC/sup_hopf/DIGEST_tau.csv')
sup_hopf_null_digest = pd.read_csv('../results/ROC/sup_hopf/DIGEST_tau_null.csv')

sup_hopf_ac = pd.read_csv('../results/ROC/sup_hopf/lag-1_tau.csv')
sup_hopf_null_ac = pd.read_csv('../results/ROC/sup_hopf/lag-1_tau_null.csv')

sup_hopf_dev = pd.read_csv('../results/ROC/sup_hopf/DEV_tau.csv')
sup_hopf_null_dev = pd.read_csv('../results/ROC/sup_hopf/DEV_tau_null.csv')

########################

trans_digest = pd.read_csv('../results/ROC/trans/DIGEST_tau.csv')
trans_null_digest = pd.read_csv('../results/ROC/trans/DIGEST_tau_null.csv')

trans_ac = pd.read_csv('../results/ROC/trans/lag-1_tau.csv')
trans_null_ac = pd.read_csv('../results/ROC/trans/lag-1_tau_null.csv')

trans_dev = pd.read_csv('../results/ROC/trans/DEV_tau.csv')
trans_null_dev = pd.read_csv('../results/ROC/trans/DEV_tau_null.csv')

########################

sub_hopf_digest = pd.read_csv('../results/ROC/sub_hopf/DIGEST_tau.csv')
sub_hopf_null_digest = pd.read_csv('../results/ROC/sub_hopf/DIGEST_tau_null.csv')

sub_hopf_ac = pd.read_csv('../results/ROC/sub_hopf/lag-1_tau.csv')
sub_hopf_null_ac = pd.read_csv('../results/ROC/sub_hopf/lag-1_tau_null.csv')

sub_hopf_dev = pd.read_csv('../results/ROC/sub_hopf/DEV_tau.csv')
sub_hopf_null_dev = pd.read_csv('../results/ROC/sub_hopf/DEV_tau_null.csv')

################################################################################################

fig, axs = plt.subplots(1, 5, figsize=(12,3))

for ax in axs:
    ax.set_aspect('equal') 

ax1, ax2, ax3, ax4, ax5 = axs[0], axs[1], axs[2], axs[3], axs[4]

################################################################################################

digest_tau = fold_digest['tau'].to_numpy()
null_digest_tau = fold_null_digest['tau'].to_numpy()

digest = np.concatenate((digest_tau, null_digest_tau), axis=0)

fpr_digest, tpr_digest, thresholds_digest = roc_curve(label, digest)
auc_digest = auc(fpr_digest, tpr_digest)

########################

ac_tau = fold_ac['tau'].to_numpy()
null_ac_tau = fold_null_ac['tau'].to_numpy()

ac = np.concatenate((ac_tau, null_ac_tau), axis=0)

label_ac = np.concatenate((np.ones(len(ac_tau)), np.zeros(n)), axis=0)

fpr_ac, tpr_ac, thresholds_ac = roc_curve(label_ac, ac)
auc_ac = auc(fpr_ac, tpr_ac)

########################

dev_tau = fold_dev['tau'].to_numpy()
null_dev_tau = fold_null_dev['tau'].to_numpy()

dev = np.concatenate((dev_tau, null_dev_tau), axis=0)

fpr_dev, tpr_dev, thresholds_dev = roc_curve(label, dev)
auc_dev = auc(fpr_dev, tpr_dev)

textstr = '\n'.join([
    r'$\text{A}_{\text{UEWS}} = %.2f$' % auc_digest,
    r'$\text{A}_{\text{AC}} = %.2f$' % auc_ac,
    r'$\text{A}_{\text{DEV}} = %.2f$' % auc_dev
])

ax1.plot(fpr_digest, tpr_digest, c='crimson',zorder=2)
ax1.plot(fpr_ac, tpr_ac, c='royalblue',zorder=1)
ax1.plot(fpr_dev, tpr_dev, c='forestgreen',zorder=1)
ax1.plot([0, 1], [0, 1], linestyle='--', linewidth=1, color='silver', zorder=0)

ax1.text(0.45, 0.35, textstr, transform=ax1.transAxes, fontsize=10,
         verticalalignment='top')

ax1.set_xlabel('False positive rate',font_x)
ax1.set_ylabel('True positive rate',font_y)

ax1.set_xticks([0,0.5,1])
ax1.set_xticklabels(['0','0.5','1'])
ax1.set_yticks([0,0.5,1])
ax1.set_yticklabels(['0','0.5','1'])

ax1.text(-0.25, 1,'a',ha='left', transform=ax1.transAxes,fontdict={'family':'DejaVu Sans','size':18,'weight':'bold'})

################################################################################################

digest_tau = fold_network_digest['tau'].to_numpy()
null_digest_tau = fold_network_null_digest['tau'].to_numpy()

digest = np.concatenate((digest_tau, null_digest_tau), axis=0)

fpr_digest, tpr_digest, thresholds_digest = roc_curve(label, digest)
auc_digest = auc(fpr_digest, tpr_digest)

########################

ac_tau = fold_network_ac['tau'].to_numpy()
null_ac_tau = fold_network_null_ac['tau'].to_numpy()

ac = np.concatenate((ac_tau, null_ac_tau), axis=0)

label_ac = np.concatenate((np.ones(len(ac_tau)), np.zeros(n)), axis=0)

fpr_ac, tpr_ac, thresholds_ac = roc_curve(label_ac, ac)
auc_ac = auc(fpr_ac, tpr_ac)

########################

dev_tau = fold_network_dev['tau'].to_numpy()
null_dev_tau = fold_network_null_dev['tau'].to_numpy()

dev = np.concatenate((dev_tau, null_dev_tau), axis=0)

fpr_dev, tpr_dev, thresholds_dev = roc_curve(label, dev)
auc_dev = auc(fpr_dev, tpr_dev)

textstr = '\n'.join([
    r'$\text{A}_{\text{UEWS}} = %.2f$' % auc_digest,
    r'$\text{A}_{\text{AC}} = %.2f$' % auc_ac,
    r'$\text{A}_{\text{DEV}} = %.2f$' % auc_dev
])

ax2.plot(fpr_digest, tpr_digest, c='crimson',zorder=2)
ax2.plot(fpr_ac, tpr_ac, c='royalblue',zorder=1)
ax2.plot(fpr_dev, tpr_dev, c='forestgreen',zorder=1)
ax2.plot([0, 1], [0, 1], linestyle='--', linewidth=1, color='silver', zorder=0)

ax2.text(0.45, 0.35, textstr, transform=ax2.transAxes, fontsize=10,
         verticalalignment='top')

ax2.set_xlabel('False positive rate',font_x)

ax2.set_xticks([0,0.5,1])
ax2.set_xticklabels(['0','0.5','1'])
ax2.set_yticks([0,0.5,1])
ax2.set_yticklabels(['0','0.5','1'])

ax2.text(-0.25, 1,'b',ha='left', transform=ax2.transAxes,fontdict={'family':'DejaVu Sans','size':18,'weight':'bold'})

################################################################################################

digest_tau = sup_hopf_digest['tau'].to_numpy()
null_digest_tau = sup_hopf_null_digest['tau'].to_numpy()

digest = np.concatenate((digest_tau, null_digest_tau), axis=0)

fpr_digest, tpr_digest, thresholds_digest = roc_curve(label, digest)
auc_digest = auc(fpr_digest, tpr_digest)

########################

ac_tau = sup_hopf_ac['tau'].to_numpy()
null_ac_tau = sup_hopf_null_ac['tau'].to_numpy()

ac = np.concatenate((ac_tau, null_ac_tau), axis=0)

label_ac = np.concatenate((np.ones(len(ac_tau)), np.zeros(n)), axis=0)

fpr_ac, tpr_ac, thresholds_ac = roc_curve(label_ac, ac)
auc_ac = auc(fpr_ac, tpr_ac)

########################

dev_tau = sup_hopf_dev['tau'].to_numpy()
null_dev_tau = sup_hopf_null_dev['tau'].to_numpy()

dev = np.concatenate((dev_tau, null_dev_tau), axis=0)

fpr_dev, tpr_dev, thresholds_dev = roc_curve(label, dev)
auc_dev = auc(fpr_dev, tpr_dev)

textstr = '\n'.join([
    r'$\text{A}_{\text{UEWS}} = %.2f$' % auc_digest,
    r'$\text{A}_{\text{AC}} = %.2f$' % auc_ac,
    r'$\text{A}_{\text{DEV}} = %.2f$' % auc_dev
])

ax3.plot(fpr_digest, tpr_digest, c='crimson',zorder=2)
ax3.plot(fpr_ac, tpr_ac, c='royalblue',zorder=1)
ax3.plot(fpr_dev, tpr_dev, c='forestgreen',zorder=1)
ax3.plot([0, 1], [0, 1], linestyle='--', linewidth=1, color='silver', zorder=0)

ax3.text(0.08, 0.48, textstr, transform=ax3.transAxes, fontsize=10,
         verticalalignment='top')

ax3.set_xlabel('False positive rate',font_x)

ax3.set_xticks([0,0.5,1])
ax3.set_xticklabels(['0','0.5','1'])
ax3.set_yticks([0,0.5,1])
ax3.set_yticklabels(['0','0.5','1'])

ax3.text(-0.25, 1,'c',ha='left', transform=ax3.transAxes,fontdict={'family':'DejaVu Sans','size':18,'weight':'bold'})

################################################################################################

digest_tau = trans_digest['tau'].to_numpy()
null_digest_tau = trans_null_digest['tau'].to_numpy()

digest = np.concatenate((digest_tau, null_digest_tau), axis=0)

fpr_digest, tpr_digest, thresholds_digest = roc_curve(label, digest)
auc_digest = auc(fpr_digest, tpr_digest)

########################

ac_tau = trans_ac['tau'].to_numpy()
null_ac_tau = trans_null_ac['tau'].to_numpy()

ac = np.concatenate((ac_tau, null_ac_tau), axis=0)

label_ac = np.concatenate((np.ones(len(ac_tau)), np.zeros(n)), axis=0)

fpr_ac, tpr_ac, thresholds_ac = roc_curve(label_ac, ac)
auc_ac = auc(fpr_ac, tpr_ac)

########################

dev_tau = trans_dev['tau'].to_numpy()
null_dev_tau = trans_null_dev['tau'].to_numpy()

dev = np.concatenate((dev_tau, null_dev_tau), axis=0)

fpr_dev, tpr_dev, thresholds_dev = roc_curve(label, dev)
auc_dev = auc(fpr_dev, tpr_dev)

textstr = '\n'.join([
    r'$\text{A}_{\text{UEWS}} = %.2f$' % auc_digest,
    r'$\text{A}_{\text{AC}} = %.2f$' % auc_ac,
    r'$\text{A}_{\text{DEV}} = %.2f$' % auc_dev
])

ax4.plot(fpr_digest, tpr_digest, c='crimson',zorder=2)
ax4.plot(fpr_ac, tpr_ac, c='royalblue',zorder=1)
ax4.plot(fpr_dev, tpr_dev, c='forestgreen',zorder=1)
ax4.plot([0, 1], [0, 1], linestyle='--', linewidth=1, color='silver', zorder=0)

ax4.text(0.45, 0.35, textstr, transform=ax4.transAxes, fontsize=10,
         verticalalignment='top')

ax4.set_xlabel('False positive rate',font_x)

ax4.set_xticks([0,0.5,1])
ax4.set_xticklabels(['0','0.5','1'])
ax4.set_yticks([0,0.5,1])
ax4.set_yticklabels(['0','0.5','1'])

ax4.text(-0.25, 1,'d',ha='left', transform=ax4.transAxes,fontdict={'family':'DejaVu Sans','size':18,'weight':'bold'})

################################################################################################

digest_tau = sub_hopf_digest['tau'].to_numpy()
null_digest_tau = sub_hopf_null_digest['tau'].to_numpy()

digest = np.concatenate((digest_tau, null_digest_tau), axis=0)

fpr_digest, tpr_digest, thresholds_digest = roc_curve(label, digest)
auc_digest = auc(fpr_digest, tpr_digest)

########################

ac_tau = sub_hopf_ac['tau'].to_numpy()
null_ac_tau = sub_hopf_null_ac['tau'].to_numpy()

ac = np.concatenate((ac_tau, null_ac_tau), axis=0)

label_ac = np.concatenate((np.ones(len(ac_tau)), np.zeros(n)), axis=0)

fpr_ac, tpr_ac, thresholds_ac = roc_curve(label_ac, ac)
auc_ac = auc(fpr_ac, tpr_ac)

########################

dev_tau = sub_hopf_dev['tau'].to_numpy()
null_dev_tau = sub_hopf_null_dev['tau'].to_numpy()

dev = np.concatenate((dev_tau, null_dev_tau), axis=0)

fpr_dev, tpr_dev, thresholds_dev = roc_curve(label, dev)
auc_dev = auc(fpr_dev, tpr_dev)

textstr = '\n'.join([
    r'$\text{A}_{\text{UEWS}} = %.2f$' % auc_digest,
    r'$\text{A}_{\text{AC}} = %.2f$' % auc_ac,
    r'$\text{A}_{\text{DEV}} = %.2f$' % auc_dev
])

ax5.plot(fpr_digest, tpr_digest, c='crimson',zorder=2)
ax5.plot(fpr_ac, tpr_ac, c='royalblue',zorder=1)
ax5.plot(fpr_dev, tpr_dev, c='forestgreen',zorder=1)
ax5.plot([0, 1], [0, 1], linestyle='--', linewidth=1, color='silver', zorder=0)

ax5.text(0.45, 0.35, textstr, transform=ax5.transAxes, fontsize=10,
         verticalalignment='top')

ax5.set_xlabel('False positive rate',font_x)

ax5.set_xticks([0,0.5,1])
ax5.set_xticklabels(['0','0.5','1'])
ax5.set_yticks([0,0.5,1])
ax5.set_yticklabels(['0','0.5','1'])

ax5.text(-0.25, 1,'e',ha='left', transform=ax5.transAxes,fontdict={'family':'DejaVu Sans','size':18,'weight':'bold'})

################################################################################################

legend_digest = mlines.Line2D([], [], color='crimson',linestyle='-',linewidth=3)
legend_ac = mlines.Line2D([], [], color='royalblue',linestyle='-',linewidth=3)
legend_dev = mlines.Line2D([], [], color='forestgreen',linestyle='-',linewidth=3)

fig.legend(handles = [legend_digest, legend_ac, legend_dev],
            labels = ['Universal EWS','Lag-1 AC','Dynamical eigenvalue'],
              loc='upper center', bbox_to_anchor=(0.5,1.01), ncol=3, frameon=False, 
              prop=font_manager.FontProperties(family='Arial Unicode MS', size=15))

plt.subplots_adjust(top=0.85, bottom=0.1, left=0.06, right=0.99, wspace=0.3)
plt.savefig('../figures/FIG6.pdf',format='pdf',dpi=600)


