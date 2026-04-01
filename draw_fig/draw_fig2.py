import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import font_manager

font_x = {'family':'Arial','weight':'normal','size': 14}
font_y = {'family':'Arial','weight':'normal','size': 14}

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

################################################################################################

n = 200

label = np.concatenate((np.ones(n), np.zeros(n)), axis=0)
label_steady = np.concatenate((np.ones(n), np.ones(n), np.ones(n), np.zeros(n), np.zeros(n)), axis=0)

################################################################################################

fold_ER_dynamics = pd.read_csv('../results/ROC/fold_ER/dynamics_tau.csv')
fold_ER_null_dynamics = pd.read_csv('../results/ROC/fold_ER/dynamics_tau_null.csv')

fold_ER_ac = pd.read_csv('../results/ROC/fold_ER/lag-1_tau.csv')
fold_ER_null_ac = pd.read_csv('../results/ROC/fold_ER/lag-1_tau_null.csv')

fold_ER_dev = pd.read_csv('../results/ROC/fold_ER/DEV_tau.csv')
fold_ER_null_dev = pd.read_csv('../results/ROC/fold_ER/DEV_tau_null.csv')

fold_BA_dynamics = pd.read_csv('../results/ROC/fold_BA/dynamics_tau.csv')
fold_BA_null_dynamics = pd.read_csv('../results/ROC/fold_BA/dynamics_tau_null.csv')

fold_BA_ac = pd.read_csv('../results/ROC/fold_BA/lag-1_tau.csv')
fold_BA_null_ac = pd.read_csv('../results/ROC/fold_BA/lag-1_tau_null.csv')

fold_BA_dev = pd.read_csv('../results/ROC/fold_BA/DEV_tau.csv')
fold_BA_null_dev = pd.read_csv('../results/ROC/fold_BA/DEV_tau_null.csv')

########################

trans_dynamics = pd.read_csv('../results/ROC/trans/dynamics_tau.csv')
trans_null_dynamics = pd.read_csv('../results/ROC/trans/dynamics_tau_null.csv')

trans_ac = pd.read_csv('../results/ROC/trans/lag-1_tau.csv')
trans_null_ac = pd.read_csv('../results/ROC/trans/lag-1_tau_null.csv')

trans_dev = pd.read_csv('../results/ROC/trans/DEV_tau.csv')
trans_null_dev = pd.read_csv('../results/ROC/trans/DEV_tau_null.csv')

########################

sup_hopf_dynamics = pd.read_csv('../results/ROC/sup_hopf/dynamics_tau.csv')
sup_hopf_null_dynamics = pd.read_csv('../results/ROC/sup_hopf/dynamics_tau_null.csv')

sup_hopf_ac = pd.read_csv('../results/ROC/sup_hopf/lag-1_tau.csv')
sup_hopf_null_ac = pd.read_csv('../results/ROC/sup_hopf/lag-1_tau_null.csv')

sup_hopf_dev = pd.read_csv('../results/ROC/sup_hopf/DEV_tau.csv')
sup_hopf_null_dev = pd.read_csv('../results/ROC/sup_hopf/DEV_tau_null.csv')

########################

sub_hopf_dynamics = pd.read_csv('../results/ROC/sub_hopf/dynamics_tau.csv')
sub_hopf_null_dynamics = pd.read_csv('../results/ROC/sub_hopf/dynamics_tau_null.csv')

sub_hopf_ac = pd.read_csv('../results/ROC/sub_hopf/lag-1_tau.csv')
sub_hopf_null_ac = pd.read_csv('../results/ROC/sub_hopf/lag-1_tau_null.csv')

sub_hopf_dev = pd.read_csv('../results/ROC/sub_hopf/DEV_tau.csv')
sub_hopf_null_dev = pd.read_csv('../results/ROC/sub_hopf/DEV_tau_null.csv')

################################################################################################

fold_ER_dynamics_rate = pd.read_csv('../results/ROC/fold_ER/dynamics_rate.csv')
fold_ER_dev_rate = pd.read_csv('../results/ROC/fold_ER/DEV_rate.csv')

fold_BA_dynamics_rate = pd.read_csv('../results/ROC/fold_BA/dynamics_rate.csv')
fold_BA_dev_rate = pd.read_csv('../results/ROC/fold_BA/DEV_rate.csv')

sup_hopf_dynamics_rate = pd.read_csv('../results/ROC/sup_hopf/dynamics_rate.csv')
sup_hopf_dev_rate = pd.read_csv('../results/ROC/sup_hopf/DEV_rate.csv')

trans_dynamics_rate = pd.read_csv('../results/ROC/trans/dynamics_rate.csv')
trans_dev_rate = pd.read_csv('../results/ROC/trans/DEV_rate.csv')

sub_hopf_dynamics_rate = pd.read_csv('../results/ROC/sub_hopf/dynamics_rate.csv')
sub_hopf_dev_rate = pd.read_csv('../results/ROC/sub_hopf/DEV_rate.csv')

################################################################################################

fig, axs = plt.subplots(1, 6, figsize=(15,3.5))

for ax in axs.flat:
    ax.set_aspect('equal', adjustable='box')

ax1, ax2, ax3, ax4, ax5, ax6 = axs[0], axs[1], axs[2], axs[3], axs[4], axs[5]

################################################################################################

dynamics_tau_ER = fold_ER_dynamics['tau'].to_numpy()
null_dynamics_tau_ER = fold_ER_null_dynamics['tau'].to_numpy()

dynamics_tau_BA = fold_BA_dynamics['tau'].to_numpy()
null_dynamics_tau_BA = fold_BA_null_dynamics['tau'].to_numpy()

dynamics_ER = np.concatenate((dynamics_tau_ER, null_dynamics_tau_ER), axis=0)
dynamics_BA = np.concatenate((dynamics_tau_BA, null_dynamics_tau_BA), axis=0)

fpr_dynamics_ER, tpr_dynamics_ER, thresholds_dynamics_ER = roc_curve(label, dynamics_ER)
auc_dynamics_ER = auc(fpr_dynamics_ER, tpr_dynamics_ER)

fpr_dynamics_BA, tpr_dynamics_BA, thresholds_dynamics_BA = roc_curve(label, dynamics_BA)
auc_dynamics_BA = auc(fpr_dynamics_BA, tpr_dynamics_BA)

########################

ac_tau_ER = fold_ER_ac['tau'].to_numpy()
null_ac_tau_ER = fold_ER_null_ac['tau'].to_numpy()

ac_tau_BA = fold_BA_ac['tau'].to_numpy()
null_ac_tau_BA = fold_BA_null_ac['tau'].to_numpy()

ac_ER = np.concatenate((ac_tau_ER, null_ac_tau_ER), axis=0)
ac_BA = np.concatenate((ac_tau_BA, null_ac_tau_BA), axis=0)

fpr_ac_ER, tpr_ac_ER, thresholds_ac_ER = roc_curve(label, ac_ER)
auc_ac_ER = auc(fpr_ac_ER, tpr_ac_ER)

fpr_ac_BA, tpr_ac_BA, thresholds_ac_BA = roc_curve(label, ac_BA)
auc_ac_BA = auc(fpr_ac_BA, tpr_ac_BA)

########################

dev_tau_ER = fold_ER_dev['tau'].to_numpy()
null_dev_tau_ER = fold_ER_null_dev['tau'].to_numpy()

dev_tau_BA = fold_BA_dev['tau'].to_numpy()
null_dev_tau_BA = fold_BA_null_dev['tau'].to_numpy()

dev_ER = np.concatenate((dev_tau_ER, null_dev_tau_ER), axis=0)
dev_BA = np.concatenate((dev_tau_BA, null_dev_tau_BA), axis=0)

fpr_dev_ER, tpr_dev_ER, thresholds_dev_ER = roc_curve(label, dev_ER)
auc_dev_ER = auc(fpr_dev_ER, tpr_dev_ER)

fpr_dev_BA, tpr_dev_BA, thresholds_dev_BA = roc_curve(label, dev_BA)
auc_dev_BA = auc(fpr_dev_BA, tpr_dev_BA)

################################################################################################

dynamics_tau = trans_dynamics['tau'].to_numpy()
null_dynamics_tau = trans_null_dynamics['tau'].to_numpy()

dynamics = np.concatenate((dynamics_tau, null_dynamics_tau), axis=0)

fpr_dynamics, tpr_dynamics, thresholds_dynamics = roc_curve(label, dynamics)
auc_dynamics = auc(fpr_dynamics, tpr_dynamics)

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
    r'$\text{A}_{\text{DEWS}} = %.2f$' % auc_dynamics,
    r'$\text{A}_{\text{AC}} = %.2f$' % auc_ac,
    r'$\text{A}_{\text{DEV}} = %.2f$' % auc_dev
])

ax1.plot(fpr_dynamics, tpr_dynamics, c='crimson',linewidth=2,zorder=3)
ax1.plot(fpr_ac, tpr_ac, c='darkslateblue',linewidth=2,zorder=2)
ax1.plot(fpr_dev, tpr_dev, c='darkorange',linewidth=2,zorder=1)
ax1.plot([0, 1], [0, 1], linestyle='--', linewidth=1, color='silver', zorder=0)

ax1.text(0.45, 0.35, textstr, transform=ax1.transAxes, fontsize=12,
         verticalalignment='top')

ax1.set_xlabel('False positive rate',font_x)
ax1.set_ylabel('True positive rate',font_y)

ax1.set_xticks([0,0.5,1])
ax1.set_xticklabels(['0','0.5','1'])
ax1.set_yticks([0,0.5,1])
ax1.set_yticklabels(['0','0.5','1'])

ax1.set_title('Predator-prey (transcritical)', fontsize=12, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.02)
ax1.text(0.83, 0.75,'(a)',ha='left', transform=ax1.transAxes,fontdict={'family':'Arial','size':18,'weight':'normal'})

################################################################################################

dynamics_tau = sup_hopf_dynamics['tau'].to_numpy()
null_dynamics_tau = sup_hopf_null_dynamics['tau'].to_numpy()

dynamics = np.concatenate((dynamics_tau, null_dynamics_tau), axis=0)

fpr_dynamics, tpr_dynamics, thresholds_dynamics = roc_curve(label, dynamics)
auc_dynamics = auc(fpr_dynamics, tpr_dynamics)

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
    r'$\text{A}_{\text{DEWS}} = %.2f$' % auc_dynamics,
    r'$\text{A}_{\text{AC}} = %.2f$' % auc_ac,
    r'$\text{A}_{\text{DEV}} = %.2f$' % auc_dev
])

ax2.plot(fpr_dynamics, tpr_dynamics, c='crimson',linewidth=2,zorder=3)
ax2.plot(fpr_ac, tpr_ac, c='darkslateblue',linewidth=2,zorder=2)
ax2.plot(fpr_dev, tpr_dev, c='darkorange',linewidth=2,zorder=1)
ax2.plot([0, 1], [0, 1], linestyle='--', linewidth=1, color='silver', zorder=0)

ax2.text(0.45, 0.35, textstr, transform=ax2.transAxes, fontsize=12,
         verticalalignment='top')

ax2.set_xlabel('False positive rate',font_x)

ax2.set_xticks([0,0.5,1])
ax2.set_xticklabels(['0','0.5','1'])
ax2.set_yticks([0,0.5,1])
ax2.set_yticklabels(['0','0.5','1'])

ax2.set_title('Predator-prey (Hopf)', fontsize=12, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.02)
ax2.text(0.83, 0.75,'(b)',ha='left', transform=ax2.transAxes,fontdict={'family':'Arial','size':18,'weight':'normal'})

################################################################################################

dynamics_tau = sub_hopf_dynamics['tau'].to_numpy()
null_dynamics_tau = sub_hopf_null_dynamics['tau'].to_numpy()

dynamics = np.concatenate((dynamics_tau, null_dynamics_tau), axis=0)

fpr_dynamics, tpr_dynamics, thresholds_dynamics = roc_curve(label, dynamics)
auc_dynamics = auc(fpr_dynamics, tpr_dynamics)

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
    r'$\text{A}_{\text{DEWS}} = %.2f$' % auc_dynamics,
    r'$\text{A}_{\text{AC}} = %.2f$' % auc_ac,
    r'$\text{A}_{\text{DEV}} = %.2f$' % auc_dev
])

ax3.plot(fpr_dynamics, tpr_dynamics, c='crimson',linewidth=2,zorder=3)
ax3.plot(fpr_ac, tpr_ac, c='darkslateblue',linewidth=2,zorder=2)
ax3.plot(fpr_dev, tpr_dev, c='darkorange',linewidth=2,zorder=1)
ax3.plot([0, 1], [0, 1], linestyle='--', linewidth=1, color='silver', zorder=0)

ax3.text(0.45, 0.35, textstr, transform=ax3.transAxes, fontsize=12,
         verticalalignment='top',zorder=3)

ax3.set_xlabel('False positive rate',font_x)

ax3.set_xticks([0,0.5,1])
ax3.set_xticklabels(['0','0.5','1'])
ax3.set_yticks([0,0.5,1])
ax3.set_yticklabels(['0','0.5','1'])

ax3.set_title('Epidemic', fontsize=12, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.02)
ax3.text(0.83, 0.75,'(c)',ha='left', transform=ax3.transAxes,fontdict={'family':'Arial','size':18,'weight':'normal'})

################################################################################################

textstr = '\n'.join([
    r'$\text{A}_{\text{DEWS}} = %.2f$' % auc_dynamics_ER,
    r'$\text{A}_{\text{AC}} = %.2f$' % auc_ac_ER,
    r'$\text{A}_{\text{DEV}} = %.2f$' % auc_dev_ER
])

ax4.plot(fpr_dynamics_ER, tpr_dynamics_ER, c='crimson',linewidth=2,zorder=3)
ax4.plot(fpr_ac_ER, tpr_ac_ER, c='darkslateblue',linewidth=2,zorder=2)
ax4.plot(fpr_dev_ER, tpr_dev_ER, c='darkorange',linewidth=2,zorder=1)
ax4.plot([0, 1], [0, 1], linestyle='--', linewidth=1, color='silver', zorder=0)

ax4.text(0.45, 0.35, textstr, transform=ax4.transAxes, fontsize=12,
         verticalalignment='top')

ax4.set_xlabel('False positive rate',font_x)

ax4.set_xticks([0,0.5,1])
ax4.set_xticklabels(['0','0.5','1'])
ax4.set_yticks([0,0.5,1])
ax4.set_yticklabels(['0','0.5','1'])

ax4.set_title('Mutualistic network (ER)', fontsize=12, fontweight='normal', fontfamily='DejaVu Sans', fontstyle= 'italic', y=1.02)
ax4.text(0.83, 0.75,'(d)',ha='left', transform=ax4.transAxes,fontdict={'family':'Arial','size':18,'weight':'normal'})

################################################################################################

textstr = '\n'.join([
    r'$\text{A}_{\text{DEWS}} = %.2f$' % auc_dynamics_BA,
    r'$\text{A}_{\text{AC}} = %.2f$' % auc_ac_BA,
    r'$\text{A}_{\text{DEV}} = %.2f$' % auc_dev_BA
])

ax5.plot(fpr_dynamics_BA, tpr_dynamics_BA, c='crimson',linewidth=2,zorder=3)
ax5.plot(fpr_ac_BA, tpr_ac_BA, c='darkslateblue',linewidth=2,zorder=2)
ax5.plot(fpr_dev_BA, tpr_dev_BA, c='darkorange',linewidth=2,zorder=1)
ax5.plot([0, 1], [0, 1], linestyle='--', linewidth=1, color='silver', zorder=0)

ax5.text(0.45, 0.35, textstr, transform=ax5.transAxes, fontsize=12,
         verticalalignment='top')

ax5.set_xlabel('False positive rate',font_x)

ax5.set_xticks([0,0.5,1])
ax5.set_xticklabels(['0','0.5','1'])
ax5.set_yticks([0,0.5,1])
ax5.set_yticklabels(['0','0.5','1'])

ax5.set_title('Mutualistic network (BA)', fontsize=12, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.02)
ax5.text(0.83, 0.75,'(e)',ha='left', transform=ax5.transAxes,fontdict={'family':'Arial','size':18,'weight':'normal'})

################################################################################################

dynamics_rate_ER = fold_ER_dynamics_rate['steady_rate'].to_numpy()
dynamics_rate_BA = fold_BA_dynamics_rate['steady_rate'].to_numpy()
dynamics_rate_sup_hopf = sup_hopf_dynamics_rate['steady_rate'].to_numpy()
dynamics_rate_trans = trans_dynamics_rate['steady_rate'].to_numpy()
dynamics_rate_sub_hopf = sub_hopf_dynamics_rate['steady_rate'].to_numpy()

dev_rate_ER = fold_ER_dev_rate['steady_rate'].to_numpy()
dev_rate_BA = fold_BA_dev_rate['steady_rate'].to_numpy()
dev_rate_sup_hopf = sup_hopf_dev_rate['steady_rate'].to_numpy()
dev_rate_trans = trans_dev_rate['steady_rate'].to_numpy()
dev_rate_sub_hopf = sub_hopf_dev_rate['steady_rate'].to_numpy()

dynamics_rate = np.concatenate((dynamics_rate_ER, dynamics_rate_BA, dynamics_rate_trans, dynamics_rate_sup_hopf, dynamics_rate_sub_hopf), axis=0)
dev_rate = np.concatenate((dev_rate_ER, dev_rate_BA, dev_rate_trans, dev_rate_sup_hopf, dev_rate_sub_hopf), axis=0)

fpr_dynamics_rate, tpr_dynamics_rate, thresholds_dynamics_rate = roc_curve(label_steady, dynamics_rate)
auc_dynamics_rate = auc(fpr_dynamics_rate, tpr_dynamics_rate)

fpr_dev_rate, tpr_dev_rate, thresholds_dev_rate = roc_curve(label_steady, dev_rate)
auc_dev_rate = auc(fpr_dev_rate, tpr_dev_rate)

textstr = '\n'.join([
    r'$\text{A}_{\text{DEWS}} = %.2f$' % auc_dynamics_rate,
    r'$\text{A}_{\text{DEV}} = %.2f$' % auc_dev_rate
])

ax6.plot(fpr_dynamics_rate, tpr_dynamics_rate, c='crimson',linewidth=2,zorder=3)
ax6.plot(fpr_dev_rate, tpr_dev_rate, c='darkorange',linewidth=2,zorder=1)
ax6.plot([0, 1], [0, 1], linestyle='--', linewidth=1, color='silver', zorder=0)

ax6.text(0.45, 0.35, textstr, transform=ax6.transAxes, fontsize=12,
         verticalalignment='top')

ax6.set_xlabel('False positive rate',font_x)

ax6.set_xticks([0,0.5,1])
ax6.set_xticklabels(['0','0.5','1'])
ax6.set_yticks([0,0.5,1])
ax6.set_yticklabels(['0','0.5','1'])

ax6.set_title('Classification', fontsize=12, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.02)
ax6.text(0.83, 0.75,'(f)',ha='left', transform=ax6.transAxes,fontdict={'family':'Arial','size':18,'weight':'normal'})

################################################################################################

legend_dynamics = mlines.Line2D([], [], color='crimson',linestyle='-',linewidth=3)
legend_ac = mlines.Line2D([], [], color='darkslateblue',linestyle='-',linewidth=3)
legend_dev = mlines.Line2D([], [], color='darkorange',linestyle='-',linewidth=3)

fig.legend(handles = [legend_dynamics, legend_ac, legend_dev],
            labels = ['Dynamics-based EWS','Lag-1 AC','Dynamical eigenvalue'],
              loc='upper center', bbox_to_anchor=(0.5,1.01), ncol=3, frameon=False, 
              prop=font_manager.FontProperties(family='Arial', size=15))

plt.subplots_adjust(top=0.8, bottom=0.1, left=0.045, right=0.995, wspace=0.25)
plt.savefig('../figures/FIG2.pdf',format='pdf')