import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

font_x = {'family':'Arial','weight':'normal','size': 12}
font_y = {'family':'Arial','weight':'normal','size': 12}

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

################################################################################################

n = 400

label = np.concatenate((np.ones(n), np.zeros(n)), axis=0)

################################################################################################

fold_ER_singlev = pd.read_csv('../results/ROC/fold_ER/singlev_tau.csv')
fold_ER_null_singlev = pd.read_csv('../results/ROC/fold_ER/singlev_tau_null.csv')

fold_BA_singlev = pd.read_csv('../results/ROC/fold_BA/singlev_tau.csv')
fold_BA_null_singlev = pd.read_csv('../results/ROC/fold_BA/singlev_tau_null.csv')

trans_singlev = pd.read_csv('../results/ROC/trans/singlev_tau.csv')
trans_null_singlev = pd.read_csv('../results/ROC/trans/singlev_tau_null.csv')

sup_hopf_singlev = pd.read_csv('../results/ROC/sup_hopf/singlev_tau.csv')
sup_hopf_null_singlev = pd.read_csv('../results/ROC/sup_hopf/singlev_tau_null.csv')

sub_hopf_singlev = pd.read_csv('../results/ROC/sub_hopf/singlev_tau.csv')
sub_hopf_null_singlev = pd.read_csv('../results/ROC/sub_hopf/singlev_tau_null.csv')

################################################################################################

fig, axs = plt.subplots(1, 5, figsize=(12,3))

for ax in axs:
    ax.set_aspect('equal') 

ax1, ax2, ax3, ax4, ax5 = axs[0], axs[1], axs[2], axs[3], axs[4]

################################################################################################

singlev_tau_ER = fold_ER_singlev['tau'].to_numpy()
null_singlev_tau_ER = fold_ER_null_singlev['tau'].to_numpy()

singlev_tau_BA = fold_BA_singlev['tau'].to_numpy()
null_singlev_tau_BA = fold_BA_null_singlev['tau'].to_numpy()

singlev_ER = np.concatenate((singlev_tau_ER, null_singlev_tau_ER), axis=0)
singlev_BA = np.concatenate((singlev_tau_BA, null_singlev_tau_BA), axis=0)

fpr_singlev_ER, tpr_singlev_ER, thresholds_singlev_ER = roc_curve(label, singlev_ER)
auc_singlev_ER = auc(fpr_singlev_ER, tpr_singlev_ER)

fpr_singlev_BA, tpr_singlev_BA, thresholds_singlev_BA = roc_curve(label, singlev_BA)
auc_singlev_BA = auc(fpr_singlev_BA, tpr_singlev_BA)

################################################################################################

singlev_tau = trans_singlev['tau'].to_numpy()
null_singlev_tau = trans_null_singlev['tau'].to_numpy()

singlev = np.concatenate((singlev_tau, null_singlev_tau), axis=0)

fpr_singlev, tpr_singlev, thresholds_singlev = roc_curve(label, singlev)
auc_singlev = auc(fpr_singlev, tpr_singlev)

textstr = '\n'.join([
    r'$\text{A} = %.2f$' % auc_singlev
])

ax1.plot(fpr_singlev, tpr_singlev, c='crimson',linewidth=2,zorder=3)
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
ax1.text(0.8, 0.7,'(a)',ha='left', transform=ax1.transAxes,fontdict={'family':'Arial','size':17,'weight':'normal'})

################################################################################################

singlev_tau = sup_hopf_singlev['tau'].to_numpy()
null_singlev_tau = sup_hopf_null_singlev['tau'].to_numpy()

singlev = np.concatenate((singlev_tau, null_singlev_tau), axis=0)

fpr_singlev, tpr_singlev, thresholds_singlev = roc_curve(label, singlev)
auc_singlev = auc(fpr_singlev, tpr_singlev)

textstr = '\n'.join([
    r'$\text{A} = %.2f$' % auc_singlev
])

ax2.plot(fpr_singlev, tpr_singlev, c='crimson',linewidth=2,zorder=3)
ax2.plot([0, 1], [0, 1], linestyle='--', linewidth=1, color='silver', zorder=0)

ax2.text(0.45, 0.35, textstr, transform=ax2.transAxes, fontsize=12,
         verticalalignment='top')

ax2.set_xlabel('False positive rate',font_x)

ax2.set_xticks([0,0.5,1])
ax2.set_xticklabels(['0','0.5','1'])
ax2.set_yticks([0,0.5,1])
ax2.set_yticklabels(['0','0.5','1'])

ax2.set_title('Predator-prey (Hopf)', fontsize=12, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.02)
ax2.text(0.8, 0.7,'(b)',ha='left', transform=ax2.transAxes,fontdict={'family':'Arial','size':17,'weight':'normal'})

################################################################################################

singlev_tau = sub_hopf_singlev['tau'].to_numpy()
null_singlev_tau = sub_hopf_null_singlev['tau'].to_numpy()

singlev = np.concatenate((singlev_tau, null_singlev_tau), axis=0)

fpr_singlev, tpr_singlev, thresholds_singlev = roc_curve(label, singlev)
auc_singlev = auc(fpr_singlev, tpr_singlev)

textstr = '\n'.join([
    r'$\text{A} = %.2f$' % auc_singlev
])

ax3.plot(fpr_singlev, tpr_singlev, c='crimson',linewidth=2,zorder=3)
ax3.plot([0, 1], [0, 1], linestyle='--', linewidth=1, color='silver', zorder=0)

ax3.text(0.45, 0.35, textstr, transform=ax3.transAxes, fontsize=12,
         verticalalignment='top',zorder=3)

ax3.set_xlabel('False positive rate',font_x)

ax3.set_xticks([0,0.5,1])
ax3.set_xticklabels(['0','0.5','1'])
ax3.set_yticks([0,0.5,1])
ax3.set_yticklabels(['0','0.5','1'])

ax3.set_title('Epidemic', fontsize=12, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.02)
ax3.text(0.8, 0.7,'(c)',ha='left', transform=ax3.transAxes,fontdict={'family':'Arial','size':17,'weight':'normal'})

################################################################################################

textstr = '\n'.join([
    r'$\text{A} = %.2f$' % auc_singlev_ER
])

ax4.plot(fpr_singlev_ER, tpr_singlev_ER, c='crimson',linewidth=2,zorder=3)
ax4.plot([0, 1], [0, 1], linestyle='--', linewidth=1, color='silver', zorder=0)

ax4.text(0.45, 0.35, textstr, transform=ax4.transAxes, fontsize=12,
         verticalalignment='top')

ax4.set_xlabel('False positive rate',font_x)

ax4.set_xticks([0,0.5,1])
ax4.set_xticklabels(['0','0.5','1'])
ax4.set_yticks([0,0.5,1])
ax4.set_yticklabels(['0','0.5','1'])

ax4.set_title('Mutualistic network (ER)', fontsize=12, fontweight='normal', fontfamily='DejaVu Sans', fontstyle= 'italic', y=1.02)
ax4.text(0.8, 0.7,'(d)',ha='left', transform=ax4.transAxes,fontdict={'family':'Arial','size':17,'weight':'normal'})

################################################################################################

textstr = '\n'.join([
    r'$\text{A} = %.2f$' % auc_singlev_BA
])

ax5.plot(fpr_singlev_BA, tpr_singlev_BA, c='crimson',linewidth=2,zorder=3)
ax5.plot([0, 1], [0, 1], linestyle='--', linewidth=1, color='silver', zorder=0)

ax5.text(0.45, 0.35, textstr, transform=ax5.transAxes, fontsize=12,
         verticalalignment='top')

ax5.set_xlabel('False positive rate',font_x)

ax5.set_xticks([0,0.5,1])
ax5.set_xticklabels(['0','0.5','1'])
ax5.set_yticks([0,0.5,1])
ax5.set_yticklabels(['0','0.5','1'])

ax5.set_title('Mutualistic network (BA)', fontsize=12, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.02)
ax5.text(0.8, 0.7,'(e)',ha='left', transform=ax5.transAxes,fontdict={'family':'Arial','size':17,'weight':'normal'})

################################################################################################

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.05, right=0.99, wspace=0.3)
plt.savefig('../figures/SFIG11.pdf',format='pdf')