import pandas
import matplotlib.pyplot as plt

font_x = {'family':'Arial','weight':'normal','size': 60}
font_y = {'family':'Arial','weight':'normal','size': 60}
font_in_x = {'family':'Arial','weight':'normal','size': 28}
font_in_y = {'family':'Arial','weight':'normal','size': 28}

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['xtick.labelsize'] = 50
plt.rcParams['ytick.labelsize'] = 50

fig, axs = plt.subplots(2, 5, figsize=(48,14))

ax1, ax2, ax3, ax4, ax5 = axs[0,0], axs[0,1], axs[0,2], axs[0,3], axs[0,4]
ax6, ax7, ax8, ax9, ax10 = axs[1,0], axs[1,1], axs[1,2], axs[1,3], axs[1,4]

df_fold_BA_results = pandas.read_csv('../results/fold_BA/fold_BA.csv')
df_fold_ER_results = pandas.read_csv('../results/fold_ER/fold_ER.csv')
df_sup_hopf_results = pandas.read_csv('../results/sup_hopf/sup_hopf.csv')
df_trans_results = pandas.read_csv('../results/trans/trans.csv')
df_sub_hopf_results = pandas.read_csv('../results/sub_hopf/sub_hopf.csv')

t_fold_ER = df_fold_ER_results['Time'] + 1
t_fold_BA = df_fold_BA_results['Time'] + 1
t_sup_hopf = df_sup_hopf_results['Time'] + 1
t_trans = df_trans_results['Time'] + 1
t_sub_hopf = df_sub_hopf_results['Time'] + 1

fold_ER_real = df_fold_ER_results['real_eval']
fold_ER_esti_real = df_fold_ER_results['esti_real_eval1']
fold_ER_imag = df_fold_ER_results['imag_eval']
fold_ER_esti_imag1 = df_fold_ER_results['esti_imag_eval1']

fold_BA_real = df_fold_BA_results['real_eval']
fold_BA_esti_real = df_fold_BA_results['esti_real_eval1']
fold_BA_imag = df_fold_BA_results['imag_eval']
fold_BA_esti_imag1 = df_fold_BA_results['esti_imag_eval1']

trans_real = df_trans_results['real_eval']
trans_esti_real = df_trans_results['esti_real_eval1']
trans_imag = df_trans_results['imag_eval']
trans_esti_imag1 = df_trans_results['esti_imag_eval1']

sup_hopf_real = df_sup_hopf_results['real_eval']
sup_hopf_esti_real = df_sup_hopf_results['esti_real_eval1']
sup_hopf_imag = df_sup_hopf_results['imag_eval']
sup_hopf_esti_imag1 = df_sup_hopf_results['esti_imag_eval1']
sup_hopf_esti_imag2 = df_sup_hopf_results['esti_imag_eval2']

sub_hopf_real = df_sub_hopf_results['real_eval']
sub_hopf_esti_real = df_sub_hopf_results['esti_real_eval1']
sub_hopf_imag = df_sub_hopf_results['imag_eval']
sub_hopf_esti_imag1 = df_sub_hopf_results['esti_imag_eval1']
sub_hopf_esti_imag2 = df_sub_hopf_results['esti_imag_eval2']

train_length = 500
length = 1000

ax1.plot(t_fold_ER,fold_ER_real,c='darkviolet',linewidth=8,alpha=0.8)
ax1.scatter(t_fold_ER[fold_ER_esti_real < 0],fold_ER_esti_real[fold_ER_esti_real < 0],s=80,marker='o',facecolors='none',edgecolors='crimson',zorder=5)
ax1.set_xlim(-58,1160)
ax1.set_ylim(-0.6,0.05)
ax1.set_yticks([-0.5,0])
ax1.set_yticklabels(['-0.5','0'])
ax1.tick_params(axis='x', bottom=False, labelbottom=False)
ax1.tick_params(direction='in', length=10)
ax1.text(0.06,0.88,'(f)',ha='left', transform=ax1.transAxes,fontdict={'family':'Arial','size':60,'weight':'normal'})
ax1.axvline(x=500, color='black', linestyle='--', linewidth=2, zorder=0)

ax1.set_ylabel('Re',font_y,labelpad=15)

################################################################################################

ax2.plot(t_fold_BA,fold_BA_real,c='darkviolet',linewidth=8,alpha=0.8)
ax2.scatter(t_fold_BA[fold_BA_esti_real < 0],fold_BA_esti_real[fold_BA_esti_real < 0],s=80,marker='o',facecolors='none',edgecolors='crimson',zorder=5)
ax2.set_xlim(-58,1160)
ax2.set_ylim(-0.8,0.05)
ax2.set_yticks([-0.7,0])
ax2.set_yticklabels(['-0.7','0'])
ax2.tick_params(axis='x', bottom=False, labelbottom=False)
ax2.tick_params(direction='in', length=10)
ax2.text(0.06,0.88,'(g)',ha='left', transform=ax2.transAxes,fontdict={'family':'Arial','size':60,'weight':'normal'})
ax2.axvline(x=500, color='black', linestyle='--', linewidth=2, zorder=0)

################################################################################################

ax3.plot(t_trans,trans_real,c='darkviolet',linewidth=8,alpha=0.8)
ax3.scatter(t_trans,trans_esti_real,s=80,marker='o',facecolors='none',edgecolors='crimson',zorder=5)
ax3.set_xlim(-60,1200)
ax3.set_yticks([-0.5,0])
ax3.set_yticklabels(['-0.5','0'])
ax3.tick_params(axis='x', bottom=False, labelbottom=False)
ax3.tick_params(direction='in', length=10)
ax3.text(0.06,0.88,'(h)',ha='left', transform=ax3.transAxes,fontdict={'family':'Arial','size':60,'weight':'normal'})
ax3.axvline(x=500, color='black', linestyle='--', linewidth=2, zorder=0)

################################################################################################

ax4.plot(t_sup_hopf,sup_hopf_real,c='darkviolet',linewidth=8,alpha=0.8)
ax4.scatter(t_sup_hopf[df_sup_hopf_results['esti_real_eval1'].between(-0.065, 0.002)],sup_hopf_esti_real[df_sup_hopf_results['esti_real_eval1'].between(-0.065, 0.002)],s=80,marker='o',facecolors='none',edgecolors='crimson',zorder=5)
ax4.set_xlim(-61,1220)
ax4.set_ylim(-0.066,0.006)
ax4.set_yticks([-0.06,0])
ax4.set_yticklabels(['-0.06','0'], fontsize=45)
ax4.tick_params(axis='x', bottom=False, labelbottom=False)
ax4.tick_params(direction='in', length=10)
ax4.text(0.06,0.88,'(i)',ha='left', transform=ax4.transAxes,fontdict={'family':'Arial','size':60,'weight':'normal'})
ax4.axvline(x=500, color='black', linestyle='--', linewidth=2, zorder=0)

################################################################################################

ax5.plot(t_sub_hopf,sub_hopf_real,c='darkviolet',linewidth=8,alpha=0.8)
ax5.scatter(t_sub_hopf[df_sub_hopf_results['esti_real_eval1'].between(-0.35, 0.02)],sub_hopf_esti_real[df_sub_hopf_results['esti_real_eval1'].between(-0.35, 0.02)],s=80,marker='o',facecolors='none',edgecolors='crimson',zorder=5)
ax5.set_xlim(-62,1265)
ax5.set_ylim(-0.35,0.03)
ax5.set_yticks([-0.3,0])
ax5.set_yticklabels(['-0.3','0'])
ax5.tick_params(axis='x', bottom=False, labelbottom=False)
ax5.tick_params(direction='in', length=10)
ax5.text(0.06,0.88,'(j)',ha='left', transform=ax5.transAxes,fontdict={'family':'Arial','size':60,'weight':'normal'})
ax5.axvline(x=500, color='black', linestyle='--', linewidth=2, zorder=0)

################################################################################################

ax6.plot(t_fold_ER,fold_ER_esti_imag1,c='darkgreen',linewidth=8,alpha=0.8)
ax6.scatter(t_fold_ER[::5],fold_ER_esti_imag1[::5], s=80, marker='s',facecolors='none',edgecolors='crimson',zorder=5)
ax6.set_xlim(-58,1160)
ax6.set_xticks([0,500,1000])
ax6.set_xticklabels(['0','500','1000'])
ax6.set_yticks([0])
ax6.set_yticklabels(['0'])
ax6.set_ylim(-0.1,0.1)
ax6.tick_params(direction='in', length=10)
ax6.tick_params(axis='x', pad=10)
ax6.text(0.06,0.88,'(k)',ha='left', transform=ax6.transAxes,fontdict={'family':'Arial','size':60,'weight':'normal'})
ax6.axvline(x=500, color='black', linestyle='--', linewidth=2, zorder=0)

ax6.set_xlabel('Timepoint',font_x)
ax6.set_ylabel('Im',font_y,labelpad=75)

################################################################################################

ax7.plot(t_fold_BA,fold_BA_esti_imag1,c='darkgreen',linewidth=8,alpha=0.8)
ax7.scatter(t_fold_BA[::5],fold_BA_esti_imag1[::5], s=80, marker='s',facecolors='none',edgecolors='crimson',zorder=5)
ax7.set_xlim(-58,1160)
ax7.set_xticks([0,500,1000])
ax7.set_xticklabels(['0','500','1000'])
ax7.set_yticks([0])
ax7.set_yticklabels(['0'])
ax7.set_ylim(-0.1,0.1)
ax7.tick_params(direction='in', length=10)
ax7.tick_params(axis='x', pad=10)
ax7.text(0.06,0.88,'(l)',ha='left', transform=ax7.transAxes,fontdict={'family':'Arial','size':60,'weight':'normal'})
ax7.axvline(x=500, color='black', linestyle='--', linewidth=2, zorder=0)

ax7.set_xlabel('Timepoint',font_x)

################################################################################################

ax8.plot(t_trans,trans_imag,c='darkgreen',linewidth=8,alpha=0.8,zorder=1)
ax8.scatter(t_trans[::5],trans_esti_imag1[::5], s=80, marker='s',facecolors='none',edgecolors='crimson',zorder=5)
ax8.set_xlim(-60,1200)
ax8.set_xticks([0,500,1000])
ax8.set_xticklabels(['0','500','1000'])
ax8.set_yticks([0])
ax8.set_yticklabels(['0'])
ax8.set_ylim(-0.1,0.1)
ax8.tick_params(direction='in', length=10)
ax8.tick_params(axis='x', pad=10)
ax8.text(0.06,0.88,'(m)',ha='left', transform=ax8.transAxes,fontdict={'family':'Arial','size':60,'weight':'normal'})
ax8.axvline(x=500, color='black', linestyle='--', linewidth=2, zorder=0)

ax8.set_xlabel('Timepoint',font_x)

################################################################################################

ax9.plot(t_sup_hopf,sup_hopf_imag,c='darkgreen',linewidth=8,alpha=0.8,zorder=1)
ax9.plot(t_sup_hopf,-sup_hopf_imag,c='darkgreen',linewidth=8,alpha=0.8,zorder=1)
ax9.scatter(t_sup_hopf[::1],sup_hopf_esti_imag1[::1], s=80, marker='s',facecolors='none',edgecolors='crimson',zorder=2)
ax9.scatter(t_sup_hopf[::1],sup_hopf_esti_imag2[::1], s=80, marker='s',facecolors='none',edgecolors='crimson',zorder=2)
ax9.set_xlim(-61,1220)
ax9.set_xticks([0,500,1000])
ax9.set_xticklabels(['0','500','1000'])
ax9.set_yticks([-0.2,0,0.2])
ax9.set_yticklabels(['-0.2','0','0.2'])
ax9.set_ylim(-0.25,0.25)
ax9.tick_params(direction='in', length=10)
ax9.tick_params(axis='x', pad=10)
ax9.text(0.06,0.88,'(n)',ha='left', transform=ax9.transAxes,fontdict={'family':'Arial','size':60,'weight':'normal'})
ax9.axvline(x=500, color='black', linestyle='--', linewidth=2, zorder=0)

ax9.set_xlabel('Timepoint',font_x)

################################################################################################

ax10.plot(t_sub_hopf,sub_hopf_imag,c='darkgreen',linewidth=8,alpha=0.8,zorder=1)
ax10.plot(t_sub_hopf,-sub_hopf_imag,c='darkgreen',linewidth=8,alpha=0.8,zorder=1)
ax10.scatter(t_sub_hopf[::1],sub_hopf_esti_imag1[::1], s=80, marker='s',facecolors='none',edgecolors='crimson',zorder=2)
ax10.scatter(t_sub_hopf[::1],sub_hopf_esti_imag2[::1], s=80, marker='s',facecolors='none',edgecolors='crimson',zorder=2)
ax10.set_xlim(-62,1265)
ax10.set_xticks([0,500,1000])
ax10.set_xticklabels(['0','500','1000'])
ax10.set_yticks([-0.8,0,0.8])
ax10.set_yticklabels(['-0.8','0','0.8'])
ax10.set_ylim(-1,1)
ax10.tick_params(direction='in', length=10)
ax10.text(0.06,0.88,'(o)',ha='left', transform=ax10.transAxes,fontdict={'family':'Arial','size':60,'weight':'normal'})
ax10.axvline(x=500, color='black', linestyle='--', linewidth=2, zorder=0)

ax10.set_xlabel('Timepoint',font_x)

plt.subplots_adjust(top=0.99, bottom=0.12, left=0.05, right=0.99, hspace=0.1, wspace=0.2)
plt.savefig('../figures/FIG1.2.pdf',format='pdf')
