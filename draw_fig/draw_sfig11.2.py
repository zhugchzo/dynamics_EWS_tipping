import pandas
import matplotlib.pyplot as plt

font_x = {'family':'Arial','weight':'normal','size': 60}
font_y = {'family':'Arial','weight':'normal','size': 60}
font_in_x = {'family':'Arial','weight':'normal','size': 28}
font_in_y = {'family':'Arial','weight':'normal','size': 28}

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['xtick.labelsize'] = 45
plt.rcParams['ytick.labelsize'] = 45

fig, axs = plt.subplots(2, 5, figsize=(48,14))

ax1, ax2, ax3, ax4, ax5 = axs[0,0], axs[0,1], axs[0,2], axs[0,3], axs[0,4]
ax6, ax7, ax8, ax9, ax10 = axs[1,0], axs[1,1], axs[1,2], axs[1,3], axs[1,4]

df_fold_BA_results = pandas.read_csv('../results/mean_lambda/singlev/fold_BA_mean_lambda.csv')
df_fold_ER_results = pandas.read_csv('../results/mean_lambda/singlev/fold_ER_mean_lambda.csv')
df_sup_hopf_results = pandas.read_csv('../results/mean_lambda/singlev/sup_hopf_mean_lambda.csv')
df_trans_results = pandas.read_csv('../results/mean_lambda/singlev/trans_mean_lambda.csv')
df_sub_hopf_results = pandas.read_csv('../results/mean_lambda/singlev/sub_hopf_mean_lambda.csv')

t_fold_ER = df_fold_ER_results['Time'] + 501
t_fold_BA = df_fold_BA_results['Time'] + 501
t_sup_hopf = df_sup_hopf_results['Time'] + 501
t_trans = df_trans_results['Time'] + 501
t_sub_hopf = df_sub_hopf_results['Time'] + 501

fold_ER_esti_re_v1 = df_fold_ER_results['esti_re_v1_mean']
fold_ER_esti_re_v1_lower = df_fold_ER_results['esti_re_v1_ci_lower']
fold_ER_esti_re_v1_upper = df_fold_ER_results['esti_re_v1_ci_upper']
fold_ER_esti_re_v2 = df_fold_ER_results['esti_re_v2_mean']
fold_ER_esti_re_v2_lower = df_fold_ER_results['esti_re_v2_ci_lower']
fold_ER_esti_re_v2_upper = df_fold_ER_results['esti_re_v2_ci_upper']

fold_BA_esti_re_v1 = df_fold_BA_results['esti_re_v1_mean']
fold_BA_esti_re_v1_lower = df_fold_BA_results['esti_re_v1_ci_lower']
fold_BA_esti_re_v1_upper = df_fold_BA_results['esti_re_v1_ci_upper']
fold_BA_esti_re_v2 = df_fold_BA_results['esti_re_v2_mean']
fold_BA_esti_re_v2_lower = df_fold_BA_results['esti_re_v2_ci_lower']
fold_BA_esti_re_v2_upper = df_fold_BA_results['esti_re_v2_ci_upper']

trans_esti_re_v1 = df_trans_results['esti_re_v1_mean']
trans_esti_re_v1_lower = df_trans_results['esti_re_v1_ci_lower']
trans_esti_re_v1_upper = df_trans_results['esti_re_v1_ci_upper']
trans_esti_re_v2 = df_trans_results['esti_re_v2_mean']
trans_esti_re_v2_lower = df_trans_results['esti_re_v2_ci_lower']
trans_esti_re_v2_upper = df_trans_results['esti_re_v2_ci_upper']

sup_hopf_esti_re_v1 = df_sup_hopf_results['esti_re_v1_mean']
sup_hopf_esti_re_v1_lower = df_sup_hopf_results['esti_re_v1_ci_lower']
sup_hopf_esti_re_v1_upper = df_sup_hopf_results['esti_re_v1_ci_upper']
sup_hopf_esti_re_v2 = df_sup_hopf_results['esti_re_v2_mean']
sup_hopf_esti_re_v2_lower = df_sup_hopf_results['esti_re_v2_ci_lower']
sup_hopf_esti_re_v2_upper = df_sup_hopf_results['esti_re_v2_ci_upper']

sub_hopf_esti_re_v1 = df_sub_hopf_results['esti_re_v1_mean']
sub_hopf_esti_re_v1_lower = df_sub_hopf_results['esti_re_v1_ci_lower']
sub_hopf_esti_re_v1_upper = df_sub_hopf_results['esti_re_v1_ci_upper']
sub_hopf_esti_re_v2 = df_sub_hopf_results['esti_re_v2_mean']
sub_hopf_esti_re_v2_lower = df_sub_hopf_results['esti_re_v2_ci_lower']
sub_hopf_esti_re_v2_upper = df_sub_hopf_results['esti_re_v2_ci_upper']

train_length = 500
length = 1000

################################################################################################

ax1.plot(t_trans,trans_esti_re_v1,c='crimson',linewidth=6,alpha=0.9,zorder=2)
ax1.fill_between(t_trans,trans_esti_re_v1_lower,trans_esti_re_v1_upper,color='crimson',alpha=0.3,linewidth=0,zorder=2)
ax1.set_xlim(-50,1220)
ax1.set_ylim(-0.84,0.084)
ax1.set_yticks([-0.7,0])
ax1.set_yticklabels(['-0.7','0'])
ax1.tick_params(axis='x', bottom=False, labelbottom=False)
ax1.tick_params(direction='in', length=10)
ax1.minorticks_on()
ax1.tick_params(axis='x', which='minor', bottom=False)
ax1.tick_params(axis='y', which='minor', direction='in', length=5)
ax1.text(0.06,0.88,'(f)',ha='left', transform=ax1.transAxes,fontdict={'family':'Arial','size':60,'weight':'normal'})
ax1.axvline(x=500, color='black', linestyle='--', linewidth=2,alpha=0.5, zorder=0)
ax1.hlines(y=0, xmin=500, xmax=1150, color='grey', linestyle='--', linewidth=4, alpha=0.5, zorder=0)

ax1.set_ylabel('Re',font_y)
ax1.yaxis.set_label_coords(-0.2, 0.5)

################################################################################################

ax2.plot(t_sup_hopf,sup_hopf_esti_re_v1,c='crimson',linewidth=6,alpha=0.9,zorder=2)
ax2.fill_between(t_sup_hopf,sup_hopf_esti_re_v1_lower,sup_hopf_esti_re_v1_upper,color='crimson',alpha=0.3,linewidth=0,zorder=2)
ax2.set_xlim(-50,1220)
ax2.set_ylim(-0.036,0.0036)
ax2.set_yticks([-0.03,0])
ax2.set_yticklabels(['-0.03','0'])
ax2.tick_params(axis='x', bottom=False, labelbottom=False)
ax2.tick_params(direction='in', length=10)
ax2.minorticks_on()
ax2.tick_params(axis='x', which='minor', bottom=False)
ax2.tick_params(axis='y', which='minor', direction='in', length=5)
ax2.text(0.06,0.88,'(g)',ha='left', transform=ax2.transAxes,fontdict={'family':'Arial','size':60,'weight':'normal'})
ax2.axvline(x=500, color='black', linestyle='--', linewidth=2,alpha=0.5, zorder=0)
ax2.hlines(y=0, xmin=500, xmax=1150, color='grey', linestyle='--', linewidth=4, alpha=0.5, zorder=0)

################################################################################################

ax3.plot(t_sub_hopf,sub_hopf_esti_re_v1,c='crimson',linewidth=6,alpha=0.9,zorder=2)
ax3.fill_between(t_sub_hopf,sub_hopf_esti_re_v1_lower,sub_hopf_esti_re_v1_upper,color='crimson',alpha=0.3,linewidth=0,zorder=2)
ax3.set_xlim(-50,1220)
ax3.set_ylim(-0.36,0.036)
ax3.set_yticks([-0.3,0])
ax3.set_yticklabels(['-0.3','0'])
ax3.tick_params(axis='x', bottom=False, labelbottom=False)
ax3.tick_params(direction='in', length=10)
ax3.minorticks_on()
ax3.tick_params(axis='x', which='minor', bottom=False)
ax3.tick_params(axis='y', which='minor', direction='in', length=5)
ax3.text(0.06,0.88,'(h)',ha='left', transform=ax3.transAxes,fontdict={'family':'Arial','size':60,'weight':'normal'})
ax3.axvline(x=500, color='black', linestyle='--', linewidth=2,alpha=0.5, zorder=0)
ax3.hlines(y=0, xmin=500, xmax=1150, color='grey', linestyle='--', linewidth=4, alpha=0.5, zorder=0)

################################################################################################

ax4.plot(t_fold_ER,fold_ER_esti_re_v1,c='crimson',linewidth=6,alpha=0.9,zorder=2)
ax4.fill_between(t_fold_ER,fold_ER_esti_re_v1_lower,fold_ER_esti_re_v1_upper,color='crimson',alpha=0.3,linewidth=0,zorder=2)
ax4.set_xlim(-50,1220)
ax4.set_ylim(-0.6,0.06)
ax4.set_yticks([-0.5,0])
ax4.set_yticklabels(['-0.5','0'])
ax4.tick_params(axis='x', bottom=False, labelbottom=False)
ax4.tick_params(direction='in', length=10)
ax4.minorticks_on()
ax4.tick_params(axis='x', which='minor', bottom=False)
ax4.tick_params(axis='y', which='minor', direction='in', length=5)
ax4.text(0.06,0.88,'(i)',ha='left', transform=ax4.transAxes,fontdict={'family':'Arial','size':60,'weight':'normal'})
ax4.axvline(x=500, color='black', linestyle='--', linewidth=2,alpha=0.5, zorder=0)
ax4.hlines(y=0, xmin=500, xmax=1150, color='grey', linestyle='--', linewidth=4, alpha=0.5, zorder=0)

################################################################################################

ax5.plot(t_fold_BA,fold_BA_esti_re_v1,c='crimson',linewidth=6,alpha=0.9,zorder=2)
ax5.fill_between(t_fold_BA,fold_BA_esti_re_v1_lower,fold_BA_esti_re_v1_upper,color='crimson',alpha=0.3,linewidth=0,zorder=2)
ax5.set_xlim(-50,1220)
ax5.set_ylim(-0.6,0.06)
ax5.set_yticks([-0.5,0])
ax5.set_yticklabels(['-0.5','0'])
ax5.tick_params(axis='x', bottom=False, labelbottom=False)
ax5.tick_params(direction='in', length=10)
ax5.minorticks_on()
ax5.tick_params(axis='x', which='minor', bottom=False)
ax5.tick_params(axis='y', which='minor', direction='in', length=5)
ax5.text(0.06,0.88,'(j)',ha='left', transform=ax5.transAxes,fontdict={'family':'Arial','size':60,'weight':'normal'})
ax5.axvline(x=500, color='black', linestyle='--', linewidth=2,alpha=0.5, zorder=0)
ax5.hlines(y=0, xmin=500, xmax=1150, color='grey', linestyle='--', linewidth=4, alpha=0.5, zorder=0)

################################################################################################

ax6.plot(t_trans,trans_esti_re_v2,c='darkviolet',linewidth=6,alpha=0.9,zorder=2)
ax6.fill_between(t_trans,trans_esti_re_v2_lower,trans_esti_re_v2_upper,color='darkviolet',alpha=0.3,linewidth=0,zorder=2)
ax6.set_xlim(-50,1220)
ax6.set_xticks([0,500,1000])
ax6.set_xticklabels(['0','500','1000'])
ax6.set_ylim(-0.6,0.06)
ax6.set_yticks([-0.5,0])
ax6.set_yticklabels(['-0.5','0'])
ax6.tick_params(direction='in', length=10)
ax6.tick_params(axis='x', pad=10)
ax6.minorticks_on()
ax6.tick_params(axis='x', which='minor', bottom=False)
ax6.tick_params(axis='y', which='minor', direction='in', length=5)
ax6.text(0.06,0.88,'(k)',ha='left', transform=ax6.transAxes,fontdict={'family':'Arial','size':60,'weight':'normal'})
ax6.axvline(x=500, color='black', linestyle='--', linewidth=2,alpha=0.5, zorder=0)
ax6.hlines(y=0, xmin=500, xmax=1150, color='grey', linestyle='--', linewidth=4, alpha=0.5, zorder=0)

ax6.set_xlabel('Time',font_x,labelpad=15)
ax6.set_ylabel('Re',font_y)
ax6.yaxis.set_label_coords(-0.2, 0.5)

################################################################################################

ax7.plot(t_sup_hopf,sup_hopf_esti_re_v2,c='darkviolet',linewidth=6,alpha=0.9,zorder=2)
ax7.fill_between(t_sup_hopf,sup_hopf_esti_re_v2_lower,sup_hopf_esti_re_v2_upper,color='darkviolet',alpha=0.3,linewidth=0,zorder=2)
ax7.set_xlim(-50,1220)
ax7.set_xticks([0,500,1000])
ax7.set_xticklabels(['0','500','1000'])
ax7.set_ylim(-0.084,0.0084)
ax7.set_yticks([-0.07,0])
ax7.set_yticklabels(['-0.07','0'])
ax7.tick_params(direction='in', length=10)
ax7.tick_params(axis='x', pad=10)
ax7.minorticks_on()
ax7.tick_params(axis='x', which='minor', bottom=False)
ax7.tick_params(axis='y', which='minor', direction='in', length=5)
ax7.text(0.06,0.88,'(l)',ha='left', transform=ax7.transAxes,fontdict={'family':'Arial','size':60,'weight':'normal'})
ax7.axvline(x=500, color='black', linestyle='--', linewidth=2,alpha=0.5, zorder=0)
ax7.hlines(y=0, xmin=500, xmax=1150, color='grey', linestyle='--', linewidth=4, alpha=0.5, zorder=0)

ax7.set_xlabel('Time',font_x,labelpad=15)

################################################################################################

ax8.plot(t_sub_hopf,sub_hopf_esti_re_v2,c='darkviolet',linewidth=6,alpha=0.9,zorder=2)
ax8.fill_between(t_sub_hopf,sub_hopf_esti_re_v2_lower,sub_hopf_esti_re_v2_upper,color='darkviolet',alpha=0.3,linewidth=0,zorder=2)
ax8.set_xlim(-50,1220)
ax8.set_xticks([0,500,1000])
ax8.set_xticklabels(['0','500','1000'])
ax8.set_ylim(-0.48,0.048)
ax8.set_yticks([-0.4,0])
ax8.set_yticklabels(['-0.4','0'])
ax8.tick_params(direction='in', length=10)
ax8.minorticks_on()
ax8.tick_params(axis='x', which='minor', bottom=False)
ax8.tick_params(axis='y', which='minor', direction='in', length=5)
ax8.text(0.06,0.88,'(m)',ha='left', transform=ax8.transAxes,fontdict={'family':'Arial','size':60,'weight':'normal'})
ax8.axvline(x=500, color='black', linestyle='--', linewidth=2,alpha=0.5, zorder=0)
ax8.hlines(y=0, xmin=500, xmax=1150, color='grey', linestyle='--', linewidth=4, alpha=0.5, zorder=0)

ax8.set_xlabel('Time',font_x,labelpad=15)
ax8.set_ylabel('')

################################################################################################

ax9.plot(t_fold_ER,fold_ER_esti_re_v2,c='darkviolet',linewidth=6,alpha=0.9,zorder=2)
ax9.fill_between(t_fold_ER,fold_ER_esti_re_v2_lower,fold_ER_esti_re_v2_upper,color='darkviolet',alpha=0.3,linewidth=0,zorder=2)
ax9.set_xlim(-50,1220)
ax9.set_xticks([0,500,1000])
ax9.set_xticklabels(['0','500','1000'])
ax9.set_ylim(-0.6,0.06)
ax9.set_yticks([-0.5,0])
ax9.set_yticklabels(['-0.5','0'])
ax9.tick_params(direction='in', length=10)
ax9.tick_params(axis='x', pad=10)
ax9.minorticks_on()
ax9.tick_params(axis='x', which='minor', bottom=False)
ax9.tick_params(axis='y', which='minor', direction='in', length=5)
ax9.text(0.06,0.88,'(n)',ha='left', transform=ax9.transAxes,fontdict={'family':'Arial','size':60,'weight':'normal'})
ax9.axvline(x=500, color='black', linestyle='--', linewidth=2,alpha=0.5, zorder=0)
ax9.hlines(y=0, xmin=500, xmax=1150, color='grey', linestyle='--', linewidth=4, alpha=0.5, zorder=0)

ax9.set_xlabel('Time',font_x,labelpad=15)

################################################################################################

ax10.plot(t_fold_BA,fold_BA_esti_re_v2,c='darkviolet',linewidth=6,alpha=0.9,zorder=2)
ax10.fill_between(t_fold_BA,fold_BA_esti_re_v2_lower,fold_BA_esti_re_v2_upper,color='darkviolet',alpha=0.3,linewidth=0,zorder=2)
ax10.set_xlim(-50,1220)
ax10.set_xticks([0,500,1000])
ax10.set_xticklabels(['0','500','1000'])
ax10.set_ylim(-0.6,0.06)
ax10.set_yticks([-0.5,0])
ax10.set_yticklabels(['-0.5','0'])
ax10.tick_params(direction='in', length=10)
ax10.tick_params(axis='x', pad=10)
ax10.minorticks_on()
ax10.tick_params(axis='x', which='minor', bottom=False)
ax10.tick_params(axis='y', which='minor', direction='in', length=5)
ax10.text(0.06,0.88,'(o)',ha='left', transform=ax10.transAxes,fontdict={'family':'Arial','size':60,'weight':'normal'})
ax10.axvline(x=500, color='black', linestyle='--', linewidth=2,alpha=0.5, zorder=0)
ax10.hlines(y=0, xmin=500, xmax=1150, color='grey', linestyle='--', linewidth=4, alpha=0.5, zorder=0)

ax10.set_xlabel('Time',font_x,labelpad=15)

plt.subplots_adjust(top=0.99, bottom=0.12, left=0.05, right=0.99, hspace=0.1, wspace=0.2)
plt.savefig('../figures/SFIG11.2.pdf',format='pdf')