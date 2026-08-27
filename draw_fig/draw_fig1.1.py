import pandas
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.lines as mlines
from matplotlib import font_manager
from matplotlib.patches import ConnectionPatch

font_x = {'family':'Arial','weight':'normal','size': 60}
font_y = {'family':'Arial','weight':'normal','size': 60}
font_in_x = {'family':'Arial','weight':'normal','size': 30}
font_in_y = {'family':'Arial','weight':'normal','size': 30}

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['xtick.labelsize'] = 45
plt.rcParams['ytick.labelsize'] = 45

fig, axs = plt.subplots(1, 5, figsize=(48,10))

ax1, ax2, ax3, ax4, ax5 = axs[0], axs[1], axs[2], axs[3], axs[4]

df_trans = pandas.read_csv('../data/trans/trans.csv')

df_sup_hopf = pandas.read_csv('../data/sup_hopf/sup_hopf.csv')
df_sup_hopf_stable = pandas.read_csv('../results/mean_lambda/sup_hopf_mean_stable.csv')

df_sub_hopf = pandas.read_csv('../data/sub_hopf/sub_hopf.csv')
df_sub_hopf_stable = pandas.read_csv('../results/mean_lambda/sub_hopf_mean_stable.csv')

df_fold_BA = pandas.read_csv('../data/fold_BA/fold_BA.csv')

df_fold_ER = pandas.read_csv('../data/fold_ER/fold_ER.csv')

truncate_length = 1185

t_fold_ER = df_fold_ER['Time'][:truncate_length] + 1
t_fold_BA = df_fold_BA['Time'][:truncate_length] + 1
t_trans = df_trans['Time'][:truncate_length] + 1
t_sup_hopf = df_sup_hopf['Time'][:truncate_length] + 1
t_sup_hopf_stable = df_sup_hopf_stable['Time'] + 951
t_sub_hopf = df_sub_hopf['Time'][:truncate_length] + 1
t_sub_hopf_stable = df_sub_hopf_stable['Time'] + 951

fold_ER_x = df_fold_ER['traj_x'][:truncate_length]
fold_BA_x = df_fold_BA['traj_x'][1:truncate_length+1]
trans_x = df_trans['traj_x'][:truncate_length]
sup_hopf_x = df_sup_hopf['traj_x'][:truncate_length]
sub_hopf_x = df_sub_hopf['traj_x'][:truncate_length]

sup_hopf_esti_stable = df_sup_hopf_stable['esti_stable_mean']
sup_hopf_esti_stable_lower = df_sup_hopf_stable['esti_stable_ci_lower']
sup_hopf_esti_stable_upper = df_sup_hopf_stable['esti_stable_ci_upper']
sub_hopf_esti_stable = df_sub_hopf_stable['esti_stable_mean']
sub_hopf_esti_stable_lower = df_sub_hopf_stable['esti_stable_ci_lower']
sub_hopf_esti_stable_upper = df_sub_hopf_stable['esti_stable_ci_upper']

train_length = 500
length = 1000

################################################################################################

ax1.scatter(t_trans[::5],trans_x[::5],s=40,c='darkslateblue',marker='o')
ax1.plot(t_trans[::5],trans_x[::5],c='darkslateblue',linewidth=3)
ax1.scatter(t_trans[train_length:length][::5],trans_x[train_length:length][::5],s=40,c='darkgrey',marker='o',zorder=4)
ax1.plot(t_trans[train_length:length][::5],trans_x[train_length:length][::5],c='darkgrey',linewidth=3)
ax1.scatter(1000,trans_x[1000],s=800, marker='o',facecolors='white', edgecolors='black',linewidths=3,zorder=5)
ax1.set_xlim(-50,1220)
ax1.set_ylim(1.35,1.85)
ax1.set_yticks([1.4,1.6,1.8])
ax1.set_yticklabels(['1.4','1.6','1.8'])
ax1.set_title('Predator-prey (transcritical)', fontsize=40, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.05)
ax1.text(0.06,0.9,'(a)',ha='left', transform=ax1.transAxes,fontdict={'family':'Arial','size':60,'weight':'normal'})
ax1.axvline(x=0, color='black', linestyle='--', linewidth=2,alpha=0.5, zorder=0)
ax1.axvline(x=500, color='black', linestyle='--', linewidth=2,alpha=0.5, zorder=0)
ax1.tick_params(axis='x', bottom=False, labelbottom=False)
ax1.tick_params(direction='in', length=10)
ax1.minorticks_on()
ax1.tick_params(axis='x', which='minor', bottom=False)
ax1.tick_params(axis='y', which='minor', direction='in', length=5)

ax1.plot([15, 484], [1.67, 1.67], linestyle='--', color='black', linewidth=3)
arrow1 = FancyArrowPatch((5, 1.67), (494, 1.67), 
                        arrowstyle='<|-|>', 
                        color='black',
                        linewidth = 0, 
                        mutation_scale=50)
ax1.add_patch(arrow1)

ax1.text(240,1.64,r'$L$', fontsize=50, ha='center', va='center', color='black')

ax1.set_ylabel('System state', font_y)
ax1.yaxis.set_label_coords(-0.2, 0.5)

################################################################################################

ax2.scatter(t_sup_hopf[::5],sup_hopf_x[::5],s=40,c='darkslateblue',marker='o')
ax2.plot(t_sup_hopf[::5],sup_hopf_x[::5],c='darkslateblue',linewidth=3)
ax2.scatter(t_sup_hopf[train_length:length][::5],sup_hopf_x[train_length:length][::5],s=40,c='darkgrey',marker='o',zorder=4)
ax2.plot(t_sup_hopf[train_length:length][::5],sup_hopf_x[train_length:length][::5],c='darkgrey',linewidth=3)
ax2.scatter(1000,sup_hopf_x[1000],s=800, marker='o',facecolors='white', edgecolors='black',linewidths=3,zorder=5)
ax2.set_xlim(-50,1220)
ax2.set_ylim(-0.2,2.2)
ax2.set_yticks([0,1,2])
ax2.set_yticklabels(['0','1','2'])
ax2.set_title('Predator-prey (Hopf)', fontsize=40, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.05)
ax2.text(0.06,0.9,'(b)',ha='left', transform=ax2.transAxes,fontdict={'family':'Arial','size':60,'weight':'normal'})
ax2.axvline(x=0, color='black', linestyle='--', linewidth=2,alpha=0.5, zorder=0)
ax2.axvline(x=500, color='black', linestyle='--', linewidth=2,alpha=0.5, zorder=0)
ax2.tick_params(axis='x', bottom=False, labelbottom=False)
ax2.tick_params(direction='in', length=10)
ax2.minorticks_on()
ax2.tick_params(axis='x', which='minor', bottom=False)
ax2.tick_params(axis='y', which='minor', direction='in', length=5)

ax2_inset = ax2.inset_axes([0.36, 0.1, 0.4, 0.3])

ax2_inset.plot(t_sup_hopf_stable, sup_hopf_esti_stable, color='darkorange', linewidth=6,alpha=0.9,zorder=2)
ax2_inset.fill_between(t_sup_hopf_stable, sup_hopf_esti_stable_lower, sup_hopf_esti_stable_upper,color='darkorange',alpha=0.3,linewidth=0,zorder=2)
ax2_inset.axhline(y=0, color='black', linestyle='--')
ax2_inset.set_xlabel('Time',font_in_x,labelpad=-20)
ax2_inset.set_xticks([951,1000])
ax2_inset.set_xticklabels([951,1000],fontsize=25)
ax2_inset.set_yticks([-0.05,0])
ax2_inset.set_yticklabels(['-0.05','0'],fontsize=25)
ax2_inset.tick_params(direction='in', length=10)
# ax2_inset.yaxis.tick_right()
ax2_inset.set_xlim(949,1002)
ax2_inset.set_ylim(-0.06,0.012)
ax2_inset.minorticks_on()
ax2_inset.tick_params(axis='x', which='minor', bottom=False)
ax2_inset.tick_params(axis='y', which='minor', direction='in', length=5)

sup_hopf_1 = (951, sup_hopf_x[951])
sup_hopf_2 = (1000, sup_hopf_x[1000])

con_left = ConnectionPatch(
    xyA=(0, 1), coordsA='axes fraction', axesA=ax2_inset,
    xyB=sup_hopf_1, coordsB='data', axesB=ax2,
    color='black', lw=2, linestyle=(0, (2, 8)),
    zorder=3, clip_on=False
)
ax2.add_artist(con_left)

con_right = ConnectionPatch(
    xyA=(1, 1), coordsA='axes fraction', axesA=ax2_inset,
    xyB=sup_hopf_2, coordsB='data', axesB=ax2,
    color='black', lw=2, linestyle=(0, (2, 8)),
    zorder=3, clip_on=False
)
ax2.add_artist(con_right)

################################################################################################

ax3.scatter(t_sub_hopf[::5],sub_hopf_x[::5],s=40,c='darkslateblue',marker='o')
ax3.plot(t_sub_hopf[::5],sub_hopf_x[::5],c='darkslateblue',linewidth=3)
ax3.scatter(t_sub_hopf[train_length:length][::5],sub_hopf_x[train_length:length][::5],s=40,c='darkgrey',marker='o',zorder=4)
ax3.plot(t_sub_hopf[train_length:length][::5],sub_hopf_x[train_length:length][::5],c='darkgrey',linewidth=3)
ax3.scatter(1000,sub_hopf_x[1000],s=800, marker='o',facecolors='white', edgecolors='black',linewidths=3,zorder=5)
ax3.set_xlim(-50,1220)
ax3.set_ylim(0,13)
ax3.set_yticks([1,6,12])
ax3.set_title('Epidemic', fontsize=40, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.05)
ax3.text(0.06,0.9,'(c)',ha='left', transform=ax3.transAxes,fontdict={'family':'Arial','size':60,'weight':'normal'})
ax3.axvline(x=0, color='black', linestyle='--', linewidth=2,alpha=0.5, zorder=0)
ax3.axvline(x=500, color='black', linestyle='--', linewidth=2,alpha=0.5, zorder=0)
ax3.tick_params(axis='x', bottom=False, labelbottom=False)
ax3.tick_params(direction='in', length=10)
ax3.minorticks_on()
ax3.tick_params(axis='x', which='minor', bottom=False)
ax3.tick_params(axis='y', which='minor', direction='in', length=5)

ax3_inset = ax3.inset_axes([0.36, 0.3, 0.4, 0.3])

ax3_inset.plot(t_sub_hopf_stable, sub_hopf_esti_stable, color='darkorange', linewidth=6,alpha=0.9,zorder=2)
ax3_inset.fill_between(t_sub_hopf_stable, sub_hopf_esti_stable_lower, sub_hopf_esti_stable_upper,color='darkorange',alpha=0.3,linewidth=0,zorder=2)
ax3_inset.axhline(y=0, color='black', linestyle='--')
ax3_inset.set_xlabel('Time',font_in_x,labelpad=-20)
ax3_inset.set_xticks([951,1000])
ax3_inset.set_xticklabels([951,1000], fontsize=25)
ax3_inset.set_yticks([0,0.06])
ax3_inset.set_yticklabels(['0','0.06'], fontsize=25)
ax3_inset.tick_params(direction='in', length=10)
# ax3_inset.yaxis.tick_right()
ax3_inset.set_xlim(949,1002)
ax3_inset.set_ylim(-0.0144,0.072)
ax3_inset.minorticks_on()
ax3_inset.tick_params(axis='x', which='minor', bottom=False)
ax3_inset.tick_params(axis='y', which='minor', direction='in', length=5)

sub_hopf_1 = (951, sub_hopf_x[951])
sub_hopf_2 = (1000, sub_hopf_x[1000])

con_left = ConnectionPatch(
    xyA=(1, 0), coordsA='axes fraction', axesA=ax3_inset,
    xyB=sub_hopf_1, coordsB='data', axesB=ax3,
    color='black', lw=2, linestyle=(0, (2, 8)),
    zorder=1, clip_on=False
)
ax3.add_artist(con_left)

con_right = ConnectionPatch(
    xyA=(1, 1), coordsA='axes fraction', axesA=ax3_inset,
    xyB=sub_hopf_2, coordsB='data', axesB=ax3,
    color='black', lw=2, linestyle=(0, (2, 8)),
    zorder=1, clip_on=False
)
ax3.add_artist(con_right)

################################################################################################

ax4.scatter(t_fold_ER[::5],fold_ER_x[::5],s=40,c='darkslateblue',marker='o')
ax4.plot(t_fold_ER[::5],fold_ER_x[::5],c='darkslateblue',linewidth=3)
ax4.scatter(t_fold_ER[train_length:length][::5],fold_ER_x[train_length:length][::5],s=40,c='darkgrey',marker='o',zorder=4)
ax4.plot(t_fold_ER[train_length:length][::5],fold_ER_x[train_length:length][::5],c='darkgrey',linewidth=3)
ax4.scatter(1000,fold_ER_x[1000],s=800, marker='o',facecolors='white', edgecolors='black',linewidths=3,zorder=5)

ax4.set_yscale('log')

ax4.set_xlim(-50,1220)
ax4.set_ylim(0.05,15)
ax4.set_yticks([0.1, 1, 10])
ax4.tick_params(axis='x', bottom=False, labelbottom=False)
ax4.tick_params(direction='in', length=10)

ax4.minorticks_on()
ax4.tick_params(axis='x', which='minor', bottom=False)
ax4.tick_params(axis='y', which='minor', direction='in', length=5)

ax4.set_title('Mutualistic network (ER)', fontsize=40, fontweight='normal', fontfamily='DejaVu Sans', fontstyle= 'italic', y=1.05)
ax4.text(0.06,0.9,'(d)',ha='left', transform=ax4.transAxes,fontdict={'family':'Arial','size':60,'weight':'normal'})
ax4.axvline(x=0, color='black', linestyle='--', linewidth=2,alpha=0.5, zorder=0)
ax4.axvline(x=500, color='black', linestyle='--', linewidth=2,alpha=0.5, zorder=0)

################################################################################################

ax5.scatter(t_fold_BA[::5],fold_BA_x[::5],s=40,c='darkslateblue',marker='o')
ax5.plot(t_fold_BA[::5],fold_BA_x[::5],c='darkslateblue',linewidth=3)
ax5.scatter(t_fold_BA[train_length:length][::5],fold_BA_x[train_length:length][::5],s=40,c='darkgrey',marker='o',zorder=4)
ax5.plot(t_fold_BA[train_length:length][::5],fold_BA_x[train_length:length][::5],c='darkgrey',linewidth=3)
ax5.scatter(1000,fold_BA_x[1000],s=800, marker='o',facecolors='white', edgecolors='black',linewidths=3,zorder=5)

ax5.set_yscale('log')

ax5.set_xlim(-50,1220)
ax5.set_ylim(0.0005,20)
ax5.set_yticks([0.001,0.1,10])
ax5.tick_params(axis='x', bottom=False, labelbottom=False)
ax5.tick_params(direction='in', length=10)

ax5.minorticks_on()
ax5.tick_params(axis='x', which='minor', bottom=False)
ax5.tick_params(axis='y', which='minor', direction='in', length=5)

ax5.set_title('Mutualistic network (BA)', fontsize=40, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.05)
ax5.text(0.06,0.9,'(e)',ha='left', transform=ax5.transAxes,fontdict={'family':'Arial','size':60,'weight':'normal'})
ax5.axvline(x=0, color='black', linestyle='--', linewidth=2,alpha=0.5, zorder=0)
ax5.axvline(x=500, color='black', linestyle='--', linewidth=2,alpha=0.5, zorder=0)

################################################################################################

legend_state = mlines.Line2D([], [], color='darkslateblue', linestyle='-',marker='o', linewidth=5, markersize=20)
legend_pstate = mlines.Line2D([], [], color='darkgrey', linestyle='-',marker='o', linewidth=5, markersize=20)
legend_bif = mlines.Line2D([], [], linestyle='None', marker='o',markerfacecolor='white', markeredgecolor='black',markersize=30, markeredgewidth=5)
legend_I = mlines.Line2D([], [], color='darkorange', linestyle='-',linewidth=8,alpha=0.9)
# legend_true_re = mlines.Line2D([], [], color='black', linestyle='-',linewidth=6)
# legend_true_im = mlines.Line2D([], [], color='black', linestyle='--',linewidth=6)
legend_esti_re = mlines.Line2D([], [], color='crimson', linestyle='-',linewidth=8,alpha=0.9)
legend_esti_im = mlines.Line2D([], [], color='royalblue', linestyle='-',linewidth=8,alpha=0.9)

fig.legend(handles = [legend_state, legend_pstate, legend_bif, legend_I, legend_esti_re, legend_esti_im],
            labels = ['State','Forecast regime','Bifurcation point', 'Estimated stability index', 'Estimated Re', 'Estimated Im'],
              loc='upper center', bbox_to_anchor=(0.5,1.01), ncol=6, frameon=False, 
              prop=font_manager.FontProperties(family='Arial', size=40, weight='bold'),handletextpad=0.4)

plt.subplots_adjust(top=0.7, bottom=0.04, left=0.05, right=0.99, wspace=0.2)
plt.savefig('../figures/FIG1.1.pdf',format='pdf')