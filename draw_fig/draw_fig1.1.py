import pandas
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.lines as mlines
from matplotlib import font_manager
from matplotlib.patches import ConnectionPatch

font_x = {'family':'Arial','weight':'normal','size': 60}
font_y = {'family':'Arial','weight':'normal','size': 60}
font_in_x = {'family':'Arial','weight':'normal','size': 35}
font_in_y = {'family':'Arial','weight':'normal','size': 35}

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['xtick.labelsize'] = 50
plt.rcParams['ytick.labelsize'] = 50

fig, axs = plt.subplots(1, 5, figsize=(48,10))

ax1, ax2, ax3, ax4, ax5 = axs[0], axs[1], axs[2], axs[3], axs[4]

df_fold_BA = pandas.read_csv('../data/fold_BA/fold_BA.csv')

df_fold_ER = pandas.read_csv('../data/fold_ER/fold_ER.csv')

df_sup_hopf = pandas.read_csv('../data/sup_hopf/sup_hopf.csv')
df_sup_hopf_stable = pandas.read_csv('../results/sup_hopf/sup_hopf_stable.csv')

df_trans = pandas.read_csv('../data/trans/trans.csv')

df_sub_hopf = pandas.read_csv('../data/sub_hopf/sub_hopf.csv')
df_sub_hopf_stable = pandas.read_csv('../results/sub_hopf/sub_hopf_stable.csv')

t_fold_ER = df_fold_ER['Time'] + 1
t_fold_BA = df_fold_BA['Time'] + 1
t_trans = df_trans['Time'][:-10] + 1
t_sup_hopf = df_sup_hopf['Time'] + 1
t_sup_hopf_stable = df_sup_hopf_stable['Time'] + 1
t_sub_hopf = df_sub_hopf['Time'] + 1
t_sub_hopf_stable = df_sub_hopf_stable['Time'] + 1

fold_ER_x = df_fold_ER['traj_x']
fold_BA_x = df_fold_BA['traj_x']
trans_x = df_trans['traj_x'][:-10]
sub_hopf_x = df_sub_hopf['traj_x']
sup_hopf_x = df_sup_hopf['traj_x']
sup_hopf_esti_stbexpnt = df_sup_hopf_stable['stable_expnt']
sub_hopf_esti_stbexpnt = df_sub_hopf_stable['stable_expnt']

train_length = 500
length = 1000

ax1.scatter(t_fold_ER[:1150][::5],fold_ER_x[:1150][::5],s=40,c='darkslateblue',marker='o')
ax1.plot(t_fold_ER[:1150][::5],fold_ER_x[:1150][::5],c='darkslateblue',linewidth=3)
ax1.scatter(t_fold_ER[train_length:length][::5],fold_ER_x[train_length:length][::5],s=40,c='crimson',marker='o')
ax1.plot(t_fold_ER[train_length:length][::5],fold_ER_x[train_length:length][::5],c='crimson',linewidth=3)
ax1.scatter(1000,fold_ER_x[1000],s=800, marker='o',facecolors='white', edgecolors='black',linewidths=3,zorder=5)

ax1.set_yscale('log')

ax1.set_xlim(-58,1160)
ax1.set_ylim(0.05,15)
ax1.set_yticks([0.1, 1, 10])
ax1.tick_params(axis='x', bottom=False, labelbottom=False)
ax1.tick_params(direction='in', length=10)

ax1.minorticks_on()
ax1.tick_params(axis='x', which='minor', bottom=False)
ax1.tick_params(axis='y', which='minor', direction='in', length=5)

ax1.set_ylabel('System state',font_y,labelpad=5)

ax1.set_title('Mutualistic network (ER)', fontsize=40, fontweight='normal', fontfamily='DejaVu Sans', fontstyle= 'italic', y=1.05)
ax1.text(0.06,0.9,'(a)',ha='left', transform=ax1.transAxes,fontdict={'family':'Arial','size':60,'weight':'normal'})
ax1.axvline(x=0, color='black', linestyle='--', linewidth=2, zorder=0)
ax1.axvline(x=500, color='black', linestyle='--', linewidth=2, zorder=0)

ax1.plot([15, 484], [1, 1], linestyle='--', color='black', linewidth=3)
arrow1 = FancyArrowPatch((5, 1), (494, 1), 
                        arrowstyle='<|-|>', 
                        color='black',
                        linewidth = 0, 
                        mutation_scale=50)
ax1.add_patch(arrow1)

ax1.text(240,0.5,r'$L$', fontsize=50, ha='center', va='center', color='black')

################################################################################################

ax2.scatter(t_fold_BA[:1150][::5][1:],fold_BA_x[:1150][::5][1:],s=40,c='darkslateblue',marker='o')
ax2.plot(t_fold_BA[:1150][::5][1:],fold_BA_x[:1150][::5][1:],c='darkslateblue',linewidth=3)
ax2.scatter(t_fold_BA[train_length:length][::5],fold_BA_x[train_length:length][::5],s=40,c='crimson',marker='o')
ax2.plot(t_fold_BA[train_length:length][::5],fold_BA_x[train_length:length][::5],c='crimson',linewidth=3)
ax2.scatter(1000,fold_BA_x[1000],s=800, marker='o',facecolors='white', edgecolors='black',linewidths=3,zorder=5)

ax2.set_yscale('log')

ax2.set_xlim(-58,1160)
ax2.set_ylim(0.0005,20)
ax2.set_yticks([0.001,0.1,10])
ax2.tick_params(axis='x', bottom=False, labelbottom=False)
ax2.tick_params(direction='in', length=10)

ax2.minorticks_on()
ax2.tick_params(axis='x', which='minor', bottom=False)
ax2.tick_params(axis='y', which='minor', direction='in', length=5)

ax2.set_title('Mutualistic network (BA)', fontsize=40, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.05)
ax2.text(0.06,0.9,'(b)',ha='left', transform=ax2.transAxes,fontdict={'family':'Arial','size':60,'weight':'normal'})
ax2.axvline(x=0, color='black', linestyle='--', linewidth=2, zorder=0)
ax2.axvline(x=500, color='black', linestyle='--', linewidth=2, zorder=0)

################################################################################################

ax3.scatter(t_trans[::5],trans_x[::5],s=40,c='darkslateblue',marker='o')
ax3.plot(t_trans[::5],trans_x[::5],c='darkslateblue',linewidth=3)
ax3.scatter(t_trans[train_length:length][::5],trans_x[train_length:length][::5],s=40,c='crimson',marker='o')
ax3.plot(t_trans[train_length:length][::5],trans_x[train_length:length][::5],c='crimson',linewidth=3)
ax3.scatter(999.5,1.004*(trans_x[999]+trans_x[1000])/2,s=800, marker='o',facecolors='white', edgecolors='black',linewidths=3,zorder=5)
ax3.set_xlim(-60,1200)
ax3.set_ylim(1.42,1.8)
ax3.set_yticks([1.5,1.7])
ax3.set_yticklabels(['1.5','1.7'])
ax3.set_title('Predator-prey (transcritical)', fontsize=40, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.05)
ax3.text(0.06,0.9,'(c)',ha='left', transform=ax3.transAxes,fontdict={'family':'Arial','size':60,'weight':'normal'})
ax3.axvline(x=0, color='black', linestyle='--', linewidth=2, zorder=0)
ax3.axvline(x=500, color='black', linestyle='--', linewidth=2, zorder=0)
ax3.tick_params(axis='x', bottom=False, labelbottom=False)
ax3.tick_params(direction='in', length=10)

################################################################################################

ax4.scatter(t_sup_hopf[:1200][::5],sup_hopf_x[:1200][::5],s=40,c='darkslateblue',marker='o')
ax4.plot(t_sup_hopf[:1200][::5],sup_hopf_x[:1200][::5],c='darkslateblue',linewidth=3)
ax4.scatter(t_sup_hopf[train_length:length][::5],sup_hopf_x[train_length:length][::5],s=40,c='crimson',marker='o')
ax4.plot(t_sup_hopf[train_length:length][::5],sup_hopf_x[train_length:length][::5],c='crimson',linewidth=3)
ax4.scatter(1000,sup_hopf_x[1000],s=800, marker='o',facecolors='white', edgecolors='black',linewidths=3,zorder=5)
ax4.set_xlim(-61,1220)
ax4.set_ylim(-1,2)
ax4.set_title('Predator-prey (Hopf)', fontsize=40, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.05)
ax4.text(0.06,0.9,'(d)',ha='left', transform=ax4.transAxes,fontdict={'family':'Arial','size':60,'weight':'normal'})
ax4.axvline(x=0, color='black', linestyle='--', linewidth=2, zorder=0)
ax4.axvline(x=500, color='black', linestyle='--', linewidth=2, zorder=0)
ax4.tick_params(axis='x', bottom=False, labelbottom=False)
ax4.tick_params(direction='in', length=10)
ax4.set_yticks([-0.5, 1.5])
ax4.set_yticklabels(['-0.5','1.5'])

ax4_inset = ax4.inset_axes([0.47, 0.2, 0.4, 0.3])

ax4_inset.scatter(t_sup_hopf_stable, sup_hopf_esti_stbexpnt, color='darkorange', marker='x',s=200,linewidths=5)
ax4_inset.axhline(y=0, color='black', linestyle='--')
ax4_inset.set_xlabel('Timepoint',font_in_x,labelpad=0, x=0.5)
ax4_inset.set_xticks([971,1000])
ax4_inset.set_xticklabels([971,1000],fontsize=30)
ax4_inset.set_yticks([-0.2,0])
ax4_inset.set_yticklabels(['-0.2','0'],fontsize=30)
ax4_inset.tick_params(direction='in', length=10)
ax4_inset.yaxis.tick_right()
ax4_inset.set_xlim(969,1002)
ax4_inset.set_ylim(-0.24,0.048)

sup_hopf_1 = (971, sup_hopf_x[971])
sup_hopf_2 = (1000, sup_hopf_x[1000])

con_left = ConnectionPatch(
    xyA=(0, 1), coordsA='axes fraction', axesA=ax4_inset,
    xyB=sup_hopf_1, coordsB='data', axesB=ax4,
    color='black', lw=2, linestyle=(0, (2, 8)),
    zorder=1, clip_on=False
)
ax4.add_artist(con_left)

con_right = ConnectionPatch(
    xyA=(1, 1), coordsA='axes fraction', axesA=ax4_inset,
    xyB=sup_hopf_2, coordsB='data', axesB=ax4,
    color='black', lw=2, linestyle=(0, (2, 8)),
    zorder=1, clip_on=False
)
ax4.add_artist(con_right)

################################################################################################

ax5.scatter(t_sub_hopf[::5],sub_hopf_x[::5],s=40,c='darkslateblue',marker='o')
ax5.plot(t_sub_hopf[::5],sub_hopf_x[::5],c='darkslateblue',linewidth=3)
ax5.scatter(t_sub_hopf[train_length:length][::5],sub_hopf_x[train_length:length][::5],s=40,c='crimson',marker='o')
ax5.plot(t_sub_hopf[train_length:length][::5],sub_hopf_x[train_length:length][::5],c='crimson',linewidth=3)
ax5.scatter(1000,sub_hopf_x[1000],s=800, marker='o',facecolors='white', edgecolors='black',linewidths=3,zorder=5)
ax5.set_xlim(-62,1265)
ax5.set_ylim(-1,12)
ax5.set_title('Epidemic', fontsize=40, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.05)
ax5.text(0.06,0.9,'(e)',ha='left', transform=ax5.transAxes,fontdict={'family':'Arial','size':60,'weight':'normal'})
ax5.axvline(x=0, color='black', linestyle='--', linewidth=2, zorder=0)
ax5.axvline(x=500, color='black', linestyle='--', linewidth=2, zorder=0)
ax5.set_yticks([1,10])
ax5.tick_params(axis='x', bottom=False, labelbottom=False)
ax5.tick_params(direction='in', length=10)

ax5_inset = ax5.inset_axes([0.46, 0.5, 0.4, 0.3])

ax5_inset.scatter(t_sub_hopf_stable, sub_hopf_esti_stbexpnt, color='darkorange', marker='x',s=200,linewidths=5)
ax5_inset.axhline(y=0, color='black', linestyle='--')
ax5_inset.set_xlabel('Timepoint',font_in_x,labelpad=0, x=0.6)
ax5_inset.set_xticks([971,1000])
ax5_inset.set_xticklabels([971,1000], fontsize=30)
ax5_inset.set_yticks([0,0.6])
ax5_inset.set_yticklabels(['0','0.6'], fontsize=30)
ax5_inset.tick_params(direction='in', length=10)
ax5_inset.yaxis.tick_right()
ax5_inset.set_xlim(969,1002)
ax5_inset.set_ylim(-0.16,0.8)

sub_hopf_1 = (971, sub_hopf_x[971])
sub_hopf_2 = (1000, sub_hopf_x[1000])

con_left = ConnectionPatch(
    xyA=(0, 0), coordsA='axes fraction', axesA=ax5_inset,
    xyB=sub_hopf_1, coordsB='data', axesB=ax5,
    color='black', lw=2, linestyle=(0, (2, 8)),
    zorder=1, clip_on=False
)
ax5.add_artist(con_left)

con_right = ConnectionPatch(
    xyA=(1, 0), coordsA='axes fraction', axesA=ax5_inset,
    xyB=sub_hopf_2, coordsB='data', axesB=ax5,
    color='black', lw=2, linestyle=(0, (2, 8)),
    zorder=1, clip_on=False
)
ax5.add_artist(con_right)

################################################################################################

legend_state = mlines.Line2D([], [], color='darkslateblue', linestyle='-',marker='o', linewidth=5, markersize=20)
legend_pstate = mlines.Line2D([], [], color='crimson', linestyle='-',marker='o', linewidth=5, markersize=20)
legend_bif = mlines.Line2D([], [], linestyle='None', marker='o',markerfacecolor='white', markeredgecolor='black',
    markersize=30, markeredgewidth=5)
legend_I = mlines.Line2D([], [], linestyle='None', marker='x',color='darkorange', markersize=30, markeredgewidth=8)
legend_true_re = mlines.Line2D([], [], color='darkviolet', linestyle='-',linewidth=10)
legend_true_im = mlines.Line2D([], [], color='darkgreen', linestyle='-',linewidth=10)
legend_esti_re = mlines.Line2D([], [], linestyle='None', marker='o',markerfacecolor='white', markeredgecolor='crimson',
    markersize=30, markeredgewidth=3)
legend_esti_im = mlines.Line2D([], [], linestyle='None', marker='s',markerfacecolor='white', markeredgecolor='crimson',
    markersize=30, markeredgewidth=3)

fig.legend(handles = [legend_state, legend_pstate, legend_bif, legend_I, legend_true_re, legend_true_im, legend_esti_re, legend_esti_im],
            labels = ['State','Forecast interval','Bifurcation point', 'Estimated stability index', 'True Re', 'True Im', 'Estimated Re', 'Estimated Im'],
              loc='upper center', bbox_to_anchor=(0.5,1.01), ncol=8, frameon=False, 
              prop=font_manager.FontProperties(family='Arial', size=40, weight='bold'),handletextpad=0.4)

plt.subplots_adjust(top=0.7, bottom=0.04, left=0.05, right=0.99, wspace=0.2)
plt.savefig('../figures/FIG1.1.pdf',format='pdf')