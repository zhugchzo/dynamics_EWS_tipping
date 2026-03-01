import pandas
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import font_manager

font_x = {'family':'Arial','weight':'normal','size': 60}
font_y = {'family':'Arial','weight':'normal','size': 60}

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['xtick.labelsize'] = 50
plt.rcParams['ytick.labelsize'] = 50

fig, axs = plt.subplots(2, 3, figsize=(36,25))

ax1, ax2, ax3, ax4, ax5, ax6 = axs[0,0], axs[0,1], axs[0,2], axs[1,0], axs[1,1], axs[1,2]

df_fold = pandas.read_csv('../results/fold/auto_output.csv')
df_fold_ER = pandas.read_csv('../results/fold_ER/bif_output.csv')
df_fold_BA = pandas.read_csv('../results/fold_BA/bif_output.csv')
df_trans = pandas.read_csv('../results/trans/auto_output.csv')
df_sup_hopf = pandas.read_csv('../results/sup_hopf/auto_output.csv')
df_sub_hopf = pandas.read_csv('../results/sub_hopf/auto_output.csv')
df_time = pandas.read_csv('../results/fold/fold.csv')

t = df_time['Time'] + 1
t1 = df_fold['Time']
t2 = df_fold_ER['Time']
t3 = df_fold_BA['Time']
t4 = df_trans['Time']
t5 = df_sup_hopf['Time']
t6 = df_sub_hopf['Time']

###################################################

fold_fold = df_fold[df_fold['bif_type'] == 'LP']
fold_NA = df_fold[df_fold['bif_type'].isna()]

t_fold_fold = fold_fold['Time']
bif_fold_fold = fold_fold['bif_value']

t_fold_NA = fold_NA['Time']
t_fold_blow = t[~t.isin(t1)]

###################################################

fold_ER_fold = df_fold_ER[df_fold_ER['bif_type'] == 'LP']
fold_ER_NA = df_fold_ER[df_fold_ER['bif_type'].isna()]

t_fold_ER_fold = fold_ER_fold['Time']
bif_fold_ER_fold = fold_ER_fold['bif_value']

t_fold_ER_NA = fold_ER_NA['Time']
t_fold_ER_blow = t[~t.isin(t2)]

###################################################

fold_BA_fold = df_fold_BA[df_fold_BA['bif_type'] == 'LP']
fold_BA_NA = df_fold_BA[df_fold_BA['bif_type'].isna()]

t_fold_BA_fold = fold_BA_fold['Time']
bif_fold_BA_fold = fold_BA_fold['bif_value']

t_fold_BA_NA = fold_BA_NA['Time']
t_fold_BA_blow = t[~t.isin(t3)]

###################################################

trans_hopf = df_trans[df_trans['bif_type'] == 'HB']
trans_fold = df_trans[df_trans['bif_type'] == 'LP']
trans_NA = df_trans[df_trans['bif_type'].isna()]

t_trans_fold = trans_fold['Time']
bif_trans_fold = trans_fold['bif_value']

t_trans_hopf = trans_hopf['Time']
bif_trans_hopf = trans_hopf['bif_value']

t_trans_NA = trans_NA['Time']
t_trans_blow = t[~t.isin(t4)]

###################################################

sup_hopf_hopf = df_sup_hopf[df_sup_hopf['bif_type'] == 'HB']
sup_hopf_fold = df_sup_hopf[df_sup_hopf['bif_type'] == 'LP']
sup_hopf_NA = df_sup_hopf[df_sup_hopf['bif_type'].isna()]

t_sup_hopf_hopf = sup_hopf_hopf['Time']
bif_sup_hopf_hopf = sup_hopf_hopf['bif_value']

t_sup_hopf_fold = sup_hopf_fold['Time']
bif_sup_hopf_fold = sup_hopf_fold['bif_value']

t_sup_hopf_NA = sup_hopf_NA['Time']
t_sup_hopf_blow = t[~t.isin(t5)]

###################################################

sub_hopf_hopf = df_sub_hopf[df_sub_hopf['bif_type'] == 'HB']
sub_hopf_fold = df_sub_hopf[df_sub_hopf['bif_type'] == 'LP']
sub_hopf_NA = df_sub_hopf[df_sub_hopf['bif_type'].isna()]

t_sub_hopf_hopf = sub_hopf_hopf['Time']
bif_sub_hopf_hopf = 1.2 - sub_hopf_hopf['bif_value']

t_sub_hopf_fold = sub_hopf_fold['Time']
bif_sub_hopf_fold = 1.2 - sub_hopf_fold['bif_value']

t_sub_hopf_NA = sub_hopf_NA['Time']
t_sub_hopf_blow = t[~t.isin(t6)]

###################################################

fold_bif = 2.37
fold_ER_bif = 0.68
fold_BA_bif = 0.11
trans_bif = 5.88
sup_hopf_bif = 2.6
sub_hopf_bif = 0.55

###################################################

ax1.scatter(t_fold_fold,bif_fold_fold,s=200,marker='o',facecolors='none',edgecolors='crimson',linewidths=2,zorder=5)
ax1.scatter(t_fold_NA,[fold_bif]*len(t_fold_NA),s=200,marker='x',color='grey',linewidths=4,zorder=4)
ax1.scatter(t_fold_blow,[fold_bif]*len(t_fold_blow),s=200,marker='x',color='darkorange',linewidths=4,zorder=4)
ax1.text(0.02,0.9,'(a)',ha='left', transform=ax1.transAxes,fontdict={'family':'Arial','size':70,'weight':'normal'})
ax1.set_ylim(0.94,3.8)
ax1.set_xticks([500,1000])
ax1.set_yticks([1.1,fold_bif,3.6])
ax1.set_yticklabels(['1.1','2.37','3.6'])
ax1.tick_params(direction='in', length=10)
ax1.tick_params(axis='x', pad=10)
ax1.axhline(y=fold_bif, color='crimson', linestyle='--', linewidth=4)
ax1.set_ylabel('Value of bifurcation', font_y,labelpad=15)

ax1.set_title('Intraspecific competition (fold)', fontsize=45, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.02)

###################################################

ax2.scatter(t_fold_ER_fold,bif_fold_ER_fold,s=200,marker='o',facecolors='none',edgecolors='crimson',linewidths=2,zorder=5)
ax2.scatter(t_fold_ER_NA,[fold_ER_bif]*len(t_fold_ER_NA),s=200,marker='x',color='grey',linewidths=4,zorder=4)
ax2.scatter(t_fold_ER_blow,[fold_ER_bif]*len(t_fold_ER_blow),s=200,marker='x',color='darkorange',linewidths=4,zorder=4)
ax2.text(0.02,0.9,'(b)',ha='left', transform=ax2.transAxes,fontdict={'family':'Arial','size':70,'weight':'normal'})
ax2.set_ylim(-0.44,1.8)
ax2.set_xticks([500,1000])
ax2.set_yticks([-0.3,fold_ER_bif,1.6])
ax2.set_yticklabels(['-0.3','0.68','1.6'])
ax2.tick_params(direction='in', length=10)
ax2.tick_params(axis='x', pad=10)
ax2.axhline(y=fold_ER_bif, color='crimson', linestyle='--', linewidth=4)

ax2.set_title('Mutualistic network (ER; fold)', fontsize=45, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.02)

###################################################

ax3.scatter(t_fold_BA_fold,bif_fold_BA_fold,s=200,marker='o',facecolors='none',edgecolors='crimson',linewidths=2,zorder=5)
ax3.scatter(t_fold_BA_NA,[fold_BA_bif]*len(t_fold_BA_NA),s=200,marker='x',color='grey',linewidths=4,zorder=4)
ax3.scatter(t_fold_BA_blow,[fold_BA_bif]*len(t_fold_BA_blow),s=200,marker='x',color='darkorange',linewidths=4,zorder=4)
ax3.text(0.02,0.9,'(c)',ha='left', transform=ax3.transAxes,fontdict={'family':'Arial','size':70,'weight':'normal'})
ax3.set_ylim(-0.01,0.23)
ax3.set_xticks([500,1000])
ax3.set_yticks([0.02,fold_BA_bif,0.2])
ax3.tick_params(direction='in', length=10)
ax3.set_yticklabels(['0.02','0.11','0.2'])
ax3.tick_params(axis='x', pad=10)
ax3.axhline(y=fold_BA_bif, color='crimson', linestyle='--', linewidth=4)

ax3.set_title('Mutualistic network (BA; fold)', fontsize=45, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.02)

###################################################

ax4.scatter(t_trans_fold,bif_trans_fold,s=200,marker='o',facecolors='none',edgecolors='crimson',linewidths=2,zorder=6)
ax4.scatter(t_trans_hopf,bif_trans_hopf,s=200,marker='x',color='darkslateblue',linewidths=4,zorder=5)
ax4.scatter(t_trans_NA,[trans_bif]*len(t_trans_NA),s=200,marker='x',color='grey',linewidths=4,zorder=4)
ax4.scatter(t_trans_blow,[trans_bif]*len(t_trans_blow),s=200,marker='x',color='darkorange',linewidths=4,zorder=4)
ax4.text(0.02,0.9,'(d)',ha='left', transform=ax4.transAxes,fontdict={'family':'Arial','size':70,'weight':'normal'})
ax4.set_xlabel('Timepoint',font_x,labelpad=-15)
ax4.set_ylim(-7.24,19)
ax4.set_xticks([500,1000])
ax4.set_yticks([-5,trans_bif,17])
ax4.set_yticklabels(['-5','5.88','17'])
ax4.tick_params(direction='in', length=10)
ax4.tick_params(axis='x', pad=10)
ax4.axhline(y=trans_bif, color='crimson', linestyle='--', linewidth=4)
ax4.set_ylabel('Value of bifurcation', font_y,labelpad=15)

ax4.set_title('Predator-prey (transcritical)', fontsize=45, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.02)

###################################################

ax5.scatter(t_sup_hopf_hopf,bif_sup_hopf_hopf,s=200,marker='o',facecolors='none',edgecolors='darkslateblue',linewidths=2,zorder=6)
ax5.scatter(t_sup_hopf_fold,bif_sup_hopf_fold,s=200,marker='x',color='crimson',linewidths=4,zorder=5)
ax5.scatter(t_sup_hopf_NA,[sup_hopf_bif]*len(t_sup_hopf_NA),s=200,marker='x',color='crimson',linewidths=4,zorder=4)
ax5.scatter(t_sup_hopf_blow,[sup_hopf_bif]*len(t_sup_hopf_blow),s=200,marker='x',color='darkorange',linewidths=4,zorder=4)
ax5.text(0.02,0.9,'(e)',ha='left', transform=ax5.transAxes,fontdict={'family':'Arial','size':70,'weight':'normal'})
ax5.set_xlabel('Timepoint',font_x,labelpad=-15)
ax5.set_ylim(1.3,3.9)
ax5.set_xticks([500,1000])
ax5.set_yticks([1.5,sup_hopf_bif,3.7])
ax5.set_yticklabels(['1.5','2.6','3.7'])
ax5.tick_params(direction='in', length=10)
ax5.tick_params(axis='x', pad=10)
ax5.axhline(y=sup_hopf_bif, color='darkslateblue', linestyle='--', linewidth=4)

ax5.set_title('Predator-prey (Hopf)', fontsize=45, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.02)

###################################################

ax6.scatter(t_sub_hopf_hopf,bif_sub_hopf_hopf,s=200,marker='o',facecolors='none',edgecolors='darkslateblue',linewidths=2,zorder=6)
ax6.scatter(t_sub_hopf_fold,bif_sub_hopf_fold,s=200,marker='x',color='crimson',linewidths=4,zorder=5)
ax6.scatter(t_sub_hopf_NA,[sub_hopf_bif]*len(t_sub_hopf_NA),s=200,marker='x',color='grey',linewidths=4,zorder=4)
ax6.scatter(t_sub_hopf_blow,[sub_hopf_bif]*len(t_sub_hopf_blow),s=200,marker='x',color='darkorange',linewidths=4,zorder=4)
ax6.text(0.02,0.9,'(f)',ha='left', transform=ax6.transAxes,fontdict={'family':'Arial','size':70,'weight':'normal'})
ax6.set_xlabel('Timepoint',font_x,labelpad=-15)
ax6.set_ylim(-1.8,2.9)
ax6.set_xticks([500,1000])
ax6.set_yticks([-1.5,sub_hopf_bif,2.6])
ax6.set_yticklabels(['-1.5','0.55','2.6'])
ax6.tick_params(direction='in', length=10)
ax6.tick_params(axis='x', pad=10)
ax6.axhline(y=sub_hopf_bif, color='darkslateblue', linestyle='--', linewidth=4)

ax6.set_title('Epidemic (Hopf)', fontsize=45, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.02)

################################################################################################

legend_steady = mlines.Line2D([], [], color='crimson',linestyle='-',linewidth=15)
legend_hopf = mlines.Line2D([], [], color='darkslateblue',linestyle='-',linewidth=15)
legend_nobif = mlines.Line2D([], [], color='grey',linestyle='-',linewidth=15)
legend_blow = mlines.Line2D([], [], color='darkorange',linestyle='-',linewidth=15)

legend_bif_right = mlines.Line2D([], [], linestyle='None', marker='o',markerfacecolor='white',
                                  markeredgecolor='black', markersize=30, markeredgewidth=3)
legend_bif_wrong = mlines.Line2D([], [], linestyle='None', marker='x',markeredgecolor='black', 
                                 markersize=30, markeredgewidth=5)
legend_bif_val = mlines.Line2D([], [], color='black',linestyle='--',linewidth=5)

fig.legend(handles = [legend_steady, legend_hopf, legend_nobif, legend_blow],
            labels = ['Steady', 'Hopf', 'No bifurcation', 'Divergence'],
              loc='upper center', bbox_to_anchor=(0.5,1.02), ncol=4, frameon=False, 
              prop=font_manager.FontProperties(family='Arial', size=50))

fig.legend(handles = [legend_bif_right, legend_bif_wrong, legend_bif_val],
            labels = ['True type', 'False type', 'Bifurcation value'],
              loc='upper center', bbox_to_anchor=(0.5,0.98), ncol=3, frameon=False, 
              prop=font_manager.FontProperties(family='Arial', size=50))

plt.subplots_adjust(top=0.85, bottom=0.06, left=0.07, right=0.99, hspace=0.3, wspace=0.2)
plt.savefig('../figures/SFIG4.pdf',format='pdf')