import pandas
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import font_manager

font_x = {'family':'Arial','weight':'normal','size': 40}
font_y = {'family':'Arial','weight':'normal','size': 40}

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['xtick.labelsize'] = 36
plt.rcParams['ytick.labelsize'] = 36

fig, axs = plt.subplots(1, 4, figsize=(36,10))

(ax1, ax2, ax3, ax4) = axs.flatten()

df_fold = pandas.read_csv('../results/fold/auto_output.csv')
df_trans = pandas.read_csv('../results/trans/auto_output.csv')
df_sup_hopf = pandas.read_csv('../results/sup_hopf/auto_output.csv')
df_sub_hopf = pandas.read_csv('../results/sub_hopf/auto_output.csv')
df_time = pandas.read_csv('../results/fold/fold.csv')

t = df_time['Time'] + 1
t1 = df_fold['Time']
t2 = df_trans['Time']
t3 = df_sup_hopf['Time']
t4 = df_sub_hopf['Time']

###################################################

fold_fold = df_fold[df_fold['bif_type'] == 'LP']
fold_NA = df_fold[df_fold['bif_type'].isna()]

t_fold_fold = fold_fold['Time']
bif_fold_fold = fold_fold['bif_value']

t_fold_NA = fold_NA['Time']
t_fold_blow = t[~t.isin(t1)]

###################################################

trans_hopf = df_trans[df_trans['bif_type'] == 'HB']
trans_fold = df_trans[df_trans['bif_type'] == 'LP']
trans_NA = df_trans[df_trans['bif_type'].isna()]

t_trans_fold = trans_fold['Time']
bif_trans_fold = trans_fold['bif_value']

t_trans_hopf = trans_hopf['Time']
bif_trans_hopf = trans_hopf['bif_value']

t_trans_NA = trans_NA['Time']
t_trans_blow = t[~t.isin(t2)]

###################################################

sup_hopf_hopf = df_sup_hopf[df_sup_hopf['bif_type'] == 'HB']
sup_hopf_fold = df_sup_hopf[df_sup_hopf['bif_type'] == 'LP']
sup_hopf_NA = df_sup_hopf[df_sup_hopf['bif_type'].isna()]

t_sup_hopf_hopf = sup_hopf_hopf['Time']
bif_sup_hopf_hopf = sup_hopf_hopf['bif_value']

t_sup_hopf_fold = sup_hopf_fold['Time']
bif_sup_hopf_fold = sup_hopf_fold['bif_value']

t_sup_hopf_NA = sup_hopf_NA['Time']
t_sup_hopf_blow = t[~t.isin(t3)]

###################################################

sub_hopf_hopf = df_sub_hopf[df_sub_hopf['bif_type'] == 'HB']
sub_hopf_fold = df_sub_hopf[df_sub_hopf['bif_type'] == 'LP']
sub_hopf_NA = df_sub_hopf[df_sub_hopf['bif_type'].isna()]

t_sub_hopf_hopf = sub_hopf_hopf['Time']
bif_sub_hopf_hopf = 1.2 - sub_hopf_hopf['bif_value']

t_sub_hopf_fold = sub_hopf_fold['Time']
bif_sub_hopf_fold = 1.2 - sub_hopf_fold['bif_value']

t_sub_hopf_NA = sub_hopf_NA['Time']
t_sub_hopf_blow = t[~t.isin(t4)]

###################################################

fold_bif = 2.37
trans_bif = 5.88
sup_hopf_bif = 2.6
sub_hopf_bif = 0.55  # a = 0.55, c = 0.65

###################################################

legend_fold_fold = ax1.scatter(t_fold_fold,bif_fold_fold,s=120,marker=r'$\checkmark$',c='crimson',zorder=5)
legend_fold_NA = ax1.scatter(t_fold_NA,[fold_bif]*len(t_fold_NA),s=80,marker=r'$\times$',c='grey',zorder=4)
legend_fold_blow = ax1.scatter(t_fold_blow,[fold_bif]*len(t_fold_blow),s=80,marker=r'$\times$',c='darkorange',zorder=4)
ax1.text(-0.1, 1.05,'(a)',ha='left', transform=ax1.transAxes,fontdict={'family':'Arial','size':40,'weight':'normal'})
ax1.set_xlabel('Timepoint',font_x,labelpad=-20)
ax1.set_ylim(2,3.1)
ax1.set_xticks([500,1000])
ax1.set_yticks([fold_bif])
ax1.tick_params(direction='in', length=10)
ax1.tick_params(axis='x', pad=10)
ax1.tick_params(axis='y', left=False)
ax1.axhline(y=fold_bif, color='crimson', linestyle='--', linewidth=4)
ax1.set_yticklabels(['2.37'])
ax1.set_ylabel('Value of bifurcation', font_y,labelpad=15)

ax1.set_title('Fold', fontsize=40, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.02)

###################################################

legend_trans_fold = ax2.scatter(t_trans_fold,bif_trans_fold,s=120,marker=r'$\checkmark$',c='crimson',zorder=6)
legend_trans_hopf = ax2.scatter(t_trans_hopf,bif_trans_hopf,s=80,marker=r'$\times$',c='darkslateblue',zorder=5)
legend_trans_NA = ax2.scatter(t_trans_NA,[trans_bif]*len(t_trans_NA),s=80,marker=r'$\times$',c='grey',zorder=4)
legend_trans_blow = ax2.scatter(t_trans_blow,[trans_bif]*len(t_trans_blow),s=80,marker=r'$\times$',c='darkorange',zorder=4)
ax2.text(-0.1, 1.05,'(b)',ha='left', transform=ax2.transAxes,fontdict={'family':'Arial','size':40,'weight':'normal'})
ax2.set_xlabel('Timepoint',font_x,labelpad=-20)
ax2.set_ylim(0,19)
ax2.set_xticks([500,1000])
ax2.set_yticks([trans_bif])
ax2.tick_params(direction='in', length=10)
ax2.tick_params(axis='x', pad=10)
ax2.tick_params(axis='y', left=False)
ax2.axhline(y=trans_bif, color='crimson', linestyle='--', linewidth=4)
ax2.set_yticklabels(['5.88'])

ax2.set_title('Transcritical', fontsize=40, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.02)

###################################################

legend_hopf_hopf = ax3.scatter(t_sup_hopf_hopf,bif_sup_hopf_hopf,s=120,marker=r'$\checkmark$',c='darkslateblue',zorder=6)
legend_hopf_fold = ax3.scatter(t_sup_hopf_fold,bif_sup_hopf_fold,s=80,marker=r'$\times$',c='crimson',zorder=5)
legend_hopf_NA = ax3.scatter(t_sup_hopf_NA,[sup_hopf_bif]*len(t_sup_hopf_NA),s=80,marker=r'$\times$',c='grey',zorder=4)
legend_hopf_blow = ax3.scatter(t_sup_hopf_blow,[sup_hopf_bif]*len(t_sup_hopf_blow),s=80,marker=r'$\times$',c='darkorange',zorder=4)
ax3.text(-0.1, 1.05,'(c)',ha='left', transform=ax3.transAxes,fontdict={'family':'Arial','size':40,'weight':'normal'})
ax3.set_xlabel('Timepoint',font_x,labelpad=-20)
ax3.set_ylim(2,3)
ax3.set_xticks([500,1000])
ax3.set_yticks([sup_hopf_bif])
ax3.tick_params(direction='in', length=10)
ax3.tick_params(axis='x', pad=10)
ax3.tick_params(axis='y', left=False)
ax3.axhline(y=sup_hopf_bif, color='darkslateblue', linestyle='--', linewidth=4)
ax3.set_yticklabels(['2.6'])

ax3.set_title('Supercritical Hopf', fontsize=40, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.02)

###################################################

legend_hopf_hopf = ax4.scatter(t_sub_hopf_hopf,bif_sub_hopf_hopf,s=120,marker=r'$\checkmark$',c='darkslateblue',zorder=6)
legend_hopf_fold = ax4.scatter(t_sub_hopf_fold,bif_sub_hopf_fold,s=80,marker=r'$\times$',c='crimson',zorder=5)
legend_hopf_NA = ax4.scatter(t_sub_hopf_NA,[sub_hopf_bif]*len(t_sub_hopf_NA),s=80,marker=r'$\times$',c='grey',zorder=4)
legend_hopf_blow = ax4.scatter(t_sub_hopf_blow,[sub_hopf_bif]*len(t_sub_hopf_blow),s=80,marker=r'$\times$',c='darkorange',zorder=4)
ax4.text(-0.1, 1.05,'(d)',ha='left', transform=ax4.transAxes,fontdict={'family':'Arial','size':40,'weight':'normal'})
ax4.set_xlabel('Timepoint',font_x,labelpad=-20)
ax4.set_ylim(0.02,0.85)
ax4.set_xticks([500,1000])
ax4.set_yticks([sub_hopf_bif])
ax4.tick_params(direction='in', length=10)
ax4.tick_params(axis='x', pad=10)
ax4.tick_params(axis='y', left=False)
ax4.axhline(y=sub_hopf_bif, color='darkslateblue', linestyle='--', linewidth=4)
ax4.set_yticklabels(['(0.55,0.65)'], fontsize=24)

ax4.set_title('Subcritical Hopf', fontsize=40, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.02)

################################################################################################

legend_bif_right = mlines.Line2D([], [], color='black', marker=r'$\checkmark$', linestyle='None', markersize=30)
legend_bif_wrong = mlines.Line2D([], [], color='black', marker=r'$\times$', linestyle='None', markersize=20)
legend_bif_val = mlines.Line2D([], [], color='black',linestyle='--',linewidth=5)
legend_steady = mlines.Line2D([], [], color='crimson',linestyle='-',linewidth=15)
legend_hopf = mlines.Line2D([], [], color='darkslateblue',linestyle='-',linewidth=15)
legend_nobif = mlines.Line2D([], [], color='grey',linestyle='-',linewidth=15)
legend_blow = mlines.Line2D([], [], color='darkorange',linestyle='-',linewidth=15)

fig.legend(handles = [legend_bif_right, legend_bif_wrong, legend_bif_val, legend_steady, legend_hopf, legend_nobif, legend_blow],
            labels = ['True type', 'False type', 'Bifurcation value', 'Steady', 'Hopf', 'No bifurcation', 'Divergence'],
              loc='upper center', bbox_to_anchor=(0.5,1.01), ncol=7, frameon=False, 
              prop=font_manager.FontProperties(family='Arial', size=36))

plt.subplots_adjust(top=0.76, bottom=0.09, left=0.05, right=0.99, wspace=0.25)
plt.savefig('../figures/SFIG4.pdf',format='pdf')



