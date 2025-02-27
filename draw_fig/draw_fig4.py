import numpy as np
import pandas
import matplotlib.pyplot as plt

font_x = {'family':'Arial','weight':'normal','size': 40}
font_y = {'family':'Arial','weight':'normal','size': 40}

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 26
plt.rcParams['ytick.labelsize'] = 26

fig, axs = plt.subplots(2, 1, figsize=(12,13))

(ax1, ax2) = axs.flatten()

df_fold = pandas.read_csv('../results/fold/auto_output.csv')
df_hopf = pandas.read_csv('../results/sub_hopf/auto_output.csv')
df_time = pandas.read_csv('../results/fold/fold.csv')

t = df_time['Time'] + 1
t1 = df_fold['Time']
t2 = df_hopf['Time']

###################################################

fold_fold = df_fold[df_fold['bif_type'] == 'LP']
fold_NA = df_fold[df_fold['bif_type'].isna()]

t_fold_fold = fold_fold['Time']
bif_fold_fold = fold_fold['bif_value']

t_fold_NA = fold_NA['Time']
bif_fold_NA = np.zeros(len(t_fold_NA))+2.1

t_fold_blow = t[~t.isin(t1)]
bif_fold_blow = np.zeros(len(t_fold_blow))+2.1

###################################################

hopf_hopf = df_hopf[df_hopf['bif_type'] == 'HB']
hopf_fold = df_hopf[df_hopf['bif_type'] == 'LP']
hopf_NA = df_hopf[df_hopf['bif_type'].isna()]

t_hopf_hopf = hopf_hopf['Time']
bif_hopf_hopf = 1.2 - hopf_hopf['bif_value']

t_hopf_fold = hopf_fold['Time']
bif_hopf_fold = 1.2 - hopf_fold['bif_value']

t_hopf_NA = hopf_NA['Time']
bif_hopf_NA = np.zeros(len(t_hopf_NA))

t_hopf_blow = t[~t.isin(t2)]
bif_hopf_blow = np.zeros(len(t_hopf_blow))

fold_bif = 2.37
hopf_bif = 0.55  # a = 0.55, c = 0.65

subplt = ax1

legend_fold_fold = subplt.scatter(t_fold_fold,bif_fold_fold,s=100,marker=r'$\checkmark$',c='crimson',zorder=5)
legend_fold_NA = subplt.scatter(t_fold_NA,bif_fold_NA,s=80,marker=r'$\times$',c='grey',zorder=5)
legend_fold_blow = subplt.scatter(t_fold_blow,bif_fold_blow,s=80,marker='^',c='forestgreen',zorder=5)
subplt.text(-0.1, 0.95,'a',ha='left', transform=subplt.transAxes,fontdict={'family':'DejaVu Sans','size':50,'weight':'bold'})
subplt.set_ylim(2,3)
subplt.set_xticks([500,1000])
subplt.set_yticks([fold_bif,2.6,2.9])
subplt.axhline(y=2.1, color='black', linestyle=':', linewidth=2)
subplt.axhline(y=fold_bif, color='crimson', linestyle='--', linewidth=2)
subplt.text(1030,fold_bif,'fold', fontsize=28, ha='left', va='center', color='black')
subplt.text(750,2.05,'divergence or no bifurcation', fontsize=24, ha='left', va='center', color='black')
subplt.tick_params(axis='x', labelbottom=False)
subplt.set_yticklabels(['2.37','2.6','2.9'])

subplt.set_title('Insect outbreak system (codim-1 fold)', fontsize=40, fontweight='normal', fontfamily='Arial', y=1.03)
subplt.legend(handles=[legend_fold_fold,legend_fold_NA,legend_fold_blow],labels=['fold','no bifurcation','divergence'],loc='center left', bbox_to_anchor=(0.6, 0.8), markerscale=2,prop={'size':24})

subplt = ax2

legend_hopf_hopf = subplt.scatter(t_hopf_hopf,bif_hopf_hopf,s=100,marker=r'$\checkmark$',c='royalblue',zorder=6)
legend_hopf_fold = subplt.scatter(t_hopf_fold,bif_hopf_fold,s=80,marker=r'$\times$',c='crimson',zorder=5)
legend_hopf_NA = subplt.scatter(t_hopf_NA,bif_hopf_NA,s=80,marker=r'$\times$',c='grey',zorder=5)
# legend_hopf_blow = subplt.scatter(t_hopf_blow,bif_hopf_blow,s=80,marker='^',c='forestgreen',zorder=5)
subplt.text(-0.1, 0.95,'b',ha='left', transform=subplt.transAxes,fontdict={'family':'DejaVu Sans','size':50,'weight':'bold'})
subplt.set_xlabel('Timepoint',font_x,labelpad=-20)
# subplt.yaxis.set_label_coords(-0.05, 0.475)
subplt.set_ylim(-0.1,0.85)
subplt.set_xticks([500,1000])
subplt.set_yticks([0.1,hopf_bif])
subplt.axhline(y=0, color='black', linestyle=':', linewidth=2)
subplt.axhline(y=hopf_bif, color='royalblue', linestyle='--', linewidth=2)
subplt.text(1030,hopf_bif,'Hopf', fontsize=28, ha='left', va='center', color='black')
subplt.text(750,-0.05,'divergence or no bifurcation', fontsize=24, ha='left', va='center', color='black')
subplt.set_yticklabels(['(0.1,1.1)','(0.55,0.65)'])

subplt.set_title('Epidemiological system (codim-2 Hopf)', fontsize=40, fontweight='normal', fontfamily='Arial', y=1.03)
subplt.legend(handles=[legend_hopf_hopf,legend_hopf_fold,legend_hopf_NA],labels=['Hopf','fold','no bifurcation'],loc='center left', bbox_to_anchor=(0.05, 0.3),markerscale=2,prop={'size':24})

fig.supylabel('Value of bifurcation',x=0.01, y=0.6, fontproperties=font_y)

plt.subplots_adjust(top=0.94, bottom=0.06, left=0.16, right=0.92, hspace=0.18)
plt.savefig('../figures/FIG4.pdf',format='pdf',dpi=600)



