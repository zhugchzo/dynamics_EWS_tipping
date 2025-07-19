import numpy as np
import pandas
import matplotlib.pyplot as plt

font_x = {'family':'Arial','weight':'normal','size': 40}
font_y = {'family':'Arial','weight':'normal','size': 40}

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 26
plt.rcParams['ytick.labelsize'] = 26

fig, axs = plt.subplots(1, 3, figsize=(36,8))

(ax1, ax2, ax3) = axs.flatten()

df_fold = pandas.read_csv('../results/fold/auto_output.csv')
df_hopf = pandas.read_csv('../results/sub_hopf/auto_output.csv')
df_trans = pandas.read_csv('../results/trans/auto_output.csv')
df_time = pandas.read_csv('../results/fold/fold.csv')

t = df_time['Time'] + 1
t1 = df_fold['Time']
t2 = df_hopf['Time']
t3 = df_trans['Time']

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

###################################################

trans_trans = df_trans[df_trans['bif_type'] == 'BP']
trans_hopf = df_trans[df_trans['bif_type'] == 'HB']
trans_fold = df_trans[df_trans['bif_type'] == 'LP']
trans_NA = df_trans[df_trans['bif_type'].isna()]

t_trans_trans = trans_trans['Time']
bif_trans_trans = trans_trans['bif_value']

t_trans_hopf = trans_hopf['Time']
bif_trans_hopf = trans_hopf['bif_value']

t_trans_fold = trans_fold['Time']
bif_trans_fold = trans_fold['bif_value']

t_trans_NA = trans_NA['Time']
bif_trans_NA = np.zeros(len(t_trans_NA)) + 2

t_trans_blow = t[~t.isin(t3)]
bif_trans_blow = np.zeros(len(t_trans_blow)) + 2

###################################################

fold_bif = 2.37
hopf_bif = 0.55  # a = 0.55, c = 0.65
trans_bif = 5.88
hopf2_bif = 15.69

###################################################

subplt = ax1

legend_fold_fold = subplt.scatter(t_fold_fold,bif_fold_fold,s=100,marker=r'$\checkmark$',c='crimson',zorder=5)
legend_fold_NA = subplt.scatter(t_fold_NA,bif_fold_NA,s=80,marker=r'$\times$',c='grey',zorder=5)
legend_fold_blow = subplt.scatter(t_fold_blow,bif_fold_blow,s=80,marker='^',c='chocolate',zorder=5)
subplt.text(-0.125, 1,'a',ha='left', transform=subplt.transAxes,fontdict={'family':'DejaVu Sans','size':50,'weight':'bold'})
subplt.set_xlabel('Timepoint',font_x,labelpad=-20)
subplt.set_ylim(2,3)
subplt.set_xticks([500,1000])
subplt.set_yticks([fold_bif,2.6,2.9])
subplt.axhline(y=2.1, color='black', linestyle=':', linewidth=2)
subplt.axhline(y=fold_bif, color='crimson', linestyle='--', linewidth=2)
subplt.text(750,2.05,'divergence or no bifurcation', fontsize=24, ha='left', va='center', color='black')
# subplt.tick_params(axis='x', labelbottom=False)
subplt.set_yticklabels(['2.37\nfold','2.6','2.9'])
subplt.set_ylabel('Value of bifurcation', font_y,labelpad=15)

subplt.set_title('Insect outbreak system (cod-1 fold)', fontsize=40, fontweight='normal', fontfamily='Arial', y=1.03)
subplt.legend(handles=[legend_fold_fold,legend_fold_NA,legend_fold_blow],labels=['fold','no bifurcation','divergence'],loc='center left', bbox_to_anchor=(0.6, 0.8), markerscale=2,prop={'size':24})

###################################################

subplt = ax2

legend_trans_trans = subplt.scatter(t_trans_trans,bif_trans_trans,s=100,marker=r'$\checkmark$',c='forestgreen',zorder=6)
legend_trans_hopf = subplt.scatter(t_trans_hopf,bif_trans_hopf,s=100,marker=r'$\checkmark$',c='royalblue',zorder=6)
legend_trans_fold = subplt.scatter(t_trans_fold,bif_trans_fold,s=80,marker=r'$\times$',c='crimson',zorder=5)
legend_trans_NA = subplt.scatter(t_trans_NA,bif_trans_NA,s=80,marker=r'$\times$',c='grey',zorder=5)
legend_trans_blow = subplt.scatter(t_trans_blow,bif_trans_blow,s=80,marker='^',c='chocolate',zorder=5)
subplt.text(-0.125, 1,'b',ha='left', transform=subplt.transAxes,fontdict={'family':'DejaVu Sans','size':50,'weight':'bold'})
subplt.set_xlabel('Timepoint',font_x,labelpad=-20)
subplt.set_ylim(0,19)
subplt.set_xticks([500,1000])
subplt.set_yticks([trans_bif,hopf2_bif])
subplt.axhline(y=2, color='black', linestyle=':', linewidth=2)
subplt.axhline(y=trans_bif, color='forestgreen', linestyle='--', linewidth=2)
subplt.axhline(y=hopf2_bif, color='royalblue', linestyle='--', linewidth=2)
subplt.text(750,0.8,'divergence or no bifurcation', fontsize=24, ha='left', va='center', color='black')
subplt.set_yticklabels(['5.88\ntranscritical','15.69\nHopf'])

subplt.set_title('Predator-prey system (cod-1 transcritical / Hopf)', fontsize=36, fontweight='normal', fontfamily='Arial', y=1.03)
subplt.legend(handles=[legend_trans_trans,legend_trans_hopf,legend_trans_fold,legend_trans_NA,legend_trans_blow],labels=['transcritical','Hopf','fold','no bifurcation','divergence'],
              loc='center left', bbox_to_anchor=(0.3, 0.6),markerscale=2,prop={'size':24})

###################################################

subplt = ax3

legend_hopf_hopf = subplt.scatter(t_hopf_hopf,bif_hopf_hopf,s=100,marker=r'$\checkmark$',c='royalblue',zorder=6)
legend_hopf_fold = subplt.scatter(t_hopf_fold,bif_hopf_fold,s=80,marker=r'$\times$',c='crimson',zorder=5)
legend_hopf_NA = subplt.scatter(t_hopf_NA,bif_hopf_NA,s=80,marker=r'$\times$',c='grey',zorder=5)
# legend_hopf_blow = subplt.scatter(t_hopf_blow,bif_hopf_blow,s=80,marker='^',c='chocolate',zorder=5)
subplt.text(-0.125, 1,'c',ha='left', transform=subplt.transAxes,fontdict={'family':'DejaVu Sans','size':50,'weight':'bold'})
subplt.set_xlabel('Timepoint',font_x,labelpad=-20)
subplt.set_ylim(-0.1,0.85)
subplt.set_xticks([500,1000])
subplt.set_yticks([0.1,hopf_bif])
subplt.axhline(y=0, color='black', linestyle=':', linewidth=2)
subplt.axhline(y=hopf_bif, color='royalblue', linestyle='--', linewidth=2)
subplt.text(850,-0.05,'no bifurcation', fontsize=24, ha='left', va='center', color='black')
subplt.set_yticklabels(['(0.1,1.1)','(0.55,0.65)\nHopf'])

subplt.set_title('Epidemiological system (cod-2 Hopf)', fontsize=40, fontweight='normal', fontfamily='Arial', y=1.03)
subplt.legend(handles=[legend_hopf_hopf,legend_hopf_fold,legend_hopf_NA],labels=['Hopf','fold','no bifurcation'],loc='center left', bbox_to_anchor=(0.05, 0.3),markerscale=2,prop={'size':24})


plt.subplots_adjust(top=0.9, bottom=0.1, left=0.05, right=0.99, wspace=0.25)
plt.savefig('../figures/FIG5.pdf',format='pdf',dpi=600)



