import numpy as np
import pandas
import matplotlib.pyplot as plt

font_x = {'family':'Arial','weight':'normal','size': 40}
font_y = {'family':'Arial','weight':'normal','size': 40}

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 26
plt.rcParams['ytick.labelsize'] = 26

fig = plt.figure(figsize=(12, 8))

df_fold = pandas.read_csv('../results/microcosm/auto_output.csv')
df_time = pandas.read_csv('../results/microcosm/microcosm.csv')

t = df_time['Time']
t1 = df_fold['Time']

###################################################

fold_fold = df_fold[df_fold['bif_type'] == 'LP']
fold_NA = df_fold[df_fold['bif_type'].isna()]

t_fold_fold = fold_fold['Time']
bif_fold_fold = fold_fold['bif_value']

t_fold_NA = fold_NA['Time']
bif_fold_NA = np.zeros(len(t_fold_NA))+22

###################################################

fold_bif = 26.7

###################################################

legend_fold_fold = plt.scatter(t_fold_fold,bif_fold_fold,s=100,marker=r'$\checkmark$',c='crimson',zorder=5)
legend_fold_NA = plt.scatter(t_fold_NA,bif_fold_NA,s=80,marker=r'$\times$',c='grey',zorder=5)
plt.xlabel('Time (day)',font_x,labelpad=-15)
plt.xlim(24,25.5)
plt.xticks([24.25,25.25],labels=['24.25','25.25'])
plt.ylim(18,60)
plt.yticks([30,50],labels=['30','50'])
# plt.yticks([26.7,50],labels=['26.7\nfold','50'])
plt.axhline(y=22, color='black', linestyle=':', linewidth=2)
# plt.axhline(y=fold_bif, color='crimson', linestyle='--', linewidth=2)
plt.text(25,20,'no bifurcation', fontsize=24, ha='left', va='center', color='black')
plt.ylabel('Value of bifurcation (day)', font_y,labelpad=5)

plt.title('Cyanobacteria microcosm system (fold)', fontsize=40, fontweight='normal', fontfamily='Arial', y=1.03)
plt.legend(handles=[legend_fold_fold,legend_fold_NA],labels=['fold','no bifurcation'],loc='center left', bbox_to_anchor=(0.05, 0.85), markerscale=2,prop={'size':24})

###################################################

plt.subplots_adjust(top=0.9, bottom=0.1, left=0.125, right=0.99)
plt.savefig('../figures/SFIG3.pdf',format='pdf',dpi=600)



