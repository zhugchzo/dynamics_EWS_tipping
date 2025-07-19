import pandas
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.lines as mlines

font_x = {'family':'Arial','weight':'normal','size': 40}
font_y1 = {'family':'Arial','weight':'normal','size': 30}
font_y2 = {'family':'Arial','weight':'normal','size': 40}

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 36
plt.rcParams['ytick.labelsize'] = 36

fig, axs = plt.subplots(2, 1, figsize=(12,15))

(ax1, ax2) = axs.flatten()

df_fold = pandas.read_csv('../data/microcosm/microcosm.csv')
df_results = pandas.read_csv('../results/microcosm/microcosm.csv')

t1 = df_fold['Time']
t2 = df_results['Time']

x = df_fold['light attenuation coefficient']
pred_x = df_results['pred_x']

rrate = df_results['esti_rrate']

subplt = ax1

subplt.scatter(t1[::5],x[::5],s=25,c='darkslateblue',marker='o')
subplt.plot(t1[::5],x[::5],c='darkslateblue')
subplt.scatter(t2[1::5],pred_x[::5],s=100, marker='o',facecolors='none',edgecolors='crimson')
subplt.scatter(t1[1209],x[1209],s=500, marker='o',facecolors='white', edgecolors='black',zorder=5)
subplt.set_ylabel('Light attenuation coefficient (m$^{-1}$)',font_y1,labelpad=10)
subplt.set_xlim(22,29)
subplt.set_xticks([22.4,24.2,28])
subplt.set_xticklabels(['22.4','24.2','28'])
subplt.set_ylim(90,155)
subplt.set_yticks([100,140])
subplt.set_yticklabels(['100','140'])
subplt.text(t1[1209]+0.2,x[1209]+3,'fold',size=48)
subplt.text(-0.075, 0.96,'a',ha='left', transform=subplt.transAxes,fontdict={'family':'DejaVu Sans','size':50,'weight':'bold'})
subplt.axvline(x=t1[0], color='black', linestyle='--', linewidth=2)
subplt.axvline(x=t1[500], color='black', linestyle='--', linewidth=2)

subplt.text(26.3,149,'dilution event', fontsize=24, ha='center', va='center', color='black')
arrow1 = FancyArrowPatch((26, 147), (25.5, 143), 
                        arrowstyle='-|>', 
                        color='black',
                        linewidth=2,
                        mutation_scale=25)
subplt.add_patch(arrow1)

subplt.plot([t1[15], t1[484]], [100, 100], linestyle='--', color='black', linewidth=2)
arrow2 = FancyArrowPatch((t1[5], 100), (t1[494], 100), 
                        arrowstyle='<|-|>', 
                        color='black',
                        linewidth = 0, 
                        mutation_scale=25)
subplt.add_patch(arrow2)

subplt.text(t1[240],105,'sliding window', fontsize=24, ha='center', va='center', color='black')
arrow3 = FancyArrowPatch((t1[190], 102.5), (t1[270], 102.5), 
                        arrowstyle='->', 
                        color='black',
                        linewidth=2,
                        mutation_scale=15)
subplt.add_patch(arrow3)

legend_state = mlines.Line2D([], [], color='darkslateblue', marker='o', markersize=5, linestyle='-', markeredgewidth=1)
legend_pstate = mlines.Line2D([], [], markerfacecolor='none',color='crimson', marker='o', markersize=10, linestyle='None', markeredgewidth=3)

subplt.legend(handles=[legend_state,legend_pstate],labels=['population','predicted population'],loc='lower center', frameon=False, bbox_to_anchor=(0.65, 0.05), handletextpad=0.2, markerscale=2.5,prop={'size':36})

subplt = ax2

subplt.scatter(t2,rrate,s=40,c='crimson',marker='^')
subplt.set_xlabel('Time (day)',font_x,labelpad=-20)
subplt.set_ylabel('Re',font_y2,labelpad=-10)
subplt.set_xlim(24,25.5)
subplt.set_xticks([24.25,25.25])
subplt.set_xticklabels(['24.25','25.25'])
subplt.set_ylim(-0.9,-0.1)
subplt.set_yticks([-0.8,-0.2])
subplt.set_yticklabels(['-0.8','-0.2'])
subplt.text(-0.075, 0.96,'b',ha='left', transform=subplt.transAxes,fontdict={'family':'DejaVu Sans','size':50,'weight':'bold'})

legend_pred = mlines.Line2D([], [], color='crimson', marker='^', markersize=10, linestyle='None')

subplt.legend(handles=[legend_pred],labels=['estimated Re'],loc='upper center', frameon=False, bbox_to_anchor=(0.7,0.3), handletextpad=0, markerscale=2.5,prop={'size':36})

plt.subplots_adjust(top=0.98, bottom=0.06, left=0.13, right=0.96, hspace=0.15)
plt.savefig('../figures/FIG4.pdf',format='pdf',dpi=600)