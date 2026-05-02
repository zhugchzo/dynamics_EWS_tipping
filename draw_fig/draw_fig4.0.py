import pandas
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.lines as mlines

font_x = {'family':'Arial','weight':'normal','size': 40}
font_y1 = {'family':'Arial','weight':'normal','size': 30}
font_y2 = {'family':'Arial','weight':'normal','size': 40}

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['xtick.labelsize'] = 36
plt.rcParams['ytick.labelsize'] = 36

fig, axs = plt.subplots(2, 1, figsize=(18,15))

ax1, ax2 = axs[0], axs[1]

df_fold = pandas.read_csv('../data/microcosm/microcosm.csv')
df_results = pandas.read_csv('../results/microcosm/microcosm.csv')

t1 = df_fold['Time']
t2 = df_results['Time']

x = df_fold['light attenuation coefficient']

rrate = df_results['esti_rrate']

train_length = 500
length = 842

ax1.scatter(t1[::5],x[::5],s=25,c='darkslateblue',marker='o')
ax1.plot(t1[::5],x[::5],c='darkslateblue',linewidth=3)
ax1.scatter(t1[train_length:length][::5],x[train_length:length][::5],s=25,c='darkgrey',marker='o')
ax1.plot(t1[train_length:length][::5],x[train_length:length][::5],c='darkgrey',linewidth=3)
ax1.scatter(t1[1209],x[1209],s=500, marker='o',facecolors='white', edgecolors='black',linewidths=3,zorder=5)
ax1.set_ylabel('Light attenuation coefficient (m$^{-1}$)',font_y1)
ax1.yaxis.set_label_coords(-0.12, 0.5)
ax1.set_xlim(22,29)
ax1.set_xticks([28])
ax1.set_xticklabels([''])
ax1.set_ylim(90,155)
ax1.set_yticks([100,125,145])
ax1.set_yticklabels(['100','125','145'])
ax1.tick_params(axis='x', pad=5)
ax1.tick_params(direction='in', length=10)
ax1.minorticks_on()
ax1.tick_params(axis='x', which='minor', bottom=False)
ax1.tick_params(axis='y', which='minor', direction='in', length=5)
ax1.text(t1[1209]+0.1,x[1209]+2,'fold',size=40)
ax1.axvline(x=t1[0], color='black', linestyle='--', linewidth=2, alpha=0.5, zorder=0)
ax1.axvline(x=t1[500], color='black', linestyle='--', linewidth=2, alpha=0.5, zorder=0)

ax1.text(22.4, 84, '22.4', ha='center', va='bottom', fontdict={'family': 'Arial', 'weight': 'normal', 'size': 36})
ax1.text(24.2, 84, '24.2', ha='center', va='bottom', fontdict={'family': 'Arial', 'weight': 'normal', 'size': 36})
ax1.text(28, 84, '28', ha='center', va='bottom', fontdict={'family': 'Arial', 'weight': 'normal', 'size': 36})

ax1.text(-0.99, 0.7,'(a)',ha='left', transform=ax1.transAxes,fontdict={'family':'Arial','size':50,'weight':'normal'})
ax1.text(-0.11, 0.95,'(b)',ha='left', transform=ax1.transAxes,fontdict={'family':'Arial','size':50,'weight':'normal'})

ax1.text(-0.9, 0.57,'Experiment illustration',ha='left', transform=ax1.transAxes,fontdict={'family':'DejaVu Sans','style':'italic','size':35,'weight':'normal'})

ax1.text(26.3,150,'dilution event', fontsize=40, ha='center', va='center', color='black')
arrow1 = FancyArrowPatch((26, 147), (25.5, 143), 
                        arrowstyle='-|>', 
                        color='black',
                        linewidth=2,
                        mutation_scale=30)
ax1.add_patch(arrow1)

ax1.plot([t1[15], t1[484]], [100, 100], linestyle='--', color='black', linewidth=2)
arrow2 = FancyArrowPatch((t1[5], 100), (t1[494], 100), 
                        arrowstyle='<|-|>', 
                        color='black',
                        linewidth = 0, 
                        mutation_scale=30)
ax1.add_patch(arrow2)

ax1.text(t1[245],104,r'$L$', fontsize=40, ha='center', va='center', color='black')
# ax1.text(t1[245],108,'sliding\nwindow', fontsize=36, ha='center', va='center', color='black')
# arrow3 = FancyArrowPatch((t1[180], 102.5), (t1[290], 102.5), 
#                         arrowstyle='->', 
#                         color='black',
#                         linewidth=2,
#                         mutation_scale=20)
# ax1.add_patch(arrow3)

legend_state = mlines.Line2D([], [], color='darkslateblue', marker='o', markersize=6, linestyle='-', linewidth=5)
legend_pstate = mlines.Line2D([], [], color='darkgrey', marker='o', markersize=6, linestyle='-', linewidth=5)

ax1.legend(handles=[legend_state,legend_pstate],labels=['Population','Forecast regime'],loc='lower center', frameon=False, bbox_to_anchor=(0.65, 0.05), handlelength=1.5,handletextpad=0.2,markerscale=2.5,prop={'size':36})

ax2.scatter(t2,rrate,s=150,color='crimson',marker='o',edgecolor=None,alpha=0.2,zorder=5)
ax2.scatter(t2,rrate,s=150,marker='o',facecolors='none',edgecolors='crimson',zorder=5)
ax2.set_xlabel('Time (day)',font_x,labelpad=-20)
ax2.set_ylabel('Re',font_y2)
ax2.yaxis.set_label_coords(-0.12, 0.5)
ax2.set_xlim(24.1,25.45)
ax2.set_xticks([24.2,25.2])
ax2.set_xticklabels(['24.2','25.2'])
ax2.set_ylim(-0.95,-0.15)
ax2.set_yticks([-0.9,-0.6,-0.3])
ax2.set_yticklabels(['-0.9','-0.6','-0.3'])
ax2.tick_params(axis='x', pad=10)
ax2.tick_params(direction='in', length=10)
ax2.minorticks_on()
ax2.tick_params(axis='x', which='minor', bottom=False)
ax2.tick_params(axis='y', which='minor', direction='in', length=5)
ax2.text(-0.11, 0.95,'(c)',ha='left', transform=ax2.transAxes,fontdict={'family':'Arial','size':50,'weight':'normal'})

legend_pred = mlines.Line2D([], [], marker='o',color='crimson',markerfacecolor='pink',markersize=10,markeredgewidth=2,alpha=1,linestyle='None')

ax2.legend(handles=[legend_pred],labels=['Estimated Re'],loc='upper center', frameon=False, bbox_to_anchor=(0.7,0.3),handletextpad=0.2,markerscale=2.5,prop={'size':36})

plt.subplots_adjust(top=0.98, bottom=0.06, left=0.5, right=0.99, hspace=0.15)
plt.savefig('../figures/FIG4.0.pdf',format='pdf')