import pandas
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.lines as mlines

font_x = {'family':'Arial','weight':'normal','size': 40}
font_y = {'family':'Arial','weight':'normal','size': 40}

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['xtick.labelsize'] = 36
plt.rcParams['ytick.labelsize'] = 36

fig, axs = plt.subplots(2, 1, figsize=(18,15))

ax1, ax2 = axs[0], axs[1]

df_abundance = pandas.read_csv('../data/rotifers-algae/rotifers-algae_train.csv')
df_results = pandas.read_csv('../results/rotifers-algae/rotifers-algae.csv')
df_tau = pandas.read_csv('../results/rotifers-algae/rotifers-algae_tau.csv')

t1 = df_abundance['time (days)']
t2 = df_results['Time']
t3 = df_tau['Time']

predator = df_abundance['rotifers (animals/ml)']
prey = df_abundance['algae (10^6 cells/ml)']

rrate = df_results['esti_rrate']
tau = df_tau['tau']

train_length = 50
length = len(t1)

ax1.scatter(t1,predator,s=25,c='darkslateblue',marker='o', zorder=1)
ax1.plot(t1,predator,c='darkslateblue',linewidth=3, zorder=1)
ax1.set_ylabel('Rotifers (animals/ml)',font_y)
ax1.yaxis.set_label_coords(-0.1, 0.5)

ax1_right = ax1.twinx()

ax1_right.scatter(t1,prey,s=25,c='darkorange',marker='o', zorder=1)
ax1_right.plot(t1,prey,c='darkorange',linewidth=3, zorder=1)
ax1_right.set_ylabel('Algae ($10^6$ cells/ml)',font_y)
ax1_right.yaxis.set_label_coords(1.07, 0.5)

ax1.set_xlim(80,201)
ax1.set_xticks([197])
ax1.set_xticklabels([''])

ax1.set_ylim(-2,32)
ax1.set_yticks([0,15,30])
ax1.set_yticklabels(['0','15','30'])
ax1.tick_params(axis='x', pad=5)
ax1.tick_params(direction='in', length=10)
ax1.minorticks_on()
ax1.tick_params(axis='x', which='minor', bottom=False)
ax1.tick_params(axis='y', which='minor', direction='in', length=5)

ax1.text(84, -5, '84', ha='center', va='bottom', fontdict={'family': 'Arial', 'weight': 'normal', 'size': 36})
ax1.text(133, -5, '133', ha='center', va='bottom', fontdict={'family': 'Arial', 'weight': 'normal', 'size': 36})
ax1.text(197, -5, '197', ha='center', va='bottom', fontdict={'family': 'Arial', 'weight': 'normal', 'size': 36})

ax1_right.set_ylim(-0.2,2.2)
ax1_right.set_yticks([0,1,2])
ax1_right.set_yticklabels(['0','1','2'])
ax1_right.tick_params(direction='in', length=10)
ax1_right.minorticks_on()
ax1_right.tick_params(axis='y', which='minor', direction='in', length=5)

ax1_right.axvspan(t1[train_length-1],t1[length-1], color='silver', linestyle='-', alpha=0.4, zorder=0)

ax1.set_zorder(ax1_right.get_zorder() + 1)
ax1.patch.set_visible(False)

ax1.axvline(x=t1[0], color='black', linestyle='--', linewidth=2, alpha=0.5, zorder=0)
ax1.axvline(x=t1[train_length-1], color='black', linestyle='--', linewidth=2, alpha=0.5, zorder=0)

ax1.text(-0.85, 0.9,'(a)',ha='left', transform=ax1.transAxes,fontdict={'family':'Arial','size':50,'weight':'normal'})
ax1.text(-0.2, 0.95,'(b)',ha='left', transform=ax1.transAxes,fontdict={'family':'Arial','size':50,'weight':'normal'})

ax1.text(-0.8, 0.75,'Experiment illustration',ha='left', transform=ax1.transAxes,fontdict={'family':'DejaVu Sans','style':'italic','size':30,'weight':'normal'})

ax1.plot([t1[2], t1[47]], [0, 0], linestyle='--', color='black', linewidth=2)
arrow1 = FancyArrowPatch((t1[1], 0), (t1[48], 0), 
                        arrowstyle='<|-|>', 
                        color='black',
                        linewidth = 0, 
                        mutation_scale=30)
ax1.add_patch(arrow1)

ax1.text(t1[25],2,r'$L_1$', fontsize=40, ha='center', va='center', color='black')

legend_predator = mlines.Line2D([], [], color='darkslateblue', marker='o', markersize=6, linestyle='-', linewidth=5)
legend_prey = mlines.Line2D([], [], color='darkorange', marker='o', markersize=6, linestyle='-', linewidth=5)

ax1.legend(handles=[legend_predator,legend_prey],labels=['Rotifers','Algae'],loc='lower center', frameon=False, bbox_to_anchor=(0.8, 0.73), handlelength=1.5,handletextpad=0.2,markerscale=2.5,prop={'size':36})


ax2.scatter(t2,rrate,s=150,marker='o',linewidths=3,facecolors='none',edgecolors='crimson',zorder=5)
ax2.plot(t2,rrate,c='crimson',linewidth=3,zorder=5)
ax2.set_xlabel('Time (day)',font_x,labelpad=-20)
ax2.set_ylabel('Value',font_y)
ax2.yaxis.set_label_coords(-0.1, 0.5)

ax2.plot(t3,tau,c='royalblue',linewidth=5,linestyle='--',alpha=0.9)

ax2.set_xlim(131,199)
ax2.set_xticks([133,197])
ax2.set_xticklabels(['133','197'])
ax2.set_ylim(-0.9,0.5)
ax2.set_yticks([-0.8,0,0.4])
ax2.set_yticklabels(['-0.8','0','0.4'])
ax2.tick_params(axis='x', pad=10)
ax2.tick_params(direction='in', length=10)
ax2.minorticks_on()
ax2.tick_params(axis='x', which='minor', bottom=False)
ax2.tick_params(axis='y', which='minor', direction='in', length=5)

ax2.hlines(y=0, xmin=131, xmax=199, color='grey', linestyle='--', linewidth=2, alpha=0.5, zorder=0)

ax2.plot([t2[1], t2[8]], [tau[0], tau[0]], linestyle='-', color='black', linewidth=2)
arrow2 = FancyArrowPatch((t2[0], tau[0]), (t2[9], tau[0]), 
                        arrowstyle='<|-|>', 
                        color='black',
                        linewidth=0, 
                        mutation_scale=30)
ax2.add_patch(arrow2)

ax2.text(t2[5],tau[0]+0.1,r'$L_2$', fontsize=40, ha='center', va='center', color='black')

ax2.text(-0.2, 0.95,'(c)',ha='left', transform=ax2.transAxes,fontdict={'family':'Arial','size':50,'weight':'normal'})

legend_rrate = mlines.Line2D([], [], markerfacecolor='none',color='crimson', marker='o', markersize=8, linestyle='-', markeredgewidth=4, linewidth=4)
legend_tau = mlines.Line2D([], [], color='royalblue', linestyle='--',linewidth=6,alpha=0.9)

ax2.legend(handles=[legend_rrate,legend_tau],labels=['Estimated Re',r"Kendall's $\tau$"],loc='upper center', frameon=False, bbox_to_anchor=(0.7,1),handletextpad=0.2,markerscale=2.5,prop={'size':36})

plt.subplots_adjust(top=0.98, bottom=0.06, left=0.43, right=0.92, hspace=0.15)
plt.savefig('../figures/SFIG16.0.pdf',format='pdf')