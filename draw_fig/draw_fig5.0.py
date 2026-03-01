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

df_PM = pandas.read_csv('../data/fish_school/fish_school.csv')
df_results = pandas.read_csv('../results/fish_school/fish_school.csv')

t1 = df_PM['Time']
t2 = df_results['Time']

p = df_PM['P']
m = df_PM['M']

rrate = df_results['esti_real_eval1']

ax1.scatter(t1[::4],m[::4],s=40,c='darkorange',marker='o')
ax1.plot(t1[::4],m[::4],c='darkorange',linewidth=3)
ax1.set_ylabel('Order parameter',font_y,labelpad=10)

ax1.set_xlim(1,21)
ax1.set_xticks([20])
ax1.set_xticklabels([''])
ax1.set_ylim(-0.05,1.05)
ax1.set_yticks([0,1])
ax1.set_yticklabels(['0','1'])

ax1.text(2, -0.14, '2', ha='center', va='bottom', fontdict={'family': 'Arial', 'weight': 'normal', 'size': 36})
ax1.text(16, -0.14, '16', ha='center', va='bottom', fontdict={'family': 'Arial', 'weight': 'normal', 'size': 36})
ax1.text(20, -0.14, '20', ha='center', va='bottom', fontdict={'family': 'Arial', 'weight': 'normal', 'size': 36})

ax1.tick_params(direction='in', length=10)

ax1.scatter(t1[::4],p[::4],s=40,c='darkslateblue',marker='o')
ax1.plot(t1[::4],p[::4],c='darkslateblue',linewidth=3)

length = 357
train_length = 300

ax1.axvline(x=t1[0], color='black', linestyle='--', linewidth=2, zorder=1)
ax1.axvline(x=t1[300], color='black', linestyle='--', linewidth=2, zorder=1)

ax1.plot([t1[15], t1[284]], [1, 1], linestyle='--', color='black', linewidth=2)
arrow1 = FancyArrowPatch((t1[5], 1), (t1[294], 1), 
                        arrowstyle='<|-|>', 
                        color='black',
                        linewidth = 0, 
                        mutation_scale=30)
ax1.add_patch(arrow1)

ax1.text(t1[200],0.95,'sliding window', fontsize=36, ha='center', va='center', color='black')

# arrow2 = FancyArrowPatch((t1[265], 0.95), (t1[285], 0.95), 
#                         arrowstyle='->', 
#                         color='black',
#                         linewidth=2,
#                         mutation_scale=20)
# ax1.add_patch(arrow2)

ax1.axvspan(t1[train_length],t1[length-1], color='silver', linestyle='-', alpha=0.4, zorder=0)

ax1.text(-1, 0.9,'(a)',ha='left', transform=ax1.transAxes,fontdict={'family':'Arial','size':50,'weight':'normal'})
ax1.text(-0.18, 0.96,'(b)',ha='left', transform=ax1.transAxes,fontdict={'family':'Arial','size':50,'weight':'normal'})

ax1.text(-0.9, 0.8,'Experiment illustration',ha='left', transform=ax1.transAxes,fontdict={'family':'DejaVu Sans','style':'italic','size':35,'weight':'normal'})

legend_p = mlines.Line2D([], [], color='darkslateblue', marker='o', markersize=5, linestyle='-', linewidth=3)
legend_m = mlines.Line2D([], [], color='darkorange', marker='o', markersize=5, linestyle='-', linewidth=3)

ax1.legend(handles=[legend_p,legend_m],labels=[r'Polarization ($P$)',r'Rotation ($R$)'],loc='lower center', frameon=False, bbox_to_anchor=(0.4, 0.35), handlelength=1.2, handletextpad=0.2, markerscale=2.5,prop={'size':32})

ax2.scatter(t2,rrate,s=80,marker='o',linewidths=3,facecolors='none',edgecolors='crimson')
ax2.plot(t2,rrate,c='crimson',linewidth=3)
ax2.set_xlabel('Time (second)',font_x,labelpad=-20)
ax2.set_ylabel('Re',font_y,labelpad=-27)
ax2.set_xlim(15.8,18.4)
ax2.set_xticks([16,18])
ax2.set_xticklabels(['16','18'])
ax2.set_ylim(-0.05,1.4)
ax2.set_yticks([0,1.3])
ax2.set_yticklabels(['0','1.3'])

ax2.tick_params(direction='in', length=10, pad=10)

ax2.text(-0.18, 0.96,'(c)',ha='left', transform=ax2.transAxes,fontdict={'family':'Arial','size':50,'weight':'normal'})

legend_rrate = mlines.Line2D([], [], markerfacecolor='none',color='crimson', marker='o', markersize=8, linestyle='-', markeredgewidth=4, linewidth=4)

ax2.legend(handles=[legend_rrate],labels=['Estimated Re'],loc='upper center', frameon=False, bbox_to_anchor=(0.75,0.99),handlelength=1.5, handletextpad=0.2, markerscale=2.5,prop={'size':32})

plt.subplots_adjust(top=0.98, bottom=0.06, left=0.5, right=0.99, hspace=0.15)
plt.savefig('../figures/FIG5.0.pdf',format='pdf')