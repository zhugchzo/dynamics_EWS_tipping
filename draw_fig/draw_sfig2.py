import pandas
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.lines as mlines

font_x = {'family':'Arial','weight':'normal','size': 40}
font_y = {'family':'Arial','weight':'normal','size': 40}
font_in_x = {'family':'Arial','weight':'normal','size': 36}
font_in_y = {'family':'Arial','weight':'normal','size': 36}

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 36
plt.rcParams['ytick.labelsize'] = 36

fig, axs = plt.subplots(2, 1, figsize=(12,15))

(ax1, ax2) = axs.flatten()

df_fold = pandas.read_csv('../data/fold_network/fold_network.csv')
df_results = pandas.read_csv('../results/fold_network/global_fold_network.csv')

t1 = df_fold['Time'] + 1
t2 = df_results['Time'] + 1

x = df_fold['traj_x']
pred_x = df_results['pred_x']

rrate = df_results['real_eval']
esti_real = df_results['esti_real_eval1']
esti_imag = df_results['esti_imag_eval1']

subplt = ax1

subplt.scatter(t1[::5],x[::5],s=25,c='darkslateblue',marker='o')
subplt.plot(t1[::5],x[::5],c='darkslateblue')
subplt.scatter(t2[::5]+1,pred_x[::5],s=100, marker='o',facecolors='none',edgecolors='crimson')
subplt.scatter(1000,x[1000],s=500, marker='o',facecolors='white', edgecolors='black',zorder=5)
subplt.set_ylabel('system state',font_y,labelpad=-25)
subplt.set_xlim(-55,1100)
subplt.set_xticks([0,1000])
subplt.set_ylim(6,13.5)
subplt.set_yticks([7,12])
subplt.set_yticklabels(['7','12'])
subplt.text(820,9.5,'fold',size=48)
subplt.tick_params(axis='x', labelbottom=False)
subplt.text(-0.075, 0.96,'a',ha='left', transform=subplt.transAxes,fontdict={'family':'DejaVu Sans','size':50,'weight':'bold'})
subplt.axvline(x=0, color='black', linestyle='--', linewidth=2)
subplt.axvline(x=500, color='black', linestyle='--', linewidth=2)

subplt.plot([15, 484], [11.5, 11.5], linestyle='--', color='black', linewidth=2)
arrow1 = FancyArrowPatch((5, 11.5), (494, 11.5), 
                        arrowstyle='<|-|>', 
                        color='black',
                        linewidth = 0, 
                        mutation_scale=25)
subplt.add_patch(arrow1)

subplt.text(240,11,'sliding window', fontsize=36, ha='center', va='center', color='black')
arrow2 = FancyArrowPatch((430, 11), (470, 11), 
                        arrowstyle='->', 
                        color='black',
                        linewidth=2,
                        mutation_scale=15)
subplt.add_patch(arrow2)

legend_state = mlines.Line2D([], [], color='darkslateblue', marker='o', markersize=5, linestyle='-', markeredgewidth=1)
legend_pstate = mlines.Line2D([], [], markerfacecolor='none',color='crimson', marker='o', markersize=10, linestyle='None', markeredgewidth=3)

subplt.legend(handles=[legend_state,legend_pstate],labels=['mean state','mean predicted\n     state'],loc='lower center', frameon=False, bbox_to_anchor=(0.7, 0.05), handletextpad=0.2, markerscale=2.5,prop={'size':36})

subplt = ax2

subplt.scatter(t2,rrate,s=20,c='darkslateblue',marker='^')
subplt.scatter(t2,esti_real,s=20,c='crimson',marker='^')
subplt.scatter(1000,rrate[499],s=500,marker='^',facecolors='white', edgecolors='black')
subplt.text(680,-0.02,'reach zero',size=40)
subplt.set_xlabel('Timepoint',font_x,labelpad=-20)
subplt.set_ylabel('Re',font_y,labelpad=-50)
subplt.set_xlim(-55,1100)
subplt.set_xticks([0,1000])
subplt.set_ylim(-0.6,0.05)
subplt.set_yticks([-0.5,0])
subplt.set_yticklabels(['-0.5','0'])
subplt.text(-0.075, 0.96,'b',ha='left', transform=subplt.transAxes,fontdict={'family':'DejaVu Sans','size':50,'weight':'bold'})
subplt.axvline(x=500, color='black', linestyle='--', linewidth=2)

ax_inset = fig.add_axes([0.12, 0.17, 0.33, 0.22])

ax_inset.scatter(t2[::5],esti_imag[::5], s=80, marker='^',facecolors='none',edgecolors='crimson')
ax_inset.axhline(y=0, color='black', linestyle='--')
ax_inset.set_xlabel('Timepoint',font_in_x,labelpad=-5)
ax_inset.set_ylabel('Im',font_in_y)
ax_inset.set_xticks([500,1000])
ax_inset.set_xticklabels([500,1000], fontsize=32)
ax_inset.set_yticks([0])
ax_inset.set_yticklabels([0], fontsize=32)
ax_inset.set_ylim(-1.1,1.1)
ax_inset.yaxis.tick_right()
ax_inset.yaxis.set_label_position("right") 
ax_inset.yaxis.set_label_coords(1.1, 0.96)
ax_inset.yaxis.label.set_rotation(360)
inset_legend_pred = mlines.Line2D([], [],  markerfacecolor='none',color='crimson', marker='^', markersize=5, linestyle='None', markeredgewidth=3)
ax_inset.legend([inset_legend_pred],['estimated Im'],fontsize=36,frameon=False,loc='upper center', bbox_to_anchor=(0.5,0.3),handletextpad=0,markerscale=3)

legend_gt = mlines.Line2D([], [], color='darkslateblue', marker='^', markersize=10, linestyle='None')
legend_pred = mlines.Line2D([], [], color='crimson', marker='^', markersize=10, linestyle='None')

subplt.legend(handles=[legend_gt,legend_pred],labels=['ground truth','estimated Re'],loc='upper center', frameon=False, bbox_to_anchor=(0.8,0.3), handletextpad=0, markerscale=2.5,prop={'size':36})

plt.subplots_adjust(top=0.98, bottom=0.06, left=0.08, right=0.99, hspace=0.05)
plt.savefig('../figures/SFIG2.pdf',format='pdf',dpi=600)