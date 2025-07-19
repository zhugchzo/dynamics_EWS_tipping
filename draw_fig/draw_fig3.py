import pandas
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

font_x = {'family':'Arial','weight':'normal','size': 40}
font_y = {'family':'Arial','weight':'normal','size': 40}
font_in_x1 = {'family':'Arial','weight':'normal','size': 32}
font_in_y1 = {'family':'Arial','weight':'normal','size': 32}
font_in_x2 = {'family':'Arial','weight':'normal','size': 24}
font_in_y2 = {'family':'Arial','weight':'normal','size': 24}

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 36
plt.rcParams['ytick.labelsize'] = 36

fig, axs = plt.subplots(2, 3, figsize=(36,15))

(ax1, ax3, ax5, ax2, ax4, ax6) = axs.flatten()

df_sup_hopf = pandas.read_csv('../data/sup_hopf/sup_hopf.csv')
df_sup_hopf_results = pandas.read_csv('../results/sup_hopf/sup_hopf.csv')
df_sup_hopf_stable = pandas.read_csv('../results/sup_hopf/sup_hopf_stable.csv')

df_trans = pandas.read_csv('../data/trans/trans.csv')
df_trans_results = pandas.read_csv('../results/trans/trans.csv')

df_sub_hopf = pandas.read_csv('../data/sub_hopf/sub_hopf.csv')
df_sub_hopf_results = pandas.read_csv('../results/sub_hopf/sub_hopf.csv')
df_sub_hopf_stable = pandas.read_csv('../results/sub_hopf/sub_hopf_stable.csv')

t1 = df_sup_hopf['Time'] + 1
t2 = df_sup_hopf_results['Time'] + 1
t3 = df_sup_hopf_stable['Time'] + 1

t4 = df_trans['Time'][:-10] + 1
t5 = df_trans_results['Time'] + 1

t6 = df_sub_hopf['Time'] + 1
t7 = df_sub_hopf_results['Time'] + 1
t8 = df_sub_hopf_stable['Time'] + 1

sup_hopf_x = df_sup_hopf['traj_x']
sup_hopf_pred_x = df_sup_hopf_results['pred_x']

trans_x = df_trans['traj_x'][:-10]
trans_pred_x = df_trans_results['pred_x']

sub_hopf_x = df_sub_hopf['traj_x']
sub_hopf_pred_x = df_sub_hopf_results['pred_x']

sup_hopf_real = df_sup_hopf_results['real_eval']
sup_hopf_esti_real = df_sup_hopf_results['esti_real_eval1']
sup_hopf_imag = df_sup_hopf_results['imag_eval']
sup_hopf_esti_imag1 = df_sup_hopf_results['esti_imag_eval1']
sup_hopf_esti_imag2 = df_sup_hopf_results['esti_imag_eval2']
sup_hopf_esti_stbexpnt = df_sup_hopf_stable['stable_expnt']

trans_real = df_trans_results['real_eval']
trans_esti_real = df_trans_results['esti_real_eval1']
trans_imag = df_trans_results['imag_eval']
trans_esti_imag1 = df_trans_results['esti_imag_eval1']

sub_hopf_real = df_sub_hopf_results['real_eval']
sub_hopf_esti_real = df_sub_hopf_results['esti_real_eval1']
sub_hopf_imag = df_sub_hopf_results['imag_eval']
sub_hopf_esti_imag1 = df_sub_hopf_results['esti_imag_eval1']
sub_hopf_esti_imag2 = df_sub_hopf_results['esti_imag_eval2']
sub_hopf_esti_stbexpnt = df_sub_hopf_stable['stable_expnt']

subplt = ax1

subplt.scatter(t1[::5],sup_hopf_x[::5],s=25,c='darkslateblue',marker='o')
subplt.plot(t1[::5],sup_hopf_x[::5],c='darkslateblue')
subplt.scatter(t2[::5]+1,sup_hopf_pred_x[::5],s=100, marker='o',facecolors='none',edgecolors='crimson')
subplt.scatter(1000,sup_hopf_x[1000],s=500, marker='o',facecolors='white', edgecolors='black',zorder=5)
subplt.set_ylabel('system state',font_y,labelpad=-30)
subplt.set_xlim(-65,1326)
subplt.set_ylim(0.5,2.2)
subplt.text(650,0.7,'supercritical Hopf',size=36)
subplt.text(-0.075, 0.96,'a',ha='left', transform=subplt.transAxes,fontdict={'family':'DejaVu Sans','size':50,'weight':'bold'})
subplt.axvline(x=0, color='black', linestyle='--', linewidth=2)
subplt.axvline(x=500, color='black', linestyle='--', linewidth=2)
subplt.set_xticks([0,500,1000])
subplt.tick_params(axis='x', labelbottom=False)
subplt.set_yticks([0.8, 2])
subplt.set_yticklabels(['0.8','2'])

subplt.plot([15, 484], [0.9, 0.9], linestyle='--', color='black', linewidth=2)
arrow1 = FancyArrowPatch((5, 0.9), (494, 0.9), 
                        arrowstyle='<|-|>', 
                        color='black',
                        linewidth = 0, 
                        mutation_scale=25)
subplt.add_patch(arrow1)

subplt.text(240,0.8,'sliding window', fontsize=36, ha='center', va='center', color='black')
arrow2 = FancyArrowPatch((450, 0.8), (490, 0.8), 
                        arrowstyle='->', 
                        color='black',
                        linewidth=2,
                        mutation_scale=15)
subplt.add_patch(arrow2)

legend_state = mlines.Line2D([], [], color='darkslateblue', marker='o', markersize=5, linestyle='-', markeredgewidth=1)
legend_pstate = mlines.Line2D([], [], markerfacecolor='none',color='crimson', marker='o', markersize=10, linestyle='None', markeredgewidth=3)

ax_inset = fig.add_axes([0.16, 0.775, 0.145, 0.2]) 

ax_inset.scatter(t3, sup_hopf_esti_stbexpnt, color='crimson', marker='x',s=75)
ax_inset.axhline(y=0, color='black', linestyle='--')
ax_inset.set_xlabel('Timepoint',font_in_x1,labelpad=-10)
ax_inset.set_xticks([971,1000])
ax_inset.set_xticklabels([971,1000],fontsize=24)
ax_inset.set_yticks([-0.2,-0.1,0])
ax_inset.set_yticklabels(['-0.2','-0.1','0'],fontsize=24)
ax_inset.yaxis.tick_right()
ax_inset.set_ylim(-0.23,0.046)
legend_index = mlines.Line2D([], [], color='crimson', marker='x', linestyle='None',markeredgewidth=2)
ax_inset.legend(handles=[legend_index],labels=['estimated stability index'],fontsize=30,loc='center', bbox_to_anchor=(0.5,0.91), frameon=False,handletextpad=-0.1,markerscale=3)

subplt.legend(handles=[legend_state,legend_pstate],labels=['state (N)','predicted state'],loc='upper center', frameon=False, bbox_to_anchor=(0.22, 0.8), handletextpad=0.2,markerscale=2.5,prop={'size':32})

subplt = ax2

subplt.scatter(t2,sup_hopf_real,s=20,c='darkslateblue',marker='^')
subplt.scatter(t2[df_sup_hopf_results['esti_real_eval1'].between(-0.065, 0.002)],sup_hopf_esti_real[df_sup_hopf_results['esti_real_eval1'].between(-0.065, 0.002)],s=20,c='crimson',marker='^')
subplt.scatter(1000,sup_hopf_real[499],s=500,marker='^',facecolors='white', edgecolors='black')
subplt.text(1020,0,'reach zero',size=36)
subplt.set_xlabel('Timepoint',font_x,labelpad=-20)
subplt.set_ylabel('Re',font_y,labelpad=-70)
subplt.text(-0.075, 0.96,'d',ha='left', transform=subplt.transAxes,fontdict={'family':'DejaVu Sans','size':50,'weight':'bold'})
subplt.axvline(x=500, color='black', linestyle='--', linewidth=2)
subplt.set_xlim(-65,1326)
subplt.set_ylim(-0.066,0.006)
subplt.set_xticks([0,1000])
subplt.set_yticks([-0.06,0])
subplt.set_yticklabels(['-0.06','0'])

ax_inset = fig.add_axes([0.041, 0.17, 0.095, 0.22]) 

ax_inset.scatter(t2[::5],sup_hopf_esti_imag1[::5], s=80, marker='^',facecolors='none',edgecolors='crimson',zorder=2)
ax_inset.scatter(t2[::5],sup_hopf_esti_imag2[::5], s=80, marker='^',facecolors='none',edgecolors='crimson',zorder=2)
ax_inset.scatter(t2,sup_hopf_imag,s=20, marker='^',c='darkslateblue',label = 'ground truth',zorder=1)
ax_inset.scatter(t2,-sup_hopf_imag,s=20, marker='^',c='darkslateblue',label = 'ground truth',zorder=1)
ax_inset.axhline(y=0, color='black', linestyle='--')
ax_inset.set_xlabel('Timepoint',font_in_x2,labelpad=-20)
ax_inset.set_ylabel('Im',font_in_y2)
ax_inset.set_xticks([500,1000])
ax_inset.set_xticklabels([500,1000],fontsize=24)
ax_inset.set_yticks([-0.2,0,0.2])
ax_inset.set_yticklabels([-0.2,0,0.2],fontsize=24)
ax_inset.set_ylim(-0.4,0.4)
ax_inset.yaxis.tick_right()
ax_inset.yaxis.set_label_position("right") 
ax_inset.yaxis.set_label_coords(1.1, 0.96)
ax_inset.yaxis.label.set_rotation(360)
inset_legend_gt = mlines.Line2D([], [], color='darkslateblue', marker='^', markersize=5, linestyle='None')
inset_legend_pred = mlines.Line2D([], [],  markerfacecolor='none',color='crimson', marker='^', markersize=5, linestyle='None', markeredgewidth=3)

legend1 = ax_inset.legend([inset_legend_gt],['ground truth'],fontsize=30,frameon=False,loc='upper center', bbox_to_anchor=(0.5,1),handletextpad=0,markerscale=3)
ax_inset.add_artist(legend1)
ax_inset.legend([inset_legend_pred],['estimated Im'],fontsize=30,frameon=False,loc='upper center', bbox_to_anchor=(0.5,0.3),handletextpad=0,markerscale=3)

legend_gt = mlines.Line2D([], [], color='darkslateblue', marker='^', markersize=10, linestyle='None')
legend_pred = mlines.Line2D([], [], color='crimson', marker='^', markersize=10, linestyle='None')

subplt.legend(handles=[legend_gt,legend_pred],labels=['ground truth','estimated Re'],loc='upper center', frameon=False, bbox_to_anchor=(0.8,0.275),handletextpad=0,markerscale=2.5,prop={'size':36})

subplt = ax3

subplt.scatter(t4[::5],trans_x[::5],s=25,c='darkslateblue',marker='o')
subplt.plot(t4[::5],trans_x[::5],c='darkslateblue')
subplt.scatter(t5[::5]+1,trans_pred_x[::5],s=100, marker='o',facecolors='none',edgecolors='crimson')
subplt.scatter(1000,trans_x[1000],s=500, marker='o',facecolors='white', edgecolors='black',zorder=5)
subplt.set_xlim(-57,1140)
subplt.set_ylim(1.675,1.8)
subplt.set_yticks([1.7,1.75])
subplt.set_yticklabels(['1.7','1.75'])
subplt.text(680,1.68,'transcritical',size=36)
subplt.text(-0.075, 0.96,'b',ha='left', transform=subplt.transAxes,fontdict={'family':'DejaVu Sans','size':50,'weight':'bold'})
subplt.axvline(x=0, color='black', linestyle='--', linewidth=2)
subplt.axvline(x=500, color='black', linestyle='--', linewidth=2)
subplt.set_xticks([0,500,1000])
subplt.tick_params(axis='x', labelbottom=False)

subplt.plot([15, 484], [1.72, 1.72], linestyle='--', color='black', linewidth=2)
arrow1 = FancyArrowPatch((5, 1.72), (494, 1.72), 
                        arrowstyle='<|-|>', 
                        color='black',
                        linewidth = 0, 
                        mutation_scale=25)
subplt.add_patch(arrow1)

subplt.text(240,1.73,'sliding window', fontsize=36, ha='center', va='center', color='black')
arrow2 = FancyArrowPatch((440, 1.73), (490, 1.73), 
                        arrowstyle='->', 
                        color='black',
                        linewidth=2,
                        mutation_scale=15)
subplt.add_patch(arrow2)

legend_state = mlines.Line2D([], [], color='darkslateblue', marker='o', markersize=5, linestyle='-', markeredgewidth=1)
legend_pstate = mlines.Line2D([], [], markerfacecolor='none',color='crimson', marker='o', markersize=10, linestyle='None', markeredgewidth=3)

subplt.legend(handles=[legend_state,legend_pstate],labels=['state (N)','predicted state'],loc='upper center', frameon=False, bbox_to_anchor=(0.75, 0.8), handletextpad=0.2,markerscale=2.5,prop={'size':32})

subplt = ax4

subplt.scatter(t5,trans_real,s=20,c='darkslateblue',marker='^')
subplt.scatter(t5,trans_esti_real,s=20,c='crimson',marker='^')
subplt.scatter(1000,trans_real[499],s=500,marker='^',facecolors='white', edgecolors='black')
subplt.text(650,0.01,'reach zero',size=36)
subplt.set_xlabel('Timepoint',font_x,labelpad=-20)
subplt.set_xlim(-57,1140)
subplt.text(-0.075, 0.96,'e',ha='left', transform=subplt.transAxes,fontdict={'family':'DejaVu Sans','size':50,'weight':'bold'})
subplt.axvline(x=500, color='black', linestyle='--', linewidth=2)
subplt.set_xticks([0,1000])
subplt.set_yticks([-0.5,0])
subplt.set_yticklabels(['-0.5','0'])

ax_inset = fig.add_axes([0.375, 0.17, 0.11, 0.22])

ax_inset.scatter(t5[::5],trans_esti_imag1[::5], s=80, marker='^',facecolors='none',edgecolors='crimson')
ax_inset.axhline(y=0, color='black', linestyle='--')
ax_inset.set_xlabel('Timepoint',font_in_x2,labelpad=-20)
ax_inset.set_ylabel('Im',font_in_y2)
ax_inset.set_xticks([500,1000])
ax_inset.set_xticklabels([500,1000],fontsize=24)
ax_inset.set_yticks([0])
ax_inset.set_yticklabels([0],fontsize=24)
ax_inset.set_ylim(-1.1,1.1)
ax_inset.yaxis.tick_right()
ax_inset.yaxis.set_label_position("right") 
ax_inset.yaxis.set_label_coords(1.08, 0.96)
ax_inset.yaxis.label.set_rotation(360)
inset_legend_pred = mlines.Line2D([], [],  markerfacecolor='none',color='crimson', marker='^', markersize=5, linestyle='None', markeredgewidth=3)
ax_inset.legend([inset_legend_pred],['estimated Im'],fontsize=30,frameon=False,loc='upper center', bbox_to_anchor=(0.5,0.3),handletextpad=0,markerscale=3)

legend_gt = mlines.Line2D([], [], color='darkslateblue', marker='^', markersize=10, linestyle='None')
legend_pred = mlines.Line2D([], [], color='crimson', marker='^', markersize=10, linestyle='None')

subplt.legend(handles=[legend_gt,legend_pred],labels=['ground truth','estimated Re'],loc='upper center', frameon=False, bbox_to_anchor=(0.8,0.275),handletextpad=0,markerscale=2.5,prop={'size':36})

subplt = ax5

subplt.scatter(t6[::5],sub_hopf_x[::5],s=25,c='darkslateblue',marker='o')
subplt.plot(t6[::5],sub_hopf_x[::5],c='darkslateblue')
subplt.scatter(t7[::5]+1,sub_hopf_pred_x[::5],s=100, marker='o',facecolors='none',edgecolors='crimson')
subplt.scatter(1000,sub_hopf_x[1000],s=500, marker='o',facecolors='white', edgecolors='black',zorder=5)
subplt.set_xlim(-62,1265)
subplt.set_ylim(0.2,8)
subplt.text(750,1.4,'subcritical Hopf',size=36)
subplt.text(-0.075, 0.96,'c',ha='left', transform=subplt.transAxes,fontdict={'family':'DejaVu Sans','size':50,'weight':'bold'})
subplt.axvline(x=0, color='black', linestyle='--', linewidth=2)
subplt.axvline(x=500, color='black', linestyle='--', linewidth=2)
subplt.set_xticks([0,500,1000])
subplt.set_yticks([1,7])
subplt.tick_params(axis='x', labelbottom=False)

subplt.plot([15, 484], [2, 2], linestyle='--', color='black', linewidth=2)
arrow1 = FancyArrowPatch((5, 2), (494, 2), 
                        arrowstyle='<|-|>', 
                        color='black',
                        linewidth = 0, 
                        mutation_scale=25)
subplt.add_patch(arrow1)

subplt.text(240,2.5,'sliding window', fontsize=36, ha='center', va='center', color='black')
arrow2 = FancyArrowPatch((440, 2.5), (490, 2.5), 
                        arrowstyle='->', 
                        color='black',
                        linewidth=2,
                        mutation_scale=15)
subplt.add_patch(arrow2)

ax_inset = fig.add_axes([0.822, 0.775, 0.145, 0.2]) 

ax_inset.scatter(t8, sub_hopf_esti_stbexpnt, color='crimson', marker='x',s=75)
ax_inset.axhline(y=0, color='black', linestyle='--')
ax_inset.set_xlabel('Timepoint',font_in_x1,labelpad=-10)
ax_inset.set_xticks([971,1000])
ax_inset.set_xticklabels([971,1000], fontsize=24)
ax_inset.set_yticks([0,2,4])
ax_inset.set_yticklabels(['0','2','4'], fontsize=24)
ax_inset.yaxis.tick_right()
ax_inset.set_ylim(-0.92,4.6)
legend_index = mlines.Line2D([], [], color='crimson', marker='x', linestyle='None',markeredgewidth=2)
ax_inset.legend(handles=[legend_index],labels=['estimated stability index'],fontsize=30,loc='center',bbox_to_anchor=(0.5,0.09), frameon=False,handletextpad=-0.1,markerscale=3)

legend_state = mlines.Line2D([], [], color='darkslateblue', marker='o', markersize=5, linestyle='-', markeredgewidth=1)
legend_pstate = mlines.Line2D([], [], markerfacecolor='none',color='crimson', marker='o', markersize=10, linestyle='None', markeredgewidth=3)

subplt.legend(handles=[legend_state,legend_pstate],labels=['state (X)','predicted state'],loc='upper center', frameon=False, bbox_to_anchor=(0.23, 0.8), handletextpad=0.2,markerscale=2.5,prop={'size':32})

subplt = ax6

subplt.scatter(t7,sub_hopf_real,s=20,c='darkslateblue',marker='^')
subplt.scatter(t7[df_sub_hopf_results['esti_real_eval1'].between(-0.39, 0.02)],sub_hopf_esti_real[df_sub_hopf_results['esti_real_eval1'].between(-0.39, 0.02)],s=20,c='crimson',marker='^')
subplt.scatter(1000,sub_hopf_real[499],s=500,marker='^',facecolors='white', edgecolors='black')
subplt.text(650,0,'reach zero',size=36)
subplt.set_xlabel('Timepoint',font_x,labelpad=-20)
subplt.set_xlim(-62,1265)
subplt.text(-0.075, 0.96,'f',ha='left', transform=subplt.transAxes,fontdict={'family':'DejaVu Sans','size':50,'weight':'bold'})
subplt.axvline(x=500, color='black', linestyle='--', linewidth=2)
subplt.set_ylim(-0.45,0.05)
subplt.set_xticks([0,1000])
subplt.set_yticks([-0.4,0])
subplt.set_yticklabels(['-0.4','0'])

ax_inset = fig.add_axes([0.6975, 0.17, 0.1, 0.22]) 

ax_inset.scatter(t7[::5],sub_hopf_esti_imag1[::5], s=80, marker='^',facecolors='none',edgecolors='crimson',zorder=2)
ax_inset.scatter(t7[::5],sub_hopf_esti_imag2[::5], s=80, marker='^',facecolors='none',edgecolors='crimson',zorder=2)
ax_inset.scatter(t2,sub_hopf_imag,s=20, marker='^',c='darkslateblue',label = 'ground truth',zorder=1)
ax_inset.scatter(t2,-sub_hopf_imag,s=20, marker='^',c='darkslateblue',label = 'ground truth',zorder=1)
ax_inset.axhline(y=0, color='black', linestyle='--')
ax_inset.set_xlabel('Timepoint',font_in_x2,labelpad=-20)
ax_inset.set_ylabel('Im',font_in_y2)
ax_inset.set_xticks([500,1000])
ax_inset.set_xticklabels([500,1000],fontsize=24)
ax_inset.set_yticks([-0.8,0,0.8])
ax_inset.set_yticklabels([-0.8,0,0.8],fontsize=24)
ax_inset.set_ylim(-1.6,1.6)
ax_inset.yaxis.tick_right()
ax_inset.yaxis.set_label_position("right") 
ax_inset.yaxis.set_label_coords(1.1, 0.96)
ax_inset.yaxis.label.set_rotation(360)
inset_legend_gt = mlines.Line2D([], [], color='darkslateblue', marker='^', markersize=5, linestyle='None')
inset_legend_pred = mlines.Line2D([], [],  markerfacecolor='none',color='crimson', marker='^', markersize=5, linestyle='None', markeredgewidth=3)

legend1 = ax_inset.legend([inset_legend_gt],['ground truth'],fontsize=30,frameon=False,loc='upper center', bbox_to_anchor=(0.5,1.05),handletextpad=0,markerscale=3)
ax_inset.add_artist(legend1)
ax_inset.legend([inset_legend_pred],['estimated Im'],fontsize=30,frameon=False,loc='upper center', bbox_to_anchor=(0.5,0.25),handletextpad=0,markerscale=3)

legend_gt = mlines.Line2D([], [], color='darkslateblue', marker='^', markersize=10, linestyle='None')
legend_pred = mlines.Line2D([], [], color='crimson', marker='^', markersize=10, linestyle='None')

subplt.legend(handles=[legend_gt,legend_pred],labels=['ground truth','estimated Re'],loc='upper center',frameon=False, bbox_to_anchor=(0.8,0.275), handletextpad=0,markerscale=2.5,prop={'size':36})

plt.subplots_adjust(top=0.98, bottom=0.06, left=0.035, right=0.99, hspace=0.05, wspace=0.1)
plt.savefig('../figures/FIG3.pdf',format='pdf',dpi=600)
