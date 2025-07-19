import pandas
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np

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

df = pandas.read_csv('../data/sub_hopf/sub_hopf.csv')

df_results = pandas.read_csv('../results/sub_hopf/sub_hopf.csv')
df_stable = pandas.read_csv('../results/sub_hopf/sub_hopf_stable.csv')

df_ac_results = pandas.read_csv('../results/sub_hopf/sub_hopf_ac.csv')
df_ac_stable = pandas.read_csv('../results/sub_hopf/sub_hopf_stable_ac.csv')

df_nop_results = pandas.read_csv('../results/sub_hopf/sub_hopf_nopars.csv')
df_nop_stable = pandas.read_csv('../results/sub_hopf/sub_hopf_stable_nopars.csv')

t1 = df['Time'] + 1
t2 = df_results['Time'] + 1
t3 = df_stable['Time'] + 1

x = df['traj_x']

real = df_results['real_eval']
imag = df_results['imag_eval']

nu_x = df_results['pred_x']
ac_x = df_ac_results['pred_x']
nop_x = df_nop_results['pred_x']

nu_real = df_results['esti_real_eval1']
nu_imag1 = df_results['esti_imag_eval1']
nu_imag2 = df_results['esti_imag_eval2']
nu_stbexpnt = df_stable['stable_expnt']

ac_real = df_ac_results['esti_real_eval1']
ac_imag1 = df_ac_results['esti_imag_eval1']
ac_imag2 = df_ac_results['esti_imag_eval2']
ac_stbexpnt = df_ac_stable['stable_expnt']

nop_real = df_nop_results['esti_real_eval1']
nop_imag1 = df_nop_results['esti_imag_eval1']
nop_imag2 = df_nop_results['esti_imag_eval2']

df_nop_stable['is_nan'] = df_nop_stable['stable_expnt'].isna()
df_nop_nan = df_nop_stable[df_nop_stable['is_nan']]
df_nop = df_nop_stable[~df_nop_stable['is_nan']]

t_nop_nan = df_nop_nan['Time'] + 1
t_nop = df_nop['Time'] + 1
nop_stbexpnt_nan = np.zeros(len(t_nop_nan))
nop_stbexpnt = df_nop['stable_expnt']

subplt = ax1

subplt.scatter(t1[::5],x[::5],s=25,c='darkslateblue',marker='o')
subplt.plot(t1[::5],x[::5],c='darkslateblue')
subplt.scatter(t2[::5]+1,nu_x[::5],s=100, marker='o',facecolors='none',edgecolors='crimson')
subplt.scatter(1000,x[1000],s=500, marker='o',facecolors='white', edgecolors='black',zorder=5)
subplt.set_xlim(-62,1265)
subplt.set_ylim(0.2,8)
subplt.set_ylabel('system state',font_y,labelpad=-10)
subplt.text(820,1.44,'subcritical Hopf',size=32)
subplt.text(-0.075, 0.96,'a',ha='left', transform=subplt.transAxes,fontdict={'family':'DejaVu Sans','size':50,'weight':'bold'})
subplt.axvline(x=0, color='black', linestyle='--', linewidth=2)
subplt.axvline(x=500, color='black', linestyle='--', linewidth=2)
subplt.set_xticks([0,1000])
subplt.set_yticks([1,7])
subplt.tick_params(axis='x', labelbottom=False)

subplt.plot([15, 484], [2, 2], linestyle='--', color='black', linewidth=2)
arrow1 = FancyArrowPatch((5, 2), (494, 2), 
                        arrowstyle='<|-|>', 
                        color='black',
                        linewidth = 0, 
                        mutation_scale=25)
subplt.add_patch(arrow1)

subplt.text(240,2.3,'sliding window', fontsize=36, ha='center', va='center', color='black')
arrow2 = FancyArrowPatch((440, 2.3), (490, 2.3),  
                        arrowstyle='->', 
                        color='black',
                        linewidth=2,
                        mutation_scale=15)
subplt.add_patch(arrow2)
subplt.text(900, 0.8, r'using $\nu$ for estimating', fontsize=25, ha='center', va='center', color='black')

legend_state = mlines.Line2D([], [], color='darkslateblue', marker='o', markersize=5, linestyle='-', markeredgewidth=1)
legend_pstate = mlines.Line2D([], [], markerfacecolor='none',color='crimson', marker='o', markersize=10, linestyle='None', markeredgewidth=3)

ax_inset = fig.add_axes([0.165, 0.775, 0.14, 0.2]) 

ax_inset.scatter(t3, nu_stbexpnt, color='crimson', marker='x',zorder=6,s=75)
ax_inset.axhline(y=0, color='black', linestyle='--')
ax_inset.set_xlabel('Timepoint',font_in_x1,labelpad=-10)
ax_inset.set_xticks([971,1000])
ax_inset.set_xticklabels([971,1000], fontsize=24)
ax_inset.set_yticks([0,2,4])
ax_inset.set_yticklabels(['0','2','4'], fontsize=24)
ax_inset.yaxis.tick_right()
ax_inset.set_ylim(-0.92,4.6)
legend_index = mlines.Line2D([], [], color='crimson', marker='x', linestyle='None',markeredgewidth=2)
ax_inset.legend(handles=[legend_index],labels=['estimated stability index'],fontsize=28,loc='center', bbox_to_anchor=(0.5,0.09), frameon=False,handletextpad=-0.1,markerscale=3)

subplt.legend(handles=[legend_state,legend_pstate],labels=['state','predicted state'],loc='upper center', frameon=False, bbox_to_anchor=(0.23, 0.8), handletextpad=0.2,markerscale=2.5,prop={'size':32})

subplt = ax2

subplt.scatter(t2,real,s=20,c='darkslateblue',marker='^')
subplt.scatter(t2[df_results['esti_real_eval1'].between(-0.39, 0.02)],nu_real[df_results['esti_real_eval1'].between(-0.39, 0.02)],s=20,c='crimson',marker='^')
subplt.scatter(1000,real[499],s=500,marker='^',facecolors='white', edgecolors='black')
subplt.text(650,0,'reach zero',size=36)
subplt.set_xlabel('Timepoint',font_x,labelpad=-20)
subplt.set_ylabel('Re',font_y,labelpad=-50)
subplt.set_xlim(-62,1265)
subplt.text(-0.075, 0.96,'d',ha='left', transform=subplt.transAxes,fontdict={'family':'DejaVu Sans','size':50,'weight':'bold'})
subplt.axvline(x=500, color='black', linestyle='--', linewidth=2)
subplt.set_ylim(-0.45,0.05)
subplt.set_xticks([0,1000])
subplt.set_yticks([-0.4,0])
subplt.set_yticklabels(['-0.4','0'])

ax_inset = fig.add_axes([0.033, 0.17, 0.105, 0.22]) 

ax_inset.scatter(t2[::5],nu_imag1[::5], s=80, marker='^',facecolors='none',edgecolors='crimson',zorder=2)
ax_inset.scatter(t2[::5],nu_imag2[::5], s=80, marker='^',facecolors='none',edgecolors='crimson',zorder=2)
ax_inset.scatter(t2,imag,s=20, marker='^',c='darkslateblue',label = 'ground truth',zorder=1)
ax_inset.scatter(t2,-imag,s=20, marker='^',c='darkslateblue',label = 'ground truth',zorder=1)
ax_inset.axhline(y=0, color='black', linestyle='--')
ax_inset.set_xlabel('Timepoint',font_in_x2,labelpad=-20)
ax_inset.set_ylabel('Im',font_in_y2)
ax_inset.set_xticks([500,1000])
ax_inset.set_xticklabels([500,1000], fontsize=24)
ax_inset.set_yticks([-0.8,0,0.8])
ax_inset.set_yticklabels([-0.8,0,0.8], fontsize=24)
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

subplt = ax3

subplt.scatter(t1[::5],x[::5],s=25,c='darkslateblue',marker='o')
subplt.plot(t1[::5],x[::5],c='darkslateblue')
subplt.scatter(t2[::5]+1,ac_x[::5],s=100, marker='o',facecolors='none',edgecolors='crimson')
subplt.scatter(1000,x[1000],s=200, marker='o',facecolors='white', edgecolors='black',zorder=5)
subplt.set_xlim(-62,1265)
subplt.set_ylim(0.2,8)
subplt.text(820,1.44,'subcritical Hopf',size=32)
subplt.text(-0.075, 0.96,'b',ha='left', transform=subplt.transAxes,fontdict={'family':'DejaVu Sans','size':50,'weight':'bold'})
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

subplt.text(240,2.3,'sliding window', fontsize=36, ha='center', va='center', color='black')
arrow2 = FancyArrowPatch((440, 2.3), (490, 2.3), 
                        arrowstyle='->', 
                        color='black',
                        linewidth=2,
                        mutation_scale=15)
subplt.add_patch(arrow2)
subplt.text(900, 0.8,r'using parameters $a$ and $c$ for estimating', fontsize=20, ha='center', va='center', color='black')

legend_state = mlines.Line2D([], [], color='darkslateblue', marker='o', markersize=5, linestyle='-', markeredgewidth=1)
legend_pstate = mlines.Line2D([], [], markerfacecolor='none',color='crimson', marker='o', markersize=10, linestyle='None', markeredgewidth=3)

ax_inset = fig.add_axes([0.495, 0.775, 0.14, 0.2]) 

ax_inset.scatter(t3, ac_stbexpnt, color='crimson', marker='x',zorder=6,s=75)
ax_inset.axhline(y=0, color='black', linestyle='--')
ax_inset.set_xlabel('Timepoint',font_in_x1,labelpad=-10)
ax_inset.set_xticks([971,1000])
ax_inset.set_xticklabels([971,1000], fontsize=24)
ax_inset.set_yticks([0,0.5,1])
ax_inset.set_yticklabels(['0','0.5','1'], fontsize=24)
ax_inset.yaxis.tick_right()
ax_inset.set_ylim(-0.3,1.5)
legend_index = mlines.Line2D([], [], color='crimson', marker='x', linestyle='None',markeredgewidth=2)
ax_inset.legend(handles=[legend_index],labels=['estimated stability index'],fontsize=28,loc='center', bbox_to_anchor=(0.5,0.09), frameon=False,handletextpad=-0.1,markerscale=3)

subplt.legend(handles=[legend_state,legend_pstate],labels=['state','predicted state'],loc='upper center', frameon=False, bbox_to_anchor=(0.23, 0.8), handletextpad=0.2,markerscale=2.5,prop={'size':32})

subplt = ax4

subplt.scatter(t2,real,s=20,c='darkslateblue',marker='^')
subplt.scatter(t2[df_ac_results['esti_real_eval1'].between(-0.39, 0.02)],ac_real[df_ac_results['esti_real_eval1'].between(-0.39, 0.02)],s=20,c='crimson',marker='^')
subplt.scatter(1000,real[499],s=200,marker='^',facecolors='white', edgecolors='black')
subplt.text(650,0,'reach zero',size=36)
subplt.set_xlabel('Timepoint',font_x,labelpad=-20)
subplt.set_xlim(-62,1265)
subplt.text(-0.075, 0.96,'e',ha='left', transform=subplt.transAxes,fontdict={'family':'DejaVu Sans','size':50,'weight':'bold'})
subplt.axvline(x=500, color='black', linestyle='--', linewidth=2)
subplt.set_ylim(-0.45,0.05)
subplt.set_xticks([0,1000])
subplt.set_yticks([-0.4,0])
subplt.set_yticklabels(['-0.4','0'])

ax_inset = fig.add_axes([0.363, 0.17, 0.105, 0.22])

ax_inset.scatter(t2[::5],ac_imag1[::5], s=80, marker='^',facecolors='none',edgecolors='crimson',zorder=2)
ax_inset.scatter(t2[::5],ac_imag2[::5], s=80, marker='^',facecolors='none',edgecolors='crimson',zorder=2)
ax_inset.scatter(t2,imag,s=20, marker='^',c='darkslateblue',label = 'ground truth',zorder=1)
ax_inset.scatter(t2,-imag,s=20, marker='^',c='darkslateblue',label = 'ground truth',zorder=1)
ax_inset.axhline(y=0, color='black', linestyle='--')
ax_inset.set_xlabel('Timepoint',font_in_x2,labelpad=-20)
ax_inset.set_ylabel('Im',font_in_y2)
ax_inset.set_xticks([500,1000])
ax_inset.set_xticklabels([500,1000], fontsize=24)
ax_inset.set_yticks([-0.8,0,0.8])
ax_inset.set_yticklabels([-0.8,0,0.8], fontsize=24)
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

subplt = ax5

subplt.scatter(t1[::5],x[::5],s=25,c='darkslateblue',marker='o')
subplt.plot(t1[::5],x[::5],c='darkslateblue')
subplt.scatter(t2[::5]+1,nop_x[::5],s=100, marker='o',facecolors='none',edgecolors='crimson')
subplt.scatter(1000,x[1000],s=200, marker='o',facecolors='white', edgecolors='black',zorder=5)
subplt.set_xlim(-62,1265)
subplt.set_ylim(0.2,8)
subplt.text(820,1.44,'subcritical Hopf',size=32)
subplt.text(-0.075, 0.96,'c',ha='left', transform=subplt.transAxes,fontdict={'family':'DejaVu Sans','size':50,'weight':'bold'})
subplt.axvline(x=0, color='black', linestyle='--', linewidth=2)
subplt.axvline(x=500, color='black', linestyle='--', linewidth=2)
subplt.set_xticks([0,1000])
subplt.set_yticks([1,7])
subplt.tick_params(axis='x', labelbottom=False)

subplt.plot([15, 484], [2, 2], linestyle='--', color='black', linewidth=2)
arrow1 = FancyArrowPatch((5, 2), (494, 2), 
                        arrowstyle='<|-|>', 
                        color='black',
                        linewidth = 0, 
                        mutation_scale=25)
subplt.add_patch(arrow1)

subplt.text(240,2.3,'sliding window', fontsize=36, ha='center', va='center', color='black')
arrow2 = FancyArrowPatch((440, 2.3), (490, 2.3), 
                        arrowstyle='->', 
                        color='black',
                        linewidth=2,
                        mutation_scale=15)
subplt.add_patch(arrow2)
subplt.text(900, 0.8,'estimating without parameters', fontsize=25, ha='center', va='center', color='black')

legend_state = mlines.Line2D([], [], color='darkslateblue', marker='o', markersize=5, linestyle='-', markeredgewidth=1)
legend_pstate = mlines.Line2D([], [], markerfacecolor='none',color='crimson', marker='o', markersize=10, linestyle='None', markeredgewidth=3)

ax_inset = fig.add_axes([0.825, 0.775, 0.14, 0.2]) 

ax_inset.scatter(t_nop, nop_stbexpnt, color='crimson', marker='x',zorder=6,s=75)
ax_inset.scatter(t_nop_nan, nop_stbexpnt_nan, color='darkorange', marker='x',zorder=6,s=75)
ax_inset.axhline(y=0, color='black', linestyle='--')
ax_inset.set_xlabel('Timepoint',font_in_x1,labelpad=-10)
ax_inset.set_xticks([971,1000])
ax_inset.set_xticklabels([971,1000], fontsize=24)
ax_inset.set_yticks([0])
ax_inset.set_yticklabels(['0'], fontsize=24)
ax_inset.yaxis.tick_right()
ax_inset.set_ylim(-0.1,0.06)
legend_index = mlines.Line2D([], [], color='crimson', marker='x', linestyle='None',markeredgewidth=2)
legend_nan = mlines.Line2D([], [], color='darkorange', marker='x', linestyle='None',markeredgewidth=2)
ax_inset.legend(handles=[legend_index,legend_nan],labels=['estimated stability index','NaN'],fontsize=28,loc='center', bbox_to_anchor=(0.5,0.15), frameon=False,handletextpad=-0.1,markerscale=3)

subplt.legend(handles=[legend_state,legend_pstate],labels=['state','predicted state'],loc='upper center', frameon=False, bbox_to_anchor=(0.23, 0.8), handletextpad=0.2,markerscale=2.5,prop={'size':32})

subplt = ax6

subplt.scatter(t2,real,s=20,c='darkslateblue',marker='^')
subplt.scatter(t2,nop_real,s=20,c='crimson',marker='^')
subplt.scatter(1000,real[499],s=200,marker='^',facecolors='white', edgecolors='black')
subplt.text(1010,0,'reach zero',size=32)
subplt.set_xlim(-62,1265)
subplt.text(-0.075, 0.96,'f',ha='left', transform=subplt.transAxes,fontdict={'family':'DejaVu Sans','size':50,'weight':'bold'})
subplt.axvline(x=500, color='black', linestyle='--', linewidth=2)
subplt.set_ylim(-0.25,0.1)
subplt.set_xticks([0,1000])
subplt.set_yticks([-0.2,0])
subplt.set_xlabel('Timepoint',font_x,labelpad=-20)
subplt.set_yticklabels(['-0.2','0'])

ax_inset = fig.add_axes([0.693, 0.17, 0.105, 0.22])

ax_inset.scatter(t2[::5],nop_imag1[::5], s=80, marker='^',facecolors='none',edgecolors='crimson',zorder=2)
ax_inset.scatter(t2[::5],nop_imag2[::5], s=80, marker='^',facecolors='none',edgecolors='crimson',zorder=2)
ax_inset.scatter(t2,imag,s=20, marker='^',c='darkslateblue',label = 'ground truth',zorder=1)
ax_inset.scatter(t2,-imag,s=20, marker='^',c='darkslateblue',label = 'ground truth',zorder=1)
ax_inset.axhline(y=0, color='black', linestyle='--')
ax_inset.set_xlabel('Time',font_in_x2,labelpad=-20)
ax_inset.set_ylabel('Im',font_in_y2,labelpad=20)
ax_inset.set_xticks([500,1000])
ax_inset.set_xticklabels([500,1000], fontsize=24)
ax_inset.set_yticks([-0.8,0,0.8])
ax_inset.set_yticklabels([-0.8,0,0.8], fontsize=24)
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

plt.subplots_adjust(top=0.98, bottom=0.06, left=0.03, right=0.99, hspace=0.05, wspace=0.1)
plt.savefig('../figures/SFIG1.pdf',format='pdf',dpi=600)