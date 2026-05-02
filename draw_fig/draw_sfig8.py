import pandas
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib as mpl

font_x = {'family':'Arial','weight':'normal','size': 35}
font_y = {'family':'Arial','weight':'normal','size': 35}

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['xtick.labelsize'] = 30
plt.rcParams['ytick.labelsize'] = 30

fig, axs = plt.subplots(2, 3, figsize=(24,14))

ax1, ax2, ax3 = axs[0,0], axs[0,1], axs[0,2]
ax4, ax5, ax6 = axs[1,0], axs[1,1], axs[1,2]

df_fold = pandas.read_csv('../results/analyze_bif/fold/bif_output.csv')
df_trans = pandas.read_csv('../results/analyze_bif/trans/bif_output.csv')
df_sup_hopf = pandas.read_csv('../results/analyze_bif/sup_hopf/bif_output.csv')
df_sub_hopf = pandas.read_csv('../results/analyze_bif/sub_hopf/bif_output.csv')
df_fold_ER = pandas.read_csv('../results/analyze_bif/fold_ER/bif_output.csv')
df_fold_BA = pandas.read_csv('../results/analyze_bif/fold_BA/bif_output.csv')

df_fold_eff = df_fold[df_fold['count'] != 0]
df_trans_eff = df_trans[df_trans['count'] != 0]
df_sup_hopf_eff = df_sup_hopf[df_sup_hopf['count'] != 0]
df_sub_hopf_eff = df_sub_hopf[df_sub_hopf['count'] != 0]
df_fold_ER_eff = df_fold_ER[df_fold_ER['count'] != 0]
df_fold_BA_eff = df_fold_BA[df_fold_BA['count'] != 0]

###################################################

t_fold = df_fold_eff['Time'].to_numpy()
t_trans = df_trans_eff['Time'].to_numpy()
t_sup_hopf = df_sup_hopf_eff['Time'].to_numpy()
t_sub_hopf = df_sub_hopf_eff['Time'].to_numpy()
t_fold_ER = df_fold_ER_eff['Time'].to_numpy()
t_fold_BA = df_fold_BA_eff['Time'].to_numpy()

###################################################

error_mean_fold = df_fold_eff['bif_error_mean'].to_numpy()
error_mean_trans = df_trans_eff['bif_error_mean'].to_numpy()
error_mean_sup_hopf = df_sup_hopf_eff['bif_error_mean'].to_numpy()
error_mean_sub_hopf = df_sub_hopf_eff['bif_error_mean'].to_numpy()
error_mean_fold_ER = df_fold_ER_eff['bif_error_mean'].to_numpy()
error_mean_fold_BA = df_fold_BA_eff['bif_error_mean'].to_numpy()

error_lower_fold = np.clip(df_fold_eff['bif_error_lower'].to_numpy(), 0, None)
error_lower_trans = np.clip(df_trans_eff['bif_error_lower'].to_numpy(), 0, None)
error_lower_sup_hopf = np.clip(df_sup_hopf_eff['bif_error_lower'].to_numpy(), 0, None)
error_lower_sub_hopf = np.clip(df_sub_hopf_eff['bif_error_lower'].to_numpy(), 0, None)
error_lower_fold_ER = np.clip(df_fold_ER_eff['bif_error_lower'].to_numpy(), 0, None)
error_lower_fold_BA = np.clip(df_fold_BA_eff['bif_error_lower'].to_numpy(), 0, None)

error_upper_fold = df_fold_eff['bif_error_upper'].to_numpy()
error_upper_trans = df_trans_eff['bif_error_upper'].to_numpy()
error_upper_sup_hopf = df_sup_hopf_eff['bif_error_upper'].to_numpy()
error_upper_sub_hopf = df_sub_hopf_eff['bif_error_upper'].to_numpy()
error_upper_fold_ER = df_fold_ER_eff['bif_error_upper'].to_numpy()
error_upper_fold_BA = df_fold_BA_eff['bif_error_upper'].to_numpy()

###################################################

type_value_mean_fold = df_fold_eff['bif_type_value_mean'].to_numpy()
type_value_mean_trans = df_trans_eff['bif_type_value_mean'].to_numpy()
type_value_mean_sup_hopf = df_sup_hopf_eff['bif_type_value_mean'].to_numpy()
type_value_mean_sub_hopf = df_sub_hopf_eff['bif_type_value_mean'].to_numpy()
type_value_mean_fold_ER = df_fold_ER_eff['bif_type_value_mean'].to_numpy()
type_value_mean_fold_BA = df_fold_BA_eff['bif_type_value_mean'].to_numpy()

###################################################

xmin=480
xmax=1020

###################################################

ax1.fill_between(t_fold, error_lower_fold, error_upper_fold,color='silver',alpha=0.3, linewidth=0, zorder=1)

ax1_points = np.array([t_fold, error_mean_fold]).T.reshape(-1, 1, 2)
ax1_segments = np.concatenate([ax1_points[:-1], ax1_points[1:]], axis=1)

ax1_norm = Normalize(vmin=-0.05, vmax=1.05)
ax1_lc = LineCollection(ax1_segments, cmap='coolwarm', norm=ax1_norm)
ax1_lc.set_array(type_value_mean_fold[:-1])
ax1_lc.set_linewidth(4)
ax1_lc.set_zorder(3)

ax1.add_collection(ax1_lc)

ax1.hlines(y=0, xmin=xmin, xmax=xmax, color='black', linestyle='--', linewidth=2, alpha=0.5, zorder=0)

ax1.set_xlim(xmin, xmax)
ax1.set_xticks([500,1000])

ax1.set_ylim(-0.07,0.7)
ax1.set_yticks([0,0.5])
ax1.set_yticklabels(['0','0.5'])

ax1.tick_params(direction='in', length=10)
ax1.minorticks_on()
ax1.tick_params(axis='x', which='minor', bottom=False)
ax1.tick_params(axis='y', which='minor', direction='in', length=5)

ax1.set_ylabel(r'Relative error ($\varepsilon$)', font_y)
ax1.yaxis.set_label_coords(-0.15, 0.5)

ax1.text(-0.2,1.05,'(a)',ha='left', transform=ax1.transAxes,fontdict={'family':'Arial','size':40,'weight':'normal'})
ax1.set_title('Intraspecific competition', fontsize=30, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.02)

###################################################

ax2.fill_between(t_trans, error_lower_trans, error_upper_trans,color='silver',alpha=0.3, linewidth=0, zorder=1)

ax2_points = np.array([t_trans, error_mean_trans]).T.reshape(-1, 1, 2)
ax2_segments = np.concatenate([ax2_points[:-1], ax2_points[1:]], axis=1)

ax2_norm = Normalize(vmin=-0.05, vmax=1.05)
ax2_lc = LineCollection(ax2_segments, cmap='coolwarm', norm=ax2_norm)
ax2_lc.set_array(type_value_mean_trans[:-1])
ax2_lc.set_linewidth(4)
ax2_lc.set_zorder(3)

ax2.add_collection(ax2_lc)

ax2.hlines(y=0, xmin=xmin, xmax=xmax, color='black', linestyle='--', linewidth=2, alpha=0.5, zorder=0)

ax2.set_xlim(xmin, xmax)
ax2.set_xticks([500,1000])

ax2.set_ylim(-0.028,0.28)
ax2.set_yticks([0,0.2])
ax2.set_yticklabels(['0','0.2'])

ax2.tick_params(direction='in', length=10)
ax2.minorticks_on()
ax2.tick_params(axis='x', which='minor', bottom=False)
ax2.tick_params(axis='y', which='minor', direction='in', length=5)

ax2.text(-0.2,1.05,'(b)',ha='left', transform=ax2.transAxes,fontdict={'family':'Arial','size':40,'weight':'normal'})
ax2.set_title('Predator-prey (transcritical)', fontsize=30, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.02)

###################################################

ax3.fill_between(t_sup_hopf, error_lower_sup_hopf, error_upper_sup_hopf,color='silver',alpha=0.3, linewidth=0, zorder=1)

ax3_points = np.array([t_sup_hopf, error_mean_sup_hopf]).T.reshape(-1, 1, 2)
ax3_segments = np.concatenate([ax3_points[:-1], ax3_points[1:]], axis=1)

ax3_norm = Normalize(vmin=-0.05, vmax=1.05)
ax3_lc = LineCollection(ax3_segments, cmap='coolwarm', norm=ax3_norm)
ax3_lc.set_array(type_value_mean_sup_hopf[:-1])
ax3_lc.set_linewidth(4)
ax3_lc.set_zorder(3)

ax3.add_collection(ax3_lc)

ax3.hlines(y=0, xmin=xmin, xmax=xmax, color='black', linestyle='--', linewidth=2, alpha=0.5, zorder=0)

ax3.set_xlim(xmin, xmax)
ax3.set_xticks([500,1000])

ax3.set_ylim(-0.056,0.56)
ax3.set_yticks([0,0.4])
ax3.set_yticklabels(['0','0.4'])

ax3.tick_params(direction='in', length=10)
ax3.minorticks_on()
ax3.tick_params(axis='x', which='minor', bottom=False)
ax3.tick_params(axis='y', which='minor', direction='in', length=5)

ax3.text(-0.2,1.05,'(c)',ha='left', transform=ax3.transAxes,fontdict={'family':'Arial','size':40,'weight':'normal'})
ax3.set_title('Predator-prey (Hopf)', fontsize=30, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.02)

###################################################

ax4.fill_between(t_sub_hopf, error_lower_sub_hopf, error_upper_sub_hopf,color='silver',alpha=0.3, linewidth=0, zorder=1)

ax4_points = np.array([t_sub_hopf, error_mean_sub_hopf]).T.reshape(-1, 1, 2)
ax4_segments = np.concatenate([ax4_points[:-1], ax4_points[1:]], axis=1)

ax4_norm = Normalize(vmin=-0.05, vmax=1.05)
ax4_lc = LineCollection(ax4_segments, cmap='coolwarm', norm=ax4_norm)
ax4_lc.set_array(type_value_mean_sub_hopf[:-1])
ax4_lc.set_linewidth(4)
ax4_lc.set_zorder(3)

ax4.add_collection(ax4_lc)

ax4.hlines(y=0, xmin=xmin, xmax=xmax, color='black', linestyle='--', linewidth=2, alpha=0.5, zorder=0)

ax4.set_xlim(xmin, xmax)
ax4.set_xticks([500,1000])

ax4.set_ylim(-0.028,0.28)
ax4.set_yticks([0,0.2])
ax4.set_yticklabels(['0','0.2'])

ax4.tick_params(direction='in', length=10)
ax4.minorticks_on()
ax4.tick_params(axis='x', which='minor', bottom=False)
ax4.tick_params(axis='y', which='minor', direction='in', length=5)

ax4.set_xlabel('Time',font_x,labelpad=0)
ax4.set_ylabel(r'Relative error ($\varepsilon$)', font_y)
ax4.yaxis.set_label_coords(-0.15, 0.5)

ax4.text(-0.2,1.05,'(d)',ha='left', transform=ax4.transAxes,fontdict={'family':'Arial','size':40,'weight':'normal'})
ax4.set_title('Epidemic', fontsize=30, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.02)

###################################################

ax5.fill_between(t_fold_ER, error_lower_fold_ER, error_upper_fold_ER,color='silver',alpha=0.3, linewidth=0, zorder=1)

ax5_points = np.array([t_fold_ER, error_mean_fold_ER]).T.reshape(-1, 1, 2)
ax5_segments = np.concatenate([ax5_points[:-1], ax5_points[1:]], axis=1)

ax5_norm = Normalize(vmin=-0.05, vmax=1.05)
ax5_lc = LineCollection(ax5_segments, cmap='coolwarm', norm=ax5_norm)
ax5_lc.set_array(type_value_mean_fold_ER[:-1])
ax5_lc.set_linewidth(4)
ax5_lc.set_zorder(3)

ax5.add_collection(ax5_lc)

ax5.hlines(y=0, xmin=xmin, xmax=xmax, color='black', linestyle='--', linewidth=2, alpha=0.5, zorder=0)

ax5.set_xlim(xmin, xmax)
ax5.set_xticks([500,1000])

ax5.set_ylim(-0.112,1.12)
ax5.set_yticks([0,0.8])
ax5.set_yticklabels(['0','0.8'])

ax5.tick_params(direction='in', length=10)
ax5.minorticks_on()
ax5.tick_params(axis='x', which='minor', bottom=False)
ax5.tick_params(axis='y', which='minor', direction='in', length=5)

ax5.set_xlabel('Time',font_x,labelpad=0)

ax5.text(-0.2,1.05,'(e)',ha='left', transform=ax5.transAxes,fontdict={'family':'Arial','size':40,'weight':'normal'})
ax5.set_title('Mutualistic network (ER)', fontsize=30, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.02)

###################################################

ax6.fill_between(t_fold_BA, error_lower_fold_BA, error_upper_fold_BA,color='silver',alpha=0.3, linewidth=0, zorder=1)

ax6_points = np.array([t_fold_BA, error_mean_fold_BA]).T.reshape(-1, 1, 2)
ax6_segments = np.concatenate([ax6_points[:-1], ax6_points[1:]], axis=1)

ax6_norm = Normalize(vmin=-0.05, vmax=1.05)
ax6_lc = LineCollection(ax6_segments, cmap='coolwarm', norm=ax6_norm)
ax6_lc.set_array(type_value_mean_fold_BA[:-1])
ax6_lc.set_linewidth(4)
ax6_lc.set_zorder(3)

ax6.add_collection(ax6_lc)

ax6.hlines(y=0, xmin=xmin, xmax=xmax, color='black', linestyle='--', linewidth=2, alpha=0.5, zorder=0)

ax6.set_xlim(xmin, xmax)
ax6.set_xticks([500,1000])

ax6.set_ylim(-0.14,1.4)
ax6.set_yticks([0,1])
ax6.set_yticklabels(['0','1'])

ax6.tick_params(direction='in', length=10)
ax6.minorticks_on()
ax6.tick_params(axis='x', which='minor', bottom=False)
ax6.tick_params(axis='y', which='minor', direction='in', length=5)

ax6.set_xlabel('Time',font_x,labelpad=0)

ax6.text(-0.2,1.05,'(f)',ha='left', transform=ax6.transAxes,fontdict={'family':'Arial','size':40,'weight':'normal'})
ax6.set_title('Mutualistic network (BA)', fontsize=30, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.02)

################################################################################################

norm = Normalize(vmin=-0.05, vmax=1.05)
sm = mpl.cm.ScalarMappable(cmap='coolwarm', norm=norm)
sm.set_array([])

cax = fig.add_axes([0.92, 0.25, 0.025, 0.5])
cbar = fig.colorbar(sm, cax=cax)
cbar.set_label(r'Classification accuracy ($r$)', rotation=270, fontsize=30)
cbar.ax.yaxis.set_label_coords(3.0, 0.5)
cbar.set_ticks([0, 0.5, 1])
cbar.set_ticklabels(['0', '0.5', '1'])
cbar.ax.tick_params(labelsize=30, length=10)

plt.subplots_adjust(top=0.94, bottom=0.06, left=0.06, right=0.88, hspace=0.35, wspace=0.3)
plt.savefig('../figures/SFIG8.pdf',format='pdf')