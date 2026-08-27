import seaborn as sns
import matplotlib.pyplot as plt
import pandas
import matplotlib.lines as mlines
from matplotlib import font_manager
from sklearn.metrics import r2_score

font_x = {'family':'Arial','weight':'normal','size': 35}
font_y = {'family':'Arial','weight':'normal','size': 35}

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['xtick.labelsize'] = 30
plt.rcParams['ytick.labelsize'] = 30

fig, axs = plt.subplots(2, 4, figsize=(30,16))

ax1, ax2, ax3, ax7 = axs[0,0], axs[0,1], axs[0,2], axs[0,3]
ax4, ax5, ax6, ax8 = axs[1,0], axs[1,1], axs[1,2], axs[1,3]

df_fold_results = pandas.read_csv('../results/mean_lambda/fold_mean_lambda.csv')
df_fold_BA_results = pandas.read_csv('../results/mean_lambda/fold_BA_mean_lambda.csv')
df_fold_ER_results = pandas.read_csv('../results/mean_lambda/fold_ER_mean_lambda.csv')
df_sup_hopf_results = pandas.read_csv('../results/mean_lambda/sup_hopf_mean_lambda.csv')
df_trans_results = pandas.read_csv('../results/mean_lambda/trans_mean_lambda.csv')
df_sub_hopf_results = pandas.read_csv('../results/mean_lambda/sub_hopf_mean_lambda.csv')

fold_esti_re = df_fold_results['esti_mean'].to_numpy()
fold_ER_esti_re = df_fold_ER_results['esti_re_mean'].to_numpy()
fold_BA_esti_re = df_fold_BA_results['esti_re_mean'].to_numpy()
trans_esti_re = df_trans_results['esti_re_mean'].to_numpy()
sup_hopf_esti_re = df_sup_hopf_results['esti_re1_mean'].to_numpy()
sub_hopf_esti_re = df_sub_hopf_results['esti_re1_mean'].to_numpy()
sup_hopf_esti_im = df_sup_hopf_results['esti_im1_mean'].to_numpy()
sub_hopf_esti_im = df_sub_hopf_results['esti_im1_mean'].to_numpy()

df_fold_true = pandas.read_csv('../data/fold/fold_train.csv')
df_trans_true = pandas.read_csv('../data/trans/trans_train.csv')
df_sup_hopf_true = pandas.read_csv('../data/sup_hopf/sup_hopf_train.csv')
df_sub_hopf_true = pandas.read_csv('../data/sub_hopf/sub_hopf_train.csv')

fold_true_re = df_fold_true['rrate'].iloc[-500:].to_numpy()
fold_ER_true_re = df_fold_ER_results['true_re_mean'].to_numpy()
fold_BA_true_re = df_fold_BA_results['true_re_mean'].to_numpy()
trans_true_re = df_trans_true['real_eval1'].iloc[-500:].to_numpy()
sup_hopf_true_re = df_sup_hopf_true['real_eval1'].iloc[-500:].to_numpy()
sub_hopf_true_re = df_sub_hopf_true['real_eval1'].iloc[-500:].to_numpy()
sup_hopf_true_im = df_sup_hopf_true['imag_eval1'].iloc[-500:].to_numpy()
sub_hopf_true_im = df_sub_hopf_true['imag_eval1'].iloc[-500:].to_numpy()

# ax1
ax1.set_aspect('equal')

fold_r2 = r2_score(fold_true_re, fold_esti_re)

sns.scatterplot(
    x=fold_true_re,
    y=fold_esti_re,
    s=50,
    color='crimson',
    edgecolor=None,
    alpha=0.2,
    zorder=1,
    ax=ax1
)
mn = fold_true_re.min()
mx = fold_true_re.max()

ax1.plot([mn, mx], [mn, mx],color='darkgrey',linestyle='--',lw=2,alpha=0.9,zorder=2)
ax1.set_xlabel('True',font_x,labelpad=-10)
ax1.set_ylabel('Estimation',font_y)
ax1.yaxis.set_label_coords(-0.1, 0.5)
ax1.tick_params(direction='in',length=10)
ax1.set_title('Intraspecific competition',fontsize=30, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.05)
ax1.text(0.1, 0.8, f'$R^2 = {fold_r2:.2f}$', transform=ax1.transAxes, fontsize=36, color='black')
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)

ax1.set_xlim(-0.55,0.05)
ax1.set_xticks([-0.5,0])
ax1.set_xticklabels(['-0.5','0'])
ax1.set_ylim(-0.55,0.05)
ax1.set_yticks([-0.5,0])
ax1.set_yticklabels(['-0.5','0'])

ax1.text(-0.25,1.1,'(a)',ha='left', transform=ax1.transAxes,fontdict={'family':'Arial','size':50,'weight':'normal'})

# ax2
ax2.set_aspect('equal')

trans_r2 = r2_score(trans_true_re, trans_esti_re)

sns.scatterplot(
    x=trans_true_re,
    y=trans_esti_re,
    s=50,
    color='crimson',
    edgecolor=None,
    alpha=0.2,
    zorder=1,
    ax=ax2
)
mn = trans_true_re.min()
mx = trans_true_re.max()

ax2.plot([mn, mx], [mn, mx],color='darkgrey',linestyle='--',lw=2,alpha=0.9,zorder=2)
ax2.set_xlabel('True',font_x,labelpad=-10)
ax2.set_ylabel('Estimation',font_y)
ax2.yaxis.set_label_coords(-0.1, 0.5)
ax2.tick_params(direction='in',length=10)
ax2.set_title('Predator-prey (transcritical)',fontsize=30, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.05)
ax2.text(0.1, 0.8, f'$R^2 = {trans_r2:.2f}$', transform=ax2.transAxes, fontsize=36, color='black')
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)

ax2.set_xlim(-0.66,0.066)
ax2.set_xticks([-0.6,0])
ax2.set_xticklabels(['-0.6','0'])
ax2.set_ylim(-0.66,0.066)
ax2.set_yticks([-0.6,0])
ax2.set_yticklabels(['-0.6','0'])

ax2.text(-0.25,1.1,'(b)',ha='left', transform=ax2.transAxes,fontdict={'family':'Arial','size':50,'weight':'normal'})

# ax3
ax3.set_aspect('equal')

sup_hopf_r2 = r2_score(sup_hopf_true_re, sup_hopf_esti_re)

sns.scatterplot(
    x=sup_hopf_true_re,
    y=sup_hopf_esti_re,
    s=50,
    color='crimson',
    edgecolor=None,
    alpha=0.2,
    zorder=1,
    ax=ax3
)
mn = sup_hopf_true_re.min()
mx = sup_hopf_true_re.max()

ax3.plot([mn, mx], [mn, mx],color='darkgrey',linestyle='--',lw=2,alpha=0.9,zorder=2)
ax3.set_xlabel('True',font_x,labelpad=-10)
ax3.set_ylabel('Estimation',font_y)
ax3.yaxis.set_label_coords(-0.1, 0.5)
ax3.tick_params(direction='in',length=10)
ax3.set_title('Predator-prey (Hopf)',fontsize=30, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.05)
ax3.text(0.1, 0.8, f'$R^2 = {sup_hopf_r2:.2f}$', transform=ax3.transAxes, fontsize=36, color='black')
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)

ax3.set_xlim(-0.044,0.0044)
ax3.set_xticks([-0.04,0])
ax3.set_xticklabels(['-0.04','0'])
ax3.set_ylim(-0.044,0.0044)
ax3.set_yticks([-0.04,0])
ax3.set_yticklabels(['-0.04','0'])

ax3.text(-0.25,1.1,'(c)',ha='left', transform=ax3.transAxes,fontdict={'family':'Arial','size':50,'weight':'normal'})

# ax4
ax4.set_aspect('equal')

sub_hopf_r2 = r2_score(sub_hopf_true_re, sub_hopf_esti_re)

sns.scatterplot(
    x=sub_hopf_true_re,
    y=sub_hopf_esti_re,
    s=50,
    color='crimson',
    edgecolor=None,
    alpha=0.2,
    zorder=1,
    ax=ax4
)
mn = sub_hopf_true_re.min()
mx = sub_hopf_true_re.max()

ax4.plot([mn, mx], [mn, mx],color='darkgrey',linestyle='--',lw=2,alpha=0.9,zorder=2)
ax4.set_xlabel('True',font_x,labelpad=-10)
ax4.set_ylabel('Estimation',font_y)
ax4.yaxis.set_label_coords(-0.1, 0.5)
ax4.tick_params(direction='in',length=10)
ax4.set_title('Epidemic',fontsize=30, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.05)
ax4.text(0.1, 0.8, f'$R^2 = {sub_hopf_r2:.2f}$', transform=ax4.transAxes, fontsize=36, color='black')
ax4.spines['right'].set_visible(False)
ax4.spines['top'].set_visible(False)

ax4.set_xlim(-0.319,0.0319)
ax4.set_xticks([-0.29,0])
ax4.set_xticklabels(['-0.29','0'])
ax4.set_ylim(-0.319,0.0319)
ax4.set_yticks([-0.29,0])
ax4.set_yticklabels(['-0.29','0'])

ax4.text(-0.25,1.1,'(d)',ha='left', transform=ax4.transAxes,fontdict={'family':'Arial','size':50,'weight':'normal'})

# ax5
ax5.set_aspect('equal')

fold_ER_r2 = r2_score(fold_ER_true_re, fold_ER_esti_re)

sns.scatterplot(
    x=fold_ER_true_re,
    y=fold_ER_esti_re,
    s=50,
    color='crimson',
    edgecolor=None,
    alpha=0.2,
    zorder=1,
    ax=ax5
)
mn = fold_ER_true_re.min()
mx = fold_ER_true_re.max()

ax5.plot([mn, mx], [mn, mx],color='darkgrey',linestyle='--',lw=2,alpha=0.9,zorder=2)
ax5.set_xlabel('True',font_x,labelpad=-10)
ax5.set_ylabel('Estimation',font_y)
ax5.yaxis.set_label_coords(-0.1, 0.5)
ax5.tick_params(direction='in',length=10)
ax5.set_title('Mutualistic network (ER)',fontsize=30, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.05)
ax5.text(0.1, 0.8, f'$R^2 = {fold_ER_r2:.2f}$', transform=ax5.transAxes, fontsize=36, color='black')
ax5.spines['right'].set_visible(False)
ax5.spines['top'].set_visible(False)

ax5.set_xlim(-0.55,0.055)
ax5.set_xticks([-0.5,0])
ax5.set_xticklabels(['-0.5','0'])
ax5.set_ylim(-0.55,0.055)
ax5.set_yticks([-0.5,0])
ax5.set_yticklabels(['-0.5','0'])

ax5.text(-0.25,1.1,'(e)',ha='left', transform=ax5.transAxes,fontdict={'family':'Arial','size':50,'weight':'normal'})

# ax6
ax6.set_aspect('equal')

fold_BA_r2 = r2_score(fold_BA_true_re, fold_BA_esti_re)

sns.scatterplot(
    x=fold_BA_true_re,
    y=fold_BA_esti_re,
    s=50,
    color='crimson',
    edgecolor=None,
    alpha=0.2,
    zorder=1,
    ax=ax6
)
mn = fold_BA_true_re.min()
mx = fold_BA_true_re.max()

ax6.plot([mn, mx], [mn, mx],color='darkgrey',linestyle='--',lw=2,alpha=0.9,zorder=2)
ax6.set_xlabel('True',font_x,labelpad=-10)
ax6.set_ylabel('Estimation',font_y)
ax6.yaxis.set_label_coords(-0.1, 0.5)
ax6.tick_params(direction='in',length=10)
ax6.set_title('Mutualistic network (BA)',fontsize=30, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.05)
ax6.text(0.1, 0.8, f'$R^2 = {fold_BA_r2:.2f}$', transform=ax6.transAxes, fontsize=36, color='black')
ax6.spines['right'].set_visible(False)
ax6.spines['top'].set_visible(False)

ax6.set_xlim(-0.88,0.088)
ax6.set_xticks([-0.8,0])
ax6.set_xticklabels(['-0.8','0'])
ax6.set_ylim(-0.88,0.088)
ax6.set_yticks([-0.8,0])
ax6.set_yticklabels(['-0.8','0'])

ax6.text(-0.25,1.1,'(f)',ha='left', transform=ax6.transAxes,fontdict={'family':'Arial','size':50,'weight':'normal'})

# ax7
ax7.set_aspect('equal')

sup_hopf_r2 = r2_score(sup_hopf_true_im, sup_hopf_esti_im)

sns.scatterplot(
    x=sup_hopf_true_im,
    y=sup_hopf_esti_im,
    s=50,
    color='royalblue',
    edgecolor=None,
    alpha=0.2,
    zorder=1,
    ax=ax7
)
mn = sup_hopf_true_im.min()
mx = sup_hopf_true_im.max()

ax7.plot([mn, mx], [mn, mx],color='darkgrey',linestyle='--',lw=2,alpha=0.9,zorder=2)
ax7.set_xlabel('True',font_x,labelpad=-10)
ax7.set_ylabel('Estimation',font_y)
ax7.yaxis.set_label_coords(-0.1, 0.5)
ax7.tick_params(direction='in',length=10)
ax7.set_title('Predator-prey (Hopf)',fontsize=30, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.05)
ax7.text(0.1, 0.8, f'$R^2 = {sup_hopf_r2:.2f}$', transform=ax7.transAxes, fontsize=36, color='black')
ax7.spines['right'].set_visible(False)
ax7.spines['top'].set_visible(False)

ax7.set_xlim(0.111,0.137)
ax7.set_xticks([0.113,0.135])
ax7.set_xticklabels(['0.113','0.135'])
ax7.set_ylim(0.111,0.137)
ax7.set_yticks([0.113,0.135])
ax7.set_yticklabels(['0.113','0.135'])

ax7.text(-0.25,1.1,'(g)',ha='left', transform=ax7.transAxes,fontdict={'family':'Arial','size':50,'weight':'normal'})

# ax8
ax8.set_aspect('equal')

sub_hopf_r2 = r2_score(sub_hopf_true_im, sub_hopf_esti_im)

sns.scatterplot(
    x=sub_hopf_true_im,
    y=sub_hopf_esti_im,
    s=50,
    color='royalblue',
    edgecolor=None,
    alpha=0.2,
    zorder=1,
    ax=ax8
)
mn = sub_hopf_true_im.min()
mx = sub_hopf_true_im.max()

ax8.plot([mn, mx], [mn, mx],color='darkgrey',linestyle='--',lw=2,alpha=0.9,zorder=2)
ax8.set_xlabel('True',font_x,labelpad=-10)
ax8.set_ylabel('Estimation',font_y)
ax8.yaxis.set_label_coords(-0.1, 0.5)
ax8.tick_params(direction='in',length=10)
ax8.set_title('Epidemic',fontsize=30, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.05)
ax8.text(0.1, 0.8, f'$R^2 = {sub_hopf_r2:.2f}$', transform=ax8.transAxes, fontsize=36, color='black')
ax8.spines['right'].set_visible(False)
ax8.spines['top'].set_visible(False)

ax8.set_xlim(0.52,0.78)
ax8.set_xticks([0.55,0.75])
ax8.set_xticklabels(['0.55','0.75'])
ax8.set_ylim(0.52,0.78)
ax8.set_yticks([0.55,0.75])
ax8.set_yticklabels(['0.55','0.75'])

ax8.text(-0.25,1.1,'(h)',ha='left', transform=ax8.transAxes,fontdict={'family':'Arial','size':50,'weight':'normal'})

legend_esti_re = mlines.Line2D([], [], linestyle='None', markeredgecolor='None', marker='o',markerfacecolor='crimson', markersize=20, alpha=0.9)
legend_esti_im = mlines.Line2D([], [], linestyle='None', markeredgecolor='None', marker='o',markerfacecolor='royalblue', markersize=20, alpha=0.9)
legend_true = mlines.Line2D([], [], color='darkgrey', linestyle='--', linewidth=5, alpha=0.9)

fig.legend(handles = [legend_esti_re, legend_esti_im, legend_true],
            labels = ['Estimated Re', 'Estimated Im', 'True'],
              loc='upper center', bbox_to_anchor=(0.5,1.01), ncol=3, frameon=False, 
              prop=font_manager.FontProperties(family='Arial', size=30, weight='bold'),handletextpad=0.4)

plt.subplots_adjust(top=0.85, bottom=0.05, left=0.03, right=1, hspace=0.4, wspace=0.15)
plt.savefig('../figures/SFIG6.pdf',format='pdf')