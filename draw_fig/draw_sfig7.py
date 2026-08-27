import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

font_x = {'family':'Arial','weight':'normal','size': 14}
font_y = {'family':'Arial','weight':'normal','size': 14}

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14


def ecdf(data):
    """计算经验累积分布函数 ECDF。"""
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]  # 删除 NaN 和 inf

    if data.size == 0:
        raise ValueError("输入数据中没有有效数值。")

    x = np.sort(data)
    y = np.arange(1, len(x) + 1) / len(x)

    # 补充起始点，使阶梯图从 0 开始
    x_plot = np.r_[x[0], x]
    y_plot = np.r_[0, y]

    return x_plot, y_plot

trans_im = []
sup_hopf_im = []
sub_hopf_im = []
fold_ER_im = []
fold_BA_im = []

for random_seed in range(200):

    trans = pd.read_csv('../results/ROC/trans/ews/dynamics_trans_{}.csv'.format(random_seed))['esti_imag_eval1'].to_numpy()
    sup_hopf = pd.read_csv('../results/ROC/sup_hopf/ews/dynamics_sup_hopf_{}.csv'.format(random_seed))['esti_imag_eval1'].to_numpy()
    sub_hopf = pd.read_csv('../results/ROC/sub_hopf/ews/dynamics_sub_hopf_{}.csv'.format(random_seed))['esti_imag_eval1'].to_numpy()
    trans_im.append(trans)
    sup_hopf_im.append(sup_hopf)
    sub_hopf_im.append(sub_hopf)

for network_size in [15, 20, 25, 30, 35, 40, 45, 50, 55, 60]:

    for random_seed in range(20):

        fold_ER = pd.read_csv('../results/ROC/fold_ER/ews/{}/dynamics_fold_ER_{}.csv'.format(network_size,random_seed))['esti_imag_dom_eval'].to_numpy()
        fold_BA = pd.read_csv('../results/ROC/fold_BA/ews/{}/dynamics_fold_BA_{}.csv'.format(network_size,random_seed))['esti_imag_dom_eval'].to_numpy()

        fold_ER_im.append(fold_ER)
        fold_BA_im.append(fold_BA)

trans_x, trans_y = ecdf(trans_im)
sup_hopf_x, sup_hopf_y = ecdf(sup_hopf_im)
sub_hopf_x, sub_hopf_y = ecdf(sub_hopf_im)
fold_ER_x, fold_ER_y = ecdf(fold_ER_im)
fold_BA_x, fold_BA_y = ecdf(fold_BA_im)

trans_ratio = np.mean(np.array(trans_im) < 1e-8)
sup_hopf_ratio = np.mean(np.array(sup_hopf_im) < 1e-8)
sub_hopf_ratio = np.mean(np.array(sub_hopf_im) < 1e-8)
fold_ER_ratio = np.mean(np.array(fold_ER_im) < 1e-8)
fold_BA_ratio = np.mean(np.array(fold_BA_im) < 1e-8)

fig, axs = plt.subplots(1, 5, figsize=(15,3))

ax1, ax2, ax3, ax4, ax5 = axs[0], axs[1], axs[2], axs[3], axs[4]

ax1.step(trans_x, trans_y, where='post', linewidth=2.5, color='royalblue', zorder=1)
ax2.step(sup_hopf_x, sup_hopf_y, where='post', linewidth=2.5, color='royalblue', zorder=1)
ax3.step(sub_hopf_x, sub_hopf_y, where='post', linewidth=2.5, color='royalblue', zorder=1)
ax4.step(fold_ER_x, fold_ER_y, where='post', linewidth=2.5, color='royalblue', zorder=1)
ax5.step(fold_BA_x, fold_BA_y, where='post', linewidth=2.5, color='royalblue', zorder=1)

ax1.axvline(0, linestyle='--', linewidth=1, color='silver', zorder=0)
ax2.axvline(0, linestyle='--', linewidth=1, color='silver', zorder=0)
ax3.axvline(0, linestyle='--', linewidth=1, color='silver', zorder=0)
ax4.axvline(0, linestyle='--', linewidth=1, color='silver', zorder=0)
ax5.axvline(0, linestyle='--', linewidth=1, color='silver', zorder=0)

ax1.set_xlabel('Im', font_x)
ax2.set_xlabel('Im', font_x)
ax3.set_xlabel('Im', font_x)
ax4.set_xlabel('Im', font_x)
ax5.set_xlabel('Im', font_x)

ax1.set_ylabel('Cumulative probability', font_y)

ax1.set_ylim(-0.05, 1.05)
ax2.set_ylim(-0.05, 1.05)
ax3.set_ylim(-0.05, 1.05)
ax4.set_ylim(-0.05, 1.05)
ax5.set_ylim(-0.05, 1.05)

ax1.set_xticks([0,0.1,0.2])
ax1.set_xticklabels(['0','0.1','0.2'])
ax1.set_yticks([0,0.5,1])
ax1.set_yticklabels(['0','0.5','1'])

ax2.set_xticks([0,0.1,0.2])
ax2.set_xticklabels(['0','0.1','0.2'])
ax2.set_yticks([0,0.5,1])
ax2.set_yticklabels(['0','0.5','1'])

ax3.set_xticks([0,0.5,1])
ax3.set_xticklabels(['0','0.5','1'])
ax3.set_yticks([0,0.5,1])
ax3.set_yticklabels(['0','0.5','1'])

ax4.set_xticks([0,0.00075,0.0015])
ax4.set_xticklabels(['0','0.00075','0.0015'])
ax4.set_yticks([0,0.5,1])
ax4.set_yticklabels(['0','0.5','1'])

ax5.set_xticks([0,0.001,0.002])
ax5.set_xticklabels(['0','0.001','0.002'])
ax5.set_yticks([0,0.5,1])
ax5.set_yticklabels(['0','0.5','1'])

trans_str = r'$\rho_{\text{Im=0}} = %.4f$' % trans_ratio
sup_hopf_str = r'$\rho_{\text{Im=0}} = %.4f$' % sup_hopf_ratio
sub_hopf_str = r'$\rho_{\text{Im=0}} = %.4f$' % sub_hopf_ratio
fold_ER_str = r'$\rho_{\text{Im=0}} = %.4f$' % fold_ER_ratio
fold_BA_str = r'$\rho_{\text{Im=0}} = %.4f$' % fold_BA_ratio

ax1.text(0.08, 0.9, trans_str, transform=ax1.transAxes, fontsize=14, verticalalignment='top')
ax2.text(0.08, 0.9, sup_hopf_str, transform=ax2.transAxes, fontsize=14, verticalalignment='top')
ax3.text(0.08, 0.9, sub_hopf_str, transform=ax3.transAxes, fontsize=14, verticalalignment='top')
ax4.text(0.08, 0.9, fold_ER_str, transform=ax4.transAxes, fontsize=14, verticalalignment='top')
ax5.text(0.08, 0.9, fold_BA_str, transform=ax5.transAxes, fontsize=14, verticalalignment='top')

ax1.set_title('Predator-prey (transcritical)', fontsize=12, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.02)
ax1.text(0.85, 0.75,'(a)',ha='left', transform=ax1.transAxes,fontdict={'family':'Arial','size':17,'weight':'normal'})

ax2.set_title('Predator-prey (Hopf)', fontsize=12, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.02)
ax2.text(0.85, 0.75,'(b)',ha='left', transform=ax2.transAxes,fontdict={'family':'Arial','size':17,'weight':'normal'})

ax3.set_title('Epidemic', fontsize=12, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.02)
ax3.text(0.85, 0.75,'(c)',ha='left', transform=ax3.transAxes,fontdict={'family':'Arial','size':17,'weight':'normal'})

ax4.set_title('Mutualistic network (ER)', fontsize=12, fontweight='normal', fontfamily='DejaVu Sans', fontstyle= 'italic', y=1.02)
ax4.text(0.85, 0.75,'(d)',ha='left', transform=ax4.transAxes,fontdict={'family':'Arial','size':17,'weight':'normal'})

ax5.set_title('Mutualistic network (BA)', fontsize=12, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.02)
ax5.text(0.85, 0.75,'(e)',ha='left', transform=ax5.transAxes,fontdict={'family':'Arial','size':17,'weight':'normal'})

plt.subplots_adjust(top=0.88, bottom=0.18, left=0.05, right=0.975, wspace=0.25)
plt.savefig('../figures/SFIG7.pdf',format='pdf')