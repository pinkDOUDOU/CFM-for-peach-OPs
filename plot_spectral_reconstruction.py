"""
Flow Matching 光谱重建质量可视化
9张子图（3迁移场景 × 3光学特性）+ 每张图单独保存为高质量TIFF格式
每个组合自动选取光谱重建质量最优的建模模式（direct_transfer / mixed）
展示：源域原始光谱 / 目标域原始光谱 / 目标域重建光谱（均值 ± 1σ）
同时将每个子图的绘图数据导出为Excel文档（含各组均值、标准差及逐样本数据）
字体：中文仿宋，英文Times New Roman
修改：增大x/y轴数字和标题字体，并加粗
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.metrics import r2_score
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# ==================== 字体 ====================
# 设置双字体：英文优先使用 Times New Roman，中文回退到 FangSong
matplotlib.rcParams['font.family']       = ['Times New Roman', 'FangSong']
matplotlib.rcParams['mathtext.fontset']  = 'stix'
matplotlib.rcParams['axes.unicode_minus'] = False

# ==================== 配置 ====================
RESULTS_DIR = Path('./results_multi_target')
OUTPUT_DIR  = RESULTS_DIR

SCENARIOS =['HJ-P_to_HJ-S', 'HJ-P_to_SW-P', 'HJ-All_to_SW-P']
OPTICAL_TYPES = ['mua', 'musp', 'mueff']

SCENARIO_LABELS = {
    'HJ-P_to_HJ-S':   'HJ-P → HJ-S',
    'HJ-P_to_SW-P':   'HJ-P → SW-P',
    'HJ-All_to_SW-P': 'HJ-All → SW-P',
}

# 核心修改1：对于希腊字母和公式，使用 \boldsymbol{} 才能将其加粗
OPTICAL_YLABELS = {
    'mua':   r'$\boldsymbol{\mu_a}$ (mm$\boldsymbol{^{-1}}$)',
    'musp':  r"$\boldsymbol{\mu_s'}$ (mm$\boldsymbol{^{-1}}$)",
    'mueff': r'$\boldsymbol{\mu_{eff}}$ (mm$\boldsymbol{^{-1}}$)',
}
MODE_CN = {'direct_transfer': '直接迁移', 'mixed': '混合建模'}

# 三组曲线的颜色
C_SRC   = '#888888'   # 源域原始
C_TGT   = '#2471A3'   # 目标域原始
C_RECON = '#E74C3C'   # 目标域重建

ALPHA_FILL = 0.18
LW         = 3

# ==================== 工具函数 ====================

def load_recon_csv(scenario, optical_type, mode):
    p = RESULTS_DIR / f"reconstruction_{scenario}_{optical_type}_{mode}.csv"
    if not p.exists():
        raise FileNotFoundError(str(p))
    return pd.read_csv(p)


def parse_wavelengths(df):
    orig_cols =[c for c in df.columns if c.endswith('_original')]
    wl = np.array([float(c.replace('_original', '')) for c in orig_cols])
    return wl, orig_cols


def spectral_r2_mean(X_orig, X_recon):
    """每个样本光谱的R²，再取均值（衡量重建质量）"""
    r2s = [r2_score(X_orig[i], X_recon[i]) for i in range(len(X_orig))]
    return float(np.mean(r2s))


def get_best_mode(scenario, optical_type):
    """
    比较两种模式对目标域(Domain=1)光谱的重建R²，返回质量更好的mode
    """
    best_mode, best_r2 = None, -np.inf
    for mode in['direct_transfer', 'mixed']:
        try:
            df = load_recon_csv(scenario, optical_type, mode)
        except FileNotFoundError:
            continue

        tgt = df[df['Domain'] == 1]
        if len(tgt) == 0:
            continue

        wl, orig_cols = parse_wavelengths(df)
        recon_cols =[c.replace('_original', '_reconstructed') for c in orig_cols]

        X_orig  = tgt[orig_cols].values.astype(float)
        X_recon = tgt[recon_cols].values.astype(float)

        r2 = spectral_r2_mean(X_orig, X_recon)
        if r2 > best_r2:
            best_r2, best_mode = r2, mode

    return best_mode, best_r2


# ==================== 单图绘制 ====================

def draw_panel(ax, scenario, optical_type, mode, df, show_xlabel=True, show_ylabel=True):
    """
    在 ax 上绘制光谱重建对比曲线
    返回 (handles, labels, mean_r2_target)
    """
    wl, orig_cols = parse_wavelengths(df)
    recon_cols =[c.replace('_original', '_reconstructed') for c in orig_cols]

    src_mask = df['Domain'] == 0
    tgt_mask = df['Domain'] == 1

    def plot_band(mask, cols, color, ls='-', label=''):
        sub = df[mask]
        if len(sub) == 0:
            return None
        X = sub[cols].values.astype(float)
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        ax.fill_between(wl, mu - sd, mu + sd, color=color, alpha=ALPHA_FILL)
        line, = ax.plot(wl, mu, color=color, lw=LW, ls=ls, label=label)
        return line

    h1 = plot_band(src_mask, orig_cols,  C_SRC,   '-',  '源域原始')
    h2 = plot_band(tgt_mask, orig_cols,  C_TGT,   '-',  '目标域原始')
    h3 = plot_band(tgt_mask, recon_cols, C_RECON, '--', '目标域重建')

    # 计算目标域重建R²
    tgt = df[tgt_mask]
    X_orig  = tgt[orig_cols].values.astype(float)
    X_recon = tgt[recon_cols].values.astype(float)
    r2_tgt  = spectral_r2_mean(X_orig, X_recon)

    # 核心修改2：去掉强制的 fontproperties，使用 fontsize 和 fontweight
    # 这样系统会自动用 Times New Roman 加粗渲染 (nm)，用 FangSong 渲染 中文，效果最佳。
    if show_xlabel:
        ax.set_xlabel('波长 (nm)', fontsize=32, fontweight='bold')
    if show_ylabel:
        ax.set_ylabel(OPTICAL_YLABELS[optical_type], fontsize=32, fontweight='bold')

    # 核心修改3：显著增大刻度标签字体（字号提至28），并对其加粗
    ax.tick_params(labelsize=28, width=2, length=6)
    for label in ax.get_xticklabels():
        label.set_weight('bold')
    for label in ax.get_yticklabels():
        label.set_weight('bold')

    handles =[h for h in [h1, h2, h3] if h is not None]
    labels  = [h.get_label() for h in handles]
    return handles, labels, r2_tgt


# ==================== 组图 ====================

def build_combined_figure(panel_data):
    """
    panel_data: dict[(r, c)] = (scenario, optical_type, mode, df)
    """
    # 核心修改4：相应扩大画布尺寸以包容变大变粗的字体
    fig, axes = plt.subplots(3, 3, figsize=(22, 18), constrained_layout=True)

    shared_handles, shared_labels = None, None

    for r, scenario in enumerate(SCENARIOS):
        for c, opt in enumerate(OPTICAL_TYPES):
            ax = axes[r][c]
            key = (r, c)

            if key not in panel_data:
                ax.text(0.5, 0.5, '数据缺失', transform=ax.transAxes,
                        ha='center', va='center', fontsize=32, fontweight='bold')
                continue

            scenario_, opt_, mode, df = panel_data[key]
            show_x = (r == 2)
            show_y = True

            h, l, r2 = draw_panel(ax, scenario_, opt_, mode, df,
                                   show_xlabel=show_x, show_ylabel=show_y)

            if shared_handles is None:
                shared_handles, shared_labels = h, l

            # 列标题（第一行）- 核心修改5：字号提升至36并加粗
            if r == 0:
                ax.set_title(
                             OPTICAL_YLABELS[opt_].split('(')[0].strip(),
                             fontsize=36,
                             fontweight='bold',
                             pad=12
                            )

            # 行标签（最左列）叠加场景名 - 核心修改6：字号提升至32并加粗
            if c == 0:
                current_ylabel = OPTICAL_YLABELS[opt_]
                ax.set_ylabel(
                            f"{SCENARIO_LABELS[scenario_]}\n\n{current_ylabel}",
                            fontsize=32,
                            fontweight='bold',
                            labelpad=12
                        )

    # 图例 - 核心修改7：用 prop 字典传参以继承双字体回退，并大幅调大图例
    if shared_handles:
        leg = fig.legend(
            shared_handles, shared_labels,
            loc='lower center', ncol=3,
            frameon=True,
            edgecolor='#cccccc',
            bbox_to_anchor=(0.5, -0.06),
            prop={'size': 26, 'weight': 'bold'}  # 增大图例字体并加粗
        )

    # 总标题 - 核心修改8：大幅增大字号
    fig.suptitle(
        'Flow Matching 光谱重建质量对比（均值 ± 1σ，各组合选取最优建模模式）',
        fontsize=38, 
        fontweight='bold',
        y=1.03
    )

    return fig


# ==================== 单独子图（用于EMF导出）====================

def build_single_figure(scenario, optical_type, mode, df):
    """生成单张独立图（用于EMF导出）"""
    # 核心修改9：稍微放大单图的画布尺寸以防止字体越界
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)

    h, l, r2 = draw_panel(ax, scenario, optical_type, mode, df,
                           show_xlabel=True, show_ylabel=True)

    # 核心修改10：用 prop 字典设置单图图例
    leg = ax.legend(h, l, loc='upper left',
              frameon=True, edgecolor='#cccccc',
              prop={'size': 24, 'weight': 'bold'})

    return fig


def save_tiff(fig, path, dpi=600):
    """保存为高质量TIFF"""
    fig.savefig(str(path), format='tiff', dpi=dpi,
                bbox_inches='tight',
                pil_kwargs={'compression': 'tiff_lzw'})


# ==================== Excel 数据导出 ====================

def export_panel_data_to_excel(scenario, optical_type, mode, df, out_path):  # noqa: ARG001
    """导出Excel数据 (代码未改动)"""
    wl, orig_cols = parse_wavelengths(df)
    recon_cols =[c.replace('_original', '_reconstructed') for c in orig_cols]

    src = df[df['Domain'] == 0].reset_index(drop=True)
    tgt = df[df['Domain'] == 1].reset_index(drop=True)

    X_src_orig  = src[orig_cols].values.astype(float)
    X_tgt_orig  = tgt[orig_cols].values.astype(float)
    X_tgt_recon = tgt[recon_cols].values.astype(float)

    # ---- Sheet1: 均值与标准差 ----
    df_stat = pd.DataFrame({'波长_nm': wl})
    df_stat['源域原始_均值']     = X_src_orig.mean(axis=0)  if len(X_src_orig)  > 0 else np.nan
    df_stat['源域原始_标准差']   = X_src_orig.std(axis=0)   if len(X_src_orig)  > 0 else np.nan
    df_stat['目标域原始_均值']   = X_tgt_orig.mean(axis=0)  if len(X_tgt_orig)  > 0 else np.nan
    df_stat['目标域原始_标准差'] = X_tgt_orig.std(axis=0)   if len(X_tgt_orig)  > 0 else np.nan
    df_stat['目标域重建_均值']   = X_tgt_recon.mean(axis=0) if len(X_tgt_recon) > 0 else np.nan
    df_stat['目标域重建_标准差'] = X_tgt_recon.std(axis=0)  if len(X_tgt_recon) > 0 else np.nan

    # ---- Sheet2/3/4: 逐样本 ----
    def make_sample_df(ids, X):
        cols = ['样本ID'] +[f'{w:.2f}nm' for w in wl]
        rows = [[ids[i]] + X[i].tolist() for i in range(len(X))]
        return pd.DataFrame(rows, columns=cols)

    df_src_orig  = make_sample_df(src['Sample_ID'].values, X_src_orig)
    df_tgt_orig  = make_sample_df(tgt['Sample_ID'].values, X_tgt_orig)
    df_tgt_recon = make_sample_df(tgt['Sample_ID'].values, X_tgt_recon)

    # ---- Sheet5: 实验信息 ----
    df_info = pd.DataFrame({
        '字段':['场景', '光学特性', '建模模式', '源域样本数', '目标域样本数'],
        '内容':  [scenario, optical_type, MODE_CN[mode], len(src), len(tgt)]
    })

    with pd.ExcelWriter(str(out_path), engine='openpyxl') as writer:
        df_info.to_excel(writer,     sheet_name='实验信息',       index=False)
        df_stat.to_excel(writer,     sheet_name='均值与标准差',   index=False)
        df_src_orig.to_excel(writer, sheet_name='源域原始_逐样本',   index=False)
        df_tgt_orig.to_excel(writer, sheet_name='目标域原始_逐样本', index=False)
        df_tgt_recon.to_excel(writer,sheet_name='目标域重建_逐样本', index=False)


# ==================== 主程序 ====================

def main():
    print("加载数据并选取最优建模模式...\n")
    print(f"{'场景':<22} {'光学特性':<8} {'最优模式':<22} {'目标域重建R²':>12}")
    print("-" * 68)

    panel_data = {}

    for r, scenario in enumerate(SCENARIOS):
        for c, opt in enumerate(OPTICAL_TYPES):
            best_mode, best_r2 = get_best_mode(scenario, opt)

            if best_mode is None:
                print(f"{scenario:<22} {opt:<8} {'N/A':<22} {'N/A':>12}")
                continue

            try:
                df = load_recon_csv(scenario, opt, best_mode)
                panel_data[(r, c)] = (scenario, opt, best_mode, df)
                print(f"{scenario:<22} {opt:<8} {best_mode:<22} {best_r2:>12.4f}")
            except Exception as e:
                print(f"{scenario:<22} {opt:<8} 加载失败: {e}")

    # ---- 组合图（高质量TIFF）----
    print("\n生成组合图（TIFF 600 dpi）...")
    fig_combined = build_combined_figure(panel_data)
    combined_tiff = OUTPUT_DIR / 'spectral_reconstruction_combined.tiff'
    save_tiff(fig_combined, combined_tiff, dpi=600)
    print(f"  已保存: {combined_tiff.name}")

    # ---- 单图TIFF + Excel 数据 ----
    tiff_dir  = OUTPUT_DIR / 'tiff_individual'
    excel_dir = OUTPUT_DIR / 'excel_data'
    tiff_dir.mkdir(exist_ok=True)
    excel_dir.mkdir(exist_ok=True)

    print("\n生成各子图 TIFF 及 Excel 数据...")
    for (r, c), (scenario, opt, mode, df) in panel_data.items():
        stem = f"recon_{scenario}_{opt}"

        # 单图 TIFF
        fig_single = build_single_figure(scenario, opt, mode, df)
        tiff_path  = tiff_dir / f"{stem}.tiff"
        save_tiff(fig_single, tiff_path, dpi=600)
        print(f"  TIFF  : {tiff_path.name}")
        plt.close(fig_single)

        # Excel 数据
        excel_path = excel_dir / f"{stem}.xlsx"
        export_panel_data_to_excel(scenario, opt, mode, df, excel_path)
        print(f"  Excel : {excel_path.name}")

    print(f"\n单图目录 : {tiff_dir}")
    print(f"数据目录 : {excel_dir}")
    print("\n完成！")
    plt.show()


if __name__ == '__main__':
    main()