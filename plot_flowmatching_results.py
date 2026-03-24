"""
对 cFlowmatching_multi_target_transfer.py 预测结果进行绘图
9张子图：3种迁移场景 × 3种光学特性
每组合自动选取最优建模模式(direct_transfer/mixed)
在标准化空间中叠加展示4个硬度指标的 预测值 vs 真实值
字体：中文仿宋，英文Times New Roman
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# ==================== 字体配置 ====================
# matplotlib搜索顺序：先用Times New Roman渲染西文，遇到汉字fallback到仿宋
matplotlib.rcParams['font.family'] = ['Times New Roman', 'FangSong']
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['axes.unicode_minus'] = False

# 仿宋字体属性（用于纯中文文本元素）
_fp_fangsong = fm.FontProperties(family='FangSong', size=9)
_fp_fangsong_sm = fm.FontProperties(family='FangSong', size=8)

# ==================== 配置 ====================
RESULTS_DIR    = Path('./results_multi_target')
HARDNESS_COLS  = ['ave_force', 'first_peak_slope', 'initial_slope', 'max_force']
OPTICAL_TYPES  = ['mua', 'musp', 'mueff']
SCENARIOS      = ['HJ-P_to_HJ-S', 'HJ-P_to_SW-P', 'HJ-All_to_SW-P']

SCENARIO_LABELS = {
    'HJ-P_to_HJ-S':   'HJ-P → HJ-S',
    'HJ-P_to_SW-P':   'HJ-P → SW-P',
    'HJ-All_to_SW-P': 'HJ-All → SW-P',
}
OPTICAL_LABELS = {
    'mua':   r'$\mu_a$',
    'musp':  r"$\mu_s'$",
    'mueff': r'$\mu_{eff}$',
}
HARDNESS_LABELS = {
    'ave_force':        'Ave Force',
    'first_peak_slope': 'First Peak Slope',
    'initial_slope':    'Initial Slope',
    'max_force':        'Max Force',
}
MODE_CN = {'direct_transfer': '直接迁移', 'mixed': '混合建模'}

# 散点样式
COLORS  = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
MARKERS = ['o', 's', '^', 'D']

# ==================== 数据处理函数 ====================

def find_latest_summary():
    files = sorted(RESULTS_DIR.glob('plsr_summary_*.csv'))
    if not files:
        raise FileNotFoundError(f"未找到 plsr_summary_*.csv 于 {RESULTS_DIR}")
    return files[-1]


def get_best_mode(df_summary, scenario, optical_type):
    """
    选取(scenario, optical_type)下，重建光谱 r2_test 跨4个硬度指标平均最大的mode
    返回 (best_mode, mean_r2)
    """
    mask = (
        (df_summary['scenario'] == scenario) &
        (df_summary['optical_type'] == optical_type) &
        (df_summary['spectrum_type'] == 'reconstructed')
    )
    sub = df_summary[mask]
    if sub.empty:
        return None, None
    avg = sub.groupby('mode')['r2_test'].mean()
    best = avg.idxmax()
    return best, avg[best]


def load_recon_csv(scenario, optical_type, mode):
    path = RESULTS_DIR / f"reconstruction_{scenario}_{optical_type}_{mode}.csv"
    if not path.exists():
        raise FileNotFoundError(str(path))
    return pd.read_csv(path)


def run_plsr(df, target_col, n_components=5):
    """
    在重建光谱上重新拟合PLSR，返回测试集标准化空间的真实值和预测值，以及R²
    标准化方式与主脚本一致
    """
    recon_cols = [c for c in df.columns if c.endswith('_reconstructed')]
    true_col   = f"{target_col}_true"

    train = df[df['Set'] == 'Train']
    test  = df[df['Set'] == 'Test']

    X_tr = train[recon_cols].values.astype(float)
    X_te = test[recon_cols].values.astype(float)
    y_tr = train[true_col].values.astype(float)
    y_te = test[true_col].values.astype(float)

    x_sc = MinMaxScaler(feature_range=(-1, 1))
    y_sc = StandardScaler()

    X_tr_s = x_sc.fit_transform(X_tr)
    X_te_s = x_sc.transform(X_te)
    y_tr_s = y_sc.fit_transform(y_tr.reshape(-1, 1)).flatten()
    y_te_s = y_sc.transform(y_te.reshape(-1, 1)).flatten()

    nc = min(n_components, X_tr_s.shape[1], X_tr_s.shape[0] - 1)
    plsr = PLSRegression(n_components=nc, scale=False)
    plsr.fit(X_tr_s, y_tr_s)

    y_pred_s = plsr.predict(X_te_s).flatten()

    r2   = r2_score(y_te_s, y_pred_s)
    rmse = np.sqrt(mean_squared_error(y_te_s, y_pred_s))

    # 反标准化供标注真实指标值用（可选），但绘图用标准化值
    return y_te_s, y_pred_s, r2, rmse


# ==================== 绘图函数 ====================

def draw_panel(ax, scenario, optical_type, mode, df_recon):
    """
    在单个Axes上绘制4个硬度指标标准化预测值 vs 真实值
    返回 legend handles 和 labels
    """
    all_true, all_pred = [], []
    handles, labels   = [], []

    for i, target in enumerate(HARDNESS_COLS):
        y_true, y_pred, r2, rmse = run_plsr(df_recon, target)

        all_true.extend(y_true)
        all_pred.extend(y_pred)

        sc = ax.scatter(
            y_true, y_pred,
            c=COLORS[i], marker=MARKERS[i],
            s=28, alpha=0.65, edgecolors='none', zorder=3,
            label=f"{HARDNESS_LABELS[target]}  $R^2$={r2:.3f}"
        )
        handles.append(sc)
        labels.append(f"{HARDNESS_LABELS[target]}  $R^2$={r2:.3f}")

    # 1:1 参考线
    vals    = all_true + all_pred
    lo, hi  = min(vals), max(vals)
    pad     = (hi - lo) * 0.06
    rng     = [lo - pad, hi + pad]
    ax.plot(rng, rng, 'k--', lw=0.9, alpha=0.55, zorder=2)
    ax.set_xlim(rng);  ax.set_ylim(rng)

    # 坐标轴（英文，Times New Roman已在rcParams设置）
    ax.set_xlabel('True Value (standardized)', fontsize=7.5)
    ax.set_ylabel('Predicted Value (standardized)', fontsize=7.5)
    ax.tick_params(labelsize=6.5)

    # 右上角注释最优模式（中文仿宋）
    ax.text(0.98, 0.02, MODE_CN[mode],
            transform=ax.transAxes, fontsize=7,
            va='bottom', ha='right',
            fontproperties=_fp_fangsong_sm,
            color='#666666')

    return handles, labels


def build_figure(df_summary):
    """生成3行×3列总图"""
    fig, axes = plt.subplots(3, 3, figsize=(11, 9.5),
                             constrained_layout=True)

    shared_handles = None
    shared_labels  = None

    for r, scenario in enumerate(SCENARIOS):
        for c, opt in enumerate(OPTICAL_TYPES):
            ax = axes[r][c]

            best_mode, _ = get_best_mode(df_summary, scenario, opt)

            if best_mode is None:
                ax.text(0.5, 0.5, '数据缺失', transform=ax.transAxes,
                        ha='center', va='center',
                        fontproperties=_fp_fangsong)
                continue

            try:
                df_recon = load_recon_csv(scenario, opt, best_mode)
                h, l = draw_panel(ax, scenario, opt, best_mode, df_recon)
                if shared_handles is None:
                    shared_handles, shared_labels = h, l
            except Exception as e:
                ax.text(0.5, 0.5, f'失败\n{str(e)[:50]}',
                        transform=ax.transAxes, ha='center', va='center',
                        fontsize=6, fontproperties=_fp_fangsong)
                print(f"  [警告] {scenario}/{opt}: {e}")
                continue

            # 列标题（第一行）：光学特性符号
            if r == 0:
                ax.set_title(OPTICAL_LABELS[opt], fontsize=12, pad=5)

            # 行标签（最左列）：场景名
            if c == 0:
                ax.set_ylabel(
                    f"{SCENARIO_LABELS[scenario]}\n\nPredicted Value (standardized)",
                    fontsize=8, labelpad=4
                )
            else:
                ax.set_ylabel('Predicted Value (standardized)', fontsize=7.5)

    # 底部统一图例
    if shared_handles:
        fig.legend(
            shared_handles, shared_labels,
            loc='lower center',
            ncol=4,
            fontsize=8,
            frameon=True,
            edgecolor='#cccccc',
            bbox_to_anchor=(0.5, -0.055),
        )

    # 总标题（中文仿宋）
    fig.suptitle(
        'Flow Matching 重建光谱 PLSR 硬度预测结果（标准化空间，各组合选取最优模式）',
        fontsize=10.5,
        fontproperties=_fp_fangsong,
        y=1.015
    )

    return fig


# ==================== 主程序 ====================

def main():
    summary_path = find_latest_summary()
    print(f"使用: {summary_path.name}\n")
    df_summary = pd.read_csv(summary_path)
    # 去掉末尾空行
    df_summary = df_summary.dropna(subset=['scenario'])

    # 打印最优模式概览
    print(f"{'场景':<22} {'光学特性':<8} {'最优模式':<22} {'平均R²':>8}")
    print("-" * 64)
    for scenario in SCENARIOS:
        for opt in OPTICAL_TYPES:
            bm, br2 = get_best_mode(df_summary, scenario, opt)
            if bm:
                print(f"{scenario:<22} {opt:<8} {bm:<22} {br2:>8.4f}")
            else:
                print(f"{scenario:<22} {opt:<8} {'N/A':<22} {'N/A':>8}")

    print("\n正在生成图形...")
    fig = build_figure(df_summary)

    out_png = RESULTS_DIR / 'flowmatching_prediction_scatter.png'
    out_pdf = RESULTS_DIR / 'flowmatching_prediction_scatter.pdf'
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
    print(f"\n已保存:\n  {out_png}\n  {out_pdf}")

    plt.show()


if __name__ == '__main__':
    main()
