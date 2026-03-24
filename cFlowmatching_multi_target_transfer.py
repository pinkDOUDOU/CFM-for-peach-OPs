"""
多目标变量、多场景迁移的 Flow Matching + PLSR 建模系统
支持：
- 4个硬度指标：ave_force, first_peak_slope, initial_slope, max_force
- 3种光学特性：mua, musp, mueff
- 3种迁移场景：HJ-P→HJ-S, HJ-P→SW-P, HJ-All→SW-P
- 2种建模方式：直接迁移、混合建模
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm
import math
import torch.nn.functional as F
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
KEY_WAVELENGTHS = [670, 950]
SAMPLING_STEPS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 2000
DOMAIN_EMBEDDING_DIM = 8

# 硬度指标列名
HARDNESS_COLS = ['ave_force', 'first_peak_slope', 'initial_slope', 'max_force']

# 光学特性类型
OPTICAL_TYPES = ['mua', 'musp', 'mueff']

# 迁移场景配置
TRANSFER_SCENARIOS = [
    {
        'name': 'HJ-P_to_HJ-S',
        'source_folder': 'HJ-P',
        'target_folder': 'HJ-S',
        'source_prefix': 'hj',
        'target_prefix': 'hj'
    },
    {
        'name': 'HJ-P_to_SW-P',
        'source_folder': 'HJ-P',
        'target_folder': 'SW-P',
        'source_prefix': 'hj',
        'target_prefix': 'zju'
    },
    {
        'name': 'HJ-All_to_SW-P',
        'source_folder': 'HJ-all',
        'target_folder': 'SW-P',
        'source_prefix': 'hj',
        'target_prefix': 'zju'
    }
]

# 建模模式
MODELING_MODES = ['direct_transfer', 'mixed']  # 直接迁移、混合建模

# 输出目录
OUTPUT_DIR = Path('./results_multi_target')
OUTPUT_DIR.mkdir(exist_ok=True)

# ==================== 工具函数 ====================
def find_nearest_idx(array, value):
    """找到数组中最接近给定值的索引"""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def get_file_path(folder, optical_type, prefix):
    """构建文件路径"""
    filename = f"{optical_type}_nb_no_{prefix}_ave_force450-1040.csv"
    return Path(folder) / filename

# ==================== 数据加载 ====================
def load_data_with_multi_targets(file_path):
    """
    加载包含多个硬度指标的数据
    返回：X (光谱), Y (多个硬度指标), sample_ids
    """
    df = pd.read_csv(file_path, encoding='gbk')

    # 第一列是ID
    sample_ids = df.iloc[:, 0].values

    # X: 从第2列到倒数第5列
    X = df.iloc[:, 1:-4].values.astype(float)

    # Y: 最后4列（硬度指标）
    Y = df.iloc[:, -4:].values.astype(float)

    # 波长信息
    wavelengths = df.columns[1:-4].astype(float).values

    return X, Y, sample_ids, wavelengths

def prepare_transfer_data(source_path, target_path, key_wavelengths, mode='direct_transfer'):
    """
    准备迁移学习数据
    mode: 'direct_transfer' (源训练→目标测试) 或 'mixed' (混合训练测试)
    """
    # 加载源域和目标域数据
    X_source, Y_source, ids_source, wavelengths = load_data_with_multi_targets(source_path)
    X_target, Y_target, ids_target, _ = load_data_with_multi_targets(target_path)

    # 提取关键波长条件
    key_idx_list = [find_nearest_idx(wavelengths, wl) for wl in key_wavelengths]

    def extract_conditions(X):
        return X[:, key_idx_list]

    # 域标签
    domain_source = np.zeros((len(X_source), 1))
    domain_target = np.ones((len(X_target), 1))

    # 标准化器
    x_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_scalers = [StandardScaler() for _ in range(Y_source.shape[1])]  # 每个Y独立标准化

    if mode == 'direct_transfer':
        # 源域训练，目标域测试
        X_train = x_scaler.fit_transform(X_source)
        X_test = x_scaler.transform(X_target)

        Y_train = np.column_stack([
            scaler.fit_transform(Y_source[:, i].reshape(-1, 1)).flatten()
            for i, scaler in enumerate(y_scalers)
        ])
        Y_test = np.column_stack([
            scaler.transform(Y_target[:, i].reshape(-1, 1)).flatten()
            for i, scaler in enumerate(y_scalers)
        ])

        C_train = extract_conditions(X_train)
        C_test = extract_conditions(X_test)

        return {
            'X_train': X_train, 'X_test': X_test,
            'C_train': C_train, 'C_test': C_test,
            'Y_train': Y_train, 'Y_test': Y_test,
            'domain_train': domain_source, 'domain_test': domain_target,
            'ids_train': ids_source, 'ids_test': ids_target,
            'x_scaler': x_scaler, 'y_scalers': y_scalers,
            'wavelengths': wavelengths
        }

    elif mode == 'mixed':
        # 混合两个域的数据
        X_all = np.vstack([X_source, X_target])
        Y_all = np.vstack([Y_source, Y_target])
        domain_all = np.vstack([domain_source, domain_target])
        ids_all = np.concatenate([ids_source, ids_target])

        # 标准化
        X_scaled = x_scaler.fit_transform(X_all)
        Y_scaled = np.column_stack([
            scaler.fit_transform(Y_all[:, i].reshape(-1, 1)).flatten()
            for i, scaler in enumerate(y_scalers)
        ])

        C_scaled = extract_conditions(X_scaled)

        # 划分训练测试集
        indices = np.arange(len(X_scaled))
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

        return {
            'X_train': X_scaled[train_idx], 'X_test': X_scaled[test_idx],
            'C_train': C_scaled[train_idx], 'C_test': C_scaled[test_idx],
            'Y_train': Y_scaled[train_idx], 'Y_test': Y_scaled[test_idx],
            'domain_train': domain_all[train_idx], 'domain_test': domain_all[test_idx],
            'ids_train': ids_all[train_idx], 'ids_test': ids_all[test_idx],
            'x_scaler': x_scaler, 'y_scalers': y_scalers,
            'wavelengths': wavelengths
        }

# ==================== 模型结构 ====================
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim, scale=1000.0):
        super().__init__()
        self.dim = dim
        self.scale = scale

    def forward(self, time):
        time = time * self.scale
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, cond_emb_dim):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.cond_mlp = nn.Linear(cond_emb_dim, out_channels)

    def forward(self, x, t_emb, c_emb):
        x = self.conv(x)
        x = self.norm(x)
        t_emb = self.time_mlp(t_emb).unsqueeze(-1)
        c_emb = self.cond_mlp(c_emb).unsqueeze(-1)
        x = x + t_emb + c_emb
        x = self.act(x)
        return x

class DomainAwareConditional1DUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_features=64,
                 n_conditions=2, domain_emb_dim=8):
        super().__init__()

        time_dim = n_features
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU()
        )

        cond_dim = n_features
        self.cond_mlp = nn.Sequential(
            nn.Linear(n_conditions, cond_dim // 2),
            nn.SiLU(),
            nn.Linear(cond_dim // 2, cond_dim),
            nn.SiLU()
        )

        self.domain_embedding = nn.Embedding(2, domain_emb_dim)
        self.domain_mlp = nn.Sequential(
            nn.Linear(domain_emb_dim, n_features // 4),
            nn.SiLU(),
            nn.Linear(n_features // 4, n_features // 2),
            nn.SiLU()
        )

        c4 = n_features // 4
        c2 = n_features // 2
        c1 = n_features

        self.init_conv = nn.Conv1d(in_channels, c4, kernel_size=3, padding=1)

        self.down1 = ConvBlock(c4, c2, time_dim, cond_dim + n_features//2)
        self.down2 = ConvBlock(c2, c1, time_dim, cond_dim + n_features//2)
        self.pool = nn.MaxPool1d(2)

        self.bot1 = ConvBlock(c1, c1, time_dim, cond_dim + n_features//2)

        self.up1 = nn.ConvTranspose1d(c1, c2, kernel_size=2, stride=2)
        self.up_conv1 = ConvBlock(c2 + c1, c2, time_dim, cond_dim + n_features//2)

        self.up2 = nn.ConvTranspose1d(c2, c4, kernel_size=2, stride=2)
        self.up_conv2 = ConvBlock(c4 + c2, c4, time_dim, cond_dim + n_features//2)

        self.out_conv = nn.Conv1d(c4, out_channels, kernel_size=1)

    def forward(self, x, t, c, domain):
        t_emb = self.time_mlp(t)
        c_emb = self.cond_mlp(c)

        domain_emb = self.domain_embedding(domain.long())
        domain_emb = self.domain_mlp(domain_emb)

        combined_emb = torch.cat([c_emb, domain_emb], dim=1)

        x1 = self.init_conv(x)
        x2 = self.down1(x1, t_emb, combined_emb)
        x2_p = self.pool(x2)
        x3 = self.down2(x2_p, t_emb, combined_emb)
        x3_p = self.pool(x3)

        xb = self.bot1(x3_p, t_emb, combined_emb)

        u1 = self.up1(xb)
        if u1.shape[-1] != x3.shape[-1]:
            u1 = F.interpolate(u1, size=x3.shape[-1], mode='linear', align_corners=False)
        u1 = torch.cat([u1, x3], dim=1)
        u1 = self.up_conv1(u1, t_emb, combined_emb)

        u2 = self.up2(u1)
        if u2.shape[-1] != x2.shape[-1]:
            u2 = F.interpolate(u2, size=x2.shape[-1], mode='linear', align_corners=False)
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.up_conv2(u2, t_emb, combined_emb)

        out = self.out_conv(u2)
        return out

# ==================== Flow Matching 训练 ====================
def compute_flow_matching_loss(model, x_1, c, domain):
    """计算 Flow Matching 损失"""
    batch_size = x_1.shape[0]

    t = torch.rand(batch_size, device=DEVICE)
    x_0 = torch.randn_like(x_1)

    t_expand = t.view(batch_size, 1, 1)
    x_t = (1 - t_expand) * x_0 + t_expand * x_1

    target_v = x_1 - x_0
    pred_v = model(x_t, t, c, domain)

    loss = F.mse_loss(pred_v, target_v)
    return loss

def train_flow_matching(model, dataloader, epochs, verbose=True):
    """训练 Flow Matching 模型"""
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        if verbose:
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        else:
            pbar = dataloader

        for batch_x, batch_c, batch_domain in pbar:
            optimizer.zero_grad()

            batch_x = batch_x.to(DEVICE)
            batch_c = batch_c.to(DEVICE)
            batch_domain = batch_domain.to(DEVICE).squeeze().long()

            loss = compute_flow_matching_loss(model, batch_x, batch_c, batch_domain)
            epoch_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if verbose and isinstance(pbar, tqdm):
                pbar.set_postfix(Loss=loss.item())

        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)

        if verbose and (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

@torch.no_grad()
def sample_ode_euler(model, c, domain, img_shape, steps=SAMPLING_STEPS):
    """使用欧拉法采样"""
    model.eval()
    batch_size = c.shape[0]

    x = torch.randn(batch_size, img_shape[0], img_shape[1], device=DEVICE)
    dt = 1.0 / steps

    for i in range(steps):
        t_value = i / steps
        t = torch.full((batch_size,), t_value, device=DEVICE)
        v = model(x, t, c, domain)
        x = x + v * dt

    return x

# ==================== PLSR 建模与评估 ====================
def build_plsr_models(X_train, Y_train, X_test, Y_test, n_components=5):
    """
    为每个硬度指标构建独立的PLSR模型
    返回：每个指标的 R², RMSE
    """
    n_targets = Y_train.shape[1]
    results = []

    for i in range(n_targets):
        plsr = PLSRegression(n_components=n_components, scale=False)
        plsr.fit(X_train, Y_train[:, i])

        y_train_pred = plsr.predict(X_train).flatten()
        y_test_pred = plsr.predict(X_test).flatten()

        r2_train = r2_score(Y_train[:, i], y_train_pred)
        r2_test = r2_score(Y_test[:, i], y_test_pred)
        rmse_train = np.sqrt(mean_squared_error(Y_train[:, i], y_train_pred))
        rmse_test = np.sqrt(mean_squared_error(Y_test[:, i], y_test_pred))

        results.append({
            'target_idx': i,
            'r2_train': r2_train,
            'r2_test': r2_test,
            'rmse_train': rmse_train,
            'rmse_test': rmse_test
        })

    return results

# ==================== 结果保存 ====================
def save_reconstruction_results(data, X_train_recon, X_test_recon, save_path):
    """
    保存光谱重建结果
    包含：样本ID、domain标签、原始光谱、重建光谱、硬度真实值
    """
    wavelengths = data['wavelengths']
    x_scaler = data['x_scaler']
    y_scalers = data['y_scalers']

    # 反归一化
    X_train_orig = x_scaler.inverse_transform(data['X_train'])
    X_train_recon_orig = x_scaler.inverse_transform(X_train_recon)
    X_test_orig = x_scaler.inverse_transform(data['X_test'])
    X_test_recon_orig = x_scaler.inverse_transform(X_test_recon)

    Y_train_orig = np.column_stack([
        scaler.inverse_transform(data['Y_train'][:, i].reshape(-1, 1)).flatten()
        for i, scaler in enumerate(y_scalers)
    ])
    Y_test_orig = np.column_stack([
        scaler.inverse_transform(data['Y_test'][:, i].reshape(-1, 1)).flatten()
        for i, scaler in enumerate(y_scalers)
    ])

    def create_df(ids, domain, X_orig, X_recon, Y_orig, set_name):
        n = len(X_orig)
        meta = {
            'Sample_ID': ids,
            'Set': [set_name] * n,
            'Domain': domain.flatten()
        }

        # 硬度指标
        for i, col in enumerate(HARDNESS_COLS):
            meta[f'{col}_true'] = Y_orig[:, i]

        df_meta = pd.DataFrame(meta)

        # 原始光谱
        df_orig = pd.DataFrame(X_orig, columns=[f"{wl:.2f}_original" for wl in wavelengths])

        # 重建光谱
        df_recon = pd.DataFrame(X_recon, columns=[f"{wl:.2f}_reconstructed" for wl in wavelengths])

        return pd.concat([df_meta, df_orig, df_recon], axis=1)

    df_train = create_df(data['ids_train'], data['domain_train'],
                         X_train_orig, X_train_recon_orig, Y_train_orig, 'Train')
    df_test = create_df(data['ids_test'], data['domain_test'],
                        X_test_orig, X_test_recon_orig, Y_test_orig, 'Test')

    df_final = pd.concat([df_train, df_test], ignore_index=True)
    df_final.to_csv(save_path, index=False)

def save_plsr_summary(all_results, save_path):
    """
    保存所有实验的PLSR建模结果汇总
    """
    df = pd.DataFrame(all_results)
    df.to_csv(save_path, index=False)

# ==================== 单次实验流程 ====================
def run_single_experiment(scenario, optical_type, mode, verbose=True):
    """
    运行单次实验
    返回：重建结果、PLSR结果
    """
    exp_name = f"{scenario['name']}_{optical_type}_{mode}"
    if verbose:
        print(f"\n{'='*60}")
        print(f"实验: {exp_name}")
        print(f"{'='*60}")

    # 构建文件路径
    source_path = get_file_path(scenario['source_folder'], optical_type, scenario['source_prefix'])
    target_path = get_file_path(scenario['target_folder'], optical_type, scenario['target_prefix'])

    if not source_path.exists() or not target_path.exists():
        print(f"警告: 文件不存在，跳过实验 {exp_name}")
        return None

    # 加载数据
    if verbose:
        print("加载数据...")
    data = prepare_transfer_data(source_path, target_path, KEY_WAVELENGTHS, mode)

    img_shape = (1, data['X_train'].shape[1])

    # 准备 DataLoader
    X_train_tensor = torch.tensor(data['X_train'], dtype=torch.float32).unsqueeze(1)
    C_train_tensor = torch.tensor(data['C_train'], dtype=torch.float32)
    domain_train_tensor = torch.tensor(data['domain_train'], dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, C_train_tensor, domain_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 初始化模型
    if verbose:
        print("初始化模型...")
    model = DomainAwareConditional1DUNet(
        in_channels=1, out_channels=1, n_features=64,
        n_conditions=len(KEY_WAVELENGTHS), domain_emb_dim=DOMAIN_EMBEDDING_DIM
    ).to(DEVICE)

    # 训练
    if verbose:
        print("开始训练...")
    train_flow_matching(model, train_loader, EPOCHS, verbose=verbose)

    # 重建光谱
    if verbose:
        print("重建光谱...")

    # 测试集
    C_test_tensor = torch.tensor(data['C_test'], dtype=torch.float32).to(DEVICE)
    domain_test_tensor = torch.tensor(data['domain_test'], dtype=torch.float32).to(DEVICE).squeeze().long()
    X_test_recon = sample_ode_euler(model, C_test_tensor, domain_test_tensor, img_shape)
    X_test_recon = X_test_recon.squeeze(1).cpu().numpy()

    # 训练集
    C_train_tensor_dev = C_train_tensor.to(DEVICE)
    domain_train_tensor_dev = domain_train_tensor.to(DEVICE).squeeze().long()
    X_train_recon = sample_ode_euler(model, C_train_tensor_dev, domain_train_tensor_dev, img_shape)
    X_train_recon = X_train_recon.squeeze(1).cpu().numpy()

    # 保存重建结果
    recon_save_path = OUTPUT_DIR / f"reconstruction_{exp_name}.csv"
    if verbose:
        print(f"保存重建结果到 {recon_save_path}")
    save_reconstruction_results(data, X_train_recon, X_test_recon, recon_save_path)

    # PLSR 建模
    if verbose:
        print("PLSR 建模...")

    plsr_results = []

    # 使用重建光谱建模
    plsr_recon = build_plsr_models(
        X_train_recon, data['Y_train'],
        X_test_recon, data['Y_test'],
        n_components=5
    )

    for res in plsr_recon:
        plsr_results.append({
            'scenario': scenario['name'],
            'optical_type': optical_type,
            'mode': mode,
            'target': HARDNESS_COLS[res['target_idx']],
            'spectrum_type': 'reconstructed',
            'r2_train': res['r2_train'],
            'r2_test': res['r2_test'],
            'rmse_train': res['rmse_train'],
            'rmse_test': res['rmse_test']
        })

    # 使用原始光谱建模（作为对比）
    plsr_orig = build_plsr_models(
        data['X_train'], data['Y_train'],
        data['X_test'], data['Y_test'],
        n_components=5
    )

    for res in plsr_orig:
        plsr_results.append({
            'scenario': scenario['name'],
            'optical_type': optical_type,
            'mode': mode,
            'target': HARDNESS_COLS[res['target_idx']],
            'spectrum_type': 'original',
            'r2_train': res['r2_train'],
            'r2_test': res['r2_test'],
            'rmse_train': res['rmse_train'],
            'rmse_test': res['rmse_test']
        })

    if verbose:
        print(f"实验 {exp_name} 完成")

    return plsr_results

# ==================== 主流程 ====================
def main():
    """
    主流程：遍历所有实验组合
    - 3种迁移场景
    - 3种光学特性
    - 2种建模模式
    - 4个硬度指标
    """
    print(f"使用设备: {DEVICE}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"训练轮数: {EPOCHS}")
    print(f"\n开始批量实验...")

    all_plsr_results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    total_experiments = len(TRANSFER_SCENARIOS) * len(OPTICAL_TYPES) * len(MODELING_MODES)
    current_exp = 0

    for scenario in TRANSFER_SCENARIOS:
        for optical_type in OPTICAL_TYPES:
            for mode in MODELING_MODES:
                current_exp += 1
                print(f"\n进度: {current_exp}/{total_experiments}")

                try:
                    results = run_single_experiment(scenario, optical_type, mode, verbose=True)
                    if results:
                        all_plsr_results.extend(results)
                except Exception as e:
                    print(f"实验失败: {scenario['name']}_{optical_type}_{mode}")
                    print(f"错误信息: {str(e)}")
                    continue

    # 保存汇总结果
    summary_path = OUTPUT_DIR / f"plsr_summary_{timestamp}.csv"
    print(f"\n保存PLSR汇总结果到 {summary_path}")
    save_plsr_summary(all_plsr_results, summary_path)

    print("\n" + "="*60)
    print("所有实验完成！")
    print(f"结果保存在: {OUTPUT_DIR}")
    print("="*60)

    # 打印简要统计
    if all_plsr_results:
        df_summary = pd.DataFrame(all_plsr_results)
        print("\n实验统计:")
        print(f"总实验数: {len(df_summary)}")
        print(f"场景数: {df_summary['scenario'].nunique()}")
        print(f"光学特性数: {df_summary['optical_type'].nunique()}")
        print(f"建模模式数: {df_summary['mode'].nunique()}")
        print(f"硬度指标数: {df_summary['target'].nunique()}")

        print("\n各场景平均测试集R²:")
        avg_r2 = df_summary.groupby(['scenario', 'mode', 'spectrum_type'])['r2_test'].mean()
        print(avg_r2)

if __name__ == "__main__":
    main()

