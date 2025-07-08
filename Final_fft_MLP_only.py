import openpyxl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import sys
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings

# --- 步骤 1: 数据加载和特征工程函数 (与之前相同) ---
def read_excel_range(file_path, sheet_name, range_string):
    try:
        match = re.match(r"([A-Z]+)(\d+):([A-Z]+)(\d+)", range_string)
        if not match: raise ValueError(f"Invalid range format: {range_string}")
        col_start, row_start, col_end, row_end = match.groups()
        row_start, row_end = int(row_start), int(row_end)
        wb = openpyxl.load_workbook(file_path, data_only=True)
        sheet = wb[sheet_name]
        col_start_idx = openpyxl.utils.column_index_from_string(col_start)
        col_end_idx = openpyxl.utils.column_index_from_string(col_end)
        data = []
        for row in sheet.iter_rows(min_row=row_start, max_row=row_end, min_col=col_start_idx, max_col=col_end_idx):
            data.append([cell.value if cell.value is not None else 0 for cell in row])
        return np.array(data)
    except Exception as e:
        print(f"Error reading Excel range '{range_string}': {e}")
        return np.array([])


def calculate_amplitude_and_frequency_fft(C, window_size=2):
    n = len(C)
    if n < window_size: return np.array([]), np.array([])
    frequencies, amplitudes = [], []
    for i in range(n - window_size + 1):
        window_data = C[i:i + window_size];
        fft_vals = np.fft.fft(window_data)
        freqs = np.fft.fftfreq(window_size)
        positive_freqs = freqs[:window_size // 2];
        positive_fft_vals = fft_vals[:window_size // 2]
        amplitude = np.abs(positive_fft_vals)
        peak_freq = positive_freqs[np.argmax(amplitude)] if len(amplitude) > 0 else 0
        peak_amplitude = np.max(amplitude) if len(amplitude) > 0 else 0
        frequencies.append(peak_freq);
        amplitudes.append(peak_amplitude)
    return np.array(frequencies), np.array(amplitudes)


# --- 步骤 2: 加载所有指定的训练数据 ---
print("\n--- 正在加载训练数据 ---")
# 注意：请将文件路径修改为您本地的正确路径
data_sources = [{
    "file_path": "C:/Users/hs/Desktop/Training data.xlsx",  # 请确保路径正确
    "ranges": [
        ("Sheet1", "B2:B2797", "C2:C2797"), ("Sheet1", "H2:H2691", "I2:I2691"),
        ("Sheet1", "N2:N3809", "O2:O3809"), ("Sheet1", "T2:T2289", "U2:U2289"),
        ("Sheet1", "Z2:Z6328", "AA2:AA6328"), ("Sheet1", "AF2:AF2582", "AG2:AG2582"),
        ("Sheet1", "AL2:AL4265", "AM2:AM4265")
    ]
}]

all_features_list = []
all_h_list = []
for source in data_sources:
    for sheet_name, c_range, h_range in source["ranges"]:
        C = read_excel_range(source["file_path"], sheet_name, c_range).flatten()
        h = read_excel_range(source["file_path"], sheet_name, h_range).flatten()
        if len(C) < 2: continue
        delta_C = np.diff(C, prepend=C[0])
        frequency, amplitude = calculate_amplitude_and_frequency_fft(C, window_size=2)
        min_len = min(len(C), len(h), len(delta_C), len(frequency), len(amplitude))
        if min_len > 0:
            features = np.vstack([C[:min_len], delta_C[:min_len], amplitude[:min_len], frequency[:min_len]]).T
            all_features_list.append(features)
            all_h_list.append(h[:min_len])

# --- 修改点: 直接将所有加载的数据作为训练集 ---
X_train = np.vstack(all_features_list)
y_train = np.concatenate(all_h_list)

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
print(f"训练数据准备完成. 总样本数: {len(X_train)}")

# --- 步骤 3: MLP超参数寻优 ---
print("\n--- 正在为MLP寻优超参数 ---")


# 修改MLPModel类以接收超参数
class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout_rate):
        super(MLPModel, self).__init__()
        layers = []
        in_features = input_dim
        for out_features in hidden_layers:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_features = out_features
        layers.append(nn.Linear(in_features, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# --- 修改点: 使用您指定的参数范围 ---
param_grid = {
    'lr': [0.1, 0.01, 0.001],
    'hidden_layers': [(64, 32), (128, 64), (256, 128, 64)],
    'dropout_rate': [0.2, 0.3, 0.5],
    'epochs': [200, 300, 400]
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_params = {}
best_score = float('inf')
input_dim = X_train_scaled.shape[1]

# 手动网格搜索循环
for lr in param_grid['lr']:
    for hidden_config in param_grid['hidden_layers']:
        for dropout in param_grid['dropout_rate']:
            for epochs in param_grid['epochs']:
                fold_scores = []
                print(f"测试MLP参数: lr={lr}, layers={hidden_config}, dropout={dropout}, epochs={epochs}")
                for train_idx, val_idx in kf.split(X_train_scaled):
                    X_train_fold, X_val_fold = X_train_tensor[train_idx], X_train_tensor[val_idx]
                    y_train_fold, y_val_fold = y_train_tensor[train_idx], y_train_tensor[val_idx]

                    model = MLPModel(input_dim, hidden_layers=hidden_config, dropout_rate=dropout)
                    criterion = nn.MSELoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                    # 在K-1折上训练
                    for epoch in range(epochs):
                        model.train()
                        optimizer.zero_grad()
                        outputs = model(X_train_fold)
                        loss = criterion(outputs, y_train_fold)
                        loss.backward()
                        optimizer.step()

                    # 在剩余的1折上验证
                    model.eval()
                    with torch.no_grad():
                        val_outputs = model(X_val_fold)
                        val_loss = F.l1_loss(val_outputs, y_val_fold)  # 使用MAE作为评估指标
                        fold_scores.append(val_loss.item())

                avg_score = np.mean(fold_scores)
                print(f"  -> 平均验证集MAE: {avg_score:.4f}")

                # 记录并更新最佳参数
                if avg_score < best_score:
                    best_score = avg_score
                    best_params = {'lr': lr, 'hidden_layers': hidden_config, 'dropout_rate': dropout, 'epochs': epochs}

print("\n找到的最佳MLP参数: ", best_params)

