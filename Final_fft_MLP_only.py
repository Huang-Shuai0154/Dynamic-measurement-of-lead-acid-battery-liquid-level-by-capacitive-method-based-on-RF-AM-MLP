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

# # --- 步骤 4: 使用最佳超参数训练最终模型 ---
# print("\n--- 正在使用最佳参数训练最终模型 ---")
# final_mlp_model = MLPModel(
#     input_dim,
#     hidden_layers=best_params['hidden_layers'],
#     dropout_rate=best_params['dropout_rate']
# )
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(final_mlp_model.parameters(), lr=best_params['lr'])
#
# print("在全部训练数据上训练最终模型...")
# for epoch in range(best_params['epochs']):
#     final_mlp_model.train()
#     optimizer.zero_grad()
#     outputs = final_mlp_model(X_train_tensor)
#     loss = criterion(outputs, y_train_tensor)
#     loss.backward()
#     optimizer.step()
#     if (epoch + 1) % 50 == 0:
#         print(f"最终模型训练周期 {epoch + 1}/{best_params['epochs']}, 损失: {loss.item():.4f}")
#
# # --- 步骤 5: 在您手动选择的测试集上进行评估 ---
# print("\n--- 在独立的测试集上评估最终模型性能 ---")
#
#
# def evaluate_on_test_set(data_sources, scaler, mlp_model, plot_title="Test Set"):
#     all_features_list_test = []
#     all_h_list_test = []
#     for source in data_sources:
#         for sheet_name, c_range, h_range in source["ranges"]:
#             C = read_excel_range(source["file_path"], sheet_name, c_range).flatten()
#             h = read_excel_range(source["file_path"], sheet_name, h_range).flatten()
#             if len(C) < 2: continue
#             delta_C = np.diff(C, prepend=C[0])
#             frequency, amplitude = calculate_amplitude_and_frequency_fft(C, window_size=2)
#             min_len = min(len(C), len(h), len(delta_C), len(frequency), len(amplitude))
#             if min_len > 0:
#                 features = np.vstack([C[:min_len], delta_C[:min_len], amplitude[:min_len], frequency[:min_len]]).T
#                 all_features_list_test.append(features)
#                 all_h_list_test.append(h[:min_len])
#
#     X_test_final = np.vstack(all_features_list_test)
#     y_test_final = np.concatenate(all_h_list_test)
#
#     # 使用在训练集上fit过的scaler来转换测试数据
#     X_test_scaled = scaler.transform(X_test_final)
#     X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
#
#     mlp_model.eval()
#     with torch.no_grad():
#         predictions = mlp_model(X_test_tensor)
#         predictions_numpy = predictions.cpu().numpy().flatten()
#
#     mae = mean_absolute_error(y_test_final, predictions_numpy)
#     rmse = np.sqrt(mean_squared_error(y_test_final, predictions_numpy))
#     r2 = r2_score(y_test_final, predictions_numpy)
#     mape = np.mean(np.abs((y_test_final - predictions_numpy) / y_test_final)) * 100
#
#     print(f"--- {plot_title} 性能指标 ---")
#     print(f"平均绝对误差 (MAE): {mae:.4f}")
#     print(f"均方根误差 (RMSE): {rmse:.4f}")
#     print(f"决定系数 (R²): {r2:.4f}")
#     print(f"平均绝对百分比误差 (MAPE): {mape:.2f}%")
#
#
# # 定义您的测试数据集（请确保文件路径正确）
# test_data_30rpm = [{"file_path": "C:/Users/hs/Desktop/Constant rates validation data.xlsx", "ranges": [("Sheet1", "B2:B6149", "C2:C6149")]}]
# test_data_60rpm = [{"file_path": "C:/Users/hs/Desktop/Constant rates validation data.xlsx", "ranges": [("Sheet1", "I2:I3652", "J2:J3652")]}]
# test_data_100rpm = [
#     {"file_path": "C:/Users/hs/Desktop/Constant rates validation data.xlsx", "ranges": [("Sheet1", "P2:P2703", "Q2:Q2703")]}]
#
# # 在每个测试集上运行评估
# evaluate_on_test_set(test_data_30rpm, scaler, final_mlp_model, plot_title="30 rpm Test Set")
# evaluate_on_test_set(test_data_60rpm, scaler, final_mlp_model, plot_title="60 rpm Test Set")
# evaluate_on_test_set(test_data_100rpm, scaler, final_mlp_model, plot_title="100 rpm Test Set")
#
# # --- 步骤 6: 保存最终的、调优后的模型 ---
# print("\n--- 正在保存最终模型 ---")
# torch.save(final_mlp_model.state_dict(), "final_tuned_mlp_model.pth")
# joblib.dump(scaler, "final_mlp_scaler.pkl")
# print("模型和scaler保存成功。")