import openpyxl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import re
import sys
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings

# --- 关键修改点 1: 检测并设置设备 ---
# 检查是否有可用的CUDA GPU，如果有就用GPU，否则用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- 将使用设备 (Using device): {device} ---")


# --- 步骤 1: 数据加载和特征工程函数 ---
def read_excel_range(file_path, sheet_name, range_string):
    # (此函数与之前相同)
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
        for row in sheet.iter_rows(min_row=row_start, max_row=row_end,
                                   min_col=col_start_idx, max_col=col_end_idx):
            data.append([cell.value if cell.value is not None else 0 for cell in row])
        return np.array(data)
    except Exception as e:
        print(f"Error reading Excel range '{range_string}': {e}")
        return np.array([])


def calculate_amplitude_and_frequency_fft(C, window_size=2):
    # (此函数与之前相同)
    n = len(C)
    if n < window_size: return np.array([]), np.array([])
    frequencies, amplitudes = [], []
    for i in range(n - window_size + 1):
        window_data = C[i:i + window_size]
        fft_vals = np.fft.fft(window_data)
        freqs = np.fft.fftfreq(window_size)
        positive_freqs = freqs[:window_size // 2]
        positive_fft_vals = fft_vals[:window_size // 2]
        amplitude = np.abs(positive_fft_vals)
        peak_freq = positive_freqs[np.argmax(amplitude)] if len(amplitude) > 0 else 0
        peak_amplitude = np.max(amplitude) if len(amplitude) > 0 else 0
        frequencies.append(peak_freq)
        amplitudes.append(peak_amplitude)
    return np.array(frequencies), np.array(amplitudes)


# --- 步骤 2: 加载所有指定的训练数据 ---
print("\n--- 正在加载训练数据 ---")
# (严格使用您提供的路径和范围)
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
            features = np.vstack([
                C[:min_len], delta_C[:min_len], amplitude[:min_len], frequency[:min_len]
            ]).T
            all_features_list.append(features)
            all_h_list.append(h[:min_len])
X_train = np.vstack(all_features_list)
y_train = np.concatenate(all_h_list)
print(f"训练数据准备完成. 总样本数: {len(X_train)}")

# --- 步骤 3: RF超参数寻优 (此部分在CPU上运行，无需修改) ---
print("\n--- 正在为随机森林(RF)寻优超参数 ---")
rf_param_grid = {'n_estimators': [5, 10, 20]}
rf_grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=rf_param_grid, cv=5, scoring='neg_mean_absolute_error',
    verbose=1, n_jobs=-1
)
rf_grid_search.fit(X_train, y_train)
best_rf_model = rf_grid_search.best_estimator_
print("找到的最佳随机森林参数: ", rf_grid_search.best_params_)

# --- 步骤 4: 使用调优后的RF模型准备MLP的输入数据 ---
X_train_leaf = best_rf_model.apply(X_train)
scaler_rf = StandardScaler()
X_train_leaf_scaled = scaler_rf.fit_transform(X_train_leaf)
# --- 关键修改点 2: 将用于MLP训练的数据张量移动到设备 ---
X_train_tensor = torch.tensor(X_train_leaf_scaled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)

# --- 步骤 5: 为MLP寻优超参数 ---
print("\n--- 正在为MLP寻优超参数 ---")


class AttentionMechanism(nn.Module):
    def __init__(self, input_dim):
        super(AttentionMechanism, self).__init__()
        self.attention_weights = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        attn_scores = F.softmax(self.attention_weights(x), dim=-1)
        return x * attn_scores, attn_scores


class MLPWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout_rate):
        super(MLPWithAttention, self).__init__()
        self.attention = AttentionMechanism(input_dim);
        layers = []
        in_features = input_dim
        for out_features in hidden_layers:
            layers.append(nn.Linear(in_features, out_features));
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout_rate));
            in_features = out_features
        layers.append(nn.Linear(in_features, 1));
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x, attn_scores = self.attention(x);
        x = self.layers(x)
        return x, attn_scores


mlp_param_grid = {
    'lr': [0.1, 0.01, 0.001],
    'hidden_layers': [(64, 32), (128, 64), (256, 128, 64)],
    'dropout_rate': [0.2, 0.3, 0.5],
    'epochs': [200, 300, 400]
}
kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_mlp_params = {};
best_mlp_score = float('inf')
input_dim = X_train_leaf_scaled.shape[1]

for lr in mlp_param_grid['lr']:
    for hidden_config in mlp_param_grid['hidden_layers']:
        for dropout in mlp_param_grid['dropout_rate']:
            for epochs in mlp_param_grid['epochs']:
                fold_scores = []
                print(f"测试MLP参数: lr={lr}, layers={hidden_config}, dropout={dropout}, epochs={epochs}")
                for train_idx, val_idx in kf.split(X_train_leaf_scaled):
                    X_train_fold, X_val_fold = X_train_tensor[train_idx], X_train_tensor[val_idx]
                    y_train_fold, y_val_fold = y_train_tensor[train_idx], y_train_tensor[val_idx]

                    # --- 关键修改点 3: 将每次新创建的MLP模型移动到GPU ---
                    mlp_model = MLPWithAttention(input_dim, hidden_layers=hidden_config, dropout_rate=dropout).to(
                        device)
                    criterion = nn.MSELoss()
                    optimizer = torch.optim.Adam(mlp_model.parameters(), lr=lr)

                    for epoch in range(epochs):
                        mlp_model.train();
                        optimizer.zero_grad()
                        outputs, _ = mlp_model(X_train_fold)
                        loss = criterion(outputs, y_train_fold)
                        loss.backward();
                        optimizer.step()

                    mlp_model.eval()
                    with torch.no_grad():
                        val_outputs, _ = mlp_model(X_val_fold)
                        val_loss = F.l1_loss(val_outputs, y_val_fold)
                        fold_scores.append(val_loss.item())

                avg_score = np.mean(fold_scores)
                print(f"  -> 平均验证集MAE: {avg_score:.4f}")

                if avg_score < best_mlp_score:
                    best_mlp_score = avg_score
                    best_mlp_params = {'lr': lr, 'hidden_layers': hidden_config, 'dropout_rate': dropout,
                                       'epochs': epochs}

print("找到的最佳MLP参数: ", best_mlp_params)

# # --- 步骤 6: 使用最佳超参数训练最终模型 ---
# print("\n--- 正在使用最佳参数训练最终模型 ---")
# final_model = MLPWithAttention(
#     input_dim,
#     hidden_layers=best_mlp_params['hidden_layers'],
#     dropout_rate=best_mlp_params['dropout_rate']
# ).to(device)  # 将最终模型移动到GPU
#
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(final_model.parameters(), lr=best_mlp_params['lr'])
# print("在全部训练数据上训练最终模型...")
# for epoch in range(best_mlp_params['epochs']):
#     final_model.train();
#     optimizer.zero_grad()
#     outputs, _ = final_model(X_train_tensor)
#     loss = criterion(outputs, y_train_tensor)
#     loss.backward();
#     optimizer.step()
#     if (epoch + 1) % 50 == 0:
#         print(f"最终模型训练周期 {epoch + 1}/{best_mlp_params['epochs']}, 损失: {loss.item():.4f}")
#
# # --- 步骤 7: 在您手动选择的测试集上进行评估 ---
# print("\n--- 在独立的测试集上评估最终模型性能 ---")
#
#
# def evaluate_on_test_set(data_sources, rf_model, scaler, mlp_model, plot_title="Test Set"):
#     all_features_list_test, all_h_list_test = [], []
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
#                 all_features_list_test.append(features);
#                 all_h_list_test.append(h[:min_len])
#
#     X_test_final = np.vstack(all_features_list_test)
#     y_test_final = np.concatenate(all_h_list_test)
#
#     X_test_leaf = rf_model.apply(X_test_final)
#     X_test_leaf_scaled = scaler.transform(X_test_leaf)
#
#     # --- 关键修改点 4: 将测试数据移动到GPU ---
#     X_test_tensor = torch.tensor(X_test_leaf_scaled, dtype=torch.float32).to(device)
#
#     mlp_model.eval()
#     with torch.no_grad():
#         predictions, _ = mlp_model(X_test_tensor)
#         predictions = predictions.cpu().numpy().flatten()  # 将结果移回CPU
#
#     mae = mean_absolute_error(y_test_final, predictions);
#     rmse = np.sqrt(mean_squared_error(y_test_final, predictions))
#     r2 = r2_score(y_test_final, predictions);
#     mape = np.mean(np.abs((y_test_final - predictions) / y_test_final)) * 100
#     print(f"--- {plot_title} 性能指标 ---")
#     print(f"平均绝对误差 (MAE): {mae:.4f}");
#     print(f"均方根误差 (RMSE): {rmse:.4f}")
#     print(f"决定系数 (R²): {r2:.4f}");
#     print(f"平均绝对百分比误差 (MAPE): {mape:.2f}%")
#
#
# # (严格使用您提供的路径和范围)
# test_data_30rpm = [{"file_path": "C:/Users/hs/Desktop/Constant rates validation data.xlsx", "ranges": [("Sheet1", "B2:B6149", "C2:C6149")]}]
# test_data_60rpm = [{"file_path": "C:/Users/hs/Desktop/Constant rates validation data.xlsx", "ranges": [("Sheet1", "I2:I3652", "J2:J3652")]}]
# test_data_100rpm = [
#     {"file_path": "C:/Users/hs/Desktop/Constant rates validation data.xlsx", "ranges": [("Sheet1", "P2:P2703", "Q2:Q2703")]}]
#
# evaluate_on_test_set(test_data_30rpm, best_rf_model, scaler_rf, final_model, plot_title="30 rpm Test Set")
# evaluate_on_test_set(test_data_60rpm, best_rf_model, scaler_rf, final_model, plot_title="60 rpm Test Set")
# evaluate_on_test_set(test_data_100rpm, best_rf_model, scaler_rf, final_model, plot_title="100 rpm Test Set")
#
# # --- 步骤 8: 保存最终的、调优后的模型 ---
# print("\n--- 正在保存最终模型 ---")
# torch.save(final_model.state_dict(), "final_rf_attention_model.pth")
# joblib.dump(best_rf_model, "final_random_forest_model.pkl")
# joblib.dump(scaler_rf, "final_scaler_rf.pkl")
# print("模型保存成功。")