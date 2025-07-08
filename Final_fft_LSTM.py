import openpyxl
import numpy as np
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
from torch.utils.data import TensorDataset, DataLoader  # 确保导入了DataLoader

# --- 设置设备 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- 将使用设备 (Using device): {device} ---")


# --- 辅助函数定义 ---
# ... (read_excel_range, calculate_amplitude_and_frequency_fft, create_sequences 函数与之前版本完全相同，此处省略) ...
def read_excel_range(file_path, sheet_name, range_string):
    try:
        match = re.match(r"([A-Z]+)(\d+):([A-Z]+)(\d+)", range_string)
        if not match: raise ValueError(f"Invalid range format: {range_string}")
        col_start, row_start, col_end, row_end = match.groups();
        row_start, row_end = int(row_start), int(row_end)
        wb = openpyxl.load_workbook(file_path, data_only=True);
        sheet = wb[sheet_name]
        col_start_idx = openpyxl.utils.column_index_from_string(col_start);
        col_end_idx = openpyxl.utils.column_index_from_string(col_end)
        data = []
        for row in sheet.iter_rows(min_row=row_start, max_row=row_end, min_col=col_start_idx, max_col=col_end_idx):
            data.append([cell.value if cell.value is not None else 0 for cell in row])
        return np.array(data)
    except Exception as e:
        return np.array([])


def calculate_amplitude_and_frequency_fft(C, window_size=2):
    n = len(C)
    if n < window_size: return np.array([]), np.array([])
    frequencies, amplitudes = [], [];
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


def create_sequences(X, y, seq_length):
    xs, ys = [], [];
    for i in range(len(X) - seq_length):
        xs.append(X[i:(i + seq_length)]);
        ys.append(y[i + seq_length])
    return np.array(xs), np.array(ys)


# --- 加载并准备完整的训练数据 ---
print("\n--- 正在加载完整的训练数据 ---")
# (与之前相同)
data_sources = [{
    "file_path": "C:/Users/hs/Desktop/Training data.xlsx",  # 请确保路径正确
    "ranges": [
        ("Sheet1", "B2:B2797", "C2:C2797"), ("Sheet1", "H2:H2691", "I2:I2691"),
        ("Sheet1", "N2:N3809", "O2:O3809"), ("Sheet1", "T2:T2289", "U2:U2289"),
        ("Sheet1", "Z2:Z6328", "AA2:AA6328"), ("Sheet1", "AF2:AF2582", "AG2:AG2582"),
        ("Sheet1", "AL2:AL4265", "AM2:AM4265")
    ]
}]
all_features_list, all_h_list = [], []
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
            all_features_list.append(features);
            all_h_list.append(h[:min_len])
X_train_full = np.vstack(all_features_list)
y_train_full = np.concatenate(all_h_list)
scaler = StandardScaler()
X_train_full_scaled = scaler.fit_transform(X_train_full)
print(f"训练数据准备完成. 总样本数: {len(X_train_full)}")

# --- LSTM超参数寻优 ---
print("\n--- 正在为LSTM寻优超参数 ---")


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout_rate if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h_lstm, _ = self.lstm(x);
        out = self.fc(h_lstm[:, -1, :]);
        return out


param_grid = {
    'lr': [0.1, 0.01, 0.001],
    'hidden_size': [16, 32, 64],
    'num_layers': [1, 2, 3],
    'dropout_rate': [0.1, 0.2, 0.3],
    'sequence_length': [10, 20],
    'batch_size': [1000],  # 将batch_size也作为超参数
    'epochs': [50]
}
kf = KFold(n_splits=3, shuffle=True, random_state=42)
best_params = {}
best_score = float('inf')

# 手动网格搜索循环
for seq_len in param_grid['sequence_length']:
    print(f"\n--- 测试 Sequence Length: {seq_len} ---")
    X_seq, y_seq = create_sequences(X_train_full_scaled, y_train_full, seq_len)

    for lr in param_grid['lr']:
        for hidden_size in param_grid['hidden_size']:
            for num_layers in param_grid['num_layers']:
                for dropout in param_grid['dropout_rate']:
                    for batch_size in param_grid['batch_size']:
                        fold_scores = []
                        print(
                            f"测试LSTM参数: seq_len={seq_len}, lr={lr}, hidden={hidden_size}, layers={num_layers}, dropout={dropout}, batch={batch_size}")

                        for train_idx, val_idx in kf.split(X_seq):
                            X_train_fold_np, X_val_fold_np = X_seq[train_idx], X_seq[val_idx]
                            y_train_fold_np, y_val_fold_np = y_seq[train_idx], y_seq[val_idx]

                            # --- 关键修改点: 在交叉验证的循环内部使用DataLoader ---
                            train_fold_dataset = TensorDataset(torch.tensor(X_train_fold_np, dtype=torch.float32),
                                                               torch.tensor(y_train_fold_np, dtype=torch.float32).view(
                                                                   -1, 1))
                            train_loader = DataLoader(train_fold_dataset, batch_size=batch_size, shuffle=True)

                            # 验证集数据可以一次性放入GPU，因为它通常不大
                            X_val_tensor = torch.tensor(X_val_fold_np, dtype=torch.float32).to(device)
                            y_val_tensor = torch.tensor(y_val_fold_np, dtype=torch.float32).view(-1, 1).to(device)

                            model = LSTMModel(input_size=X_seq.shape[2], hidden_size=hidden_size, num_layers=num_layers,
                                              dropout_rate=dropout).to(device)
                            criterion = nn.MSELoss()
                            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                            # --- 关键修改点: 训练循环遍历 DataLoader ---
                            for epoch in range(param_grid['epochs'][0]):
                                model.train()
                                for X_batch, y_batch in train_loader:
                                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                                    optimizer.zero_grad()
                                    outputs = model(X_batch)
                                    loss = criterion(outputs, y_batch)
                                    loss.backward()
                                    optimizer.step()

                            model.eval()
                            with torch.no_grad():
                                val_outputs = model(X_val_tensor)
                                val_loss = F.l1_loss(val_outputs, y_val_tensor)
                                fold_scores.append(val_loss.item())

                        avg_score = np.mean(fold_scores)
                        print(f"  -> 平均验证集MAE: {avg_score:.4f}")

                        if avg_score < best_score:
                            best_score = avg_score
                            best_params = {'sequence_length': seq_len, 'lr': lr, 'hidden_size': hidden_size,
                                           'num_layers': num_layers, 'dropout_rate': dropout, 'batch_size': batch_size,
                                           'epochs': param_grid['epochs'][0]}

print("\n找到的最佳LSTM参数: ", best_params)

# # ... (后续的最终模型训练和评估部分，也应该使用最佳的batch_size，与之前的代码逻辑相同)
#
# # --- 步骤 4: 使用最佳超参数训练最终模型 ---
# print("\n--- 正在使用最佳参数训练最终LSTM模型 ---")
# X_train_final_seq, y_train_final_seq = create_sequences(X_train_full_scaled, y_train_full,
#                                                         best_params['sequence_length'])
# # --- 关键修改点 3: 将最终训练数据移动到设备 ---
# X_train_final_tensor = torch.tensor(X_train_final_seq, dtype=torch.float32).to(device)
# y_train_final_tensor = torch.tensor(y_train_final_seq, dtype=torch.float32).view(-1, 1).to(device)
#
# final_lstm_model = LSTMModel(
#     input_size=X_train_final_seq.shape[2],
#     hidden_size=best_params['hidden_size'],
#     num_layers=best_params['num_layers'],
#     dropout_rate=best_params['dropout_rate']
# ).to(device)  # 模型上GPU
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(final_lstm_model.parameters(), lr=best_params['lr'])
#
# print("在全部训练数据上训练最终模型...")
# for epoch in range(best_params['epochs']):
#     final_lstm_model.train()
#     optimizer.zero_grad()
#     outputs = final_lstm_model(X_train_final_tensor)
#     loss = criterion(outputs, y_train_final_tensor)
#     loss.backward()
#     optimizer.step()
#     if (epoch + 1) % 50 == 0:
#         print(f"最终模型训练周期 {epoch + 1}/{best_params['epochs']}, 损失: {loss.item():.4f}")
#
# # --- 步骤 5: 在您手动选择的测试集上进行评估 ---
# print("\n--- 在独立的测试集上评估最终LSTM模型性能 ---")
#
#
# def evaluate_lstm_on_test_set(data_sources, scaler, model, seq_length, plot_title="Test Set"):
#     # (此函数与之前相同，但在内部将数据移动到GPU)
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
#     X_test_full = np.vstack(all_features_list_test)
#     y_test_full = np.concatenate(all_h_list_test)
#     X_test_scaled = scaler.transform(X_test_full)
#     X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_full, seq_length)
#
#     # --- 关键修改点 4: 将测试数据移动到设备 ---
#     X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32).to(device)
#
#     model.eval()
#     with torch.no_grad():
#         predictions = model(X_test_tensor)
#         # 将结果移回CPU进行后续处理（如计算指标、绘图）
#         predictions_numpy = predictions.cpu().numpy().flatten()
#
#     # 计算指标
#     mae = mean_absolute_error(y_test_seq, predictions_numpy)
#     rmse = np.sqrt(mean_squared_error(y_test_seq, predictions_numpy))
#     r2 = r2_score(y_test_seq, predictions_numpy)
#     mape = np.mean(np.abs((y_test_seq - predictions_numpy) / y_test_seq)) * 100
#
#     print(f"--- {plot_title} 性能指标 ---")
#     print(f"平均绝对误差 (MAE): {mae:.4f}")
#     print(f"均方根误差 (RMSE): {rmse:.4f}")
#     print(f"决定系数 (R²): {r2:.4f}")
#     print(f"平均绝对百分比误差 (MAPE): {mape:.2f}%")
#
#
# # 定义您的测试数据集
# test_data_30rpm = [{"file_path": "C:/Users/hs/Desktop/Constant rates validation data.xlsx", "ranges": [("Sheet1", "B2:B6149", "C2:C6149")]}]
# test_data_60rpm = [{"file_path": "C:/Users/hs/Desktop/Constant rates validation data.xlsx", "ranges": [("Sheet1", "I2:I3652", "J2:J3652")]}]
# test_data_100rpm = [
#     {"file_path": "C:/Users/hs/Desktop/Constant rates validation data.xlsx", "ranges": [("Sheet1", "P2:P2703", "Q2:Q2703")]}]
#
# evaluate_lstm_on_test_set(test_data_30rpm, scaler, final_lstm_model, best_params['sequence_length'],
#                           plot_title="30 rpm Test Set")
# evaluate_lstm_on_test_set(test_data_60rpm, scaler, final_lstm_model, best_params['sequence_length'],
#                           plot_title="60 rpm Test Set")
# evaluate_lstm_on_test_set(test_data_100rpm, scaler, final_lstm_model, best_params['sequence_length'],
#                           plot_title="100 rpm Test Set")
#
# # --- 步骤 6: 保存模型 ---
# print("\n--- 正在保存最终LSTM模型 ---")
# torch.save(final_lstm_model.state_dict(), "final_tuned_lstm_model.pth")
# joblib.dump(scaler, "final_lstm_scaler.pkl")
# print("模型和scaler保存成功。")