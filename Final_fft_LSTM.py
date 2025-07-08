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

