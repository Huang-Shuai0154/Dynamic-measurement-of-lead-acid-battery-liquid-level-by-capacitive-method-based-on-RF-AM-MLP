# Read me
# 这是一个完整的、经过修改的程序版本。
#


import numpy as np
import pandas as pd
import openpyxl
import re
import sys

# 导入 PolynomialFeatures
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
# 导入 DataLoader
from torch.utils.data import TensorDataset, DataLoader
import seaborn as sns

# 【新增】导入中值滤波库
from scipy.ndimage import median_filter

# === 1. GPU 设备检测 ==============================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- 启动程序 (Poly-LSTM 无注意力机制 + 中值滤波版) ---")
print(f"--- 将使用设备 (Using device): {device} ---")


# === 辅助函数 =======================================================
def read_excel_range(file_path, sheet_name, range_string):
    try:
        match = re.match(r"([A-Z]+)(\d+):([A-Z]+)(\d+)", range_string)
        if not match:
            raise ValueError(f"Invalid range format: {range_string}")
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
        print(f"Error reading Excel range: {e}")
        sys.exit(1)


def calculate_amplitude_and_frequency_fft(C, window_size=2, sample_rate=1.0):
    n = len(C)
    frequencies = []
    amplitudes = []
    for i in range(0, n - window_size + 1):
        window_data = C[i:i + window_size]
        fft_vals = np.fft.fft(window_data)
        freqs = np.fft.fftfreq(window_size, d=sample_rate)
        positive_freqs = freqs[:window_size // 2]
        positive_fft_vals = fft_vals[:window_size // 2]
        amplitude = np.abs(positive_fft_vals)
        peak_freq = positive_freqs[np.argmax(amplitude)] if len(amplitude) > 0 else 0
        peak_amplitude = np.max(amplitude) if len(amplitude) > 0 else 0
        frequencies.append(peak_freq)
        amplitudes.append(peak_amplitude)
    return np.array(frequencies), np.array(amplitudes)


def create_sequences(X, y, seq_length):
    """将数据转换为LSTM所需的序列格式。"""
    xs, ys = [], []
    for i in range(len(X) - seq_length):
        xs.append(X[i:(i + seq_length)])
        ys.append(y[i + seq_length])
    return np.array(xs), np.array(ys)


# === 数据加载与预处理 ==============================================

data_sources = [{
    "file_path": "C:/Users/hs/Desktop/Sensor.xlsx",  # 请确保路径正确
    "ranges": [
        ("Sheet4", "A1:A1223", "D1:D1223"),
        ("Sheet4", "G1:G1226", "J1:J1226"),
        ("Sheet4", "M1:M1291", "P1:P1291"),
        ("Sheet2", "A1:A2018", "D1:D2018"),
        # ("Sheet2", "G94:G2134", "J94:J2134"),
        ("Sheet2", "M1:M2001", "P1:P2001"),
        ("Sheet2", "S1:S1132", "V1:V1132"),
        ("Sheet3", "A1:A4385", "D1:D4385"),
        ("Sheet3", "G1:G4002", "J1:J4002"),
        ("Sheet3", "M1:M3976", "P1:P3976"),
        # ("Sheet3", "S78:S4300", "V78:V4300"),

    ]
}]
all_amplitude = []
all_frequency = []
all_avg_C = []
all_delta_C_window = []
all_h = []

print("--- 开始加载并预处理训练数据 ---")
# 按时间顺序加载所有数据片段
for source in data_sources:
    file_path = source["file_path"]
    for sheet_name, c_range, h_range in source["ranges"]:
        C = read_excel_range(file_path, sheet_name, c_range).flatten()
        h = read_excel_range(file_path, sheet_name, h_range).flatten()


        delta_C_window = [C[i + 1] - C[i] for i in range(len(C) - 1)]
        frequency, amplitude = calculate_amplitude_and_frequency_fft(C, window_size=2)
        min_len = min(len(C), len(delta_C_window), len(amplitude), len(frequency), len(h))
        C, delta_C_window, amplitude, frequency, h = C[:min_len], delta_C_window[:min_len], amplitude[
                                                                                            :min_len], frequency[
                                                                                                       :min_len], h[
                                                                                                                  :min_len]

        all_avg_C.extend(C)
        all_delta_C_window.extend(delta_C_window)
        all_amplitude.extend(amplitude)
        all_frequency.extend(frequency)
        all_h.extend(h)

all_avg_C = np.array(all_avg_C).reshape(-1, 1)
all_delta_C_window = np.array(all_delta_C_window).reshape(-1, 1)
all_h = np.array(all_h)

X = np.hstack((all_avg_C, all_delta_C_window))
y = all_h.flatten()

# --- 使用多项式特征进行特征转换 ---

# 1. 在 *全部* X 数据上拟合原始 Scaler
print("--- 正在标准化原始输入特征 [C, dC/dt] ---")
scaler_X_original = StandardScaler()
X_scaled = scaler_X_original.fit_transform(X)

# 2. 在 *全部* X_scaled 数据上拟合多项式转换器
N_FEATURES = 10
DEGREE = 3
print(f"--- 正在应用3次多项式 (2维 -> {N_FEATURES}维) ---")
poly_transformer = PolynomialFeatures(degree=DEGREE, include_bias=True)
X_poly = poly_transformer.fit_transform(X_scaled)

# 3. 在 *全部* X_poly 数据上拟合第二个 Scaler
print("--- 正在标准化多项式特征 ---")
scaler_poly = StandardScaler()
X_poly_scaled = scaler_poly.fit_transform(X_poly)

# --- 创建时间序列 ---
SEQUENCE_LENGTH = 20  # 序列长度
BATCH_SIZE = 512  # 批量大小
EPOCHS = 200  # 训练周期
LEARNING_RATE = 0.01  # 学习率
HIDDEN_SIZE = 64  # LSTM 隐藏层大小
NUM_LAYERS = 1  # LSTM 层数
DROPOUT_RATE = 0.3  # Dropout 率

print(f"--- 正在创建长度为 {SEQUENCE_LENGTH} 的时间序列 ---")
X_seq, y_seq = create_sequences(X_poly_scaled, y, SEQUENCE_LENGTH)

# 转换为Tensors并创建DataLoader
X_train_tensor = torch.tensor(X_seq, dtype=torch.float32)
y_train_tensor = torch.tensor(y_seq, dtype=torch.float32).view(-1, 1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


# --- 【修改】定义纯 LSTM 模型 (移除 Attention) ---

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,  # 输入形状 (Batch, SeqLen, Features)
            dropout=dropout_rate if num_layers > 1 else 0
        )
        # 移除了 AttentionMechanism
        self.fc = nn.Linear(hidden_size, 1)  # 最终预测层

    def forward(self, x):
        # x 形状: (Batch, SeqLen, InputSize)

        # lstm_out 形状: (Batch, SeqLen, HiddenSize)
        lstm_out, _ = self.lstm(x)

        # 【修改】不使用 Attention，直接取最后一个时间步的输出
        # 取序列中最后一个时间步的 Hidden State 作为上下文特征
        # last_step_out 形状: (Batch, HiddenSize)
        last_step_out = lstm_out[:, -1, :]

        # 从最后一个时间步进行最终预测
        # out 形状: (Batch, 1)
        out = self.fc(last_step_out)
        return out


# --- 训练模型的循环 (使用DataLoader) ---

input_dim = X_seq.shape[2]  # 10

# 将模型实例化并发送到GPU
model = LSTMModel(
    input_size=input_dim,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout_rate=DROPOUT_RATE
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"--- 开始训练 Standard LSTM (Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}) ---")

model.train()  # 设置模型为训练模式
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    for X_batch, y_batch in train_loader:
        # 将数据批量发送到GPU
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # 前向传播
        optimizer.zero_grad()
        # 【修改】模型现在只返回预测值
        outputs = model(X_batch)

        # 计算损失
        loss = criterion(outputs, y_batch)

        # 反向传播
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}/{EPOCHS}, 平均损失: {epoch_loss / len(train_loader):.4f}")

print("--- 训练完成 ---")

# --- 验证部分 ---

# 验证集数据源
print("--- 警告：验证集与训练数据分离 ---")
validation_data_100 = [
    {
        "file_path": "C:/Users/hs/Desktop/Sensor.xlsx",
        "ranges": [
            ("Sheet4", "S1:S1266", "V1:V1266"),
        ]
    }
]
validation_data_60 = [
    {
        "file_path": "C:/Users/hs/Desktop/Sensor.xlsx",
        "ranges": [
            ("Sheet2", "G1:G2041", "J1:J2041"),
            # ("Sheet2", "M41:M2041", "P41:P2041"),
        ]
    }
]
validation_data_30 = [
    {
        "file_path": "C:/Users/hs/Desktop/Sensor.xlsx",
        "ranges": [
            ("Sheet3", "S1:S4074", "V1:V4074"),
            # ("Sheet3", "M176:M4151", "P176:P4151"),
        ]
    }
]


def plot_predictions_comparison(y_true, predictions, title, file_name):
    plt.figure(figsize=(10, 7.5))
    plt.plot(y_true, label='True Values', color='black', linewidth=2)
    plt.plot(predictions, label='Predictions (LSTM)', color='red', linestyle='--', alpha=0.7)
    plt.title(f"Predictions vs True Values: {title}", fontsize=22)
    plt.xlabel('Sample Number (Time Steps)', fontsize=20)
    plt.ylabel('Liquid Level (mm)', fontsize=20)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()
    print(f"{title} - Predictions vs True Values plot saved as {file_name}")


# 【注意】此函数保留以防调用报错，但由于没有attention_scores传入，实际不会执行绘图
def plot_attention_heatmap(attention_scores, y_true, y_pred, seq_length, title, filename):
    print("提示: 模型中未包含注意力机制，跳过热图绘制。")
    return


# 【修改】process_validation_data 流程 (移除 Attention 收集)
def process_validation_data(data_sources,
                            scaler_X_original,
                            poly_transformer,
                            scaler_poly,
                            model,
                            seq_length,
                            plot_title="Validation"):
    model.eval()  # 设置模型为评估模式

    # 1. 加载和特征工程 (与训练时相同)
    all_avg_C, all_delta_C_window, all_h = [], [], []
    for source in data_sources:
        file_path = source["file_path"]
        for sheet_name, c_range, h_range in source["ranges"]:
            C = read_excel_range(file_path, sheet_name, c_range).flatten()
            h = read_excel_range(file_path, sheet_name, h_range).flatten()

            # ===============================================================
            # 【新增】预处理滤波：对验证集原始电容数据 C 进行同样的降噪
            # ===============================================================
            try:
                if len(C) > 5:
                    C = median_filter(C, size=5, mode='nearest')
            except Exception as e:
                print(f"验证数据滤波失败: {e}，保持原样")
            # ===============================================================

            delta_C_window = [C[i + 1] - C[i] for i in range(len(C) - 1)]
            min_len = min(len(C), len(delta_C_window), len(h))
            C, delta_C_window, h = C[:min_len], delta_C_window[:min_len], h[:min_len]
            all_avg_C.extend(C)
            all_delta_C_window.extend(delta_C_window)
            all_h.extend(h)

    all_avg_C = np.array(all_avg_C).reshape(-1, 1)
    all_delta_C_window = np.array(all_delta_C_window).reshape(-1, 1)
    all_h = np.array(all_h)

    X_validation = np.hstack((all_avg_C, all_delta_C_window))
    y_validation = all_h.flatten()

    # --- 开始四阶段转换 ---

    # 1. 标准化 *原始* 验证数据
    X_validation_scaled = scaler_X_original.transform(X_validation)

    # 2. 用多项式转换器 *转换* 验证数据
    X_validation_poly = poly_transformer.transform(X_validation_scaled)

    # 3. 标准化 *转换后* 的特征
    X_validation_poly_scaled = scaler_poly.transform(X_validation_poly)

    # 4. 创建验证序列
    X_val_seq, y_val_seq = create_sequences(X_validation_poly_scaled, y_validation, seq_length)

    if len(X_val_seq) == 0:
        print(f"警告: {plot_title} 数据不足，无法创建序列。")
        return 0, 0, 0, 0, 0, None, None, None

    # 5. 送入 LSTM 模型
    X_val_tensor = torch.tensor(X_val_seq, dtype=torch.float32).to(device)

    all_predictions = []
    # 移除了 all_attn_scores

    # 使用DataLoader进行批量预测
    val_dataset = TensorDataset(X_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    with torch.no_grad():
        for X_batch in val_loader:
            # 【修改】模型只返回预测值
            outputs = model(X_batch[0])
            all_predictions.append(outputs.cpu())

    validation_predictions = torch.cat(all_predictions).numpy().flatten()
    # validation_attn_scores 设为 None
    validation_attn_scores = None

    # 真实值 (y) 必须与预测值对齐
    y_true = y_val_seq

    # 绘图和指标计算
    plot_predictions_comparison(y_true, validation_predictions, title=plot_title,
                                file_name=f"{plot_title}_comparison.png")

    mse_validation = mean_squared_error(y_true, validation_predictions)
    rmse_validation = np.sqrt(mse_validation)
    mae_validation = mean_absolute_error(y_true, validation_predictions)
    r2_validation = r2_score(y_true, validation_predictions)
    # 避免 y_true 中有 0
    y_true_safe = np.where(y_true == 0, 1e-6, y_true)
    mape_validation = np.mean(np.abs((y_true - validation_predictions) / y_true_safe)) * 100

    # 【修改】返回 None 作为 attention scores
    return mse_validation, rmse_validation, mae_validation, r2_validation, mape_validation, \
           y_true, validation_predictions, validation_attn_scores


# --- 评估调用 ---
print("\n=== 验证集评估（30转） ===")
mse_30, rmse_30, mae_30, r2_30, mape_30, \
y_true_30, y_pred_30, attn_30 = process_validation_data(
    validation_data_30, scaler_X_original, poly_transformer, scaler_poly, model, SEQUENCE_LENGTH,
    plot_title="30rpm Sulfuric Acid"
)
print(f"均方误差 (MSE): {mse_30:.4f}")
print(f"均方根误差 (RMSE): {rmse_30:.4f}")
print(f"平均绝对误差 (MAE): {mae_30:.4f}")
print(f"决定系数 (R²): {r2_30:.4f}")
print(f"平均绝对百分比误差 (MAPE): {mape_30:.2f}%")
# 由于 attn_30 是 None，这里不会绘图
if attn_30 is not None:
    plot_attention_heatmap(attn_30, y_true_30, y_pred_30, SEQUENCE_LENGTH,
                           "30rpm Sulfuric Acid", "attention_30rpm.png")

print("\n=== 验证集评估（60转） ===")
mse_60, rmse_60, mae_60, r2_60, mape_60, \
y_true_60, y_pred_60, attn_60 = process_validation_data(
    validation_data_60, scaler_X_original, poly_transformer, scaler_poly, model, SEQUENCE_LENGTH,
    plot_title="60rpm Sulfuric Acid"
)
print(f"均方误差 (MSE): {mse_60:.4f}")
print(f"均方根误差 (RMSE): {rmse_60:.4f}")
print(f"平均绝对误差 (MAE): {mae_60:.4f}")
print(f"决定系数 (R²): {r2_60:.4f}")
print(f"平均绝对百分比误差 (MAPE): {mape_60:.2f}%")
if attn_60 is not None:
    plot_attention_heatmap(attn_60, y_true_60, y_pred_60, SEQUENCE_LENGTH,
                           "60rpm Sulfuric Acid", "attention_60rpm.png")

print("\n=== 验证集评估（100转） ===")
mse_100, rmse_100, mae_100, r2_100, mape_100, \
y_true_100, y_pred_100, attn_100 = process_validation_data(
    validation_data_100, scaler_X_original, poly_transformer, scaler_poly, model, SEQUENCE_LENGTH,
    plot_title="100rpm Sulfuric Acid"
)
print(f"均方误差 (MSE): {mse_100:.4f}")
print(f"均方根误差 (RMSE): {rmse_100:.4f}")
print(f"平均绝对误差 (MAE): {mae_100:.4f}")
print(f"决定系数 (R²): {r2_100:.4f}")
print(f"平均绝对百分比误差 (MAPE): {mape_100:.2f}%")
if attn_100 is not None:
    plot_attention_heatmap(attn_100, y_true_100, y_pred_100, SEQUENCE_LENGTH,
                           "100rpm Sulfuric Acid", "attention_100rpm.png")

# # --- 保存模型 ---
# print("\n--- G:/My Drive/")
# joblib.dump(scaler_X_original, "Scaler_Original_X_PolyLSTM.pkl")
# joblib.dump(poly_transformer, "Polynomial_Transformer_model_PolyLSTM.pkl")
# joblib.dump(scaler_poly, "Scaler_Poly_X_PolyLSTM.pkl")
# torch.save(model.state_dict(), "Final_LSTM_model_PolyLSTM.pth")

# --- 保存模型 ---
print("\n--- G:/My Drive/")
joblib.dump(scaler_X_original, "Scaler_Original_X_PolyLSTM1.pkl")
joblib.dump(poly_transformer, "Polynomial_Transformer_model_PolyLSTM1.pkl")
joblib.dump(scaler_poly, "Scaler_Poly_X_PolyLSTM1.pkl")
torch.save(model.state_dict(), "Final_LSTM_model_PolyLSTM1.pth")

print("所有组件保存成功！")