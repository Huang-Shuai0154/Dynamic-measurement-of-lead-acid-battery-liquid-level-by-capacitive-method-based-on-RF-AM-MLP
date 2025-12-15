import openpyxl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

# --- 1. 全局设置 ---

# --- 设备检测 ---
# 检查是否有可用的CUDA GPU，如果有就用GPU，否则用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(f"--- 将使用设备 (Using device): {device} ---")

# --- 手动设置超参数 ---
# 您可以在这里手动调整所有参数，然后运行整个脚本来训练和评估模型
SEQUENCE_LENGTH = 20
HIDDEN_SIZE = 64
NUM_LAYERS = 1
DROPOUT_RATE = 0.3
LEARNING_RATE = 0.01
EPOCHS = 200
BATCH_SIZE = 500  # 设置批量大小

# --- 全局绘图字体设置 ---
plt.rcParams.update({
    'font.size': 20, 'axes.titlesize': 22, 'axes.labelsize': 20,
    'legend.fontsize': 16.5, 'xtick.labelsize': 20, 'ytick.labelsize': 20
})


# --- 2. 辅助函数定义 ---

def read_excel_range(file_path, sheet_name, range_string):
    """从Excel文件中读取指定范围的数据。"""
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
    """计算振幅和频率特征。"""
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


def create_sequences(X, y, seq_length):
    """将数据转换为LSTM所需的序列格式。"""
    xs, ys = [], []
    for i in range(len(X) - seq_length):
        xs.append(X[i:(i + seq_length)])
        ys.append(y[i + seq_length])
    return np.array(xs), np.array(ys)


def process_data_and_get_features(data_sources):
    """
    一个通用的数据处理函数，它会读取所有数据源，进行特征工程，
    并返回一个完整的特征矩阵(X)和对应的真实标签(y)。
    """
    all_features_list, all_h_list = [], []
    for source in data_sources:
        for sheet_name, c_range, h_range in source["ranges"]:
            C = read_excel_range(source["file_path"], sheet_name, c_range).flatten()
            h = read_excel_range(source["file_path"], sheet_name, h_range).flatten()

            if len(C) < 2:
                continue

            # 特征工程: 计算电容变化率、振幅和频率
            delta_C = np.diff(C, prepend=C[0])
            frequency, amplitude = calculate_amplitude_and_frequency_fft(C, window_size=2)

            # 对齐所有数据数组的长度
            min_len = min(len(C), len(h), len(delta_C), len(frequency), len(amplitude))

            if min_len > 0:
                # 构建特征矩阵
                features = np.vstack([
                    C[:min_len],
                    delta_C[:min_len],
                ]).T
                all_features_list.append(features)
                all_h_list.append(h[:min_len])

    # 将所有数据片段合并成一个大的Numpy数组
    X_full = np.vstack(all_features_list)
    y_full = np.concatenate(all_h_list)

    return X_full, y_full

def plot_test_results(y_true, y_pred, title, filename):
    """绘制并保存真实值与预测值的对比图。"""
    plt.figure(figsize=(12, 8))
    plt.plot(y_true, label='True Value', color='black', linewidth=2.5, alpha=0.8)
    plt.plot(y_pred, label='LSTM Prediction', color='red', linestyle='--', linewidth=2, alpha=0.9)
    plt.title(f'Test Set: Prediction vs. True Value ({title})', fontsize=20)
    plt.xlabel('Sample Index', fontsize=16)
    plt.ylabel('Liquid Level (mm)', fontsize=16)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()  # 关闭图形，防止在循环中重复显示
    print(f"对比图已保存至: {filename}")


# --- 3. LSTM模型类定义 ---
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout_rate if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        out = self.fc(h_lstm[:, -1, :])
        return out


# --- 4. 评估函数定义 ---
def evaluate_lstm_on_test_set(data_sources, scaler, model, seq_length, plot_title="Test Set"):
    model.eval()  # 确保模型在评估模式
    device = next(model.parameters()).device  # 获取模型所在的设备

    # 加载和处理测试数据
    all_features_list_test, all_h_list_test = [], []
    for source in data_sources:
        for sheet_name, c_range, h_range in source["ranges"]:
            C = read_excel_range(source["file_path"], sheet_name, c_range).flatten()
            h = read_excel_range(source["file_path"], sheet_name, h_range).flatten()
            if len(C) < 2: continue
            delta_C = np.diff(C, prepend=C[0])
            frequency, amplitude = calculate_amplitude_and_frequency_fft(C, window_size=2)
            min_len = min(len(C), len(h), len(delta_C), len(frequency), len(amplitude))
            if min_len > 0:
                features = np.vstack([C[:min_len], delta_C[:min_len]]).T
                all_features_list_test.append(features)
                all_h_list_test.append(h[:min_len])

    if not all_features_list_test:
        print(f"警告: 在 {plot_title} 中没有找到有效数据。")
        return

    X_test_full = np.vstack(all_features_list_test)
    y_test_full = np.concatenate(all_h_list_test)
    X_test_scaled = scaler.transform(X_test_full)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_full, seq_length)

    # 为防止测试集过大也导致显存溢出，同样使用DataLoader
    test_dataset = TensorDataset(torch.tensor(X_test_seq, dtype=torch.float32))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    all_predictions = []
    with torch.no_grad():
        for X_batch in test_loader:
            X_batch = X_batch[0].to(device)  # DataLoader返回的是一个元组
            predictions_batch = model(X_batch)
            all_predictions.append(predictions_batch.cpu())

    predictions_numpy = torch.cat(all_predictions).numpy().flatten()

    # 绘图
    plot_filename = f"test_comparison_{plot_title.replace(' ', '_')}.png"
    plot_test_results(y_test_seq, predictions_numpy, plot_title, plot_filename)

    # 计算指标
    mae = mean_absolute_error(y_test_seq, predictions_numpy)
    rmse = np.sqrt(mean_squared_error(y_test_seq, predictions_numpy))
    r2 = r2_score(y_test_seq, predictions_numpy)
    mape = np.mean(np.abs((y_test_seq - predictions_numpy) / y_test_seq)) * 100

    print(f"--- {plot_title} 性能指标 ---")
    print(f"平均绝对误差 (MAE): {mae:.4f}")
    print(f"均方根误差 (RMSE): {rmse:.4f}")
    print(f"决定系数 (R²): {r2:.4f}")
    print(f"平均绝对百分比误差 (MAPE): {mape:.2f}%")


# --- 5. 主执行流程 ---
def main():
    # 加载并准备训练数据
    print("\n--- 正在加载完整的训练数据 ---")

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
    X_train, y_train = process_data_and_get_features(data_sources)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    print(f"训练数据准备完成. 总样本数: {len(X_train)}")

    # 准备小批量训练数据
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, SEQUENCE_LENGTH)
    train_dataset = TensorDataset(torch.tensor(X_train_seq, dtype=torch.float32),
                                  torch.tensor(y_train_seq, dtype=torch.float32).view(-1, 1))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 初始化模型并移至GPU
    lstm_model = LSTMModel(
        input_size=X_train_seq.shape[2],
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout_rate=DROPOUT_RATE
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=LEARNING_RATE)

    # 训练模型
    print("\n--- 正在训练LSTM模型 ---")
    for epoch in range(EPOCHS):
        lstm_model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = lstm_model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 50 == 0:
            print(f"训练周期 {epoch + 1}/{EPOCHS}, 平均损失: {epoch_loss / len(train_loader):.4f}")

    # 在独立的测试集上评估模型
    print("\n--- 正在独立的测试集上评估模型性能 ---")

    test_data_30rpm = [{"file_path": "C:/Users/hs/Desktop/Sensor.xlsx",
                        "ranges": [("Sheet3", "S1:S4074", "V1:V4074")]}]
    test_data_60rpm = [{"file_path": "C:/Users/hs/Desktop/Sensor.xlsx",
                        "ranges": [("Sheet2", "G1:G2041", "J1:J2041")]}]
    test_data_100rpm = [
        {"file_path": "C:/Users/hs/Desktop/Sensor.xlsx",
         "ranges": [("Sheet4", "S1:S1266", "V1:V1266")]}]

    evaluate_lstm_on_test_set(test_data_30rpm, scaler, lstm_model, SEQUENCE_LENGTH, plot_title="Slow rate Test Set")
    evaluate_lstm_on_test_set(test_data_60rpm, scaler, lstm_model, SEQUENCE_LENGTH, plot_title="Medium rate Test Set")
    evaluate_lstm_on_test_set(test_data_100rpm, scaler, lstm_model, SEQUENCE_LENGTH, plot_title="Fast rate Test Set")

    # 保存模型
    print("\n--- 正在保存最终模型 ---")
    torch.save(lstm_model.state_dict(), "Circuit_board_lstm_model.pth")
    joblib.dump(scaler, "Circuit_board_lstm_scaler.pkl")
    print("模型和scaler保存成功。")
    # torch.save(lstm_model.state_dict(), "Circuit_board_lstm_model1.pth")
    # joblib.dump(scaler, "Circuit_board_lstm_scaler1.pkl")
    # print("模型和scaler保存成功。")


if __name__ == "__main__":
    # 运行主函数
    main()