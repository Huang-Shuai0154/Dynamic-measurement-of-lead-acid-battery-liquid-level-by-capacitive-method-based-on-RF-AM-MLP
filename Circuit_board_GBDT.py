# Read me
# 这是一个完整的、未经省略的程序版本。
#
# 我们使用 "梯度提升决策树" (GBDT) 模型进行预测。
#
# 【重要】:
# 1. 特征:   不使用特征扩展。直接使用 [C, dC/dt] 2维原始特征。
# 2. 模型:   不使用注意力或LSTM。
# 3. 数据:   GBDT 是静态模型，因此数据将被打乱 (Shuffle)。
# 4. 加速:   Scikit-learn (GBDT) 不支持 GPU，将自动在 CPU 上运行。

import numpy as np
import pandas as pd
import openpyxl
import re
import sys

# 【修改】导入 GBDT 和相关工具
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

# === 滤波工具 (保留) ============================================================
from scipy.signal import medfilt, savgol_filter

# --- 设置Matplotlib以支持中文黑体显示 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({
    'font.size': 20, 'axes.titlesize': 22, 'axes.labelsize': 20,
    'legend.fontsize': 16.5, 'xtick.labelsize': 20, 'ytick.labelsize': 20
})


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
    # (此函数在主循环中被调用，但其结果不用于GBDT的特征，保持不变)
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

for source in data_sources:
    file_path = source["file_path"]
    for sheet_name, c_range, h_range in source["ranges"]:
        C = read_excel_range(file_path, sheet_name, c_range).flatten()
        h = read_excel_range(file_path, sheet_name, h_range).flatten()

        # 计算特征
        delta_C_window = [C[i + 1] - C[i] for i in range(len(C) - 1)]
        frequency, amplitude = calculate_amplitude_and_frequency_fft(C, window_size=2)

        # 对齐长度
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

# 【修改】只使用2维原始特征
X = np.hstack((all_avg_C, all_delta_C_window))
y = all_h.flatten()

# --- 【重要】为GBDT打乱数据 ---
X, y = shuffle(X, y, random_state=42)
print(f"--- 数据加载完成. 总样本数: {len(X)} ---")

# --- 【修改】GBDT 训练 ---

# 1. 【重要】GBDT 推荐标准化 X 和 y (尽管树模型对缩放不敏感，但标准化y有助于收敛和评估)
print("--- 正在标准化 X (特征) 和 y (标签) ---")
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
# y 需要是 (n_samples, 1) 形状才能被 scaler 拟合
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# 2. 定义 GBDT 模型
# n_estimators: 迭代次数（树的数量）
# learning_rate: 学习率，通常越小越好但需要更多树
# max_depth: 树的深度，防止过拟合
print("--- 正在初始化 GBDT 模型 ---")
gbdt_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.01,
    max_depth=5,
    random_state=42,
    verbose=1  # 打印训练进度
)

# 3. 训练 GBDT 模型
print("--- 正在训练 GBDT 模型 (这可能需要一些时间)... ---")
# .ravel() 将 y_scaled 变回 (n_samples,) 形状
gbdt_model.fit(X_scaled, y_scaled.ravel())
print("--- GBDT 训练完成 ---")

# --- 【修改】验证部分 ---

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
        ]
    }
]
validation_data_30 = [
    {
        "file_path": "C:/Users/hs/Desktop/Sensor.xlsx",
        "ranges": [
            ("Sheet3", "S1:S4074", "V1:V4074"),
        ]
    }
]


def plot_predictions_comparison(y_true, predictions, title, file_name):
    plt.figure(figsize=(10, 7.5))
    plt.plot(y_true, label='True Values', color='black', linewidth=2)
    plt.plot(predictions, label='Predictions (GBDT)', color='blue', linestyle='--', alpha=0.7)
    plt.title(f"Predictions vs True Values: {title}", fontsize=22)
    plt.xlabel('Sample Number', fontsize=20)
    plt.ylabel('Liquid Level (mm)', fontsize=20)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()
    print(f"{title} - Predictions vs True Values plot saved as {file_name}")


# 【关键修改】process_validation_data 流程
def process_validation_data(data_sources,
                            scaler_X,  # <--- 原始特征 Scaler
                            scaler_y,  # <--- 标签 Scaler
                            model,  # <--- GBDT 模型
                            plot_title="Validation"):
    # GBDT 是 sklearn 模型，不需要 .eval()

    # 1. 加载和特征工程 (与训练时相同)
    all_avg_C, all_delta_C_window, all_h = [], [], []
    for source in data_sources:
        file_path = source["file_path"]
        for sheet_name, c_range, h_range in source["ranges"]:
            C = read_excel_range(file_path, sheet_name, c_range).flatten()
            h = read_excel_range(file_path, sheet_name, h_range).flatten()
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
    y_validation = all_h.flatten()  # 真实的、未缩放的 Y

    if len(X_validation) == 0:
        print(f"警告: {plot_title} 数据不足。")
        return 0, 0, 0, 0, 0

    # --- 开始 GBDT 预测流程 ---

    # 1. 标准化 *原始* 验证数据
    X_validation_scaled = scaler_X.transform(X_validation)

    # 2. 用 GBDT 模型预测（得到的是 *缩放后* 的 y）
    y_pred_scaled = model.predict(X_validation_scaled)

    # 3. 【重要】将预测结果反标准化，还原为真实尺度
    validation_predictions = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
    validation_predictions = validation_predictions.flatten()  # 变为 1D

    # 4. 真实值 (y_validation) 不需要反标准化
    y_true = y_validation

    # 绘图和指标计算
    plot_predictions_comparison(y_true, validation_predictions, title=plot_title,
                                file_name=f"{plot_title}_comparison_GBDT.png")

    mse_validation = mean_squared_error(y_true, validation_predictions)
    rmse_validation = np.sqrt(mse_validation)
    mae_validation = mean_absolute_error(y_true, validation_predictions)
    r2_validation = r2_score(y_true, validation_predictions)
    # 避免 y_true 中有 0
    y_true_safe = np.where(y_true == 0, 1e-6, y_true)
    mape_validation = np.mean(np.abs((y_true - validation_predictions) / y_true_safe)) * 100

    return mse_validation, rmse_validation, mae_validation, r2_validation, mape_validation


# --- 【修改】评估调用 ---
print("\n=== 验证集评估（30转） ===")
mse_30, rmse_30, mae_30, r2_30, mape_30 = process_validation_data(
    validation_data_30, scaler_X, scaler_y, gbdt_model, plot_title="30rpm Sulfuric Acid"
)
print(f"均方误差 (MSE): {mse_30:.4f}")
print(f"均方根误差 (RMSE): {rmse_30:.4f}")
print(f"平均绝对误差 (MAE): {mae_30:.4f}")
print(f"决定系数 (R²): {r2_30:.4f}")
print(f"平均绝对百分比误差 (MAPE): {mape_30:.2f}%")

print("\n=== 验证集评估（60转） ===")
mse_60, rmse_60, mae_60, r2_60, mape_60 = process_validation_data(
    validation_data_60, scaler_X, scaler_y, gbdt_model, plot_title="60rpm Sulfuric Acid"
)
print(f"均方误差 (MSE): {mse_60:.4f}")
print(f"均方根误差 (RMSE): {rmse_60:.4f}")
print(f"平均绝对误差 (MAE): {mae_60:.4f}")
print(f"决定系数 (R²): {r2_60:.4f}")
print(f"平均绝对百分比误差 (MAPE): {mape_60:.2f}%")

print("\n=== 验证集评估（100转） ===")
mse_100, rmse_100, mae_100, r2_100, mape_100 = process_validation_data(
    validation_data_100, scaler_X, scaler_y, gbdt_model, plot_title="100rpm Sulfuric Acid"
)
print(f"均方误差 (MSE): {mse_100:.4f}")
print(f"均方根误差 (RMSE): {rmse_100:.4f}")
print(f"平均绝对误差 (MAE): {mae_100:.4f}")
print(f"决定系数 (R²): {r2_100:.4f}")
print(f"平均绝对百分比误差 (MAPE): {mape_100:.2f}%")

# --- 【修改】保存模型 ---
print("\n--- 正在保存所有模型和Scalers ---")
joblib.dump(scaler_X, "Scaler_X_GBDT.pkl")
joblib.dump(scaler_y, "Scaler_y_GBDT.pkl")
joblib.dump(gbdt_model, "Final_GBDT_model.pkl")

print("所有组件保存成功！")