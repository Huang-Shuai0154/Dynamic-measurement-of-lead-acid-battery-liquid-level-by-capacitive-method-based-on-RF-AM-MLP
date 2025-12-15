# Read me
# 这是一个完整的评估与绘图程序版本。
#
# 【修改说明】:
# 1. 【绘图修改】: 在 plot_predictions_zoom (缩略图) 函数中，移除了 X轴名称、Y轴名称和图例，仅保留刻度。
# 2. 【保留】所有模型 (MLP, LSTM, SVR, Poly-LSTM, GBDT) 和预处理逻辑不变。

import openpyxl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import sys
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor  # 导入GBDT
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

# 仍然需要导入 median_filter 用于输入端处理
from scipy.ndimage import median_filter

# --- 设置Matplotlib以支持中文黑体显示 ---
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

# --- 全局绘图和配置设置 ---
plt.rcParams.update({
    'font.size': 20, 'axes.titlesize': 22, 'axes.labelsize': 20,
    'legend.fontsize': 16.5, 'xtick.labelsize': 20, 'ytick.labelsize': 20
})

PLOT_CONFIG = {
    "enable_prediction_plot": True,
    "enable_residual_plot": True,
    "enable_zoom_plot": True,
    "enable_attention_heatmap": False,
    "zoom_ranges": {
        "慢速": (114, 134),
        "中速": (114, 134),
        "快速": (60, 80),
        "5℃": (380, 400),
        "20℃": (670, 690),
        "25℃": (695, 715),
        "15℃": (420, 440),
        "slow rate": (3140, 3160),
        "medium rate": (60, 80),
        "fast rate": (670, 690),
        "variable rate": (60, 80),
    }
}


# --- 1. 辅助函数定义 ---
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
        print(f"读取Excel文件 '{file_path}' 的工作表 '{sheet_name}' 时出错: {e}")
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


def create_sequences(X, y, seq_length):
    xs, ys = [], [];
    for i in range(len(X) - seq_length):
        xs.append(X[i:(i + seq_length)]);
        ys.append(y[i + seq_length])
    return np.array(xs), np.array(ys)


def process_data_and_get_features(data_sources):
    """
    读取数据并进行特征提取。
    保留输入端的滤波，以确保特征提取的质量。
    """
    all_features_list, all_h_list = [], []
    for source in data_sources:
        for sheet_name, c_range, h_range in source["ranges"]:
            C = read_excel_range(source["file_path"], sheet_name, c_range).flatten()
            h = read_excel_range(source["file_path"], sheet_name, h_range).flatten()
            if len(C) < 2: continue


            delta_C = [C[i + 1] - C[i] for i in range(len(C) - 1)]
            frequency, amplitude = calculate_amplitude_and_frequency_fft(C, window_size=2)
            min_len = min(len(C), len(h), len(delta_C), len(frequency), len(amplitude))
            if min_len > 0:
                features = np.vstack([C[:min_len], delta_C[:min_len]]).T
                all_features_list.append(features)
                all_h_list.append(h[:min_len])

    if not all_features_list:
        return np.array([]), np.array([])

    X_full = np.vstack(all_features_list)
    y_full = np.concatenate(all_h_list)
    return X_full, y_full


def evaluate_model_metrics(y_true, y_pred, model_name):
    mask = y_true != 0
    y_true_safe = y_true[mask]
    y_pred_safe = y_pred[mask]

    mae = mean_absolute_error(y_true, y_pred);
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred);
    mape = np.mean(np.abs((y_true_safe - y_pred_safe) / y_true_safe)) * 100 if len(y_true_safe) > 0 else float('nan')

    print(f"--- {model_name} 性能指标 ---")
    print(f"平均绝对误差 (MAE): {mae:.4f}");
    print(f"均方根误差 (RMSE): {rmse:.4f}")
    print(f"决定系数 (R²): {r2:.4f}");
    print(f"平均绝对百分比误差 (MAPE): {mape:.4f}%\n")


# --- 2. 模型类定义 ---
class MLPWithoutAttention(nn.Module):
    def __init__(self, input_dim):
        super(MLPWithoutAttention, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        return x, None


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


class LSTMModel_Poly(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate):
        super(LSTMModel_Poly, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout_rate if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_step_out = lstm_out[:, -1, :]
        out = self.fc(last_step_out)
        return out


# --- 3. 绘图函数 ---
# 【修改】添加 GBDT 的颜色
model_color = {
    "MLP": "orange",
    "LSTM": "red",
    # "SVR": "green",
    "Poly-LSTM": "blue",
    "GBDT": "green"  # GBDT 颜色
}


def plot_predictions_full(y_true, predictions_dict, title):
    print(f"绘制 {title} 的完整预测对比图...")
    time_values = np.arange(len(y_true)) * 0.20
    fig, ax = plt.subplots(figsize=(10, 7.5))

    ax.plot(time_values, y_true, label='True Value', color='black', linewidth=4, zorder=10)

    for model_name, preds in predictions_dict.items():
        if model_name in model_color:
            ax.plot(time_values, preds, label=model_name, color=model_color[model_name], linewidth=4, alpha=0.8)

    ax.set_title(f"Liquid level descent at {title}", fontsize=20)
    ax.set_xlabel('Time (s)', fontsize=20);
    ax.set_ylabel('Liquid level (mm)', fontsize=20)

    # ==========================================
    # 【修改】手动设置 Y 轴刻度最大值
    # ==========================================
    ax.set_ylim(top=420)
    # ==========================================

    ax.legend(fontsize=20, loc='lower left');
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid(True);
    plt.tight_layout();
    plt.savefig(f"{title}_full_comparison.png");
    plt.close(fig)
    print(f"图表已保存至: {title}_full_comparison.png")


def plot_predictions_zoom(y_true, predictions_dict, title, start_sample, end_sample):
    print(f"绘制 {title} 的局部放大对比图 (样本点 {start_sample}-{end_sample})...")
    sample_indices = np.arange(len(y_true))
    time_values = sample_indices * 0.20
    mask = (sample_indices >= start_sample) & (sample_indices <= end_sample)
    if not np.any(mask):
        print(f"警告: 在 {title} 中找不到样本范围 {start_sample}-{end_sample} 的数据。")
        return
    fig, ax = plt.subplots(figsize=(4, 3))

    ax.plot(time_values[mask], y_true[mask], label='True Value', color='black', linewidth=4, zorder=10)

    for model_name, preds in predictions_dict.items():
        if model_name in model_color:
            ax.plot(time_values[mask], preds[mask], label=model_name, color=model_color[model_name], linewidth=4,
                    alpha=0.8)

    # ==========================================
    # 【修改】移除 X轴、Y轴标签和图例
    # ==========================================
    # ax.set_xlabel('Time (s)', fontsize=20);       # <--- 已移除
    # ax.set_ylabel('Liquid level (mm)', fontsize=20) # <--- 已移除
    # ax.legend(fontsize=16);                       # <--- 已移除
    # ==========================================

    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid(True);
    plt.tight_layout();
    plt.savefig(f"{title}_zoom_comparison.png");
    plt.close(fig)
    print(f"放大图已保存至: {title}_zoom_comparison.png")


def plot_residuals_full(y_true, predictions_dict, title):
    print(f"绘制 {title} 的完整误差分布图...")
    time_values = np.arange(len(y_true)) * 0.20
    fig, ax = plt.subplots(figsize=(10, 6))
    for model_name, preds in predictions_dict.items():
        if model_name in model_color:
            residuals = preds - y_true
            ax.plot(time_values, residuals, label=model_name, alpha=0.7, color=model_color[model_name], linewidth=4)
    ax.axhline(0, color='black', linestyle='--', linewidth=1.5)

    ax.set_title(f"Prediction Error at {title}", fontsize=22)
    ax.set_xlabel('Time (s)', fontsize=20);
    ax.set_ylabel('Error (mm)', fontsize=20)

    ax.legend(fontsize=20, loc='lower right');
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid(True);
    plt.tight_layout();
    plt.savefig(f"{title}_residual_full.png");
    plt.close(fig)
    print(f"误差图已保存至: {title}_residual_full.png")


def plot_attention_heatmap(attention_scores, y_true, y_pred, seq_length, title, filename):
    return


# --- 4. 主执行函数 ---
def main():
    try:
        print("--- 正在加载所有已保存的模型和Scaler ---")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu");
        print(f"将使用设备 (Using device): {device}")

        # === 1. MLP ===
        scaler_mlp = joblib.load("Scaler_no_RF_Circuit.pkl")
        model_mlp = MLPWithoutAttention(input_dim=2).to(device)
        model_mlp.load_state_dict(torch.load("No_RF_no_attention_model_Circuit.pth"));
        model_mlp.eval()

        # === 2. LSTM ===
        scaler_lstm = joblib.load("Circuit_board_lstm_scaler.pkl")
        lstm_config = {
            "SEQUENCE_LENGTH": 20, "hidden_size": 64,
            "num_layers": 1, "dropout_rate": 0.3
        }
        model_lstm = LSTMModel(input_size=2, **{k: v for k, v in lstm_config.items() if k != 'SEQUENCE_LENGTH'})
        model_lstm.load_state_dict(torch.load("Circuit_board_lstm_model.pth"));
        model_lstm.to(device).eval()

        # === 3. SVR ===
        scaler_X_svr = joblib.load("Scaler_X_SVR.pkl")
        scaler_y_svr = joblib.load("Scaler_y_SVR.pkl")
        model_svr = joblib.load("Final_SVR_model.pkl")

        # === 4. Poly-LSTM ===
        scaler_X_original_poly = joblib.load("Scaler_Original_X_PolyLSTM1.pkl")
        poly_transformer = joblib.load("Polynomial_Transformer_model_PolyLSTM1.pkl")
        scaler_poly = joblib.load("Scaler_Poly_X_PolyLSTM1.pkl")

        poly_lstm_config = {
            "SEQUENCE_LENGTH": 20, "input_size": 10, "hidden_size": 64,
            "num_layers": 1, "dropout_rate": 0.3
        }
        model_poly_lstm = LSTMModel_Poly(
            **{k: v for k, v in poly_lstm_config.items() if k != 'SEQUENCE_LENGTH'}
        ).to(device)
        model_poly_lstm.load_state_dict(torch.load("Final_LSTM_model_PolyLSTM1.pth"));
        model_poly_lstm.eval()

        # === 5. 【新增】GBDT ===
        # 请确保您已经训练并保存了这些文件，名称需要和训练代码一致
        scaler_X_gbdt = joblib.load("Scaler_X_GBDT.pkl")
        scaler_y_gbdt = joblib.load("Scaler_y_GBDT.pkl")
        model_gbdt = joblib.load("Final_GBDT_model.pkl")

        assert lstm_config["SEQUENCE_LENGTH"] == poly_lstm_config["SEQUENCE_LENGTH"]
        COMMON_SEQ_LENGTH = lstm_config["SEQUENCE_LENGTH"]

        print("所有模型加载成功。")

    except Exception as e:
        print(f"加载模型或scaler时出错: {e}");
        sys.exit(1)

    # 定义测试数据集
    test_datasets = {
        "fast rate": [{"file_path": "C:/Users/hs/Desktop/Sensor.xlsx",
                       "ranges": [("Sheet4", "S1:S1266", "V1:V1266")]}],
        "medium rate": [{"file_path": "C:/Users/hs/Desktop/Sensor.xlsx",
                         "ranges": [("Sheet2", "G1:G2041", "J1:J2041")]}],
        "slow rate": [{"file_path": "C:/Users/hs/Desktop/Sensor.xlsx",
                       "ranges": [("Sheet3", "S1:S4074", "V1:V4074")]}],
        "5℃": [{"file_path": "C:/Users/hs/Desktop/Sensor.xlsx",
                "ranges": [("Sheet1", "A1:A1432", "D1:D1432")]}],
        "15℃": [{"file_path": "C:/Users/hs/Desktop/Sensor.xlsx",
                 "ranges": [("Sheet1", "G1:G1327", "J1:J1327")]}],
        "20℃": [{"file_path": "C:/Users/hs/Desktop/Sensor.xlsx",
                 "ranges": [("Sheet4", "S1:S1266", "V1:V1266")]}],
        "25℃": [{"file_path": "C:/Users/hs/Desktop/Sensor.xlsx",
                 "ranges": [("Sheet1", "Y1:Y1217", "AB1:AB1217")]}],
        "variable rate": [{"file_path": "C:/Users/hs/Desktop/Sensor.xlsx",
                           "ranges": [("Sheet1", "M1:M1836", "P1:P1836")]}],
    }

    # 循环处理每个测试集
    results_all = {}
    for name, dataset in test_datasets.items():
        print(f"\n===== 正在评估: {name} =====")
        X_test, y_test = process_data_and_get_features(dataset)

        if X_test.size == 0: continue

        _, y_test_seq = create_sequences(X_test, y_test, COMMON_SEQ_LENGTH)
        if y_test_seq.size == 0: continue

        # --- 评估所有模型 ---
        preds_mlp, preds_lstm, preds_svr, preds_poly_lstm, preds_gbdt = None, None, None, None, None

        with torch.no_grad():
            # 1. MLP
            X_test_scaled_mlp = scaler_mlp.transform(X_test)
            preds_mlp_full, _ = model_mlp(torch.tensor(X_test_scaled_mlp, dtype=torch.float32).to(device))
            preds_mlp = preds_mlp_full.squeeze().cpu().numpy()
            preds_mlp = preds_mlp[-len(y_test_seq):]

            # 2. LSTM
            X_test_scaled_lstm = scaler_lstm.transform(X_test)
            X_test_seq_lstm, _ = create_sequences(X_test_scaled_lstm, y_test, COMMON_SEQ_LENGTH)
            preds_lstm_tensor = model_lstm(torch.tensor(X_test_seq_lstm, dtype=torch.float32).to(device))
            preds_lstm = preds_lstm_tensor.squeeze().cpu().numpy()

            # 3. Poly-LSTM
            X_test_scaled_orig_poly = scaler_X_original_poly.transform(X_test)
            X_test_poly = poly_transformer.transform(X_test_scaled_orig_poly)
            X_test_poly_scaled = scaler_poly.transform(X_test_poly)
            X_test_seq_poly, _ = create_sequences(X_test_poly_scaled, y_test, COMMON_SEQ_LENGTH)
            preds_poly_tensor = model_poly_lstm(torch.tensor(X_test_seq_poly, dtype=torch.float32).to(device))
            preds_poly_lstm = preds_poly_tensor.squeeze().cpu().numpy()

        # 4. SVR
        X_test_scaled_svr = scaler_X_svr.transform(X_test)
        preds_svr_scaled = model_svr.predict(X_test_scaled_svr)
        preds_svr_full = scaler_y_svr.inverse_transform(preds_svr_scaled.reshape(-1, 1)).flatten()
        preds_svr = preds_svr_full[-len(y_test_seq):]

        # 5. 【新增】GBDT
        X_test_scaled_gbdt = scaler_X_gbdt.transform(X_test)
        preds_gbdt_scaled = model_gbdt.predict(X_test_scaled_gbdt)
        preds_gbdt_full = scaler_y_gbdt.inverse_transform(preds_gbdt_scaled.reshape(-1, 1)).flatten()
        preds_gbdt = preds_gbdt_full[-len(y_test_seq):]


        print(f"基准真实值 y_true (来自时序模型) 长度: {len(y_test_seq)}")
        evaluate_model_metrics(y_test_seq, preds_mlp, "MLP")
        evaluate_model_metrics(y_test_seq, preds_lstm, "LSTM")
        evaluate_model_metrics(y_test_seq, preds_poly_lstm, "Poly-LSTM")
        evaluate_model_metrics(y_test_seq, preds_gbdt, "GBDT")  # 评估 GBDT



        results_all[name] = {
            "y_true": y_test_seq,
            "predictions": {
                "MLP": preds_mlp,
                "LSTM": preds_lstm,
                # "SVR": preds_svr,
                "Poly-LSTM": preds_poly_lstm,
                "GBDT": preds_gbdt,  # 添加到绘图数据
            },
            "attention_scores": None,
            "poly_preds": preds_poly_lstm
        }

    print("\n--- 正在生成所有图表 ---")
    for title, data in results_all.items():
        y_true_plot = data["y_true"]
        predictions_plot = data["predictions"]

        if PLOT_CONFIG["enable_prediction_plot"]:
            plot_predictions_full(y_true_plot, predictions_plot, title)

        if PLOT_CONFIG["enable_residual_plot"]:
            plot_residuals_full(y_true_plot, predictions_plot, title)

        if PLOT_CONFIG["enable_zoom_plot"] and title in PLOT_CONFIG["zoom_ranges"]:
            start, end = PLOT_CONFIG["zoom_ranges"][title]
            adjusted_start = max(0, start - COMMON_SEQ_LENGTH)
            adjusted_end = max(0, end - COMMON_SEQ_LENGTH)
            if adjusted_end > adjusted_start:
                plot_predictions_zoom(y_true_plot, predictions_plot, title, start_sample=adjusted_start,
                                      end_sample=adjusted_end)


if __name__ == "__main__":
    main()