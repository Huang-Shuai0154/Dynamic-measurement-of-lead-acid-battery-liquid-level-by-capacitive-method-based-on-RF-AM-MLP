import openpyxl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # <--- 修正 1：添加此导入
import re
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import sys

# --- 全局绘图和配置设置 ---
plt.rcParams.update({
    'font.size': 20, 'axes.titlesize': 22, 'axes.labelsize': 20,
    'legend.fontsize': 16.5, 'xtick.labelsize': 20, 'ytick.labelsize': 20
})

# 在这里可以方便地修改绘图选项和范围
PLOT_CONFIG = {
    "enable_prediction_plot": True,  # 是否绘制总的预测对比图
    "enable_residual_plot": True,  # 是否绘制误差对比图
    "enable_zoom_plot": True,  # 是否绘制局部放大图
    # 您可以在这里为每个测试集自定义想要放大的“样本点”范围
    "zoom_ranges": {
        "30 rpm": (4485, 4505),  # (起始样本点, 结束样本点)
        "60 rpm": (1860, 1880),
        "100 rpm": (1300, 1320),
        "Variable rates":(2870, 2890)
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
    if not all_features_list:
        return np.array([]), np.array([])
    X_full = np.vstack(all_features_list);
    y_full = np.concatenate(all_h_list)
    return X_full, y_full


def evaluate_model_metrics(y_true, y_pred, model_name):
    # 确保y_true不为0，避免MAPE计算时除以0
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
class AttentionMechanism(nn.Module):
    def __init__(self, input_dim):
        super(AttentionMechanism, self).__init__()
        # 这里的线性层名叫 attention_weights
        self.attention_weights = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        attn_scores = F.softmax(self.attention_weights(x), dim=-1)
        return x * attn_scores, attn_scores


class MLPWithAttention(nn.Module):
    def __init__(self, input_dim):
        super(MLPWithAttention, self).__init__()
        # 这里的模块名叫 attention，它引用了上面的 AttentionMechanism 类
        self.attention = AttentionMechanism(input_dim)
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x, attn_scores = self.attention(x)  # 调用 attention 模块
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        return x, attn_scores


class MLPWithoutAttention(nn.Module):
    def __init__(self, input_dim):
        super(MLPWithoutAttention, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128);
        self.fc3 = nn.Linear(128, 64);
        self.fc4 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x));
        x = self.dropout(x);
        x = F.leaky_relu(self.fc3(x));
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


# --- 3. 绘图函数 ---
model_color = {"RF-AM-MLP": "blue", "MLP": "orange", "LSTM": "red"}


def plot_predictions_full(y_true, predictions_dict, title):
    print(f"绘制 {title} 的完整预测对比图...")
    time_values = np.arange(len(y_true)) * 0.14
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.plot(time_values, y_true, label='True Value', color='black', linewidth=4, zorder=10)
    for model_name, preds in predictions_dict.items():
        if model_name in model_color:
            ax.plot(time_values, preds, label=model_name, color=model_color[model_name], linewidth=4, alpha=0.8)
    ax.set_title(f"Peristaltic pump rotates at {title}", fontsize=20)
    ax.set_xlabel('Time (s)', fontsize=20);
    ax.set_ylabel('Liquid level (mm)', fontsize=20)
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
    time_values = sample_indices * 0.14
    mask = (sample_indices >= start_sample) & (sample_indices <= end_sample)
    if not np.any(mask):
        print(f"警告: 在 {title} 中找不到样本范围 {start_sample}-{end_sample} 的数据。")
        return
    fig, ax = plt.subplots(figsize=(4, 3))  # 放大图用稍大的尺寸
    ax.plot(time_values[mask], y_true[mask], label='True Value', color='black', linewidth=4, zorder=10)
    for model_name, preds in predictions_dict.items():
        if model_name in model_color:
            ax.plot(time_values[mask], preds[mask], label=model_name, color=model_color[model_name], linewidth=4,
                    alpha=0.8)
    ax.grid(True);
    plt.tight_layout();
    plt.savefig(f"{title}_zoom_comparison.png");
    plt.close(fig)
    print(f"放大图已保存至: {title}_zoom_comparison.png")


def plot_residuals_full(y_true, predictions_dict, title):
    print(f"绘制 {title} 的完整误差分布图...")
    time_values = np.arange(len(y_true)) * 0.14
    fig, ax = plt.subplots(figsize=(10, 6))
    for model_name, preds in predictions_dict.items():
        if model_name in model_color:
            residuals = preds - y_true
            ax.plot(time_values, residuals, label=model_name, alpha=0.7, color=model_color[model_name], linewidth=4)
    ax.axhline(0, color='black', linestyle='--', linewidth=1.5)  # 添加y=0的参考线
    ax.set_title(f"Peristaltic pump rotates at {title}", fontsize=22)
    ax.set_xlabel('Time (s)', fontsize=20);
    ax.set_ylabel('Error (mm)', fontsize=20)
    ax.legend(fontsize=20, loc='lower right');
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid(True);
    plt.tight_layout();
    plt.savefig(f"{title}_residual_full.png");
    plt.close(fig)
    print(f"误差图已保存至: {title}_residual_full.png")


# --- 4. 主执行函数 ---
def main():
    try:
        print("--- 正在加载已保存的模型和Scaler ---")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu");
        print(f"将使用设备 (Using device): {device}")

        # 加载模型
        rf_model = joblib.load("random_forest_model1.pkl")
        scaler_rf = joblib.load("scaler_rf1.pkl")
        rf_attention_model = MLPWithAttention(input_dim=rf_model.n_estimators).to(device)
        rf_attention_model.load_state_dict(torch.load("rf_attention_model1.pth"));
        rf_attention_model.eval()

        scaler_no_rf = joblib.load("scaler_no_rf.pkl")
        no_rf_no_attention_model = MLPWithoutAttention(input_dim=4).to(device)
        no_rf_no_attention_model.load_state_dict(torch.load("no_rf_no_attention_model.pth"));
        no_rf_no_attention_model.eval()

        scaler_lstm = joblib.load("manual_lstm_scaler.pkl")
        # <--- 修正 2：将字典的键改为小写，以匹配类定义 ---
        lstm_config = {
            "SEQUENCE_LENGTH": 20,
            "hidden_size": 64,
            "num_layers": 1,
            "dropout_rate": 0.3
        }
        lstm_model = LSTMModel(input_size=4, **{k: v for k, v in lstm_config.items() if k != 'SEQUENCE_LENGTH'})
        lstm_model.load_state_dict(torch.load("manual_tuned_lstm_model.pth"));
        lstm_model.to(device).eval()
        print("所有模型加载成功。")

    except Exception as e:
        print(f"加载模型或scaler时出错: {e}");
        sys.exit(1)

    # 定义测试数据集 (请确保路径正确)
    test_datasets = {
        "30 rpm": [{"file_path": "C:/Users/hs/Desktop/Constant rates validation data.xlsx",
                    "ranges": [("Sheet1", "B2:B6149", "C2:C6149")]}],
        "60 rpm": [{"file_path": "C:/Users/hs/Desktop/Constant rates validation data.xlsx",
                    "ranges": [("Sheet1", "I2:I3652", "J2:J3652")]}],
        "100 rpm": [{"file_path": "C:/Users/hs/Desktop/Constant rates validation data.xlsx",
                     "ranges": [("Sheet1", "P2:P2703", "Q2:Q2703")]}],
        "Variable rates": [{"file_path": "C:/Users/hs/Desktop/Variable rates validation data.xlsx",
                     "ranges": [("Sheet1", "B2:B4360", "C2:C4360")]}]
    }

    # 循环处理每个测试集
    results_all = {}
    for name, dataset in test_datasets.items():
        print(f"\n===== 正在评估: {name} =====")
        X_test, y_test = process_data_and_get_features(dataset)

        if X_test.size == 0:
            print(f"无法从数据集 '{name}' 中处理得到特征，跳过评估。")
            continue

        # 评估所有模型
        with torch.no_grad():
            X_test_leaf = rf_model.apply(X_test)
            X_test_scaled_rf = scaler_rf.transform(X_test_leaf)
            preds_rf_am_mlp, _ = rf_attention_model(torch.tensor(X_test_scaled_rf, dtype=torch.float32).to(device))
            preds_rf_am_mlp = preds_rf_am_mlp.squeeze().cpu().numpy()

            X_test_scaled_no_rf = scaler_no_rf.transform(X_test)
            preds_mlp, _ = no_rf_no_attention_model(torch.tensor(X_test_scaled_no_rf, dtype=torch.float32).to(device))
            preds_mlp = preds_mlp.squeeze().cpu().numpy()

            X_test_scaled_lstm = scaler_lstm.transform(X_test)
            X_test_seq, y_test_seq = create_sequences(X_test_scaled_lstm, y_test, lstm_config["SEQUENCE_LENGTH"])

            if X_test_seq.size == 0:
                print(f"为LSTM创建序列后数据为空，跳过 {name} 的评估。")
                continue

            preds_lstm = lstm_model(torch.tensor(X_test_seq, dtype=torch.float32).to(device))
            preds_lstm = preds_lstm.squeeze().cpu().numpy()

        # 打印评估指标
        evaluate_model_metrics(y_test_seq, preds_lstm, "LSTM")
        # 注意：为公平对比，所有模型都应与LSTM的真实值(y_test_seq)进行比较，因其长度最短
        evaluate_model_metrics(y_test_seq, preds_rf_am_mlp[-len(y_test_seq):], "RF-AM-MLP")
        evaluate_model_metrics(y_test_seq, preds_mlp[-len(y_test_seq):], "MLP")

        # 存储结果用于绘图
        results_all[name] = {
            "y_true": y_test_seq,
            "predictions": {
                "RF-AM-MLP": preds_rf_am_mlp[-len(y_test_seq):],
                "MLP": preds_mlp[-len(y_test_seq):],
                "LSTM": preds_lstm,
            }
        }

    # --- 绘图 ---
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
            # 调整缩放范围以适应LSTM序列的长度
            seq_len = lstm_config["SEQUENCE_LENGTH"]
            adjusted_start = max(0, start - seq_len)
            adjusted_end = max(0, end - seq_len)
            if adjusted_end > adjusted_start:
                plot_predictions_zoom(y_true_plot, predictions_plot, title, start_sample=adjusted_start,
                                      end_sample=adjusted_end)
            else:
                print(f"警告: 调整后的缩放范围 ({adjusted_start}-{adjusted_end}) 无效，跳过 {title} 的放大图绘制。")


if __name__ == "__main__":
    main()