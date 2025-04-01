#Read me
#In this part, the prediction effects of different models RF-AM-MLP,MLP, and static linear models on different verification sets were compared and different curves were drawn for comparative analysis and error analysis
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
import openpyxl
import re
import joblib
import matplotlib.pyplot as plt
import sys
import time
import csv

plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 22,
    'axes.labelsize': 20,
    'legend.fontsize': 16.5,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20
})

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
        data_array = np.array(data)
        return data_array
    except Exception as e:
        print(f"Error reading Excel range: {e}")
        sys.exit(1)


class AttentionMechanism(nn.Module):
    def __init__(self, input_dim):
        super(AttentionMechanism, self).__init__()
        self.attention_weights = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        attn_scores = F.softmax(self.attention_weights(x), dim=-1)
        return x * attn_scores, attn_scores


class MLPWithAttention(nn.Module):
    def __init__(self, input_dim):
        super(MLPWithAttention, self).__init__()
        self.attention = AttentionMechanism(input_dim)
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x, attn_scores = self.attention(x)
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
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        return x, None


def plot_model_comparison(y_true, preds_dict, sample_numbers, title, model1, model2, label1, label2):
    """
    绘制两个模型之间的对比图
    """
    print(f"绘制模型对比图: {title}")

    # 计算时间 t (单位为秒)
    time_values = sample_numbers * 0.14  # 0.14 秒为每个样本点的时间间隔

    plt.figure(figsize=(10, 7.5))
    plt.plot(time_values, y_true, label='True Value', color='black', linewidth=2)

    # 绘制两个模型的预测
    plt.plot(time_values, preds_dict[model1], label=label1, color='red', alpha=0.7)
    plt.plot(time_values, preds_dict[model2], label=label2, color='blue', alpha=0.7)

    plt.title(f"{title} - {label1} vs {label2}", fontsize=22)
    plt.xlabel('Time (s)', fontsize=20)
    plt.ylabel('Liquid Level (mm)', fontsize=20)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{title}_{label1}_vs_{label2}.png")
    plt.close()
    print(f"{title} 对比图已保存为 {title}_{label1}_vs_{label2}.png 并关闭")
    time.sleep(0.5)


def plot_all_comparisons(results):
    """
    绘制所有速率的模型对比图。
    """
    for title, preds_dict in results.items():
        sample_numbers = np.arange(1, len(preds_dict["y_true"]) + 1)

        # 比较 RF-AM-MLP 和 MLP only
        plot_model_comparison(
            y_true=preds_dict["y_true"],
            preds_dict=preds_dict,
            sample_numbers=sample_numbers,
            title=title,
            model1="RF-AM-MLP",
            model2="MLP only",
            label1="RF-AM-MLP",
            label2="MLP only"
        )

        # 比较 RF-AM-MLP 和 Linear model
        plot_model_comparison(
            y_true=preds_dict["y_true"],
            preds_dict=preds_dict,
            sample_numbers=sample_numbers,
            title=title,
            model1="RF-AM-MLP",
            model2="Linear model",
            label1="RF-AM-MLP",
            label2="Linear model"
        )

def plot_model_vs_true(y_true, preds_dict, sample_numbers, title, model_name, label, color):
    print(f"绘制模型与真实值对比图: {title} - {model_name}")
    time_values = sample_numbers * 0.14
    plt.figure(figsize=(10, 6))
    plt.plot(time_values, y_true, label='True Value', color='black', linewidth=3)
    plt.plot(time_values, preds_dict[model_name], label=label, color=color, alpha=0.7, linewidth=3)
    plt.title(f"Peristaltic pump {title}", fontsize=22)
    plt.xlabel('Time (s)', fontsize=20)
    plt.ylabel('Liquid level (mm)', fontsize=20)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{title}_{label}_vs_True.png")
    plt.close()
    print(f"{title} - {label} vs True 图表已保存为 {title}_{label}_vs_True.png 并关闭")
    time.sleep(0.5)


def plot_all_model_vs_true(results):
    """
    绘制所有速率下的三个模型与真实值的对比图
    """
    # Define a color map for different speed rates
    color_map = {
        "30 rpm": "blue",
        "60 rpm": "blue",
        "100 rpm": "blue",
        "Variable rates": "blue"
    }

    for title, preds_dict in results.items():
        sample_numbers = np.arange(1, len(preds_dict["y_true"]) + 1)

        # Get the color for the current rate
        color = color_map.get(title, "black")  # Default to black if the title is not in the color map

        # 绘制每个模型与真实值的对比图
        plot_model_vs_true(
            y_true=preds_dict["y_true"],
            preds_dict=preds_dict,
            sample_numbers=sample_numbers,
            title=title,
            model_name="RF-AM-MLP",
            label="RF-AM-MLP",
            color=color
        )

        plot_model_vs_true(
            y_true=preds_dict["y_true"],
            preds_dict=preds_dict,
            sample_numbers=sample_numbers,
            title=title,
            model_name="MLP",
            label="MLP",
            color=color
        )

        plot_model_vs_true(
            y_true=preds_dict["y_true"],
            preds_dict=preds_dict,
            sample_numbers=sample_numbers,
            title=title,
            model_name="Static linear model",
            label="Static linear model",
            color=color
        )


# 定义模型颜色字典
model_color = {
    "RF-AM-MLP": "blue",
    "MLP": "orange",
    "Static linear model": "green"
}

def plot_predictions_full(y_true, predictions_dict, sample_numbers, title):
    print(f"绘制完整预测图表: {title}")

    # 计算时间 t (单位为秒)
    time_values = sample_numbers * 0.14  # 0.14 秒为每个样本点的时间间隔

    plt.figure(figsize=(8, 7.5))
    plt.plot(time_values, y_true, label='True Value', color='black', linewidth=4)

    # 使用字典中定义的颜色绘制各模型预测曲线
    for model_name, preds in predictions_dict.items():
        if model_name in model_color:
            plt.plot(time_values, preds, label=model_name, color=model_color[model_name], linewidth=4, alpha=0.7)

    plt.title(f"Peristaltic pump rotates at {title} ", fontsize=17)
    plt.xlabel('Time (s)', fontsize=16.5)
    plt.ylabel('Liquid level (mm)', fontsize=16.5)
    plt.legend(fontsize=16.5)
    plt.legend(loc='lower left')
    plt.xticks(fontsize=16.5)
    plt.yticks(fontsize=16.5)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{title}_full.png")
    plt.close()
    print(f"{title} 图表已保存为 {title}_full.png 并关闭")
    time.sleep(0.5)


def plot_predictions_zoom(y_true, predictions_dict, sample_numbers, title, min_sample=20, max_sample=40):
    print(f"绘制缩放预测图表: {title} (Samples {min_sample}-{max_sample})")

    # 计算时间 t (单位为秒)
    time_values = sample_numbers * 0.14  # 0.14 秒为每个样本点的时间间隔

    plt.figure(figsize=(4, 3))
    mask = (sample_numbers >= min_sample) & (sample_numbers <= max_sample)
    if np.any(mask):
        y_true_zoom = y_true[mask]
        sample_numbers_zoom = sample_numbers[mask]
        time_values_zoom = time_values[mask]  # 对时间也应用掩码

        plt.plot(time_values_zoom, y_true_zoom, label='True Value', color='black', linewidth=4)
        # 同样使用字典中定义的颜色
        for model_name, preds in predictions_dict.items():
            if model_name in model_color:
                preds = np.array(preds)
                preds_zoom = preds[mask]
                plt.plot(time_values_zoom, preds_zoom, label=model_name, color=model_color[model_name],linewidth=4, alpha=0.7)
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{title}_zoom.png")
        plt.close()
        print(f"{title} 缩放图表已保存为 {title}_zoom.png 并关闭")
        time.sleep(0.5)
    else:
        print(f"No data in Samples {min_sample}-{max_sample} range for {title}.")

def plot_residuals_full(y_true, predictions_dict, sample_numbers, title):
    print(f"绘制完整残差分布图表: {title}")
    time_values = sample_numbers * 0.14
    plt.figure(figsize=(10, 6))
    colors = ['red', 'blue', 'green', 'orange']
    for (model_name, preds), color in zip(predictions_dict.items(), colors):
        if model_name != "y_true":
            residuals = preds - y_true
            plt.plot(time_values, residuals, label=model_name, alpha=0.5, color=color, linewidth=4)
    plt.title(f"Peristaltic pump rotates at {title} ", fontsize=22)
    plt.xlabel('Time (s)', fontsize=20)
    plt.ylabel('Error (mm)', fontsize=20)
    plt.legend(fontsize=15)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    plt.legend(loc='lower right')
    plt.savefig(f"{title}_residual_full.png")
    plt.close()
    print(f"{title} 残差分布图表已保存为 {title}_residual_full.png 并关闭")
    time.sleep(0.5)


def plot_residuals_zoom(y_true, predictions_dict, sample_numbers, title, min_sample=20, max_sample=40):
    print(f"绘制缩放残差分布图表: {title} (Samples {min_sample}-{max_sample})")

    # 计算时间 t (单位为秒)
    time_values = sample_numbers * 0.14  # 0.14 秒为每个样本点的时间间隔

    plt.figure(figsize=(10, 6))
    mask = (sample_numbers >= min_sample) & (sample_numbers <= max_sample)
    if np.any(mask):
        y_true_zoom = y_true[mask]
        sample_numbers_zoom = sample_numbers[mask]
        time_values_zoom = time_values[mask]  # 对时间也应用掩码

        preds_dict_zoom = {}
        for k, v in predictions_dict.items():
            if k not in ["y_true", "C"]:
                v_array = np.array(v)
                if v_array.ndim == 0:
                    print(f"Model {k} has a scalar prediction. Skipping.")
                    continue
                if len(v_array) == 0:
                    print(f"Model {k} has no predictions. Skipping.")
                    continue
                preds_zoom = v_array[mask]
                preds_dict_zoom[k] = preds_zoom

        residuals_zoom = {k: y_true_zoom - preds for k, preds in preds_dict_zoom.items()}

        if residuals_zoom:
            colors = ['red', 'blue', 'green']
            for (model_name, residuals), color in zip(residuals_zoom.items(), colors):
                plt.plot(time_values_zoom, residuals, label=model_name, alpha=0.5, color=color)
            plt.title(f"{title} error distribution", fontsize=22)
            plt.xlabel('Time (s)', fontsize=20)
            plt.ylabel('Error (mm)', fontsize=20)
            plt.legend(fontsize=15)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.grid(True)
            plt.tight_layout()
            plt.legend(loc='upper right')
            plt.savefig(f"{title}_residual_zoom.png")
            plt.close()
            print(f"{title} 缩放残差分布图表已保存为 {title}_residual_zoom.png 并关闭")
            time.sleep(0.5)
        else:
            print(f"No residual data available in Samples {min_sample}-{max_sample} range for {title}.")
    else:
        print(f"No residual data in Samples {min_sample}-{max_sample} range for {title}.")


def plot_all_residuals(results):
    for title, preds_dict in results.items():
        sample_numbers = np.arange(1, len(preds_dict["y_true"]) + 1)
        plot_residuals_full(y_true=preds_dict["y_true"], predictions_dict=preds_dict,
                              sample_numbers=sample_numbers, title=title)
        plot_residuals_zoom(y_true=preds_dict["y_true"], predictions_dict=preds_dict,
                              sample_numbers=sample_numbers, title=title,
                              min_sample=900, max_sample=1000)

def plot_all_predictions(results):
    for title, preds_dict in results.items():
        sample_numbers = np.arange(1, len(preds_dict["y_true"]) + 1)
        plot_predictions_full(y_true=preds_dict["y_true"], predictions_dict=preds_dict,
                              sample_numbers=sample_numbers, title=title)
        plot_predictions_zoom(y_true=preds_dict["y_true"], predictions_dict=preds_dict,
                              sample_numbers=sample_numbers, title=title,
                              min_sample=4140, max_sample=4170)


def calculate_amplitude_and_frequency_fft(C, window_size=2, sample_rate=1.0):
    n = len(C)
    frequencies = []
    amplitudes = []

    # 滑动窗口进行 FFT 计算
    for i in range(0, n - window_size + 1):
        window_data = C[i:i + window_size]

        # 计算 FFT
        fft_vals = np.fft.fft(window_data)
        freqs = np.fft.fftfreq(window_size, d=sample_rate)

        # 取正频率部分（window_size // 2）
        positive_freqs = freqs[:window_size // 2]
        positive_fft_vals = fft_vals[:window_size // 2]

        # 计算幅度
        amplitude = np.abs(positive_fft_vals)

        # 找到最大振幅对应的频率
        peak_freq = positive_freqs[np.argmax(amplitude)] if len(amplitude) > 0 else 0
        peak_amplitude = np.max(amplitude) if len(amplitude) > 0 else 0

        frequencies.append(peak_freq)
        amplitudes.append(peak_amplitude)

    return np.array(frequencies), np.array(amplitudes)


def main():
    try:
        print("Loading models and scalers...")
        rf_model = joblib.load("random_forest_model.pkl")
        scaler_rf = joblib.load("scaler_rf.pkl")
        scaler_no_rf = joblib.load("scaler_no_rf.pkl")

        sample_input = np.zeros((1, 4))
        X_leaf_sample = rf_model.apply(sample_input)
        input_dim_rf = X_leaf_sample.shape[1]
        input_dim_no_rf = 4
        print(f"Input dimensions - RF: {input_dim_rf}, No RF: {input_dim_no_rf}")

        print("Loading MLP models...")
        rf_attention_model = MLPWithAttention(input_dim=input_dim_rf)
        rf_attention_model.load_state_dict(torch.load("rf_attention_model.pth", map_location=torch.device('cpu')))
        rf_attention_model.eval()

        no_rf_no_attention_model = MLPWithoutAttention(input_dim=input_dim_no_rf)
        no_rf_no_attention_model.load_state_dict(
            torch.load("no_rf_no_attention_model.pth", map_location=torch.device('cpu')))
        no_rf_no_attention_model.eval()

        print("All models loaded successfully.")

    except Exception as e:
        print(f"Error loading models or scalers: {e}")
        sys.exit(1)

    validation_data_30 = [
        {
            "file_path": "C:/Users/hs/Desktop/Constant rates validation data.xlsx",
            "ranges": [
                ("Sheet1", "B2:B6149", "C2:C6149"),
            ]
        }
    ]
    validation_data_60 = [
        {
            "file_path": "C:/Users/hs/Desktop/Constant rates validation data.xlsx",
            "ranges": [
                ("Sheet1", "I2:I3652", "J2:J3652"),
            ]
        }
    ]
    validation_data_100 = [
        {
            "file_path": "C:/Users/hs/Desktop/Constant rates validation data.xlsx",
            "ranges": [
                ("Sheet1", "P2:P2703", "Q2:Q2703"),
            ]
        }
    ]
    validation_data_random_speed = [
        {
            "file_path": "C:/Users/hs/Desktop/Variable rates validation data.xlsx",
            "ranges": [
                ("Sheet1", "B2:B4360", "C2:C4360"),
            ]
        }
    ]

    validation_datasets = {
        "30 rpm": validation_data_30,
        "60 rpm": validation_data_60,
        "100 rpm": validation_data_100,
        "variable rates": validation_data_random_speed,
    }

    results = {
        "30 rpm": {},
        "60 rpm": {},
        "100 rpm": {},
        "variable rates": {},
    }

    def calculate_additional_model_prediction(C):
        # 计算新模型的预测：h = 1.3613 * C + 58.5865
        return 1.3613 * C + 58.8591

    def evaluate_model(y_true, y_pred, model_name):
        # 均方误差
        mse = mean_squared_error(y_true, y_pred)
        # 均方根误差
        rmse = np.sqrt(mse)
        # 平均绝对误差
        mae = mean_absolute_error(y_true, y_pred)
        # 决定系数
        r2 = r2_score(y_true, y_pred)
        # 平均绝对百分比误差
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        # 打印评估指标
        print(f"评估结果 ({model_name}):")
        print(f"均方误差 (MSE): {mse:.4f}")
        print(f"均方根误差 (RMSE): {rmse:.4f}")
        print(f"平均绝对误差 (MAE): {mae:.4f}")
        print(f"决定系数 (R²): {r2:.4f}")
        print(f"平均绝对百分比误差 (MAPE): {mape:.4f}%")
        print("\n")

    def process_validation_data_rf(data_sources, rf_model, scaler_rf, model, process_var=1, measurement_var=0.001):
        all_amplitude = []  # 振幅
        all_frequency = []  # 频率
        all_avg_C = []  # 每个窗口的电容平均值
        all_delta_C_window = []  # 每个窗口的电容首尾差值
        all_h = []  # h值
        all_new_model_h = []  # 新模型的 h 值

        for source in data_sources:
            file_path = source["file_path"]
            for sheet_name, c_range, h_range in source["ranges"]:
                C = read_excel_range(file_path, sheet_name, c_range).flatten()
                h = read_excel_range(file_path, sheet_name, h_range).flatten()

                # 计算每个点的电容差值：每个点与前一个点的差
                delta_C_window = [C[i + 1] - C[i] for i in range(len(C) - 1)]  # 差值（每个点与前一个点的差值）

                # 计算每个点的频率和振幅
                frequency, amplitude = calculate_amplitude_and_frequency_fft(C, window_size=2)

                # 对齐数据长度
                min_len = min(len(C), len(delta_C_window), len(amplitude), len(frequency), len(h))
                C = C[:min_len]
                delta_C_window = delta_C_window[:min_len]
                amplitude = amplitude[:min_len]
                frequency = frequency[:min_len]
                h = h[:min_len]

                # 计算新模型的 h 值并添加到列表中
                new_model_h = calculate_additional_model_prediction(C)

                # 将每个点的电容值、电容差值、振幅、频率、新模型的 h 值添加到列表中
                all_avg_C.extend(C)
                all_delta_C_window.extend(delta_C_window)
                all_amplitude.extend(amplitude)
                all_frequency.extend(frequency)
                all_h.extend(h)
                all_new_model_h.extend(new_model_h)

        # 转换为数组
        all_avg_C = np.array(all_avg_C).reshape(-1, 1)
        all_delta_C_window = np.array(all_delta_C_window).reshape(-1, 1)
        all_amplitude = np.array(all_amplitude).reshape(-1, 1)
        all_frequency = np.array(all_frequency).reshape(-1, 1)
        all_h = np.array(all_h)
        all_new_model_h = np.array(all_new_model_h)

        # 合并特征
        X_validation = np.hstack((all_avg_C, all_delta_C_window, all_amplitude, all_frequency))
        y_validation = all_h.flatten()

        X_validation_leaf = rf_model.apply(X_validation)

        try:
            X_validation_scaled = scaler_rf.transform(X_validation_leaf)
        except Exception as e:
            print(f"Error scaling validation features: {e}")
            sys.exit(1)

        X_validation_tensor = torch.tensor(X_validation_scaled, dtype=torch.float32)
        y_validation_tensor = torch.tensor(y_validation, dtype=torch.float32).view(-1)

        model.eval()
        with torch.no_grad():
            try:
                validation_predictions, _ = model(X_validation_tensor)  # 这里返回的是一个元组 (output, attn_scores)
                validation_predictions = validation_predictions.squeeze()  # 获取预测输出并进行 squeeze
            except Exception as e:
                print(f"Error during model prediction on validation data: {e}")
                sys.exit(1)


        smoothed_validation_predictions = validation_predictions
        smoothed_validation_predictions = smoothed_validation_predictions.cpu().numpy()  # 将 tensor 转换为 numpy 数组
        # 对线性模型进行评估
        evaluate_model(y_validation, new_model_h, "Linear Model")

        # 调用评估函数进行评估
        evaluate_model(y_validation, smoothed_validation_predictions, "RF-AM-MLP")

        return smoothed_validation_predictions, y_validation, all_new_model_h

    def process_validation_data_no_rf(data_sources, scaler_no_rf, model, process_var=1, measurement_var=0.001):
        all_amplitude = []  # 振幅
        all_frequency = []  # 频率
        all_avg_C = []  # 每个窗口的电容平均值
        all_delta_C_window = []  # 每个窗口的电容首尾差值
        all_h = []  # h值
        all_new_model_h = []  # 新模型的 h 值

        for source in data_sources:
            file_path = source["file_path"]
            for sheet_name, c_range, h_range in source["ranges"]:
                C = read_excel_range(file_path, sheet_name, c_range).flatten()
                h = read_excel_range(file_path, sheet_name, h_range).flatten()

                # 计算每个点的电容差值：每个点与前一个点的差
                delta_C_window = [C[i + 1] - C[i] for i in range(len(C) - 1)]  # 差值（每个点与前一个点的差值）

                # 计算每个点的频率和振幅
                frequency, amplitude = calculate_amplitude_and_frequency_fft(C, window_size=2)

                # 对齐数据长度
                min_len = min(len(C), len(delta_C_window), len(amplitude), len(frequency), len(h))
                C = C[:min_len]
                delta_C_window = delta_C_window[:min_len]
                amplitude = amplitude[:min_len]
                frequency = frequency[:min_len]
                h = h[:min_len]

                # 计算新模型的 h 值并添加到列表中
                new_model_h = calculate_additional_model_prediction(C)

                # 将每个点的电容值、电容差值、振幅、频率、新模型的 h 值添加到列表中
                all_avg_C.extend(C)
                all_delta_C_window.extend(delta_C_window)
                all_amplitude.extend(amplitude)
                all_frequency.extend(frequency)
                all_h.extend(h)
                all_new_model_h.extend(new_model_h)

        # 转换为数组
        all_avg_C = np.array(all_avg_C).reshape(-1, 1)
        all_delta_C_window = np.array(all_delta_C_window).reshape(-1, 1)
        all_amplitude = np.array(all_amplitude).reshape(-1, 1)
        all_frequency = np.array(all_frequency).reshape(-1, 1)
        all_h = np.array(all_h)
        all_new_model_h = np.array(all_new_model_h)

        X_validation = np.hstack((all_avg_C, all_delta_C_window, all_amplitude, all_frequency))
        y_validation = all_h.flatten()

        try:
            X_validation_scaled = scaler_no_rf.transform(X_validation)
        except Exception as e:
            print(f"Error scaling validation features: {e}")
            sys.exit(1)

        X_validation_tensor = torch.tensor(X_validation_scaled, dtype=torch.float32)
        y_validation_tensor = torch.tensor(y_validation, dtype=torch.float32).view(-1)

        model.eval()
        with torch.no_grad():
            try:
                validation_predictions, attn_scores = model(X_validation_tensor) # 这里返回的是一个元组 (output, attn_scores)
                validation_predictions = validation_predictions.squeeze()  # 获取预测输出并进行 squeeze
            except Exception as e:
                print(f"Error during model prediction on validation data: {e}")
                sys.exit(1)

        smoothed_predictions = validation_predictions
        smoothed_predictions = smoothed_predictions.cpu().numpy()  # 将 tensor 转换为 numpy 数组
        evaluate_model(y_validation, smoothed_predictions, "MLP")

        return smoothed_predictions, y_validation, all_new_model_h

    # Process and store results
    results = {}
    for name, dataset in validation_datasets.items():
        smoothed_predictions_rf, y_true_rf, new_model_h_rf = process_validation_data_rf(dataset, rf_model, scaler_rf, rf_attention_model)
        smoothed_predictions_no_rf, y_true_no_rf, new_model_h_no_rf = process_validation_data_no_rf(dataset, scaler_no_rf, no_rf_no_attention_model)

        results[name] = {
            "y_true": y_true_rf,
            "RF-AM-MLP": smoothed_predictions_rf,
            "MLP": smoothed_predictions_no_rf,
            "Static linear model": new_model_h_rf,  # 添加新模型的预测
        }

    # Plot all results
    plot_all_predictions(results)
    # plot_all_residuals(results)
    # plot_all_comparisons(results)
    # plot_all_model_vs_true(results)
if __name__ == "__main__":
    main()