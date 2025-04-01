#Read me
#This part uses the measured data at 10rpm to fit the linear relationship between liquid level and capacitance
import openpyxl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 读取 Excel 文件中液位和电容数据的函数
def read_excel_range(file_path, sheet_name, capacitance_range, level_range):
    wb = openpyxl.load_workbook(file_path, data_only=True)
    sheet = wb[sheet_name]

    # 解析液位和电容范围
    level_cells = sheet[level_range]
    capacitance_cells = sheet[capacitance_range]

    # 将单元格数据转换为列表
    levels = [cell[0].value for cell in level_cells]
    capacitance = [cell[0].value for cell in capacitance_cells]

    # 转换为 NumPy 数组
    levels = np.array(levels, dtype=float)
    capacitance = np.array(capacitance, dtype=float)

    # 去除无效值（None 或 NaN）
    valid_indices = ~np.isnan(levels) & ~np.isnan(capacitance)
    levels = levels[valid_indices]
    capacitance = capacitance[valid_indices]

    return levels, capacitance

# 定义线性回归模型拟合 h = a * C + b
def fit_linear_model(C, h):
    model = LinearRegression()
    C = C.reshape(-1, 1)  # 将 C 转换为二维数组，以便线性回归模型拟合
    model.fit(C, h)
    a = model.coef_[0]  # 获取拟合的斜率
    b = model.intercept_  # 获取拟合的截距

    return a, b

# 创建一个函数来绘制拟合结果
def plot_fit(data_dict):
    plt.figure(figsize=(10, 6))

    # 遍历数据字典进行拟合和绘图
    for data in data_dict:
        file_path = data["file_path"]
        ranges = data["ranges"][0]  # 只取第一个范围
        sheet_name = ranges[0]
        capacitance_range_cells = ranges[1]
        level_range_cells = ranges[2]

        # 读取C和h数据
        levels, capacitance = read_excel_range(file_path, sheet_name, capacitance_range_cells, level_range_cells)

        # 使用线性模型拟合 h 和 C 的关系
        a, b = fit_linear_model(capacitance, levels)
        print(f"Fitted {data['label']} equation: h = {a:.4f} * C + {b:.4f}")
        # 绘制原始数据和拟合结果
        plt.plot(capacitance, levels, label=f'{data["label"]}', color=data["color"], alpha=0.7, linewidth=3)
        # plt.plot(capacitance, a * capacitance + b, label=f'Fitted {data["label"]}: h = {a:.4f} * C + {b:.4f}', color=data["color"], linestyle='--', linewidth=3)

    # 设置图表标签和标题
    plt.title("", fontsize=22)
    plt.xlabel('Capacitance (pF)', fontsize=20)
    plt.ylabel('Liquid level (mm)', fontsize=20)
    # plt.legend(fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 数据字典：包含文件路径、范围、标签和颜色
data_dict = [
    {
        "file_path": "C:/Users/hs/Desktop/Static.xlsx",#You can put the file to your own desktop and change the path to your desktop accordingly
        "ranges": [("Sheet1", "B2:B8489", "C2:C8489")],
        "label": "10 rpm",  # 第一组数据标注
        "color": "blue",     # 蓝色
    },
    # 可以继续添加其他数据组
]

# 调用绘图函数进行拟合和绘制
plot_fit(data_dict)
