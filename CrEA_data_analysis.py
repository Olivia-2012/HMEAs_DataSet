import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import chardet
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import precision_score,recall_score,f1_score,confusion_matrix
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import QuantileTransformer
import lightgbm as lgb
import configparser
import os.path
from math import sqrt
from sklearn.metrics import accuracy_score
import sys
import chardet
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
import scipy
from scipy import stats
from scipy.stats import norm
from matplotlib.ticker import MaxNLocator, MultipleLocator
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import smogn
from imblearn.over_sampling import SMOTE
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import xlsxwriter
from matplotlib.ticker import FuncFormatter

predata = pd.read_excel('D:\Doctor\PMHMs-DataSet\data_new\PMCr.xlsx')


# 删除PMCr列下值为0的行
predata = predata[predata['PMCr'] != 0]

# 要剔除异常值的列
columns_to_process = ['PMCr', 'indSO2', 'indNOx', 'carNOx', 'carSmoke', 'pop',  'temp', 'rh', 'sd', 'wsp', 'preci']

# 逐列处理数据
for column_name in columns_to_process:
    median = predata[column_name].median()
    std_dev = predata[column_name].std()
    lower_bound = median - 3 * std_dev
    upper_bound = median + 3 * std_dev
    predata = predata[(predata[column_name] >= lower_bound) & (predata[column_name] <= upper_bound)]

columns_to_delete = ['pro', 'city']
data = predata.drop(columns=columns_to_delete)

# 自定义分箱的数据范围和标签
bin_ranges = [38, 800, 1500, 2500, 5000, 10000, 15000, 30000, 60000, 85374] # As数据扩充范围
bin_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8']

# 使用自定义数据范围和标签进行分箱
data['PMCr_binned_labels'] = pd.cut(data['PMCr'], bins=bin_ranges, labels=bin_labels)

# 统计每个分箱中的数据数量
bin_counts = data['PMCr_binned_labels'].value_counts()

# 对分箱标签按照你指定的顺序排序
bin_counts = bin_counts.reindex(bin_labels)

# 定义区间范围列表
interval_ranges = [f"{start} - {end}" for start, end in zip(bin_ranges, bin_ranges[1:])]

# 输出每个分箱的标签、数量和数据范围
for bin_label, count, data_range in zip(bin_counts.index, bin_counts.values, interval_ranges):
    print(f"分箱标签: {bin_label}, 分箱数量: {count}, 数据范围: {data_range}")

# 计算每个分箱的均值和标准差
bin_means = data.groupby('PMCr_binned_labels')['PMCr'].transform('mean')
bin_std = data.groupby('PMCr_binned_labels')['PMCr'].transform('std')

# 定义浮动因子
std_multiplier = 0.2  # 可以根据需求调整

# 计算每行的估计值
pmcd_estimates = []

for index, row in data.iterrows():
    mean = bin_means[index]
    std = bin_std[index] if bin_std[index] > 0 else 1  # 将标准差小于等于0的情况设置为1
    estimate = mean + np.random.uniform(-std_multiplier * std, std_multiplier * std)
    pmcd_estimates.append(estimate)

data['PMCr_estimate'] = pmcd_estimates

# 输出每行的估计值
for index, row in data.iterrows():
    print(f"真实值: {row['PMCr']}, 行索引: {index}, 分箱标签: {row['PMCr_binned_labels']}, 估计值: {row['PMCr_estimate']}")

# 分割自变量和因变量
X = data[['indSO2', 'indNOx', 'carNOx', 'carSmoke', 'pop', 'temp', 'rh', 'sd', 'wsp', 'preci']]
y = data['PMCr_binned_labels']

# 使用SMOTE生成合成样本
smote = SMOTE(sampling_strategy='auto', k_neighbors=3, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 将生成的合成样本添加回原始数据集
resampled_data = pd.concat([X_resampled, y_resampled], axis=1)
final_data = pd.concat([data, resampled_data], ignore_index=True)

# 创建一个布尔列'is_real'，用于标识真实数据，用于区分真实数据和合成数据
final_data['is_real'] = True  

# 标记合成数据为False
final_data.loc[len(final_data) - len(resampled_data):, 'is_real'] = False

# 定义一个函数，用于生成随机数
def generate_random_estimate(row):
    if pd.isna(row['PMCr_estimate']):
        bin_label = row['PMCr_binned_labels']
        mean = bin_means[bin_label]
        std = bin_std[bin_label]
        # 计算估计值为均值加减0.2倍标准差的浮动值
        estimated_value = mean + np.random.uniform(-0.2 * std, 0.2 * std)
        return estimated_value
    else:
        return row['PMCr_estimate']

# 计算每个分箱的均值和标准差
bin_means = final_data.groupby('PMCr_binned_labels')['PMCr'].mean()
bin_std = final_data.groupby('PMCr_binned_labels')['PMCr'].std()

# 使用 apply 方法为 PMAs_estimate 列的空值生成随机估计值
final_data['PMCr_estimate'] = final_data.apply(generate_random_estimate, axis=1)

final_data.to_excel('Cr_expanded_data2.xlsx', index=False)

expanded_data = pd.read_excel("D:\Doctor\PMHMs-DataSet\data_new\Cr_expanded_data2.xlsx")

expanded_data.head()

# 将 PMCr_all 转换为以t为单位
expanded_data['PMCr_all'] = expanded_data['PMCr_all'] / 1e3

# 创建一个新的图形
plt.figure(figsize=(15, 5))

# 设置子图
plt.subplot(131)  # 一行三列中的第一个图
plt.hist(expanded_data[expanded_data['is_real']]['PMCr_all'], bins=10, color='green', alpha=0.5, label='PMCr (Real)', 
         edgecolor='black', hatch='//')
plt.xlabel('Real PMCr values (t)')
plt.ylabel('Frequency')
plt.xlim([min(expanded_data['PMCr_all']), max(expanded_data['PMCr_all'])])
plt.ylim([0, 160])

plt.subplot(132)  # 一行三列中的第二个图
plt.hist(expanded_data[~expanded_data['is_real']]['PMCr_all'], bins=9, color='blue', alpha=0.5, label='PMCr (Synthetic)', 
         edgecolor='black', hatch='\\')
plt.xlabel('Synthetic PMCr values (t)')
plt.xlim([min(expanded_data['PMCr_all']), max(expanded_data['PMCr_all'])])
plt.ylim([0, 160])  

plt.subplot(133)  # 一行三列中的第三个图
plt.hist(expanded_data['PMCr_all'], bins=10, color='yellow', alpha=0.5, label='PMCr (All)', edgecolor='black', hatch='--')
plt.xlabel('Total PMCr values (t)')
plt.xlim([min(expanded_data['PMCr_all']), max(expanded_data['PMCr_all'])])
plt.ylim([0, 160])  

# 显示图形
plt.tight_layout()
plt.show()

# 列出需要进行对数转换的列
log_columns = ['PMCr_all', 'indSO2', 'indNOx', 'carNOx', 'carSmoke', 'pop', 'sd', 'wsp', 'preci', 'temp', 'rh']

# 进行对数转换
for col in log_columns:
    expanded_data[col + '_log'] = np.log(expanded_data[col])

# 检查 'PMAs_log' 列是否包含 NaN 值
has_nan = expanded_data['PMCr_all_log'].isna().any()

if has_nan:
    print("Column 'PMCr_all_log' contains NaN values.")
else:
    print("Column 'PMCr_all_log' does not contain NaN values.")

# 删除包含空值的行
expanded_data.dropna(subset=['PMCr_all_log'], inplace=True)

true_count = expanded_data['is_real'].sum()
print("Number of True values in 'is_real' column:", true_count)

# 定义模型列表
models = [
    LinearRegression(),
    Ridge(),
    BayesianRidge(),
    Lasso(),
    ElasticNet(),
    RandomForestRegressor(n_estimators=200),
    KNeighborsRegressor(),
    MLPRegressor(),
    LGBMRegressor()
]

# 定义性能指标的列表
metrics = {
    "Model": [],
    "R2_test": [],
    "RMSE_test": [],
    "MAE_test": [],
    "R2_train": [],
    "RMSE_train": [],
    "MAE_train": []
}
# 创建一个空的DataFrame来存储模型参数
model_params_Cr = pd.DataFrame(columns=["Model", "Parameters"])

# 训练和评估模型
n_repeats = 10  # 每个模型重复训练的次数

best_model = None
min_rmse_test = float('inf')  # 初始化为正无穷

for model in models:
    for _ in range(n_repeats):
        # 确保训练集包括70%真实数据和全部合成数据,验证集包括30%真实数据
        # 真实数据的索引
        real_data_index = expanded_data[expanded_data['is_real']].index
        # 随机划分真实数据为训练集（70%）和验证集（30%）
        train_real_index, test_real_index = train_test_split(real_data_index, test_size=0.5)   
        # 合成数据的索引
        synthetic_data_index = expanded_data[~expanded_data['is_real']].index
        # 组合真实数据和合成数据的索引来创建训练集
        train_data_index = list(train_real_index) + list(synthetic_data_index)
        # 创建训练集和验证集
        train_data = expanded_data.loc[train_data_index]
        test_data = expanded_data.loc[test_real_index]
        # 分离自变量和因变量
        X_train = train_data[['indSO2_log', 'indNOx_log', 'carNOx_log', 'carSmoke_log', 'pop', 'sd_log', 'wsp_log', 'preci_log', 'temp', 'rh']]
        y_train = train_data['PMCr_all_log']
        X_test = test_data[['indSO2_log', 'indNOx_log', 'carNOx_log', 'carSmoke_log', 'pop', 'sd_log', 'wsp_log', 'preci_log', 'temp', 'rh']]
        y_test = test_data['PMCr_all']

        # 创建模型实例并训练模型        
        model.fit(X_train, y_train)
        
        # 输出模型参数
        model_params = model.get_params()
        model_params_Cr = model_params_Cr.append({"Model": type(model).__name__, "Parameters": model_params}, ignore_index=True)

        # 预测
        y_train_original = np.exp(y_train)
        y_train_pred = model.predict(X_train)
        y_train_pred_original = np.exp(y_train_pred)
        y_test_pred = model.predict(X_test)
        y_test_pred_original = np.exp(y_test_pred)

        # 评估模型性能
        r2_train = r2_score(y_train_original, y_train_pred_original)
        rmse_train = mean_squared_error(y_train_original, y_train_pred_original, squared=False)
        mae_train = mean_absolute_error(y_train_original, y_train_pred_original)        
        
        r2_test = r2_score(y_test, y_test_pred_original)
        rmse_test = mean_squared_error(y_test, y_test_pred_original, squared=False)
        mae_test = mean_absolute_error(y_test, y_test_pred_original)

        # 将性能指标和模型名称添加到列表
        metrics["Model"].append(type(model).__name__)
        metrics["R2_test"].append(r2_test)
        metrics["RMSE_test"].append(rmse_test)
        metrics["MAE_test"].append(mae_test)
        metrics["R2_train"].append(r2_train)
        metrics["RMSE_train"].append(rmse_train)
        metrics["MAE_train"].append(mae_train)

        # 检查当前模型在测试集上是否具有更小的RMSE
        if rmse_test < min_rmse_test:
            min_rmse_test = rmse_test
            best_model = model

# 保存最佳模型到文件
joblib.dump(best_model, 'best_model_rmse_Cr.joblib')

# 创建DataFrame
metrics_df = pd.DataFrame(metrics)
print(metrics_df)

# 保存模型参数到Excel文件
model_params_Cr.to_excel("model_parameters_Cr.xlsx", index=False)

# 计算每个模型的性能指标均值
average_metrics = metrics_df.groupby('Model').mean().reset_index()
print(average_metrics)

# 保存为Excel文件
# average_metrics.to_excel('Cr_model_metrics_averages.xlsx', index=False)
# metrics_df.to_excel('Cr_model_metrics.xlsx', index=False)

# 加载保存的最佳模型
# mymodel = joblib.load('best_model_rmse_Cr.joblib')

# 计算95%置信区间
lower, upper = np.percentile(y_test_pred_original, [2.5, 97.5])

# PICP
true_values_inside_interval = np.logical_and(y_test >= lower, y_test <= upper)
picp = np.mean(true_values_inside_interval)

# 计算不在置信区间内的数据
outside_interval = np.logical_not(true_values_inside_interval)
num_outside_interval = np.sum(outside_interval)

# 打印结果
print(f"Confidence Interval: [{lower}, {upper}]")
print(f"PICP: {picp}")
print(f"True values inside interval: {np.sum(true_values_inside_interval)} out of {len(y_test)}")
print(f"True values outside interval: {num_outside_interval} out of {len(y_test)}")

# 将结果存储到字典中
result_dict_Cd = {
    "Confidence Interval Lower": [lower],
    "Confidence Interval Upper": [upper],
    "PICP": [picp],
    "True Values Inside Interval": [np.sum(true_values_inside_interval)],
    "True Values Outside Interval": [num_outside_interval]
}

# 创建DataFrame
result_df = pd.DataFrame(result_dict_Cd)

# 将DataFrame保存到Excel文件
result_df.to_excel("PICP_results_Cr.xlsx", index=False)

y_train_pred_rf = mymodel.predict(X_train)
y_train_pred_original_rf = np.exp(y_train_pred)
y_test_pred_rf = mymodel.predict(X_test)
y_test_pred_original_rf = np.exp(y_test_pred)

data_predict = pd.read_excel(r"D:\Doctor\PMHMs-DataSet\data_new\2015input.xlsx")

data_predict.columns

# 列出需要进行对数转换的列
log_columns = ['indSO2', 'indNOx', 'carNOx', 'carSmoke', 'sd', 'wsp', 'preci']

# 进行对数转换
for col in log_columns:
    data_predict[col + '_log'] = np.log(data_predict[col])

columns_to_delete = ['省', '市', 'indSO2', 'indNOx', 'carNOx', 'carSmoke', 'sd', 'wsp', 'preci']
data_predict = data_predict.drop(columns=columns_to_delete)

# 加载保存的最佳模型
mymodel = joblib.load('best_model_rmse_Cr.joblib')

# 提取自变量数据
X = data_predict[['indSO2_log', 'indNOx_log', 'carNOx_log', 'carSmoke_log', 'pop', 
                  'sd_log', 'wsp_log', 'preci_log', 'temp', 'rh']]

# 使用模型进行预测
y_pred = mymodel.predict(X)

# 将预测结果添加到新数据中
data_predict['RF_PMCr_2015'] = y_pred

# 打印包含预测值的新数据
data_predict.to_excel('Predicted_PMCr_2015.xlsx', index=False)

