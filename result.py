import pandas as pd
import matplotlib.pyplot as plt

result = pd.read_csv('metrics.csv')
result = result.drop(result.columns[11:], axis=1)  # 删除第一列
# print(result.columns)
# 选择需要绘制的列
col = result.columns

# 绘制每一列
for i in range(len(col)):
    plt.plot(result['global_step'], result[col[i]], label=col[i])
    plt.title(col[i])
    plt.show()

# 每22250行计算一次平均值
def calculate_average(df, interval):
    return df.groupby(df.index // interval).mean()

# # 计算平均值
# average_df = calculate_average(result, 22250 // 50)
# print(average_df.head())
# # 绘制平均值
# for i in range(len(col)):
#     plt.plot(average_df['global_step'], average_df[col[i]], label=col[i])
#     plt.title(col[i])
#     plt.show()