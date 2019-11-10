# 利用时间序列进行天气预测

# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import randrange
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.api import tsa

dataList = []

fr = open('./weatherH')

for line in fr.readlines():
    dataList.append(int(line))

data = pd.DataFrame(dataList, columns=['temperatureH'])

# 差分运算
# 默认1阶差分
data_diff = data.diff(1)

data_diff = data_diff.dropna()

data_diff.plot()
plt.show()
# 目测已经平稳

plot_acf(data_diff).show()
plot_pacf(data_diff).show()

arima = ARIMA(data, order=(2, 1, 1))
result = arima.fit(disp=False)
print(result.aic, result.bic, result.hqic)

# plt.plot(data_diff)
# plt.plot(result.fittedvalues, color='red')
# plt.title('ARIMA RSS: %.4f' % sum(result.fittedvalues - data_diff['temperatureH']) ** 2)
# plt.show()

# ARIMA   Ljung-Box检验 -----模型显著性检验，Prod> 0.05，说明该模型适合样本
resid = result.resid
r, q, p = tsa.acf(resid.values.squeeze(), qstat=True)
print(len(r), len(q), len(p))
test_data = np.c_[range(40), r[1:], q, p]
table = pd.DataFrame(test_data, columns=['lag', 'AC', 'Q', 'Prob(>Q)'])
print(table.set_index('lag'))

# 模型预测
pred = result.predict(411, 500, typ='levels')
print(pred)
x = pd.date_range(0, 500)
# plt.plot(pred)
# plt.plot(pred)
# plt.show()
print('end')