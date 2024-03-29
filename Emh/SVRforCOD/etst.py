from random import randint

import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import xlrd

# 读取数据

excel = xlrd.open_workbook('./#1decentralization++.xlsx')
table = excel.sheet_by_index(0)

# 行
nRows = table.nrows

X = []
y = []

# 数据从第五行开始
for it in range(nRows):
    X.append([float(table.cell_value(it, 0)), float(table.cell_value(it, 1)), float(table.cell_value(it, 2)),
              float(table.cell_value(it, 3)), float(table.cell_value(it, 4)), float(table.cell_value(it, 5)),
              float(table.cell_value(it, 6)), float(table.cell_value(it, 7)), float(table.cell_value(it, 8)), ])
              # float(table.cell_value(it, 9)), float(table.cell_value(it, 10))])
    y.append(float(table.cell_value(it, 12)))

# SVRforCOD

rate = 0.4
size = int(rate * (nRows - 4))

# 生成随机数列
randList = []
while True:
    rand = randint(0, len(X) - 1)
    if rand in randList:
        pass
    else:
        randList.append(rand)
    if len(randList) == size:
        break

randList.sort()

"""for i in range(size):
    randList.append(i)"""

# 调用模型
svr_rbf = SVR(C=0.1, epsilon=0.0002, gamma=2, kernel='rbf', max_iter=500, shrinking=True, tol=0.005, )

svr_rbf.fit(np.mat(X), y)
y_rbf = svr_rbf.predict(np.mat(X))

# 可视化结果
eta0 = 0.05
eta1 = 0.1

lw = 2
plt.scatter([i for i in range(len(y))], y, color='darkorange', label='Real Data')
plt.scatter([i for i in range(len(y))], y_rbf, color='navy', lw=lw, label='RBF predict')
plt.xlabel('number')
plt.ylabel('COD_Out')
plt.title('SVRforCOD for COD OUT')
plt.legend()
plt.show()

error = np.abs(np.array(y) - np.array(y_rbf))
plt.plot([i for i in range(len(y))], [eta0] * len(y), color='navy')
plt.plot([i for i in range(len(y))], [eta1] * len(y), color='navy')
plt.scatter([i for i in range(len(y))], error, color='darkorange', lw=lw, label='error')
perErrorRate = error / np.array(y)
MeanErrorRate = np.sum(perErrorRate) / len(y)

counter0 = 0
for e in error:
    if e >= eta0:
        counter0 += 1

counter1 = 0
for e in error:
    if e >= eta1:
        counter1 += 1

title = 'Mean error rate: ' + (str(MeanErrorRate))[:6] + '\nMax error rate: ' \
        + (str(perErrorRate.max()))[:6] + '\n' + 'Eta0-ErrorRate is: ' \
        + (str(counter0 / len(y))[:6]) + '\n' + 'Eta1-ErrorRate is: ' \
        + (str(counter1 / len(y))[:6])
plt.title(title)
plt.show()

outer = []

for i in range(len(error)):
    if error[i] > 0.05:
        outer.append(i+5)

print(outer)
