#  尝试计算VFA的差分情况

import numpy as np
import matplotlib.pyplot as plt
import xlrd
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.svm import SVR

# 读取数据

excel = xlrd.open_workbook('./#1decentralization++.xlsx')
table = excel.sheet_by_index(0)

# 行
nRows = table.nrows

X = []
VFA = []
VFA_diff = []

# 数据从第五行开始
for it in range(nRows-1):
    if table.cell_value(it+1, 15) - table.cell_value(it, 15) != 1:
        continue
    X.append([float(table.cell_value(it, 0)), float(table.cell_value(it, 1)), float(table.cell_value(it, 2)),
              float(table.cell_value(it, 3)), float(table.cell_value(it, 4)), float(table.cell_value(it, 5)),
              float(table.cell_value(it, 6)), float(table.cell_value(it, 7)), float(table.cell_value(it, 8)),
              float(table.cell_value(it, 13)),
              float(table.cell_value(it+1, 0)), float(table.cell_value(it+1, 1)), float(table.cell_value(it+1, 2)),
              float(table.cell_value(it+1, 3)), float(table.cell_value(it+1, 4)), float(table.cell_value(it+1, 5)),
              float(table.cell_value(it+1, 6)), float(table.cell_value(it+1, 7)), float(table.cell_value(it+1, 8)),
              ])
    VFA.append(table.cell_value(it, 13))
    VFA_diff.append(table.cell_value(it+1, 13)-table.cell_value(it, 13))

VFA.append(table.cell_value(nRows-1, 13))

plt.plot([i for i in range(len(VFA_diff))], VFA_diff)
plt.title('diff_1')
plt.show()

# SVRforCOD

rate = 0.4
size = int(rate * (nRows - 4))

# 生成随机数列
randList = []
"""while True:
    rand = randint(0, len(X) - 1)
    if rand in randList:
        pass
    else:
        randList.append(rand)
    if len(randList) == size:
        break

randList.sort()"""

for i in range(size):
    randList.append(i)

trainList = []
trainLabel = []
testList = []
testLabel = []
realLabel = []

for i in range(len(X)):
    if i in randList:
        trainList.append(X[i])
        trainLabel.append(VFA_diff[i])
    else:
        testList.append(X[i])
        testLabel.append(VFA_diff[i])
        realLabel.append(VFA[i+1])


# 调用模型
# RM = GBR(n_estimators=3000, max_depth=2, min_samples_split=2, learning_rate=0.1, loss='lad')
RM = SVR(C=0.2, epsilon=0.0002, gamma=2, kernel='rbf', max_iter=500, shrinking=True, tol=0.005, )

RM.fit(np.mat(trainList), trainLabel)
y_rbf = RM.predict(np.mat(testList))

# 可视化结果
eta0 = 0.05
eta1 = 0.1

"""RM1 = SVR(C=0.2, epsilon=0.0002, gamma=2, kernel='rbf', max_iter=500, shrinking=True, tol=0.005, )
gap = np.array(testLabel) - np.array(y_rbf)

RM1.fit(np.mat(testList), gap)
y_rbf += RM.predict(np.mat(testList))"""

lw = 2
error = np.abs(np.array(y_rbf) - np.array(testLabel))
plt.plot([i for i in range(len(testLabel))], testLabel, color='darkorange', label='Real Data')
plt.plot([i for i in range(len(testLabel))], y_rbf, color='navy', lw=lw, label='predict')
plt.plot([i for i in range(len(testLabel))], error, color='green', lw=lw, label='error')
plt.xlabel('NO.')
plt.ylabel('VFA_Out')
plt.title('VFA OUT')
plt.legend()
plt.show()

plt.plot([i for i in range(len(testLabel))], [eta0] * len(testLabel), color='navy')
plt.plot([i for i in range(len(testLabel))], [eta1] * len(testLabel), color='navy')
plt.scatter([i for i in range(len(testLabel))], error, color='darkorange', lw=lw, label='error')
perErrorRate = error / np.array(realLabel)
MeanErrorRate = np.sum(perErrorRate) / len(perErrorRate)

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
        + (str(counter0 / len(testLabel))[:6]) + '\n' + 'Eta1-ErrorRate is: ' \
        + (str(counter1 / len(testLabel))[:6])
plt.title(title)
plt.show()

