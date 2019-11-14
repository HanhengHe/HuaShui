from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import xlrd
from sklearn.svm import SVR
import numpy as np
from random import randint

# 读取数据

excel = xlrd.open_workbook('./#1decentralization++.xlsx')
table = excel.sheet_by_index(0)

# 行
nRows = table.nrows

X = []
testVFA = []
VFA = []

# 数据从第五行开始
for it in range(nRows):
    if table.cell_value(it, 13) == 0:
        continue
    X.append([float(table.cell_value(it, 0)), float(table.cell_value(it, 1)), float(table.cell_value(it, 2)),
              float(table.cell_value(it, 3)), float(table.cell_value(it, 4)), float(table.cell_value(it, 5)),
              float(table.cell_value(it, 6)), float(table.cell_value(it, 7)), float(table.cell_value(it, 8)),])
              # float(table.cell_value(it, 9)), float(table.cell_value(it, 10))])
    # testVFA.append([float(table.cell_value(it, 13)), float(table.cell_value(it, 13))])
    VFA.append(float(table.cell_value(it, 13)))

y_Predict = KMeans(n_clusters=3).fit_predict(X)

plt.plot([i for i in range(len(VFA))], VFA)
plt.scatter([i for i in range(len(VFA))], VFA, c=y_Predict)
plt.plot([i for i in range(len(VFA))], [0.37037037037037035]*len(VFA))
plt.title('cluster without vfa')
plt.show()

type1 = []
type2 = []

for i in range(len(y_Predict)):
    if y_Predict[i] == y_Predict[0]:
        type1.append(i)
    else:
        type2.append(i)

rate = 0.4
size = int(rate * (len(type1)))
randList = []

"""while True:
    rand = randint(0, len(type1) - 1)
    if rand in randList:
        pass
    else:
        randList.append(rand)
    if len(randList) == size:
        break

randList.sort()"""

for i in range(size):
    randList.append(type1[i])

trainList = []
trainLabel = []
testList = []
testLabel = []

for i in type1:
    if i in randList:
        trainList.append(X[i])
        trainLabel.append(VFA[i])
    else:
        testList.append(X[i])
        testLabel.append(VFA[i])

svr_rbf1 = SVR(C=0.2, epsilon=0.0002, gamma=3, kernel='rbf', max_iter=500, shrinking=True, tol=0.005, )

svr_rbf1.fit(np.mat(trainList), trainLabel)
y_rbf = svr_rbf1.predict(np.mat(testList))

# 可视化结果
eta0 = 0.05
eta1 = 0.1

lw = 2
plt.plot([i for i in range(len(testLabel))], testLabel, color='darkorange', label='Real Data')
plt.scatter([i for i in range(len(testLabel))], y_rbf, color='navy', lw=lw, label='RBF predict')
plt.xlabel('number')
plt.ylabel('VFA_Out')
plt.title('SVR for VFA OUT')
plt.legend()
plt.show()

error = np.abs(np.array(testLabel) - np.array(y_rbf))
plt.plot([i for i in range(len(testLabel))], [eta0] * len(testLabel), color='navy')
plt.plot([i for i in range(len(testLabel))], [eta1] * len(testLabel), color='navy')
plt.scatter([i for i in range(len(testLabel))], error, color='darkorange', lw=lw, label='error')
perErrorRate = error / np.array(testLabel)
MeanErrorRate = np.sum(perErrorRate) / len(testLabel)
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
