from random import randint

import numpy as np
import target as target
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import xlrd

# 读取数据
excel = xlrd.open_workbook('./warm.xlsx')
#excel = xlrd.open_workbook('./#1decentralization++.xlsx')
table = excel.sheet_by_index(0)

# 行
nRows = table.nrows

X = []
X1 = []
y = []



# 数据从第五行开始
for it in range(nRows):

    X.append([float(table.cell_value(it, 1)), float(table.cell_value(it, 4)), float(table.cell_value(it, 5)),
              float(table.cell_value(it, 6)), float(table.cell_value(it, 7)), float(table.cell_value(it, 8)) ,
              float(table.cell_value(it, 9)), float(table.cell_value(it, 10)),float(table.cell_value(it, 11))]
             )
    # float(table.cell_value(it, 9)), float(table.cell_value(it, 10))])
    y.append(float(table.cell_value(it, 13)))







'''
Cols = []

for it in range(1, nRows):
    temp = []
    try:
        for jt in range(1, 15):#不要当地温度,VFA为第13列
            if table.cell_value(it, 0) != '':
                temp.append(float(table.cell_value(it, jt)))
        Cols.append(temp)
    except ValueError as ve:
        pass
matrixTable = np.mat(Cols)
n, m = np.shape(matrixTable)
for i in range(m):
    matrixTable[:, i] = matrixTable[:, i]/np.sum(matrixTable[:, i])

X = matrixTable[:, :-2]
y = matrixTable[:, 12]
# SVR
'''

rate = 0.7
size = int(rate * (nRows - 4))

# 生成随机数列
randList = []
while True:
    rand = randint(0, len(X)-1)
    if rand in randList:
        pass
    else:
        randList.append(rand)
    if len(randList) == size:
        break


randList.sort()

trainList = []
trainLabel = []
trainList1 = []
trainLabel1 = []
testList = []
testLabel = []
testList1 = []
testLabel1 = []

'''for i in range(len(X)):
    if i in randList:
        if X[i][-1] > 33:
            trainList.append(X[i])
            trainLabel.append(y[i])
        else:
            trainList1.append(X[i])
            trainLabel1.append(y[i])

    else:
        if X[i][-1] > 33:
            testList.append(X[i])
            testLabel.append(y[i])
        else:
            testList1.append(X[i])
            testLabel1.append(y[i])
'''

for i in range(len(X)):
    if i in randList:
        trainList.append(X[i])
        trainLabel.append(y[i])
    else:
        testList.append(X[i])
        testLabel.append(y[i])



# 调用模型
#svr_rbf = SVR(C=0.025, epsilon=0.0002, gamma=2,  kernel='rbf', max_iter=500, shrinking=True, tol=0.005, )
gbr = GradientBoostingRegressor(learning_rate=0.08, n_estimators=120,max_depth=9, subsample=0.85, random_state=10)  # 建立梯度增强回归模型对象
#gbr = GradientBoostingRegressor(learning_rate=0.06)

gbr.fit(np.mat(trainList), trainLabel)
gbr_rbf = gbr.predict(np.mat(testList))

'''
gbr1 = GradientBoostingRegressor(learning_rate=0.08, n_estimators=100,max_depth=9, subsample=0.85, random_state=10)  # 建立梯度增强回归模型对象
gbr1.fit(np.mat(trainList1), trainLabel1)
gbr1_rbf = gbr.predict(np.mat(testList1))
'''


#svr_rbf.fit(np.mat(trainList), trainLabel)
#y_rbf = svr_rbf.predict(np.mat(testList))

# 可视化结果
eta0 = 0.05
eta1 = 0.1

lw = 2
plt.plot([i for i in range(len(testLabel))], testLabel, color='darkorange', label='Real Data')
plt.plot([i for i in range(len(testLabel))], gbr_rbf, color='navy', lw=lw, label='RBF predict')
#plt.scatter([i for i in range(len(testLabel))], y_rbf, color='navy', lw=lw, label='RBF predict')
plt.xlabel('number')
plt.ylabel('COD_Out')
#plt.title('SVR for COD OUT')
plt.title('GBR for VFA OUT')
plt.legend()
plt.show()
error = np.abs(np.array(testLabel)-np.array(gbr_rbf))
#error = np.abs(np.array(testLabel)-np.array(y_rbf))
plt.plot([i for i in range(len(testLabel))], [eta0]*len(testLabel), color='navy')
plt.plot([i for i in range(len(testLabel))], [eta1]*len(testLabel), color='navy')
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
