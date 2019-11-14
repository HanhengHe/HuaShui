from random import randint

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import xlrd

# 读取数据
#excel = xlrd.open_workbook('./warm.xlsx')
excel = xlrd.open_workbook('./all.xlsx')
table = excel.sheet_by_index(0)

# 行
nRows = table.nrows

X = []
X1 = []
y = []

#  数据从第五行开始
for it in range(nRows):
    X.append([float(table.cell_value(it, 1)), float(table.cell_value(it, 4)), float(table.cell_value(it, 5)),
              float(table.cell_value(it, 6)), float(table.cell_value(it, 7)), float(table.cell_value(it, 8)),
              float(table.cell_value(it, 9)), float(table.cell_value(it, 10)), float(table.cell_value(it, 11))]
             )
    # float(table.cell_value(it, 9)), float(table.cell_value(it, 10))])
    y.append(float(table.cell_value(it, 13)))

mat = np.mat(X)
n, m = np.shape(mat)
for i in range(m):
    mat[:, i] = (mat[:, i] - np.mean(mat[:, i])) / np.mean(mat[:, i])

sum = 0
for it in y:
    sum += it
minY = min(y)
maxY = max(y)
for i in range(len(y)):
    y[i] = (y[i] - minY + 0.01) / (maxY - minY)

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
size = int(rate * (nRows))

# 生成随机数列
randList = []
while True:
    rand = randint(0, n - 1)
    if rand in randList:
        pass
    else:
        randList.append(rand)
    if len(randList) == size:
        break

randList.sort()
#randList = [i * 3 for i in range(int(nRows / 3))]

trainList = []
trainLabel = []
testList = []
testLabel = []

for i in range(n):
    if i in randList:
        trainList.append(mat[i, :].tolist()[0])
        trainLabel.append(y[i])
    else:
        testList.append(mat[i, :].tolist()[0])
        testLabel.append(y[i])

# 调用模型
# gbr = SVR(C=5, epsilon=0.2, gamma=3,  kernel='rbf', max_iter=500, shrinking=True, tol=0.005, )
 #+ j * 500,
grid = []
for i in range(4):
    for j in range(5):
            gbr = GradientBoostingRegressor(learning_rate=0.1 + i * 0.05, max_depth=9 + j * 1,
                                            subsample=0.85,
                                            random_state=10)
            gbr.fit(np.mat(trainList), trainLabel)
            gbr_predict = gbr.predict(np.mat(testList))
            counter = 0
            for e in np.abs(np.array(testLabel) - np.array(gbr_predict)):
                if e >= 0.1:
                    counter += 1
            print(counter/len(testList))
            grid.append([counter/len(testList), 0.1 + i * 0.05, 9 + j * 1])

minG = 10000
index = 0
for i in range(len(grid)):
    if grid[i][0] <= minG:
        minG = grid[i][0]
        index = i

print(grid[index])


gbr = GradientBoostingRegressor(learning_rate=grid[index][1], max_depth=grid[index][2],n_estimators=120,  subsample=0.85, random_state=10)  # 建立梯度增强回归模型对象
# gbr = GradientBoostingRegressor(learning_rate=0.06)

gbr.fit(np.mat(trainList), trainLabel)
gbr_predict = gbr.predict(np.mat(testList))

'''
gbr1 = GradientBoostingRegressor(learning_rate=0.08, n_estimators=100,max_depth=9, subsample=0.85, random_state=10)  # 建立梯度增强回归模型对象
gbr1.fit(np.mat(trainList1), trainLabel1)
gbr1_rbf = gbr.predict(np.mat(testList1))
'''

# svr_rbf.fit(np.mat(trainList), trainLabel)
# y_rbf = svr_rbf.predict(np.mat(testList))

# 可视化结果
eta0 = 0.15
eta1 = 0.1

lw = 2
error = np.abs(np.array(testLabel) - np.array(gbr_predict))
plt.plot([i for i in range(len(testLabel))], testLabel, color='darkorange', label='Real Data')
plt.plot([i for i in range(len(testLabel))], gbr_predict, color='navy', label='predict')
#plt.plot([i for i in range(len(testLabel))], error, color='green', label='error')
# plt.scatter([i for i in range(len(testLabel))], y_rbf, color='navy', lw=lw, label='RBF predict')
plt.xlabel('number')
plt.ylabel('COD_Out')
# plt.title('SVR for COD OUT')
plt.title('GBR for VFA OUT')
plt.legend()
plt.show()

# error = np.abs(np.array(testLabel)-np.array(y_rbf))
plt.plot([i for i in range(len(testLabel))], [eta0] * len(testLabel), color='navy')
plt.plot([i for i in range(len(testLabel))], [eta1] * len(testLabel), color='navy')
plt.scatter([i for i in range(len(testLabel))], error, color='darkorange', lw=lw, label='error')
perErrorRate = error / np.array(np.abs(testLabel))
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
