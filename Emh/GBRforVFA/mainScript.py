import numpy as np
import xlrd
import matplotlib.pyplot as plt
from random import randint
from sklearn.ensemble import GradientBoostingRegressor as GBR
from Emh.GBRforVFA.gbr4VFA import gbrSearcher

#  gbr 暴力搜参函数


# 读取数据

excel = xlrd.open_workbook('./Normalized.xlsx')
table = excel.sheet_by_index(0)

# 行
nRows = table.nrows

X = []
VFA = []
temperature = []
temperatureList = []

# 跳过列名
for it in range(1, nRows):
    try:
        X.append([float(table.cell_value(it, 1)), float(table.cell_value(it, 4)), float(table.cell_value(it, 5)),
                  float(table.cell_value(it, 8)),  # 这一列的影响不明
                  float(table.cell_value(it, 7)), float(table.cell_value(it, 11)),
                  # 入水温度和罐内温度， 我们怀疑这两个个对VFA有一定影响。罐内温度对聚类结果有很明显的影响作用
                  float(table.cell_value(it, 9)),  # 入水VFA
                  float(table.cell_value(it, 10)),  # 罐内pH，理论上这里列该是稳定的， 考虑删除
                  float(table.cell_value(it, 2)), float(table.cell_value(it, 3)), float(table.cell_value(it, 6)),
                  # 进水量、进水COD和进水pH， 这行代表系统负荷
                  float(table.cell_value(it, 12)), float(table.cell_value(it, 14)),  # 入水COD和COD去除率， 我们怀疑COD和VFA有很大关系
                  ])
        # 获取温度范围

        temperatureList.append(table.cell_value(it, 7))

        if table.cell_value(it, 11) not in temperature:
            temperature.append(table.cell_value(it, 11))

        VFA.append(table.cell_value(it, 13) * 0.9 + 0.1)  # [0,1]区间映射到[0.1,1]
    except ValueError:
        pass

rate = 0.4
size = int(rate * (nRows - 4))

"""# 生成随机数列
randList = []
leftList = []
while True:
    rand = randint(0, len(X) - 1)
    if rand in randList:
        pass
    else:
        randList.append(rand)
    if len(randList) == size:
        break

randList.sort()

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
        trainLabel.append(VFA[i])
    else:
        leftList.append(i)
        testList.append(X[i])
        testLabel.append(VFA[i])
# 调用模型
RM = GBR(n_estimators=3000, max_depth=2, min_samples_split=2, learning_rate=0.1, loss='lad')

RM.fit(np.mat(trainList), trainLabel)
y_rbf = RM.predict(np.mat(testList))

# 可视化结果
eta0 = 0.05
eta1 = 0.1

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
perErrorRate = np.array(error) / np.array(testLabel)
MeanErrorRate = np.sum(perErrorRate) / len(perErrorRate)"""

errorRecorder005 = [0]*len(X)
errorRecorder01 = [0]*len(X)

eta0 = 0.05
eta1 = 0.1

for index in range(500):
    # 生成随机数列
    randList = []
    leftList = []
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

    trainList = []
    trainLabel = []
    testList = []
    testLabel = []
    realLabel = []

    for i in range(len(X)):
        if i in randList:
            trainList.append(X[i])
            trainLabel.append(VFA[i])
        else:
            leftList.append(i)
            testList.append(X[i])
            testLabel.append(VFA[i])

    [MeanErrorRate, _, counter005, counter01, error] = gbrSearcher(trainList, trainLabel, testList, testLabel)
    print('%s, u 0.05: %s, u 0.1: %s, meanER: %s' % (index, str(counter005/len(leftList))[:5], str(counter01/len(leftList))[:5], str(MeanErrorRate)[:5]))
    for i in range(len(error)):
        if error[i] >= 0.05:
            errorRecorder005[leftList[i]] += 1
        if error[i] >= 0.1:
            errorRecorder01[leftList[i]] += 1

print(errorRecorder005)
print(errorRecorder01)

"""title = 'Mean error rate: ' + (str(MeanErrorRate))[:6] + '\nMax error rate: ' \
        + (str(perErrorRate.max()))[:6] + '\n' + 'Eta0-ErrorRate is: ' \
        + (str(counter0 / len(testLabel))[:6]) + '\n' + 'Eta1-ErrorRate is: ' \
        + (str(counter1 / len(testLabel))[:6])
plt.title(title)
plt.show()"""
