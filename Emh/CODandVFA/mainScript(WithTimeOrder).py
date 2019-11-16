import numpy as np
import xlrd
import xlwt
import matplotlib.pyplot as plt
from random import randint

from Emh.CODandVFA.gbr4VFA import gbrSearcher
from Emh.CODandVFA.svr4VFA import svrSearcher

# 读取数据

excel = xlrd.open_workbook('./Normalized(separated).xlsx')
table = excel.sheet_by_index(0)

# 行
nRows = table.nrows

X = []
VFA = []
temperature = []
temperatureList = []

# 跳过列名
# 2 days
for it in range(1, nRows):
    try:
        if True: # table.cell_value(it + 1, 0) - table.cell_value(it,
                                                          # 0) == 1 and table.cell_value(it+2, 0)-table.cell_value(it+1, 0) == 1:
            X.append([float(table.cell_value(it, 1)), float(table.cell_value(it, 4)), float(table.cell_value(it, 5)),
                      float(table.cell_value(it, 8)),  # 这一列的影响不明
                      float(table.cell_value(it, 7)), float(table.cell_value(it, 11)),
                      # 入水温度和罐内温度， 我们怀疑这两个个对VFA有一定影响。罐内温度对聚类结果有很明显的影响作用
                      float(table.cell_value(it, 9)),  # 入水VFA
                      float(table.cell_value(it, 10)),  # 罐内pH，理论上这里列该是稳定的， 考虑删除
                      float(table.cell_value(it, 2)), float(table.cell_value(it, 3)), float(table.cell_value(it, 6)),
                      # 进水量、进水COD和进水pH， 这行代表系统负荷
                      # float(table.cell_value(it, 12)), float(table.cell_value(it, 14)),  # 出水COD和COD去除率， 我们怀疑COD和VFA有很大关系

                      # float(table.cell_value(it, 13)),  # 昨天的出水VFA

                      # float(table.cell_value(it + 1, 1)), float(table.cell_value(it + 1, 4)),
                      # float(table.cell_value(it + 1, 5)),
                      # float(table.cell_value(it + 1, 8)),  # 这一列的影响不明
                      # float(table.cell_value(it + 1, 7)), float(table.cell_value(it + 1, 11)),
                      # 入水温度和罐内温度， 我们怀疑这两个个对VFA有一定影响。罐内温度对聚类结果有很明显的影响作用
                      # float(table.cell_value(it + 1, 9)),  # 入水VFA
                      # float(table.cell_value(it + 1, 10)),  # 罐内pH，理论上这里列该是稳定的， 考虑删除
                      # float(table.cell_value(it + 1, 2)), float(table.cell_value(it + 1, 3)),
                      # float(table.cell_value(it + 1, 6)),
                      # 进水量、进水COD和进水pH， 这行代表系统负荷
                      # float(table.cell_value(it, 12)), float(table.cell_value(it, 14)),  # 出水COD和COD去除率， 我们怀疑COD和VFA有很大关系

                      # float(table.cell_value(it+1, 13)),  # 昨天的出水VFA

                      # float(table.cell_value(it + 2, 1)), float(table.cell_value(it + 2, 4)),
                      # float(table.cell_value(it + 2, 5)),
                      # float(table.cell_value(it + 2, 8)),  # 这一列的影响不明
                      # float(table.cell_value(it + 2, 7)), float(table.cell_value(it + 2, 11)),
                      # 入水温度和罐内温度， 我们怀疑这两个个对VFA有一定影响。罐内温度对聚类结果有很明显的影响作用
                      # float(table.cell_value(it + 2, 9)),  # 入水VFA
                      # float(table.cell_value(it + 2, 10)),  # 罐内pH，理论上这里列该是稳定的， 考虑删除
                      # float(table.cell_value(it + 2, 2)), float(table.cell_value(it + 2, 3)),
                      # float(table.cell_value(it + 2, 6)),
                      # 进水量、进水COD和进水pH， 这行代表系统负荷
                      # float(table.cell_value(it, 12)), float(table.cell_value(it, 14)),  # 出水COD和COD去除率， 我们怀疑COD和VFA有很大关系
                      ])
            # 获取温度范围

            """temperatureList.append(table.cell_value(it, 7))

            if table.cell_value(it, 11) not in temperature: 
                temperature.append(table.cell_value(it, 11))"""

            # VFA.append(table.cell_value(it, 13) * 0.9 + 0.1)  # [0,1]区间映射到[0.1,1]
            VFA.append(table.cell_value(it, 14))  # 这个是COD！
    except ValueError:
        pass

rate = 0.6
size = int(rate * (nRows - 1))

errorRecorder005 = [0] * len(X)
errorRecorder01 = [0] * len(X)

eta0 = 0.05
eta1 = 0.1

GBRMAESum = 0
GBRR2Sum = 0

SVRMAESum = 0
SVRR2Sum = 0

maeRecorder = [0] * len(X)

gap = 0.3

steps = 500

for index in range(steps):
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

    for i in range(len(X)):
        if i in randList:
            trainList.append(X[i])
            trainLabel.append(VFA[i])
        else:
            leftList.append(i)
            testList.append(X[i])
            testLabel.append(VFA[i])

    """y_fix = classifier(testList)

    markerSizeU = [40]
    markerSizeD = [40]
    color = 'red'"""

    [MeanErrorRate, y_predict, counter005, counter01, error] = gbrSearcher(trainList, trainLabel, testList, testLabel)
    for i in range(len(error)):
        if abs(error[i] - testLabel[i]) >= gap:
            maeRecorder[leftList[i]] += (error[i] - testLabel[i]) / abs(error[i] - testLabel[i])
    """y_valueU = []
    y_fixU = []
    y_valueD = []
    y_fixD = []
    for i in range(len(y_fix)):
        if y_fix[i] == 1:
            y_fixU.append(i)
            y_valueU.append(y_predict[i])
        elif y_fix[i] == -1:
            y_fixD.append(i)
            y_valueD.append(y_predict[i])
    markerSizeU = markerSizeU*len(y_fixU)
    markerSizeD = markerSizeD*len(y_fixD)"""
    mean = np.mean(testLabel)
    R2 = 1 - np.sum(np.square(error)) / np.sum(np.square(np.array(testLabel) - mean))
    print('%s,GBR: u 0.05: %s, u 0.1: %s, meanER: %s, R2: %s' % (
    index + 1, str(counter005 / len(leftList))[:5], str(counter01 / len(leftList))[:5], str(MeanErrorRate)[:5],
    str(R2)[:5]))
    # title = 'GBR for VFA out (3 Consecutive Day)\nMAE: ' + (str(MeanErrorRate))[:6] + '; R2: ' + str(R2)[:5]
    # plt.figure(figsize=(18, 12), dpi=300)
    # plt.scatter(y_fixU, y_valueU, s=markerSizeU, color=color, marker='+', label='classifier predict(predict to low)')
    # plt.scatter(y_fixD, y_valueD, s=markerSizeD, color=color, marker='x', label='classifier predict(predict to high)')
    # plt.plot([i for i in range(len(error))], error, color='green', label='gap')
    # plt.plot([i for i in range(len(testLabel))], testLabel, color='navy', label='real value')
    # plt.plot([i for i in range(len(y_predict))], y_predict, color='darkorange', label='predict value')
    # plt.title(title)
    # plt.xlabel('samples')
    # plt.ylabel('VFA Out')
    # plt.savefig("./Result/3 Consecutive Day GBR4VFA "+str(index)+".png", dpi=300)
    # plt.show()

    GBRMAESum += MeanErrorRate
    GBRR2Sum += R2

    [MeanErrorRate, y_predict, counter005, counter01, error] = svrSearcher(trainList, trainLabel, testList, testLabel)
    """y_valueU = []
    y_fixU = []
    y_valueD = []
    y_fixD = []
    for i in range(len(y_fix)):
        if y_fix[i] == 1:
            y_fixU.append(i)
            y_valueU.append(y_predict[i])
        elif y_fix[i] == -1:
            y_fixD.append(i)
            y_valueD.append(y_predict[i])"""
    # mean = np.mean(testLabel)
    # R2 = 1 - np.sum(np.square(error)) / np.sum(np.square(np.array(testLabel) - mean))
    print('  SVR: u 0.05: %s, u 0.1: %s, meanER: %s, R2: %s\n' % (str(counter005 / len(leftList))[:5], str(counter01 / len(leftList))[:5], str(MeanErrorRate)[:5], str(R2)[:5]))
    # title = 'SVR for VFA out (3 Consecutive Day)\nMAE: ' + (str(MeanErrorRate))[:6] + '; R2: ' + str(R2)[:5]
    # plt.figure(figsize=(18, 12), dpi=300)
    # plt.scatter(y_fixU, y_valueU, s=markerSizeU, color=color, marker='+', label='classifier predict(predict to low)')
    # plt.scatter(y_fixD, y_valueD, s=markerSizeD, color=color, marker='x', label='classifier predict(predict to high)')
    # plt.plot([i for i in range(len(error))], error, color='green', label='gap')
    # plt.plot([i for i in range(len(testLabel))], testLabel, color='navy', label='real value')
    # plt.plot([i for i in range(len(y_predict))], y_predict, color='darkorange', label='predict value')
    # plt.title(title)
    # plt.xlabel('samples')
    # plt.ylabel('VFA Out')
    # plt.savefig("./Result/3 Consecutive Day SVR4VFA " + str(index) + ".png", dpi=300)
    # plt.show()

    SVRMAESum += MeanErrorRate
    SVRR2Sum += R2

    """for i in range(len(error)):
        if error[i] >= 0.05:
            errorRecorder005[leftList[i]] += 1
        if error[i] >= 0.1:
            errorRecorder01[leftList[i]] += 1"""

print(maeRecorder)

for i in range(len(maeRecorder)):
    maeRecorder[i] = maeRecorder[i] / steps
    if maeRecorder[i] <= 0.5:
        maeRecorder[i] = 0
    maeRecorder[i] = maeRecorder[i] / abs(maeRecorder[i]) if maeRecorder[i] != 0 else 0

print("GBR: ", end='')
print(GBRMAESum / steps, end=' ')
print(GBRR2Sum / steps)

print("SVR: ", end='')
print(SVRMAESum / steps, end=' ')
print(SVRR2Sum / steps)

"""print(errorRecorder005)
print(errorRecorder01)"""

"""title = 'Mean error rate: ' + (str(MeanErrorRate))[:6] + '\nMax error rate: ' \
        + (str(perErrorRate.max()))[:6] + '\n' + 'Eta0-ErrorRate is: ' \
        + (str(counter0 / len(testLabel))[:6]) + '\n' + 'Eta1-ErrorRate is: ' \
        + (str(counter1 / len(testLabel))[:6])
plt.title(title)
plt.show()"""

"""wb = xlwt.Workbook()
ws = wb.add_sheet('errorCounter')
ws.write(0, 0, '0.1')
ws.write(0, 1, '0.05')
ws.write(0, 2, 'data')
ws.write(0, len(X[0])+2, 'VFA')
for i in range(1, len(errorRecorder005)):
    ws.write(i, 0, errorRecorder01[i-1])
    ws.write(i, 1, errorRecorder005[i-1])
    for j in range(len(X[0])):
        ws.write(i, j+2, X[i-1][j])
    ws.write(i, len(X[0])+2, VFA[i-1])
wb.save('./errorCounterFix.xls')"""

"""wb = xlwt.Workbook()
ws = wb.add_sheet('newEC')
ws.write(0, 0, 'C')
ws.write(0, 1, 'data')
ws.write(0, len(X[0]) + 1, 'VFA')
for i in range(1, len(errorRecorder005)):
    ws.write(i, 0, maeRecorder[i - 1])
    for j in range(len(X[0])):
        ws.write(i, j + 1, X[i - 1][j])
    ws.write(i, len(X[0]) + 1, VFA[i - 1])
wb.save('./newEC.xls')"""
