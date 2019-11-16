import numpy as np
import xlrd
import matplotlib.pyplot as plt


def Var(X):
    return np.mean(np.square(np.array(X) - np.mean(X)))


def Threshold(X):
    if len(X) < 4:
        return [np.mean(X)]*len(X)
    off = np.percentile(X, (25, 50, 80), interpolation='midpoint')
    return [off[2] * (1 - ScalingFactor)] * len(X)


# parameter here
tipRate = 0.55
ScalingFactor = 0.1

# 读取数据

excel = xlrd.open_workbook('./Normalized.xlsx')
table = excel.sheet_by_index(0)

# 行
nRows = table.nrows

# rTemperature = []
temperature = []
VFA = []

# 跳过列名
for it in range(1, nRows):
    try:
        # rTemperature.append(table.cell_value(it, 11) * 30 + 12)  # 逆归一化
        temperature.append(table.cell_value(it, 11))
        VFA.append(table.cell_value(it, 13) * 0.9 + 0.1)  # [0,1]区间映射到[0.1,1]
    except ValueError:
        pass

# 判断尖点
if VFA[0] > VFA[1]:
    tipPointsIndex = [0]
else:
    tipPointsIndex = [1]
for i in range(tipPointsIndex[0]+1, len(temperature)):
    if (VFA[i] - VFA[i - 1])/VFA[i] >= tipRate:
        tipPointsIndex.append(i)

temperatureSet = []
start = 0

while True:
    try:
        if start >= len(tipPointsIndex)-1:
            temperatureSet.append(start)
            break
        for k in range(start+1, len(tipPointsIndex)-1):
            if Var(VFA[start:tipPointsIndex[k]]) <= min(Var(VFA[start:tipPointsIndex[k + 1]]),
                                                        Var(VFA[start:tipPointsIndex[k + 2]])):
                temperatureSet.append((start, k))
                start = k
                break
    except IndexError:
        try:
            if Var(VFA[start:tipPointsIndex[k]]) <= Var(VFA[start:tipPointsIndex[k + 1]]):
                temperatureSet.append((start, k))
                start = k
                continue
            else:
                temperatureSet.append((start, k + 1))
                start = k + 1
                continue
        except IndexError:
            if start != len(tipPointsIndex) - 1:
                temperatureSet.append((start, start + 1))
            else:
                temperatureSet.append(start)
        break

print(tipPointsIndex)
print(temperatureSet)

title = 'Threshold of VFA out'
plt.figure(figsize=(18, 12), dpi=300)
# plt.scatter([i for i in range(len(VFA))], VFA, color='green', label='VFA OUT')
plt.plot([i for i in range(len(VFA))], VFA, color='green', label='VFA OUT')
plt.plot([i for i in range(len(temperature))], temperature, color='navy', label='real temperature')
for i in range(len(temperatureSet)):
    try:
        index1 = tipPointsIndex[temperatureSet[i][0]]
        index2 = tipPointsIndex[temperatureSet[i][1]] - 1
    except TypeError:
        plt.plot([tipPointsIndex[temperatureSet[i]]], VFA[tipPointsIndex[temperatureSet[i]]], color='darkorange', label='VFA OUT')
        continue
    vfaB = VFA[index1:index2]
    temp = Threshold(vfaB)
    print(temp)
    plt.plot([i for i in range(index1, index2)], temp, color='darkorange', label='VFA OUT')

plt.title(title)
plt.xlabel('samples')
plt.ylabel('VFA Out')
plt.show()
