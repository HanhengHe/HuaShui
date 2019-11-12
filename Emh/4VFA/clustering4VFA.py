from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import xlrd

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
    X.append([float(table.cell_value(it, 0)), float(table.cell_value(it, 1)), float(table.cell_value(it, 2)),
              float(table.cell_value(it, 3)), float(table.cell_value(it, 4)), float(table.cell_value(it, 5)),
              float(table.cell_value(it, 6)), float(table.cell_value(it, 7)), float(table.cell_value(it, 8)),])
              # float(table.cell_value(it, 9)), float(table.cell_value(it, 10))])
    testVFA.append([float(table.cell_value(it, 13)), float(table.cell_value(it, 13))])
    VFA.append(float(table.cell_value(it, 13)))

y_Predict = KMeans(n_clusters=2).fit_predict(X)

"""for i in range(len(VFA)):
    if i == len(VFA)-1:
        break
    if y_Predict[i] != y_Predict[i+1]:
        print(i+5)"""

"""test = []

for i in range(len(VFA)):
    if y_Predict[i] == 1:
        test.append(VFA[i])

print(min(test))"""

plt.plot([i for i in range(len(VFA))], VFA)
plt.scatter([i for i in range(len(VFA))], VFA, c=y_Predict)
plt.plot([i for i in range(len(VFA))], [0.37037037037037035]*len(VFA))
plt.title('cluster without vfa')
plt.show()

"""
a = []
b = []
c = []
for i in range(len(VFA)):
    if y_Predict[i] == 0:
        a.append(VFA[i])
    if y_Predict[i] == 1:
        b.append(VFA[i])
    if y_Predict[i] == 2:
        c.append(VFA[i])

plt.scatter([i for i in range(len(a))], a)
plt.title('type 1')
plt.show()

plt.scatter([i for i in range(len(b))], b)
plt.title('type 2')
plt.show()

plt.scatter([i for i in range(len(c))], c)
plt.title('type 3')
plt.show()
"""