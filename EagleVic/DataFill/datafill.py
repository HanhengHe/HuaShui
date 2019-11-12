from random import randint
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import xlrd


X1 = []


#读取二号数据
exc2 = xlrd.open_workbook('./#2.xlsx')
table = exc2.sheet_by_index(0)
rows = table.nrows


#第九列求和求平均值
for i in range(4,rows):
    if table.cell_value(i,8)!='':
        X1.append(table.cell_value(i,8))
x1 = sum(X1)
n1 = len(X1)
average_X1 = x1/n1


#中心化,减去平均值
X1.sort()
print(average_X1)
print(n1)
for i in range(n1):
    X1[i] = X1[i]-average_X1
print(X1)


#划分区间分组
del X1[n1-1]
size = X1[n1-2] - X1[0]
print(size)
sections = np.zeros(64)
for i in range(64):
    sections[i] = -50 + i*50


#画图
plt.hist(X1, sections, histtype='bar', rwidth=0.8)
plt.legend()
plt.xlabel('value')
plt.ylabel('number')
plt.title(u'plot1')
plt.show()