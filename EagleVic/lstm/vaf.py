import numpy as np
import xlrd
import matplotlib.pyplot as plt
import xlwt
##两个vaf相除观察其分布规律

X = []
Y = []

#读数据
exc1 = xlrd.open_workbook('./#1.xlsx')
table = exc1.sheet_by_index(0)
rows = table.nrows

# 读取两个先后vaf
for it in range(4, rows):
    X.append([float(table.cell_value(it, 9)),float(table.cell_value(it, 12))])

#做除法
x = np.array(X)
(n,m) = np.shape(x)
y = np.zeros((n,1))
for i in range(n-1):
    y[i] = x[i, 1] / x[i, 0]

#画图
sections = np.zeros(40)
for i in range(40):
    sections[i] = 0.005 + i*0.005
plt.hist(y, sections, histtype='bar', rwidth=0.8)
plt.legend()
plt.xlabel('value')
plt.ylabel('number')
plt.title(u'plot1')
plt.show()