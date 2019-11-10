import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import xlrd

# 读取数据

excel = xlrd.open_workbook('./#1.xlsx')
table = excel.sheet_by_index(0)

# 行
nRows = table.nrows

X = []
y = []

# 数据从第五行开始
for it in range(4, nRows):
    X.append([float(table.cell_value(it, 1)), float(table.cell_value(it, 4)), float(table.cell_value(it, 5)),
                 float(table.cell_value(it, 6)), float(table.cell_value(it, 7)), float(table.cell_value(it, 8)),
                 float(table.cell_value(it, 9)), float(table.cell_value(it, 13)), float(table.cell_value(it, 14)),
                 float(table.cell_value(it, 15)), float(table.cell_value(it, 16))])
    y.append(float(table.cell_value(it, 10)))

# SVR

rate = 0.5
size = rate * (nRows-4)

# 调用模型
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
# svr_poly = SVR(kernel='poly', C=1e3, degree=2)
svr_rbf.fit(np.mat(X[:size]), np.mat(y))
y_rbf = svr_rbf.predict(np.mat(X[size:]))
svr_lin.fit(np.mat(X[:size]), np.mat(y))
y_lin = svr_lin.predict(np.mat(X[size:]))
# y_poly = svr_poly.fit(X, y).predict(X)

# 可视化结果
lw = 2
plt.scatter(X, y, color='darkorange', label='data')
plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model')
plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')
# plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()
