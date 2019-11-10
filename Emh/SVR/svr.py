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
size = int(rate * (nRows - 4))

# 调用模型
svr_rbf = SVR(C=1.0, cache_size=200, coef0=0.0, epsilon=0.2, gamma=0.1,
              kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)

svr_rbf.fit(np.mat(X[:size]), y[:size])
y_rbf = svr_rbf.predict(np.mat(X[size:]))

# 可视化结果
eta = 0.05

lw = 2
plt.scatter([i for i in range(len(y[size:]))], y[size:], color='darkorange', label='Real Data')
plt.plot([i for i in range(len(y[size:]))], y_rbf, color='navy', lw=lw, label='RBF predict')
plt.xlabel('number')
plt.ylabel('COD_Out')
plt.title('SVR for COD OUT')
plt.legend()
plt.show()

error = np.abs(np.array(y[size:])-np.array(y_rbf))
plt.plot([i for i in range(len(y[size:]))], [eta]*len(y[size:]), color='darkorange', label='errorAllow')
plt.plot([i for i in range(len(y[size:]))], error, color='navy', lw=lw, label='error')
perErrorRate = error/np.array(y[size:])
errorRate = np.sum(perErrorRate)/len(y[size:])

counter = 0
for e in perErrorRate:
    if e >= eta:
        counter += 1

etaRate = eta/len(y[size:])
title = 'Mean error rate: ' + str(errorRate) + '\nMax error rate: ' \
        + str(perErrorRate.max()) + '\n' + 'Eta-ErrorRate is: ' + str(counter/len(y[size:]))
plt.title(title)
plt.show()