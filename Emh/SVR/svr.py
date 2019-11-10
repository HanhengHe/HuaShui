import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import xlrd

# 读取数据

excel = xlrd.open_workbook('./#1decentralization.xlsx')
table = excel.sheet_by_index(0)

# 行
nRows = table.nrows

X = []
y = []

# 数据从第五行开始
for it in range(4, nRows):
    X.append([float(table.cell_value(it, 0)), float(table.cell_value(it, 1)), float(table.cell_value(it, 2)),
              float(table.cell_value(it, 3)), float(table.cell_value(it, 4)), float(table.cell_value(it, 5)),
              float(table.cell_value(it, 6)), float(table.cell_value(it, 7)), float(table.cell_value(it, 8)),
              float(table.cell_value(it, 9)), float(table.cell_value(it, 10))])
    y.append(float(table.cell_value(it, 12)))

# SVR

rate = 0.4
size = int(rate * (nRows - 4))

# 调用模型
svr_rbf = SVR(C=0.03, epsilon=0.0019, gamma=2.0, kernel='rbf', max_iter=500, shrinking=True, tol=0.005)

svr_rbf.fit(np.mat(X[:size]), y[:size])
y_rbf = svr_rbf.predict(np.mat(X[size:]))

# 可视化结果
eta = 0.05

lw = 2
plt.scatter([i for i in range(len(y[size:]))], y[size:], color='darkorange', label='Real Data')
plt.scatter([i for i in range(len(y[size:]))], y_rbf, color='navy', lw=lw, label='RBF predict')
plt.xlabel('number')
plt.ylabel('COD_Out')
plt.title('SVR for COD OUT')
plt.legend()
plt.show()

error = np.abs(np.array(y[size:])-np.array(y_rbf))
plt.plot([i for i in range(len(y[size:]))], [eta]*len(y[size:]), color='navy', label='errorAllow')
plt.scatter([i for i in range(len(y[size:]))], error, color='darkorange', lw=lw, label='error')
perErrorRate = error / np.array(y[size:])
MeanErrorRate = np.sum(perErrorRate) / len(y[size:])

counter = 0
for e in error:
    if e >= eta:
        counter += 1

title = 'Mean error rate: ' + (str(MeanErrorRate))[:6] + '\nMax error rate: ' \
        + (str(perErrorRate.max()))[:6] + '\n' + 'Eta-ErrorRate is: ' \
        + (str(counter/len(y[size:]))[:6])
plt.title(title)
plt.show()