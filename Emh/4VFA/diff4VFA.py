#  尝试计算VFA的差分情况

import numpy as np
import matplotlib.pyplot as plt
import xlrd

# 读取数据

excel = xlrd.open_workbook('./#1decentralization++.xlsx')
table = excel.sheet_by_index(0)

# 行
nRows = table.nrows

VFA = []

# 数据从第五行开始
for it in range(nRows):
    VFA.append(table.cell_value(it, 13))

diff = np.array(VFA[1:]) - np.array(VFA[:-1])
diff = diff[1:] - diff[:-1]
diff = diff[1:] - diff[:-1]

plt.plot([i for i in range(len(diff))], diff)
plt.title('diff_3')
plt.show()
