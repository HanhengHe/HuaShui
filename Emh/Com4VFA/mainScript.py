import numpy as np
import xlrd
from sklearn.ensemble import GradientBoostingRegressor as GBR

#  gbr 暴力搜参函数


# 读取数据

excel = xlrd.open_workbook('./Normalized++.xlsx')
table = excel.sheet_by_index(0)

# 行
nRows = table.nrows

X = []
VFA = []
temperature = []
# 跳过列名
for it in range(1, nRows):
    X.append([float(table.cell_value(it, 1)), float(table.cell_value(it, 4)), float(table.cell_value(it, 5)),
              float(table.cell_value(it, 6)), float(table.cell_value(it, 7)), float(table.cell_value(it, 8)),
              float(table.cell_value(it, 9)), float(table.cell_value(it, 10)), float(table.cell_value(it, 11)),
              float(table.cell_value(it, 14)),
              ])
    # 获取温度范围
    if table.cell_value(it, 11) not in temperature:
        temperature.append(table.cell_value(it, 11))
    VFA.append(table.cell_value(it, 13))
