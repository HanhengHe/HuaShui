import numpy as np
import xlrd
from sklearn.ensemble import GradientBoostingRegressor as GBR
#  gbr 暴力搜参函数

def LenthSearch()

if __name__=='__main__':
    # 读取数据

    excel = xlrd.open_workbook('./#1decentralization++.xlsx')
    table = excel.sheet_by_index(0)

    # 行
    nRows = table.nrows

    X = []
    VFA = []
    # 数据从第五行开始
    for it in range(1, nRows):
        if table.cell_value(it + 1, 15) - table.cell_value(it, 15) != 1:
            continue
        X.append([float(table.cell_value(it, 0)), float(table.cell_value(it, 1)), float(table.cell_value(it, 2)),
                  float(table.cell_value(it, 3)), float(table.cell_value(it, 4)), float(table.cell_value(it, 5)),
                  float(table.cell_value(it, 6)), float(table.cell_value(it, 7)), float(table.cell_value(it, 8)),
                  float(table.cell_value(it, 13)),
                  ])
        VFA.append(table.cell_value(it, 13))

    VFA.append(table.cell_value(nRows - 1, 13))