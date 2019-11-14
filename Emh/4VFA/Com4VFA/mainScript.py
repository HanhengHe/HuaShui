import numpy as np
import xlrd
import matplotlib.pyplot as plt

#  gbr 暴力搜参函数


# 读取数据

excel = xlrd.open_workbook('./Normalized.xlsx')
table = excel.sheet_by_index(0)

# 行
nRows = table.nrows

X = []
VFA = []
temperature = []
temperatureList = []

# 跳过列名
for it in range(1, nRows):
    try:
        X.append([float(table.cell_value(it, 1)), float(table.cell_value(it, 4)), float(table.cell_value(it, 5)), float(table.cell_value(it, 8)),  # 这一列的影响不明
                  float(table.cell_value(it, 7)), float(table.cell_value(it, 11)),  # 入水温度和罐内温度， 我们怀疑这两个个对VFA有一定影响。罐内温度对聚类结果有很明显的影响作用
                  float(table.cell_value(it, 9)),  # 入水VFA
                  float(table.cell_value(it, 10)),  # 罐内pH，理论上这里列该是稳定的， 考虑删除
                  float(table.cell_value(it, 2)), float(table.cell_value(it, 3)), float(table.cell_value(it, 6)),  # 进水量、进水COD和进水pH， 这行代表系统负荷
                  float(table.cell_value(it, 12)), float(table.cell_value(it, 14)),  # 入水COD和COD去除率， 我们怀疑COD和VFA有很大关系
                  ])
        # 获取温度范围

        temperatureList.append(table.cell_value(it, 7))

        if table.cell_value(it, 11) not in temperature:
            temperature.append(table.cell_value(it, 11))

        VFA.append(table.cell_value(it, 13))
    except ValueError:
        pass


