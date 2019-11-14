import numpy as np
import xlrd
import xlwt

X=[]

exc1 = xlrd.open_workbook('./#1.xlsx')
table = exc1.sheet_by_index(0)
rows = table.nrows

# 数据从第五行开始
for it in range(4, rows):
    X.append([float(table.cell_value(it, 1)), float(table.cell_value(it, 2)),float(table.cell_value(it, 3)),
              float(table.cell_value(it, 4)), float(table.cell_value(it, 5)),float(table.cell_value(it, 6)),
              float(table.cell_value(it, 7)), float(table.cell_value(it, 8)),float(table.cell_value(it, 9)),
              float(table.cell_value(it, 10)), float(table.cell_value(it, 11)), float(table.cell_value(it, 12)),
               float(table.cell_value(it, 13)), float(table.cell_value(it, 14)),float(table.cell_value(it, 15)),
              float(table.cell_value(it, 16))])

a = np.array(X)
a = a.T
corr = np.corrcoef(a)

print(corr)


#读数据
myWorkbook = xlwt.Workbook()
mySheet = myWorkbook.add_sheet('Test_Sheet')
# 数据格式
myStyle = xlwt.easyxf('font: name Times New Roman, color-index red, bold on', num_format_str='#,##0.00')
(n, m) = np.shape(corr)
print(n)
print(m)
for i in range(n):
    for j in range(m):
        mySheet.write(i, j, corr[i, j])
myWorkbook.save('excelFileeee.xls')
