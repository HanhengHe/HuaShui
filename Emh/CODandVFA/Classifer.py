import numpy as np
import xlrd
from random import randint
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.svm import SVC


def classifier(target):
    # 读取数据

    wb = xlrd.open_workbook('./newEC.xls')
    ws = wb.sheet_by_index(0)

    rX = []
    ry = []

    X = []
    X0 = []
    y = []
    y0 = []

    gap = 20  # changeable

    c = 0
    c0 = 0

    # 跳过表头 data: 2:14, 上四分位数：0.1735 下四分位数：0.2378
    for it in range(1, ws.nrows):
        try:
            rX.append([float(ws.cell_value(it, 1)), float(ws.cell_value(it, 2)), float(ws.cell_value(it, 3)),
                       float(ws.cell_value(it, 4)), float(ws.cell_value(it, 5)), float(ws.cell_value(it, 6)),
                       float(ws.cell_value(it, 7)), float(ws.cell_value(it, 8)), float(ws.cell_value(it, 9)),
                       float(ws.cell_value(it, 10)), float(ws.cell_value(it, 11)), float(ws.cell_value(it, 12)),
                       float(ws.cell_value(it, 13)),
                       ])
            if ws.cell_value(it, 1) == 0:
                X0.append([float(ws.cell_value(it, 1)), float(ws.cell_value(it, 2)), float(ws.cell_value(it, 3)),
                       float(ws.cell_value(it, 4)), float(ws.cell_value(it, 5)), float(ws.cell_value(it, 6)),
                       float(ws.cell_value(it, 7)), float(ws.cell_value(it, 8)), float(ws.cell_value(it, 9)),
                       float(ws.cell_value(it, 10)), float(ws.cell_value(it, 11)), float(ws.cell_value(it, 12)),
                       float(ws.cell_value(it, 13)),
                       ])
                y0.append(0)
                c0 += 1
            else:
                X.append([float(ws.cell_value(it, 1)), float(ws.cell_value(it, 2)), float(ws.cell_value(it, 3)),
                           float(ws.cell_value(it, 4)), float(ws.cell_value(it, 5)), float(ws.cell_value(it, 6)),
                           float(ws.cell_value(it, 7)), float(ws.cell_value(it, 8)), float(ws.cell_value(it, 9)),
                           float(ws.cell_value(it, 10)), float(ws.cell_value(it, 11)), float(ws.cell_value(it, 12)),
                           float(ws.cell_value(it, 13)),
                           ])
                y.append(0)
                c += 1
            """if float(ws.cell_value(it, 0)) == 1:
                if float(ws.cell_value(it, 0)) >= 0.2378:
                    X.append([float(ws.cell_value(it, 2)), float(ws.cell_value(it, 3)), float(ws.cell_value(it, 4)),
                              float(ws.cell_value(it, 5)), float(ws.cell_value(it, 6)), float(ws.cell_value(it, 7)),
                              float(ws.cell_value(it, 8)), float(ws.cell_value(it, 9)),
                              float(ws.cell_value(it, 10)),
                              float(ws.cell_value(it, 11)), float(ws.cell_value(it, 12)),
                              float(ws.cell_value(it, 13)),
                              float(ws.cell_value(it, 14)),
                              ])
                    c1 += 1
                    y.append(1)  # 大VFA
                    ry.append(1)
                elif float(ws.cell_value(it, 15)) <= 0.1735:
                    X.append([float(ws.cell_value(it, 2)), float(ws.cell_value(it, 3)), float(ws.cell_value(it, 4)),
                              float(ws.cell_value(it, 5)), float(ws.cell_value(it, 6)), float(ws.cell_value(it, 7)),
                              float(ws.cell_value(it, 8)), float(ws.cell_value(it, 9)),
                              float(ws.cell_value(it, 10)),
                              float(ws.cell_value(it, 11)), float(ws.cell_value(it, 12)),
                              float(ws.cell_value(it, 13)),
                              float(ws.cell_value(it, 14)),
                              ])
                    c_1 += 1
                    y.append(-1)  # 小VFA
                    ry.append(-1)
                else:
                    X0.append(
                        [float(ws.cell_value(it, 2)), float(ws.cell_value(it, 3)), float(ws.cell_value(it, 4)),
                         float(ws.cell_value(it, 5)), float(ws.cell_value(it, 6)), float(ws.cell_value(it, 7)),
                         float(ws.cell_value(it, 8)), float(ws.cell_value(it, 9)), float(ws.cell_value(it, 10)),
                         float(ws.cell_value(it, 11)), float(ws.cell_value(it, 12)), float(ws.cell_value(it, 13)),
                         float(ws.cell_value(it, 14)),
                         ])
                    c0 += 1
                    y0.append(0)
                    ry.append(0)
            else:
                X0.append([float(ws.cell_value(it, 2)), float(ws.cell_value(it, 3)), float(ws.cell_value(it, 4)),
                           float(ws.cell_value(it, 5)), float(ws.cell_value(it, 6)), float(ws.cell_value(it, 7)),
                           float(ws.cell_value(it, 8)), float(ws.cell_value(it, 9)), float(ws.cell_value(it, 10)),
                           float(ws.cell_value(it, 11)), float(ws.cell_value(it, 12)), float(ws.cell_value(it, 13)),
                           float(ws.cell_value(it, 14)),
                           ])
                c0 += 1
                y0.append(0)  # 中间值
                ry.append(0)"""
        except ValueError:
            pass
    print('%s, %s' % (c, c0))
    print(c / (c + c0))

    #  随机生成(c0+c1)/2个0类插入y
    r0 = []
    while True:
        rand = randint(0, len(y0) - 1)
        if len(r0) > c:
            break
        if rand in r0:
            continue
        else:
            r0.append(rand)

    for i in r0:
        X.append(X0[i])
        y.append(y0[i])

    rate = 0.6
    size = int(rate * len(X))

    classifiers = []

    for index in range(1):
        # 生成随机数列
        randList = []
        while True:
            if len(randList) == size:
                break
            rand = randint(0, len(X) - 1)
            if rand in randList:
                continue
            else:
                randList.append(rand)

        trainList = []
        trainLabel = []
        testList = []
        testLabel = []

        y0C = 0

        for i in range(len(X)):
            if y[i] == 0:
                if y0C > c:
                    continue
            if i in randList:
                trainList.append(X[i])
                trainLabel.append(y[i])
            else:
                testList.append(X[i])
                testLabel.append(y[i])
        classifier = SVC(C=4, kernel='rbf', gamma=4, tol=1e-3, )
        classifier.fit(np.mat(trainList), trainLabel)
        classifiers.append(classifier)
        """counter = 0
        mat1 = np.mat(np.zeros((4, 4)))
        mat1[0, 1] = -1
        mat1[0, 2] = 0
        mat1[0, 3] = 1
        mat1[1, 0] = -1
        mat1[2, 0] = 0
        mat1[3, 0] = 1
        for i in range(len(testList)):
            if testLabel[i] != pre_test[i]:
                mat1[pre_test[i] + 2, testLabel[i] + 2] += 1
                counter += 1
            else:
                mat1[pre_test[i] + 2, ry[i] + 2] += 1
        MeanErrorRate = counter / len(testList)
        print('meanER: %s' % (str(MeanErrorRate)[:5]))
        print(mat1)"""

        pre_test = classifier.predict(np.mat(target)).tolist()
        """pre_test = classifier.predict(np.mat(rX)).tolist()
        counter = 0
        mat1 = np.mat(np.zeros((4, 4)))
        mat1[0, 1] = -1
        mat1[0, 2] = 0
        mat1[0, 3] = 1
        mat1[1, 0] = -1
        mat1[2, 0] = 0
        mat1[3, 0] = 1
        for i in range(len(ry)):
            if ry[i] != pre_test[i]:
                mat1[pre_test[i] + 2, ry[i] + 2] += 1
                counter += 1
            else:
                mat1[pre_test[i] + 2, ry[i] + 2] += 1
        MeanErrorRate = counter / len(ry)
        print('meanER hold set: %s' % (str(MeanErrorRate)[:5]))
        print(mat1)
        mat1[1:, 1] = mat1[1:, 1] / c_1
        mat1[1:, 2] = mat1[1:, 2] / c0
        mat1[1:, 3] = mat1[1:, 3] / c1
        print(mat1)"""

        """for i in range(len(pre_test)):
            if pre_test[i] == -1:
                pre_test[i] = 0"""

        return pre_test


