import numpy as np
from sklearn.svm import SVR


def svrSearcher(trainList, trainLabel, testList, testLabel):
    # 调用模型
    RM = SVR(C=0.1, epsilon=0.0002, gamma=2, kernel='rbf', max_iter=1500, shrinking=True, tol=0.005, )

    RM.fit(trainList, trainLabel)
    y_predict = RM.predict(np.mat(testList))

    # 计算误差
    error = np.abs(np.array(y_predict) - np.array(testLabel))
    perErrorRate = np.array(error) / np.array(testLabel)
    MeanErrorRate = np.sum(perErrorRate) / len(perErrorRate)
    eta0 = 0.05
    eta1 = 0.1
    counter005 = 0
    counter01 = 0
    for e in error:
        if e >= eta0:
            counter005 += 1
        if e >= eta1:
            counter01 += 1

    return [MeanErrorRate, y_predict, counter005, counter01, error]
