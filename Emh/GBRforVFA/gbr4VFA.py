import numpy as np
from sklearn.ensemble import GradientBoostingRegressor as GBR


def gbrSearcher(trainList, trainLabel, testList, testLabel):
    # 调用模型
    RM = GBR(n_estimators=3000, max_depth=2, min_samples_split=2, learning_rate=0.1, loss='lad')

    RM.fit(trainList, trainLabel)
    y_rbf = RM.predict(np.mat(testList))

    # 计算误差
    error = np.abs(np.array(y_rbf) - np.array(testLabel))
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

    return [MeanErrorRate, perErrorRate, counter005, counter01, error]
