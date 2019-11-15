import keras as K
import numpy as np
from keras import optimizers
from keras.layers import Dropout


def nnSearcher(trainList, trainLabel, testList, testLabel):
    # 调用模型

    # 2. 定义模型
    model = K.models.Sequential()
    model.add(K.layers.Dense(units=512, input_dim=len(trainList[0]), activation='relu'))
    model.add(K.layers.Dense(units=1, activation='softmax'))
    sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_absolute_error', optimizer=sgd, metrics=['mae', 'acc'])

    b_size = 10
    max_epochs = 500
    print("Starting training ")
    _ = model.fit(np.mat(trainList), np.array(trainLabel), batch_size=b_size, epochs=max_epochs, shuffle=True, verbose=2)
    print("Training finished \n")

    eval = model.evaluate(np.mat(testList), np.array(testLabel), verbose=0)
    print("Evaluation on test data: loss = %0.6f accuracy = %0.2f%% \n" % (eval[0], eval[1] * 100))

    # 计算误差
    """error = np.abs(np.array(y_rbf) - np.array(testLabel))
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
            counter01 += 1"""

    return [1, 1, -1, -1, []]
