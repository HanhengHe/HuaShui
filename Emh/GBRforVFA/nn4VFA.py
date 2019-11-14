import keras as K


def nnSearcher(trainList, trainLabel, testList, testLabel):
    # 调用模型

    # 2. 定义模型
    init = K.initializers.glorot_uniform(seed=1)
    simple_adam = K.optimizers.Adam()
    model = K.models.Sequential()
    model.add(K.layers.Dense(units=5, input_dim=len(trainList[0]), kernel_initializer=init, activation='relu'))
    model.add(K.layers.Dense(units=6, kernel_initializer=init, activation='relu'))
    model.add(K.layers.Dense(units=1, kernel_initializer=init, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=simple_adam, metrics=['accuracy'])

    b_size = 1
    max_epochs = 100
    print("Starting training ")
    h = model.fit(trainList, trainLabel, batch_size=b_size, epochs=max_epochs, shuffle=True, verbose=1)
    print("Training finished \n")

    eval = model.evaluate(testList, testLabel, verbose=0)
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

    return [1, 1, -1, -1, -1]
