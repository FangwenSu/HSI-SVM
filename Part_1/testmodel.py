# -*- coding: utf-8 -*-
"""
 @Time    : 19-9-15 下午10:54
 @Author  : yangzh
 @Email   : 1725457378@qq.com
 @File    : train.py
"""
import warnings
warnings.filterwarnings('ignore')

import h5py
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, Dropout, Flatten, Dense
from keras import backend as K
from keras.optimizers import SGD
from keras.models import load_model

K.set_image_dim_ordering('th')
from keras.utils import np_utils


def test_model(input_shape=None,
               C1=None,
               loss='categorical_crossentropy',
               optimizer=None,
               metrics=['accuracy']):
    model = Sequential( )

    model.add(Conv2D(C1, (3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(3 * C1, (3, 3), activation='relu'))
    model.add(Dropout(0.25))

    model.add(Flatten( ))
    model.add(Dense(6 * numPCAcomponents, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()

    return model

if __name__ == '__main__':
    windowSize = 5
    numPCAcomponents = 30
    testRatio = 0.20

    X_train = np.load("../data/XtrainWindowSize"
                      + str(windowSize) + "PCA" + str(numPCAcomponents) + "testRatio" + str(testRatio) + ".npy")
    y_train = np.load("../data/ytrainWindowSize"
                      + str(windowSize) + "PCA" + str(numPCAcomponents) + "testRatio" + str(testRatio) + ".npy")

    # 转化为(num,channels,height,width)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[3], X_train.shape[1], X_train.shape[2]))

    # 转化标签
    y_train = np_utils.to_categorical(y_train)

    # 定义输入的形状
    input_shape = X_train[0].shape
    print(input_shape)

    # filters
    C1 = 3 * numPCAcomponents

    # 定义优化器
    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)

    # 获取测试模型
    model = test_model(input_shape=input_shape, C1=C1, optimizer=sgd)

    # 训练模型
    model.fit(X_train, y_train, batch_size=32, epochs=50)

    # 保存模型
    model.save('./model/my_model.h5')
