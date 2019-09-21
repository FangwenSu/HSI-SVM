# -*- coding: utf-8 -*-
"""
 @Time    : 19-9-15 上午3:29
 @Author  : yangzh
 @Email   : 1725457378@qq.com
 @File    : HSI.py
"""
from __future__ import print_function

import keras
from keras import layers, models
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.utils import plot_model
from keras.models import Model
import numpy as np
from sklearn.svm import SVC

keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
np.random.seed(1337)


# HSI 空间特征提取模型
def HSI_K(include_top=True,
          input_tensor=None,
          input_shape=(),
          classes=10,
          **kwargs):
    H, W, D = input_shape[0], input_shape[1], input_shape[2]

    # 大小为k×k×d的样本输入
    img_input = layers.Input(shape=input_shape)
    # 第一层卷积核的大小为5x5xd,数量100个
    x = layers.Conv2D(filters=100, kernel_size=(5, 5), name='conv21', activation='tanh', padding="valid",
                      bias_initializer='zeros')(img_input)
    # 第二层最大池化层,池化核的大小为2x2
    x = layers.MaxPooling2D(pool_size=(2, 2), name='pool22')(x)
    # 第三层卷积核的大小为3x3xd,数量300个
    x = layers.Conv2D(filters=300, kernel_size=(3, 3), name='conv23', activation='tanh', padding="valid",
                      bias_initializer='zeros')(x)
    # 第四层最大池化层,池化核的大小为2x2
    x = layers.MaxPooling2D(pool_size=(2, 2), name='pool24')(x)
    # 将第四个池化层输出的特征图转换成一维向量
    x = layers.Flatten( )(x)
    # 第五层、第六层和第七层三个全连接运算
    x = layers.Dense(units=200, name='fc25', activation='tanh', bias_initializer='zeros')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(units=100, name='fc26', activation='tanh', bias_initializer='zeros')(x)
    x = layers.Dropout(0.4)(x)
    # x = layers.Dense(units=classes, activation='softmax', name='HSI_K-predictions', bias_initializer='zeros')(x)

    model = models.Model(img_input, x, name='HSI_K')

    # plot_model(model, to_file='HSI_K.png', show_shapes=True, show_layer_names=True)

    return model


# HSI 光谱特征提取模型
def HSI_G(include_top=True,
          input_tensor=None,
          input_shape=(),
          classes=10,
          **kwargs):
    H, W, D = input_shape[0], input_shape[1], input_shape[2]

    img_input = layers.Input(shape=input_shape)
    x = layers.Reshape(target_shape=(input_shape[0] * input_shape[1], input_shape[2]))(img_input)
    x = layers.Conv1D(filters=90, kernel_size=7, activation='relu', name='conv11', bias_initializer='zeros')(x)
    x = layers.MaxPooling1D(pool_size=2, name='pool11')(x)
    x = layers.Conv1D(filters=270, kernel_size=5, activation='relu', name='conv12', bias_initializer='zeros')(x)
    x = layers.MaxPooling1D(pool_size=2, name='pool12')(x)
    x = layers.Conv1D(filters=810, kernel_size=3, activation='relu', name='conv13', bias_initializer='zeros')(x)
    x = layers.MaxPooling1D(pool_size=2, name='pool13')(x)
    x = layers.Flatten( )(x)
    x = layers.Dense(units=200, name='fc11', activation='relu', bias_initializer='zeros')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(units=100, name='fc12', activation='relu', bias_initializer='zeros')(x)
    x = layers.Dropout(0.4)(x)
    # x = layers.Dense(units=classes, activation='softmax', name='HSI_G-predictions', bias_initializer='zeros')(x)

    model = models.Model(img_input, x, name='HSI_G')

    # plot_model(model, to_file='HSI_G.png', show_shapes=True, show_layer_names=True)

    return model


def HSI(objective, optimizer, metrics, input_shape, filters, K_num):
    img_input = layers.Input(shape=input_shape)
    model_K = HSI_K(include_top=True, input_shape=input_shape, classes=K_num)(img_input)
    model_G = HSI_G(include_top=True, input_shape=input_shape, classes=K_num)(img_input)

    x = layers.concatenate([model_K, model_G], axis=1)

    x = layers.Dense(units=100, name='split', activation='relu', bias_initializer='zeros')(x)

    predictions = layers.Dense(units=K_num, activation='softmax', name='HSI-predictions',use_bias=False)(x)

    model = models.Model(inputs=img_input, outputs=predictions)
    model.compile(loss=objective, optimizer=optimizer, metrics=metrics)
    model.summary( )

    # plot_model(model, to_file='HSI_A.png', show_shapes=True, show_layer_names=True)

    return model



def train_model():
    learning_rate = 0.001
    epoch = 50
    challes = 1
    windowSize = 19
    numPCAcomponents = 20
    testRatio = 0.20
    K_num = 16
    filepath = '../model/HSI_PCA_svm_model.h5'
    X_train = np.load("../data/XtrainWindowSize"
                      + str(windowSize) + "PCA" + str(numPCAcomponents) + "testRatio" + str(testRatio) + ".npy")
    y_train_ = np.load("../data/ytrainWindowSize"
                      + str(windowSize) + "PCA" + str(numPCAcomponents) + "testRatio" + str(testRatio) + ".npy")

    X_test = np.load("../data/XtestWindowSize"
                     + str(windowSize) + "PCA" + str(numPCAcomponents) + "testRatio" + str(testRatio) + ".npy")
    y_test_ = np.load("../data/ytestWindowSize"
                     + str(windowSize) + "PCA" + str(numPCAcomponents) + "testRatio" + str(testRatio) + ".npy")

    # 转化标签
    y_train = np_utils.to_categorical(y_train_)
    y_test = np_utils.to_categorical(y_test_)

    print(X_train.shape)
    print(y_train.shape)

    # 定义输入的形状
    input_shape = X_train[0].shape
    print(input_shape)

    # filters
    C1 = 3 * numPCAcomponents

    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False,
                                 mode='auto', period=1)
    tensorboard = TensorBoard(log_dir='../logs/HSI_PCA_SVM')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0000000001)

    # model = model.fit(X_train, y_train, batch_size=32, validation_data=(X_test, y_test),
    #                   callbacks=[checkpoint, tensorboard, reduce_lr], epochs=50, verbose=1)


    optimizer = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    objective = 'categorical_crossentropy'
    metrics = ['accuracy']
    model = HSI(objective=objective, optimizer=optimizer, metrics=metrics, input_shape=input_shape, filters=C1,
                K_num=K_num)

    model.load_weights('../model/HSI_PCA_svm_model.h5')

    intermediate_model = Model(inputs=model.input, outputs=[model.get_layer('split').output])
    intermediate_output = intermediate_model.predict(X_train)
    print(intermediate_output)

    svm = SVC(kernel='rbf',gamma='auto')
    svm.fit(intermediate_output, y_train_)

    test_output = intermediate_model.predict(X_test)
    pre = svm.predict(test_output)



    print(pre)
    print(y_test_)
    print(pre.shape)
    print(y_test_.shape)

    acc = 0
    for i in range(len(pre)):
        if (pre[i] == y_test_[i]):
            acc += 1
    acc_rate = acc / len(pre)
    print(acc_rate)


if __name__ == '__main__':
    train_model( )
