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
from sklearn.decomposition import PCA
import os
import scipy.io as sio
from keras.utils import np_utils
from sklearn.metrics import classification_report, confusion_matrix
import spectral
from matplotlib import pyplot as plt

import numpy as np

np.random.seed(1337)
keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
PATCH_SIZE = 5

def loadIndianPinesData():
    data_path = os.path.join(os.getcwd( ), '../Indian Pines')
    data = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
    labels = sio.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))['indian_pines_gt']

    return data, labels

def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
    return newX, pca


def Patch(data, height_index, width_index):
    # transpose_array = data.transpose((2,0,1))
    # print transpose_array.shape
    height_slice = slice(height_index, height_index + PATCH_SIZE)
    width_slice = slice(width_index, width_index + PATCH_SIZE)
    patch = data[height_slice, width_slice, :]


def reports(model, X_test, y_test):
    Y_pred = model.predict(X_test)
    y_pred = np.argmax(Y_pred, axis=1)
    target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
        , 'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                    'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                    'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                    'Stone-Steel-Towers']

    classification = classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target_names)
    confusion = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
    score = model.evaluate(X_test, y_test, batch_size=32)
    Test_Loss = score[0] * 100
    Test_accuracy = score[1] * 100

    return classification, confusion, Test_Loss, Test_accuracy



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
    # x = layers.Dense(units=200, name='fc31', activation='relu', bias_initializer='zeros')(x)
    # x = layers.Dropout(0.5)(x)
    # x = layers.Dense(units=100, name='fc32', activation='relu', bias_initializer='zeros')(x)
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
    filepath = '../model/HSI_PCA_model_2.h5'
    X_train = np.load("../data/XtrainWindowSize"
                      + str(windowSize) + "PCA" + str(numPCAcomponents) + "testRatio" + str(testRatio) + ".npy")
    y_train = np.load("../data/ytrainWindowSize"
                      + str(windowSize) + "PCA" + str(numPCAcomponents) + "testRatio" + str(testRatio) + ".npy")

    X_test = np.load("../data/XtestWindowSize"
                     + str(windowSize) + "PCA" + str(numPCAcomponents) + "testRatio" + str(testRatio) + ".npy")
    y_test = np.load("../data/ytestWindowSize"
                     + str(windowSize) + "PCA" + str(numPCAcomponents) + "testRatio" + str(testRatio) + ".npy")

    # 转化标签
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    print(X_train.shape)
    print(y_train.shape)

    # 定义输入的形状
    input_shape = X_train[0].shape
    print(input_shape)

    # filters
    C1 = 3 * numPCAcomponents

    optimizer = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    objective = 'categorical_crossentropy'
    metrics = ['accuracy']
    model = HSI(objective=objective, optimizer=optimizer, metrics=metrics, input_shape=input_shape, filters=C1,
                K_num=K_num)

    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False,
                                 mode='auto', period=1)
    tensorboard = TensorBoard(log_dir='../logs/HSI_PCA_softmax_2')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0000000001)

    model = model.fit(X_train, y_train, batch_size=32, validation_data=(X_test, y_test),
                         callbacks=[checkpoint, tensorboard, reduce_lr], epochs=20, verbose=1)
    model.load_weights('../model/HSI_PCA_model_2.h5')
    # 预测
    classification, confusion, Test_loss, Test_accuracy = reports(model, X_test, y_test)

    # 写入
    classification = str(classification)
    confusion = str(confusion)
    file_name = './result/report' + "WindowSize" + str(windowSize) + "PCA" + str(numPCAcomponents) + "testRatio" + str(
        testRatio) + ".txt"
    with open(file_name, 'w') as x_file:
        x_file.write('{} Test loss (%)'.format(Test_loss))
        x_file.write('\n')
        x_file.write('{} Test accuracy (%)'.format(Test_accuracy))
        x_file.write('\n')
        x_file.write('\n')
        x_file.write('{}'.format(classification))
        x_file.write('\n')
        x_file.write('{}'.format(confusion))



if __name__ == '__main__':
    train_model( )

### 0.9792
