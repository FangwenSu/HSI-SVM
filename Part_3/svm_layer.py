# -*- coding: utf-8 -*-
"""
 @Time    : 19-9-16 下午2:32
 @Author  : yangzh
 @Email   : 1725457378@qq.com
 @File    : svm_layer.py
"""
from keras.models import Model
from sklearn.svm import SVC
from keras.utils import np_utils


class SvmLayer:
    """
    Linear stack of layers with the option to replace the end of the stack with a Support Vector Machine
    # Arguments
        layers: list of layers to add to the model.
        svm: The Support Vector Machine to use.
    """
    def __init__(self, model, svm=None):
        super().__init__()

        self.model = model
        self.intermediate_model = None  # type: Model
        self.svm = svm

        if svm is None:
            self.svm = SVC(kernel='linear')

    def add(self, layer):
        return self.model.add(layer)

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.,
            validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0,
            steps_per_epoch=None, validation_steps=None, **kwargs):

        fit = self.model.fit(x, y, batch_size, epochs, verbose, callbacks, validation_split,
                             validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch,
                             validation_steps, **kwargs)

        self.fit_svm(x, y, self.__get_split_layer())

        return fit

    def fit_svm(self, x, y, split_layer):

        self.intermediate_model = Model(inputs=self.model.input,
                                        outputs=split_layer.output)
        intermediate_output = self.intermediate_model.predict(x)
        self.svm.fit(intermediate_output, y)

    def evaluate(self, x=None, y=None, batch_size=None, verbose=1, steps=None):

        if self.intermediate_model is None:
            raise Exception("A model must be fit before running evaluate")
        output = self.predict(x, batch_size, verbose, steps)
        correct = [output[i] == y[i]
                   for i in range(len(output))]

        accuracy = sum(correct) / len(correct)

        return accuracy

    def predict(self, x, batch_size=None, verbose=0, steps=None):

        intermediate_prediction = self.intermediate_model.predict(x, batch_size, verbose, steps)
        output = self.svm.predict(intermediate_prediction)

        return output

    def __get_split_layer(self):

        if len(self.model.layers) < 3:
            raise ValueError('self.layers to small for a relevant split')

        for layer in self.model.layers:
            if layer.name == "split_layer":
                return layer

        return self.model.layers[-3]
