import gin
import logging
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split

from evaluation.metrics import *
from input_pipeline import datasets
from models.architectures import *
from models.transfer_learning import inception_resnet_v2, mobilenet


@gin.configurable
def ensemble_learning(train, val, test, type):
    '''
    Combine the output of several models and vote for a prediction which has maximum possibility
    Args:
        train: train set
        val: validation set
        test: test set
        type: type of label, s2s or s2l
    '''

    gin.parse_config_files_and_bindings(['configs/config_ensemble_learning.gin'], [])

    model_list = ['RNN', 'GRU', 'BRNN']

    # record the predictions and labels of each model
    binary_predictions_list = []
    multiple_predictions_list = []

    # the times that we split dataset for each model
    times = 5

    # Only use for save the original probability of predictions in multi-class case
    y_pred_with_probability = np.zeros([103, 5])

    # evaluate each model
    for model_name in model_list:
        # setup model
        if model_name == 'RNN':
            model = RNN(type=type)
        elif model_name == 'GRU':
            model = GRU(type=type)
        elif model_name == 'BRNN':
            model = BRNN(type=type)
        elif model_name == 'CNN_RNN':
            model = CNN_RNN(type=type)
        elif model_name == 'RNN_CNN':
            model = RNN_CNN(type=type)

        opt = tf.keras.optimizers.Adam(lr=0.0005 / 10)
        model.compile(loss="CategoricalCrossentropy", optimizer=opt)

        # for time in range(times):

            # train_split, val_split = sklearn.model_selection.train_test_split(train, test_size=0.2)
        model.fit(train, epochs=100, batch_size=16, validation_data=val)

        for index, (window, label) in enumerate(test):
            y_pred_origin = model.predict(window)
            y_pred = []
            if type == 's2s':
                for i in range(y_pred_origin.shape[0]):
                    for j in range(y_pred_origin.shape[1]):
                        idx = np.argmax(y_pred_origin[i, j, :]) + 1
                        y_pred.append(idx)
            elif type == 's2l':
                for i in range(y_pred_origin.shape[0]):
                    idx = np.argmax(y_pred_origin[i, :]) + 1
                    y_pred.append(idx)
            if model_name == 'RNN':
                multiple_predictions_list = y_pred
            elif model_name != 'RNN':
                multiple_predictions_list = np.concatenate((multiple_predictions_list, y_pred), axis=0)

            if type == 's2l':
                if index == 0 & model_name == 'RNN':
                    y_pred_with_probability = y_pred_origin
                else:
                    y_pred_with_probability += y_pred_origin
                # print(y_pred_origin_matrix.shape)
            elif type == 's2s':
                y_pred_origin_new = np.reshape(y_pred_origin, (-1, 12))
                if index == 0 & model_name == 'RNN':
                    y_pred_with_probability = y_pred_origin_new
                else:
                    y_pred_with_probability += y_pred_origin_new
                print(y_pred_with_probability.shape)


    # evaluate the ensemble learning model
    print('---Evaluation of Ensemble Learning---')

    y_pred = []
    for i in range(multiple_predictions_list.shape[1]):
        pred = np.argmax(np.bincount(multiple_predictions_list[:, i]))
        y_pred.append(pred)
    print(y_pred)
    print(len(y_pred))


    # Get the true test labels
    y_true = []
    for idx, (test_windows, test_labels) in enumerate(test):
        # dim = test_labels.shape[0]
        # print(test_labels.shape)
        if type == 's2s':
            for i in range(test_labels.shape[0]):
                for j in range(test_labels.shape[1]):
                    # print(test_labels[i, j, :])
                    idx = tf.argmax(test_labels[i, j, :]) + 1
                    y_true.append(idx)
        elif type == 's2l':
            for i in range(test_labels.shape[0]):
                # print(test_labels[i, j, :])
                idx = tf.argmax(test_labels[i, :]) + 1
                y_true.append(idx)

    y_pred_with_probability = y_pred_with_probability / len(model_list)
    ConfusionMatrix(y_pred_with_probability, y_pred, y_true, classification='multiple', type=type)


@gin.configurable
def main():
    data_dir = "/Users/yinzheming/Desktop/Deep_Learning/Lab/HAPT_dataset"
    # data_dir = "/Users/rocker/Desktop/Uni Stuttgart/DL Lab/dataset/HAPT_dataset"

    # data_dir = "/home/data/IDRID_dataset"
    ds_train, ds_val, ds_test, ds_info = datasets.load(name='HAPT', data_dir=data_dir, type='s2s')
    ensemble_learning(ds_train, ds_val, ds_test, type='s2s')


main()