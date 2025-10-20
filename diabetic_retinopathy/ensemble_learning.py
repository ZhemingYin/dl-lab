import gin
import logging
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

from evaluation.metrics import *
from input_pipeline import datasets
from models.architectures import *
from models.transfer_learning import inception_resnet_v2, mobilenet
from utils import utils_params, utils_misc


@gin.configurable
def ensemble_learning(train, validation, test, classification):

    '''
    Combine the output of several models and vote for a prediction which has maximum possibility
    Args:
        train: train set
        validation: validation set
        test: test set
        classification: type of classification
    '''

    # The list of models which will be used
    model_list = ['CNN', 'VGG', 'ResNet101', 'inception_resnet_v2', 'mobilenet']

    # record the predictions and labels of each model
    predictions_list = []

    # Only use for save the original probability of predictions in multi-class case
    y_pred_with_probability = np.zeros([103, 5])

    # load and compile each model
    for model_name in model_list:
        # setup model
        if model_name == 'CNN':
            model = CNN(classification=classification)
        elif model_name == 'VGG':
            model = VGG(classification=classification)
        elif model_name == 'ResNet101':
            model = ResNet101(classification=classification)
        elif model_name == 'inception_resnet_v2':
            model = inception_resnet_v2()
        elif model_name == 'mobilenet':
            model = mobilenet()

        if classification == 'binary':
            opt = tf.keras.optimizers.Adam(lr=0.0005 / 10)
            model.compile(loss="BinaryCrossentropy", optimizer=opt, metrics=["accuracy"])
        elif classification == 'multiple':
            opt = tf.keras.optimizers.Adam(lr=0.0005 / 10)
            model.compile(loss="SparseCategoricalCrossentropy", optimizer=opt, metrics=["accuracy"])

        model.fit(train, epochs=20, batch_size=16, validation_data=validation)

        # Predict test set with each model
        if classification == 'binary':
            y_pred_origin = model.predict(test)
            y_pred = np.where(y_pred_origin > 0.5, 1, 0)
            y_pred = np.ndarray.tolist(y_pred)
            y_pred = [x[0] for x in y_pred]

            # Use first result as template to concatenate the other results
            if model_name == 'CNN':
                predictions_list = y_pred
            elif model_name != 'CNN':
                predictions_list = np.vstack((predictions_list, y_pred))

        elif classification == 'multiple':
            y_pred_origin = model.predict(test)
            y_pred = []
            for i in range(y_pred_origin.shape[0]):
                idx = np.argmax(y_pred_origin[i])
                y_pred.append(idx)
            if model_name == 'CNN':
                predictions_list = y_pred
            elif model_name != 'CNN':
                predictions_list = np.vstack((predictions_list, y_pred))
            y_pred_with_probability += y_pred_origin


    # evaluate the ensemble learning model
    print('---Evaluation of Ensemble Learning---')

    y_pred = []
    predictions_list = np.array(predictions_list)
    # Vote for the predicted result which has the maximum probability
    for i in range(predictions_list.shape[1]):
        pred = np.argmax(np.bincount(predictions_list[:, i]))
        y_pred.append(pred)
    print(y_pred)
    print(len(y_pred))

    # Get the true test labels
    y_true = []
    for idx, (test_images, test_labels) in enumerate(test):
        dim = test_labels.shape[0]
        for i in range(dim):
            y_true.append(test_labels[i].numpy())
    print(y_true)
    print(len(y_true))

    if classification == 'multiple':
        y_pred_with_probability = y_pred_with_probability / len(model_list)

    ConfusionMatrix(y_pred_with_probability, y_pred, y_true, classification)


@gin.configurable
def main():
    gin.parse_config_files_and_bindings(['configs/config_ensemble.gin'], [])
    ds_train, ds_val, ds_test, ds_info = datasets.load()
    ensemble_learning(ds_train, ds_val, ds_test)

main()
