import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from models.layers import *


@gin.configurable
def inception_resnet_v2(classification, fine_tune_at, img_size):
    '''
    Inception-resnet-v2 using transfer learning
    Args:
        classification: type of classification
        fine_tune_at: the number of freezing layers
        img_size: the size of input images
    Return:
        keras model object
    '''

    # Set the input
    inputs = tf.keras.Input(shape=img_size)
    # output = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 127.5, offset=-1)(inputs)
    # Build the model with transfer learning
    preprocess_inputs = tf.keras.applications.inception_resnet_v2.preprocess_input
    #rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1)
    base_model = tf.keras.applications.InceptionResNetV2(input_shape=img_size, include_top=False, weights='imagenet')
    # Freezing the layers before fine_tune_at
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    activation = tf.keras.layers.Activation(activation='linear', name='last_conv')

    x = preprocess_inputs(inputs)
    # x = rescale(x)
    x = base_model(x)
    x = activation(x)
    output = output_block(x, classification=classification)

    return tf.keras.Model(inputs=inputs, outputs=output, name='inception_resnet_v2')


@gin.configurable
def mobilenet(classification, fine_tune_at, img_size):
    '''
    Mobilenet using transfer learning
    Args:
        classification: type of classification
        fine_tune_at: the number of freezing layers
        img_size: the size of input images
    Return:
        keras model object
    '''

    # Set the input
    inputs = tf.keras.Input(shape=img_size)
    # Build the model with transfer learning
    preprocess_inputs = tf.keras.applications.mobilenet.preprocess_input
    # rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1)
    base_model = tf.keras.applications.MobileNet(input_shape=img_size, include_top=False, weights='imagenet')
    # Freezing the layers before fine_tune_at
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    activation = tf.keras.layers.Activation(activation='linear', name='last_conv')

    x = preprocess_inputs(inputs)
    # x = rescale(x)
    x = base_model(x)
    x = activation(x)
    output = output_block(x, classification=classification)

    return tf.keras.Model(inputs=inputs, outputs=output, name='mobilenet')
