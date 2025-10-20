import gin
import tensorflow as tf

from models.layers import *


@gin.configurable
def CNN(img_size, neuron_list, number_of_layer, kernel_size, stride, pool_size, classification):

    """
    Define a simple CNN model
    Args:
        img_size: the size of input images
        neuron_list: which neurons will be used for this model
        number_of_layer: the times that each type of neuron will be repeated
        kernel_size: kernel size used for single_conv and last_conv_layer blocks
        stride: the number of stride used for single_conv block
        pool_size: pool size for last_conv_layer block
        classification: type of dataset
    Return:
        keras model object
    """

    # set the input
    inputs = tf.keras.Input(shape=img_size)
    output = inputs
    for i in range(len(neuron_list)):  # i means the times that the blocks will be used with different neurons
        for j in range(number_of_layer[i]):  # j means the times that the block will be used with particular neuron
            if i != len(neuron_list)-1:
                output = conv_bn_maxpooling(output, conv_units=neuron_list[i], kernel_size=kernel_size, pool_size=pool_size)
            elif i == len(neuron_list)-1 and j == number_of_layer[i]-1:
                output = last_conv_layer(output, conv_units=neuron_list[i], kernel_size=kernel_size, pool_size=pool_size)

    # output block with dense layers
    output = output_block(output, classification=classification)

    return tf.keras.Model(inputs=inputs, outputs=output, name='CNN')


@gin.configurable
def VGG(img_size, neuron_list, number_of_layer, dropout_rate, kernel_size, pool_size, classification):

    """
    Define a VGG model
    Args:
        img_size: the size of input images
        neuron_list: which neurons will be used for this model
        number_of_layer: the times that each type of neuron will be repeated
        dropout_rate: the number of dropout rate for each dropout layer in blocks
        kernel_size: kernel size used for conv_bn_dropout and conv_bn_maxpooling blocks
        pool_size: pool size for last_conv_layer block
        classification: type of dataset
    Return:
        keras model object
    """

    # set the input
    inputs = tf.keras.Input(shape=img_size)
    output = inputs
    for i in range(len(neuron_list)):  # i means the times that the blocks will be used with different neurons
        for j in range(number_of_layer[i]):  # j means the times that the block will be used with particular neuron
            if j != number_of_layer[i]-1:
                output = conv_bn_dropout(output, dropout_rate=dropout_rate, conv_units=neuron_list[i], kernel_size=kernel_size)
            elif i != len(neuron_list)-1 and j == number_of_layer[i]-1:
                output = conv_bn_maxpooling(output, conv_units=neuron_list[i], kernel_size=kernel_size, pool_size=pool_size)
            elif i == len(neuron_list)-1 and j == number_of_layer[i]-1:
                output = last_conv_layer(output, conv_units=neuron_list[i], kernel_size=kernel_size, pool_size=pool_size)

    # output block with dense layers
    output = output_block(output, classification=classification)

    return tf.keras.Model(inputs=inputs, outputs=output, name='VGG')


@gin.configurable
def ResNet101(img_size, bottleneck_list, neurons, classification):

    """
    Define a ResNet101 model
    Args:
        img_size: the size of input images
        bottleneck_list: which neurons will be used for this model
        neurons: the number of initial neuron
        classification: type of dataset
    Return:
        keras model object
    """

    # set the input
    inputs = tf.keras.Input(shape=img_size)
    output = inputs
    output = conv_maxpooling(output, conv_units=256, kernel_size=7, pool_size=3, stride=2)

    # bottleneck_id means the times that the blocks will be used with different neurons
    for bottleneck_id in range(len(bottleneck_list)):
        # layer_id means the times that the block will be used with particular neuron
        for layer_id in range(bottleneck_list[bottleneck_id]):
            # Only the first time in the first Bottleneck with residual step
            if bottleneck_id != 0 and layer_id == 0:
                output = Bottleneck(output, conv_units=neurons, stride=2, residual_path=True)
            else:
                output = Bottleneck(output, conv_units=neurons, stride=1, residual_path=False)
        neurons *= 2  # the number of neurons will be twice than the last block

    # Use the last number of neuron in neurons_list to set the last convolutional block
    output = last_conv_layer(output, conv_units=neurons/2, kernel_size=1, pool_size=1)

    # output block with dense layers
    output = output_block(output, classification=classification)

    return tf.keras.Model(inputs=inputs, outputs=output, name='ResNet101')
