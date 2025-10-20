import gin
import tensorflow as tf

from models.layers import *


@gin.configurable
def RNN(input_shape, neuron_num, number_of_layer, dropout_rate, name, label_type):
    """
    Define a LSTM model
    Args:
        input_shape: the shape of input sequence
        neuron_num: the number of neurons which will be used for this model
        number_of_layer: the times of the layer
        dropout_rate: the number of dropout rate for each dropout layer in blocks
        name: name of dataset, HAPT or HAR
        label_type: type of label, s2s or s2l
    Return:
        keras model object
    """

    inputs = tf.keras.Input(input_shape)
    output = inputs
    for i in range(number_of_layer):
        output = lstm_dropout(output, lstm_units=neuron_num * (i+1), dropout_rate=dropout_rate)
    output = output_block(output, dropout_rate=dropout_rate, name=name, label_type=label_type)

    return tf.keras.Model(inputs=inputs, outputs=output, name='RNN')


@gin.configurable
def GRU(input_shape, neuron_list, number_of_layer, dropout_rate, name, label_type):
    """
    Define a GRU model
    Args:
        input_shape: the shape of input sequence
        neuron_list: the list of neurons which will be used for this model
        number_of_layer: the times that each type of neuron will be repeated
        dropout_rate: the number of dropout rate for each dropout layer in blocks
        name: name of dataset, HAPT or HAR
        label_type: type of label, s2s or s2l
    Return:
        keras model object
    """

    inputs = tf.keras.Input(input_shape)
    output = inputs
    for i in range(number_of_layer):
            output = gru_dropout(output, gru_units=neuron_num * (i+1), dropout_rate=dropout_rate)
    output = output_block(output, dropout_rate=dropout_rate, name=name, label_type=label_type)

    return tf.keras.Model(inputs=inputs, outputs=output, name='GRU')


@gin.configurable
def BRNN(input_shape, neuron_num, number_of_layer, dropout_rate, name, label_type):
    """
        Define a BRNN model
        Args:
            input_shape: the shape of input sequence
            neuron_num: the list of neurons which will be used for this model
            number_of_layer: the times that each type of neuron will be repeated
            dropout_rate: the number of dropout rate for each dropout layer in blocks
            name: name of dataset, HAPT or HAR
            label_type: type of label, s2s or s2l
        Return:
            keras model object
        """

    inputs = tf.keras.Input(input_shape)
    output = inputs
    for i in range(number_of_layer):
            output = brnn_dropout(output, lstm_units=neuron_num * (i+1), dropout_rate=dropout_rate)
    output = output_block(output, dropout_rate=dropout_rate, name=name, label_type=label_type)

    return tf.keras.Model(inputs=inputs, outputs=output, name='BRNN')


@gin.configurable
def RNN_CNN(input_shape, rnn_neuron_list, rnn_number_of_layer, cnn_neuron_list, cnn_number_of_layer, dropout_rate,
            kernel_size, pool_size, name, label_type):

    inputs = tf.keras.Input(input_shape)
    output = inputs
    for i in range(len(rnn_neuron_list)):
        for j in range(rnn_number_of_layer[i]):
            output = lstm_dropout(output, lstm_units=rnn_neuron_list[i], dropout_rate=dropout_rate)
    for i in range(len(cnn_neuron_list)):
        for j in range(cnn_number_of_layer[i]):
            output = conv_maxpooling(output, conv_units=cnn_neuron_list[i], kernel_size=kernel_size, pool_size=pool_size)
    output = output_block(output, dropout_rate=dropout_rate, name=name, label_type=label_type)

    return tf.keras.Model(inputs=inputs, outputs=output, name='RNN_CNN')


@gin.configurable
def CNN_RNN(input_shape, rnn_neuron_list, rnn_number_of_layer, cnn_neuron_list, cnn_number_of_layer, dropout_rate,
            kernel_size, pool_size, name, label_type):

    inputs = tf.keras.Input(input_shape)
    output = inputs
    for i in range(len(cnn_neuron_list)):
        for j in range(cnn_number_of_layer[i]):
            output = conv_maxpooling(output, conv_units=cnn_neuron_list[i], kernel_size=kernel_size,
                                     pool_size=pool_size)
    for i in range(len(rnn_neuron_list)):
        for j in range(rnn_number_of_layer[i]):
            output = lstm_dropout(output, lstm_units=rnn_neuron_list[i], dropout_rate=dropout_rate)
    output = output_block(output, dropout_rate=dropout_rate, name=name, label_type=label_type)

    return tf.keras.Model(inputs=inputs, outputs=output, name='CNN_RNN')