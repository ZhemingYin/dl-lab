import gin
import tensorflow as tf


@gin.configurable

def output_block(input, dropout_rate, label_type, name, dense_units):
    """
    A output block consisting of several dense layers, global average pooling layer (if s2l) and dropout layer
    Args:
        input: input of dense layer
        dropout_rate: the number of dropout rate for dropout layer
        label_type: type of label, s2s or s2l
        name: name of dataset, HAPT or HAR
        dense_units: the number of neurons used for dense layer
    Return:
        output: output of last dense layer
    """

    output = input
    if label_type == 's2l':
        output = tf.keras.layers.GlobalAveragePooling1D()(output)
    output = tf.keras.layers.Dense(dense_units, activation='relu')(output)
    output = tf.keras.layers.Dropout(rate=dropout_rate)(output)
    if name == "HAPT":
        output = tf.keras.layers.Dense(12, activation='softmax', name='last_output')(output)
    elif name == "HAR":
        output = tf.keras.layers.Dense(8, activation='softmax', name='last_output')(output)
    else:
        raise ValueError

    return output


@gin.configurable
def lstm_dropout(input, lstm_units, dropout_rate, return_sequences, stateful):
    """
    A output block consisting of a single lstm layer and a dropout layer
    Args:
        input: input of dense layer
        lstm_units: the number of the filters
        dropout_rate: the number of dropout rate for dropout layer
        return_sequences: whether output is sequence or not
        stateful: whether we reset the inner state and the outputs after each batch or not
    Return:
        output: output after lstm and dropout layer
    """

    output = tf.keras.layers.LSTM(lstm_units, return_sequences=return_sequences, stateful=stateful)(input)
    output = tf.keras.layers.Dropout(dropout_rate)(output)

    return output


@gin.configurable
def gru_dropout(input, gru_units, dropout_rate, return_sequences, stateful):
    """
    A output block consisting of a single gru layer and a dropout layer
    Args:
        input: input of dense layer
        gru_units: the number of the filters
        dropout_rate: the number of dropout rate for dropout layer
        return_sequences: whether output is sequence or not
        stateful: whether we reset the inner state and the outputs after each batch or not
    Return:
        output: output after gru and dropout layer
    """

    output = tf.keras.layers.GRU(gru_units, return_sequences=return_sequences, stateful=stateful)(input)
    output = tf.keras.layers.Dropout(dropout_rate)(output)

    return output


@gin.configurable
def brnn_dropout(input, lstm_units, dropout_rate, return_sequences, stateful):
    """
    A output block consisting of a single brnn layer and a dropout layer
    Args:
        input: input of dense layer
        lstm_units: the number of the filters
        dropout_rate: the number of dropout rate for dropout layer
        return_sequences: whether output is sequence or not
        stateful: whether we reset the inner state and the outputs after each batch or not
    Return:
        output: output after brnn and dropout layer
    """

    output = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units, return_sequences=return_sequences, stateful=stateful))(input)
    output = tf.keras.layers.Dropout(dropout_rate)(output)

    return output


@gin.configurable
def conv_maxpooling(input, conv_units, kernel_size, pool_size):
    """
    A convolutional layer with a maxpooling layer
    Args:
        input: input of convolutional layer
        conv_units: the number of neurons used for convolutional layer
        kernel_size: kernel size used for convolutional layer
        pool_size: pool size used for maxpooling layer
    Return:
        output: output of maxpooling layer with convolutional layer
    """

    output = tf.keras.layers.Conv2D(conv_units, kernel_size=kernel_size, padding='same', activation='relu')(input)
    output = tf.keras.layers.MaxPooling2D(pool_size=pool_size)(output)

    return output