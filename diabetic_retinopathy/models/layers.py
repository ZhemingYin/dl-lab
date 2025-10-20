import gin
import tensorflow as tf

@gin.configurable
def conv_maxpooling(input, conv_units, kernel_size, pool_size, stride):

    """
    A convolutional layer with a maxpooling layer
    Args:
        input: input of convolutional layer
        conv_units: the number of neurons used for convolutional layer
        kernel_size: kernel size used for convolutional layer
        pool_size: pool size used for maxpooling layer
        stride: the number of stride used for both convolutional layer and maxpooling layer
    Return:
        output: output of maxpooling layer with convolutional layer
    """

    output = tf.keras.layers.Conv2D(conv_units, kernel_size=kernel_size, strides=stride, padding='same',
                                    activation=tf.nn.relu)(input)
    output = tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=stride, padding='same')(output)

    return output


@gin.configurable
def single_conv(input, conv_units, kernel_size, stride):

    """
        A single convolutional layer
        Args:
            input: input of convolutional layer
            conv_units: the number of neurons used for convolutional layer
            kernel_size: kernel size used for convolutional layer
            stride: the number of stride used for convolutional layer
        Return:
            output: output of convolutional layer
        """

    output = tf.keras.layers.Conv2D(conv_units, kernel_size=kernel_size, strides=stride, padding='SAME')(input)

    return output


@gin.configurable
def output_block(input, dense_units, dropout_rate, classification):

    """
        A output block consisting of several dense layers, global average pooling layer and dropout layer
        Args:
            input: input of dense layer
            dense_units: the number of neurons used for dense layer
            dropout_rate: the number of dropout rate for dropout layer
            classification: the type of dataset
        Return:
            output: output of last dense layer
        """

    output = tf.keras.layers.GlobalAveragePooling2D()(input)
    output = tf.keras.layers.Dense(dense_units, activation='relu')(output)
    output = tf.keras.layers.Dense(1000, activation='relu')(output)
    output = tf.keras.layers.BatchNormalization()(output)
    output = tf.keras.layers.Dropout(rate=dropout_rate)(output)

    # According to the different types of dataset, choose different numbers of neurons and types of activation functions
    if classification == 'binary':
        output = tf.keras.layers.Dense(1, activation='sigmoid', name='last_output')(output)
    elif classification == 'multiple':
        output = tf.keras.layers.Dense(5, activation='softmax', name='last_output')(output)
    elif classification == 'regression':
        output = tf.keras.layers.Dense(1, activation='linear', name='last_output')(output)

    return output


@gin.configurable
def conv_bn_dropout(input, dropout_rate, conv_units, kernel_size):

    """
        A convolutional layer with a batch normalization layer and a dropout layer
        Args:
            input: input of convolutional layer
            dropout_rate: the number of dropout rate for dropout layer
            conv_units: the number of neurons used for convolutional layer
            kernel_size: kernel size used for convolutional layer
        Return:
             output: output of dropout layer with a convolutional layer and a batch normalization layer
        """

    output = tf.keras.layers.Conv2D(conv_units, kernel_size=kernel_size, padding='same', activation='relu')(input)
    output = tf.keras.layers.BatchNormalization()(output)
    output = tf.keras.layers.Dropout(dropout_rate)(output)

    return output


@gin.configurable
def conv_bn_maxpooling(input, conv_units, kernel_size, pool_size):

    """
        A convolutional layer with a batch normalization layer and a maxpooling layer
        Args:
            input: input of convolutional layer
            conv_units: the number of neurons used for convolutional layer
            kernel_size: kernel size used for convolutional layer
            pool_size: pool size used for maxpooling layer
        Return:
            output: output of maxpooling layer with a convolutional layer and a batch normalization layer
        """

    output = tf.keras.layers.Conv2D(conv_units, kernel_size=kernel_size, padding='same', activation='relu')(input)
    output = tf.keras.layers.BatchNormalization()(output)
    output = tf.keras.layers.MaxPooling2D(pool_size=pool_size)(output)

    return output


@gin.configurable
def last_conv_layer(input, conv_units, kernel_size, pool_size):

    """
        A convolutional layer with a maxpooling layer marked as the last convolution layer of a model
        Args:
            input: input of convolutional layer
            conv_units: the number of neurons used for convolutional layer
            kernel_size: kernel size used for convolutional layer
            pool_size: pool size used for maxpooling layer
        Return:
            output: output of maxpooling layer with a convolutional layer
        """

    output = tf.keras.layers.Conv2D(conv_units, kernel_size=kernel_size, padding='same', activation='relu',
                                    name='last_conv')(input)
    output = tf.keras.layers.MaxPooling2D(pool_size=pool_size)(output)

    return output


@gin.configurable
def conv_bn(input, conv_units, kernel_size, stride):

    """
        A convolutional layer with a batch normalization layer
        Args:
            input: input of convolutional layer
            conv_units: the number of neurons used for convolutional layer
            kernel_size: kernel size used for convolutional layer
            stride: the number of stride used for convolutional layer
        Return:
            output: output of batch normalization layer with a convolutional layer
        """

    output = tf.keras.layers.Conv2D(conv_units, kernel_size=kernel_size, strides=stride, activation='relu', padding='same')(input)
    output = tf.keras.layers.BatchNormalization(axis=3)(output)

    return output


@gin.configurable
def Bottleneck(input, conv_units, stride, residual_path=False):

    """
    Block for ResNet101 model
    Args:
        input: input of the first conv_bn block
        conv_units: the number of neurons used for each convolutional layer
        stride: the number of stride used for each convolutional layer
        residual_path: determine whether residual is needed after each block
    Return:
        output: output of three conv_bn blocks added with identity
    """

    identity = input
    output = conv_bn(input=input, conv_units=conv_units, kernel_size=1, stride=stride)
    output = conv_bn(input=output, conv_units=conv_units, kernel_size=3, stride=1)
    output = conv_bn(input=output, conv_units=conv_units*4, kernel_size=1, stride=stride)

    # After three conv_bn, change the dimension of identity to fit the dimension of output
    if residual_path:
        identity = single_conv(input, conv_units=conv_units*4, kernel_size=1, stride=stride**2)

    # Add identity and output to implement residual step
    output = tf.keras.layers.add([output, identity])
    output = tf.nn.relu(output)

    return output
