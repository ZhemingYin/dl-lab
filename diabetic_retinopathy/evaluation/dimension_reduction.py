import tensorflow as tf
import umap
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px


def dimensional_reduction(model, dataset, labels):

    """ Diemnsionality reduction with UMAP method
    Args:
        model: the model which is used for training
        dataset: test set
        labels: true labels
    return:
        plot the image of dimensionality reduction
    """

    # get the model
    umap_model = tf.keras.models.Model([model.inputs], [model.get_layer('last_conv').output])
    for idx, (image, label) in enumerate(dataset):
        if idx == 0:
            values = umap_model(image)
            # convert tensor 'values` to a proto tensor
            proto_tensor = tf.make_tensor_proto(values)
            values = tf.make_ndarray(proto_tensor)
            # use the dimension of the first value as template to concat another values
            for i, value in enumerate(values):
                if i == 0:
                    value = tf.reshape(value, [1, -1])
                    umap_values = value
                    print(value.shape)
                    print(value.dtype)
                elif i != 0:
                    value = tf.reshape(value, [1, -1])
                    umap_values = tf.concat([umap_values, value], axis=0)
        elif idx != 0:
            values = umap_model(image)
            # convert `values` to a proto tensor
            proto_tensor = tf.make_tensor_proto(values)
            values = tf.make_ndarray(proto_tensor)
            for i, value in enumerate(values):
                value = tf.reshape(value, [1, -1])
                umap_values = tf.concat([umap_values, value], axis=0)

    print(umap_values.shape)
    print(umap_values.dtype)

    # Except the batch dimension, convert another dimension into 2D
    reducer = umap.UMAP(n_neighbors=10, n_components=2)
    X_test_res = reducer.fit_transform(umap_values)
    print('Shape of X_train_res: ', X_test_res.shape)

    y_true = tf.reshape(labels, [-1])
    print(y_true.shape)
    print(y_true.dtype)
    y = np.array(y_true)
    arr_concat = np.concatenate((X_test_res, y.reshape(y.shape[0], 1)), axis=1)
    # Create a Pandas dataframe using the above array
    df = pd.DataFrame(arr_concat, columns=['x', 'y', 'label'])
    # Convert label data type from float to integer
    df['label'] = df['label'].astype(int)
    # Finally, sort the dataframe by label
    df.sort_values(by='label', axis=0, ascending=True, inplace=True)
    # Create graph
    fig = px.scatter(df, x='x', y='y', color=df['label'].astype(str))
    fig.show()

    return
