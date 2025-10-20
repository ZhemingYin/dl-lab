import logging
import gin
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from evaluation.metrics import *
import tensorflow as tf
from input_pipeline.data_visualization import bg_color
from evaluation.dimension_reduction import dimensional_reduction


@gin.configurable
def evaluate(model, ds_test, label_type, name):

    """evaluate performance of the model
        Args:
            model: model which will be evaluated
            ds_test: test set
            label_type: type of label, s2s or s2l
            name: name of dataset, HAPT or HAR
        """

    # Load the checkpoint
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        "/Users/rocker/Desktop/Uni Stuttgart/DL Lab/runs/run_2023-02-03T04-06-04-898610/ckpts",
        max_to_keep=10
    )
    checkpoint.restore(checkpoint_manager.latest_checkpoint)

    # Check if the checkpoint was restored correctly
    if checkpoint_manager.latest_checkpoint:
        logging.info(f"Restored from {checkpoint_manager.latest_checkpoint}")
    else:
        logging.info("Initializing from scratch.")

    # Read true labels and predicted labels from test dataset and model
    y_true, y_pred, y_pred_origin_matrix = [], [], []
    for index, (window, label) in enumerate(ds_test):
        y_pred_origin = model.predict(window)
        # Read labels
        if label_type == 's2l':
            # Read predicted labels
            if index == 0:
                y_pred_origin_matrix = y_pred_origin
            elif index != 0:
                y_pred_origin_matrix = np.concatenate((y_pred_origin_matrix, y_pred_origin), axis=0)
            for i in range(y_pred_origin.shape[0]):
                idx = np.argmax(y_pred_origin[i, :]) + 1
                y_pred.append(idx)
            # Read true labels
            for i in range(label.shape[0]):
                idx = tf.argmax(label[i, :]) + 1
                y_true.append(idx)

        elif label_type == 's2s':
            # Read predicted labels
            if name == "HAPT":
                y_pred_origin_new = np.reshape(y_pred_origin, (-1, 12))
            elif name == "HAR":
                y_pred_origin_new = np.reshape(y_pred_origin, (-1, 8))
            else:
                raise ValueError
            if index == 0:
                y_pred_origin_matrix = y_pred_origin_new
            else:
                y_pred_origin_matrix = np.concatenate((y_pred_origin_matrix, y_pred_origin_new), axis=0)
            for i in range(y_pred_origin.shape[0]):
                for j in range(y_pred_origin.shape[1]):
                    idx = np.argmax(y_pred_origin[i, j, :]) + 1
                    y_pred.append(idx)
            # Read true labels
            label = label.numpy()
            for i in range(label.shape[0]):
                for j in range(label.shape[1]):
                    idx = np.argmax(label[i, j, :]) + 1
                    y_true.append(idx)

    # Plot the evaluation result
    eval_plot(name, ds_test, y_pred)

    # Make folder for saving visualization image
    make_folder()

    # Plot Confusion Matrix
    ConfusionMatrix(y_pred_origin_matrix, y_pred, y_true, label_type=label_type, name=name)

    # Dimensional reduction
    dimensional_reduction(model=model, dataset=ds_test, labels=y_true)

    return


def make_folder():

    visualization_path = os.getcwd() + "/visualization"
    if os.path.exists(visualization_path):
        shutil.rmtree(visualization_path)
    os.makedirs(visualization_path)

    return


def eval_plot(name, dataset, pred_labels, plot_range=20000):
    """conduct the activity recognition for a single position
    Args:
        name: name of the dataset
        dataset: the dataset for plot
        pred_labels: the label obtained from the model
        plot_range: the range of the features to be plotted
    Returns: the plot with colored background
    """

    acc_x, acc_y, acc_z = [], [], []
    gyro_x, gyro_y, gyro_z = [], [], []
    true_labels = []
    for idx, (feature_window, label_window) in enumerate(dataset):
        feature_window = feature_window.numpy()
        label_window = label_window.numpy()
        labels = np.concatenate(label_window, axis=0)
        true_labels.append(labels)
        features = np.concatenate(feature_window, axis=0)
        acc_x.append(features[:, 0])
        acc_y.append(features[:, 1])
        acc_z.append(features[:, 2])
        gyro_x.append(features[:, 3])
        gyro_y.append(features[:, 4])
        gyro_z.append(features[:, 5])

    acc_x, acc_y, acc_z = np.concatenate(acc_x), np.concatenate(acc_y), np.concatenate(acc_z)
    gyro_x, gyro_y, gyro_z = np.concatenate(gyro_x), np.concatenate(gyro_y), np.concatenate(gyro_z)
    true_labels = np.concatenate(true_labels)
    true_labels = np.argmax(true_labels, axis=1) + 1
    pred_labels = np.asarray(pred_labels)

    # Define the plotted range
    acc_x, acc_y, acc_z = acc_x[:plot_range], acc_y[:plot_range], acc_z[:plot_range]
    gyro_x, gyro_y, gyro_z = gyro_x[:plot_range], gyro_y[:plot_range], gyro_z[:plot_range]
    true_labels = true_labels[:plot_range]
    pred_labels = np.asarray(pred_labels[:plot_range])

    # Plot
    plt.figure(figsize=(9, 6), dpi=150)
    plt.subplot(2, 1, 1)
    plt.title('Ground Truth')
    plt.plot(acc_x)
    plt.plot(acc_y)
    plt.plot(acc_z)
    plt.plot(gyro_x)
    plt.plot(gyro_y)
    plt.plot(gyro_z)
    bg_color(name, true_labels)

    plt.subplot(2, 1, 2)
    plt.title('Prediction')
    plt.plot(acc_x)
    plt.plot(acc_y)
    plt.plot(acc_z)
    plt.plot(gyro_x)
    plt.plot(gyro_y)
    plt.plot(gyro_z)
    bg_color(name, pred_labels)

    plt.tight_layout()
    plt.savefig(os.getcwd() + '/visualization/vis.png')
    plt.show()
