import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


# Get features and labels from the datasets
def plot(name, dataset, feature, label, input, plot_range=20000):
    """plot the training features either from the dataset or the raw data
    Args:
        name: specify which dataset to plot
        dataset: choose train, test or validation set
        feature: the features for raw data plot
        label: the labels for raw data plot
        input: extract the data from "raw_data" or generated "data_set"
        plot_range: the range of features to be visualized
    Returns: features with colored background
    """

    acc_x, acc_y, acc_z = [], [], []
    gyro_x, gyro_y, gyro_z = [], [], []
    labels = []

    if input == "data_set":
        for idx, (feature_window, label_window) in enumerate(dataset):
            feature_window = feature_window.numpy()
            label_window = label_window.numpy()
            labels.append(label_window)
            acc_x.append(feature_window[:, 0])
            acc_y.append(feature_window[:, 1])
            acc_z.append(feature_window[:, 2])
            gyro_x.append(feature_window[:, 3])
            gyro_y.append(feature_window[:, 4])
            gyro_z.append(feature_window[:, 5])
            if idx == 50:
                break
        acc_x, acc_y, acc_z = np.concatenate(acc_x), np.concatenate(acc_y), np.concatenate(acc_z)
        gyro_x, gyro_y, gyro_z = np.concatenate(gyro_x), np.concatenate(gyro_y), np.concatenate(gyro_z)
        labels = np.concatenate(labels)

    elif input == "raw_data":
        acc_x, acc_y, acc_z = feature[:, 0], feature[:, 1], feature[:, 2]
        gyro_x, gyro_y, gyro_z = feature[:, 3], feature[:, 4], feature[:, 5]
        labels = label

    else:
        raise ValueError

    # Define the plotted range
    acc_x, acc_y, acc_z = acc_x[:plot_range], acc_y[:plot_range], acc_z[:plot_range]
    gyro_x, gyro_y, gyro_z = gyro_x[:plot_range], gyro_y[:plot_range], gyro_z[:plot_range]
    labels = labels[:plot_range]

    # Plot
    plt.figure(figsize=(9, 6), dpi=150)
    plt.subplot(2, 1, 1)
    plt.title('Accelerometer')
    plt.plot(acc_x)
    plt.plot(acc_y)
    plt.plot(acc_z)
    bg_color(name, labels)

    plt.subplot(2, 1, 2)
    plt.title('Gyroscope')
    plt.plot(gyro_x)
    plt.plot(gyro_y)
    plt.plot(gyro_z)
    bg_color(name, labels)

    plt.tight_layout()
    plt.show()

    return


# Background color according to the labels
def bg_color(name, labels):

    if name == 'HAPT':
        label_color = ['lawngreen', 'gold', 'steelblue', 'darkorange', 'violet', 'purple',
                       'salmon', 'sienna', 'yellow', 'pink', 'firebrick', 'chocolate']
    elif name == 'HAR':
        label_color = ['lawngreen', 'gold', 'steelblue', 'darkorange', 'violet', 'purple',
                       'salmon', 'sienna']
    else:
        raise ValueError

    start = 0
    for i in range(1, int(labels.size)):
        if labels[i] != labels[i - 1]:
            end = i - 1
            plt.axvspan(xmin=start, xmax=end, facecolor=label_color[labels[i-1] - 1], alpha=0.5)
            start = i
    plt.axvspan(start, int(labels.size) - 1, facecolor=label_color[labels[-1] - 1], alpha=0.5)
    return label_color
